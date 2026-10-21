"""Verify member bundles, join by (query, model, sample), and publish an immutable release."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.benchmark import Benchmark, safe_name
from . import CHANNELS, PROTOCOL, channels, is_scored, allowed_missing_score
from .collect import generation_config, job_spec, response_hash
from .dataset import load_dataset, model_pool
from .io import digest, environment, key, locked, read_jsonl, seal, successful, verify, write_json


def release_spec(cfg):
    """Storage locations and selected view do not change benchmark identity."""
    return {k: v for k, v in cfg.items() if k not in ("path", "hub_repo", "hub_revision", "contribution_paths", "quality_metric")}


# Benchmark extra name -> long-table column, written into every view (see data/benchmark.py).
SAMPLE_EXTRAS = {"sample_cost": "cost", "completion_tokens": "completion_tokens", "seconds": "seconds"}


def sample_columns(cfg):
    # Opt in for new releases only; old frozen views keep their layout/fingerprint.
    public = {"prompt_tokens": "prompt_tokens"} if "prompt_tokens" in cfg.get("input_features", []) else {}
    return {**SAMPLE_EXTRAS, **public}


def frozen_dir(cfg):
    version = cfg["version"]
    if not version or Path(version).name != version or version in (".", ".."):
        raise ValueError("version must be a simple directory name")
    return Path(cfg["path"]) / "frozen" / version


def package(cfg, model, member="local", shard_index=0, num_shards=1):
    source, _, spec = job_spec(cfg, model, member, shard_index, num_shards)
    target = Path(cfg["path"]) / "exports" / f"{member}-{safe_name(model)}-{shard_index:05d}-of-{num_shards:05d}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with locked(source / ".writer.lock"):
        if target.exists():
            raise ValueError(f"bundle already exists: {target}; use a new run/member name")
        if json.loads((source / "run.json").read_text())["run_id"] != digest(spec):
            raise ValueError("bundle run configuration mismatch")
        if not (source / "responses.jsonl").exists():
            raise ValueError("no responses to package")
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            tmp = Path(tmp)
            for name in ("run.json", "responses.jsonl", *(f"{c}.jsonl" for c in CHANNELS)):
                if (source / name).exists():
                    shutil.copyfile(source / name, tmp / name)
            seal(tmp, {"protocol": PROTOCOL, "dataset_id": spec["dataset_id"], "run_id": digest(spec)})
            tmp.rename(target)
    return target


def _cost(model, metadata, definition):
    values = [metadata.get("prompt_tokens"), metadata.get("completion_tokens")]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("successful generations require nonnegative integer prompt/completion token usage")
    p, c = values
    if definition == "usd_tokens":
        price = model["pricing"]
        rates = [price["input_per_1m"], price["output_per_1m"]]
        scale = 1e6
    elif definition == "token_proxy":
        rates = [model["cost_per_1k_tokens"]] * 2
        scale = 1e3
    else:
        raise ValueError("cost.definition must be usd_tokens or token_proxy")
    if any(not math.isfinite(v) or v < 0 for v in rates):
        raise ValueError("missing/invalid price; zero must be an explicit choice")
    return p * rates[0] / scale, c * rates[1] / scale


def _ingest(cfg, rows, dataset, bundles):
    models = model_pool(cfg)
    by_id = {r["id"]: r for r in rows}
    responses, scores, evaluator_ids = {}, {c: {} for c in channels(cfg)}, {}
    for bundle in bundles:
        man = verify(bundle, ("run.json", "responses.jsonl"))
        run = json.loads((bundle / "run.json").read_text())
        spec = run["spec"]
        name = spec["model"]["name"]
        if man.get("protocol") != PROTOCOL or man.get("dataset_id") != dataset["dataset_id"] or man.get("run_id") != digest(spec) or run["run_id"] != digest(spec):
            raise ValueError(f"{bundle}: dataset/run identity mismatch")
        if name not in models or spec["model"] != models[name] or spec["generation"] != generation_config(cfg, models[name]) or spec["seed"] != cfg["seed"]:
            raise ValueError(f"{bundle}: model/generation configuration mismatch")
        nshard, shard = spec["num_shards"], spec["shard_index"]
        if nshard < 1 or not 0 <= shard < nshard:
            raise ValueError("invalid bundle shard")
        records = read_jsonl(bundle / "responses.jsonl")
        for r in records:
            k = key(r)
            qid, model, sid = k
            if qid not in by_id or model != name or sid >= spec["generation"]["k"] or r["prompt"] != by_id[qid]["query"]:
                raise ValueError(f"{bundle}: unknown query/model/sample or prompt mismatch: {k}")
            expected_seed = int(digest([spec["seed"], qid, model, sid])[:8], 16) % (2**31)
            if (int(digest(qid)[:16], 16) % nshard != shard or r["dataset_id"] != dataset["dataset_id"]
                    or r["run_id"] != run["run_id"] or r["generation"] != spec["generation"] or r["seed"] != expected_seed
                    or r["response_hash"] != response_hash(r)):
                raise ValueError(f"{bundle}: response provenance mismatch: {k}")
            if type(r["success"]) is not bool or (r["success"] and not isinstance(r["response"], str)):
                raise ValueError("invalid response status/text")
        for k, r in successful(records).items():
            if k in responses:
                raise ValueError(f"overlapping member/shard contributions: {k}")
            responses[k] = {**r, "bundle": str(bundle)}
        for channel in scores:
            records = read_jsonl(bundle / f"{channel}.jsonl")
            for r in records:
                k = key(r)
                bound = responses.get(k)
                if (not bound or bound["bundle"] != str(bundle) or not bound["success"]
                        or bound["response_hash"] != r["response_hash"] or r["dataset_id"] != dataset["dataset_id"]
                        or r["score_hash"] != digest({a: b for a, b in r.items() if a != "score_hash"})
                        or r["evaluator_id"] != digest(r["evaluator"]) or r["evaluator"]["scorer"] != channel):
                    raise ValueError(f"{bundle}: orphan/stale score or score hash mismatch: {k}")
                expected = dict(cfg.get("scoring", {}).get(channel, {}))
                policy = expected.pop("text_policy", "strip_think")
                if r["evaluator"] != {"scorer": channel, "options": expected, "text_policy": policy, "protocol": PROTOCOL}:
                    raise ValueError(f"{bundle}: evaluator configuration mismatch")
                if r["success"]:
                    value = r["score"]
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or (channel == "ground_truth" and not 0 <= value <= 1):
                        raise ValueError("invalid quality score")
                old_id = evaluator_ids.setdefault(channel, r["evaluator_id"])
                if old_id != r["evaluator_id"]:
                    raise ValueError("mixed evaluator versions")
            for k, r in successful(records, "score_hash").items():
                if k in scores[channel]:
                    raise ValueError(f"duplicate score: {k}")
                scores[channel][k] = r
    return responses, scores


def derive_wide(long, queries, names, quality=("ground_truth", "armorm")):
    long = long.copy()
    for column in (*quality, "cost"):
        long[column] = pd.to_numeric(long[column], errors="raise").astype(float)
    wide = pd.DataFrame([{k: r[k] for k in ("id", "query", "source", "split")} for r in queries]).set_index("id")
    for m in names:
        block = long[long.model == m]
        g = block.groupby("query_id")
        wide[m + "__n_samples"] = g.size().reindex(wide.index, fill_value=0)
        for metric in (*quality, "cost"):
            for stat in ("mean", "max"):
                wide[m + "__" + metric + "_" + stat] = getattr(g[metric], stat)().reindex(wide.index)
            first = block[block.sample_id == 0].set_index("query_id")[metric]
            wide[m + "__" + metric + "_first"] = first.reindex(wide.index)
        wide[m + "__cost_sum"] = g["cost"].sum(min_count=1).reindex(wide.index)
    return wide.reset_index()


def assemble(cfg):
    rows, dataset = load_dataset(cfg)
    models = model_pool(cfg)
    target = frozen_dir(cfg)
    if target.exists():
        raise ValueError(f"immutable release exists: {target}; use a new benchmark version/path")
    import glob
    patterns = cfg.get("contribution_paths", [str(Path(cfg["path"]) / "exports" / "*")])
    bundles = sorted({Path(p).resolve() for pattern in patterns for p in glob.glob(pattern) if Path(p).is_dir()})
    if not bundles:
        raise ValueError("no packaged contributions found; run package-contribution first")
    responses, scores = _ingest(cfg, rows, dataset, bundles)
    required = cfg.get("required_scores", ["ground_truth", "armorm"])
    if set(required) - set(scores):
        raise ValueError("unknown required scoring channel")
    missing = []
    allowed_missing = {c: 0 for c in scores}
    for q in rows:
        for m in models:
            for sid in range(generation_config(cfg, models[m])["k"]):
                k = q["id"], m, sid
                r = responses.get(k)
                if not r or not r["success"]:
                    missing.append((k, "generation"))
                    continue
                for channel in required:
                    if not is_scored(channel, q):
                        continue
                    record = scores[channel].get(k, {})
                    if allowed_missing_score(cfg, channel, record):
                        allowed_missing[channel] += 1
                    elif not record.get("success"):
                        missing.append((k, channel))
    if missing and not cfg.get("allow_partial", False):
        raise ValueError(f"incomplete collection: {len(missing)} missing/failed items, e.g. {missing[:3]}")
    definition = cfg["cost"]["definition"]
    records = []
    query_lookup = {q["id"]: q for q in rows}
    for k, r in sorted(responses.items()):
        costs = _cost(models[k[1]], r["metadata"], definition) if r["success"] else (None, None)
        q = query_lookup[k[0]]
        records.append({"query_id": k[0], "model": k[1], "sample_id": k[2], "split": q["split"], "source": q["source"],
                        "prompt": r["prompt"], "response": r["response"], "success": r["success"], "seed": r["seed"],
                        "response_hash": r["response_hash"], "dataset_id": dataset["dataset_id"], "run_id": r["run_id"],
                        "model_spec_json": json.dumps(models[k[1]], sort_keys=True),
                        "prompt_tokens": r["metadata"].get("prompt_tokens"),
                        "completion_tokens": r["metadata"].get("completion_tokens"),
                        "seconds": r["metadata"].get("batch_seconds_per_sample"),
                        "input_cost": costs[0], "output_cost": costs[1],
                        "cost": sum(costs) if costs[0] is not None else None,
                        "metadata_json": json.dumps(r["metadata"], sort_keys=True),
                        "generation_json": json.dumps(r["generation"], sort_keys=True),
                        **{c: scores[c][k]["score"] if scores[c].get(k, {}).get("success") else None for c in scores}})
    long = pd.DataFrame(records)
    if long.empty:
        raise ValueError("no response records")
    names = list(models)
    max_k = max(generation_config(cfg, m)["k"] for m in models.values())
    shape = len(rows), len(names), max_k
    arrays = {c: np.full(shape, np.nan, dtype=np.float32) for c in scores}
    costs = np.full(shape[:2], np.nan, dtype=np.float32)
    per_sample = {name: np.full(shape, np.nan, dtype=np.float32) for name in sample_columns(cfg)}
    qi, mi = {r["id"]: i for i, r in enumerate(rows)}, {m: j for j, m in enumerate(names)}
    for r in records:
        i, j, sid = qi[r["query_id"]], mi[r["model"]], r["sample_id"]
        for c in arrays:
            if r[c] is not None:
                arrays[c][i, j, sid] = r[c]
        for name, column in sample_columns(cfg).items():
            if r[column] is not None:
                per_sample[name][i, j, sid] = r[column]
        if sid == 0 and r["cost"] is not None:
            costs[i, j] = r["cost"]  # realized sample 0, never mean cost of a different sample
    wide = derive_wide(long, rows, names, list(scores))
    target.parent.mkdir(parents=True, exist_ok=True)
    with locked(target.parent / ".assemble.lock"):
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            tmp = Path(tmp)
            long.to_parquet(tmp / "long.parquet", index=False)
            wide.to_parquet(tmp / "wide.parquet", index=False)
            shutil.copytree(Path(cfg["path"]) / "dataset", tmp / "dataset")
            for i, bundle in enumerate(bundles):
                shutil.copytree(bundle, tmp / "inputs" / f"bundle-{i:05d}", ignore=shutil.ignore_patterns(".cache"))
            write_json(tmp / "config.json", cfg)
            for channel, values in arrays.items():
                # Each view carries the other quality channel (a best-of-n selector, e.g. ArmoRM picking
                # the sample whose ground truth is then scored) and the per-sample cost/length/latency.
                extras = {**per_sample, **{other: v for other, v in arrays.items() if other != channel}}
                b = Benchmark(cfg["name"], [r["id"] for r in rows], [r["query"] for r in rows], names, values, costs,
                              np.array([r["split"] for r in rows]), sources=[r["source"] for r in rows],
                              references=[r["reference"] for r in rows],
                              meta={"protocol": PROTOCOL, "version": cfg["version"], "dataset_id": dataset["dataset_id"],
                                    "metric": channel, "cost_unit": "USD" if definition == "usd_tokens" else "token_proxy",
                                    "cost_definition": cfg["cost"], "ragged": True, "missing_samples": "NaN padding; never a negative label"},
                              extras=extras)
                b.save(str(tmp / "views" / channel), with_responses=False)
            seal(tmp, {"protocol": PROTOCOL, "version": cfg["version"], "dataset_id": dataset["dataset_id"],
                       "models": names, "channels": list(scores), "config_hash": digest(release_spec(cfg)), "n_rows": len(long), "missing_items": len(missing),
                       "content_hash": _long_content_hash(long), "environment": environment(),
                       "allowed_missing_scores": allowed_missing})
            validate_release(tmp)
            tmp.rename(target)
    return load_view(cfg)


def _long_content_hash(frame):
    return digest([{k: None if isinstance(v, float) and math.isnan(v) else v for k, v in r.items()}
                   for r in frame.to_dict("records")])


def validate_release(path):
    path = Path(path)
    man = verify(path, ("long.parquet", "wide.parquet", "config.json", "dataset/queries.jsonl", "dataset/splits.json"))
    long = pd.read_parquet(path / "long.parquet")
    if long.duplicated(["query_id", "model", "sample_id"]).any():
        raise ValueError("duplicate long-table sample keys")
    if len(long) != man["n_rows"] or _long_content_hash(long) != man["content_hash"]:
        raise ValueError("long-table content hash/row count mismatch")
    quality = man.get("channels", ["ground_truth", "armorm"])  # releases before bartscore had these two
    for column in (*quality, "cost"):
        long[column] = pd.to_numeric(long[column], errors="raise").astype(float)
    wide = pd.read_parquet(path / "wide.parquet")
    queries = read_jsonl(path / "dataset" / "queries.jsonl")
    if digest(queries) != man["dataset_id"] or wide.id.tolist() != [q["id"] for q in queries]:
        raise ValueError("dataset/wide row identity mismatch")
    splits = json.loads((path / "dataset" / "splits.json").read_text())
    if splits != {s: [q["id"] for q in queries if q["split"] == s] for s in ("train", "val", "test")}:
        raise ValueError("frozen split manifest mismatch")
    query_lookup = {q["id"]: q for q in queries}
    for r in long.to_dict("records"):
        q = query_lookup.get(r["query_id"])
        if not q or r["prompt"] != q["query"] or r["split"] != q["split"] or r["source"] != q["source"]:
            raise ValueError("long-table prompt/source/split mismatch")
    try:
        pd.testing.assert_frame_equal(wide, derive_wide(long, queries, man["models"], quality), check_dtype=False)
    except AssertionError as e:
        raise ValueError("wide table does not match long-table aggregates") from e
    cfg = json.loads((path / "config.json").read_text())
    if digest(release_spec(cfg)) != man["config_hash"]:
        raise ValueError("resolved configuration hash mismatch")
    for channel in quality:
        b = Benchmark.load(str(path / "views" / channel))
        if (b.models != man["models"] or b.query_ids != [q["id"] for q in queries]
                or b.queries != [q["query"] for q in queries] or b.references != [q["reference"] for q in queries]
                or list(b.split_of) != [q["split"] for q in queries]):
            raise ValueError("view/model/split mismatch")
        qi = {q: i for i, q in enumerate(b.query_ids)}
        mi = {m: j for j, m in enumerate(b.models)}
        for r in long.to_dict("records"):
            i, j, s = qi[r["query_id"]], mi[r["model"]], r["sample_id"]
            if not np.isclose(b.scores[i, j, s], r[channel], equal_nan=True):
                raise ValueError("long/view quality mismatch")
            if s == 0 and not np.isclose(b.costs[i, j], r["cost"], equal_nan=True):
                raise ValueError("long/view cost mismatch")
            expected = {**sample_columns(cfg), **{c: c for c in quality if c != channel}}
            for name, column in expected.items():
                value = r.get(column)
                if not np.isclose(b.extras[name][i, j, s], np.nan if value is None else value, equal_nan=True):
                    raise ValueError(f"long/view extra mismatch: {name}")
    return man


def model_info(models):
    """Static model facts and frozen tariffs, independent of observed response quality."""
    keys = ("model_id", "revision", "parameters_b", "pricing", "cost_per_1k_tokens")
    return {name: {k: spec[k] for k in keys if k in spec} for name, spec in models.items()}


def load_view(cfg, with_responses=False):
    target = frozen_dir(cfg)
    man = verify(target, ("long.parquet", "wide.parquet", "config.json"))
    if man["config_hash"] != digest(release_spec(cfg)):
        raise ValueError("release configuration mismatch; only quality_metric may change when reading")
    metric = cfg.get("quality_metric", "ground_truth")
    available = man.get("channels", ["ground_truth", "armorm"])
    if metric not in available:
        raise ValueError(f"quality_metric must be one of this release's channels {available}")
    bench = Benchmark.load(str(target / "views" / metric))
    # Static facts about each model, from the pinned pool the release was built with (config_hash
    # checked above), for routers that describe models (GraphRouter's LLM nodes). Not in the fingerprint.
    bench.meta["model_info"] = model_info(model_pool(cfg))
    if with_responses:
        # Text lives once in the canonical long view, rather than duplicated in both metric views.
        models = model_pool(cfg)
        texts = {m: [[None] * generation_config(cfg, models[m])["k"] for _ in bench.query_ids] for m in bench.models}
        qi = {q: i for i, q in enumerate(bench.query_ids)}
        records = pd.read_parquet(target / "long.parquet", columns=["query_id", "model", "sample_id", "response"])
        for qid, model, sample, response in records.itertuples(index=False, name=None):
            texts[model][qi[qid]][sample] = response if isinstance(response, str) else None
        bench.responses = texts
    return bench
