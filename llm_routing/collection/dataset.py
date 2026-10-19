"""Build queries ONCE, before any member generates answers. Never silently overwrite."""
from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

from . import PROTOCOL
from .io import append, digest, file_hash, locked, read_jsonl, seal, verify, write_json


def model_pool(cfg):
    models = cfg["models"]
    names = [m["name"] for m in models]
    if not models or len(names) != len(set(names)):
        raise ValueError("model names must be nonempty and unique")
    from ..data.benchmark import safe_name
    if any(not isinstance(n, str) or not n.strip() or not safe_name(n).strip(".") for n in names):
        raise ValueError("invalid model name")
    if len({safe_name(n) for n in names}) != len(names):
        raise ValueError("model names collide after filename normalization")
    if cfg.get("cost", {}).get("definition") == "usd_tokens":
        for m in models:
            price = m.get("pricing", {})
            for key in ("input_per_1m", "output_per_1m"):
                rate = price.get(key)
                if type(rate) not in (float, int) or not math.isfinite(rate) or rate < 0:
                    raise ValueError(f"{m['name']}: missing/invalid pricing.{key}")
    return {m["name"]: m for m in models}


def specification(cfg):
    spec = {k: cfg[k] for k in ("name", "version", "sources", "splits", "seed")}
    # How splits are drawn is part of the dataset's identity, like the ratios.
    spec.update(near_duplicates=cfg.get("near_duplicates", {"method": "exact"}), ood_datasets=cfg.get("ood_datasets", []))
    return spec


def dataset_dir(cfg):
    return Path(cfg["path"]) / "dataset"


def load_dataset(cfg):
    path = dataset_dir(cfg)
    man = verify(path, ("queries.jsonl", "splits.json", "config.json"))
    if man["protocol"] != PROTOCOL or man["spec_hash"] != digest(specification(cfg)):
        raise ValueError("dataset configuration changed; create a new version/path")
    rows = read_jsonl(path / "queries.jsonl")
    if not rows or len({r["id"] for r in rows}) != len(rows) or digest(rows) != man["dataset_id"]:
        raise ValueError("invalid query IDs or dataset hash")
    return rows, man


def _load_source(src, seed):
    """Examples of one source, shuffled with the global seed. HF sources must pin a commit."""
    if "local_path" in src:
        return list(read_jsonl(src["local_path"])), {"name": src["name"], "sha256": file_hash(src["local_path"])}
    from datasets import concatenate_datasets, load_dataset as hf_load
    if not src.get("revision") or src["revision"].startswith("PIN_"):
        raise ValueError(f"{src['name']}: pin a Hugging Face dataset revision")
    configs = src.get("hf_config")
    configs = configs if isinstance(configs, list) else [configs]
    parts, fingerprints = [], []
    for c in configs:
        ds = hf_load(src["hf_id"], c, split=src.get("hf_split", "train"), revision=src["revision"],
                     data_files=src.get("data_files"))
        fingerprints.append(ds._fingerprint)
        parts.append(ds.add_column("_config", [c] * len(ds)).add_column("_row", list(range(len(ds)))))
    ds = concatenate_datasets(parts).shuffle(seed=seed)
    return list(ds), {"name": src["name"], "revision": src["revision"], "configs": configs, "fingerprints": fingerprints}


def _row(src, ex):
    """One frozen query: text, gold answer and how it is scored."""
    from .formatters import FORMATTERS
    if src.get("formatter"):
        out = dict(FORMATTERS[src["formatter"]](ex))
    else:
        from ..ground_truth import LEGACY_SCORERS
        answer_type = src.get("answer_type", LEGACY_SCORERS.get(src.get("ground_truth_scorer"), src.get("ground_truth_scorer")))
        gold = ex.get(src.get("reference_field", "reference"))
        out = {"query": src["query_template"].format_map(ex).strip(),  # missing fields are errors
               "gold": None if answer_type is None else gold, "answer_type": answer_type,
               "reference": "" if gold is None else str(gold)}
    if out["answer_type"] is not None:
        from ..ground_truth import supported_answer_types
        if out["answer_type"] not in supported_answer_types():
            raise ValueError(f"{src['name']}: unknown answer_type {out['answer_type']!r}")
        if out["gold"] in (None, "", []):
            raise ValueError(f"{src['name']}: example without a gold answer")
    return out


def build_queries(cfg, embed=None):
    """Freeze the prompt set and its splits. ``embed`` replaces the near-duplicate embedder (tests)."""
    model_pool(cfg)
    target = dataset_dir(cfg)
    if target.exists():
        rows, _ = load_dataset(cfg)
        print(f"[build-queries] frozen dataset reused: {len(rows)} queries")
        return
    fractions = cfg["splits"]
    if set(fractions) != {"train", "val", "test"} or any(v < 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("splits must contain train/val/test and sum to 1")
    rows, source_meta, seen = [], [], set()
    sources = cfg["sources"]
    if len({s["name"] for s in sources}) != len(sources):
        raise ValueError("source names must be unique")
    for src in sources:
        data, meta = _load_source(src, cfg["seed"])
        per_config = src.get("n_per_config")
        taken, counts, skipped = 0, {}, {"duplicate": 0, "too_long": 0}
        for ex in data:
            group_key = ex.get("_config")
            if per_config is not None and counts.get(group_key, 0) >= per_config:
                continue
            if per_config is None and taken >= src["n"]:
                break
            row = _row(src, ex)
            query = row["query"]
            if not query:
                raise ValueError("empty query")
            if src.get("max_query_chars") and len(query) > src["max_query_chars"]:
                skipped["too_long"] += 1  # e.g. Multi-News beyond the ArmoRM context: dropped, never truncated
                continue
            # An exact duplicate (Alpaca has some) enters once, whatever its source.
            group = re.sub(r"\s+", " ", query.lower()).strip()
            if group in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(group)
            original_id = ex.get(src["id_field"]) if src.get("id_field") else (
                f"{ex['_config']}/{ex['_row']}" if "_row" in ex else ex.get("id"))
            qid = src["name"] + "/" + digest({"id": original_id, "query": query})[:24]
            rows.append({"id": qid, "query": query, "reference": row.get("reference", ""), "gold": row["gold"],
                         "answer_type": row["answer_type"], "aux": row.get("aux") or {},
                         "gt_dataset": row.get("gt_dataset", src["name"]), "source": src["name"], "split": None,
                         "original_id": original_id})
            taken += 1
            counts[group_key] = counts.get(group_key, 0) + 1
        meta.update(n=taken, skipped=skipped)
        source_meta.append(meta)
    rows.sort(key=lambda r: r["id"])
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty dataset or duplicate query IDs")
    # Splits as in the reference repo: near-duplicate groups first, then 60/10/30 within each source.
    from .splits import group_stats, near_duplicate_groups, stratified_splits
    nd = dict(specification(cfg)["near_duplicates"])
    groups = near_duplicate_groups([r["id"] for r in rows], [r["query"] for r in rows], embed=embed, **nd)
    split_of = stratified_splits({r["id"]: r["source"] for r in rows}, fractions, cfg["seed"], groups,
                                 specification(cfg)["ood_datasets"])
    for r in rows:
        r["split"] = split_of[r["id"]]
    duplicates = group_stats(groups)
    target.parent.mkdir(parents=True, exist_ok=True)
    with locked(target.parent / ".dataset.lock"):
        if target.exists():
            raise ValueError("another process already published this dataset")
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            tmp = Path(tmp)
            for row in rows:
                append(tmp / "queries.jsonl", row)
            write_json(tmp / "splits.json", {s: [r["id"] for r in rows if r["split"] == s] for s in fractions})
            write_json(tmp / "config.json", specification(cfg))
            seal(tmp, {"protocol": PROTOCOL, "dataset_id": digest(rows), "spec_hash": digest(specification(cfg)),
                       "sources": source_meta, "n_queries": len(rows), "near_duplicates": duplicates})
            tmp.rename(target)
    print(f"[build-queries] {len(rows)} frozen queries -> {target}; near-duplicate groups: "
          f"{duplicates['n_groups_with_duplicates']} ({duplicates['n_prompts_in_duplicate_groups']} prompts)")
