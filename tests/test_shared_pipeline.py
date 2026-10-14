"""Protocol regressions: synthetic backends only, no GPU, API, downloads or router training."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_routing.config import REPO_ROOT, load_benchmark_config
from llm_routing.core.base import Decision
from llm_routing.core.evaluation import curve
from llm_routing.data.benchmark import Benchmark
from llm_routing.data.loaders import load_benchmark
from llm_routing.pipeline.shared.assemble import assemble, frozen_dir, package, validate_release
from llm_routing.pipeline.shared.collect import generate, job_spec, score
from llm_routing.pipeline.shared.dataset import build_queries, load_dataset
from llm_routing.pipeline.shared.io import digest, read_jsonl, seal, verify, write_json
from llm_routing.pipeline.shared.smoke import run_smoke
from llm_routing.routers.hybrid_llm.labels import gap_probability, find_relaxation


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return run_smoke(tmp_path_factory.mktemp("shared") / "collection")


@pytest.fixture
def cfg(tmp_path):
    c = load_benchmark_config(str(Path(REPO_ROOT) / "configs/benchmarks/shared_smoke.yaml"))
    c["path"] = str(tmp_path / "collection")
    c["sources"][0]["local_path"] = str(Path(REPO_ROOT) / "tests/fixtures/shared_queries.jsonl")
    c["sources"][0]["n"] = 5
    c["models"] = c["models"][:1]
    c["models"][0]["generation"]["k"] = 2
    build_queries(c)
    return c


def fake(query, seed):
    return str(int(query.split()[1]) + 1), {"prompt_tokens": 5, "completion_tokens": 2,
                                           "engine": "test-double", "formatted_prompt": query}


def collect(cfg):
    generate(cfg, "tiny", backend=fake, retry_delay=0)
    score(cfg, "tiny")
    score(cfg, "tiny", scorer="armorm", evaluator=lambda q, r: (0.5, {"synthetic": True}))
    return package(cfg, "tiny")


def test_smoke_roundtrip_and_ragged_views(smoke):
    cfg, report = smoke
    gt = load_benchmark(cfg, with_responses=True)
    rm = load_benchmark({**cfg, "quality_metric": "armorm"})
    assert gt.scores.shape == (30, 2, 3)
    assert np.isnan(gt.scores[:, 1, 1:]).all()
    assert gt.query_ids == rm.query_ids and np.array_equal(gt.split_of, rm.split_of)
    assert np.array_equal(gt.costs, rm.costs)
    assert gt.fingerprint() != rm.fingerprint()
    assert gt.responses["tiny"][0] and len(gt.responses["large"][0]) == 1
    man = validate_release(frozen_dir(cfg))
    assert man["n_rows"] == 120
    assert "views/ground_truth/manifest.json" in man["outputs"]
    long = pd.read_parquet(frozen_dir(cfg) / "long.parquet")
    assert long.response.notna().all() and long.metadata_json.notna().all()
    assert set(report["channels"]) == {"ground_truth", "armorm"}
    for channel in report["channels"].values():
        assert channel["n_scored"] == channel["n_test"]
        assert channel["oracle"]["quality"] >= max(p["quality"] for p in channel["single_models"].values())


def test_dataset_id_stable_when_input_reordered(cfg, tmp_path):
    rows, man = load_dataset(cfg)
    alternate = copy.deepcopy(cfg)
    p = tmp_path / "reordered.jsonl"
    source = read_jsonl(cfg["sources"][0]["local_path"])[:5]
    p.write_text("\n".join(json.dumps(r) for r in reversed(source)) + "\n")
    alternate["sources"][0]["local_path"] = str(p)
    alternate["path"] = str(tmp_path / "other")
    build_queries(alternate)
    rows2, man2 = load_dataset(alternate)
    assert rows == rows2 and man["dataset_id"] == man2["dataset_id"]


def test_existing_dataset_cannot_change(cfg):
    altered = copy.deepcopy(cfg)
    altered["seed"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        build_queries(altered)


def test_shards_are_disjoint_and_resume_seeds_are_stable(cfg):
    got = []
    for shard in range(3):
        path = generate(cfg, "tiny", num_shards=3, shard_index=shard, backend=fake)
        got += read_jsonl(path / "responses.jsonl")
        def forbidden(*args):
            pytest.fail("resume called the backend")
        generate(cfg, "tiny", num_shards=3, shard_index=shard, backend=forbidden)
    assert len(got) == len({(r["query_id"], r["sample_id"]) for r in got}) == 10
    path = generate(cfg, "tiny", member="independent", backend=fake)
    expected = {(r["query_id"], r["sample_id"]): r["seed"] for r in read_jsonl(path / "responses.jsonl")}
    assert all(expected[r["query_id"], r["sample_id"]] == r["seed"] for r in got)


def test_retry_and_resume_failure_audit(cfg):
    calls = 0
    def transient(q, seed):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("test interruption")
        return fake(q, seed)
    path = generate(cfg, "tiny", backend=transient, retry_delay=0)
    records = read_jsonl(path / "responses.jsonl")
    assert len(records) == 11 and not records[0]["success"] and records[1]["success"]
    score(cfg, "tiny")
    score(cfg, "tiny", scorer="armorm", evaluator=lambda q, r: (0.3, {}))
    package(cfg, "tiny")
    assert len(assemble(cfg)) == 5


def test_changed_generation_rejects_resume(cfg):
    generate(cfg, "tiny", backend=fake)
    cfg["generation"]["temperature"] = 0.1
    with pytest.raises(ValueError, match="resume configuration mismatch"):
        generate(cfg, "tiny", backend=fake)


def test_torn_last_line_preserved_and_resumed(cfg):
    path = generate(cfg, "tiny", backend=fake)
    with (path / "responses.jsonl").open("ab") as f:
        f.write(b'{"incomplete":')
    generate(cfg, "tiny", backend=fake)
    assert (path / "responses.jsonl.interrupted-tail").read_bytes().startswith(b'{"incomplete":')
    assert len(read_jsonl(path / "responses.jsonl")) == 10


def test_bundle_tampering_detected(cfg):
    bundle = collect(cfg)
    with (bundle / "responses.jsonl").open("a") as f:
        f.write('{}\n')
    with pytest.raises(ValueError, match="hash/path mismatch"):
        assemble(cfg)


@pytest.mark.parametrize("mutation", ["model", "prompt", "response", "seed"])
def test_resealed_wrong_response_rejected(cfg, mutation):
    bundle = collect(cfg)
    rows = read_jsonl(bundle / "responses.jsonl")
    rows[0][mutation] = 123 if mutation == "seed" else "WRONG"
    (bundle / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    man = json.loads((bundle / "manifest.json").read_text())
    seal(bundle, {k: v for k, v in man.items() if k != "outputs"})
    with pytest.raises(ValueError):
        assemble(cfg)


def test_stale_score_binding_rejected(cfg):
    bundle = collect(cfg)
    records = read_jsonl(bundle / "ground_truth.jsonl")
    records[0]["response_hash"] = "stale"
    records[0]["score_hash"] = digest({k: v for k, v in records[0].items() if k != "score_hash"})
    (bundle / "ground_truth.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    man = json.loads((bundle / "manifest.json").read_text())
    seal(bundle, {k: v for k, v in man.items() if k != "outputs"})
    with pytest.raises(ValueError, match="stale"):
        assemble(cfg)


def test_duplicate_contributions_rejected(cfg):
    bundle = collect(cfg)
    shutil.copytree(bundle, bundle.parent / "duplicate")
    with pytest.raises(ValueError, match="overlapping"):
        assemble(cfg)


def test_missing_model_or_scores_fail_publication(cfg):
    generate(cfg, "tiny", backend=fake)
    package(cfg, "tiny")
    with pytest.raises(ValueError, match="incomplete collection"):
        assemble(cfg)
    assert not frozen_dir(cfg).exists()


def test_release_cannot_be_overwritten(smoke):
    cfg, _ = smoke
    with pytest.raises(ValueError, match="immutable release exists"):
        assemble(cfg)


def test_wide_recomputation_detects_bad_aggregates(smoke, tmp_path):
    cfg, _ = smoke
    target = tmp_path / "release"
    shutil.copytree(frozen_dir(cfg), target)
    wide = pd.read_parquet(target / "wide.parquet")
    wide.loc[0, "tiny__cost_mean"] += 1
    wide.to_parquet(target / "wide.parquet", index=False)
    man = json.loads((target / "manifest.json").read_text())
    seal(target, {k: v for k, v in man.items() if k != "outputs"})
    with pytest.raises(ValueError, match="wide table"):
        validate_release(target)


def test_selected_sample_quality_and_total_cost(smoke):
    cfg, _ = smoke
    bench = load_benchmark(cfg).select_models(["tiny"])
    n = len(bench)
    sample = np.full(n, 1, dtype=int)
    d = Decision(0, np.zeros(n, dtype=int), np.full(n, 12.0), sample_index=sample)
    point = curve([d], bench)[0]
    assert point[0] == 12 and point[1] == bench.scores[:, 0, 1].mean()
    bad = Decision(0, np.zeros(n, dtype=int), np.ones(n), sample_index=np.full(n, 3))
    with pytest.raises(ValueError):
        curve([bad], bench)


def test_ragged_padding_does_not_count_as_negative_labels():
    s = np.array([[1, np.nan, np.nan], [0, 1, np.nan]])
    l = np.array([[1, 0, np.nan], [1, np.nan, np.nan]])
    assert np.allclose(gap_probability(s, l), [1, 0.5])
    assert np.allclose(gap_probability(s, l, estimator="paired"), [1, 0])
    assert np.isfinite(find_relaxation(s, l)[0])


def test_query_text_changes_fingerprint_and_load_detects_tamper(smoke, tmp_path):
    cfg, _ = smoke
    b = load_benchmark(cfg)
    before = b.fingerprint()
    b.queries = list(b.queries)
    b.queries[0] += " changed"
    assert b.fingerprint() != before
    b.save(str(tmp_path / "view"))
    p = tmp_path / "view" / "queries.jsonl"
    p.write_text(p.read_text().replace(" changed", " edited"))
    with pytest.raises(ValueError):
        Benchmark.load(str(p.parent))


def test_cli_dotlist_after_options_and_model_pool(cfg, monkeypatch):
    from llm_routing.cli import main
    from llm_routing.pipeline import build
    captured = []
    monkeypatch.setattr(build, "generate", lambda c, m, **kw: captured.append((c, m, kw)))
    main(["generate", "configs/benchmarks/shared_v1.yaml", "--model", "qwen2.5-0.5b", "--member", "alice", "generation.k=2"])
    c, m, kw = captured[0]
    assert c["generation"]["k"] == 2 and c["models"][0]["name"] == m and kw["member"] == "alice"


def test_empty_shards_can_be_packaged(cfg):
    path, rows, _ = job_spec(cfg, "tiny", "empty", 0, 1000)
    # Pick a shard containing no queries; a valid empty shard is still an artifact.
    for shard in range(1000):
        path, rows, _ = job_spec(cfg, "tiny", "empty", shard, 1000)
        if not rows:
            break
    generate(cfg, "tiny", member="empty", shard_index=shard, num_shards=1000, backend=fake)
    score(cfg, "tiny", member="empty", shard_index=shard, num_shards=1000)
    bundle = package(cfg, "tiny", member="empty", shard_index=shard, num_shards=1000)
    assert read_jsonl(bundle / "responses.jsonl") == []
    verify(bundle, ("run.json", "responses.jsonl"))


def test_partial_release_preserves_failure_and_null_cost(cfg):
    calls = 0
    def fail_once(query, seed):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("no answer")
        return fake(query, seed)
    with pytest.raises(RuntimeError, match="samples failed"):
        generate(cfg, "tiny", backend=fail_once, max_retries=0)
    score(cfg, "tiny")
    score(cfg, "tiny", scorer="armorm", evaluator=lambda q, r: (.4, {}))
    package(cfg, "tiny")
    cfg["allow_partial"] = True
    assemble(cfg)
    man = validate_release(frozen_dir(cfg))
    long = pd.read_parquet(frozen_dir(cfg) / "long.parquet")
    assert man["missing_items"] > 0
    assert long.loc[~long.success, "cost"].isna().all()
    assert long.loc[~long.success, "ground_truth"].isna().all()


def test_unavailable_gold_stays_null(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["path"] = str(tmp_path / "no-gold")
    cfg["sources"][0]["ground_truth_scorer"] = None
    build_queries(cfg)
    collect(cfg)
    gt = assemble(cfg)
    assert np.isnan(gt.scores).all()
    rm = load_benchmark({**cfg, "quality_metric": "armorm"})
    assert np.isfinite(rm.scores).all()


def test_unknown_token_usage_cannot_be_published(cfg):
    def missing_usage(q, seed):
        return "answer", {"prompt_tokens": 5}
    generate(cfg, "tiny", backend=missing_usage)
    score(cfg, "tiny")
    score(cfg, "tiny", scorer="armorm", evaluator=lambda q, r: (.4, {}))
    package(cfg, "tiny")
    with pytest.raises(ValueError, match="token usage"):
        assemble(cfg)


def test_release_can_move_to_another_storage_root(smoke, tmp_path):
    cfg, _ = smoke
    moved = {**cfg, "path": str(tmp_path / "moved")}
    target = frozen_dir(moved)
    target.parent.mkdir(parents=True)
    shutil.copytree(frozen_dir(cfg), target)
    assert load_benchmark(cfg).fingerprint() == load_benchmark(moved).fingerprint()
