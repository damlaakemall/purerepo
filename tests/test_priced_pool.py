"""Frozen public tariffs; tests never call a pricing or generation API."""
import copy
import re

import numpy as np
import pytest

from llm_routing.config import load_benchmark_config, load_experiment
from llm_routing.contract import _shuffle_outcomes, check_router
from llm_routing.collection.assemble import _cost, assemble, frozen_dir, load_view, package, validate_release
from llm_routing.collection.collect import generate, score
from llm_routing.collection.dataset import build_queries, model_pool
from llm_routing.collection.evaluate import evaluate_data
from llm_routing.data.loaders import prepare
from llm_routing.data.benchmark import Benchmark
from llm_routing.leaderboard import build_table
from llm_routing.runner import run_experiment


def test_v3_pool_is_pinned_priced_and_preserves_source_protocol():
    cfg = load_benchmark_config("configs/benchmarks/shared_v3.yaml")
    old = load_benchmark_config("configs/benchmarks/shared_v2.yaml")
    assert len(model_pool(cfg)) == 12
    assert cfg["sources"] == old["sources"] and cfg["splits"] == old["splits"]
    assert cfg["generation"] == old["generation"] and cfg["scoring"] == old["scoring"]
    assert cfg["path"] != old["path"] and cfg["version"] != old["version"]
    assert len(set(m["model_id"] for m in cfg["models"]) & set(m["model_id"] for m in old["models"])) == 6
    old_models = {m["model_id"]: m for m in old["models"]}
    for m in cfg["models"]:
        if m["model_id"] in old_models:
            assert m["revision"] == old_models[m["model_id"]]["revision"]
        assert re.fullmatch("[0-9a-f]{40}", m["revision"])
        assert 3 < m["parameters_b"] < 33 and "cost_per_1k_tokens" not in m
        p = m["pricing"]
        assert p["input_per_1m"] > 0 and p["output_per_1m"] > 0
        assert p["snapshot_date"] == cfg["cost"]["snapshot_date"]
        assert p["source"].endswith(p["openrouter_id"] + "/endpoints")
        assert p["provider"] and p["endpoint"] and p["source_hf_id"]
    qwen = next(m for m in cfg["models"] if m["name"] == "qwen3-32b")
    assert qwen["chat_template_kwargs"]["enable_thinking"]


@pytest.mark.parametrize("bad", [None, -1, float("nan"), float("inf"), "0.1", True])
def test_invalid_usd_prices_rejected_before_collection(bad):
    cfg = load_benchmark_config("configs/benchmarks/shared_v3.yaml")
    cfg["models"][0]["pricing"]["output_per_1m"] = bad
    with pytest.raises(ValueError, match="invalid pricing"):
        model_pool(cfg)


def test_synthetic_v3_exercises_priced_contract_without_network(tmp_path):
    cfg = load_benchmark_config("configs/benchmarks/synthetic_shared_v3.yaml", ["params.n=100"])
    b = prepare(cfg)
    assert b.models == [m["name"] for m in cfg["models"]]
    assert b.scores.shape == (100, 12, 10)
    assert b.meta["model_info"][b.models[0]]["pricing"] == cfg["models"][0]["pricing"]
    p = cfg["models"][0]["pricing"]
    expected = (b.extras["prompt_tokens"][:, 0].astype(float) * p["input_per_1m"] +
                b.extras["completion_tokens"][:, 0].astype(float) * p["output_per_1m"]) / 1e6
    np.testing.assert_allclose(b.extras["sample_cost"][:, 0], expected)
    shuffled = _shuffle_outcomes(b, 123)
    np.testing.assert_array_equal(shuffled.extras["prompt_tokens"], b.extras["prompt_tokens"])
    assert not np.array_equal(shuffled.extras["completion_tokens"], b.extras["completion_tokens"])
    b.save(str(tmp_path / "bench"))
    assert Benchmark.load(str(tmp_path / "bench")).fingerprint() == b.fingerprint()
    assert all(ok for _, ok, _ in check_router({"type": "quality_regression", "params": {}}, b))


def test_usd_release_roundtrip_tariff_freeze_and_training(tmp_path):
    cfg = load_benchmark_config("configs/benchmarks/shared_smoke.yaml")
    cfg.update(name="priced_smoke", version="priced-test", path=str(tmp_path / "release"),
               cost={"definition": "usd_tokens", "kind": "estimated_api_usd", "snapshot_date": "2026-10-19"},
               input_features=["prompt_tokens"])
    pool = load_benchmark_config("configs/benchmarks/shared_v3.yaml")["models"]
    cfg["models"] = [copy.deepcopy(pool[i]) for i in (0, 1)]
    cfg["generation"]["k"] = 2
    build_queries(cfg)
    for m in cfg["models"]:
        generate(cfg, m["name"], backend=lambda q, seed: (str(int(q.split()[1]) + 1),
                 {"prompt_tokens": 1000, "completion_tokens": 2000}), retry_delay=0)
        score(cfg, m["name"])
        score(cfg, m["name"], scorer="armorm", evaluator=lambda q, r: (0.5, {"synthetic": True}))
        package(cfg, m["name"])
    b = assemble(cfg)
    assert validate_release(frozen_dir(cfg))["n_rows"] == 120
    # Llama3.2: 1000 * $0.05/M + 2000 * $0.33/M = $0.00071.
    assert _cost(cfg["models"][0], {"prompt_tokens": 1000, "completion_tokens": 2000}, "usd_tokens") == pytest.approx((.00005, .00066))
    np.testing.assert_allclose(b.costs[:, 0], .00071)
    np.testing.assert_array_equal(b.extras["prompt_tokens"], 1000)
    assert b.meta["model_info"][b.models[0]]["pricing"] == cfg["models"][0]["pricing"]
    changed = copy.deepcopy(cfg)
    changed["models"][0]["pricing"]["input_per_1m"] *= 2
    with pytest.raises(ValueError, match="release configuration mismatch"):
        load_view(changed)
    report = evaluate_data(cfg)
    assert report["channels"]["ground_truth"]["cost_definition"] == cfg["cost"]
    exp = load_experiment("configs/experiments/quality_regression__synthetic.yaml")
    exp.update(benchmark=cfg, output_dir=str(tmp_path / "runs"))
    record = run_experiment(exp)
    assert record["cost_unit"] == "USD" and record["cost_definition"] == cfg["cost"]
    assert "Estimated API cost (USD), frozen tariff 2026-10-19" in build_table([record])
