"""Token-cost accounting and frozen-release regressions; no API or GPU calls."""
import copy

import numpy as np
import pytest

from llm_routing.config import load_benchmark_config, load_experiment
from llm_routing.contract import _shuffle_outcomes
from llm_routing.collection.assemble import _cost, assemble, frozen_dir, load_view, package, validate_release
from llm_routing.collection.collect import generate, score
from llm_routing.collection.dataset import build_queries, model_pool
from llm_routing.collection.evaluate import evaluate_data
from llm_routing.data.loaders import prepare
from llm_routing.data.benchmark import Benchmark
from llm_routing.leaderboard import build_table
from llm_routing.runner import run_experiment


@pytest.mark.parametrize("bad", [None, -1, float("nan"), float("inf"), "0.1", True])
def test_invalid_usd_prices_rejected_before_collection(bad):
    cfg = {"models": [{"name": "toy", "pricing": {"input_per_1m": 0.05, "output_per_1m": 0.33}}],
           "cost": {"definition": "usd_tokens"}}
    cfg["models"][0]["pricing"]["output_per_1m"] = bad
    with pytest.raises(ValueError, match="invalid pricing"):
        model_pool(cfg)


def test_synthetic_v3_exercises_priced_contract_without_network(tmp_path):
    cfg = load_benchmark_config("configs/benchmarks/synthetic_shared_v3.yaml", ["params.n=100"])
    b = prepare(cfg)
    p = cfg["models"][0]["pricing"]
    expected = (b.extras["prompt_tokens"][:, 0].astype(float) * p["input_per_1m"] +
                b.extras["completion_tokens"][:, 0].astype(float) * p["output_per_1m"]) / 1e6
    np.testing.assert_allclose(b.extras["sample_cost"][:, 0], expected)
    shuffled = _shuffle_outcomes(b, 123)
    np.testing.assert_array_equal(shuffled.extras["prompt_tokens"], b.extras["prompt_tokens"])
    assert not np.array_equal(shuffled.extras["completion_tokens"], b.extras["completion_tokens"])
    b.save(str(tmp_path / "bench"))
    assert Benchmark.load(str(tmp_path / "bench")).fingerprint() == b.fingerprint()


def test_usd_release_roundtrip_tariff_freeze_and_training(tmp_path):
    cfg = load_benchmark_config("configs/benchmarks/shared_smoke.yaml")
    cfg.update(name="priced_smoke", version="priced-test", path=str(tmp_path / "release"),
               cost={"definition": "usd_tokens", "kind": "estimated_api_usd", "snapshot_date": "2026-10-19"},
               input_features=["prompt_tokens"])
    # Toy tariffs test the formula independently of the real model list and dated prices.
    for model, rates in zip(cfg["models"], [(0.05, 0.33), (0.04, 0.05)]):
        model.pop("generation", None)
        model["pricing"] = dict(zip(("input_per_1m", "output_per_1m"), rates))
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
    # Toy model: 1000 * $0.05/M + 2000 * $0.33/M = $0.00071.
    assert _cost(cfg["models"][0], {"prompt_tokens": 1000, "completion_tokens": 2000}, "usd_tokens") == pytest.approx((.00005, .00066))
    np.testing.assert_allclose(b.costs[:, 0], .00071)
    np.testing.assert_array_equal(b.extras["prompt_tokens"], 1000)
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
