import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_routing.config import load_experiment  # noqa: E402
from llm_routing.core.evaluation import aiq, curve, oracle_point, summarize  # noqa: E402
from llm_routing.data.benchmark import Benchmark, assign_splits  # noqa: E402
from llm_routing.data.synthetic import make_synthetic_benchmark  # noqa: E402
from llm_routing.pipeline.scorers import get_scorer  # noqa: E402
from llm_routing.registry import build_router, list_routers  # noqa: E402
from llm_routing.routers.baselines import OracleRouter  # noqa: E402


@pytest.fixture(scope="module")
def bench():
    return make_synthetic_benchmark(n=1500, k=4, seed=0)


def test_save_load_roundtrip(tmp_path, bench):
    bench.save(str(tmp_path))
    b2 = Benchmark.load(str(tmp_path))
    assert b2.fingerprint() == bench.fingerprint()
    assert b2.models == bench.models and np.allclose(b2.scores, bench.scores)


def test_tampering_detected(tmp_path, bench):
    bench.save(str(tmp_path))
    s = np.load(tmp_path / "scores.npy"); s[0, 0, 0] += 1
    np.save(tmp_path / "scores.npy", s)
    with pytest.raises(ValueError):
        Benchmark.load(str(tmp_path))


def test_splits_disjoint_and_deterministic(bench):
    a, b = assign_splits(1000, {"train": .7, "val": .1, "test": .2}, 3), assign_splits(1000, {"train": .7, "val": .1, "test": .2}, 3)
    assert (a == b).all() and set(a) == {"train", "val", "test"}
    ids = [set(bench.split(s).query_ids) for s in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_pair_view(bench):
    p = bench.pair("small", "large")
    assert p.small_scores.shape == (len(bench), bench.k)


def test_oracle_dominates_single_models(bench):
    test = bench.split("test")
    s = summarize(OracleRouter().sweep(test), test)
    best_single = test.realized().mean(axis=0).max()
    assert s["max_quality"] >= best_single
    assert oracle_point(test)[1] >= best_single


def test_aiq_constant_curve():
    pts = np.array([[1.0, 0.5], [3.0, 0.5]])
    assert np.isclose(aiq(pts, 1.0, 3.0), 0.5)


@pytest.mark.parametrize("cfg", [
    {"type": "hybrid_llm", "models": {"small": "small", "large": "large"}, "params": {"mode": "prob"}},
    {"type": "quality_regression", "models": None},
    {"type": "random_binary", "models": {"small": "tiny", "large": "large"}},
])
def test_every_router_produces_valid_decisions(cfg, bench):
    r = build_router(cfg, seed=0)
    from llm_routing.runner import candidate_models
    sub = bench.select_models(candidate_models(r, bench))
    r.fit(sub.split("train"), sub.split("val"))
    test = sub.split("test")
    ds = r.sweep(test, 11)
    assert len(ds) == 11
    for d in ds:
        assert d.choice.shape == (len(test),) and d.cost.shape == (len(test),)
        assert d.choice.min() >= 0 and d.choice.max() < test.num_models
    pts = curve(ds, test)
    assert pts[0, 0] <= pts[-1, 0] + 1e-9  # cheapest first


def test_registry_lists_routers():
    assert {"hybrid_llm", "oracle", "quality_regression", "random_binary"} <= set(list_routers())


def test_experiment_configs_load():
    root = os.path.join(os.path.dirname(__file__), "..", "configs", "experiments")
    for f in os.listdir(root):
        cfg = load_experiment(os.path.join(root, f), ["seed=3"])
        assert cfg["seed"] == 3 and "type" in cfg["router"] and "loader" in cfg["benchmark"]


def test_end_to_end_run(tmp_path):
    from llm_routing.runner import run_experiment
    cfg = load_experiment(os.path.join(os.path.dirname(__file__), "..", "configs", "experiments",
                                       "hybrid_llm__synthetic.yaml"),
                          [f"output_dir={tmp_path}/runs", f"benchmark.path={tmp_path}/bench", "benchmark.params.n=800"])
    rec = run_experiment(cfg)
    assert os.path.exists(os.path.join(rec["run_dir"], "metrics.json"))
    assert {r["router"] for r in rec["results"]} >= {"hybrid_llm/r_trans", "random", "oracle"}


def test_scorers():
    em = get_scorer("exact_match_number")
    assert em(["so the answer is 1,234."], ["blah #### 1234"]).tolist() == [1.0]
    mc = get_scorer("multiple_choice")
    assert mc(["The answer is B because"], ["B"]).tolist() == [1.0]


def test_router_test_pool_restricts_evaluation_and_baselines(tmp_path):
    from llm_routing.registry import register_router
    from llm_routing.routers.baselines import QualityRegressionRouter
    from llm_routing.runner import run_experiment

    @register_router("_test_pool_router")
    class PoolRouter(QualityRegressionRouter):
        def fit(self, train, val=None):
            self.test_pool = list(train.models[1:])  # trained on all, tested on unseen LLMs only
            self.models = self.test_pool
            return super().fit(train.select_models(self.test_pool), val)

        def evaluation_models(self):
            return self.test_pool

    cfg = load_experiment(os.path.join(os.path.dirname(__file__), "..", "configs", "experiments",
                                       "quality_regression__synthetic.yaml"),
                          [f"output_dir={tmp_path}/runs", f"benchmark.path={tmp_path}/bench", "benchmark.params.n=600",
                           "router.type=_test_pool_router"])
    cfg["baselines"] = ["oracle", "quality_regression"]
    rec = run_experiment(cfg)
    assert rec["models"] == list(rec["single_models"]) and len(rec["models"]) == 3
