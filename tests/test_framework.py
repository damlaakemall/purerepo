import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_routing.config import load_experiment  # noqa: E402
from llm_routing.core.evaluation import aiq, oracle_point, summarize  # noqa: E402
from llm_routing.data.benchmark import Benchmark, assign_splits  # noqa: E402
from llm_routing.data.synthetic import make_synthetic_benchmark  # noqa: E402
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


def test_oracle_dominates_single_models(bench):
    test = bench.split("test")
    s = summarize(OracleRouter().sweep(test), test)
    best_single = test.realized().mean(axis=0).max()
    assert s["max_quality"] >= best_single
    assert oracle_point(test)[1] >= best_single


def test_aiq_constant_curve():
    pts = np.array([[1.0, 0.5], [3.0, 0.5]])
    assert np.isclose(aiq(pts, 1.0, 3.0), 0.5)


def test_end_to_end_run(tmp_path):
    from llm_routing.runner import run_experiment
    cfg = load_experiment(os.path.join(os.path.dirname(__file__), "..", "configs", "experiments",
                                       "quality_regression__synthetic.yaml"),
                          [f"output_dir={tmp_path}/runs", f"benchmark.path={tmp_path}/bench", "benchmark.params.n=800"])
    rec = run_experiment(cfg)
    assert os.path.exists(os.path.join(rec["run_dir"], "metrics.json"))
    assert {r["router"] for r in rec["results"]} >= {"quality_regression", "oracle"}


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
