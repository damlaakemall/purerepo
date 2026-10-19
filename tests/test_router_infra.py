"""Shared pieces a paper router builds on: dev data, information boundaries, evaluation, helpers."""

import json
import os

import numpy as np
import pytest

from llm_routing.config import load_experiment
from llm_routing.core.base import BinaryRouter, Decision, Router
from llm_routing.core.evaluation import summarize
from llm_routing.data.synthetic import make_synthetic_benchmark
from llm_routing.registry import list_routers, register_router
from llm_routing.routers.baselines import QualityRegressionRouter
from llm_routing.runner import run_experiment, val_threshold_metrics

ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCES = {"gsm8k": "binary", "mbpp": "binary", "alpaca": "continuous"}


def shared_like(**kw):
    return make_synthetic_benchmark(**{"n": 600, "k": 4, "seed": 0, "sources": SOURCES, "missing_rate": 0.02, **kw})


def test_shared_like_benchmark_has_the_release_shape():
    b = shared_like()
    src = np.array(b.sources)
    binary = b.scores[np.isin(src, ["gsm8k", "mbpp"])]
    assert set(np.unique(binary[np.isfinite(binary)])) <= {0.0, 1.0}
    cont = b.scores[src == "alpaca"]
    assert np.nanmin(cont) >= 0 and np.nanmax(cont) <= 1 and len(np.unique(cont[np.isfinite(cont)])) > 2
    assert np.isnan(b.realized()).any() and b.extras["sample_cost"].shape == b.scores.shape
    assert all(q.startswith(s + ":") for q, s in zip(b.queries, b.sources))
    assert make_synthetic_benchmark(n=50, k=2).sources == ["synthetic"] * 50  # default data unchanged


def _experiment(tmp_path, router_type, n=600):
    cfg = load_experiment(os.path.join(ROOT, "configs", "experiments", "quality_regression__synthetic_shared_v3.yaml"),
                          [f"output_dir={tmp_path}/runs", f"benchmark.path={tmp_path}/bench", f"benchmark.params.n={n}",
                           "benchmark.params.k=2", f"router.type={router_type}"])
    cfg["baselines"] = ["oracle", "quality_regression"]
    return cfg


SEEN = {}


@register_router("_test_dynamic_pool")
class DynamicPoolRouter(QualityRegressionRouter):
    """Trains on the first half of the pool, is tested on the second half (UniRoute's protocol)."""

    def __init__(self, models=None, **kw):
        super().__init__(None, **kw)

    def training_models(self):
        return ["llama-3.2-3b", "gemma-3-4b", "qwen2.5-7b", "llama-3.1-8b", "mistral-nemo-12b", "gemma-3-12b"]

    def evaluation_models(self):
        return ["qwen3-14b", "phi-4", "mistral-small-24b", "gemma-3-27b", "qwen2.5-coder-32b", "qwen3-32b"]

    def fit(self, train, val=None):
        SEEN["train_models"], SEEN["val_models"] = list(train.models), list(val.models)
        # a stand-in for UniRoute: profile the new models on VAL, never on TRAIN
        self.models = self.evaluation_models()
        return super().fit(val.select_models(self.models), None)


def test_dynamic_pool_router_and_baselines_never_see_new_models_train_labels(tmp_path):
    rec = run_experiment(_experiment(tmp_path / "a", "_test_dynamic_pool"))
    assert SEEN["train_models"] == DynamicPoolRouter().training_models()
    assert len(SEEN["val_models"]) == 12
    assert rec["models"] == DynamicPoolRouter().evaluation_models() == list(rec["single_models"])
    assert {r["fit_on"] for r in rec["results"] if r["is_baseline"]} == {"val"}

    # perturb the new models' TRAIN scores in the frozen data: every result must stay identical
    from llm_routing.data.benchmark import Benchmark

    bench = Benchmark.load(str(tmp_path / "a" / "bench"))
    rows = bench.split_of == "train"
    new = [bench.model_index(m) for m in DynamicPoolRouter().evaluation_models()]
    scores = bench.scores.copy()
    scores[np.ix_(rows, new)] = 1.0 - scores[np.ix_(rows, new)]
    bench.scores = scores
    bench.save(str(tmp_path / "b" / "bench"))
    rec2 = run_experiment(_experiment(tmp_path / "b", "_test_dynamic_pool"))
    strip = lambda r: {k: v for k, v in r.items() if k != "fit_seconds"}  # noqa: E731
    assert [strip(r) for r in rec["results"]] == [strip(r) for r in rec2["results"]]


def test_leaderboard_keeps_dynamic_pool_runs_apart(tmp_path):
    from llm_routing.leaderboard import build_table

    rec = run_experiment(_experiment(tmp_path, "_test_dynamic_pool"))
    static = json.loads(json.dumps(rec))
    static["training_models"] = None
    table = build_table([rec, static])
    assert table.count("### ") == 2 and "trained on llama-3.2-3b" in table


class ScoreRouter(BinaryRouter):
    """Deterministic score that reads only the query text."""

    def __init__(self):
        super().__init__("small", "large")

    def score(self, queries):
        return np.array([1.0 - sum(w in q for w in ("prove", "theorem", "derive", "quantum")) / 4 for q in queries])


def test_val_threshold_is_chosen_on_val_and_ignores_test_labels():
    from llm_routing.collection.evaluate import complete_cohort

    b = shared_like(n=1500).select_models(["small", "large"])
    val, test = complete_cohort(b.split("val")), complete_cohort(b.split("test"))
    m = val_threshold_metrics(ScoreRouter(), val, test)
    assert m["val_quality_drop_pct"] <= 1.0 or m["val_small_share"] == 0.0
    test.scores = np.where(np.isfinite(test.scores), 1.0 - test.scores, test.scores)
    m2 = val_threshold_metrics(ScoreRouter(), val, test)
    assert m2["val_threshold"] == m["val_threshold"] and m2["test@val_threshold_small_share"] == m["test@val_threshold_small_share"]


def test_single_operating_point_has_no_aiq():
    b = shared_like().select_models(["tiny", "large"])
    b = b.subset(np.flatnonzero(np.isfinite(b.realized()).all(1)))
    pick = np.zeros(len(b), dtype=int)
    s = summarize([Decision(0.0, pick, b.costs[:, 0])], b)
    assert s["aiq"] is None and s["max_quality"] == pytest.approx(float(b.realized()[:, 0].mean()))


def test_transformer_classifier_keeps_best_val_epoch(monkeypatch):
    torch = pytest.importorskip("torch")
    from llm_routing.core.classifiers import TransformerClassifier

    class Enc(dict):
        def to(self, _):
            return self

    class Tok:
        def __call__(self, texts, **_):
            return Enc(x=torch.tensor([[float(len(t))] for t in texts]))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(1, 1)

        def forward(self, x):
            return type("Out", (), {"logits": self.w(x)})

    def setup(self):
        torch.manual_seed(0)
        self.device, self.tokenizer, self.model = "cpu", Tok(), Net()

    monkeypatch.setattr(TransformerClassifier, "_setup", setup)
    texts = ["a", "bb", "ccc", "dddd"] * 4
    clf = TransformerClassifier(epochs=6, batch_size=4, lr=0.5, warmup_ratio=0.0)
    # val labels are the opposite of train labels: later epochs fit train and get worse on val
    clf.fit(texts, np.array([0, 0, 1, 1] * 4, dtype=float), val_texts=texts, val_labels=np.array([1, 1, 0, 0] * 4, dtype=float))
    assert 1 <= clf.best_epoch_ <= 6
    assert clf.bce(texts, np.array([1, 1, 0, 0] * 4, dtype=float)) == pytest.approx(clf.best_val_bce_, rel=1e-5)


def test_embedding_pooling_and_cache(tmp_path, monkeypatch):
    from llm_routing.core import embeddings

    hidden = np.array([[[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]]])
    mask = np.array([[1, 1, 0]])
    assert np.allclose(embeddings.pool(hidden, mask, "mean"), [[2.0, 1.0]])
    assert np.allclose(embeddings.pool(hidden, mask, "cls"), [[1.0, 0.0]])

    calls = []

    def fake(texts, **kw):
        calls.append(kw["model_name"])
        return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(embeddings, "_encode", fake)
    a = embeddings.encode(["x", "y"], "m", cache_dir=str(tmp_path))
    b = embeddings.encode(["x", "y"], "m", cache_dir=str(tmp_path))
    embeddings.encode(["x", "y"], "m", pooling="cls", cache_dir=str(tmp_path))
    assert np.array_equal(a, b) and len(calls) == 2


def test_label_manifest_identifies_data_and_policy():
    from llm_routing.core.labels import label_manifest

    b = shared_like(n=60)
    m = label_manifest(b, {"variant": "r_prob", "tie": ">="}, labels=np.zeros(60))
    m2 = label_manifest(b, {"tie": ">=", "variant": "r_prob"})
    assert m["fingerprint"] == b.fingerprint() and m["policy_hash"] == m2["policy_hash"] and m["num_labels"] == 60


def test_dev_data_carries_model_info():
    from llm_routing.config import load_benchmark_config
    from llm_routing.data.loaders import prepare

    b = prepare(load_benchmark_config(os.path.join(ROOT, "configs", "benchmarks", "synthetic_shared.yaml"),
                                      ["params.n=200", "params.k=2"]))
    pool = load_benchmark_config(os.path.join(ROOT, "configs", "benchmarks", "shared_v2.yaml"))["models"]
    assert b.models == [m["name"] for m in pool]  # router configs run unchanged on dev and real data
    assert b.meta["model_info"]["qwen2.5-32b"]["cost_per_1k_tokens"] == 32


@register_router("_test_reads_test_labels")
class CheatingRouter(Router):
    def __init__(self, models=None, **_):
        self.models = models

    def sweep(self, bench, num=51):
        pick = np.nan_to_num(bench.realized()).argmax(1)
        return [Decision(0.0, pick, bench.costs[np.arange(len(bench)), pick])]


@pytest.mark.parametrize("cfg,expect", [
    ({"type": "quality_regression", "models": None}, True),
    ({"type": "random_binary", "models": {"small": "gemma-3-4b", "large": "gemma-3-27b"}}, True),
    ({"type": "_test_dynamic_pool", "models": None}, True),
    ({"type": "_test_reads_test_labels", "models": None}, False),
])
def test_check_router(cfg, expect):
    from llm_routing.config import load_benchmark_config
    from llm_routing.contract import check_router
    from llm_routing.data.loaders import prepare

    # the default check-router data: the v3 pool's names, so router configs for shared_v3 run unchanged
    b = prepare(load_benchmark_config(os.path.join(ROOT, "configs", "benchmarks", "synthetic_shared_v3.yaml"),
                                      ["params.n=600", "params.k=2"]))
    results = check_router(cfg, b)
    assert all(ok for _, ok, _ in results) == expect, results


@register_router("_test_reads_extra_labels")
class ExtraLabelsRouter(Router):
    def __init__(self, models=None, **_):
        pass

    def sweep(self, bench, num=51):
        pick = bench.extras["armorm"][:, :, 0].argmax(1)
        return [Decision(0.0, pick, bench.costs[np.arange(len(bench)), pick])]


@register_router("_test_numpy_description")
class NumpyDescriptionRouter(Router):
    def __init__(self, models=None, **_):
        pass

    def sweep(self, bench, num=51):
        return [Decision(0.0, np.zeros(len(bench), dtype=int), bench.costs[:, 0])]

    def describe(self):
        return {"profile": np.array([0.1, 0.2])}


def test_check_router_rejects_extra_channel_leak_and_numpy_description():
    from llm_routing.contract import check_router

    bench = shared_like()
    bench.extras["armorm"] = np.random.default_rng(4).random(bench.scores.shape).astype(np.float32)
    results = {name: ok for name, ok, _ in check_router({"type": "_test_reads_extra_labels"}, bench)}
    assert results["decisions"] and not results["no test labels"]
    results = {name: ok for name, ok, _ in check_router({"type": "_test_numpy_description"}, bench)}
    assert results["decisions"] and not results["describe"]


def test_checker_and_runner_use_the_same_dynamic_cohorts(tmp_path):
    from llm_routing.contract import check_router
    from llm_routing.data.benchmark import Benchmark

    cfg = _experiment(tmp_path, "_test_dynamic_pool")
    from llm_routing.data.loaders import prepare
    bench = prepare(cfg["benchmark"])
    train_models = DynamicPoolRouter().training_models()
    new_models = DynamicPoolRouter().evaluation_models()
    old = [bench.model_index(m) for m in train_models]
    new = [bench.model_index(m) for m in new_models]
    # Neither the fit cohort nor its cost estimates may depend on hidden train cells.
    for values in (bench.scores, bench.costs, *bench.extras.values()):
        values[np.ix_(bench.split_of == "train", new)] = np.nan
        values[np.ix_(bench.split_of == "test", old)] = np.nan
    bench.save(cfg["benchmark"]["path"])
    results = check_router({"type": "_test_dynamic_pool"}, Benchmark.load(cfg["benchmark"]["path"]))
    assert all(ok for _, ok, _ in results), results
    rec = run_experiment(cfg)
    assert rec["num_test"] > 0
    assert rec["models"] == new_models
    assert all(rec["coverage"][s]["cohort_hash"] for s in ("train", "val", "test"))


def test_leaderboard_separates_different_cohorts():
    from llm_routing.leaderboard import build_table

    base = {"benchmark": "b", "fingerprint": "fp", "models": ["m"], "seed": 0,
            "results": [{"router": "r", "aiq": 0.5}],
            "coverage": {"test": {"query_ids": ["q1"]}}}
    other = {**base, "coverage": {"test": {"query_ids": ["q2"]}}}
    assert build_table([base, other]).count("### ") == 2


def test_collected_release_can_train_without_collection_backends(tmp_path, monkeypatch):
    from llm_routing.collection.smoke import run_smoke
    from llm_routing.collection import collect

    benchmark, _ = run_smoke(str(tmp_path / "release"))

    def forbidden(*args, **kwargs):
        raise AssertionError("training tried to generate or score responses")

    monkeypatch.setattr(collect, "generate", forbidden)
    monkeypatch.setattr(collect, "score", forbidden)
    cfg = _experiment(tmp_path, "quality_regression")
    cfg["benchmark"] = benchmark
    rec = run_experiment(cfg)
    assert rec["models"] == ["tiny", "large"]
    assert rec["num_train"] == 18 and rec["num_test"] == 6  # smoke config uses 60/20/20
    assert rec["quality_metric"] == "ground_truth" and rec["cost_unit"] == "token_proxy"
    assert os.path.exists(os.path.join(rec["run_dir"], "metrics.json"))
