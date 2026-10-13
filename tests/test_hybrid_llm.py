import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_routing.core.metrics import area_under_curve, evaluate_decisions, sweep  # noqa: E402
from llm_routing.data.synthetic import make_synthetic  # noqa: E402
from llm_routing.routers.baselines import RandomRouter  # noqa: E402
from llm_routing.routers.hybrid_llm import (  # noqa: E402
    HybridLLMRouter,
    deterministic_labels,
    find_relaxation,
    gap_probability,
    label_spread,
)


def test_label_spread_matches_bruteforce():
    rng = np.random.default_rng(0)
    y = rng.random(50)
    brute = np.abs(y[:, None] - y[None, :]).mean()
    assert np.isclose(label_spread(y), brute)
    assert label_spread(np.zeros(10)) == 0.0
    assert np.isclose(label_spread(np.array([0.0, 1.0])), 0.5)


def test_gap_probability():
    s = np.array([[0.0, 1.0], [5.0, 5.0]])
    l = np.array([[0.5, 0.5], [0.0, 10.0]])
    assert np.allclose(gap_probability(s, l), [0.5, 0.5])
    assert np.allclose(gap_probability(s[:1], np.array([[0.5, -1.0]]), estimator="paired"), [0.5])
    assert np.allclose(gap_probability(s[:1], np.array([[0.5, -1.0]])), [0.75])
    assert np.allclose(gap_probability(s, l, t=1.0), [1.0, 0.5])
    assert np.allclose(deterministic_labels(s, l), [0.0, 1.0])


def test_relaxation_fixes_imbalance():
    data = make_synthetic(n=500, offset=1.5, seed=1)
    y0 = gap_probability(data.small_scores, data.large_scores, 0.0)
    t, _, _ = find_relaxation(data.small_scores, data.large_scores)
    yt = gap_probability(data.small_scores, data.large_scores, t)
    assert t > 0
    assert y0.mean() < 0.1  # heavy imbalance without relaxation
    assert label_spread(yt) > label_spread(y0)


def test_metrics_endpoints():
    q_s, q_l = np.array([-2.0, -3.0]), np.array([-1.0, -1.0])
    r = evaluate_decisions(np.array([False, False]), q_s, q_l)
    assert r.cost_advantage == 0 and r.quality_drop_pct == 0
    r = evaluate_decisions(np.array([True, True]), q_s, q_l)
    assert r.cost_advantage == 1 and np.isclose(r.quality_drop_pct, 150.0)


def test_routers_beat_random():
    train, test = make_synthetic(n=3000, offset=0.3, seed=0).split((0.75, 0.25))
    q_s, q_l = test.realized_scores()
    rand_auc = area_under_curve(sweep(RandomRouter(0).score(test.queries), q_s, q_l))
    for mode in ("det", "prob", "trans"):
        r = HybridLLMRouter(mode).fit(train)
        assert area_under_curve(sweep(r.score(test.queries), q_s, q_l)) > rand_auc, mode
