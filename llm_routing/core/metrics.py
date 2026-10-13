"""Evaluation metrics for binary routing, as used in Hybrid LLM (Sec. 4).

Cost advantage
    Fraction of queries routed to the small model. Since the small model is
    (assumed) much cheaper, this approximates the fraction of cost saved.

Quality drop
    How much worse the routed system's average response quality is than
    "always use the large model", in percent:

        drop = (q_all_large - q_routed) / |q_all_large| * 100

    It can be negative: on queries where the small model happens to answer
    better, routing can *improve* quality.

A router is good if it achieves high cost advantage at low quality drop. The
whole trade-off is captured by sweeping the threshold and plotting the curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoutingResult:
    threshold: float
    cost_advantage: float  # in [0, 1]
    quality: float  # mean quality of the routed responses
    quality_drop_pct: float  # vs. all-to-large


def evaluate_decisions(to_small: np.ndarray, q_small: np.ndarray, q_large: np.ndarray, threshold=np.nan) -> RoutingResult:
    to_small = np.asarray(to_small, dtype=bool)
    routed = np.where(to_small, q_small, q_large)
    base = q_large.mean()
    return RoutingResult(
        threshold=float(threshold),
        cost_advantage=float(to_small.mean()),
        quality=float(routed.mean()),
        quality_drop_pct=float((base - routed.mean()) / abs(base) * 100.0),
    )


def sweep(scores: np.ndarray, q_small: np.ndarray, q_large: np.ndarray, num: int = 101) -> list[RoutingResult]:
    """Evaluate a router's scores at many thresholds (quantiles -> evenly spaced cost advantages)."""
    scores = np.asarray(scores, dtype=float)
    qs = np.linspace(0.0, 1.0, num)
    thresholds = np.concatenate([[np.inf], np.quantile(scores, 1.0 - qs[1:])])
    return [evaluate_decisions(scores >= t, q_small, q_large, t) for t in thresholds]


def quality_at_cost_advantage(results: list[RoutingResult], targets=(0.1, 0.2, 0.4)) -> dict[float, float]:
    """Interpolated quality drop (%) at given cost advantages — the paper's main table format."""
    ca = np.array([r.cost_advantage for r in results])
    qd = np.array([r.quality_drop_pct for r in results])
    order = np.argsort(ca)
    return {t: float(np.interp(t, ca[order], qd[order])) for t in targets}


def cost_advantage_at_quality_drop(results: list[RoutingResult], max_drop_pct: float) -> float:
    """Largest cost advantage whose quality drop stays <= ``max_drop_pct``."""
    ok = [r.cost_advantage for r in results if r.quality_drop_pct <= max_drop_pct]
    return max(ok) if ok else 0.0


def area_under_curve(results: list[RoutingResult]) -> float:
    """Area under the (cost advantage -> routed quality) curve. Router-agnostic single number.

    Higher is better. Compare it against the random router's AUC.
    """
    ca = np.array([r.cost_advantage for r in results])
    q = np.array([r.quality for r in results])
    order = np.argsort(ca)
    ca, q = ca[order], q[order]
    return float(np.sum((ca[1:] - ca[:-1]) * (q[1:] + q[:-1]) / 2.0))
