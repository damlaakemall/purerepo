"""Router-agnostic evaluation: every router -> the same quality/cost curve and summary numbers.

For each operating point we measure, on the TEST split:

    quality = mean realized quality of the returned answers  (sample 0 of the chosen model)
    cost    = mean cost paid per query

Summary numbers (all computed from the curve, so they apply to any router family):

aiq                 Average Improvement in Quality (RouterBench, Hu et al. 2024): area under the
                    router's non-decreasing quality-vs-cost envelope, cost normalized to
                    [cheapest model, most expensive model]. Higher is better.
quality_at_max_cost quality of the most expensive operating point.
cost_to_match       lowest cost at which the router reaches the quality of a reference model
                    (default: the best single model), as a fraction of that model's cost.
                    < 1 means the router saves money at equal quality.
max_quality         best quality reached at any operating point (can exceed every single model).

Binary routers additionally get the Hybrid LLM metrics (cost advantage, quality drop), see
``runner.binary_metrics``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Decision

if TYPE_CHECKING:  # type hints only; avoids a core <-> data import cycle
    from ..data.benchmark import Benchmark


def curve(decisions: list[Decision], bench: Benchmark) -> np.ndarray:
    """(P, 2) array of (cost, quality) per operating point."""
    rows = np.arange(len(bench))
    points = []
    for d in decisions:
        choice = np.asarray(d.choice)
        sample = np.zeros(len(bench), dtype=int) if d.sample_index is None else np.asarray(d.sample_index)
        cost = np.asarray(d.cost)
        if (choice.shape != (len(bench),) or sample.shape != choice.shape or cost.shape != choice.shape
                or choice.dtype.kind not in "iu" or sample.dtype.kind not in "iu"
                or (choice < 0).any() or (choice >= bench.num_models).any()
                or (sample < 0).any() or (sample >= bench.k).any()):
            raise ValueError("invalid returned model/sample indices or decision dimensions")
        quality = bench.scores[rows, choice, sample]
        if not np.isfinite(quality).all() or not np.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("evaluation needs a fixed complete cohort; missing quality/cost may not disappear from the denominator")
        points.append([float(cost.mean()), float(quality.mean())])
    return np.array(points)


def single_model_points(bench: Benchmark) -> dict[str, tuple[float, float]]:
    q, c = bench.realized(), bench.costs
    return {m: (float(c[:, i].mean()), float(np.nanmean(q[:, i]))) for i, m in enumerate(bench.models)}


def oracle_point(bench: Benchmark) -> tuple[float, float]:
    """Best quality per query, cheapest model among ties."""
    q, c = np.nan_to_num(bench.realized(), nan=-np.inf), bench.costs
    best = q.max(axis=1, keepdims=True)
    cand = np.where(q >= best, c, np.inf)
    pick = cand.argmin(axis=1)
    rows = np.arange(len(bench))
    return float(c[rows, pick].mean()), float(q[rows, pick].mean())


def _envelope(pts: np.ndarray) -> np.ndarray:
    pts = pts[np.argsort(pts[:, 0])]
    env = pts.copy()
    env[:, 1] = np.maximum.accumulate(pts[:, 1])  # paying more never has to hurt: keep best so far
    return env


def aiq(pts: np.ndarray, c_min: float, c_max: float, grid: int = 512) -> float:
    if c_max <= c_min:
        return float(pts[:, 1].max())
    env = _envelope(pts)
    xs = np.linspace(c_min, c_max, grid)
    # below the cheapest operating point the router has no answer: use its cheapest quality
    ys = np.interp(xs, env[:, 0], env[:, 1], left=env[0, 1], right=env[-1, 1])
    return float(np.trapezoid(ys, xs) / (c_max - c_min))


def cost_to_match(pts: np.ndarray, target_quality: float, target_cost: float) -> float:
    ok = pts[pts[:, 1] >= target_quality - 1e-12]
    if len(ok) == 0:
        return float("inf")
    return float(ok[:, 0].min() / max(target_cost, 1e-12))


def summarize(decisions: list[Decision], bench: Benchmark, reference_model: str | None = None) -> dict:
    pts = curve(decisions, bench)
    singles = single_model_points(bench)
    costs = np.array([c for c, _ in singles.values()])
    ref = reference_model or max(singles, key=lambda m: singles[m][1])
    ref_c, ref_q = singles[ref]
    return {
        "aiq": aiq(pts, costs.min(), costs.max()),
        "max_quality": float(pts[:, 1].max()),
        "quality_at_max_cost": float(pts[np.argmax(pts[:, 0]), 1]),
        "reference_model": ref,
        "cost_to_match_reference": cost_to_match(pts, ref_q, ref_c),
    }
