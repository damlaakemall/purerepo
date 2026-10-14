"""CPU-only benchmark audit: identical cohorts for all strategies within each quality channel."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ...core.base import Decision
from ...core.evaluation import curve, oracle_point, single_model_points
from ...routers.baselines import OracleRouter
from .assemble import frozen_dir, load_view
from .io import digest, write_json


def complete_cohort(bench):
    mask = np.isfinite(bench.realized()).all(axis=1) & np.isfinite(bench.costs).all(axis=1)
    if not mask.any():
        raise ValueError("no queries with sample-0 quality and cost for the entire model pool")
    return bench.subset(np.flatnonzero(mask))


def evaluate_data(cfg):
    output = Path(cfg["path"]) / "reports" / cfg["version"]
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    curve_rows = []
    for metric in ("ground_truth", "armorm"):
        all_rows = load_view({**cfg, "quality_metric": metric}).split("test")
        if not len(all_rows):
            reports[metric] = {"status": "unavailable", "reason": "empty test split"}
            continue
        try:
            bench = complete_cohort(all_rows)
        except ValueError as e:
            reports[metric] = {"status": "unavailable", "reason": str(e), "n_test": len(all_rows)}
            continue
        singles = single_model_points(bench)
        best = max(singles, key=lambda m: singles[m][1])
        cheapest = min(singles, key=lambda m: singles[m][0])
        rng = np.random.default_rng(cfg["seed"])
        pick = rng.integers(0, bench.num_models, len(bench))
        random = Decision(0, pick, bench.costs[np.arange(len(bench)), pick])
        oracle_curve = curve(OracleRouter().sweep(bench), bench)
        reports[metric] = {"status": "ok", "n_test": len(all_rows), "n_scored": len(bench),
                           "query_ids": bench.query_ids, "cohort_hash": digest(bench.query_ids),
                           "fingerprint": bench.fingerprint(), "models": bench.models, "cost_unit": bench.meta["cost_unit"],
                           "single_models": {m: {"cost": c, "quality": q} for m, (c, q) in singles.items()},
                           "best_single_hindsight": best, "cheapest_single": cheapest,
                           "random": dict(zip(("cost", "quality"), curve([random], bench)[0].tolist())),
                           "oracle": dict(zip(("cost", "quality"), oracle_point(bench))),
                           "per_source": {s: {m: {"cost": c, "quality": q} for m, (c, q) in
                                               single_model_points(bench.subset(np.flatnonzero(np.array(bench.sources) == s))).items()}
                                          for s in sorted(set(bench.sources))}}
        for label, points in [(m, np.array([p])) for m, p in singles.items()] + [("random", curve([random], bench)), ("oracle", oracle_curve)]:
            curve_rows.extend({"metric": metric, "strategy": label, "cost": float(c), "quality": float(q)} for c, q in points)
    report = {"version": cfg["version"], "release": str(frozen_dir(cfg)), "channels": reports,
              "note": "GT and ArmoRM are separate outcomes; oracle and hindsight best single are evaluation bounds, never train labels."}
    write_json(output / "evaluation.json", report)
    with (output / "curves.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "strategy", "cost", "quality"])
        w.writeheader()
        w.writerows(curve_rows)
    if curve_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ("ground_truth", "armorm")):
            for label in sorted({r["strategy"] for r in curve_rows if r["metric"] == metric}):
                pts = sorted((r["cost"], r["quality"]) for r in curve_rows if r["metric"] == metric and r["strategy"] == label)
                ax.plot(*zip(*pts), marker=".", label=label)
            ax.set(title=metric, xlabel="mean cost / query", ylabel="mean quality")
            if ax.lines:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "quality_cost.png", dpi=140)
        plt.close(fig)
    return report
