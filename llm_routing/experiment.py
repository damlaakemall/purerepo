"""Run several routers on one dataset and report the cost/quality trade-off.

Reusable for every paper: give it any list of ``Router`` objects.
"""

from __future__ import annotations

import numpy as np

from .core.base import Router
from .core.metrics import area_under_curve, cost_advantage_at_quality_drop, quality_at_cost_advantage, sweep
from .core.types import RoutingDataset


def compare_routers(routers: list[Router], train: RoutingDataset, test: RoutingDataset,
                    val: RoutingDataset | None = None, cost_targets=(0.1, 0.2, 0.4),
                    max_drop_pct: float = 1.0, fit: bool = True):
    q_s, q_l = test.realized_scores()
    rows, curves = [], {}
    for r in routers:
        if fit:
            r.fit(train, val)
        res = sweep(r.score(test.queries), q_s, q_l)
        curves[r.name] = res
        row = {"router": r.name, "auc": area_under_curve(res),
               f"cost_adv@drop<={max_drop_pct}%": cost_advantage_at_quality_drop(res, max_drop_pct)}
        for t, v in quality_at_cost_advantage(res, cost_targets).items():
            row[f"drop%@{int(t * 100)}%"] = v
        if hasattr(r, "t") and r.t is not None:
            row["t*"] = r.t
        rows.append(row)
    return rows, curves


def print_table(rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    widths = {k: max(len(k), 10) for k in keys}
    widths["router"] = max(len(r["router"]) for r in rows) + 2
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(widths[k]))
        print("  ".join(cells))


def plot_curves(curves: dict, path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, res in curves.items():
        ca = np.array([r.cost_advantage for r in res]) * 100
        qd = np.array([r.quality_drop_pct for r in res])
        style = "--" if name in ("random", "oracle") else "-"
        ax.plot(ca, qd, style, label=name)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("cost advantage (% queries to small model)")
    ax.set_ylabel("quality drop vs. all-large (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
