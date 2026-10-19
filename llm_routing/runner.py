"""Train + evaluate one experiment config. Same procedure for every router -> fair comparison.

1. load the frozen benchmark (same files for everyone, fingerprint recorded)
2. restrict it to the router's candidate models (binary: small+large; multi: its list or all);
   a router's ``training_models()`` further restricts the TRAIN split it may read (UniRoute)
3. fit on the TRAIN split (VAL available for tuning); never touch TEST while fitting
4. sweep the router's knob on TEST -> quality/cost curve -> summary metrics; a binary router also
   gets the threshold chosen on VAL, applied once to TEST (Hybrid LLM App. A)
5. evaluate the configured baselines on exactly the same candidate models and split
   (on a dynamic pool they fit on VAL only: they may not see the new models' TRAIN labels)
6. write runs/<benchmark>/<experiment>/<timestamp>_s<seed>/ and log to W&B
"""

from __future__ import annotations

import csv
import json
import os
import random
import time

import numpy as np
from omegaconf import OmegaConf

from .core.base import BinaryRouter, Router
from .core.evaluation import curve, oracle_point, single_model_points, summarize
from .data.loaders import load_benchmark
from .registry import build_router
from .tracking import Tracker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def candidate_models(router: Router, bench) -> list[str]:
    if isinstance(router, BinaryRouter):
        return [router.small, router.large]
    return list(getattr(router, "models", None) or bench.models)


def training_splits(router: Router, bench):
    """One information boundary for both real training and ``check-router``.

    Restrict TRAIN before inspecting missing labels. Keep TEST unfiltered until
    ``fit`` has declared the evaluation pool; its other model columns may be absent.
    """
    from .collection.evaluate import complete_cohort

    sub = bench.select_models(candidate_models(router, bench))
    train, val, test = (sub.split(s) for s in ("train", "val", "test"))
    models = router.training_models()
    if models is not None:
        if not models or not set(models) <= set(sub.models):
            raise ValueError("training_models must be a nonempty subset of the candidate models")
        train = train.select_models(list(models))
    train = complete_cohort(train)
    val = complete_cohort(val) if len(val) else val
    if models is not None and not len(val):
        raise ValueError("a dynamic pool needs validation data to onboard new models")
    return train, val, test


def evaluation_split(router: Router, test):
    """Choose the evaluation models after fit, then freeze their complete cohort."""
    from .collection.evaluate import complete_cohort

    models = router.evaluation_models()
    if models is not None:
        if not models or not set(models) <= set(test.models):
            raise ValueError("evaluation_models must be a nonempty subset of the candidate models")
        test = test.select_models(list(models))
    return complete_cohort(test)


def cohort_record(available: int, bench) -> dict:
    from .collection.io import digest

    return {"available": available, "scored": len(bench), "query_ids": bench.query_ids,
            "cohort_hash": digest(bench.query_ids)}


def binary_metrics(decisions, test, small_idx: int, large_idx: int) -> dict:
    """Hybrid LLM's metrics: share of traffic to the small model vs. quality drop relative to all-large."""
    q = test.realized()
    rows = np.arange(len(test))
    base = float(np.nanmean(q[:, large_idx]))
    pts = []
    for d in decisions:
        share = float(np.mean(d.choice == small_idx))
        drop = (base - float(np.nanmean(q[rows, d.choice]))) / abs(base) * 100 if base != 0 else float("nan")
        pts.append((share, drop))
    pts = np.array(sorted(pts))
    out = {f"quality_drop_pct@{int(t * 100)}%": float(np.interp(t, pts[:, 0], pts[:, 1])) for t in (0.1, 0.2, 0.4)}
    ok = pts[pts[:, 1] <= 1.0]
    out["cost_advantage@drop<=1%"] = float(ok[:, 0].max()) if len(ok) else 0.0
    return out


def _share_and_drop(to_small: np.ndarray, bench, small: str, large: str) -> tuple[float, float]:
    q = bench.realized()
    i, j = bench.model_index(small), bench.model_index(large)
    base = float(q[:, j].mean())
    routed = float(np.where(to_small, q[:, i], q[:, j]).mean())
    drop = (base - routed) / abs(base) * 100 if base != 0 else float("nan")  # undefined if all-large scores 0
    return float(to_small.mean()), drop


def val_threshold_metrics(router: BinaryRouter, val, test, max_drop_pct: float = 1.0) -> dict:
    """Hybrid LLM App. A: pick the threshold on VAL with the most traffic to the small model while the
    quality drop stays <= ``max_drop_pct``, freeze it, then report what it does on TEST.

    The test sweep above is descriptive (its thresholds are test-score quantiles); this is the
    deployable operating point. Test labels never choose the threshold.
    """
    s = np.asarray(router.score(val.queries), dtype=float)
    best_t, best_share = np.inf, 0.0
    for t in np.unique(s):
        share, drop = _share_and_drop(s >= t, val, router.small, router.large)
        if np.isfinite(drop) and drop <= max_drop_pct and share > best_share:
            best_t, best_share = float(t), share
    val_share, val_drop = _share_and_drop(s >= best_t, val, router.small, router.large)
    to_small = router.route(test.queries, best_t)
    test_share, test_drop = _share_and_drop(to_small, test, router.small, router.large)
    choice = np.where(to_small, test.model_index(router.small), test.model_index(router.large))
    return {"val_threshold": best_t, "val_small_share": val_share, "val_quality_drop_pct": val_drop,
            "test@val_threshold_small_share": test_share, "test@val_threshold_quality_drop_pct": test_drop,
            "test@val_threshold_cost": float(test.costs[np.arange(len(test)), choice].mean())}


def evaluate_router(router: Router, test, num_points: int = 51, val=None) -> tuple[dict, list[dict]]:
    decisions = router.sweep(test, num_points)
    m = summarize(decisions, test)
    if isinstance(router, BinaryRouter):
        m.update(binary_metrics(decisions, test, test.model_index(router.small), test.model_index(router.large)))
        if val is not None and len(val):
            m.update(val_threshold_metrics(router, val, test))
    pts = curve(decisions, test)
    rows = [{"router": router.name, "knob": d.knob, "cost": float(c), "quality": float(q)}
            for d, (c, q) in zip(decisions, pts)]
    return m, rows


def run_experiment(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    bench = load_benchmark(cfg["benchmark"])
    fp = bench.fingerprint()
    router = build_router(cfg["router"], seed=cfg["seed"])
    models = candidate_models(router, bench)
    train_models = router.training_models()
    train, val, test = training_splits(router, bench)
    coverage = {s: cohort_record(len(bench.split(s)), b)
                for s, b in (("train", train), ("val", val))}
    print(f"[run] {cfg['name']}  benchmark={bench.name} ({fp})  models={models}  "
          f"train/val/test={len(train)}/{len(val)}/{len(test)}"
          + (f"  training models={list(train_models)}" if train_models else ""))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg["output_dir"], bench.name, cfg["name"], f"{stamp}_s{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), os.path.join(run_dir, "config.yaml"))
    tracker = Tracker(cfg, f"{cfg['name']}_s{cfg['seed']}", run_dir)

    t0 = time.time()
    router.fit(train, val)
    fit_seconds = time.time() - t0
    eval_models = router.evaluation_models()
    before = len(test)
    test = evaluation_split(router, test)
    models = list(test.models)
    if eval_models is not None:
        print(f"[run] evaluating on the router's test pool: {models}")
    coverage["test"] = cohort_record(before, test)
    print(f"[run] complete-case test cohort: {len(test)} of {before} queries")
    eval_val = val.select_models(models)

    results, curve_rows = [], []
    m, rows = evaluate_router(router, test, val=eval_val)
    results.append({"router": router.name, "type": cfg["router"]["type"], "family": router.family,
                    "is_baseline": False, "fit_seconds": fit_seconds, **m, "learned": router.describe()})
    curve_rows += rows
    for b in cfg.get("baselines", []):
        bcfg = {"type": b, "models": eval_models or cfg["router"].get("models")}
        if b == "oracle":
            bcfg["models"] = models
        elif b == "random_binary" and not isinstance(router, BinaryRouter):
            print("[run] skipping random_binary baseline: router is not binary")
            continue
        br = build_router(bcfg, seed=cfg["seed"])
        if train_models is not None:  # same information as the router: test-pool labels on VAL only
            br.fit(eval_val, None)
        else:
            br.fit(train, val)
        bm, brows = evaluate_router(br, test, val=eval_val)
        results.append({"router": br.name, "type": b, "family": br.family, "is_baseline": True,
                        "fit_on": "val" if train_models is not None else "train", **bm})
        curve_rows += brows

    singles = single_model_points(test)
    record = {"experiment": cfg["name"], "benchmark": bench.name, "fingerprint": fp, "seed": cfg["seed"],
              "quality_metric": bench.meta.get("metric"), "cost_unit": bench.meta.get("cost_unit"),
              "coverage": coverage, "models": models, "training_models": train_models, "num_test": len(test), "num_train": len(train),
              "single_models": {k: {"cost": c, "quality": q} for k, (c, q) in singles.items()},
              "oracle_point": dict(zip(("cost", "quality"), oracle_point(test))), "results": results,
              "run_dir": run_dir}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(run_dir, "curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["router", "knob", "cost", "quality"])
        w.writeheader(); w.writerows(curve_rows)
    plot_path = os.path.join(run_dir, "curve.png")
    plot_run(record, curve_rows, plot_path)

    tracker.log_summary({f"{r['router']}/{k}": v for r in results for k, v in r.items()
                         if isinstance(v, (int, float))} | {"fingerprint": fp})
    tracker.log_curve("test", curve_rows)
    tracker.log_image("curve", plot_path)
    tracker.finish()
    print_results(record)
    print(f"[run] saved {run_dir}")
    return record


def print_results(record: dict):
    keys = ["aiq", "max_quality", "cost_to_match_reference", "cost_advantage@drop<=1%",
            "quality_drop_pct@20%", "quality_drop_pct@40%"]
    print("\n" + "router".ljust(26) + "".join(k[:22].ljust(24) for k in keys))
    for r in record["results"]:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(24))
        print(r["router"].ljust(26) + "".join(cells))
    print("single models (cost, quality): " +
          ", ".join(f"{m}=({v['cost']:.3g}, {v['quality']:.3f})" for m, v in record["single_models"].items()))


def plot_run(record: dict, curve_rows: list[dict], path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name in dict.fromkeys(r["router"] for r in curve_rows):
        pts = sorted((r["cost"], r["quality"]) for r in curve_rows if r["router"] == name)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "--" if name in ("random", "oracle") else "-", label=name, lw=1.4,
                marker="o" if len(pts) == 1 else None)
    for m, v in record["single_models"].items():
        ax.scatter([v["cost"]], [v["quality"]], marker="o", s=28, zorder=3)
        ax.annotate(m, (v["cost"], v["quality"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean cost per query")
    ax.set_ylabel("mean quality (test)")
    ax.set_title(f"{record['experiment']} — {record['benchmark']}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
