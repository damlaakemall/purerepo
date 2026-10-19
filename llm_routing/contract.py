"""`llmr check-router <router.yaml>`: the checks every router must pass before a Tosun run.

They turn the fairness rules (AGENTS.md, CONTRIBUTING.md) into something a person or a coding
agent can run in seconds on CPU, on data shaped like the shared release
(configs/benchmarks/synthetic_shared.yaml: the team's 12 model names, 7 sources, 0/1 and
token-F1 scores, k=10 samples, a few missing cells):

registered        the config builds a registered router
decisions         fit(train, val) then sweep(test): valid, finite, cheapest operating point first
no test labels    shuffling scores, costs, extras and gold references changes no decision
dynamic pool      changing hidden models' TRAIN scores, costs, extras and missingness changes no decision
describe          describe() is JSON-serializable, so it can be saved in metrics.json

Passing them is necessary, not sufficient: the paper-specific parts (labels, loss, threshold
selection) still need their own unit tests.
"""

from __future__ import annotations

import json

import numpy as np

DEV_BENCHMARK = "configs/benchmarks/synthetic_shared.yaml"


def _decisions(router, test, num):
    out = []
    for d in router.sweep(test, num):
        sample = np.zeros(len(test), dtype=int) if d.sample_index is None else np.asarray(d.sample_index)
        out.append((np.asarray(d.choice).tolist(), sample.tolist()))
    return out


def _shuffle_outcomes(bench, seed):
    """Keep observable queries/sources fixed, perturb every stored outcome channel."""
    shuffled = bench.subset(np.arange(len(bench)))
    perm = np.random.default_rng(seed).permutation(len(bench))
    shuffled.scores, shuffled.costs = bench.scores[perm], bench.costs[perm]
    shuffled.extras = {name: values[perm] for name, values in bench.extras.items()}
    if bench.references is not None:
        shuffled.references = [bench.references[i] for i in perm]
    return shuffled


def check_router(router_cfg: dict, bench, seed: int = 0, num: int = 11) -> list[tuple[str, bool, str]]:
    from .core.evaluation import curve
    from .registry import build_router
    from .runner import evaluation_split, set_seed, training_splits

    results = []

    def build():
        set_seed(seed)
        return build_router(dict(router_cfg), seed=seed)

    try:
        router = build()
    except Exception as e:  # noqa: BLE001 - report, do not crash the checklist
        return [("registered", False, f"{type(e).__name__}: {e}")]
    results.append(("registered", True, f"{type(router).__name__} ({router.family})"))

    try:
        train, val, test = training_splits(router, bench)
        router.fit(train, val)
        evaluated = evaluation_split(router, test)
        pts = curve(router.sweep(evaluated, num), evaluated)
        if not len(pts):
            return results + [("decisions", False, "sweep() returned no operating points")]
        ok = bool(pts[0, 0] <= pts[-1, 0] + 1e-9)
        results.append(("decisions", ok, f"{len(pts)} operating point(s); cost {pts[0, 0]:.3g} -> {pts[-1, 0]:.3g}"))
    except Exception as e:  # report invalid fits/decisions as a failed check, not a traceback
        return results + [("decisions", False, f"{type(e).__name__}: {e}")]

    try:
        shuffled = _shuffle_outcomes(evaluated, seed + 1)
        same = _decisions(router, evaluated, num) == _decisions(router, shuffled, num)
        results.append(("no test labels", same, "decisions unchanged when all test outcome channels are shuffled" if same else
                        "decisions changed: sweep() reads a test outcome channel"))
    except Exception as e:
        results.append(("no test labels", False, f"{type(e).__name__}: {e}"))

    if router.training_models() is not None:
        other = [j for j, m in enumerate(bench.models) if m not in router.training_models()]
        flipped = bench.subset(np.arange(len(bench)))
        rows = flipped.split_of == "train"
        hidden = np.ix_(rows, other)
        # Alter missingness as well: hidden train labels must not choose the fit cohort.
        flipped.scores[hidden] = np.nan
        flipped.costs[hidden] = np.nan
        for values in flipped.extras.values():
            values[hidden] = np.nan
        again = build()
        tr, va, _ = training_splits(again, flipped)
        again.fit(tr, va)
        same = (router.evaluation_models() == again.evaluation_models() and
                _decisions(router, evaluated, num) == _decisions(again, evaluated, num))
        results.append(("dynamic pool", same, "new models' TRAIN labels unused" if same else
                        "decisions changed: fit() reads TRAIN labels of models outside training_models()"))

    try:
        json.dumps(router.describe())  # exactly the serializer used for metrics.json
        results.append(("describe", True, "JSON-serializable"))
    except (TypeError, ValueError) as e:
        results.append(("describe", False, str(e)))
    return results


def run_check(router_path: str, benchmark_path: str = DEV_BENCHMARK, overrides: list[str] | None = None) -> bool:
    from omegaconf import OmegaConf

    from .config import load_benchmark_config, load_yaml
    from .data.loaders import load_benchmark, prepare

    cfg = OmegaConf.to_container(load_yaml(router_path), resolve=True)
    if overrides:
        cfg = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.from_dotlist(overrides)))
    bcfg = load_benchmark_config(benchmark_path)
    # dev data is generated in memory (seconds); a real shared release is read from disk, never rebuilt
    bench = load_benchmark(bcfg) if bcfg.get("protocol") == "shared-v1" else prepare(bcfg)
    print(f"[check-router] {router_path} on {bench.name} ({len(bench)} queries, {bench.num_models} models)")
    results = check_router(cfg, bench, seed=int(cfg.get("seed", 0) or 0))
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<15} {detail}")
    passed = all(ok for _, ok, _ in results)
    print("[check-router] " + ("all checks passed" if passed else "FAILED"))
    return passed
