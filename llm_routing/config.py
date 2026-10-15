"""YAML experiment configs.

An experiment file points to one benchmark config and one router config and may
override any field of either:

    # configs/experiments/quality_regression__routerbench.yaml
    name: quality_regression
    seed: 0
    benchmark: configs/benchmarks/routerbench.yaml
    router: configs/routers/quality_regression.yaml
    overrides:
      router: {params: {alpha: 1.0}}
    baselines: [oracle]
    wandb: {enabled: false, project: llm-routing}

Command-line overrides use dot notation:  router.params.alpha=0.5  seed=1
``${oc.env:LLMR_DATA_ROOT,data}`` in any path reads an environment variable
(used on the cluster to put data on scratch storage).
"""

from __future__ import annotations

import os

from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) or os.path.exists(p) else os.path.join(REPO_ROOT, p)


def load_yaml(path: str):
    return OmegaConf.load(_resolve_path(path))


def load_experiment(path: str, cli_overrides: list[str] | None = None) -> dict:
    exp = load_yaml(path)
    cfg = OmegaConf.create({k: v for k, v in exp.items() if k not in ("benchmark", "router", "overrides")})
    cfg.benchmark = load_yaml(exp.benchmark) if isinstance(exp.benchmark, str) else exp.benchmark
    cfg.router = load_yaml(exp.router) if isinstance(exp.router, str) else exp.router
    if "overrides" in exp:
        cfg = OmegaConf.merge(cfg, exp.overrides)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    cfg.setdefault("seed", 0)
    cfg.setdefault("baselines", [])
    cfg.setdefault("wandb", {"enabled": False})
    cfg.setdefault("output_dir", "runs")
    return _with_model_pool(OmegaConf.to_container(cfg, resolve=True))


def load_benchmark_config(path: str, cli_overrides: list[str] | None = None) -> dict:
    cfg = load_yaml(path)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    return _with_model_pool(OmegaConf.to_container(cfg, resolve=True))


def _with_model_pool(cfg):
    """Allow a shared YAML model pool without changing existing inline configs."""
    benchmark = cfg.get("benchmark", cfg)
    if isinstance(benchmark, dict) and benchmark.get("model_pool"):
        if "models" in benchmark:
            raise ValueError("use model_pool or inline models, not both")
        pool = OmegaConf.to_container(load_yaml(benchmark["model_pool"]), resolve=True)
        benchmark["models"] = pool["models"]
    return cfg
