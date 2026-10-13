"""Name -> class registries, so experiments are chosen in YAML instead of code.

    @register_router("hybrid_llm")
    class HybridLLMRouter(BinaryRouter): ...

    build_router({"type": "hybrid_llm", "models": {...}, "params": {...}, "encoder": {...}})
"""

from __future__ import annotations

import importlib
import inspect

ROUTERS: dict[str, type] = {}

# modules that register routers; add one line per new paper
ROUTER_MODULES = [
    "llm_routing.routers.baselines",
    "llm_routing.routers.hybrid_llm",
]


def register_router(name: str):
    def deco(cls):
        if name in ROUTERS and ROUTERS[name] is not cls:
            raise KeyError(f"router {name!r} registered twice")
        ROUTERS[name] = cls
        cls.registry_name = name
        return cls

    return deco


def _load_all():
    for m in ROUTER_MODULES:
        importlib.import_module(m)


def build_router(cfg: dict, seed: int | None = None):
    """cfg: {type, models: {small, large} | [names] | null, params: {...}, encoder: {...}}"""
    _load_all()
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind not in ROUTERS:
        raise KeyError(f"unknown router {kind!r}; known: {sorted(ROUTERS)}")
    kwargs = dict(cfg.get("params") or {})
    models = cfg.get("models")
    if isinstance(models, dict):
        kwargs.update(models)  # small=..., large=...
    elif models is not None:
        kwargs["models"] = list(models)
    if cfg.get("encoder") is not None:
        kwargs["encoder"] = dict(cfg["encoder"])
    cls = ROUTERS[kind]
    if seed is not None and "seed" not in kwargs and "seed" in inspect.signature(cls.__init__).parameters:
        kwargs["seed"] = seed
    router = cls(**kwargs)
    if cfg.get("name"):
        router.name = cfg["name"]
    return router


def list_routers() -> list[str]:
    _load_all()
    return sorted(ROUTERS)
