"""Benchmark config -> frozen Benchmark on disk -> loaded Benchmark.

``load_benchmark(cfg)`` returns the frozen copy at ``cfg.path`` if it exists;
otherwise it pulls it from the Hub (if ``hub_repo`` is set) or prepares it with
the configured loader and saves it. Training runs therefore always read the
same files.
"""

from __future__ import annotations

import os

from .benchmark import Benchmark


def prepare(cfg: dict) -> Benchmark:
    kind = cfg["loader"]
    p = dict(cfg.get("params") or {})
    if kind == "synthetic":
        from .synthetic import make_synthetic_benchmark

        models = p.pop("models", None)
        if models:
            models = {k: tuple(v) for k, v in models.items()}
        return make_synthetic_benchmark(models=models, name=cfg["name"], **p)
    if kind == "mixinstruct":
        from .mixinstruct import load_mixinstruct

        return load_mixinstruct(name=cfg["name"], **p)
    if kind == "routerbench":
        from .routerbench import load_routerbench

        return load_routerbench(name=cfg["name"], **p)
    if kind == "generated":
        from ..pipeline.build import assemble

        return assemble(cfg)
    raise KeyError(f"unknown loader {kind!r}")


def load_benchmark(cfg: dict, with_responses: bool = False, rebuild: bool = False) -> Benchmark:
    path = cfg["path"]
    if cfg.get("protocol") == "shared-v1":
        from ..pipeline.shared.assemble import load_view
        # Training never assembles, downloads, or regenerates a shared release.
        return load_view(cfg, with_responses)
    if not rebuild and os.path.exists(os.path.join(path, "manifest.json")):
        return Benchmark.load(path, with_responses)
    if not rebuild and cfg.get("hub_repo"):
        from .hub import pull

        try:
            pull(cfg["hub_repo"], path, cfg.get("hub_revision"))
            return Benchmark.load(path, with_responses)
        except Exception as e:  # repo not created yet, offline, ...
            print(f"[data] could not pull {cfg['hub_repo']}: {e}; preparing locally")
    b = prepare(cfg)
    if cfg["loader"] != "generated":  # `assemble` already saved it
        b.save(path)
    print(f"[data] prepared {cfg['name']} -> {path} (fingerprint {b.fingerprint()})")
    return Benchmark.load(path, with_responses)
