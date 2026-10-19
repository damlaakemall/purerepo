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
        bench = make_synthetic_benchmark(models=models, name=cfg["name"], **p)
        if cfg.get("models") and cfg.get("cost", {}).get("definition") == "usd_tokens":
            import numpy as np
            from ..collection.assemble import model_info
            from ..collection.dataset import model_pool

            pool = model_pool(cfg)
            if list(pool) != bench.models:
                raise ValueError("synthetic models must match the model_pool order")
            # Invented token lengths, not downloads/tokenizer calls. Exercise the real cost formula.
            prompt = np.broadcast_to(np.array([len(q.split()) for q in bench.queries])[:, None, None], bench.scores.shape).astype(np.float32)
            completion = np.random.default_rng(p.get("seed", 0)).integers(8, 64, bench.scores.shape).astype(np.float32)
            rates = np.array([[m["pricing"][k] for k in ("input_per_1m", "output_per_1m")] for m in pool.values()])
            sample_cost = ((prompt * rates[None, :, 0, None] + completion * rates[None, :, 1, None]) / 1e6).astype(np.float32)
            bench.costs = sample_cost[:, :, 0]
            bench.extras.update(prompt_tokens=prompt, completion_tokens=completion, sample_cost=sample_cost)
            bench.meta.update(model_info=model_info(pool), cost_unit="USD", cost_definition=cfg["cost"],
                              token_counts="invented word counts and random output lengths; not HF tokenization")
        return bench
    if kind == "mixinstruct":
        from .mixinstruct import load_mixinstruct

        return load_mixinstruct(name=cfg["name"], **p)
    if kind == "routerbench":
        from .routerbench import load_routerbench

        return load_routerbench(name=cfg["name"], **p)
    if kind == "generated":
        from ..legacy_pipeline.build import assemble

        return assemble(cfg)
    raise KeyError(f"unknown loader {kind!r}")


def load_benchmark(cfg: dict, with_responses: bool = False, rebuild: bool = False) -> Benchmark:
    path = cfg["path"]
    if cfg.get("protocol") == "shared-v1":
        from ..collection.assemble import load_view
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
