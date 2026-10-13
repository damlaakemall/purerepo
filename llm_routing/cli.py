"""`llmr` — one command for everything (installed by `pip install -e .`; or `python -m llm_routing.cli`).

    llmr train      configs/experiments/X.yaml [key=value ...]   train + evaluate one router (+ baselines)
    llmr prepare    configs/benchmarks/X.yaml                    build/download a frozen benchmark
    llmr info       configs/benchmarks/X.yaml                    print benchmark summary + fingerprint
    llmr build-queries | generate --model M | score --model M | assemble   configs/benchmarks/ours_*.yaml
    llmr push-data | pull-data  configs/benchmarks/X.yaml        share frozen data via Hugging Face Hub
    llmr leaderboard [--runs runs] [--out results/leaderboard.md]
    llmr routers                                                 list registered routers
"""

from __future__ import annotations

import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(prog="llmr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train"); t.add_argument("config"); t.add_argument("overrides", nargs="*")
    for name in ("prepare", "info", "build-queries", "assemble", "push-data", "pull-data"):
        s = sub.add_parser(name); s.add_argument("config"); s.add_argument("overrides", nargs="*")
        if name == "prepare":
            s.add_argument("--rebuild", action="store_true")
    for name in ("generate", "score"):
        s = sub.add_parser(name); s.add_argument("config"); s.add_argument("--model", required=True)
        s.add_argument("overrides", nargs="*")
    lb = sub.add_parser("leaderboard"); lb.add_argument("--runs", default="runs")
    lb.add_argument("--out", default="results/leaderboard.md")
    sub.add_parser("routers")
    a = p.parse_args(argv)

    from .config import load_benchmark_config, load_experiment

    if a.cmd == "train":
        from .runner import run_experiment

        run_experiment(load_experiment(a.config, a.overrides))
    elif a.cmd == "routers":
        from .registry import list_routers

        print("\n".join(list_routers()))
    elif a.cmd == "leaderboard":
        from .leaderboard import write_leaderboard

        print(write_leaderboard(a.runs, a.out))
    else:
        cfg = load_benchmark_config(a.config, a.overrides)
        if a.cmd in ("prepare", "info"):
            from .data.loaders import load_benchmark

            b = load_benchmark(cfg, rebuild=getattr(a, "rebuild", False))
            print(json.dumps(b.summary(), indent=2))
        elif a.cmd == "build-queries":
            from .pipeline.build import build_queries

            build_queries(cfg)
        elif a.cmd == "generate":
            from .pipeline.build import generate

            generate(cfg, a.model)
        elif a.cmd == "score":
            from .pipeline.build import score

            score(cfg, a.model)
        elif a.cmd == "assemble":
            from .pipeline.build import assemble

            assemble(cfg)
        elif a.cmd == "push-data":
            from .data.hub import push

            push(cfg["path"], cfg["hub_repo"])
        elif a.cmd == "pull-data":
            from .data.hub import pull

            pull(cfg["hub_repo"], cfg["path"], cfg.get("hub_revision"))


if __name__ == "__main__":
    main()
