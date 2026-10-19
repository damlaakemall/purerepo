"""`llmr` — one command for everything (installed by `pip install -e .`; or `python -m llm_routing.cli`).

    llmr train      configs/experiments/X.yaml [key=value ...]   train + evaluate one router (+ baselines)
    llmr prepare    configs/benchmarks/X.yaml                    build/download a frozen benchmark
    llmr info       configs/benchmarks/X.yaml                    print benchmark summary + fingerprint
    llmr build-queries | generate --model M | score --model M | assemble   configs/benchmarks/ours_*.yaml
    llmr push-data | pull-data  configs/benchmarks/X.yaml        share frozen data via Hugging Face Hub
    llmr leaderboard [--runs runs] [--out results/leaderboard.md]
    llmr routers                                                 list registered routers
    llmr check-router configs/routers/X.yaml [--benchmark B]    fairness/contract checks on CPU dev data
"""

from __future__ import annotations

import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(prog="llmr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train"); t.add_argument("config"); t.add_argument("overrides", nargs="*")
    for name in ("prepare", "info", "build-queries", "assemble", "push-data", "pull-data", "validate-data", "evaluate-data"):
        s = sub.add_parser(name); s.add_argument("config"); s.add_argument("overrides", nargs="*")
        if name == "prepare":
            s.add_argument("--rebuild", action="store_true")
    for name in ("generate", "score", "package-contribution"):
        s = sub.add_parser(name); s.add_argument("config"); s.add_argument("--model", required=True)
        s.add_argument("overrides", nargs="*")
        s.add_argument("--member")
        s.add_argument("--shard-index", type=int)
        s.add_argument("--num-shards", type=int)
        if name == "score":
            s.add_argument("--scorer", choices=["ground_truth", "armorm", "bartscore"])
        if name == "generate":
            s.add_argument("--max-retries", type=int)
            s.add_argument("--retry-delay", type=float)
            s.add_argument("--batch-size", type=int, help="requests per vLLM call (throughput only)")
    lb = sub.add_parser("leaderboard"); lb.add_argument("--runs", default="runs")
    lb.add_argument("--out", default="results/leaderboard.md")
    sub.add_parser("routers")
    cr = sub.add_parser("check-router"); cr.add_argument("config"); cr.add_argument("overrides", nargs="*")
    cr.add_argument("--benchmark", default="configs/benchmarks/synthetic_shared.yaml")
    smoke = sub.add_parser("smoke-pipeline")
    smoke.add_argument("--output", required=True)
    a, extras = p.parse_known_args(argv)
    # argparse otherwise rejects dotlist overrides placed after optional flags.
    if extras:
        if not hasattr(a, "overrides") or any(x.startswith("-") or "=" not in x for x in extras):
            p.error("unrecognized arguments: " + " ".join(extras))
        a.overrides.extend(extras)

    from .config import load_benchmark_config, load_experiment

    if a.cmd == "smoke-pipeline":
        from .collection.smoke import run_smoke
        run_smoke(a.output)
    elif a.cmd == "train":
        from .runner import run_experiment

        run_experiment(load_experiment(a.config, a.overrides))
    elif a.cmd == "check-router":
        from .contract import run_check

        raise SystemExit(0 if run_check(a.config, a.benchmark, a.overrides) else 1)
    elif a.cmd == "routers":
        from .registry import list_routers

        print("\n".join(list_routers()))
    elif a.cmd == "leaderboard":
        from .leaderboard import write_leaderboard

        print(write_leaderboard(a.runs, a.out))
    else:
        cfg = load_benchmark_config(a.config, a.overrides)
        shared = cfg.get("protocol") == "shared-v1"  # team data: llm_routing.collection; older configs: legacy_pipeline
        if a.cmd in ("prepare", "info"):
            from .data.loaders import load_benchmark

            b = load_benchmark(cfg, rebuild=getattr(a, "rebuild", False))
            print(json.dumps(b.summary(), indent=2))
        elif a.cmd == "build-queries":
            if shared:
                from .collection.dataset import build_queries
            else:
                from .legacy_pipeline.build import build_queries

            build_queries(cfg)
        elif a.cmd == "validate-data":
            from .collection.assemble import frozen_dir, validate_release
            print(json.dumps(validate_release(frozen_dir(cfg)), indent=2))
        elif a.cmd == "evaluate-data":
            from .collection.evaluate import evaluate_data
            print(json.dumps(evaluate_data(cfg), indent=2))
        elif a.cmd == "package-contribution":
            from .collection.assemble import package
            options = {k: getattr(a, k) for k in ("member", "shard_index", "num_shards") if getattr(a, k) is not None}
            print(package(cfg, a.model, **options))
        elif a.cmd == "generate":
            if shared:
                from .collection.collect import generate
            else:
                from .legacy_pipeline.build import generate

            options = {k: getattr(a, k) for k in ("member", "shard_index", "num_shards", "max_retries", "retry_delay", "batch_size") if getattr(a, k) is not None}
            generate(cfg, a.model, **options)
        elif a.cmd == "score":
            if shared:
                from .collection.collect import score
            else:
                from .legacy_pipeline.build import score

            options = {k: getattr(a, k) for k in ("member", "shard_index", "num_shards", "scorer") if getattr(a, k) is not None}
            score(cfg, a.model, **options)
        elif a.cmd == "assemble":
            if shared:
                from .collection.assemble import assemble
            else:
                from .legacy_pipeline.build import assemble

            assemble(cfg)
        elif a.cmd == "push-data":
            from .data.hub import push

            if shared:
                from .collection.assemble import frozen_dir
                push(str(frozen_dir(cfg)), cfg["hub_repo"])
            else:
                push(cfg["path"], cfg["hub_repo"])
        elif a.cmd == "pull-data":
            from .data.hub import pull

            if shared:
                from .collection.assemble import frozen_dir
                if not cfg.get("hub_revision"):
                    raise ValueError("shared releases require a pinned hub_revision commit")
                pull(cfg["hub_repo"], str(frozen_dir(cfg)), cfg["hub_revision"])
            else:
                pull(cfg["hub_repo"], cfg["path"], cfg.get("hub_revision"))


if __name__ == "__main__":
    main()
