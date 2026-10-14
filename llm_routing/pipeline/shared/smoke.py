"""Exercise collection -> scoring -> packaging -> release -> evaluation, with test doubles only."""
from pathlib import Path

from ...config import REPO_ROOT, load_benchmark_config
from .assemble import assemble, package, validate_release, frozen_dir
from .collect import generate, score
from .dataset import build_queries
from .evaluate import evaluate_data


def run_smoke(output):
    cfg = load_benchmark_config(str(Path(REPO_ROOT) / "configs/benchmarks/shared_smoke.yaml"))
    cfg["path"] = str(Path(output).resolve())
    cfg["sources"][0]["local_path"] = str(Path(REPO_ROOT) / "tests/fixtures/shared_queries.jsonl")
    build_queries(cfg)
    for name, member in (("tiny", "alice"), ("large", "bob")):
        def fake_generate(query, seed):
            n = int(query.split()[1])
            answer = n + 1 if name == "large" or (n % 2 == 0 and seed % 3 != 0) else -1
            return str(answer), {"engine": "synthetic-test-double", "prompt_tokens": 5,
                                  "completion_tokens": 1 + seed % 3, "seed_applied": True}
        for shard in range(2):
            args = dict(member=member, shard_index=shard, num_shards=2)
            generate(cfg, name, backend=fake_generate, retry_delay=0, **args)
            # Repeating must not call any backend for completed samples.
            def forbidden(*_):
                raise AssertionError("resume unexpectedly called a backend")
            generate(cfg, name, backend=forbidden, retry_delay=0, **args)
            score(cfg, name, scorer="ground_truth", **args)
            score(cfg, name, scorer="armorm", evaluator=lambda q, r: (0.2 if r == "-1" else 0.8,
                                                                    {"synthetic": True, "rewards": [0.8], "gating": [1.0]}), **args)
            package(cfg, name, **args)
    bench = assemble(cfg)
    validate_release(frozen_dir(cfg))
    from ...routers.hybrid_llm.labels import gap_probability
    from .io import write_json
    pair = bench.split("train").pair("tiny", "large")
    labels = gap_probability(pair.small_scores, pair.large_scores)
    report = evaluate_data(cfg)
    write_json(Path(cfg["path"]) / "reports" / cfg["version"] / "label_smoke.json",
               {"method": "existing Hybrid LLM all-pairs labels", "n_train": len(labels),
                "labels": labels.tolist(), "note": "label construction only; no router training"})
    print(f"[smoke] {len(bench)} queries, {bench.num_models} models, 120 responses; release validated; no models loaded")
    return cfg, report
