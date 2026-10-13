"""Train and compare Hybrid LLM routers (r_det, r_prob, r_trans) against baselines.

Examples
    # CPU, seconds — synthetic data
    python scripts/run_hybrid_llm.py --data synthetic --offset 0.3
    python scripts/run_hybrid_llm.py --data synthetic --offset 1.0      # weak small model -> class imbalance

    # real data: MixInstruct precomputed BARTScores (needs `pip install datasets`)
    python scripts/run_hybrid_llm.py --data mixinstruct --small flan-t5-xxl --large vicuna-13b-1.1

    # your own generated + scored responses (see generate_responses.py / score_responses.py)
    python scripts/run_hybrid_llm.py --data jsonl --train data/train.jsonl --test data/test.jsonl

    # paper setup: DeBERTa-v3-large router (GPU recommended)
    python scripts/run_hybrid_llm.py --data mixinstruct ... --backend deberta
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_routing.experiment import compare_routers, plot_curves, print_table  # noqa: E402
from llm_routing.routers.baselines import OracleRouter, RandomRouter  # noqa: E402
from llm_routing.routers.hybrid_llm import HybridLLMRouter  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synthetic", "mixinstruct", "jsonl"], default="synthetic")
    p.add_argument("--offset", type=float, default=0.3, help="synthetic: how much weaker the small model is")
    p.add_argument("--n", type=int, default=4000, help="synthetic: number of queries")
    p.add_argument("--small", default="flan-t5-xxl")
    p.add_argument("--large", default="vicuna-13b-1.1")
    p.add_argument("--max-train", type=int, default=10000)
    p.add_argument("--max-test", type=int, default=5000)
    p.add_argument("--train"); p.add_argument("--test")
    p.add_argument("--backend", choices=["tfidf", "deberta"], default="tfidf")
    p.add_argument("--model-name", default="microsoft/deberta-v3-large")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.data == "synthetic":
        from llm_routing.data.synthetic import make_synthetic

        train, test = make_synthetic(n=a.n, offset=a.offset, seed=a.seed).split((0.75, 0.25), seed=a.seed)
        tag = f"synthetic_offset{a.offset}"
    elif a.data == "mixinstruct":
        from llm_routing.data.mixinstruct import load_mixinstruct_pair

        train = load_mixinstruct_pair(a.small, a.large, "train", a.max_train)
        test = load_mixinstruct_pair(a.small, a.large, "test", a.max_test)
        tag = f"mixinstruct_{a.small}_vs_{a.large}"
    else:
        from llm_routing.data.io import load_jsonl

        train, test = load_jsonl(a.train), load_jsonl(a.test)
        tag = f"jsonl_{train.small_model}_vs_{train.large_model}".replace("/", "_")

    q_s, q_l = test.realized_scores()
    print(f"[{tag}] train={len(train)} test={len(test)} k={train.num_samples}")
    print(f"  mean quality small={q_s.mean():.3f}  large={q_l.mean():.3f}  "
          f"Pr[small >= large] on test={(q_s >= q_l).mean():.3f}\n")

    clf_kwargs = {"model_name": a.model_name} if a.backend == "deberta" else {}
    routers = [HybridLLMRouter(m, backend=a.backend, **clf_kwargs) for m in ("det", "prob", "trans")]
    routers += [RandomRouter(a.seed), OracleRouter(test)]
    rows, curves = compare_routers(routers, train, test)
    print_table(rows)

    os.makedirs(a.out, exist_ok=True)
    plot_curves(curves, os.path.join(a.out, f"{tag}.png"), tag)
    with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {a.out}/{tag}.png and .json")


if __name__ == "__main__":
    main()
