"""Step 1 of the full paper pipeline: sample k responses per query from a small and a large model.

    python scripts/generate_responses.py --small meta-llama/Llama-2-7b-chat-hf \
        --large meta-llama/Llama-2-13b-chat-hf --split train --max-examples 10000 --k 10 --out data/raw_train.jsonl

Writes one JSON line per query: {"query", "reference", "small": [k texts], "large": [k texts]}.
Requires a GPU and `pip install torch transformers datasets accelerate`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_routing.core.generation import sample_responses  # noqa: E402
from llm_routing.data.mixinstruct import HF_ID, query_text  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", required=True); p.add_argument("--large", required=True)
    p.add_argument("--split", default="train"); p.add_argument("--max-examples", type=int, default=1000)
    p.add_argument("--k", type=int, default=10); p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from datasets import load_dataset

    rows = list(itertools.islice(load_dataset(HF_ID, split=a.split, streaming=True), a.max_examples))
    queries = [query_text(r) for r in rows]
    small = sample_responses(a.small, queries, k=a.k, max_new_tokens=a.max_new_tokens)
    large = sample_responses(a.large, queries, k=a.k, max_new_tokens=a.max_new_tokens)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(json.dumps({"_header": True, "small_model": a.small, "large_model": a.large}) + "\n")
        for r, q, s, l in zip(rows, queries, small, large):
            f.write(json.dumps({"query": q, "reference": r["output"], "small": s, "large": l}) + "\n")


if __name__ == "__main__":
    main()
