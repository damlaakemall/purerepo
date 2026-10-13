"""Step 2 of the full paper pipeline: BARTScore every sampled response against the reference.

    python scripts/score_responses.py --inp data/raw_train.jsonl --out data/train.jsonl

Output is a RoutingDataset JSONL usable by `run_hybrid_llm.py --data jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_routing.core.quality import BARTScorer  # noqa: E402
from llm_routing.core.types import RoutingDataset  # noqa: E402
from llm_routing.data.io import save_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()

    header, rows = {}, []
    with open(a.inp) as f:
        for line in f:
            r = json.loads(line)
            if r.get("_header"):
                header = r
            else:
                rows.append(r)

    scorer = BARTScorer()

    def score_side(side):
        k = len(rows[0][side])
        resp = [r[side][j] for r in rows for j in range(k)]
        refs = [r["reference"] for r in rows for _ in range(k)]
        return scorer.score(resp, refs).reshape(len(rows), k)

    data = RoutingDataset([r["query"] for r in rows], score_side("small"), score_side("large"),
                          header.get("small_model", "small"), header.get("large_model", "large"))
    save_jsonl(data, a.out)
    print(f"wrote {len(data)} rows, mean small={np.mean(data.small_scores):.3f} large={np.mean(data.large_scores):.3f}")


if __name__ == "__main__":
    main()
