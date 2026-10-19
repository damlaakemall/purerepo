"""Summarize one generated + scored shard, to check it before packaging or assembling.

    python cluster/tosun/inspect_shard.py CONFIG MODEL MEMBER [--num-shards N --shard-index I] [--show-failed K]

Prints what docs/walkthrough/01_egitim_verisi.md (section 5) walks through: how responses ended (stop / length,
reasoning trace closed or cut off), GT per source with its evaluation modes, ArmoRM and BARTScore per
source, ArmoRM on GT-correct vs GT-wrong answers, and the last lines of a few GT-failed answers to read.
Reads files only; loads no model.
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict

from llm_routing.config import load_benchmark_config
from llm_routing.collection.collect import answer_text, job_spec
from llm_routing.collection.io import read_jsonl


def by_key(rows):
    """Last successful record per (query_id, sample_id), as the pipeline itself keeps them."""
    return {(r["query_id"], r["sample_id"]): r for r in rows if r["success"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config"); p.add_argument("model"); p.add_argument("member")
    p.add_argument("--num-shards", type=int, default=1); p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--show-failed", type=int, default=3, help="how many GT-failed answers to print")
    a = p.parse_args()
    cfg = load_benchmark_config(a.config)
    path, queries, _ = job_spec(cfg, a.model, a.member, a.shard_index, a.num_shards)
    q = {r["id"]: r for r in queries}
    responses = by_key(read_jsonl(path / "responses.jsonl"))
    print(f"{path}\n")

    meta = [r["metadata"] for r in responses.values()]
    tokens = sorted(m["completion_tokens"] for m in meta)
    print(f"responses: {len(responses)}  finish_reason: {dict(Counter(m['finish_reason'] for m in meta))}")
    print(f"completion tokens: min {tokens[0]}  median {tokens[len(tokens) // 2]}  max {tokens[-1]}")
    texts = [r["response"] for r in responses.values()]
    if any("<think>" in t for t in texts):
        prefix = Counter(repr(m.get("response_prefix", "")) for m in meta)
        print(f"reasoning: response_prefix {dict(prefix)}  trace closed {sum('</think>' in t for t in texts)}  "
              f"empty after strip {sum(not answer_text(t, 'strip_think') for t in texts)}")

    scores = {}
    for channel in ("ground_truth", "armorm", "bartscore"):
        if (path / f"{channel}.jsonl").exists():
            rows = read_jsonl(path / f"{channel}.jsonl")
            scores[channel] = by_key(rows)
            failed = [r["error"] for r in rows if not r["success"]]
            print(f"{channel}: {len(scores[channel])} scored, {len(failed)} errors" + (f" e.g. {failed[0]}" if failed else ""))

    print("\nper source (mean)")
    per = defaultdict(lambda: defaultdict(list))
    modes = defaultdict(Counter)
    for channel, rows in scores.items():
        for k, r in rows.items():
            source = q[k[0]]["source"]
            per[source][channel].append(r["score"])
            if channel == "ground_truth":
                modes[source][r["details"]["mode"]] += 1
    for source in sorted(per):
        cells = "  ".join(f"{c} {statistics.mean(v):9.3f}" for c, v in per[source].items())
        print(f"  {source:14} n={len(next(iter(per[source].values())))}  {cells}  GT modes {dict(modes[source])}")

    gt, rm = scores.get("ground_truth", {}), scores.get("armorm", {})
    binary = [k for k in gt if q[k[0]]["answer_type"] != "reference_text"]
    right = [rm[k]["score"] for k in binary if k in rm and gt[k]["score"] == 1.0]
    wrong = [rm[k]["score"] for k in binary if k in rm and gt[k]["score"] == 0.0]
    if right and wrong:
        print(f"\nArmoRM on 0/1 sources: GT-correct {statistics.mean(right):.3f} (n={len(right)})  "
              f"GT-wrong {statistics.mean(wrong):.3f} (n={len(wrong)})  -- correct should score higher")

    failed = [k for k in binary if gt[k]["details"]["mode"] == "failed"][:a.show_failed]
    for k in failed:
        query, details = q[k[0]], gt[k]["details"]
        tail = answer_text(responses[k]["response"], "strip_think")[-300:].replace("\n", " | ")
        print(f"\n--- GT failed: {query['source']}  gold {str(query['gold'])[:80]}  read as {details['predicted']}\n    ...{tail}")


if __name__ == "__main__":
    main()
