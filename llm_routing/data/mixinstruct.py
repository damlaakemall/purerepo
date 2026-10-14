"""MixInstruct (Jiang et al., 2023 — LLM-Blender), the dataset Hybrid LLM uses.

Each example is an instruction-following query with ONE response from each of
12 open LLMs and *precomputed* quality scores per response (BARTScore,
BERTScore, BLEURT, ...). That lets us train/evaluate routers among those models
with zero generation cost.

Limitation: k = 1 sample per model, so Hybrid LLM's r_prob equals r_det here.
MixInstruct has no costs; give a per-model cost (e.g. parameter count in
billions) in the benchmark config.

Splits: our train/val/test are taken from MixInstruct's own train/validation/test.
"""

from __future__ import annotations

import itertools

import numpy as np

from .benchmark import Benchmark

HF_ID = "llm-blender/mix-instruct"
MODELS = ["vicuna-13b-1.1", "flan-t5-xxl", "stablelm-tuned-alpha-7b", "koala-7B-HF", "dolly-v2-12b",
          "chatglm-6b", "oasst-sft-4-pythia-12b-epoch-3.5", "llama-7b-hf-baize-lora-bf16",
          "moss-moon-003-sft", "mpt-7b", "mpt-7b-instruct", "alpaca-native"]


def query_text(ex: dict) -> str:
    instr, inp = (ex.get("instruction") or "").strip(), (ex.get("input") or "").strip()
    return f"{instr}\n{inp}".strip()


def load_mixinstruct(models: list[str] | None = None, metric: str = "bartscore", model_costs: dict | None = None,
                     max_per_split: dict | None = None, name: str = "mixinstruct",
                     keep_responses: bool = False) -> Benchmark:
    from datasets import load_dataset

    models = models or MODELS
    max_per_split = max_per_split or {"train": 10000, "val": 1000, "test": 5000}
    hf_split = {"train": "train", "val": "validation", "test": "test"}
    ids, qs, refs, splits, scores = [], [], [], [], []
    responses = {m: [] for m in models} if keep_responses else None
    for ours, theirs in hf_split.items():
        stream = load_dataset(HF_ID, split=theirs, streaming=True)
        n = 0
        for ex in itertools.takewhile(lambda _: n < max_per_split[ours], stream):
            by_model = {c["model"]: c for c in ex["candidates"]}
            row = [by_model.get(m, {}).get("scores", {}).get(metric) for m in models]
            if any(v is None for v in row):
                continue
            ids.append(ex["id"]); qs.append(query_text(ex)); refs.append(ex["output"]); splits.append(ours)
            scores.append(row)
            if keep_responses:
                for m in models:
                    responses[m].append([by_model[m]["text"]])
            n += 1
    costs = np.array([(model_costs or {}).get(m, 1.0) for m in models], dtype=float)
    return Benchmark(name, ids, qs, list(models), np.array(scores)[:, :, None], costs, np.array(splits),
                     sources=[i.split("/")[0] for i in ids], references=refs,
                     meta={"source": HF_ID, "metric": metric, "cost_unit": "config model_costs"},
                     responses=responses)


def load_mixinstruct_pair(small_model: str, large_model: str, split: str = "train",
                          max_examples: int | None = None, metric: str = "bartscore"):
    """Binary small/large view of MixInstruct (RoutingDataset), for two-model routers."""
    ours = {"train": "train", "validation": "val", "test": "test"}.get(split, split)
    cap = max_examples or 10**9
    b = load_mixinstruct([small_model, large_model], metric,
                         max_per_split={s: (cap if s == ours else 0) for s in ("train", "val", "test")})
    return b.pair(small_model, large_model)
