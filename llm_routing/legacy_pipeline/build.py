"""Legacy single-user pipeline (`ours_v1`); team `shared-v1` data lives in `llm_routing.collection`.

Build OUR OWN benchmark: queries -> responses (per model) -> scores (per model) -> frozen Benchmark.

    llmr build-queries configs/benchmarks/ours_v1.yaml          # CPU, once
    llmr generate      configs/benchmarks/ours_v1.yaml --model X   # GPU, one job per model
    llmr score         configs/benchmarks/ours_v1.yaml --model X   # GPU, one job per model
    llmr assemble      configs/benchmarks/ours_v1.yaml          # CPU, once -> frozen benchmark

Directory layout under ``cfg.path``:

    queries.jsonl                       fixed ids, text, reference, source, split
    responses/<model>.jsonl             k samples per query (resumable)
    scores/<model>.npz                  scores (N, k), cost (N,)
    manifest.json, scores.npy, costs.npy   written by `assemble` (the frozen benchmark)
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

from ..data.benchmark import Benchmark, assign_splits, safe_name
from .generate import generate_for_model
from .scorers import get_scorer


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def _mmlu(ex):
    letters = "ABCDE"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
    q = f"{ex['question']}\n{opts}\nAnswer with the letter of the correct option."
    return q, letters[int(ex["answer"])]


# sources whose fields can't be expressed with a simple query_template
FORMATTERS = {"mmlu": _mmlu}


def _read_queries(path: str) -> list[dict]:
    with open(os.path.join(path, "queries.jsonl")) as f:
        return [json.loads(line) for line in f]


def build_queries(cfg: dict) -> None:
    from datasets import load_dataset

    rows = []
    for src in cfg["sources"]:
        ds = load_dataset(src["hf_id"], src.get("hf_config"), split=src.get("hf_split", "train"), streaming=True)
        ds = ds.shuffle(seed=cfg.get("seed", 0), buffer_size=10_000)
        for i, ex in enumerate(itertools.islice(ds, src["n"])):
            if src.get("formatter"):
                q, ref = FORMATTERS[src["formatter"]](ex)
            else:
                q = src["query_template"].format_map(_SafeDict(ex)).strip()
                ref = str(ex.get(src["reference_field"], ""))
            rows.append({"id": f"{src['name']}/{ex.get('id', i)}", "query": q, "reference": ref,
                         "source": src["name"]})
    splits = assign_splits(len(rows), dict(cfg["splits"]), cfg.get("seed", 0))
    os.makedirs(cfg["path"], exist_ok=True)
    with open(os.path.join(cfg["path"], "queries.jsonl"), "w") as f:
        for r, s in zip(rows, splits):
            f.write(json.dumps({**r, "split": str(s)}, ensure_ascii=False) + "\n")
    print(f"[build-queries] {len(rows)} queries -> {cfg['path']}/queries.jsonl")


def _model_cfg(cfg: dict, name: str) -> dict:
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"model {name!r} not in config; available: {[m['name'] for m in cfg['models']]}")


def generate(cfg: dict, model: str, **job_options) -> None:
    if job_options:
        raise ValueError("member/shard options require protocol: shared-v1")
    m = _model_cfg(cfg, model)
    g = dict(cfg.get("generation", {}))
    g.update(m.get("generation", {}))
    generate_for_model(m["hf_id"], _read_queries(cfg["path"]),
                       os.path.join(cfg["path"], "responses", safe_name(model) + ".jsonl"),
                       seed=cfg.get("seed", 0), **g)


def score(cfg: dict, model: str, **job_options) -> None:
    if job_options:
        raise ValueError("separate scorer/member/shard options require protocol: shared-v1")
    m = _model_cfg(cfg, model)
    rows = _read_queries(cfg["path"])
    by_id = {}
    with open(os.path.join(cfg["path"], "responses", safe_name(model) + ".jsonl")) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]] = r
    missing = [r["id"] for r in rows if r["id"] not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} queries have no responses for {model}; finish `generate` first")
    k = cfg["generation"]["k"]
    scores = np.full((len(rows), k), np.nan, dtype=np.float32)
    source_scorer = {s["name"]: s["scorer"] for s in cfg["sources"]}
    scorer_kwargs = cfg.get("scoring", {}) or {}
    for scorer_name in sorted(set(source_scorer.values())):
        fn = get_scorer(scorer_name, **(scorer_kwargs.get(scorer_name) or {}))
        idx = [i for i, r in enumerate(rows) if source_scorer[r["source"]] == scorer_name]
        resp = [by_id[rows[i]["id"]]["responses"][j] for i in idx for j in range(k)]
        refs = [rows[i]["reference"] for i in idx for _ in range(k)]
        print(f"[score] {model}: {scorer_name} on {len(resp)} responses")
        scores[idx] = np.asarray(fn(resp, refs), dtype=np.float32).reshape(len(idx), k)
    tokens = np.array([by_id[r["id"]]["prompt_tokens"] + np.mean(by_id[r["id"]]["completion_tokens"])
                       for r in rows])
    cost = m["cost_per_1k_tokens"] * tokens / 1000.0
    os.makedirs(os.path.join(cfg["path"], "scores"), exist_ok=True)
    np.savez(os.path.join(cfg["path"], "scores", safe_name(model) + ".npz"), scores=scores, cost=cost)


def assemble(cfg: dict) -> Benchmark:
    rows = _read_queries(cfg["path"])
    names = [m["name"] for m in cfg["models"]]
    S, C = [], []
    for n in names:
        z = np.load(os.path.join(cfg["path"], "scores", safe_name(n) + ".npz"))
        S.append(z["scores"]); C.append(z["cost"])
    responses = {}
    for n in names:
        p = os.path.join(cfg["path"], "responses", safe_name(n) + ".jsonl")
        with open(p) as f:
            by_id = {json.loads(line)["id"]: json.loads(line)["responses"] for line in f}
        responses[n] = [by_id[r["id"]] for r in rows]
    b = Benchmark(cfg["name"], [r["id"] for r in rows], [r["query"] for r in rows], names,
                  np.stack(S, axis=1), np.stack(C, axis=1), np.array([r["split"] for r in rows]),
                  sources=[r["source"] for r in rows], references=[r["reference"] for r in rows],
                  meta={"generator": "llm_routing.legacy_pipeline", "generation": dict(cfg["generation"]),
                        "models": {m["name"]: m["hf_id"] for m in cfg["models"]},
                        "scorers": {s["name"]: s["scorer"] for s in cfg["sources"]},
                        "cost_unit": "cost_per_1k_tokens * tokens / 1000"},
                  responses=responses)
    b.save(cfg["path"])
    print(f"[assemble] {b.summary()}")
    return b
