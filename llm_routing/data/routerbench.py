"""RouterBench (Hu et al., 2024 — Martian), a public multi-model routing benchmark.

~36k prompts from 8+ tasks (MMLU, HellaSwag, GSM8K, MBPP, ARC, Winogrande, MT-Bench, RAG),
11 models (GPT-4, GPT-3.5, Claude, Llama-2-70B, Mixtral, Mistral-7B, Yi-34B, ...).
Per (prompt, model): correctness score in [0, 1], the response, and the real $ cost.
k = 1 sample per model. Splits are assigned by us (seeded), stratified by task.

The file is a pandas pickle. Unpickling can execute code, so we load it with a
restricted unpickler that only accepts pandas / numpy / builtin classes.
"""

from __future__ import annotations

import ast
import pickle

import numpy as np

from .benchmark import Benchmark

HF_ID = "withmartian/routerbench"
_ALLOWED_PREFIXES = ("pandas.", "numpy.", "builtins", "datetime", "collections", "copyreg", "_codecs")


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(_ALLOWED_PREFIXES) or module in ("numpy", "pandas"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked class {module}.{name} in RouterBench pickle")


def _prompt_text(p) -> str:
    if isinstance(p, str) and p.startswith("["):
        try:
            parts = ast.literal_eval(p)
            if isinstance(parts, list):
                return "\n".join(map(str, parts))
        except (ValueError, SyntaxError):
            pass
    return str(p)


def load_routerbench(variant: str = "0shot", models: list[str] | None = None, splits: dict | None = None,
                     seed: int = 0, max_queries: int | None = None, keep_responses: bool = False,
                     name: str = "routerbench") -> Benchmark:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_ID, f"routerbench_{variant}.pkl", repo_type="dataset")
    with open(path, "rb") as f:
        df = _SafeUnpickler(f).load()
    all_models = [c for c in df.columns if "|" not in c and c not in ("sample_id", "prompt", "eval_name",
                                                                       "oracle_model_to_route_to")]
    models = models or all_models
    if max_queries:
        df = df.sample(n=min(max_queries, len(df)), random_state=seed)
    df = df.reset_index(drop=True)

    # stratified split by task
    fr = splits or {"train": 0.7, "val": 0.1, "test": 0.2}
    rng = np.random.default_rng(seed)
    split_of = np.empty(len(df), dtype=object)
    for _, idx in df.groupby("eval_name").indices.items():
        idx = rng.permutation(idx)
        a = int(round(fr["train"] * len(idx))); b = a + int(round(fr["val"] * len(idx)))
        split_of[idx[:a]], split_of[idx[a:b]], split_of[idx[b:]] = "train", "val", "test"

    scores = df[models].to_numpy(dtype=float)[:, :, None]
    costs = df[[f"{m}|total_cost" for m in models]].to_numpy(dtype=float)
    responses = ({m: [[str(r)] for r in df[f"{m}|model_response"]] for m in models} if keep_responses else None)
    return Benchmark(name, df["sample_id"].astype(str).tolist(), [_prompt_text(p) for p in df["prompt"]], models,
                     scores, costs, split_of.astype(str), sources=df["eval_name"].astype(str).tolist(),
                     meta={"source": HF_ID, "variant": variant, "metric": "correctness", "cost_unit": "USD"},
                     responses=responses)
