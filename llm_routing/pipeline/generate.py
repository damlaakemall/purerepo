"""Generate k sampled responses per query for one model. Resumable (cluster jobs get killed).

Writes <benchmark_dir>/responses/<model>.jsonl, one line per query:
    {"id", "responses": [k texts], "prompt_tokens": int, "completion_tokens": [k ints]}

Engines:
    vllm  fast batched sampling on GPU (recommended on the cluster; `pip install vllm`)
    hf    plain transformers.generate (works everywhere, slower)
"""

from __future__ import annotations

import json
import os


def _done_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _chat(tok, prompt: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
    return prompt


def generate_for_model(hf_id: str, rows: list[dict], out_path: str, k: int = 10, temperature: float = 0.7,
                       top_p: float = 0.95, max_new_tokens: int = 512, engine: str = "vllm",
                       batch_size: int = 64, seed: int = 0, **engine_kwargs) -> None:
    done = _done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"[generate] {hf_id}: {len(done)} done, {len(todo)} to go -> {out_path}")
    if not todo:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gen = _vllm if engine == "vllm" else _hf
    with open(out_path, "a") as f:
        for b in range(0, len(todo), batch_size):
            batch = todo[b : b + batch_size]
            for r, out in zip(batch, gen(hf_id, [x["query"] for x in batch], k, temperature, top_p,
                                         max_new_tokens, seed, **engine_kwargs)):
                f.write(json.dumps({"id": r["id"], **out}, ensure_ascii=False) + "\n")
            f.flush()


_CACHE: dict = {}


def _vllm(hf_id, prompts, k, temperature, top_p, max_new_tokens, seed, **kw):
    from vllm import LLM, SamplingParams

    if hf_id not in _CACHE:
        _CACHE.clear()
        _CACHE[hf_id] = LLM(model=hf_id, seed=seed, **kw)
    llm = _CACHE[hf_id]
    tok = llm.get_tokenizer()
    params = SamplingParams(n=k, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed)
    outs = llm.generate([_chat(tok, p) for p in prompts], params, use_tqdm=False)
    return [{"responses": [c.text for c in o.outputs], "prompt_tokens": len(o.prompt_token_ids),
             "completion_tokens": [len(c.token_ids) for c in o.outputs]} for o in outs]


def _hf(hf_id, prompts, k, temperature, top_p, max_new_tokens, seed, **kw):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if hf_id not in _CACHE:
        _CACHE.clear()
        tok = AutoTokenizer.from_pretrained(hf_id, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype="auto", device_map="auto", **kw).eval()
        _CACHE[hf_id] = (tok, model)
    tok, model = _CACHE[hf_id]
    torch.manual_seed(seed)
    enc = tok([_chat(tok, p) for p in prompts], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                             max_new_tokens=max_new_tokens, num_return_sequences=k, pad_token_id=tok.pad_token_id)
    new = gen[:, enc.input_ids.shape[1]:]
    texts = tok.batch_decode(new, skip_special_tokens=True)
    lens = (new != tok.pad_token_id).sum(dim=1).tolist()
    plen = enc.attention_mask.sum(dim=1).tolist()
    return [{"responses": texts[i * k:(i + 1) * k], "prompt_tokens": int(plen[i]),
             "completion_tokens": lens[i * k:(i + 1) * k]} for i in range(len(prompts))]
