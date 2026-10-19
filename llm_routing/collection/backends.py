"""Local GPU backends for Tosun (vLLM, or plain transformers). No paid APIs.

Every sample is its own request with its own seed, so resume and sharding never change a seed.
vLLM batches many such requests in one ``generate`` call (one SamplingParams per request), which
keeps that contract while using the GPU properly.
"""
from __future__ import annotations

import re
import time

from .io import digest

ENGINES = ("vllm", "hf")


def messages_for(model, query):
    messages = []
    if model.get("system_prompt"):
        messages.append({"role": "system", "content": model["system_prompt"]})
    return messages + [{"role": "user", "content": query}]


class Generator:
    def __init__(self, model, generation):
        self.model, self.generation = model, generation
        self.engine = generation.get("engine", "vllm")
        self.loaded = False
        if self.engine not in ENGINES:
            raise ValueError(f"unsupported engine: {self.engine} (Tosun runs local models only: {ENGINES})")

    def _load(self):
        if self.loaded:
            return
        m = self.model
        self.model_id = m.get("model_id", m.get("hf_id"))
        if not self.model_id:
            raise ValueError("model_id or hf_id is required")
        if not m.get("revision") or m["revision"].startswith("PIN_"):
            raise ValueError("pin the model revision before generation")
        engine_kwargs = dict(m.get("engine_kwargs", {}))
        if self.engine == "vllm":
            from vllm import LLM
            if m.get("gpus", 1) > 1:
                engine_kwargs.setdefault("tensor_parallel_size", m["gpus"])
            self.llm = LLM(model=self.model_id, revision=m.get("revision"), tokenizer_revision=m.get("tokenizer_revision", m.get("revision")),
                           generation_config="vllm", **engine_kwargs)
            self.tok = self.llm.get_tokenizer()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            loader = AutoModelForCausalLM
            if m.get("hf_loader", "causal_lm") == "multimodal_lm":
                # Gemma 3 4/12/27B: conditional checkpoint, used with text only.
                from transformers import AutoModelForImageTextToText
                loader = AutoModelForImageTextToText
            elif m.get("hf_loader", "causal_lm") != "causal_lm":
                raise ValueError("hf_loader must be causal_lm or multimodal_lm")
            # max_model_len is a vLLM scheduler option, not a transformers constructor argument.
            engine_kwargs.pop("max_model_len", None)
            self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=m.get("tokenizer_revision", m.get("revision")))
            self.llm = loader.from_pretrained(self.model_id, revision=m.get("revision"),
                                                           device_map="auto", **engine_kwargs).eval()
        self.loaded = True

    def _prompt(self, query, seed):
        m = self.model
        messages = messages_for(m, query)
        meta = {"engine": self.engine, "model_id": self.model_id, "requested_revision": m.get("revision"),
                "seed_requested": seed, "messages": messages}
        template = getattr(self.tok, "chat_template", None)
        if not template and m.get("system_prompt"):
            raise ValueError("system_prompt requires a chat template")
        formatted = (self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **m.get("chat_template_kwargs", {})) if template else query)
        # Some templates open the reasoning block themselves (DeepSeek-R1-Distill ends the prompt with
        # "<think>\n"), so the model's text holds only "...</think> answer". The prefix is put back in
        # front of the response so it is the whole assistant turn and strip_think can find the trace.
        opened = re.search(r"<(think|thinking|reasoning)>\s*$", formatted) if template else None
        meta["response_prefix"] = opened.group(0) if opened else ""
        meta.update(formatted_prompt=formatted, chat_template=template, chat_template_hash=digest(template),
                    chat_template_kwargs=m.get("chat_template_kwargs", {}), tokenizer_name=self.tok.name_or_path,
                    tokenizer_revision=self.tok.init_kwargs.get("_commit_hash"), seed_applied=True)
        # Already-formatted chat prompts include special tokens. Do not add them twice.
        ids = self.tok.encode(formatted, add_special_tokens=not bool(template))
        meta.update(prompt_token_ids_hash=digest(ids), prompt_tokens=len(ids))
        return ids, meta

    def generate_batch(self, items):
        """items: [(query, seed)] -> [(text, metadata) or Exception], in order."""
        self._load()
        g = self.generation
        prepared = []
        for query, seed in items:
            try:
                prepared.append(self._prompt(query, seed))
            except Exception as e:  # one malformed prompt must not sink the batch
                prepared.append(e)
        if self.engine != "vllm":
            return [p if isinstance(p, Exception) else self._hf(*p, seed=s) for p, (_, s) in zip(prepared, items)]
        from vllm import SamplingParams
        live = [(i, p) for i, p in enumerate(prepared) if not isinstance(p, Exception)]
        params = [SamplingParams(n=1, temperature=g["temperature"], top_p=g["top_p"], max_tokens=g["max_new_tokens"],
                                 seed=items[i][1]) for i, _ in live]
        start = time.perf_counter()
        try:
            outputs = self.llm.generate([{"prompt_token_ids": p[0]} for _, p in live], params, use_tqdm=False) if live else []
        except Exception as e:
            return [p if isinstance(p, Exception) else e for p in prepared]
        per_sample = (time.perf_counter() - start) / max(len(live), 1)
        results = list(prepared)
        for (i, (_, meta)), out, p in zip(live, outputs, params):
            o = out.outputs[0]
            results[i] = (meta["response_prefix"] + o.text, {**meta, "sampling_params": str(p), "completion_tokens": len(o.token_ids),
                                   "finish_reason": o.finish_reason, "batch_size": len(live),
                                   "batch_seconds_per_sample": per_sample})
        return results

    def _hf(self, ids, meta, seed):
        import torch
        from transformers import set_seed
        g = self.generation
        try:
            set_seed(seed)
            inputs = torch.tensor([ids], device=self.llm.device)
            kwargs = {"max_new_tokens": g["max_new_tokens"], "do_sample": g["temperature"] > 0}
            if kwargs["do_sample"]:
                kwargs.update(temperature=g["temperature"], top_p=g["top_p"])
            meta["effective_generation"] = {**self.llm.generation_config.to_dict(), **kwargs}
            start = time.perf_counter()
            with torch.inference_mode():
                seq = self.llm.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                        pad_token_id=self.tok.eos_token_id, **kwargs)[0, len(ids):]
            meta.update(completion_tokens=len(seq), finish_reason="length" if len(seq) == g["max_new_tokens"] else "stop",
                        resolved_revision=getattr(self.llm.config, "_commit_hash", None), batch_size=1,
                        batch_seconds_per_sample=time.perf_counter() - start)
            return meta["response_prefix"] + self.tok.decode(seq, skip_special_tokens=True), meta
        except Exception as e:
            return e

    def __call__(self, query, seed):
        out = self.generate_batch([(query, seed)])[0]
        if isinstance(out, Exception):
            raise out
        return out
