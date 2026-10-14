"""Lazy backends. One independent sample per request keeps resume/sharding seeds stable."""
from __future__ import annotations

from .io import digest


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
        if self.engine not in ("vllm", "hf", "api"):
            raise ValueError(f"unsupported engine: {self.engine}")

    def _load(self):
        if self.loaded:
            return
        m, g = self.model, self.generation
        self.model_id = m.get("model_id", m.get("hf_id"))
        if not self.model_id:
            raise ValueError("model_id or hf_id is required")
        if self.engine != "api" and (not m.get("revision") or m["revision"].startswith("PIN_")):
            raise ValueError("pin the model revision before generation")
        if self.engine == "api":
            import os
            from openai import OpenAI
            options = m.get("api", {})
            # Only environment variable NAMES enter artifacts, never credential values.
            api_key = os.environ[options.get("key_env", "OPENAI_API_KEY")]
            self.client = OpenAI(api_key=api_key, base_url=os.environ.get(options.get("base_url_env", "OPENAI_BASE_URL")),
                                 max_retries=0, timeout=options.get("timeout", 120))
        elif self.engine == "vllm":
            from vllm import LLM
            self.llm = LLM(model=self.model_id, revision=m.get("revision"), tokenizer_revision=m.get("tokenizer_revision", m.get("revision")),
                           generation_config="vllm", **m.get("engine_kwargs", {}))
            self.tok = self.llm.get_tokenizer()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=m.get("tokenizer_revision", m.get("revision")))
            self.llm = AutoModelForCausalLM.from_pretrained(self.model_id, revision=m.get("revision"),
                                                           device_map="auto", **m.get("engine_kwargs", {})).eval()
        self.loaded = True

    def __call__(self, query, seed):
        self._load()
        m, g = self.model, self.generation
        messages = messages_for(m, query)
        meta = {"engine": self.engine, "model_id": self.model_id, "requested_revision": m.get("revision"),
                "seed_requested": seed, "messages": messages}
        if self.engine == "api":
            options = m.get("api", {})
            request = {"model": self.model_id, "messages": messages}
            if options.get("send_sampling_parameters", True):
                request.update(temperature=g["temperature"], top_p=g["top_p"])
            if options.get("supports_seed", False):
                request["seed"] = seed
            request[options.get("max_tokens_field", "max_tokens")] = g["max_new_tokens"]
            extra = options.get("request_kwargs", {})
            if set(extra) & set(request):
                raise ValueError("api.request_kwargs may not override recorded generation fields")
            request.update(extra)
            result = self.client.chat.completions.create(**request)
            choice, usage = result.choices[0], result.usage
            meta.update(request=request, seed_applied="seed" in request, served_model=result.model,
                        system_fingerprint=getattr(result, "system_fingerprint", None),
                        prompt_tokens=getattr(usage, "prompt_tokens", None),
                        completion_tokens=getattr(usage, "completion_tokens", None), finish_reason=choice.finish_reason,
                        usage=usage.model_dump() if usage else None)
            return choice.message.content or "", meta
        template = getattr(self.tok, "chat_template", None)
        if not template and m.get("system_prompt"):
            raise ValueError("system_prompt requires a chat template")
        formatted = (self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **m.get("chat_template_kwargs", {})) if template else query)
        meta.update(formatted_prompt=formatted, chat_template=template, chat_template_hash=digest(template),
                    chat_template_kwargs=m.get("chat_template_kwargs", {}), tokenizer_name=self.tok.name_or_path,
                    tokenizer_revision=self.tok.init_kwargs.get("_commit_hash"), seed_applied=True)
        # Already-formatted chat prompts include special tokens. Do not add them twice.
        ids = self.tok.encode(formatted, add_special_tokens=not bool(template))
        meta.update(prompt_token_ids_hash=digest(ids), prompt_tokens=len(ids))
        if self.engine == "vllm":
            from vllm import SamplingParams
            p = SamplingParams(n=1, temperature=g["temperature"], top_p=g["top_p"],
                               max_tokens=g["max_new_tokens"], seed=seed)
            meta["sampling_params"] = str(p)
            out = self.llm.generate([{"prompt_token_ids": ids}], p, use_tqdm=False)[0].outputs[0]
            meta.update(completion_tokens=len(out.token_ids), finish_reason=out.finish_reason)
            return out.text, meta
        import torch
        from transformers import set_seed
        set_seed(seed)
        inputs = torch.tensor([ids], device=self.llm.device)
        kwargs = {"max_new_tokens": g["max_new_tokens"], "do_sample": g["temperature"] > 0}
        if kwargs["do_sample"]:
            kwargs.update(temperature=g["temperature"], top_p=g["top_p"])
        meta["effective_generation"] = {**self.llm.generation_config.to_dict(), **kwargs}
        with torch.inference_mode():
            seq = self.llm.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    pad_token_id=self.tok.eos_token_id, **kwargs)[0, len(ids):]
        meta.update(completion_tokens=len(seq), finish_reason="length" if len(seq) == g["max_new_tokens"] else "stop",
                    resolved_revision=getattr(self.llm.config, "_commit_hash", None))
        return self.tok.decode(seq, skip_special_tokens=True), meta
