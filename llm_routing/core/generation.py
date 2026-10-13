"""Sample several responses per query from a Hugging Face causal LM.

The probabilistic router in Hybrid LLM needs k > 1 stochastic samples per
(query, model) to estimate Pr[small model is at least as good as large]. The
paper uses k = 10 with sampling decoding.
"""

from __future__ import annotations


def sample_responses(model_name: str, prompts: list[str], k: int = 10, max_new_tokens: int = 256,
                     temperature: float = 0.7, top_p: float = 0.95, batch_size: int = 4,
                     device_map: str = "auto") -> list[list[str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device_map).eval()

    def fmt(p: str) -> str:
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        return p

    results: list[list[str]] = []
    for b in range(0, len(prompts), batch_size):
        batch = [fmt(p) for p in prompts[b : b + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                                 max_new_tokens=max_new_tokens, num_return_sequences=k,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i in range(len(batch)):
            results.append(texts[i * k : (i + 1) * k])
    return results
