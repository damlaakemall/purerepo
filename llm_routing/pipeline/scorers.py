"""Response-quality scorers. Routers are trained on *how good each model's answer was*.

Every scorer maps (responses, references) -> float array, higher = better.
Choose per query source in the benchmark config, e.g. exact match for math with
a known answer, BARTScore for open-ended instructions (as in Hybrid LLM).

bartscore      mean log-likelihood of the reference given the response under BART
               (Yuan et al., 2021). Negative; closer to 0 = better. Used by Hybrid LLM.
exact_match_number
               1.0 if the last number in the response equals the reference's final
               number (GSM8K-style "#### 42"), else 0.0.
multiple_choice
               1.0 if the first standalone option letter (A-E) in the response equals
               the reference letter.

Later papers use LLM-as-judge or human preference labels; add them here with
``@register_scorer``.
"""

from __future__ import annotations

import re

import numpy as np

SCORERS: dict = {}


def register_scorer(name):
    def deco(fn):
        SCORERS[name] = fn
        return fn
    return deco


def get_scorer(name: str, **kwargs):
    if name not in SCORERS:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(SCORERS)}")
    return SCORERS[name](**kwargs)


class BARTScorer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str | None = None, max_length: int = 1024,
                 batch_size: int = 16):
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device).eval()

    def __call__(self, responses: list[str], references: list[str]) -> np.ndarray:
        import torch

        out = []
        bs = self.batch_size
        for b in range(0, len(responses), bs):
            src = self.tokenizer(responses[b:b + bs], max_length=self.max_length, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            tgt = self.tokenizer(references[b:b + bs], max_length=self.max_length, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=src.input_ids, attention_mask=src.attention_mask,
                                    labels=tgt.input_ids).logits
                logp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.input_ids.unsqueeze(-1)).squeeze(-1)
                mask = tgt.attention_mask.float()
                out.append(((logp * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    # backward-compatible name
    score = __call__


@register_scorer("bartscore")
def _bartscore(**kw):
    return BARTScorer(**kw)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(s: str):
    if "####" in s:
        s = s.split("####")[-1]
    nums = _NUM.findall(s or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


@register_scorer("exact_match_number")
def _exact_number(**_):
    def score(responses, references):
        out = []
        for r, ref in zip(responses, references):
            a, b = _last_number(r), _last_number(ref)
            out.append(float(a is not None and b is not None and abs(a - b) < 1e-6))
        return np.array(out)
    return score


_LETTER = re.compile(r"\b([A-E])\b")


@register_scorer("multiple_choice")
def _multiple_choice(**_):
    def score(responses, references):
        out = []
        for r, ref in zip(responses, references):
            m = _LETTER.search(r or "")
            out.append(float(m is not None and m.group(1) == ref.strip()[:1].upper()))
        return np.array(out)
    return score
