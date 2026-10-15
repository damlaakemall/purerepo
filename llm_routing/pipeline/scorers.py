"""Response-quality scorers. Routers are trained on *how good each model's answer was*.

Every scorer maps (responses, references) -> float array, higher = better.
Choose per query source in the benchmark config, e.g. exact match for math with
a known answer, BARTScore for open-ended instructions (as in Hybrid LLM).

bartscore      mean log-likelihood of the reference given the response under BART
               (Yuan et al., 2021). Negative; closer to 0 = better. Used by Hybrid LLM.
exact_match_number
               GSM8K: the hardened numeric evaluator of ``ground_truth`` (explicit "Final Answer",
               "Answer: 42", \\boxed{42}, "#### 42", then the last number) against the
               reference's "#### <n>" answer.
multiple_choice
               the hardened MCQ evaluator of ``ground_truth``: an explicit "Answer: X" conclusion,
               then the decorated last line / prose patterns. (A first-letter regex would read
               "A cache is ..." as option A.)

Later papers use LLM-as-judge or human preference labels; add them here with
``@register_scorer``.
"""

from __future__ import annotations


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
                 batch_size: int = 16, revision: str | None = None):
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.tokenizer = BartTokenizer.from_pretrained(model_name, revision=revision)
        self.model = BartForConditionalGeneration.from_pretrained(model_name, revision=revision).to(device).eval()

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


@register_scorer("exact_match_number")
def _exact_number(**_):
    from .ground_truth.evaluators import _gsm8k_strict
    from .ground_truth.metrics import extract_gsm8k_answer

    def score(responses, references):
        return np.array([float(_gsm8k_strict(r or "", extract_gsm8k_answer(ref) or "")[0]) for r, ref in zip(responses, references)])
    return score


@register_scorer("multiple_choice")
def _multiple_choice(**_):
    from .ground_truth import evaluate

    def score(responses, references):
        return np.array([evaluate("mcq", r or "", ref.strip()[:1].upper(), answer_type="mcq_letter").score
                         for r, ref in zip(responses, references)])
    return score
