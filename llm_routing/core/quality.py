"""Response-quality metrics.

Routers are trained on *how good each model's answer was*. Hybrid LLM uses
BARTScore (Yuan et al., 2021): the average log-likelihood of the reference
answer given the candidate response, under a BART model:

    BARTScore(response, reference) = (1/|ref|) * sum_t log p_BART(ref_t | ref_<t, response)

Scores are negative; closer to 0 = better. The paper reports it correlates well
with human judgement while being cheap — important because you need quality
scores for every (query, model, sample) triple.

Other choices used in later routing papers: LLM-as-judge scores, exact match /
accuracy on benchmarks with ground truth, human preference (Chatbot Arena)
labels. Keep the rest of the pipeline metric-agnostic.
"""

from __future__ import annotations

import numpy as np


class BARTScorer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str | None = None, max_length: int = 1024):
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device, self.max_length = device, max_length
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(device).eval()

    def score(self, responses: list[str], references: list[str], batch_size: int = 8) -> np.ndarray:
        """Mean token log-likelihood of ``references`` conditioned on ``responses``."""
        import torch

        out = []
        for b in range(0, len(responses), batch_size):
            src = self.tokenizer(responses[b : b + batch_size], max_length=self.max_length, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            tgt = self.tokenizer(references[b : b + batch_size], max_length=self.max_length, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=src.input_ids, attention_mask=src.attention_mask,
                                    labels=tgt.input_ids).logits
                logp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.input_ids.unsqueeze(-1)).squeeze(-1)
                mask = tgt.attention_mask.float()
                out.append(((logp * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(out)
