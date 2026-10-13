"""Hybrid LLM router (Ding et al., ICLR 2024 — arXiv:2404.14618).

    train:  (query, labels from labels.py)  ->  soft-label classifier p_w(x)
    test:   send x to the SMALL model  iff  p_w(x) >= threshold

One trained router supports every cost/quality operating point: only the
threshold changes.
"""

from __future__ import annotations

import numpy as np

from ...core.base import BinaryRouter
from ...core.classifiers import SoftLabelClassifier, make_classifier
from ...core.types import RoutingDataset
from ...data.benchmark import Benchmark
from ...registry import register_router
from . import labels as L


@register_router("hybrid_llm")
class HybridLLMRouter(BinaryRouter):
    """``mode`` is one of "det", "prob", "trans" (r_det / r_prob / r_trans in the paper)."""

    def __init__(self, small: str, large: str, mode: str = "trans", encoder: dict | None = None,
                 classifier: SoftLabelClassifier | None = None, t: float | None = None,
                 estimator: str = "all_pairs", seed: int = 0):
        assert mode in ("det", "prob", "trans"), mode
        super().__init__(small, large)
        self.mode, self.t, self.estimator = mode, t, estimator
        self.name = f"hybrid_llm/r_{mode}"
        enc = dict(encoder or {"backend": "tfidf"})
        backend = enc.pop("backend", "tfidf")
        if backend == "transformer":
            enc.setdefault("seed", seed)
        self.classifier = classifier or make_classifier(backend, **enc)

    def make_labels(self, data: RoutingDataset) -> np.ndarray:
        s, l = data.small_scores, data.large_scores
        if self.mode == "det":
            return L.deterministic_labels(s, l)
        if self.mode == "prob":
            return L.probabilistic_labels(s, l, self.estimator)
        y, self.t = L.transformed_labels(s, l, self.t, self.estimator)
        return y

    def fit(self, train: Benchmark, val: Benchmark | None = None):
        pair = train.pair(self.small, self.large)
        self.train_labels_ = self.make_labels(pair)
        self.classifier.fit(pair.queries, self.train_labels_)
        return self

    def score(self, queries):
        return self.classifier.predict_proba(list(queries))

    def describe(self):
        d = {"mode": self.mode, "small": self.small, "large": self.large,
             "train_label_mean": float(np.mean(self.train_labels_)) if hasattr(self, "train_labels_") else None}
        if self.mode == "trans":
            d["t_star"] = self.t
        return d
