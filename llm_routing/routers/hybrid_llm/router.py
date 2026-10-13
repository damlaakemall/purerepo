"""Hybrid LLM router (Ding et al., ICLR 2024 — arXiv:2404.14618).

    train:  (query, labels from labels.py)  ->  soft-label classifier p_w(x)
    test:   send x to the SMALL model  iff  p_w(x) >= threshold

One trained router supports every cost/quality operating point: only the
threshold changes.
"""

from __future__ import annotations

import numpy as np

from ...core.base import Router
from ...core.classifiers import SoftLabelClassifier, make_classifier
from ...core.types import RoutingDataset
from . import labels as L


class HybridLLMRouter(Router):
    """``mode`` is one of "det", "prob", "trans" (r_det / r_prob / r_trans in the paper)."""

    def __init__(self, mode: str = "trans", classifier: SoftLabelClassifier | None = None,
                 backend: str = "tfidf", t: float | None = None, estimator: str = "all_pairs", **clf_kwargs):
        assert mode in ("det", "prob", "trans"), mode
        self.mode, self.t, self.estimator = mode, t, estimator
        self.name = f"hybrid_llm/r_{mode}"
        self.classifier = classifier or make_classifier(backend, **clf_kwargs)

    def make_labels(self, data: RoutingDataset) -> np.ndarray:
        s, l = data.small_scores, data.large_scores
        if self.mode == "det":
            return L.deterministic_labels(s, l)
        if self.mode == "prob":
            return L.probabilistic_labels(s, l, self.estimator)
        y, self.t = L.transformed_labels(s, l, self.t, self.estimator)
        return y

    def fit(self, train: RoutingDataset, val: RoutingDataset | None = None):
        self.train_labels_ = self.make_labels(train)
        self.classifier.fit(train.queries, self.train_labels_)
        return self

    def score(self, queries):
        return self.classifier.predict_proba(list(queries))
