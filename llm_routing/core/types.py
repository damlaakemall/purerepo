"""Shared data containers used by every router in this repo.

A *routing dataset* is the minimum information any binary small/large router
needs:

* the query text (what the router sees at inference time), and
* quality scores of responses from each candidate model (what the router is
  trained on — never available at inference time).

We store *several* sampled responses per model (shape ``(N, k)``) because LLM
decoding is stochastic: the same query can get a good answer one time and a bad
one the next. Hybrid LLM's probabilistic router is built exactly around that
fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoutingDataset:
    queries: list[str]
    # quality of k sampled responses per query, higher is better. shape (N, k)
    small_scores: np.ndarray
    large_scores: np.ndarray
    small_model: str = "small"
    large_model: str = "large"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.small_scores = np.atleast_2d(np.asarray(self.small_scores, dtype=float))
        self.large_scores = np.atleast_2d(np.asarray(self.large_scores, dtype=float))
        if self.small_scores.shape[0] != len(self.queries):
            # a 1-D input of length N becomes (1, N) through atleast_2d; fix it
            self.small_scores = self.small_scores.reshape(len(self.queries), -1)
            self.large_scores = self.large_scores.reshape(len(self.queries), -1)
        assert self.small_scores.shape[0] == self.large_scores.shape[0] == len(self.queries)

    def __len__(self) -> int:
        return len(self.queries)

    @property
    def num_samples(self) -> int:
        return self.small_scores.shape[1]

    def realized_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Quality of *one* response per model (the first sample).

        Evaluation uses a single response per query — that is what a user would
        actually receive once the router has made its decision.
        """
        return self.small_scores[:, 0], self.large_scores[:, 0]

    def subset(self, idx) -> "RoutingDataset":
        idx = np.asarray(idx)
        return RoutingDataset(
            queries=[self.queries[i] for i in idx],
            small_scores=self.small_scores[idx],
            large_scores=self.large_scores[idx],
            small_model=self.small_model,
            large_model=self.large_model,
            meta=dict(self.meta),
        )

    def split(self, fractions=(0.8, 0.1, 0.1), seed: int = 0):
        """Random train/val/test split."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self))
        cuts = np.cumsum([int(f * len(self)) for f in fractions[:-1]])
        return [self.subset(p) for p in np.split(perm, cuts)]
