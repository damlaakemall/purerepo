"""Reference routers every new method should be compared against.

RandomRouter  — sends a random fraction of queries to the small model. Its
                quality falls linearly with cost advantage; a learned router
                must beat this line to be worth anything.
OracleRouter  — cheats: looks at the *realized* quality of both responses.
                Upper bound on what any router could achieve with these models.
"""

from __future__ import annotations

import numpy as np

from ..core.base import Router
from ..core.types import RoutingDataset


class RandomRouter(Router):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, queries):
        return self.rng.random(len(queries))


class OracleRouter(Router):
    """Score = realized quality gap q(S) - q(L) of the first sample. Only usable on labelled data."""

    name = "oracle"

    def __init__(self, data: RoutingDataset):
        s, l = data.realized_scores()
        self._gap = dict(zip(data.queries, s - l))

    def score(self, queries):
        gap = np.array([self._gap[q] for q in queries])
        return 1.0 / (1.0 + np.exp(-gap))  # squash into [0, 1], order is what matters
