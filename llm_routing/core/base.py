"""The interface every router in this repo implements.

Convention (keep this identical across papers so routers are comparable):

* ``score(queries)`` returns a number in [0, 1] per query. **Higher means "the
  cheap/small model is good enough for this query".**
* ``route(queries, threshold)`` returns a boolean mask: ``True`` = send to the
  small model, ``False`` = send to the large model.

Moving the threshold is how you trade quality for cost *at test time* without
retraining: threshold 0 sends everything to the small model, threshold > 1
sends everything to the large model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import RoutingDataset


class Router(ABC):
    name: str = "router"

    def fit(self, train: RoutingDataset, val: RoutingDataset | None = None) -> "Router":
        """Learn from queries + response-quality scores. Default: nothing to learn."""
        return self

    @abstractmethod
    def score(self, queries: list[str]) -> np.ndarray:
        """Probability-like score that the small model is good enough."""

    def route(self, queries: list[str], threshold: float) -> np.ndarray:
        return self.score(queries) >= threshold

    def threshold_for_cost_advantage(self, queries: list[str], target: float) -> float:
        """Threshold that sends roughly ``target`` (0..1) of these queries to the small model.

        Handy in practice: you usually think in terms of "I can afford X% of
        traffic on the big model", not in terms of raw scores. Calibrate it on a
        validation set, then apply to live traffic.
        """
        s = self.score(queries)
        if target <= 0:
            return float(np.inf)
        return float(np.quantile(s, 1.0 - target))
