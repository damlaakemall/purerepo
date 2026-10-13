"""Router interfaces shared by every paper in this repo.

All routers answer the same question — *which model's answer does the user get,
and what did it cost?* — so they can be evaluated by the same code. A router
exposes a family of operating points (one per value of its cost/quality knob):

    Router.sweep(benchmark) -> list[Decision]
    Decision.choice[n]  index of the model whose answer is returned for query n
    Decision.cost[n]    total cost paid for query n (a cascade may pay for several models)

Three families are anticipated:

BinaryRouter      small vs. large model, a score per query + a threshold
                  (Hybrid LLM, RouteLLM, ...)
MultiModelRouter  predicts quality of every model, picks argmax(quality - lambda * cost)
                  (RouterBench baselines, RouterDC, kNN/MLP routers, ...)
CascadeRouter     calls models in order of cost and stops once an answer is judged good
                  enough (FrugalGPT, AutoMix, ...). Sees responses, so it implements
                  ``sweep`` directly with accumulated costs.

Inference-time rule: a router may look at query text (and, for cascades, at
responses of models it has *paid for*). It must never read test-set scores or
test-set costs; costs it needs are estimated from the training split.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type hints only; avoids a core <-> data import cycle
    from ..data.benchmark import Benchmark


@dataclass
class Decision:
    knob: float
    choice: np.ndarray  # (N,) int
    cost: np.ndarray  # (N,) float


class Router(ABC):
    name: str = "router"
    family: str = "generic"

    def fit(self, train: Benchmark, val: Benchmark | None = None) -> "Router":
        return self

    @abstractmethod
    def sweep(self, bench: Benchmark, num: int = 51) -> list[Decision]:
        """Operating points from cheapest to most expensive."""

    def describe(self) -> dict:
        """Hyperparameters / learned values worth logging (e.g. t* in Hybrid LLM)."""
        return {}


class BinaryRouter(Router):
    """``score(queries)`` in [0, 1]: higher = the small model is good enough. Route to small iff score >= threshold."""

    family = "binary"

    def __init__(self, small: str, large: str):
        self.small, self.large = small, large

    @abstractmethod
    def score(self, queries: list[str]) -> np.ndarray: ...

    def route(self, queries: list[str], threshold: float) -> np.ndarray:
        return self.score(queries) >= threshold

    def sweep(self, bench: Benchmark, num: int = 51) -> list[Decision]:
        i, j = bench.model_index(self.small), bench.model_index(self.large)
        s = np.asarray(self.score(bench.queries), dtype=float)
        # thresholds at score quantiles -> evenly spaced fractions of traffic to the small model
        fracs = np.linspace(1, 0, num)  # all-small (cheapest) first
        out = []
        for f in fracs:
            t = np.inf if f == 0 else float(np.quantile(s, 1 - f))
            to_small = s >= t
            choice = np.where(to_small, i, j)
            out.append(Decision(t, choice, bench.costs[np.arange(len(bench)), choice]))
        return out


class MultiModelRouter(Router):
    """Predict per-model quality, then trade it off against estimated cost with a knob lambda."""

    family = "multi"

    def __init__(self, models: list[str] | None = None):
        self.models = models  # None = all models of the benchmark

    @abstractmethod
    def predict_quality(self, queries: list[str]) -> np.ndarray:
        """(N, M) predicted quality for ``self.models``."""

    def fit(self, train: Benchmark, val: Benchmark | None = None):
        self.models = self.models or list(train.models)
        self.est_cost_ = train.select_models(self.models).costs.mean(axis=0)  # estimated from TRAIN only
        return self

    def sweep(self, bench: Benchmark, num: int = 51) -> list[Decision]:
        cols = np.array([bench.model_index(m) for m in self.models])
        q = self.predict_quality(bench.queries)
        c = self.est_cost_ / max(self.est_cost_.max(), 1e-12)
        qr = max(float(np.nanmax(q) - np.nanmin(q)), 1e-9)
        # lambda from "ignore cost" to "cost dominates"
        lams = np.concatenate([[0.0], np.geomspace(1e-4, 1e3, num - 1)]) * qr
        out = []
        for lam in lams[::-1]:  # cheapest first
            pick = cols[np.argmax(q - lam * c[None, :], axis=1)]
            out.append(Decision(float(lam), pick, bench.costs[np.arange(len(bench)), pick]))
        return out


class CascadeRouter(Router):
    """Placeholder base for cascades; subclasses implement ``sweep`` with accumulated cost."""

    family = "cascade"
