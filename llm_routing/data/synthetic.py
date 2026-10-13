"""A synthetic routing dataset — learn the mechanics without GPUs or API keys.

Each query has a hidden *difficulty* d in [0, 1]. The query text is built from
words whose distribution depends on d (easy chit-chat vs. hard reasoning
vocabulary), so a text classifier can partially recover d — just like a real
router must infer difficulty from the prompt alone.

Quality of k sampled responses from model m (BARTScore-like, higher = better):

    q_m ~ N(base - slope_m * d - offset_m,  noise)

Cheap models have a steep slope (fine on easy queries, bad on hard ones);
expensive models are flat. ``make_synthetic`` returns the 2-model RoutingDataset
used in the Hybrid LLM tests; ``make_synthetic_benchmark`` returns an M-model
Benchmark with costs and frozen splits.
"""

from __future__ import annotations

import numpy as np

from ..core.types import RoutingDataset
from .benchmark import Benchmark, assign_splits

EASY = ("hi hello thanks name color capital city weather favorite animal translate word define simple "
        "list greeting recommend movie song joke birthday spell synonym fruit").split()
HARD = ("prove theorem derive optimize algorithm complexity integral recursion distributed consensus "
        "quantum entropy eigenvalue bayesian regression proof lemma compile concurrency cryptography "
        "multistep reasoning counterfactual legal contract analyze tradeoffs").split()
NEUTRAL = ("please explain write describe what how why the a about for me in of and with can you "
           "give short detailed answer question").split()

DEFAULT_MODELS = {  # name: (slope, offset, cost)
    "tiny": (2.2, 0.3, 0.1),
    "small": (1.6, 0.1, 0.3),
    "medium": (1.0, 0.0, 1.0),
    "large": (0.6, 0.0, 3.0),
}


def _queries(d: np.ndarray, rng, text_signal: float) -> list[str]:
    out = []
    for di in d:
        p_hard = np.clip(0.5 + text_signal * (di - 0.5), 0, 1)
        words = []
        for _ in range(rng.integers(6, 16)):
            if rng.random() < 0.4:
                words.append(rng.choice(NEUTRAL))
            else:
                words.append(rng.choice(HARD) if rng.random() < p_hard else rng.choice(EASY))
        out.append(" ".join(words))
    return out


def make_synthetic(n: int = 4000, k: int = 10, offset: float = 0.3, noise: float = 0.35,
                   a_small: float = 1.6, a_large: float = 0.6, base: float = -2.0,
                   text_signal: float = 1.0, seed: int = 0) -> RoutingDataset:
    rng = np.random.default_rng(seed)
    d = rng.beta(1.5, 1.5, size=n)
    queries = _queries(d, rng, text_signal)
    large = (base - a_large * d)[:, None] + noise * rng.standard_normal((n, k))
    small = (base - a_small * d - offset)[:, None] + noise * rng.standard_normal((n, k))
    return RoutingDataset(queries, small, large, "synthetic-small", "synthetic-large",
                          meta={"difficulty": d, "offset": offset})


def make_synthetic_benchmark(n: int = 4000, k: int = 10, models: dict | None = None, noise: float = 0.35,
                             base: float = -2.0, text_signal: float = 1.0, seed: int = 0,
                             splits: dict | None = None, name: str = "synthetic") -> Benchmark:
    models = models or DEFAULT_MODELS
    rng = np.random.default_rng(seed)
    d = rng.beta(1.5, 1.5, size=n)
    queries = _queries(d, rng, text_signal)
    names = list(models)
    scores = np.stack([(base - models[m][0] * d - models[m][1])[:, None] + noise * rng.standard_normal((n, k))
                       for m in names], axis=1)
    lengths = np.array([len(q.split()) for q in queries], dtype=float)
    costs = np.stack([models[m][2] * lengths / lengths.mean() for m in names], axis=1)  # cost grows with length
    return Benchmark(name, [f"syn-{i}" for i in range(n)], queries, names, scores, costs,
                     assign_splits(n, splits or {"train": 0.7, "val": 0.1, "test": 0.2}, seed),
                     sources=["synthetic"] * n, meta={"generator": "synthetic", "seed": seed})
