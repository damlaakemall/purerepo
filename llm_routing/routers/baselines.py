"""Reference routers every method is compared against.

random_binary   send a random fraction of queries to the small model. Quality falls
                linearly with savings; a learned binary router must beat this line.
oracle          cheats by reading the realized test scores: best answer per query,
                cheapest among ties, with a cost knob. Upper bound for any router.
quality_regression
                simplest learned multi-model router: one TF-IDF ridge regressor per
                model predicts its quality, route by argmax(quality - lambda * cost).
                (The "predictive router" baseline of RouterBench.)
"""

from __future__ import annotations

import numpy as np

from ..core.base import BinaryRouter, Decision, MultiModelRouter, Router
from ..data.benchmark import Benchmark
from ..registry import register_router


@register_router("random_binary")
class RandomBinaryRouter(BinaryRouter):
    def __init__(self, small: str, large: str, seed: int = 0, **_):
        super().__init__(small, large)
        self.name = "random"
        self.seed = seed

    def score(self, queries):
        return np.random.default_rng(self.seed).random(len(queries))


@register_router("oracle")
class OracleRouter(Router):
    """Reads test scores — for reference only. Knob: lambda in argmax(q - lambda * cost)."""

    family = "reference"

    def __init__(self, models: list[str] | None = None, **_):
        self.models, self.name = models, "oracle"

    def sweep(self, bench: Benchmark, num: int = 51):
        cols = np.array([bench.model_index(m) for m in (self.models or bench.models)])
        q = np.nan_to_num(bench.realized()[:, cols], nan=-1e9)
        c = bench.costs[:, cols]
        cn = c / max(c.max(), 1e-12)
        qr = max(float(q.max() - q.min()), 1e-9)
        rows = np.arange(len(bench))
        out = []
        for lam in (np.concatenate([[0.0], np.geomspace(1e-4, 1e3, num - 1)]) * qr)[::-1]:
            pick = cols[np.argmax(q - lam * cn, axis=1)]
            out.append(Decision(float(lam), pick, bench.costs[rows, pick]))
        return out


@register_router("quality_regression")
class QualityRegressionRouter(MultiModelRouter):
    def __init__(self, models: list[str] | None = None, alpha: float = 1.0, max_features: int = 50_000, **_):
        super().__init__(models)
        self.name = "quality_regression"
        self.alpha, self.max_features = alpha, max_features

    def fit(self, train, val=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge

        super().fit(train, val)
        self.vec = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), sublinear_tf=True)
        X = self.vec.fit_transform(train.queries)
        y = np.nanmean(train.select_models(self.models).scores, axis=2)  # mean over k samples
        self.regs = []
        for j in range(y.shape[1]):
            ok = ~np.isnan(y[:, j])
            self.regs.append(Ridge(alpha=self.alpha).fit(X[ok], y[ok, j]))
        return self

    def predict_quality(self, queries):
        X = self.vec.transform(queries)
        return np.stack([r.predict(X) for r in self.regs], axis=1)
