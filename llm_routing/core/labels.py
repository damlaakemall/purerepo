"""What every router records about the training targets it built.

Paper-specific labels are built inside each router package (``routers/<paper>/labels.py``),
never written into the shared data. To keep them traceable, a router puts this manifest in
``describe()``, which the runner saves into ``metrics.json``:

    self.label_manifest_ = label_manifest(train, policy={"variant": "r_trans", "t": t_star, ...})
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data.benchmark import Benchmark


def label_manifest(bench: Benchmark, policy: dict, labels: np.ndarray | None = None, paper: str | None = None) -> dict:
    """Benchmark identity + the label policy (metric, sample rule, variant, hyperparameters).

    ``policy`` is everything that decides the labels, e.g. ``{"metric": "ground_truth",
    "samples": "same-index pairs", "tie": ">="}``; its hash tells two label sets apart.
    ``labels`` (optional) adds their count and a content hash.
    """
    out = {"benchmark": bench.name, "fingerprint": bench.fingerprint(), "split_rows": len(bench),
           "models": list(bench.models), "quality_metric": bench.meta.get("metric"), "paper": paper,
           "policy": policy,
           "policy_hash": hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode()).hexdigest()[:16]}
    if labels is not None:
        labels = np.asarray(labels)
        out["num_labels"] = int(labels.shape[0])
        out["labels_hash"] = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()[:16]
    return out
