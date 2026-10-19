"""The shared benchmark format. EVERY router is trained and evaluated on one of these.

Fair comparison rules this format enforces:

* frozen splits — each query carries its split ("train" / "val" / "test") on disk,
  so every router sees exactly the same train data and is scored on the same test data;
* frozen scores — response quality is computed once, at build time, and stored;
  routers never re-score anything;
* a fingerprint (hash of ids, models, scores, costs) is written into every run's
  results, so two numbers are only compared if they came from the same data.

On disk (one directory per benchmark):

    manifest.json        name, models, k, metric, fingerprint, meta
    queries.jsonl        {"id", "query", "source", "split", "reference"?} per line
    scores.npy           float32 (N, M, k)  quality of k sampled responses, NaN = missing
    costs.npy            float32 (N, M)     cost of answering query n with model m
    responses/<m>.jsonl  optional response texts (needed by cascade routers)
    extras/<name>.npy    optional per-sample arrays (N, M, k), NaN = missing. The shared-v1 release writes
                         sample_cost        realized cost of every sample (best-of-n pays for n samples)
                         armorm / ground_truth   the OTHER quality channel (e.g. a best-of-n selector)
                         completion_tokens  output length per sample
                         seconds            GPU wall-clock share per sample (latency-aware routers)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from ..core.types import RoutingDataset

SPLITS = ("train", "val", "test")


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model)


@dataclass
class Benchmark:
    name: str
    query_ids: list[str]
    queries: list[str]
    models: list[str]
    scores: np.ndarray  # (N, M, k)
    costs: np.ndarray  # (N, M)
    split_of: np.ndarray  # (N,) array of "train"/"val"/"test"
    sources: list[str] | None = None  # task / dataset each query came from
    references: list[str] | None = None
    meta: dict = field(default_factory=dict)
    responses: dict[str, list[list[str]]] | None = None  # model -> N x k texts
    extras: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (N, M, k) per-sample arrays

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float32)
        if self.scores.ndim == 2:
            self.scores = self.scores[:, :, None]
        self.costs = np.asarray(self.costs, dtype=np.float32)
        if self.costs.ndim == 1:  # one cost per model -> same for every query
            self.costs = np.broadcast_to(self.costs, (len(self.queries), len(self.models))).copy()
        self.split_of = np.asarray(self.split_of)
        n, m = len(self.queries), len(self.models)
        if self.scores.ndim != 3 or self.scores.shape[:2] != (n, m) or self.scores.shape[2] < 1:
            raise ValueError("scores must have shape (queries, models, samples)")
        if self.costs.shape != (n, m) or len(self.query_ids) != n or len(self.split_of) != n:
            raise ValueError("benchmark dimensions disagree")
        if len(set(self.query_ids)) != n or len(set(self.models)) != m:
            raise ValueError("query IDs and model names must be unique")
        if not set(self.split_of) <= set(SPLITS):
            raise ValueError("unknown split name")
        if np.isinf(self.scores).any() or np.isinf(self.costs).any() or (self.costs < 0).any():
            raise ValueError("scores/costs cannot be infinite; costs cannot be negative")
        self.extras = {k: np.asarray(v, dtype=np.float32) for k, v in (self.extras or {}).items()}
        for name, value in self.extras.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", name) or value.shape != (n, m, self.scores.shape[2]):
                raise ValueError(f"extra {name!r} must be a simple name with shape (queries, models, samples)")

    # ---- shape ---------------------------------------------------------------
    def __len__(self):
        return len(self.queries)

    @property
    def num_models(self):
        return len(self.models)

    @property
    def k(self):
        return self.scores.shape[2]

    def model_index(self, name: str) -> int:
        return self.models.index(name)

    def realized(self) -> np.ndarray:
        """(N, M) quality of the response a user would actually get (sample 0)."""
        return self.scores[:, :, 0]

    # ---- views ---------------------------------------------------------------
    def subset(self, idx) -> "Benchmark":
        idx = np.asarray(idx)
        pick = lambda xs: None if xs is None else [xs[i] for i in idx]  # noqa: E731
        return Benchmark(
            self.name, pick(self.query_ids), pick(self.queries), list(self.models), self.scores[idx],
            self.costs[idx], self.split_of[idx], pick(self.sources), pick(self.references), dict(self.meta),
            None if self.responses is None else {m: pick(r) for m, r in self.responses.items()},
            {k: v[idx] for k, v in self.extras.items()},
        )

    def split(self, name: str) -> "Benchmark":
        return self.subset(np.flatnonzero(self.split_of == name))

    def select_models(self, names: list[str]) -> "Benchmark":
        j = [self.model_index(n) for n in names]
        b = Benchmark(self.name, self.query_ids, self.queries, list(names), self.scores[:, j], self.costs[:, j],
                      self.split_of, self.sources, self.references, dict(self.meta),
                      None if self.responses is None else {m: self.responses[m] for m in names if m in self.responses},
                      {k: v[:, j] for k, v in self.extras.items()})
        return b

    def pair(self, small: str, large: str) -> RoutingDataset:
        """Binary view used by small/large routers (e.g. Hybrid LLM)."""
        i, j = self.model_index(small), self.model_index(large)
        return RoutingDataset(self.queries, self.scores[:, i], self.scores[:, j], small, large,
                              meta={"benchmark": self.name})

    # ---- integrity -----------------------------------------------------------
    def _legacy_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update("\n".join(self.query_ids).encode())
        h.update("\n".join(self.models).encode())
        h.update("\n".join(map(str, self.split_of)).encode())
        h.update(np.ascontiguousarray(self.scores).tobytes())
        h.update(np.ascontiguousarray(self.costs).tobytes())
        return h.hexdigest()[:16]

    def fingerprint(self) -> str:
        h = hashlib.sha256(self._legacy_fingerprint().encode())
        h.update(json.dumps({"queries": self.queries, "sources": self.sources,
                             "references": self.references,
                             "protocol": {k: self.meta.get(k) for k in
                                          ("metric", "cost_unit", "cost_definition", "version", "dataset_id")}},
                            sort_keys=True, ensure_ascii=False).encode())
        for name in sorted(self.extras):  # absent extras leave older fingerprints unchanged
            h.update(name.encode())
            h.update(np.ascontiguousarray(self.extras[name]).tobytes())
        return h.hexdigest()[:16]

    def summary(self) -> dict:
        counts = {s: int((self.split_of == s).sum()) for s in SPLITS}
        q = np.nanmean(self.realized(), axis=0)
        c = self.costs.mean(axis=0)
        return {"name": self.name, "num_queries": len(self), "splits": counts, "k": self.k,
                "models": {m: {"mean_quality": float(q[i]), "mean_cost": float(c[i])} for i, m in enumerate(self.models)},
                "fingerprint": self.fingerprint()}

    # ---- io ------------------------------------------------------------------
    def save(self, path: str, with_responses: bool = True) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "scores.npy"), self.scores)
        np.save(os.path.join(path, "costs.npy"), self.costs)
        with open(os.path.join(path, "queries.jsonl"), "w") as f:
            for i in range(len(self)):
                row = {"id": self.query_ids[i], "query": self.queries[i], "split": str(self.split_of[i])}
                if self.sources is not None:
                    row["source"] = self.sources[i]
                if self.references is not None:
                    row["reference"] = self.references[i]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if with_responses and self.responses:
            os.makedirs(os.path.join(path, "responses"), exist_ok=True)
            for m, resp in self.responses.items():
                with open(os.path.join(path, "responses", safe_name(m) + ".jsonl"), "w") as f:
                    for qid, r in zip(self.query_ids, resp):
                        f.write(json.dumps({"id": qid, "responses": r}, ensure_ascii=False) + "\n")
        if self.extras:
            os.makedirs(os.path.join(path, "extras"), exist_ok=True)
            for name, value in self.extras.items():
                np.save(os.path.join(path, "extras", name + ".npy"), value)
        from ..collection.io import file_hash
        from pathlib import Path
        artifacts = [Path(path) / n for n in ("queries.jsonl", "scores.npy", "costs.npy")]
        artifacts += [Path(path) / "extras" / (name + ".npy") for name in self.extras]
        if with_responses and self.responses:
            artifacts += [Path(path) / "responses" / (safe_name(m) + ".jsonl") for m in self.responses]
        outputs = {p.relative_to(path).as_posix(): {"sha256": file_hash(p)} for p in artifacts}
        manifest = {"fingerprint_version": 2, "outputs": outputs, "name": self.name, "models": self.models, "k": self.k, "num_queries": len(self),
                    "extras": sorted(self.extras),
                    "fingerprint": self.fingerprint(), "meta": self.meta}
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str, with_responses: bool = False) -> "Benchmark":
        with open(os.path.join(path, "manifest.json")) as f:
            man = json.load(f)
        extra_names = man.get("extras", [])
        if man.get("outputs"):
            from ..collection.io import verify
            verify(path, ("queries.jsonl", "scores.npy", "costs.npy", *(f"extras/{n}.npy" for n in extra_names)))
        ids, qs, splits, srcs, refs = [], [], [], [], []
        with open(os.path.join(path, "queries.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                ids.append(r["id"]); qs.append(r["query"]); splits.append(r["split"])
                srcs.append(r.get("source")); refs.append(r.get("reference"))
        responses = None
        rdir = os.path.join(path, "responses")
        if with_responses and os.path.isdir(rdir):
            responses = {}
            for m in man["models"]:
                p = os.path.join(rdir, safe_name(m) + ".jsonl")
                if os.path.exists(p):
                    by_id = {}
                    with open(p) as f:
                        for line in f:
                            r = json.loads(line)
                            by_id[r["id"]] = r["responses"]
                    responses[m] = [by_id.get(i, []) for i in ids]
        b = cls(man["name"], ids, qs, man["models"], np.load(os.path.join(path, "scores.npy")),
                np.load(os.path.join(path, "costs.npy")), np.array(splits),
                srcs if any(s is not None for s in srcs) else None,
                refs if any(r is not None for r in refs) else None, man.get("meta", {}), responses,
                {n: np.load(os.path.join(path, "extras", n + ".npy")) for n in extra_names})
        expected = b.fingerprint() if man.get("fingerprint_version") == 2 else b._legacy_fingerprint()
        if man.get("fingerprint") and man["fingerprint"] != expected:
            raise ValueError(f"benchmark at {path} does not match its manifest fingerprint — data was modified")
        return b


def assign_splits(n: int, fractions: dict, seed: int) -> np.ndarray:
    """Deterministic random split assignment."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    out = np.empty(n, dtype=object)
    start = 0
    names = list(fractions)
    for i, s in enumerate(names):
        end = n if i == len(names) - 1 else start + int(round(fractions[s] * n))
        out[perm[start:end]] = s
        start = end
    return out.astype(str)
