"""Frozen text embeddings for routers that represent a query as a vector.

GraphRouter (MiniLM, mean pooling), UniRoute (a 768-d encoder, CLS pooling), RouteLLM and
kNN routers all embed the query with a frozen pretrained encoder. This module only does
that one step; it does not choose an encoder for anyone. Each router passes the model,
revision, pooling and maximum length its paper uses, and records them in ``describe()``.

    X = encode(train.queries, "sentence-transformers/all-MiniLM-L6-v2", revision="...", pooling="mean")

Embeddings use only the query text, never scores, so caching them is safe for every split.
With ``cache_dir`` set, each call is stored once as ``<cache_dir>/<key>.npy`` where the key
hashes the encoder settings and the exact texts. Needs ``torch`` + ``transformers``
(``pip install -e '.[gpu]'``); runs on CPU for small inputs.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def pool(hidden: np.ndarray, mask: np.ndarray, pooling: str) -> np.ndarray:
    """(B, T, D) token states + (B, T) attention mask -> (B, D).

    mean: average over real tokens (padding excluded), as sentence-transformers does.
    cls:  the first token's state (BERT/BGE convention).
    """
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        m = mask[:, :, None].astype(hidden.dtype)
        return (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    raise ValueError(f"unknown pooling {pooling!r}; use 'mean' or 'cls'")


def cache_key(texts: list[str], **settings) -> str:
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for t in texts:
        h.update(hashlib.sha256(t.encode()).digest())
    return h.hexdigest()[:24]


def encode(texts: list[str], model_name: str, revision: str | None = None, pooling: str = "mean",
           normalize: bool = True, max_length: int = 512, batch_size: int = 64, device: str | None = None,
           cache_dir: str | None = None) -> np.ndarray:
    """(N, D) float32 embeddings of ``texts``; L2-normalised rows when ``normalize``."""
    settings = dict(model_name=model_name, revision=revision, pooling=pooling, normalize=normalize,
                    max_length=max_length)
    path = os.path.join(cache_dir, cache_key(texts, **settings) + ".npy") if cache_dir else None
    if path and os.path.exists(path):
        return np.load(path)
    out = _encode(texts, batch_size=batch_size, device=device, **settings)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, out)
    return out


def _encode(texts, model_name, revision, pooling, normalize, max_length, batch_size, device) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    chunks = []
    with torch.no_grad():
        for b in range(0, len(texts), batch_size):
            enc = tok(texts[b : b + batch_size], truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state.float().cpu().numpy()
            chunks.append(pool(hidden, enc["attention_mask"].cpu().numpy(), pooling))
    x = np.concatenate(chunks) if chunks else np.zeros((0, model.config.hidden_size))
    if normalize:
        x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return x.astype(np.float32)
