"""Who goes in train / val / test: stratified by source, near-duplicates kept together.

Follows SU Lab LLM-Routing-Project (data_generator/shared_protocol/splits.py):

1. Near-duplicate groups. Two prompts whose embeddings have cosine >= threshold (default 0.99,
   all-MiniLM-L6-v2) are linked, and linked prompts form one group (union-find). ``method: exact``
   only links identical normalised text and needs no model.
2. Stratified split. Each source is split on its own: its groups are shuffled with
   ``seed ^ crc32(source)`` and dealt out until train holds floor(n * train) prompts and val
   floor(n * (train + val)) - that; the rest is test. Every source therefore gets the same
   60/10/30 (up to group sizes), and a whole group always lands in one split, so no router can
   learn a test prompt from its twin in train.
3. Sources in ``ood_datasets`` go entirely to test.

As in the reference, groups are formed within a source; exact duplicates across sources never reach
this point because build-queries keeps only the first copy of a text.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def _union_find(n: int, pairs) -> list[int]:
    """Connected components: label[i] is the smallest index linked to i."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


def embed_minilm(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", revision=None, batch_size=64):
    """Mean-pooled, L2-normalised sentence embeddings (the reference's grouping embedder)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).eval()
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).float().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)


def near_duplicate_groups(ids, texts, method="embedding", threshold=0.99, embed=None, block=512, **embed_options):
    """prompt id -> group number. ``embed(texts) -> [N, D]`` unit vectors can be injected (tests)."""
    if method == "exact":
        seen = {}
        return {pid: seen.setdefault(normalize_text(t), len(seen)) for pid, t in zip(ids, texts)}
    if method != "embedding":
        raise ValueError(f"unknown near-duplicate method {method!r}; use exact or embedding")
    emb = np.asarray((embed or embed_minilm)(list(texts), **embed_options), dtype=np.float32)
    pairs = []
    for start in range(0, len(ids), block):  # blocks keep the similarity matrix small
        rows, cols = np.where(emb[start:start + block] @ emb.T >= threshold)
        pairs += [(start + int(r), int(c)) for r, c in zip(rows, cols) if start + int(r) < int(c)]
    labels = _union_find(len(ids), pairs)
    renumber = {}
    return {pid: renumber.setdefault(label, len(renumber)) for pid, label in zip(ids, labels)}


def stratified_splits(sources, fractions, seed, groups=None, ood_datasets=()):
    """sources: prompt id -> source. Returns prompt id -> "train" | "val" | "test"."""
    groups = groups or {pid: pid for pid in sources}
    split_of = {}
    for source in sorted(set(sources.values())):
        ids = sorted(pid for pid, s in sources.items() if s == source)  # independent of input order
        if source in ood_datasets:
            split_of.update({pid: "test" for pid in ids})
            continue
        units = {}  # group -> its prompts in this source
        for pid in ids:
            units.setdefault(str(groups[pid]), []).append(pid)
        order = [units[g] for g in sorted(units)]
        rng = np.random.default_rng(seed ^ zlib.crc32(source.encode("utf-8")))
        n_train = int(np.floor(len(ids) * fractions["train"]))
        n_train_val = int(np.floor(len(ids) * (fractions["train"] + fractions["val"])))
        placed = 0
        for i in rng.permutation(len(order)):
            name = "train" if placed < n_train else ("val" if placed < n_train_val else "test")
            split_of.update({pid: name for pid in order[i]})
            placed += len(order[i])
    return split_of


def group_stats(groups):
    """How much of the prompt set is near-duplicated (recorded in the dataset manifest)."""
    sizes = {}
    for g in groups.values():
        sizes[g] = sizes.get(g, 0) + 1
    multi = [n for n in sizes.values() if n > 1]
    return {"n_prompts": len(groups), "n_groups": len(sizes), "n_groups_with_duplicates": len(multi),
            "n_prompts_in_duplicate_groups": sum(multi), "largest_group": max(sizes.values(), default=0)}
