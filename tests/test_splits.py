"""Splits as in the reference repo: stratified per source, near-duplicate groups never split up."""
import numpy as np

from llm_routing.collection.splits import near_duplicate_groups, stratified_splits

FRACTIONS = {"train": 0.6, "val": 0.1, "test": 0.3}


def test_every_source_gets_the_same_ratios():
    sources = {f"a/{i}": "a" for i in range(100)} | {f"b/{i}": "b" for i in range(37)}
    split_of = stratified_splits(sources, FRACTIONS, seed=42)
    for source, n in (("a", 100), ("b", 37)):
        counts = {s: sum(1 for p, x in split_of.items() if sources[p] == source and x == s) for s in FRACTIONS}
        n_train, n_val = int(np.floor(n * 0.6)), int(np.floor(n * 0.7)) - int(np.floor(n * 0.6))
        assert counts == {"train": n_train, "val": n_val, "test": n - n_train - n_val}


def test_split_does_not_depend_on_input_order():
    sources = {f"a/{i}": "a" for i in range(50)}
    backwards = dict(reversed(list(sources.items())))
    assert stratified_splits(sources, FRACTIONS, 7) == stratified_splits(backwards, FRACTIONS, 7)


def test_near_duplicate_groups_stay_in_one_split():
    texts = {f"a/{i}": f"question number {i}" for i in range(60)}
    texts["a/twin"] = "What is 17 times 19?"
    texts["a/twin2"] = "Compute 17 times 19."
    ids = list(texts)

    def embed(batch, **_):  # paraphrases share a direction; everything else is orthogonal
        vecs = np.eye(len(batch), dtype=np.float32)
        twin = [i for i, t in enumerate(batch) if "times" in t]
        vecs[twin] = vecs[twin[0]]
        return vecs

    groups = near_duplicate_groups(ids, [texts[i] for i in ids], method="embedding", threshold=0.99, embed=embed)
    assert groups["a/twin"] == groups["a/twin2"] and len(set(groups.values())) == 61
    for seed in range(20):
        split_of = stratified_splits({i: "a" for i in ids}, FRACTIONS, seed, groups)
        assert split_of["a/twin"] == split_of["a/twin2"]


def test_exact_grouping_and_ood_sources():
    groups = near_duplicate_groups(["x", "y", "z"], ["Hello  World", "hello world", "other"], method="exact")
    assert groups["x"] == groups["y"] != groups["z"]
    split_of = stratified_splits({"x": "in", "y": "in", "z": "held_out"}, FRACTIONS, 0, groups, ood_datasets=["held_out"])
    assert split_of["z"] == "test" and split_of["x"] == split_of["y"]
