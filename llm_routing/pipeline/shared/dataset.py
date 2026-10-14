"""Build queries ONCE, before any member generates answers. Never silently overwrite."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

from . import PROTOCOL
from .io import append, digest, file_hash, locked, read_jsonl, seal, verify, write_json


def model_pool(cfg):
    models = cfg["models"]
    names = [m["name"] for m in models]
    if not models or len(names) != len(set(names)):
        raise ValueError("model names must be nonempty and unique")
    from ...data.benchmark import safe_name
    if any(not isinstance(n, str) or not n.strip() or not safe_name(n).strip(".") for n in names):
        raise ValueError("invalid model name")
    if len({safe_name(n) for n in names}) != len(names):
        raise ValueError("model names collide after filename normalization")
    return {m["name"]: m for m in models}


def specification(cfg):
    return {k: cfg[k] for k in ("name", "version", "sources", "splits", "seed")}


def dataset_dir(cfg):
    return Path(cfg["path"]) / "dataset"


def load_dataset(cfg):
    path = dataset_dir(cfg)
    man = verify(path, ("queries.jsonl", "splits.json", "config.json"))
    if man["protocol"] != PROTOCOL or man["spec_hash"] != digest(specification(cfg)):
        raise ValueError("dataset configuration changed; create a new version/path")
    rows = read_jsonl(path / "queries.jsonl")
    if not rows or len({r["id"] for r in rows}) != len(rows) or digest(rows) != man["dataset_id"]:
        raise ValueError("invalid query IDs or dataset hash")
    return rows, man


def build_queries(cfg):
    from ..build import FORMATTERS
    model_pool(cfg)
    target = dataset_dir(cfg)
    if target.exists():
        rows, _ = load_dataset(cfg)
        print(f"[build-queries] frozen dataset reused: {len(rows)} queries")
        return
    fractions = cfg["splits"]
    if set(fractions) != {"train", "val", "test"} or any(v < 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("splits must contain train/val/test and sum to 1")
    rows, source_meta = [], []
    sources = cfg["sources"]
    if len({s["name"] for s in sources}) != len(sources):
        raise ValueError("source names must be unique")
    for src in sources:
        if "local_path" in src:
            data = read_jsonl(src["local_path"])
            source_meta.append({"name": src["name"], "sha256": file_hash(src["local_path"])})
        else:
            from datasets import load_dataset as hf_load
            if not src.get("revision") or src["revision"].startswith("PIN_"):
                raise ValueError(f"{src['name']}: pin a Hugging Face dataset revision")
            ds = hf_load(src["hf_id"], src.get("hf_config"), split=src.get("hf_split", "train"), revision=src["revision"])
            data = ds.shuffle(seed=cfg["seed"]).select(range(min(src["n"], len(ds))))
            source_meta.append({"name": src["name"], "revision": src["revision"], "fingerprint": ds._fingerprint})
        for ex in list(data)[:src["n"]]:
            if src.get("formatter"):
                query, reference = FORMATTERS[src["formatter"]](ex)
            else:
                query = src["query_template"].format_map(ex).strip()  # missing fields are errors
                reference = str(ex.get(src.get("reference_field", "reference"), ""))
            if not query:
                raise ValueError("empty query")
            original_id = ex.get(src.get("id_field", "id"))
            qid = src["name"] + "/" + digest({"id": original_id, "query": query})[:24]
            # Global normalized-text grouping prevents exact duplicates leaking even across sources.
            group = re.sub(r"\s+", " ", query.lower()).strip()
            u = int(digest([cfg["seed"], group])[:16], 16) / 2**64
            split = "train" if u < fractions["train"] else ("val" if u < fractions["train"] + fractions["val"] else "test")
            rows.append({"id": qid, "query": query, "reference": reference, "source": src["name"], "split": split,
                         "ground_truth_scorer": src.get("ground_truth_scorer"), "original_id": original_id})
    rows.sort(key=lambda r: r["id"])
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty dataset or duplicate query IDs")
    target.parent.mkdir(parents=True, exist_ok=True)
    with locked(target.parent / ".dataset.lock"):
        if target.exists():
            raise ValueError("another process already published this dataset")
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            tmp = Path(tmp)
            for row in rows:
                append(tmp / "queries.jsonl", row)
            write_json(tmp / "splits.json", {s: [r["id"] for r in rows if r["split"] == s] for s in fractions})
            write_json(tmp / "config.json", specification(cfg))
            seal(tmp, {"protocol": PROTOCOL, "dataset_id": digest(rows), "spec_hash": digest(specification(cfg)),
                       "sources": source_meta, "n_queries": len(rows)})
            tmp.rename(target)
    print(f"[build-queries] {len(rows)} frozen queries -> {target}")
