"""Small, strict file helpers for shared artifacts; no model dependencies."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                rows.append(row)
            except ValueError as e:
                raise ValueError(f"{path}:{n}: invalid JSONL: {e}") from e
    return rows


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def recover_tail(path):
    """Preserve an interrupted final write, then truncate to the last complete line."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, 2)
        if f.read(1) == b"\n":
            return
        end = f.tell()
        pos = end
        while pos:
            start = max(0, pos - 8192)
            f.seek(start)
            block = f.read(pos - start)
            ix = block.rfind(b"\n")
            if ix >= 0:
                pos = start + ix + 1
                break
            pos = start
        f.seek(pos)
        tail = f.read()
        try:
            json.loads(tail)
        except ValueError:
            with open(str(path) + ".interrupted-tail", "ab") as audit:
                audit.write(tail + b"\n")
            f.truncate(pos)
        else:
            f.seek(end)
            f.write(b"\n")


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as e:
        raise RuntimeError(f"writer lock exists: {path}; check the owning job before removing it") from e
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        os.close(fd)
        path.unlink()


def environment():
    packages = {}
    for name in ("numpy", "pyarrow", "torch", "transformers", "vllm", "sympy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    source_hash = digest({p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "llm_routing").rglob("*.py"))})
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "git_commit": result.stdout.strip(), "git_dirty": bool(status.stdout), "source_hash": source_hash}


def key(row):
    sid = row.get("sample_id")
    if type(sid) is not int or sid < 0:
        raise ValueError("sample_id must be a nonnegative integer")
    return row["query_id"], row["model"], sid


def successful(rows, identity="response_hash"):
    """Retry failures may repeat; distinct successful answers for one key are an error."""
    out = {}
    for r in rows:
        k = key(r)
        old = out.get(k)
        if old and old["success"]:
            if r["success"] and old[identity] != r[identity]:
                raise ValueError(f"conflicting successful records for {k}")
            continue
        out[k] = r
    return out


def seal(directory, extra):
    directory = Path(directory)
    outputs = {p.relative_to(directory).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
               for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "manifest.json"}
    write_json(directory / "manifest.json", {**extra, "outputs": outputs})


def verify(directory, required=()):
    directory = Path(directory).resolve()
    man = json.loads((directory / "manifest.json").read_text())
    outputs = man.get("outputs", {})
    if not outputs or not set(required) <= set(outputs):
        raise ValueError(f"{directory}: missing required manifest artifacts")
    if man.get("protocol") == "shared-v1":
        payload = set()
        for p in directory.rglob("*"):
            relative = p.relative_to(directory)
            if relative.parts[0] == ".cache":
                continue  # HF snapshot download bookkeeping, never part of the release
            if p.is_symlink():
                raise ValueError(f"{directory}: symlink is not an artifact: {relative}")
            if p.is_file() and relative.as_posix() != "manifest.json":
                payload.add(relative.as_posix())
        if payload != set(outputs):
            raise ValueError(f"{directory}: artifact inventory differs from manifest")
    for name, meta in outputs.items():
        p = (directory / name).resolve()
        if not p.is_relative_to(directory) or not p.is_file() or file_hash(p) != meta["sha256"]:
            raise ValueError(f"{directory}: artifact hash/path mismatch: {name}")
    return man
