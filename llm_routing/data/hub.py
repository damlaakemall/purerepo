"""Share validated artifacts. Shared releases should be downloaded at a fixed Hub commit."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _validate(path):
    path = Path(path)
    man = json.loads((path / "manifest.json").read_text())
    if man.get("protocol") == "shared-v1":
        if (path / "long.parquet").exists():
            from ..pipeline.shared.assemble import validate_release
            validate_release(path)
        else:
            from ..pipeline.shared.io import verify
            verify(path, ("run.json", "responses.jsonl"))
    else:
        from .benchmark import Benchmark
        Benchmark.load(str(path))


def push(path: str, repo_id: str, private: bool = True, message: str = "publish benchmark"):
    from huggingface_hub import HfApi
    if not repo_id:
        raise ValueError("set hub_repo to a team dataset repository")
    _validate(path)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(folder_path=path, repo_id=repo_id, repo_type="dataset", commit_message=message,
                               ignore_patterns=[".cache/**", "*.lock"])
    print(f"[hub] pushed {path} -> https://huggingface.co/datasets/{repo_id}; pin hub_revision={commit.oid}")
    return commit.oid


def pull(repo_id: str, path: str, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download
    target = Path(path)
    if target.exists():
        raise ValueError(f"refusing to overwrite existing data: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        tmp = Path(tmp)
        snapshot_download(repo_id, repo_type="dataset", local_dir=tmp, revision=revision)
        _validate(tmp)
        tmp.rename(target)
    print(f"[hub] pulled verified artifact {repo_id}@{revision} -> {path}")
