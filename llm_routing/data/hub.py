"""Share frozen benchmarks through the Hugging Face Hub so every machine uses identical data.

    llmr push-data configs/benchmarks/ours_v1.yaml     # after `assemble`, from the cluster
    llmr pull-data configs/benchmarks/ours_v1.yaml     # anywhere else

Needs `huggingface-cli login` (token with write access) once per machine, and
``hub_repo: <user-or-org>/<name>`` in the benchmark config. Repos are created private.
"""

from __future__ import annotations


def push(path: str, repo_id: str, private: bool = True, message: str = "update benchmark") -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=path, repo_id=repo_id, repo_type="dataset", commit_message=message)
    print(f"[hub] pushed {path} -> https://huggingface.co/datasets/{repo_id}")


def pull(repo_id: str, path: str, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=path, revision=revision)
    print(f"[hub] pulled {repo_id} -> {path}")
