"""Independent member jobs: generation and two separately resumable offline scoring passes."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from . import PROTOCOL
from .backends import Generator
from .dataset import load_dataset, model_pool
from .io import append, digest, environment, key, locked, read_jsonl, recover_tail, successful, write_json


def generation_config(cfg, model):
    g = {"k": 1, "engine": "vllm", "temperature": 0.7, "top_p": 0.95, "max_new_tokens": 512,
         **cfg.get("generation", {}), **model.get("generation", {})}
    if set(g) - {"k", "engine", "temperature", "top_p", "max_new_tokens"}:
        raise ValueError(f"unknown generation parameters: {set(g) - {'k', 'engine', 'temperature', 'top_p', 'max_new_tokens'}}")
    if type(g["k"]) is not int or g["k"] < 1 or g["max_new_tokens"] < 1 or g["temperature"] < 0 or not 0 < g["top_p"] <= 1:
        raise ValueError("invalid generation parameters")
    return g


def job_spec(cfg, model_name, member, shard_index, num_shards):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", member) or member in (".", ".."):
        raise ValueError("member must be a simple name")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    rows, dataset = load_dataset(cfg)
    model = model_pool(cfg)[model_name]
    g = generation_config(cfg, model)
    spec = {"protocol": PROTOCOL, "dataset_id": dataset["dataset_id"], "model": model,
            "generation": g, "seed": cfg["seed"], "member": member, "shard_index": shard_index, "num_shards": num_shards}
    from ...data.benchmark import safe_name
    path = Path(cfg["path"]) / "contributions" / member / safe_name(model_name) / f"shard-{shard_index:05d}-of-{num_shards:05d}"
    selected = [r for r in rows if int(digest(r["id"])[:16], 16) % num_shards == shard_index]
    return path, selected, spec


def init_job(path, spec):
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "run.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["environment"].get("source_hash") != environment()["source_hash"]:
            raise ValueError("code changed since generation; use a new contribution path")
        if old["run_id"] != digest(spec):
            raise ValueError("resume configuration mismatch; use a new member/run path")
    else:
        write_json(manifest, {"run_id": digest(spec), "spec": spec, "environment": environment()})


def response_hash(row):
    return digest({k: v for k, v in row.items() if k not in ("response_hash", "attempt", "error")})


def _run_batch(generator, items):
    """[(query, seed)] -> [(text, metadata) or Exception]. Plain callables (test doubles) run one by one."""
    if hasattr(generator, "generate_batch"):
        return generator.generate_batch(items)
    out = []
    for query, seed in items:
        try:
            out.append(generator(query, seed))
        except Exception as e:
            out.append(e)
    return out


def generate(cfg, model, member="local", shard_index=0, num_shards=1, max_retries=2, retry_delay=1.0, backend=None,
             batch_size=256):
    """Generate every missing (query, sample) of this shard. ``batch_size`` requests go to vLLM at once;
    it changes throughput only (it is not part of the run identity), and each request keeps its own seed."""
    if max_retries < 0 or retry_delay < 0 or batch_size < 1:
        raise ValueError("invalid retry/batch settings")
    path, rows, spec = job_spec(cfg, model, member, shard_index, num_shards)
    g = spec["generation"]
    with locked(path / ".writer.lock"):
        init_job(path, spec)
        output = path / "responses.jsonl"
        recover_tail(output)
        output.touch(exist_ok=True)
        done = successful(read_jsonl(output))
        generator = backend or Generator(spec["model"], g)
        pending = []
        for r in rows:
            for sid in range(g["k"]):
                if (r["id"], model, sid) in done and done[(r["id"], model, sid)]["success"]:
                    continue
                pending.append((r, sid, int(digest([spec["seed"], r["id"], model, sid])[:8], 16) % (2**31)))
        errors = 0
        for start in range(0, len(pending), batch_size):
            todo = pending[start:start + batch_size]
            for attempt in range(max_retries + 1):
                results = _run_batch(generator, [(r["query"], seed) for r, _, seed in todo])
                failed = []
                for (r, sid, seed), result in zip(todo, results):
                    record = {"query_id": r["id"], "model": model, "sample_id": sid,
                              "dataset_id": spec["dataset_id"], "run_id": digest(spec), "prompt": r["query"],
                              "seed": seed, "generation": g, "attempt": attempt}
                    try:
                        if isinstance(result, Exception):
                            raise result
                        text, metadata = result
                        if not isinstance(text, str):
                            raise ValueError("backend response must be text")
                        record.update(response=text, metadata=metadata, success=True, error=None)
                    except Exception as e:
                        record.update(response=None, metadata={}, success=False, error=f"{type(e).__name__}: {e}")
                        failed.append((r, sid, seed))
                    record["response_hash"] = response_hash(record)
                    append(output, record)
                todo = failed
                if not todo:
                    break
                if attempt < max_retries:
                    time.sleep(min(retry_delay * 2**attempt, 30))
            errors += len(todo)
            print(f"[generate] {model}: {min(start + batch_size, len(pending))}/{len(pending)} samples", flush=True)
        if errors:
            raise RuntimeError(f"{errors} samples failed; rerun the same command to resume; details in {output}")
    return path


def answer_text(text, policy):
    if policy == "raw":
        return text
    if policy != "strip_think":
        raise ValueError("text_policy must be raw or strip_think")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"<think>.*$", "", text, flags=re.S).strip()


class ArmoRM:
    """Offline evaluator only. No labels, routing decisions, or generation calls."""
    def __init__(self, model_id="RLHFlow/ArmoRM-Llama3-8B-v0.1", revision=None, device_map="auto"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not revision or revision.startswith("PIN_"):
            raise ValueError("pin scoring.armorm.revision before an ArmoRM run")
        # The pinned ArmoRM custom model imports this documentation-only constant.
        # Recent transformers removed it; restoring an empty docstring does not alter forward().
        from transformers.models.llama import modeling_llama
        self.docstring_compat = not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING")
        if self.docstring_compat:
            modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device_map).eval()

    def __call__(self, prompt, response):
        import torch
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        tokens = self.tok.apply_chat_template(messages, return_tensors="pt", return_dict=False).to(self.model.device)
        limit = self.model.config.max_position_embeddings
        if tokens.shape[-1] > limit:
            raise ValueError(f"ArmoRM context overflow: {tokens.shape[-1]} > {limit}; no silent truncation")
        with torch.inference_mode():
            out = self.model(tokens)
        return float(out.score.item()), {"rewards": out.rewards.flatten().float().cpu().tolist(),
                                         "gating": out.gating_output.flatten().float().cpu().tolist(),
                                         "docstring_compat": self.docstring_compat,
                                         "chat_template_hash": digest(self.tok.chat_template), "tokens": tokens.shape[-1]}


def score(cfg, model, scorer="ground_truth", member="local", shard_index=0, num_shards=1, evaluator=None):
    if scorer not in ("ground_truth", "armorm"):
        raise ValueError("scorer must be ground_truth or armorm")
    path, queries, spec = job_spec(cfg, model, member, shard_index, num_shards)
    by_id = {r["id"]: r for r in queries}
    options = dict(cfg.get("scoring", {}).get(scorer, {}))
    policy = options.pop("text_policy", "strip_think")
    evaluator_spec = {"scorer": scorer, "options": options, "text_policy": policy, "protocol": PROTOCOL}
    evaluator_id = digest(evaluator_spec)
    scoring_environment = environment()
    with locked(path / ".writer.lock"):
        init_job(path, spec)
        output = path / f"{scorer}.jsonl"
        recover_tail(output)
        old = read_jsonl(output)
        if any(r["evaluator_id"] != evaluator_id for r in old):
            raise ValueError("scorer configuration changed; use a new contribution path")
        done = successful(old, "score_hash")
        if not (path / "responses.jsonl").exists():
            raise ValueError("generate this shard before scoring")
        responses = successful(read_jsonl(path / "responses.jsonl"))
        errors = 0
        for k, r in sorted(responses.items()):
            if not r["success"]:
                continue
            if r["run_id"] != digest(spec) or r["response_hash"] != response_hash(r) or r["query_id"] not in by_id:
                raise ValueError("response provenance mismatch")
            if k in done and done[k]["success"]:
                if done[k]["response_hash"] != r["response_hash"]:
                    raise ValueError("stale score bound to a different response")
                continue
            query = by_id[r["query_id"]]
            if scorer == "ground_truth" and not query.get("answer_type"):
                continue  # unavailable is null, never a fabricated correctness label
            record = {"query_id": k[0], "model": k[1], "sample_id": k[2], "response_hash": r["response_hash"],
                      "dataset_id": spec["dataset_id"], "evaluator_id": evaluator_id, "evaluator": evaluator_spec, "environment": scoring_environment}
            if scorer == "armorm" and evaluator is None:
                evaluator = ArmoRM(**options)
            try:
                text = answer_text(r["response"], policy)
                if scorer == "armorm":
                    value, details = evaluator(query["query"], text)
                else:
                    from ..ground_truth import score_query
                    ev = score_query(query, text, mbpp_timeout=int(options.get("mbpp_timeout", 5)))
                    value = float(ev.score)
                    details = {"answer_type": query["answer_type"], "mode": ev.mode,
                               "strict_correct": ev.strict_correct, "predicted": ev.predicted}
                    if not 0 <= value <= 1:
                        raise ValueError("ground truth must be in [0,1]; reference metrics need a separate channel")
                if not math.isfinite(value):
                    raise ValueError("nonfinite quality score")
                record.update(score=value, details=details, success=True, error=None)
            except Exception as e:
                record.update(score=None, details={}, success=False, error=f"{type(e).__name__}: {e}")
                errors += 1
            record["score_hash"] = digest(record)
            append(output, record)
        if errors:
            raise RuntimeError(f"{errors} scoring failures; resume the same scorer; inspect {output}")
    return path
