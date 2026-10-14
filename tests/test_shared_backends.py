"""Backend request/formatting tests with vLLM doubles; never load an actual LLM."""
import sys
from types import SimpleNamespace

import pytest

from llm_routing.pipeline.shared.backends import Generator


class Tokenizer:
    chat_template = "test-template"
    name_or_path = "fake/tokenizer"
    init_kwargs = {"_commit_hash": "tokenizer-sha"}
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<start>system/user<assistant>"
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens  # special tokens already in the rendered chat
        return [1, 2, 3]


def _fake_vllm(monkeypatch, calls, seen):
    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def get_tokenizer(self):
            return Tokenizer()
        def generate(self, prompts, params, use_tqdm):
            assert all(p == {"prompt_token_ids": [1, 2, 3]} for p in prompts) and len(prompts) == len(params)
            seen.append([p.seed for p in params])
            return [SimpleNamespace(outputs=[SimpleNamespace(text=f"answer-{p.seed}", token_ids=[4, 5], finish_reason="stop")])
                    for p in params]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: SimpleNamespace(**kw)))


def test_vllm_template_and_seed(monkeypatch):
    calls, seen = [], []
    _fake_vllm(monkeypatch, calls, seen)
    generator = Generator({"model_id": "fake/model", "revision": "fixed-sha", "chat_template_kwargs": {"enable_thinking": False},
                           "system_prompt": "system"},
                          {"engine": "vllm", "temperature": .7, "top_p": .95, "max_new_tokens": 16})
    text, metadata = generator("question", 77)
    assert text == "answer-77" and metadata["prompt_tokens"] == 3 and metadata["completion_tokens"] == 2
    assert metadata["formatted_prompt"] == "<start>system/user<assistant>"
    assert metadata["messages"][0]["role"] == "system" and metadata["seed_applied"]
    generator("question again", 77)
    assert len(calls) == 1 and calls[0]["generation_config"] == "vllm"
    assert "tensor_parallel_size" not in calls[0]


def test_vllm_batch_keeps_one_seed_per_request_and_sets_tensor_parallel(monkeypatch):
    calls, seen = [], []
    _fake_vllm(monkeypatch, calls, seen)
    generator = Generator({"model_id": "fake/model", "revision": "fixed-sha", "gpus": 2,
                           "chat_template_kwargs": {"enable_thinking": False}},
                          {"engine": "vllm", "temperature": .7, "top_p": .95, "max_new_tokens": 16})
    out = generator.generate_batch([("q1", 5), ("q2", 6), ("q3", 7)])
    assert seen == [[5, 6, 7]] and [t for t, _ in out] == ["answer-5", "answer-6", "answer-7"]
    assert all(m["batch_size"] == 3 and m["batch_seconds_per_sample"] >= 0 for _, m in out)
    assert calls[0]["tensor_parallel_size"] == 2


def test_unpinned_revision_rejected():
    with pytest.raises(ValueError, match="pin the model revision"):
        Generator({"model_id": "fake/model", "revision": "PIN_MODEL_COMMIT"}, {"engine": "vllm"})("q", 1)


@pytest.mark.parametrize("engine", ["typo", "api"])
def test_unknown_engine_rejected(engine):
    with pytest.raises(ValueError, match="unsupported engine"):
        Generator({}, {"engine": engine})
