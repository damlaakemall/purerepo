"""Backend request/formatting tests with SDK doubles; never load an actual LLM."""
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


def test_vllm_template_and_seed(monkeypatch):
    calls = []
    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def get_tokenizer(self):
            return Tokenizer()
        def generate(self, prompts, params, use_tqdm):
            assert prompts == [{"prompt_token_ids": [1, 2, 3]}]
            assert params.seed == 77 and params.n == 1
            return [SimpleNamespace(outputs=[SimpleNamespace(text="answer", token_ids=[4, 5], finish_reason="stop")])]
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    generator = Generator({"model_id": "fake/model", "revision": "fixed-sha", "chat_template_kwargs": {"enable_thinking": False},
                           "system_prompt": "system"},
                          {"engine": "vllm", "temperature": .7, "top_p": .95, "max_new_tokens": 16})
    text, metadata = generator("question", 77)
    assert text == "answer" and metadata["prompt_tokens"] == 3 and metadata["completion_tokens"] == 2
    assert metadata["formatted_prompt"] == "<start>system/user<assistant>"
    assert metadata["messages"][0]["role"] == "system" and metadata["seed_applied"]
    generator("question again", 77)
    assert len(calls) == 1 and calls[0]["generation_config"] == "vllm"


@pytest.mark.parametrize("supports_seed", [False, True])
def test_api_usage_seed_and_no_credentials_in_metadata(monkeypatch, supports_seed):
    monkeypatch.setenv("TEAM_KEY", "secret-test-value")
    monkeypatch.setenv("TEAM_URL", "https://inference.example/v1")
    requests = []
    def create(**request):
        requests.append(request)
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=11, model_dump=lambda: {"prompt_tokens": 7, "completion_tokens": 11})
        return SimpleNamespace(model="served-fixed-model", usage=usage, system_fingerprint="server-v1",
                               choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")])
    def client(**kw):
        assert kw["api_key"] == "secret-test-value" and kw["max_retries"] == 0
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client))
    g = Generator({"model_id": "requested-model", "system_prompt": "be concise", "api": {
        "key_env": "TEAM_KEY", "base_url_env": "TEAM_URL", "supports_seed": supports_seed,
        "max_tokens_field": "max_completion_tokens"}},
        {"engine": "api", "temperature": .7, "top_p": .95, "max_new_tokens": 12})
    text, meta = g("question", 44)
    assert text == "answer" and meta["completion_tokens"] == 11
    assert ("seed" in requests[0]) == supports_seed and meta["seed_applied"] == supports_seed
    assert "max_completion_tokens" in requests[0] and "secret-test-value" not in str(meta)
    assert requests[0]["messages"][0] == {"role": "system", "content": "be concise"}


def test_unknown_engine_rejected():
    with pytest.raises(ValueError, match="unsupported engine"):
        Generator({}, {"engine": "typo"})
