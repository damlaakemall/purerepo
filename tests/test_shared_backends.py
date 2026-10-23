"""Backend request/formatting tests with vLLM doubles; never load an actual LLM."""
import sys
from types import SimpleNamespace

import pytest

from llm_routing.collection.backends import Generator


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


def test_template_opened_think_block_is_put_back(monkeypatch):
    # DeepSeek-R1-Distill's template ends the prompt with "<think>\n"; the model's own text starts mid-trace.
    calls, seen = [], []
    _fake_vllm(monkeypatch, calls, seen)
    monkeypatch.setattr(Tokenizer, "apply_chat_template", lambda self, messages, **kw: "<user>q<assistant><think>\n")
    generator = Generator({"model_id": "fake/model", "revision": "fixed-sha"},
                          {"engine": "vllm", "temperature": .7, "top_p": .95, "max_new_tokens": 16})
    text, metadata = generator("q", 3)
    assert text == "<think>\nanswer-3" and metadata["response_prefix"] == "<think>\n"


def test_strip_think_matches_reference():
    from llm_routing.collection.collect import answer_text
    assert answer_text("<think>\nmaybe (D)</think>\nAnswer: (B)", "strip_think") == "Answer: (B)"
    assert answer_text("<Thinking>x</Thinking> ok", "strip_think") == "ok"
    assert answer_text("<think>\ncut off mid-thought, Answer: (D)", "strip_think") == ""  # truncated: no answer
    assert answer_text("<think>\n\n</think>\nplain", "strip_think") == "plain"           # SmolLM3 no-think
    assert answer_text("<think>x</think>y", "raw") == "<think>x</think>y"


@pytest.mark.parametrize("multimodal", [False, True])
def test_hf_loader_selects_checkpoint_class_and_removes_vllm_options(monkeypatch, multimodal):
    calls = []
    def loader(kind):
        def load(model_id, **kwargs):
            calls.append((kind, model_id, kwargs))
            return SimpleNamespace(eval=lambda: "loaded")
        return SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        AutoModelForCausalLM=loader("causal"), AutoModelForImageTextToText=loader("conditional")))
    g = Generator({"model_id": "fake/model", "revision": "pinned",
                   "hf_loader": "multimodal_lm" if multimodal else "causal_lm",
                   "engine_kwargs": {"max_model_len": 16384}}, {"engine": "hf"})
    g._load()
    assert calls == [("conditional" if multimodal else "causal", "fake/model",
                      {"revision": "pinned", "device_map": "auto"})]


def test_native_mistral_prompt_keeps_native_chat_ids():
    from llm_routing.collection.io import digest
    class Native:
        IS_MISTRAL_TOKENIZER = True
        instruct = object()
        version = 7
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return [1, 42, 3]
        def decode(self, ids, skip_special_tokens):
            assert ids == [1, 42, 3] and not skip_special_tokens
            return "<s>[INST]question[/INST]"
    gen = Generator({"model_id": "native/model", "revision": "pinned"}, {})
    gen.model_id = "native/model"
    gen.tok = Native()
    ids, meta = gen._prompt("question", 5)
    assert ids == [1, 42, 3]
    assert meta["prompt_tokens"] == 3 and meta["prompt_token_ids_hash"] == digest(ids)
    assert meta["tokenizer_revision"] == "pinned" and meta["seed_applied"]


def test_output_budget_reserves_prompt_without_truncating_it(monkeypatch):
    calls, seen = [], []
    _fake_vllm(monkeypatch, calls, seen)
    generator = Generator({"model_id": "fake/model", "revision": "pinned",
                           "chat_template_kwargs": {"enable_thinking": False},
                           "engine_kwargs": {"max_model_len": 5}},
                          {"engine": "vllm", "temperature": .7, "top_p": 1., "max_new_tokens": 4,
                           "context_policy": "cap_output"})
    # Three prompt tokens leave room for two output tokens, without editing the prompt.
    _, meta = generator("question", 1)
    assert meta["prompt_tokens"] == 3 and meta["effective_max_new_tokens"] == 2
    assert meta["context_limited"] and meta["completion_tokens"] == 2
    generator.model["engine_kwargs"]["max_model_len"] = 3
    with pytest.raises(ValueError, match="no output room"):
        generator("question", 1)


@pytest.mark.parametrize("ids,text,prefix,reasoning,answer", [
    ([10, 3, 4, 11, 5], "<think>work</think>answer", "", 4, 1),
    ([3, 11, 5], "work</think>answer", "<think>\n", 2, 1),
    ([10, 3, 4], "<think>unfinished", "", 3, 0),
    ([3, 4], "ordinary answer", "", 0, 2),
])
def test_reasoning_counts_partition_actual_generated_ids(ids, text, prefix, reasoning, answer):
    from llm_routing.collection.backends import reasoning_token_counts
    class Tags:
        def encode(self, text, add_special_tokens):
            return {"<think>": [10], "</think>": [11], "<thinking>": [12],
                    "</thinking>": [13], "<reasoning>": [14], "</reasoning>": [15]}[text]
    meta = reasoning_token_counts(ids, Tags(), text, prefix)
    assert meta["reasoning_tokens"] == reasoning and meta["answer_tokens"] == answer
    assert reasoning + answer == len(ids)


def test_unresolvable_reasoning_tags_do_not_invent_token_counts():
    from llm_routing.collection.backends import reasoning_token_counts
    class Tags:
        def encode(self, text, add_special_tokens):
            return [100] if not text.startswith("</") else [101]
    meta = reasoning_token_counts([2, 3], Tags(), "text</think>answer", "<think>")
    assert meta["reasoning_tokens"] is None and meta["answer_tokens"] is None
    assert meta["token_partition"] == "unavailable"


class RewardTokenizer:
    chat_template = 'reward-template'
    def apply_chat_template(self, messages, tokenize):
        assert tokenize
        # Deliberately differs from the producer tokenizer: characters, not words.
        return list(range(4 + len(messages[0]['content']) + len(messages[1]['content'])))


def test_reward_budget_counts_whole_chat_and_refuses_overruns():
    from llm_routing.collection.armorm_budget import ArmoRMBudget
    budget = ArmoRMBudget({'model_id': 'reward', 'revision': 'pinned', 'limit': 20,
                          'reserve_tokens': 8}, RewardTokenizer())
    assert budget.check_prompt('abc') == 7
    assert not budget.stop('abc', '1234')
    assert budget.stop('abc', '12345')
    assert budget.metadata('abc', '12345')['armorm_generation_budget']['tokens'] == 12
    with pytest.raises(ValueError, match='no safe ArmoRM output room'):
        budget.check_prompt('x' * 8)
    with pytest.raises(ValueError, match='guard overrun'):
        budget.stop('abc', 'x' * 14)
    with pytest.raises(ValueError, match='final answer exceeds'):
        budget.metadata('abc', 'x' * 14)


def test_reward_budget_stops_vllm_without_trimming_or_replacing_tokens():
    from llm_routing.collection.armorm_budget import ArmoRMBudget
    class Engine:
        def __init__(self):
            self.active = False; self.step_number = 0; self.aborts = []
        def add_request(self, request_id, prompt, sampling):
            assert prompt == {'prompt_token_ids': [1, 2]}
            assert sampling.seed == 77
            self.active = True
        def has_unfinished_requests(self):
            return self.active
        def step(self):
            self.step_number += 1
            return [SimpleNamespace(request_id='armorm-0', finished=False, outputs=[
                SimpleNamespace(text='a' * self.step_number, token_ids=list(range(self.step_number)), finish_reason=None)])]
        def abort_request(self, request_ids):
            self.aborts += request_ids; self.active = False
    generator = Generator({'model_id': 'fake', 'revision': 'pinned'}, {})
    generator.armorm_budget = ArmoRMBudget({'model_id': 'reward', 'revision': 'pinned',
                                           'limit': 20, 'reserve_tokens': 8}, RewardTokenizer())
    engine = Engine(); generator.llm = SimpleNamespace(llm_engine=engine)
    outputs = generator._budgeted_vllm([(0, ([1, 2], {'budget_query': 'abc', 'response_prefix': ''}))],
                                       [SimpleNamespace(seed=77)])
    sample = outputs[0].outputs[0]
    assert engine.step_number == 5 and sample.text == 'aaaaa'
    assert sample.token_ids == [0, 1, 2, 3, 4] and sample.finish_reason == 'armorm_budget'
    assert generator.armorm_budget.count('abc', sample.text) == 12 < 20


def test_reward_budget_step_guard_requires_room_for_producer_tokens():
    from llm_routing.collection.armorm_budget import ArmoRMBudget
    budget = ArmoRMBudget({'revision': 'pin', 'limit': 64, 'reserve_tokens': 32}, RewardTokenizer())
    producer = SimpleNamespace(vocab_size=2, decode=lambda ids, **kw: 'x' * (ids[0] + 1))
    assert budget.validate_step_guard(producer) == 2
    producer.decode = lambda ids, **kw: 'x' * 33
    with pytest.raises(ValueError, match='producer token can exceed'):
        budget.validate_step_guard(producer)


def test_generation_reward_budget_matches_scorer_and_all_v6_models():
    from llm_routing.config import load_benchmark_config
    from llm_routing.collection.collect import generation_config
    from llm_routing.collection.dataset import model_pool
    cfg = load_benchmark_config('configs/benchmarks/shared_v6.yaml')
    for model in model_pool(cfg).values():
        generation = generation_config(cfg, model)
        assert generation['max_new_tokens'] == 8192
        assert generation['armorm_budget']['limit'] == 8192
        assert generation['armorm_budget']['reserve_tokens'] == 1024
    cfg['scoring']['armorm']['revision'] = 'different'
    with pytest.raises(ValueError, match='must match'):
        generation_config(cfg, model)
