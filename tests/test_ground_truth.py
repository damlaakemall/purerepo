"""Ground-truth scoring: the reference repo's regression cases plus the fixes made here."""
import pytest

from llm_routing.ground_truth import score_query
from llm_routing.ground_truth.evaluators import (
    _bbh_fallback, _bbh_strict, _gsm8k_strict, _mbpp_strict, _mmlupro_strict, _olympiad_strict,
)
from llm_routing.collection import formatters as F


# ---- ported from SU Lab LLM-Routing-Project tests/scoring/test_ground_truth_regressions.py ----

def test_bbh_word_sorting_commas():
    prompt = "Sort the following words alphabetically: List: testate orchestra permafrost leasehold shiva"
    ok, _ = _bbh_fallback("leasehold, orchestra, permafrost, shiva, testate",
                          "leasehold orchestra permafrost shiva testate", aux={"_prompt": prompt})
    assert ok


def test_bbh_repeated_prefix():
    prompt = ("Complete the rest of the sequence, making sure that the parentheses are closed properly. "
              "Input: [ < < > >")
    assert _bbh_fallback("[ < < > > ]", "]", aux={"_prompt": prompt})[0]


def test_mmlu_explicit_conclusions():
    assert _mmlupro_strict("Reasoning...\nThe correct option is E.", "E") == (True, "E")
    assert _mmlupro_strict("The calculated probability matches option I.", "I") == (True, "I")


def test_gsm8k_final_answer_section():
    assert _gsm8k_strict("### Final Answer:\n\n**1,128 minutes** (or **18.8 hours**) per month.", "1128") == (True, "1,128")


def test_olympiad_final_answer_variants():
    assert _olympiad_strict("There are 49 possibilities.\n\nAnswer: 49", ["49"])[0]
    assert _olympiad_strict("The largest positive integer N is n.\n\nAnswer: n", [r"$N=n$"])[0]


def test_mbpp_ignores_model_main_tests():
    response = '```python\ndef f(x):\n    return x + 1\n\nif __name__ == "__main__":\n    assert f(1) == 999\n```'
    assert _mbpp_strict(response, ["assert f(1) == 2", "assert f(5) == 6"], setup_code="", timeout=5) == (True, "passed")


# ---- fixes made in this repo ----

@pytest.mark.parametrize("response,gold,expected", [
    ("A cache is a fast store of data.", "A", 0.0),   # an article is not option A
    ("I think the second one.", "I", 0.0),            # nor is the pronoun option I
    ("Reasoning...\nAnswer: C", "C", 1.0),
    ("**(B)**", "B", 1.0),
    ("B. Aladdin", "B", 1.0),
    ("So my choice:\nD", "D", 1.0),
])
def test_mcq_letters_need_a_separator(response, gold, expected):
    row = {**F.mmlu_pro({"question": "Q", "options": list("abcdefghij"), "answer": gold}), "source": "mmlu_pro"}
    assert score_query(row, response).score == expected


@pytest.mark.parametrize("line,correct", [("Answer: B", True), ("Answer: B.", True), ("Answer: (B) Aladdin", True),
                                          ("Answer: B) Aladdin", True), ("Answer: C", False), ("Answer: A dog", False)])
def test_bbh_option_letter_on_answer_line(line, correct):
    assert _bbh_strict(f"reasoning\n{line}", "(B)")[0] is correct


def test_mbpp_timeout_is_a_wrong_answer_not_a_scoring_error():
    row = {**F.mbpp({"text": "loop", "test_list": ["assert f(1) == 2"], "test_setup_code": ""}), "source": "mbpp"}
    ev = score_query(row, "```python\ndef f(x):\n    while True:\n        pass\n```", mbpp_timeout=1)
    assert (ev.score, ev.mode) == (0.0, "timeout")


def test_formatters_match_their_evaluators():
    cases = [
        (F.gsm8k({"question": "Q?", "answer": "steps\n#### 1,128"}), "work\nAnswer: 1128", 1.0),
        (F.bbh({"input": "Which?\nOptions:\n(A) x\n(B) Aladdin", "target": "(B)"}), "think\nAnswer: (B)", 1.0),
        (F.olympiadbench({"question": "Find x", "final_answer": ["$\\frac{1}{2}$"], "unit": None}), "so \\boxed{0.5}", 1.0),
        (F.mbpp({"text": "add one", "test_list": ["assert f(1) == 2", "assert f(5) == 6"], "test_setup_code": ""}),
         "```python\ndef f(x):\n    return x + 1\n```", 1.0),
        (F.alpaca({"instruction": "Say hi", "input": "", "output": "hello there"}), "hello there", 1.0),
        (F.multi_news({"document": "a ||||| b", "summary": "– a and b"}), "a and b", 1.0),
    ]
    for row, response, expected in cases:
        assert score_query({**row, "source": "x"}, response).score == pytest.approx(expected)
    assert F.mmlu_pro({"question": "Q", "options": "['x', 'y']", "answer": "B"})["query"].count("\n(") == 2
    assert F.olympiadbench({"question": "Q", "final_answer": "['2']", "unit": "m"})["gold"] == ["2"]


def test_legacy_scorers_use_the_hardened_evaluators():
    from llm_routing.legacy_pipeline.scorers import get_scorer
    assert get_scorer("multiple_choice")(["A cache is a store."], ["A"]).tolist() == [0.0]
    assert get_scorer("exact_match_number")(["### Final Answer:\n**1,128**"], ["x #### 1128"]).tolist() == [1.0]
