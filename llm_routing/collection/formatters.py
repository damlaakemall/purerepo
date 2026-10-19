"""Source example -> frozen query row for the shared-v1 prompt set.

Each formatter returns ``{"query", "gold", "answer_type", "aux"?, "reference"?}``. The prompt asks for
the answer in the shape the ground-truth evaluator reads first ("Answer: X", ``\\boxed{}``, a fenced
code block), so the strict evaluator decides most responses and the prose fallback stays a rescue.

The prompt mix follows SU Lab LLM-Routing-Project: GSM8K, MMLU-Pro, BBH, OlympiadBench, MBPP
(checkable gold) and Alpaca, Multi-News (free-text reference, token F1).
"""

from __future__ import annotations

import ast

from ..ground_truth.metrics import extract_gsm8k_answer

LETTERS = "ABCDEFGHIJ"


def _as_list(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return list(value)


def gsm8k(ex):
    gold = extract_gsm8k_answer(ex["answer"].split("####")[-1])
    if gold is None:
        raise ValueError("GSM8K example without a '#### <number>' answer")
    return {"query": f"{ex['question'].strip()}\n\nSolve the problem step by step. End your response with a "
                     "final line of the form 'Answer: <number>'.",
            "gold": gold, "answer_type": "numeric", "reference": ex["answer"]}


def _mcq(question, options, gold):
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return {"query": f"{question.strip()}\n\nOptions:\n{opts}\n\nThink step by step, then end your response with "
                     "a final line of the form 'Answer: X', where X is the letter of the correct option.",
            "gold": gold, "answer_type": "mcq_letter", "reference": gold}


def mmlu_pro(ex):
    return _mcq(ex["question"], _as_list(ex["options"]), str(ex["answer"]).strip().upper())


def mmlu(ex):
    return _mcq(ex["question"], _as_list(ex["choices"]), LETTERS[int(ex["answer"])])


def bbh(ex):
    return {"query": f"{ex['input'].strip()}\n\nThink step by step, then end your response with a final line of "
                     "the form 'Answer: <answer>'. If the question lists options, answer with the option label, "
                     "for example 'Answer: (B)'.",
            "gold": str(ex["target"]).strip(), "answer_type": "exact_match_text", "reference": str(ex["target"])}


def olympiadbench(ex):
    unit = ex.get("unit")
    unit_note = f" The unit of the answer is {unit}; do not include it inside the box." if unit not in (None, "", "None") else ""
    gold = _as_list(ex["final_answer"])
    return {"query": f"{ex['question'].strip()}\n\nSolve the problem step by step and put the final answer in "
                     f"\\boxed{{}}.{unit_note}",
            "gold": gold, "answer_type": "exact_match_text", "gt_dataset": "olympiadbench", "reference": ", ".join(gold)}


def mbpp(ex):
    tests = _as_list(ex["test_list"])
    # The first assertion shows the function name and signature; all of them are the gold.
    return {"query": f"{ex['text'].strip()}\nYour code should pass this test:\n{tests[0]}\n\nWrite the complete "
                     "Python solution in a single ```python code block.",
            "gold": tests, "answer_type": "code_tests", "aux": {"test_setup_code": ex.get("test_setup_code") or ""},
            "reference": "\n".join(tests)}


def alpaca(ex):
    query = ex["instruction"].strip() + (f"\n\n{ex['input'].strip()}" if ex.get("input", "").strip() else "")
    return {"query": query, "gold": ex["output"], "answer_type": "reference_text", "reference": ex["output"]}


def multi_news(ex):
    articles = [a.strip() for a in ex["document"].split("|||||") if a.strip()]
    body = "\n\n".join(f"Article {i + 1}:\n{a}" for i, a in enumerate(articles))
    return {"query": f"Summarize the following news articles in one coherent summary.\n\n{body}",
            "gold": ex["summary"].lstrip("– ").strip(), "answer_type": "reference_text", "reference": ex["summary"]}


FORMATTERS = {"gsm8k": gsm8k, "mmlu_pro": mmlu_pro, "mmlu": mmlu, "bbh": bbh, "olympiadbench": olympiadbench,
              "mbpp": mbpp, "alpaca": alpaca, "multi_news": multi_news}
