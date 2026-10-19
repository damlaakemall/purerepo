"""Ground-truth answer scoring, ported from SU Lab LLM-Routing-Project (`data_generator/scoring/ground_truth`).

The evaluators were hardened there by blind audits; they are copied unchanged so both repos score
identically. Dispatch is on the query's ``answer_type`` (numeric | mcq_letter | exact_match_text |
code_tests | reference_text), with a dataset override for ``olympiadbench`` (sympy equivalence judge).

    score_query(query_row, response_text) -> Evaluation(score in [0, 1], mode, strict_correct, predicted)

``reference_text`` is continuous token F1; every other type is 0/1. MBPP executes model code in a
subprocess (``python -I``, timeout) — run scoring on a compute node, not on a machine with secrets.

The OlympiadBench judge is vendored from OpenBMB under the bundled ``OLYMPIADBENCH_LICENSE`` (MIT).
"""

from __future__ import annotations

from .evaluators import Evaluation, evaluate, supported_answer_types

# Older configs named the scorer instead of the answer type.
LEGACY_SCORERS = {"exact_match_number": "numeric", "multiple_choice": "mcq_letter"}


def score_query(query: dict, response: str, mbpp_timeout: int = 5) -> Evaluation:
    """Score one response against a frozen query row (``answer_type``, ``gold``, ``aux``, ``gt_dataset``)."""
    aux = dict(query.get("aux") or {})
    aux["_prompt"] = query["query"]  # BBH's fallback reads the options from the prompt
    try:
        return evaluate(query.get("gt_dataset") or query.get("source", ""), response, query["gold"],
                        answer_type=query["answer_type"], aux=aux, mbpp_timeout=mbpp_timeout)
    except TimeoutError:
        # Code that does not finish within the limit fails the tests (an infinite loop is a wrong
        # answer, not a scoring error). The mode records it so a too-tight limit is visible.
        return Evaluation(score=0.0, mode="timeout", strict_correct=False, predicted="timeout")


__all__ = ["Evaluation", "evaluate", "score_query", "supported_answer_types", "LEGACY_SCORERS"]
