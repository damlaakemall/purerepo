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

import math
import multiprocessing

from .evaluators import Evaluation, evaluate, supported_answer_types

LEGACY_SCORERS = {"exact_match_number": "numeric", "multiple_choice": "mcq_letter"}


class MathTimeout(TimeoutError):
    """The math judge did not finish; this does not establish an incorrect answer."""
    def __init__(self, seconds):
        self.seconds = seconds
        super().__init__(f"math evaluation exceeded {seconds}s")


def _math_worker(connection, args, kwargs):
    # Process isolation also bounds code inside upstream bare except blocks.
    try:
        connection.send((True, evaluate(*args, **kwargs)))
    except Exception as error:
        connection.send((False, error))
    finally:
        connection.close()


def _bounded_math(args, kwargs, seconds):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_math_worker, args=(writer, args, kwargs))
    process.start()
    writer.close()
    try:
        if not reader.poll(seconds):
            raise MathTimeout(seconds)
        ok, result = reader.recv()
        if not ok:
            raise result
        return result
    finally:
        reader.close()
        process.join(timeout=0.1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def score_query(query: dict, response: str, mbpp_timeout: int = 5,
                math_timeout: float | None = None) -> Evaluation:
    """Optional math deadline is opt-in; existing benchmark scoring stays unchanged."""
    if math_timeout is not None and (not math.isfinite(math_timeout) or math_timeout <= 0):
        raise ValueError("math_timeout must be finite and positive")
    aux = dict(query.get("aux") or {})
    aux["_prompt"] = query["query"]
    dataset = query.get("gt_dataset") or query.get("source", "")
    args = (dataset, response, query["gold"])
    kwargs = dict(answer_type=query["answer_type"], aux=aux, mbpp_timeout=mbpp_timeout)
    if math_timeout is not None and dataset == "olympiadbench":
        return _bounded_math(args, kwargs, math_timeout)
    try:
        return evaluate(*args, **kwargs)
    except TimeoutError:
        # MBPP has its existing, separate execution limit: failing tests is GT=0.
        return Evaluation(score=0.0, mode="timeout", strict_correct=False, predicted="timeout")


__all__ = ["Evaluation", "MathTimeout", "evaluate", "score_query", "supported_answer_types", "LEGACY_SCORERS"]
