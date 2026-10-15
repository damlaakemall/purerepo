"""
evaluators.py

Per-dataset ground-truth evaluators, ported from the shared-data scorer on
`feature/system-prompt-nemotron` (`scripts/icl_data/*`) and extended with the
continuous reference-text branch.

Protocol per response:
    strict    the dataset's canonical evaluator, which requires the answer in
              the requested format ("Answer: X", "\\boxed{...}", tests pass)
    fallback  an approved, task-specific rescue of an answer stated in prose
              (multiple choice only). OlympiadBench and MBPP stay strict:
              mathematical equivalence and executable tests must not be
              rescued by a regex.
    score     1.0 / 0.0 for binary datasets; token F1 in [0, 1] for
              `answer_type: reference_text`.

`evaluate()` returns an `Evaluation` carrying the score, the mode that
produced it, the strict verdict and the extracted prediction, so the matrix
keeps all variants in `score_metadata` instead of collapsing them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .metrics import (
    answer_line,
    extract_gsm8k_answer, extract_mcq_fallback, normalize_numeric, token_f1,
)

# Datasets whose gold answer is free text: scored continuously.
CONTINUOUS_ANSWER_TYPES = {"reference_text"}


@dataclass(frozen=True)
class Evaluation:
    score: float                      # in [0, 1]
    mode: str                         # strict | fallback | token_f1 | failed
    strict_correct: Optional[bool]    # None for continuous metrics
    predicted: Optional[str]


# --------------------------------------------------------------------- #
# BBH
# --------------------------------------------------------------------- #

def normalize_bbh_answer(answer: Any) -> str:
    """Normalise case and whitespace while preserving answer punctuation."""
    return " ".join(str(answer).strip().lower().split())


def extract_bbh_answer(response: str) -> Optional[str]:
    """Final non-empty ``Answer: ...`` line of a response."""
    matches = [m.strip() for m in re.findall(r"(?im)^\s*Answer\s*:\s*(.*?)\s*$", str(response))]
    matches = [m for m in matches if m]
    return matches[-1] if matches else None


def _bbh_strict(response: str, gold: Any) -> tuple[bool, Optional[str]]:
    prediction = extract_bbh_answer(response)
    if prediction is None:
        return False, None
    if normalize_bbh_answer(prediction) == normalize_bbh_answer(gold):
        return True, prediction
    # An option gold "(B)" stated on the Answer line as "B", "B.", "B) Aladdin" or "(B) Aladdin".
    # A bare letter needs a separator or the end of the line, so "Answer: A dog" is not option A.
    option = re.fullmatch(r"\(([A-Z])\)", str(gold).strip())
    letter = re.match(r"^(?:\(([A-Z])\)(?:\s.*)?|([A-Z])(?:[.:)](?:\s.*)?)?)$", prediction.strip())
    if option and letter:
        return (letter.group(1) or letter.group(2)) == option.group(1), prediction
    return False, prediction


def _last_nonempty_line(response: str) -> str:
    """The decorated final line; see metrics.answer_line."""
    return answer_line(response)


def _bbh_fallback(
    response: str,
    gold: Any,
    aux: Optional[dict] = None,
) -> tuple[bool, Optional[str]]:
    """
    Rescue an answer stated in prose, when the strict ``Answer:`` line is
    absent — which it usually is, because the prompts say "give only the
    final answer".

    Two gold shapes occur in BBH and both are handled:

    * an option letter, ``(B)``;
    * a bare literal, ``valid`` / ``invalid`` / ``Yes`` / ``No`` /
      ``True`` / ``False`` (formal_fallacies, navigate, web_of_lies,
      boolean_expressions have no options).

    The **last** line is consulted first and prose patterns are read
    last-match-first, so a concluding answer always beats an aside made
    while deliberating.
    """
    gold_text = str(gold).strip()
    response_text = str(response).strip()
    last = _last_nonempty_line(response)
    prompt = str((aux or {}).get("_prompt") or "")

    # BBH word sorting is sequence-valued; commas are presentation only.
    if prompt.lstrip().lower().startswith(
        "sort the following words alphabetically"
    ):
        def _word_sequence(text: str) -> list[str]:
            return [
                x.lower()
                for x in re.findall(r"[A-Za-z]+", text)
            ]

        if _word_sequence(response_text) == _word_sequence(gold_text):
            return True, response_text

    # In the Dyck/completion task, some models repeat the supplied prefix
    # and append the requested completion. Accept that only for this task.
    if prompt.lstrip().lower().startswith(
        "complete the rest of the sequence, making sure that the parentheses are closed properly"
    ):
        match = re.search(r"(?im)\bInput\s*:\s*(.*?)\s*$", prompt)
        if match:
            given = normalize_bbh_answer(match.group(1))
            pred = normalize_bbh_answer(response_text)
            target = normalize_bbh_answer(gold_text)

            if pred == f"{given} {target}".strip():
                return True, response_text

    # 0. Exact answer modulo whitespace. This rescues structured answers such
    # as BBH word-sorting outputs emitted one token per line.
    if (
        response_text
        and normalize_bbh_answer(response_text)
        == normalize_bbh_answer(gold_text)
    ):
        return True, response_text

    # 1. The model did what it was asked: the final line *is* the answer.
    if last and normalize_bbh_answer(last) == normalize_bbh_answer(gold_text):
        return True, last

    option = re.fullmatch(r"\(([A-Z])\)", gold_text)
    if option is None:
        # 2. A literal gold: accept it as the final token of the last line,
        #    so "the argument is invalid" counts but "invalid" inside
        #    "...is not invalid" does not get mined out of the middle.
        if last and re.search(rf"(?i)\b{re.escape(gold_text)}\s*[.!]?$", last):
            return True, gold_text
        return False, last or None

    # 3. If the prompt exposes labelled options and the complete model
    # response exactly matches one unique option text, map that text back to
    # its letter. This is deliberately exact: partial/semantic matching would
    # risk inventing correctness.
    if prompt and response_text:
        options = {}
        for raw_line in prompt.splitlines():
            match = re.match(r"^\s*\(([A-Z])\)\s*(.+?)\s*$", raw_line)
            if match:
                options[match.group(1).upper()] = match.group(2)

        matched_letters = [
            letter
            for letter, text in options.items()
            if normalize_bbh_answer(response_text)
            == normalize_bbh_answer(text)
        ]

        if len(matched_letters) == 1:
            predicted = matched_letters[0]
            return (
                predicted == option.group(1).upper(),
                f"({predicted})",
            )

    # 4. An option gold: the letter may lead the last line ("(G) Ana ..."),
    #    be the whole line ("G"), or be named in prose. Later matches win, so
    #    a stated conclusion beats the first word of a sentence.
    #
    #    BBH options run to (R), so the letter class is the full A-Z and "A"
    #    and "I" are also English words. The leading-letter patterns therefore
    #    demand a separator or an otherwise empty line — "A dog ..." is not an
    #    answer, "A." and "A" are.
    patterns = [
        r"(?i)^\s*\(?([A-Z])[\)\].:\-]",
        r"(?i)^\s*([A-Z])\s*$",
        r"(?i)\bcorrect\s+answer\s+is\s*[:\-]?\s*\(?([A-Z])\)?",
        r"(?i)\banswer\s+is\s*[:\-]?\s*\(?([A-Z])\)?",
        r"(?i)\boption\s+\(?([A-Z])\)?\s+is\s+correct",
    ]
    for text in (last, str(response).strip()):     # the last line, then the whole response
        candidates = [c for pat in patterns for c in re.findall(pat, text)]
        if candidates:
            predicted = candidates[-1].upper()
            return predicted == option.group(1).upper(), f"({predicted})"
    return False, last or None


# --------------------------------------------------------------------- #
# MMLU-Pro
# --------------------------------------------------------------------- #

#: Final ``Answer: <A-J>`` line, tolerating markdown emphasis around it
#: (`**Answer: F**`) and the bracketed letter form. Only the explicit answer
#: line is relaxed this way — an option list must not match it.
_MMLUPRO_ANSWER = re.compile(
    r"(?i)^(?:the\s+correct\s+answer\s+is|(?:final|correct)\s+answer|answer)"
    r"\s*[:\-]?\s*[\(\[]?([A-J])(?:[\)\]]|\b)"
)

_MMLUPRO_CONCLUSION = re.compile(
    r"(?i)^(?:"
    r"(?:the\s+)?(?:final|correct)\s+answer\s+is"
    r"|(?:the\s+)?correct\s+option\s+is"
    r"|therefore[,\s:\-]*the\s+correct\s+answer\s+is"
    r"|this\s+corresponds\s+to\s+option"
    r"|.*\bmatches\s+option"
    r"|let['’]?\s+select"
    r")"
    r"\s*[:\-]?\s*[\(\[]?([A-J])(?:[\)\]]|\b)"
)


def extract_mmlupro_answer(response: str) -> Optional[str]:
    """Return the last explicit A-J answer conclusion."""
    for raw in reversed(str(response).splitlines()):
        line = answer_line(raw)
        if not line:
            continue

        # Drop leading decorative symbols such as ✅ after markdown cleanup.
        line = re.sub(r"^[^\w]+", "", line).strip()

        match = _MMLUPRO_ANSWER.match(line)
        if match:
            return match.group(1).upper()

        match = _MMLUPRO_CONCLUSION.match(line)
        if match:
            return match.group(1).upper()

    return None



def _mmlupro_strict(response: str, gold: Any) -> tuple[bool, Optional[str]]:
    prediction = extract_mmlupro_answer(response)
    if prediction is None:
        return False, None
    return prediction == str(gold).strip().upper(), prediction


def _mmlupro_fallback(response: str, gold: Any) -> tuple[bool, Optional[str]]:
    predicted = extract_mcq_fallback(response, "ABCDEFGHIJ")
    if predicted is None:
        return False, None
    return predicted == str(gold).strip().upper(), predicted


# --------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------- #

def _gsm8k_strict(response: str, gold: Any) -> tuple[bool, Optional[str]]:
    predicted = extract_gsm8k_answer(response)
    if predicted is None:
        return False, None
    pred_norm, gold_norm = normalize_numeric(predicted), normalize_numeric(str(gold))
    if pred_norm is None or gold_norm is None:
        return str(predicted).strip() == str(gold).strip(), predicted
    return pred_norm == gold_norm, predicted


# --------------------------------------------------------------------- #
# OlympiadBench (strict only — sympy equivalence judge)
# --------------------------------------------------------------------- #

_JUDGE = None


def _olympiad_judge():
    global _JUDGE
    if _JUDGE is None:
        from .olympiad_judge import AutoScoringJudge

        _JUDGE = AutoScoringJudge()
    return _JUDGE


def prepare_olympiad_gold(final_answer: Any) -> str:
    """OlympiadBench golds are lists; the judge takes a comma-joined string."""
    if isinstance(final_answer, str):
        return final_answer
    if not isinstance(final_answer, list):
        raise TypeError(f"Unexpected final_answer type: {type(final_answer)}")
    if not final_answer:
        raise ValueError("Empty final_answer")
    return ", ".join(str(x) for x in final_answer)


#: A `\text{...}` run inside an answer, e.g. `\boxed{\frac{600}{7}\text{ km/h}}`.
_LATEX_TEXT = re.compile(r"\\text\s*\{[^{}]*\}")
_PERCENT_NUMBER = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(%)?\s*$")


def _percentage_numeric_equal(gold: str, predicted: str, precision: float) -> bool:
    """Compare pure numeric answers, applying /100 only when '%' is explicit."""
    g = _PERCENT_NUMBER.fullmatch(str(gold))
    p = _PERCENT_NUMBER.fullmatch(str(predicted))
    if not g or not p:
        return False

    gv = float(g.group(1)) / (100.0 if g.group(2) else 1.0)
    pv = float(p.group(1)) / (100.0 if p.group(2) else 1.0)
    return abs(gv - pv) <= precision * 1.01


def _judge(gold_text: str, response: str, precision: float) -> bool:
    try:
        return bool(_olympiad_judge().judge(gold_text, response, precision))
    except Exception:  # noqa: BLE001 - judge failures count as incorrect
        return False


def _boxed(response: str) -> Optional[str]:
    """
    Content of the last `\\boxed{...}`, brace-matched so `\\frac{1}{2}` survives.

    Diagnostic only: the judge reads the whole response, and this is recorded
    as `predicted_answer` so a wrong answer can be told from an unreadable one.
    """
    start = response.rfind(r"\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len(r"\boxed{") - 1
    for j in range(i, len(response)):
        if response[j] == "{":
            depth += 1
        elif response[j] == "}":
            depth -= 1
            if depth == 0:
                return response[i + 1:j].strip() or None
    return None



def _strip_math_wrappers(text: str) -> str:
    """Strip outer inline/display LaTeX delimiters."""
    text = str(text).strip()

    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        text = text[2:-2]
    elif text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1]
    elif text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2]
    elif text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2]

    return text.strip()


def _olympiad_candidate_variants(text: str, gold_text: str) -> list[str]:
    """Conservative candidates extracted from concluding answer text."""
    if not text:
        return []

    out: list[str] = []

    def add(value: str) -> None:
        value = str(value).strip().rstrip(".").strip()
        value = _strip_math_wrappers(value)

        if value and value not in out:
            out.append(value)

    add(text)

    # "Answer: 49", "Final Answer: ..."
    payload = re.sub(
        r"(?i)^\s*(?:final\s+answer|answer)\s*:\s*",
        "",
        text,
    )
    add(payload)

    # "The length of QT is 2\sqrt{10}."
    parts = re.split(r"(?i)\s+is\s+", payload)
    if len(parts) > 1:
        tail = parts[-1].strip()
        tail = re.split(
            r"(?i)\s*\(\s*or\b",
            tail,
            maxsplit=1,
        )[0]
        add(tail)

    # Final equation such as n=37 or ... = 45/2
    # when the canonical answer itself is scalar.
    if "=" not in gold_text and "=" in payload:
        rhs = payload.rsplit("=", 1)[1].strip()
        rhs = rhs.rstrip(".").strip()
        rhs = rhs.strip("$").strip()
        add(rhs)

    return out


def _olympiad_gold_variants(gold_text: str) -> list[str]:
    """Gold plus safe RHS for simple assignments such as N=n."""
    compact = _strip_math_wrappers(gold_text)
    out = [compact]

    if compact.count("=") == 1:
        lhs, rhs = compact.split("=", 1)

        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", lhs.strip()):
            rhs = rhs.strip()
            if rhs and rhs not in out:
                out.append(rhs)

    return out


def _olympiad_strict(response: str, gold: Any, precision: float = 1e-8) -> tuple[bool, Optional[str]]:
    gold_text = prepare_olympiad_gold(gold)
    # Models write percentages as \%; the upstream scorer strips % but not the
    # backslash, so canonicalise the escape first.
    gold_text = gold_text.replace(r"\%", "%")
    response = str(response).replace(r"\%", "%")
    predicted = _boxed(response) or answer_line(response) or None

    if predicted and _percentage_numeric_equal(gold_text, predicted, precision):
        return True, predicted

    # The judge compares two expressions, so a whole multi-step solution can
    # fail on a correct answer: the other equations in it are compared too.
    # Try the full text first, then the model's own final answer, then each
    # without `\text{...}` units the golds never carry. Every retry is a
    # narrower comparison than the one before and only runs after a failure,
    # so none of them can turn a wrong answer into a right one.
    candidates: list[str] = [response]

    if predicted:
        candidates.extend(
            _olympiad_candidate_variants(predicted, gold_text)
        )

    # Only inspect a small concluding tail; never mine arbitrary
    # intermediate calculations from the entire reasoning.
    tail_lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ][-6:]

    for line in reversed(tail_lines):
        candidates.extend(
            _olympiad_candidate_variants(line, gold_text)
        )

    gold_variants = _olympiad_gold_variants(gold_text)
    tried: set[tuple[str, str]] = set()

    for text in candidates:
        if not text:
            continue

        for variant in (text, _LATEX_TEXT.sub("", text)):
            if not variant:
                continue

            for gold_variant in gold_variants:
                key = (gold_variant, variant)

                if key in tried:
                    continue

                tried.add(key)

                if _judge(gold_variant, variant, precision):
                    return True, predicted

    return False, predicted


# --------------------------------------------------------------------- #
# MBPP (strict only — executes the model's code against the tests)
# --------------------------------------------------------------------- #

def _mbpp_strict(response: str, gold: Any, setup_code: str = "", timeout: int = 5) -> tuple[bool, Optional[str]]:
    from .mbpp_exec import extract_python_code, run_mbpp_tests

    tests = gold if isinstance(gold, list) else [gold]
    code = extract_python_code(response)
    passed, status = run_mbpp_tests(code=code, test_list=tests, test_setup_code=setup_code, timeout=timeout)
    if status == "timeout":
        raise TimeoutError(f"MBPP execution exceeded {timeout}s")
    return passed, status


# --------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------- #

# Primary dispatch is on the **answer type** the answer key declares, not on
# the dataset name: `numeric`, `mcq_letter`, `exact_match_text`, `code_tests`
# and `reference_text` describe what a gold answer *is*, so a dataset this
# repo has never seen scores correctly as long as it declares one of them.
#
# Dataset overrides exist for the cases where the answer type is not enough.
# OlympiadBench golds are `exact_match_text` like BBH's, but comparing them
# as strings is wrong — `\frac{1}{2}` and `0.5` are the same answer — so it
# gets the sympy equivalence judge instead.
ANSWER_TYPE_EVALUATORS = {
    "numeric": _gsm8k_strict,
    "mcq_letter": _mmlupro_strict,
    "exact_match_text": _bbh_strict,
    "code_tests": _mbpp_strict,
}

DATASET_EVALUATORS = {
    "olympiadbench": _olympiad_strict,
}

# The approved prose fallback, tried only when the strict evaluator failed:
# pull a bare option letter out of a chatty answer. Not offered for numeric,
# code or free-text golds, where a loose match would invent correctness.
ANSWER_TYPE_FALLBACKS = {
    "mcq_letter": _mmlupro_fallback,
    "exact_match_text": _bbh_fallback,
}

DATASET_FALLBACKS: dict[str, Any] = {
    "olympiadbench": None,      # the judge is already lenient; no prose rescue
}

#: Strict evaluators that need the extra arguments in `aux`.
_NEEDS_AUX = {_mbpp_strict}

#: Fallback evaluators that need response-side context such as the prompt.
_FALLBACK_NEEDS_AUX = {_bbh_fallback}


def supported_answer_types() -> set[str]:
    return set(ANSWER_TYPE_EVALUATORS) | set(CONTINUOUS_ANSWER_TYPES)


def _pick(dataset: str, answer_type: Optional[str], by_dataset: dict, by_type: dict):
    """Dataset override first, then the answer type."""
    if dataset in by_dataset:
        return by_dataset[dataset]
    return by_type.get(answer_type or "")


def evaluate(dataset: str, response: str, gold: Any, *, answer_type: Optional[str] = None,
             aux: Optional[dict] = None, mbpp_timeout: int = 5) -> Evaluation:
    """
    Score one response.

    Free-text golds (`answer_type in CONTINUOUS_ANSWER_TYPES`) get a
    continuous token F1 against the reference; everything else runs the
    strict evaluator for its answer type and then, where approved, the prose
    fallback.
    """
    if answer_type in CONTINUOUS_ANSWER_TYPES:
        reference = gold if isinstance(gold, str) else " ".join(map(str, gold or []))
        return Evaluation(score=token_f1(response, reference), mode="token_f1",
                          strict_correct=None, predicted=None)

    strict_fn = _pick(dataset, answer_type, DATASET_EVALUATORS, ANSWER_TYPE_EVALUATORS)
    if strict_fn is None:
        raise ValueError(
            f"No ground-truth evaluator for dataset '{dataset}' (answer_type={answer_type!r}). "
            f"Declare one of {sorted(supported_answer_types())} in the answer key, "
            f"or add a dataset override in DATASET_EVALUATORS.")

    if strict_fn in _NEEDS_AUX:
        correct, predicted = strict_fn(response, gold, (aux or {}).get("test_setup_code", ""), mbpp_timeout)
    else:
        correct, predicted = strict_fn(response, gold)
    if correct:
        return Evaluation(score=1.0, mode="strict", strict_correct=True, predicted=predicted)

    fallback_fn = _pick(dataset, answer_type, DATASET_FALLBACKS, ANSWER_TYPE_FALLBACKS)
    if fallback_fn is not None:
        if fallback_fn in _FALLBACK_NEEDS_AUX:
            rescued, fallback_pred = fallback_fn(response, gold, aux)
        else:
            rescued, fallback_pred = fallback_fn(response, gold)
        if rescued:
            return Evaluation(score=1.0, mode="fallback", strict_correct=False, predicted=fallback_pred)
        # Keep what the fallback read even when it did not match: a wrong
        # answer and an unreadable one both score 0, and `predicted_answer`
        # is the only way to tell them apart afterwards.
        predicted = predicted or fallback_pred

    return Evaluation(score=0.0, mode="failed", strict_correct=False, predicted=predicted)
