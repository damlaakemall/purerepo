"""
metrics.py

Answer-extraction and comparison primitives shared by the ground-truth
evaluators.

Two families:

*binary* — the answer is either right or wrong (multiple choice, numeric,
  exact match, executable tests). `gt_answer_score` is 1.0 / 0.0.
*continuous* — the answer is free text judged against a reference
  (`answer_type: reference_text`, i.e. alpaca and multi_news here).
  `gt_answer_score` is SQuAD-style **token F1** in [0, 1], the same metric
  GraphRouter uses for its alpaca / multi_news tasks
  (`data_processing/utils.py::f1_score` after `normalize_answer`).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Optional

# --------------------------------------------------------------------- #
# SQuAD normalisation + token F1 (GraphRouter / SQuAD convention)
# --------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    s = str(text).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD token-level F1 in [0, 1]."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        # Both empty -> perfect agreement; one empty -> no overlap.
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------- #
# Numeric comparison (GSM8K)
# --------------------------------------------------------------------- #

def normalize_numeric(value: str) -> Optional[str]:
    """Canonical decimal string, or None when the text is not numeric."""
    text = str(value).strip().replace(",", "").replace("$", "")
    try:
        return str(Decimal(text).normalize())
    except InvalidOperation:
        return None


def extract_gsm8k_answer(response: str) -> Optional[str]:
    """
    Final numeric answer of a GSM8K response.

    Tries the explicit formats first ("Answer: 42", "\\boxed{42}", "#### 42")
    and falls back to the last number in the text.
    """
    text = str(response).strip()

    # Prefer an explicit Final Answer heading. Markdown responses often put
    # the answer on the next non-empty line after the heading.
    lines = text.splitlines()

    for i in range(len(lines) - 1, -1, -1):
        heading = re.sub(
            r"^[^A-Za-z0-9]+",
            "",
            lines[i],
        ).strip()

        match = re.match(
            r"(?i)^final\s+answer\s*:?\s*(.*)$",
            heading,
        )

        if not match:
            continue

        candidates = []

        inline = match.group(1).strip()
        if inline:
            candidates.append(inline)

        # Look only immediately after the Final Answer heading.
        for following in lines[i + 1:i + 6]:
            following = following.strip()

            if following:
                candidates.append(following)

        for candidate in candidates:
            nums = re.findall(
                r"-?\d[\d,]*(?:\.\d+)?",
                candidate,
            )

            if nums:
                return nums[0]

    patterns = [
        r"(?im)^\s*Answer\s*:\s*\$?(-?\d[\d,]*(?:\.\d+)?)\s*$",
        r"\\boxed\{\s*\$?(-?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"####\s*\$?(-?\d[\d,]*(?:\.\d+)?)",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, text))
    if candidates:
        return candidates[-1]
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1] if matches else None


# --------------------------------------------------------------------- #
# The answer line
# --------------------------------------------------------------------- #

#: Decoration models wrap a final answer in: markdown emphasis, code ticks,
#: headings, list bullets, numbered-list prefixes.
#:
#: Emphasis comes first so `**-24**` loses the stars and keeps the sign. A
#: bullet or list number only counts as one when whitespace follows it,
#: otherwise `-24` reads as a bullet and `336.` as a list marker, and both
#: answers are destroyed.
_DECORATION = re.compile(r"(?:[*_`#]+)|(?:^(?:[\s>]|[-+*](?=\s))*(?:\d+[.)]\s+)?)")


def answer_line(response: str) -> str:
    """
    The last non-empty line of a response, stripped of decoration.

    Models state a final answer in many shapes and all of them must reduce to
    the same thing before matching::

        (B)            **(B)**          ### Final Answer:\n**(B) Aladdin**
        - (B)          `B`             2. B.

    Looking at the *last* line is what makes a concluding answer beat a letter
    floated earlier while reasoning.
    """
    for raw in reversed(str(response).splitlines()):
        line = _DECORATION.sub("", raw).strip()
        if line:
            return line
    return ""


# --------------------------------------------------------------------- #
# Multiple-choice fallback extraction (task_evaluators convention)
# --------------------------------------------------------------------- #

def extract_mcq_fallback(response: str, allowed_letters: str) -> Optional[str]:
    """
    The multiple-choice letter a response settles on.

    The decorated last line is read first — that is where a model puts its
    final answer, in any of the shapes `answer_line` normalises. Only then
    are the prose patterns tried over the whole text, last match first.
    """
    final = answer_line(response)
    if final:
        # the line is the letter, or the letter followed by a separator and
        # the option text: "B", "(B)", "B.", "B) Aladdin", "B: Aladdin"
        # A bare leading letter must be followed by a separator or end the line: "A" and "A." are
        # answers, "A cache is ..." and "I think ..." are not (the same guard as BBH's fallback).
        head = re.match(rf"(?i)^(?:[\(\[]([{allowed_letters}])[\)\]]|([{allowed_letters}])(?:[\).:\-](?:\s|$)|$))", final)
        if head:
            return (head.group(1) or head.group(2)).upper()

    # "option" first, so an explicit "the answer is D" still wins over an
    # option named earlier while reasoning: later matches take precedence.
    patterns = [
        rf"(?i)\boption\s+(?:is\s+)?[\(\[]?([{allowed_letters}])[\)\].:\"']?",
        rf"(?i)\bcorrect\s+answer\s+is\s*[:\-]?\s*[\(\[]?([{allowed_letters}])[\)\]]?",
        rf"(?i)\banswer\s+is\s*[:\-]?\s*[\(\[]?([{allowed_letters}])[\)\]]?",
        rf"(?im)^\s*[\(\[]?([{allowed_letters}])[\)\]]?\s*[\.\:\-]\s+",
    ]
    candidates = [
        match
        for pattern in patterns
        for match in re.findall(pattern, str(response).strip())
    ]
    return candidates[-1].upper() if candidates else None
