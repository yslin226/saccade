"""Grading answers against ground truth.

A bug here invalidates every number the benchmark produces, so the parsing
is deliberately strict about what counts as an answer and deliberately
generous about the surrounding prose that models tend to add.

The published prompts ask for answers in a fixed shape — ``Yes``/``No`` for
Touching Circles, ``{3}`` for the counting tasks — so that shape is what we
look for first, falling back to looser forms only when it is absent.
"""

from __future__ import annotations

import re

__all__ = ["extract_count", "extract_pair", "extract_yes_no", "score"]

_BRACED = re.compile(r"\{\s*(-?\d+)\s*\}")
_BARE_INT = re.compile(r"-?\d+")

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)


def extract_yes_no(text: str) -> bool | None:
    """Pull a yes/no answer out of a reply.

    Returns None when the reply contains both or neither — an answer that
    cannot be read is scored wrong, but it must not be silently guessed at.
    """
    has_yes = bool(_YES.search(text))
    has_no = bool(_NO.search(text))

    if has_yes == has_no:
        # Both present, or neither. Fall back to whichever the reply opened
        # with, since models often state the answer then hedge.
        first_yes = _YES.search(text)
        first_no = _NO.search(text)
        if first_yes and first_no:
            return first_yes.start() < first_no.start()
        return None
    return has_yes


def extract_count(text: str) -> int | None:
    """Pull an integer answer out of a reply.

    Braced form wins: the prompts ask for ``{3}``, and a model that complies
    has told us exactly which number is the answer. Only if there is no
    braced value do we fall back to the last bare integer, which is where an
    answer usually lands in a sentence.
    """
    braced = _BRACED.findall(text)
    if braced:
        return int(braced[-1])

    bare = _BARE_INT.findall(text)
    if bare:
        return int(bare[-1])
    return None


def extract_pair(text: str) -> tuple[int, int] | None:
    """Pull two integers out of a reply, in the order they appear.

    The grid tasks ask for rows and columns — "rows={5} columns={6}", or
    "(5,6)", or a bare "5,6" from the ground truth. Both numbers matter, so
    reading only the last one would score "rows=3 columns=4" against a truth
    of 3,3 by comparing 4 with 3: wrong for the right answer and right for
    some wrong ones.
    """
    braced = _BRACED.findall(text)
    numbers = braced if len(braced) >= 2 else _BARE_INT.findall(text)
    if len(numbers) < 2:
        return None
    return int(numbers[0]), int(numbers[1])


def score(task: str, answer: str, groundtruth: str) -> bool:
    """Whether ``answer`` matches ``groundtruth`` for ``task``.

    Args:
        task: Dataset task name.
        answer: The model's reply, in full.
        groundtruth: The dataset's expected answer.

    Returns:
        True only on a confident match. An unreadable answer scores False —
        never None, never skipped, since dropping unparseable replies would
        quietly inflate accuracy.
    """
    if task == "Touching Circles":
        expected = extract_yes_no(groundtruth)
        given = extract_yes_no(answer)
        return expected is not None and given == expected

    if task.startswith("Counting Grid"):
        expected_pair = extract_pair(groundtruth)
        given_pair = extract_pair(answer)
        return expected_pair is not None and given_pair == expected_pair

    expected_count = extract_count(groundtruth)
    given_count = extract_count(answer)
    return expected_count is not None and given_count == expected_count
