"""Internal: confronting what the VLM said with what was measured.

This module is the project's actual claim. A conventional ReAct agent treats
a tool as an extension of the model and believes what comes back. Here a
tool is the model's referee, and only a computed result may overrule a
statement.

The rule that makes it work is narrow and absolute: a result may only be
used as evidence if ``is_measurement`` is True. A description produced by
another model is just a second opinion, and two blind witnesses agreeing
does not make either of them right.

Pure logic — no I/O (rule 3).
"""

from __future__ import annotations

import re
from typing import Any

from saccade._observer import NON_ANSWERS
from saccade.models import Observation, Verification
from saccade.tools import ToolResult

__all__ = [
    "UNVERIFIED_CEILING",
    "VERIFIED_FLOOR",
    "adjust_confidence",
    "verify",
]

# The most an agent may believe on the model's word alone. Anything above
# this has to be earned against a measurement — that is the entire point of
# the project, so it is a hard ceiling rather than a tunable default.
UNVERIFIED_CEILING = 0.6

# Where a single confirmed measurement puts confidence. Above the default
# threshold on purpose: the claim has been checked against a computed
# result, which is the best evidence this design can produce, and treating
# it as merely incremental left verified answers unable to converge.
VERIFIED_FLOOR = 0.85

# How far a verification moves confidence beyond the floor.
AGREEMENT_STEP = 0.25
CONFLICT_PENALTY = 0.35

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z']+")

# Words that assert something is or is not the case. A statement carrying
# none of these is descriptive, and there is nothing to contradict.
_NEGATIONS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "neither",
        "nor",
        "without",
        "cannot",
        "doesn't",
        "don't",
        "isn't",
        "aren't",
    }
)


def verify(observation: Observation, results: list[ToolResult]) -> Verification:
    """Check an observation against tool results.

    Args:
        observation: What the VLM claimed to see.
        results: Everything the tools returned this step. Entries with
            ``is_measurement=False`` are ignored for verification purposes.

    Returns:
        A :class:`~saccade.models.Verification`. When nothing measurable was
        available, ``method`` is ``"none"`` and ``passed`` is False — not
        because the observation is wrong, but because it is unconfirmed.
    """
    measurements = [result for result in results if result.is_measurement]

    if not measurements:
        return Verification(
            passed=False,
            method="none",
            computed={},
            conflict=None,
        )

    if _declined(observation.statement):
        # The model said it could not tell. There is no claim to confront, so
        # a measurement cannot confirm anything — agreeing with silence is
        # not agreement. Counting it as confirmation let a magnified corner,
        # where the tool measures fragments of shapes, raise confidence in an
        # answer nobody had given.
        return Verification(
            passed=False,
            method="none",
            computed=_collect(measurements),
            conflict=None,
        )

    computed = _collect(measurements)
    conflict = _find_conflict(observation.statement, computed, _answer_keys(measurements))

    if conflict is None:
        return Verification(passed=True, method=_method_name(measurements), computed=computed)

    return Verification(
        passed=False,
        method=_method_name(measurements),
        computed=computed,
        conflict=conflict,
    )


def adjust_confidence(current: float, verification: Verification) -> float:
    """Move confidence in response to a verification.

    Unverified observations are capped at :data:`UNVERIFIED_CEILING`. That
    cap is the mechanism behind the project's core rule: the system is not
    allowed to be sure of anything a measurement has not confirmed.
    """
    if verification.method == "none":
        # Nothing was measured. Creep towards, but never past, the ceiling.
        return min(UNVERIFIED_CEILING, current + 0.1)

    if verification.passed:
        # A measurement agreeing with the claim is the strongest evidence the
        # system can obtain, so it lands above the unverified ceiling in one
        # step rather than needing several. Requiring repeat confirmations
        # meant an answer the tools had already backed could never converge,
        # and the loop kept magnifying until the subject left the view.
        return max(min(1.0, current + AGREEMENT_STEP), VERIFIED_FLOOR)

    return max(0.0, current - CONFLICT_PENALTY)


def _collect(measurements: list[ToolResult]) -> dict[str, Any]:
    """Flatten measurement values into one dict for the evidence chain."""
    computed: dict[str, Any] = {}
    for index, result in enumerate(measurements):
        if isinstance(result.value, dict):
            computed.update(result.value)
        else:
            computed[f"value_{index}" if index else "value"] = result.value
    return computed


def _method_name(measurements: list[ToolResult]) -> str:
    names = [
        str(result.value.get("method"))
        for result in measurements
        if isinstance(result.value, dict) and "method" in result.value
    ]
    return ", ".join(names) if names else "measurement"


def _find_conflict(statement: str, computed: dict[str, Any], answer_keys: set[str]) -> str | None:
    """Look for a measurement the statement contradicts.

    Two checks, both deliberately conservative — a false conflict sends the
    agent chasing a problem that is not there, which costs more than a missed
    one:

    1. A boolean measurement against an affirmed or negated claim.
    2. A numeric measurement against a number named in the statement.

    When a tool named its ``answer_key``, only that key is judged. The rest
    is diagnostic context: a line counter reporting 1 crossing across 300
    shared columns must not have "1" checked against 300.
    """
    lowered = statement.lower()
    judged = {k: v for k, v in computed.items() if k in answer_keys} if answer_keys else computed

    for key, value in judged.items():
        if isinstance(value, bool):
            conflict = _boolean_conflict(lowered, key, value)
            if conflict:
                return conflict

    for key, value in judged.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        conflict = _count_conflict(lowered, key, value)
        if conflict:
            return conflict

    return None


def _answer_keys(measurements: list[ToolResult]) -> set[str]:
    """The keys tools declared as holding their verdict."""
    return {m.answer_key for m in measurements if m.answer_key is not None}


def _boolean_conflict(lowered: str, key: str, measured: bool) -> str | None:
    """Detect a claim that contradicts a measured true/false fact.

    Two shapes of claim, because models produce both:

    - A bare verdict — "Yes", "No.", "No, they are apart." Benchmarks ask
      for exactly this, so it is the common case. Requiring the measurement's
      own key word to appear in the statement missed all of them, and every
      answer silently passed verification.
    - A sentence naming the measured property — "the circles overlap".
    """
    claimed = _verdict(lowered)

    if claimed is None:
        subject_words = key.replace("_", " ").split()
        head = subject_words[0] if subject_words else key
        if head not in lowered:
            return None
        claimed = not any(word in _NEGATIONS for word in _words(lowered))

    if claimed == measured:
        return None

    subject = key.replace("_", " ")
    return (
        f"the observation states {subject} is {str(claimed).lower()}, "
        f"but the measurement found {str(measured).lower()}"
    )


def _verdict(lowered: str) -> bool | None:
    """Read a leading yes/no verdict, if the statement opens with one.

    Position matters: "No, the circles are not touching" is a No, while
    "Yes, though no gap is visible" is a Yes. Whichever comes first is the
    answer; the rest is justification.
    """
    words = _words(lowered)
    if not words:
        return None

    for word in words[:3]:
        if word == "yes":
            return True
        if word == "no":
            return False
    return None


def _words(text: str) -> list[str]:
    """Split into bare words, so punctuation cannot hide a keyword.

    "No." and "no," must both read as the word "no".
    """
    return _WORD.findall(text)


def _declined(statement: str) -> bool:
    """Whether the observation declined to answer.

    Kept in step with the same phrases the evidence chain skips over: a
    statement that is not an answer cannot be verified, and must not be
    treated as one.
    """
    lowered = statement.lower()
    return any(phrase in lowered for phrase in NON_ANSWERS)


def _count_conflict(lowered: str, key: str, measured: float) -> str | None:
    """Detect a number in the statement that disagrees with a measurement."""
    stated = [float(match) for match in _NUMBER.findall(lowered)]
    if not stated:
        return None

    # Any stated number matching the measurement is treated as agreement:
    # a sentence may legitimately mention several figures.
    if any(abs(value - measured) < 1e-6 for value in stated):
        return None

    return (
        f"the observation mentions {_format(stated)}, "
        f"but {key} was measured as {_format([measured])}"
    )


def _format(values: list[float]) -> str:
    rendered = [str(int(v)) if v == int(v) else f"{v:g}" for v in values]
    return " and ".join(rendered)
