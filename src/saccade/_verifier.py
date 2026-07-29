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

from saccade.models import Observation, Verification
from saccade.tools import ToolResult

__all__ = [
    "UNVERIFIED_CEILING",
    "adjust_confidence",
    "verify",
]

# The most an agent may believe on the model's word alone. Anything above
# this has to be earned against a measurement — that is the entire point of
# the project, so it is a hard ceiling rather than a tunable default.
UNVERIFIED_CEILING = 0.6

# How far a single verification moves confidence.
AGREEMENT_STEP = 0.25
CONFLICT_PENALTY = 0.35

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

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

    computed = _collect(measurements)
    conflict = _find_conflict(observation.statement, computed)

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
        return min(1.0, current + AGREEMENT_STEP)

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


def _find_conflict(statement: str, computed: dict[str, Any]) -> str | None:
    """Look for a measurement the statement contradicts.

    Two checks, both deliberately conservative — a false conflict sends the
    agent chasing a problem that is not there, which costs more than a missed
    one:

    1. A boolean measurement against an affirmed or negated claim.
    2. A numeric measurement against a number named in the statement.
    """
    lowered = statement.lower()

    for key, value in computed.items():
        if isinstance(value, bool):
            conflict = _boolean_conflict(lowered, key, value)
            if conflict:
                return conflict

    for key, value in computed.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        conflict = _count_conflict(lowered, key, value)
        if conflict:
            return conflict

    return None


def _boolean_conflict(lowered: str, key: str, measured: bool) -> str | None:
    """Detect a claim that contradicts a measured true/false fact."""
    subject = key.replace("_", " ")
    head = subject.split()[0] if subject.split() else subject
    if head not in lowered and subject not in lowered:
        return None

    negated = any(word in _NEGATIONS for word in lowered.split())
    claimed = not negated

    if claimed == measured:
        return None
    return (
        f"the observation states {subject} is {str(claimed).lower()}, "
        f"but the measurement found {str(measured).lower()}"
    )


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
