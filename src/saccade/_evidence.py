"""Internal: accumulating the evidence chain.

Rule 8 says no claim leaves the system without numbers behind it. This is
where those numbers are kept, in the order they were produced, so any answer
can be traced back to the looks that produced it.

``image_ref`` holds a reference — a cache key or caller-supplied label —
never image bytes and never a path this module wrote. Persistence is
somebody else's job (rule 3).
"""

from __future__ import annotations

from saccade.models import EvidenceStep, Observation, Verification, Viewport

__all__ = ["EvidenceChain"]


class EvidenceChain:
    """An append-only record of what the agent looked at and what it found."""

    def __init__(self) -> None:
        self._steps: list[EvidenceStep] = []

    def add(
        self,
        *,
        action_name: str,
        viewport: Viewport,
        observation: Observation,
        verification: Verification | None = None,
        reason: str | None = None,
        image_ref: str | None = None,
    ) -> EvidenceStep:
        """Append a step. The index is assigned here so it cannot drift."""
        step = EvidenceStep(
            index=len(self._steps),
            action_name=action_name,
            viewport=viewport,
            observation=observation,
            verification=verification,
            reason=reason,
            image_ref=image_ref,
        )
        self._steps.append(step)
        return step

    @property
    def steps(self) -> list[EvidenceStep]:
        return list(self._steps)

    @property
    def verified_steps(self) -> list[EvidenceStep]:
        """Steps a measurement confirmed — the ones that can support a claim."""
        return [
            step
            for step in self._steps
            if step.verification is not None and step.verification.passed
        ]

    @property
    def conflicts(self) -> list[EvidenceStep]:
        """Steps where a measurement contradicted the model."""
        return [
            step
            for step in self._steps
            if step.verification is not None and step.verification.conflict is not None
        ]

    def best_statement(self) -> str | None:
        """The most defensible statement recorded so far.

        Three rules, in order:

        1. A step that could not answer is skipped. Later looks are often
           magnified corners that no longer contain what the question is
           about, and taking the last statement blindly means answering with
           "I cannot tell from this view" after an earlier step answered it.
        2. A verified observation beats an unverified one, regardless of how
           confident the model sounded. That preference is the whole point.
        3. Among equals the most recent wins, since later looks are better
           informed.
        """
        verified = [step for step in self.verified_steps if _is_answer(step)]
        if verified:
            return verified[-1].observation.statement

        answered = [step for step in self._steps if _is_answer(step)]
        if answered:
            return answered[-1].observation.statement

        # Nothing conclusive. Report the last thing seen rather than nothing,
        # so the caller can tell the difference between "no answer" and
        # "never looked".
        if self._steps:
            return self._steps[-1].observation.statement
        return None

    def __len__(self) -> int:
        return len(self._steps)

    def __bool__(self) -> bool:
        return bool(self._steps)


# What a model says when the view does not settle the question. The observer
# prompt asks for the first form; the others are what models produce anyway.
_NON_ANSWERS = (
    "cannot tell",
    "can't tell",
    "cannot determine",
    "unable to determine",
    "not possible to tell",
    "insufficient",
)


def _is_answer(step: EvidenceStep) -> bool:
    """Whether a step actually answered, rather than declining to."""
    lowered = step.observation.statement.lower()
    return not any(phrase in lowered for phrase in _NON_ANSWERS)
