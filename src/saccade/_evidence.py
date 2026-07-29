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
        image_ref: str | None = None,
    ) -> EvidenceStep:
        """Append a step. The index is assigned here so it cannot drift."""
        step = EvidenceStep(
            index=len(self._steps),
            action_name=action_name,
            viewport=viewport,
            observation=observation,
            verification=verification,
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

        A verified observation beats an unverified one regardless of how
        confident the model sounded — that preference is the whole point.
        Among equals, the most recent wins, since later looks are better
        informed.
        """
        verified = self.verified_steps
        if verified:
            return verified[-1].observation.statement
        if self._steps:
            return self._steps[-1].observation.statement
        return None

    def __len__(self) -> int:
        return len(self._steps)

    def __bool__(self) -> bool:
        return bool(self._steps)
