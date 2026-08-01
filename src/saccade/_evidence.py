"""Internal: accumulating the evidence chain.

Rule 8 says no claim leaves the system without numbers behind it. This is
where those numbers are kept, in the order they were produced, so any answer
can be traced back to the looks that produced it.

``image_ref`` holds a reference — a cache key or caller-supplied label —
never image bytes and never a path this module wrote. Persistence is
somebody else's job (rule 3).
"""

from __future__ import annotations

from saccade._observer import NON_ANSWERS
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

        The rules, in order:

        0. A measurement that contradicted the model wins outright. The model
           is then not a source for this answer at all — the tool is.
        1. A step that could not answer is skipped. Later looks are often
           magnified corners that no longer contain what the question is
           about, and taking the last statement blindly means answering with
           "I cannot tell from this view" after an earlier step answered it.
        2. A statement made about part of the image is not an answer about
           the whole image. This matters most for counting: asked how many
           times two lines cross, a model looking at one quadrant answers
           correctly about that quadrant and wrongly about the picture. On
           BlindTest's line task this alone dropped accuracy from 98.7% to
           56.7% — 62 items broken, none fixed, every one an undercount.
        3. A verified observation beats an unverified one, regardless of how
           confident the model sounded. That preference is the whole point.
        4. Among equals the most recent wins, since later looks are better
           informed.

        Rule 0 is what makes this a referee rather than an adviser. Without
        it, detecting a conflict only lowered confidence and the model's
        wrong answer was still returned: on BlindTest's touching-circles task
        the 53 items where a measurement contradicted the model scored 1.9%,
        against 97.9% on the 97 where nothing objected. The geometry had the
        right answer in all 53 and was not asked for it.
        """
        overruled = self._overruling_step()
        if overruled is not None:
            return _state(overruled.verification)

        whole = [step for step in self._steps if _is_answer(step) and _sees_whole_image(step)]

        verified = [s for s in whole if s.verification is not None and s.verification.passed]
        if verified:
            return verified[-1].observation.statement
        if whole:
            return whole[-1].observation.statement

        # No full-image answer. Fall back to a partial one rather than
        # nothing — it is weak evidence, but the caller can see from the
        # chain which viewport produced it.
        partial = [step for step in self._steps if _is_answer(step)]
        if partial:
            return partial[-1].observation.statement

        # Nothing conclusive at all. Report the last thing seen, so "no
        # answer" stays distinguishable from "never looked".
        if self._steps:
            return self._steps[-1].observation.statement
        return None

    def _overruling_step(self) -> EvidenceStep | None:
        """The last step where a whole-image measurement named a verdict the
        model contradicted.

        Two conditions, both load-bearing:

        - ``verdict_key`` must be set. A conflict without one means something
          disagreed but no tool claimed to answer, and reporting a diagnostic
          figure as the answer would be worse than reporting the model's.
        - The viewport must cover the whole image, for the reason in rule 2:
          a tool measuring a magnified corner is answering about the corner.
          An earlier version of this project let cropped measurements referee
          and produced 305 false conflicts.
        """
        for step in reversed(self._steps):
            check = step.verification
            if check is None or check.conflict is None or check.verdict_key is None:
                continue
            if not _sees_whole_image(step):
                continue
            return step
        return None

    def __len__(self) -> int:
        return len(self._steps)

    def __bool__(self) -> bool:
        return bool(self._steps)


def _state(verification: Verification | None) -> str:
    """Phrase a measured verdict as an answer.

    Booleans become "Yes"/"No" because that is what the question asked for
    and what a caller compares against; anything else is rendered as itself.
    """
    if verification is None:
        return "no observation could be made"

    verdict = verification.verdict
    if isinstance(verdict, bool):
        return "Yes" if verdict else "No"
    return str(verdict)


def _is_answer(step: EvidenceStep) -> bool:
    """Whether a step actually answered, rather than declining to."""
    lowered = step.observation.statement.lower()
    return not any(phrase in lowered for phrase in NON_ANSWERS)


def _sees_whole_image(step: EvidenceStep) -> bool:
    """Whether this step was looking at the entire image.

    A magnified crop can answer a question about itself, but the question
    was about the picture. Only a full view can settle that.
    """
    return step.viewport.covers_full_image
