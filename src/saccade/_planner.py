"""Internal: deciding where to look next.

This is the agency the position paper says VLMs lack. A passive model gets
one fixation; this module chooses the next one based on what the previous
looks produced.

M1 keeps the policy rule-based rather than asking a model where to look.
Two reasons: the loop has to work before it can be tuned, and a rule-based
planner is a fixed baseline that M2's ablations can measure a learned
planner against. Swapping the policy is a change to this file alone.

Pure logic — no I/O (rule 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from saccade.models import BBox, EvidenceStep, Viewport

__all__ = ["Action", "PlannedAction", "Planner"]

# Below this, a region is too small to be worth magnifying further; the
# model is looking at noise rather than structure.
MIN_REGION_PIXELS = 16

# How much of the image a single scan step examines.
SCAN_DIVISIONS = 2

DEFAULT_ZOOM = 2.0
CONFLICT_ZOOM = 3.0


class Action(StrEnum):
    """What to do next."""

    LOOK = "look"
    """Examine a region at 1x — the opening move."""

    ZOOM = "zoom"
    """Magnify a region, because detail is what is missing."""

    STOP = "stop"
    """Nothing further would help."""


@dataclass(frozen=True)
class PlannedAction:
    """The planner's decision, and why.

    ``reason`` goes into the evidence chain: an auditor should be able to see
    not just where the agent looked but what made it look there.
    """

    action: Action
    viewport: Viewport
    reason: str
    zoom_factor: float = 1.0


@dataclass
class Planner:
    """Chooses the next viewport from the investigation so far.

    Args:
        source_size: Size of the image under investigation.
        max_steps: Ceiling on iterations, used to stop scanning near the end.
    """

    source_size: tuple[int, int]
    max_steps: int = 8
    _explored: list[BBox] = field(default_factory=list, init=False)

    @property
    def explored(self) -> list[BBox]:
        """Regions already examined. Prevents looking at the same place twice."""
        return list(self._explored)

    def record(self, viewport: Viewport) -> None:
        """Note that a region has been examined."""
        self._explored.append(viewport.bbox)

    def plan_next(
        self,
        evidence: list[EvidenceStep],
        confidence: float,
    ) -> PlannedAction:
        """Decide the next move.

        The order of the rules is the policy:

        1. Nothing seen yet, so look at the whole image.
        2. A measurement contradicted the model — magnify the disputed
           region, since a conflict is the strongest signal available that
           more detail is needed.
        3. Confidence is low and unexplored area remains — scan it.
        4. Otherwise stop: repeating a look that has already failed to
           resolve the question only costs tokens.
        """
        if not evidence:
            return PlannedAction(
                action=Action.LOOK,
                viewport=Viewport.full(self.source_size),
                reason="opening look at the whole image",
            )

        conflicted = _latest_conflict(evidence)
        if conflicted is not None:
            zoomed = self._tighten(conflicted.viewport.bbox)
            if zoomed is not None:
                return PlannedAction(
                    action=Action.ZOOM,
                    viewport=Viewport(
                        bbox=zoomed, zoom=CONFLICT_ZOOM, source_size=self.source_size
                    ),
                    reason=(
                        f"measurement conflicted with the observation at step "
                        f"{conflicted.index}; magnifying the disputed region"
                    ),
                    zoom_factor=CONFLICT_ZOOM,
                )

        settled = _settled_by_measurement(evidence)
        if settled is not None:
            return PlannedAction(
                action=Action.STOP,
                viewport=settled.viewport,
                reason=(
                    f"a measurement confirmed the observation at step "
                    f"{settled.index}; looking again cannot improve on that"
                ),
            )

        if len(evidence) >= self.max_steps:
            return PlannedAction(
                action=Action.STOP,
                viewport=Viewport.full(self.source_size),
                reason=f"reached the {self.max_steps}-step budget",
            )

        unexplored = self._next_unexplored()
        if unexplored is not None:
            return PlannedAction(
                action=Action.ZOOM,
                viewport=Viewport(bbox=unexplored, zoom=DEFAULT_ZOOM, source_size=self.source_size),
                reason=f"confidence {confidence:.2f} is short of the threshold; "
                f"examining an unexplored region",
                zoom_factor=DEFAULT_ZOOM,
            )

        return PlannedAction(
            action=Action.STOP,
            viewport=Viewport.full(self.source_size),
            reason="every region has been examined and no conflict remains",
        )

    def _tighten(self, bbox: BBox) -> BBox | None:
        """Halve a region around its centre, to look harder at the same place.

        Returns None once the region is too small to subdivide usefully —
        magnifying noise produces confident nonsense, not detail.
        """
        new_w = bbox.w // 2
        new_h = bbox.h // 2
        if new_w * new_h < MIN_REGION_PIXELS:
            return None

        centre_x = bbox.x + bbox.w // 2
        centre_y = bbox.y + bbox.h // 2
        x = max(0, centre_x - new_w // 2)
        y = max(0, centre_y - new_h // 2)

        width, height = self.source_size
        new_w = min(new_w, width - x)
        new_h = min(new_h, height - y)
        if new_w <= 0 or new_h <= 0:
            return None

        candidate = BBox(x=x, y=y, w=new_w, h=new_h)
        return None if self._already_explored(candidate) else candidate

    def _next_unexplored(self) -> BBox | None:
        """Pick the first quadrant-sized region not yet examined."""
        width, height = self.source_size
        tile_w = max(1, width // SCAN_DIVISIONS)
        tile_h = max(1, height // SCAN_DIVISIONS)

        for row in range(SCAN_DIVISIONS):
            for col in range(SCAN_DIVISIONS):
                x = col * tile_w
                y = row * tile_h
                # Let the last tile in each direction absorb any remainder.
                w = width - x if col == SCAN_DIVISIONS - 1 else tile_w
                h = height - y if row == SCAN_DIVISIONS - 1 else tile_h
                if w <= 0 or h <= 0:
                    continue

                candidate = BBox(x=x, y=y, w=w, h=h)
                if not self._already_explored(candidate):
                    return candidate
        return None

    def _already_explored(self, candidate: BBox) -> bool:
        """Whether this exact region has been examined.

        Only exact matches count. Treating "contained in something already
        seen" as explored looks reasonable but is wrong: the opening move
        examines the whole image, which contains every subsequent region, so
        the agent would conclude there was nothing left to look at and stop
        after one step. Looking closer at part of an area already seen at a
        wider view is the entire point of a saccade.
        """
        return any(seen == candidate for seen in self._explored)


def _settled_by_measurement(evidence: list[EvidenceStep]) -> EvidenceStep | None:
    """The step where a measurement confirmed what the model said, if any.

    Once a computed result has backed an answer, looking again cannot make it
    more true, so the referee agreeing is the strongest stopping signal the
    loop has.

    Note what this does *not* say: it is not a claim that one look is enough
    in general. It stops only when a measurement has settled the question.
    Tasks with no measurement tool keep exploring, and so do tasks where the
    subject is small enough that magnifying genuinely helps — reading a
    circled letter, or judging an occluded joint in a video frame. On
    Touching Circles the two shapes fill the frame, so a magnified crop
    always cuts one of them; that is a property of the task, not a rule about
    saccades.
    """
    for step in evidence:
        if step.verification is not None and step.verification.passed:
            return step
    return None


def _latest_conflict(evidence: list[EvidenceStep]) -> EvidenceStep | None:
    """The most recent step where a measurement contradicted the model."""
    for step in reversed(evidence):
        if step.verification is not None and step.verification.conflict is not None:
            return step
    return None
