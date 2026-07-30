"""Detecting pose readings that cannot be true.

MediaPipe reports a confidence for every joint and it is not a reliable
error signal: on a real pitching clip it flagged none of its own failures.
Measured on happy/林永閎.MOV — 450 frames, zero frames where it admitted it
could not see, and four readings where a joint moved further in 1/30s than a
human limb can move, each reported with confidence between 0.79 and 0.99.

So the error signal has to come from physics rather than from the detector.
A wrist cannot travel 286 pixels between adjacent frames of 30fps video. If
it appears to, one of the two readings is wrong, whatever confidence came
attached.

This is a benchmark tool, not part of the library: rule 2 keeps MediaPipe out
of src/saccade entirely. It reaches the agent through register_tool().
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL.Image import Image
from pydantic import BaseModel

from saccade.tools import Tool, ToolResult

__all__ = [
    "MAX_JOINT_TRAVEL_PX",
    "ContinuityParams",
    "JointReading",
    "continuity_tool",
    "implausible_joints",
]

# The furthest a tracked joint may plausibly travel between adjacent frames
# of 30fps video, in pixels, for a 1920x1080 clip of a person throwing.
#
# Derived from the clip rather than chosen: across 450 frames the 4500 joint
# transitions sit far below this, and the four that exceed it are all visibly
# wrong when the frames are inspected. It scales with resolution and frame
# rate, so a caller working at a different scale must pass its own.
MAX_JOINT_TRAVEL_PX = 150.0


@dataclass(frozen=True)
class JointReading:
    """One joint's position in one frame, as the detector reported it."""

    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class Implausible:
    """A reading that physics rules out."""

    name: str
    travel_px: float
    reported_confidence: float

    def describe(self) -> str:
        return (
            f"{self.name} moved {self.travel_px:.0f}px in one frame "
            f"(limit {MAX_JOINT_TRAVEL_PX:.0f}px), reported at "
            f"confidence {self.reported_confidence:.2f}"
        )


class ContinuityParams(BaseModel):
    """This tool measures the frame it is given, so it takes no arguments."""


def implausible_joints(
    previous: list[JointReading],
    current: list[JointReading],
    *,
    limit: float = MAX_JOINT_TRAVEL_PX,
) -> list[Implausible]:
    """Which joints moved further than a body can between two frames.

    Args:
        previous: Readings from the preceding frame.
        current: Readings from the frame under examination.
        limit: Maximum plausible travel in pixels.

    Returns:
        The implausible readings, largest travel first. Empty means the frame
        is continuous with the one before it — which is not proof that it is
        correct, only that this particular check found nothing.
    """
    before = {reading.name: reading for reading in previous}

    found: list[Implausible] = []
    for reading in current:
        earlier = before.get(reading.name)
        if earlier is None:
            continue
        travel = ((reading.x - earlier.x) ** 2 + (reading.y - earlier.y) ** 2) ** 0.5
        if travel > limit:
            found.append(
                Implausible(
                    name=reading.name,
                    travel_px=travel,
                    reported_confidence=reading.confidence,
                )
            )

    found.sort(key=lambda i: -i.travel_px)
    return found


def continuity_tool(
    previous: list[JointReading],
    current: list[JointReading],
    *,
    limit: float = MAX_JOINT_TRAVEL_PX,
) -> Tool:
    """Build a tool that reports whether this frame's pose is physically possible.

    The verdict is a measurement: it comes from arithmetic on coordinates, not
    from anything's opinion. What it cannot say is *why* a joint jumped — motion
    blur, occlusion and a lost track all look identical in the numbers, and they
    call for different handling. That question is what the agent is for.
    """

    def run(image: Image, viewport: object) -> ToolResult:
        found = implausible_joints(previous, current, limit=limit)
        return ToolResult(
            value={
                "method": "pose_continuity",
                "plausible": not found,
                "implausible_count": len(found),
                "worst_travel_px": round(found[0].travel_px, 1) if found else 0.0,
                "detail": [i.describe() for i in found],
            },
            is_measurement=True,
            answer_key="plausible",
        )

    return Tool(
        name="pose_continuity",
        description=(
            "Check whether the detected joint positions could physically follow "
            "from the previous frame"
        ),
        fn=run,
        params_schema=ContinuityParams,
    )
