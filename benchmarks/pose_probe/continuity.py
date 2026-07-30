"""Detecting pose readings that cannot be true — an approach that failed.

The premise held up: a detector's confidence is not a reliable error signal.
On a pitching clip MediaPipe produced four physically impossible readings and
reported confidence between 0.79 and 0.99 for every one, never once admitting
it could not see.

The proposed remedy did not. Frame-to-frame joint travel, normalised by the
subject's own shoulder width, scored 2.78x on the clips it was tuned on and
0.64x on clips it had not seen — below 1.0, meaning the frames it flagged were
the *better* ones. Two other signals fared no better: absolute pixel travel
1.02x, bone-length consistency 1.00x.

Kept rather than deleted because the failure is worth reproducing, and because
the arithmetic is a fair starting point for anyone trying a different signal.
Do not treat a flag from this module as evidence a frame is wrong.

The one honest finding is negative and worth stating: on 1139 hand-labelled
frames, no purely geometric single-frame signal we tried separated a bad pose
estimate from a good one.

This is a benchmark tool, not part of the library: rule 2 keeps MediaPipe out
of src/saccade entirely. It reaches the agent through register_tool().
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL.Image import Image
from pydantic import BaseModel

from saccade.tools import Tool, ToolResult

__all__ = [
    "FALLBACK_TRAVEL_PX",
    "MAX_TRAVEL_PER_SHOULDER_WIDTH",
    "ContinuityParams",
    "JointReading",
    "continuity_tool",
    "implausible_joints",
    "shoulder_width",
]

# The furthest a joint may plausibly travel between adjacent frames, as a
# multiple of the subject's shoulder width.
#
# DOES NOT WORK as an error detector. Kept because the arithmetic is sound and
# because the failure is worth being able to reproduce, but do not build on it.
#
# Measured on Penn Action against hand-labelled joints:
#
#   calibration set (baseball_pitch, bench_press)   flagged frames 2.78x error
#   held-out set (golf, tennis_serve, squat, pushup)              0.64x
#
# Below 1.0 means the flagged frames were the *better* ones. The 2.78x came
# from tuning and evaluating on the same clips; it did not survive contact
# with actions the threshold had never seen. Per action on the held-out set:
# pushup 0.37x, squat 0.48x, golf 0.75x, tennis_serve 1.08x.
#
# An absolute pixel threshold failed differently and worse — 1.02x — because
# it conflated limb speed with apparent body size. Normalising fixed that
# confound without making the signal predictive.
MAX_TRAVEL_PER_SHOULDER_WIDTH = 0.55

# Fallback when shoulders are not among the readings. Absolute pixels, with
# the caveat above: only sound when every frame is framed alike.
FALLBACK_TRAVEL_PX = 150.0


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
    threshold_px: float
    reported_confidence: float

    def describe(self) -> str:
        return (
            f"{self.name} moved {self.travel_px:.0f}px in one frame "
            f"(limit {self.threshold_px:.0f}px for this subject's size), "
            f"reported at confidence {self.reported_confidence:.2f}"
        )


class ContinuityParams(BaseModel):
    """This tool measures the frame it is given, so it takes no arguments."""


def shoulder_width(readings: list[JointReading]) -> float | None:
    """Distance between the shoulders, the scale everything else is measured in.

    Shoulders because they are the most reliably detected pair on a torso and
    barely change apparent separation as a person turns — unlike hips, which
    foreshorten badly from the side.
    """
    positions = {r.name: r for r in readings}
    left, right = positions.get("L shoulder"), positions.get("R shoulder")
    if left is None or right is None:
        return None

    width = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
    # A near-zero width means the shoulders coincide, which is a detection
    # failure rather than a scale.
    return width if width > 1.0 else None


def implausible_joints(
    previous: list[JointReading],
    current: list[JointReading],
    *,
    limit: float | None = None,
) -> list[Implausible]:
    """Which joints moved further than a body can between two frames.

    Travel is measured in shoulder widths, so the verdict does not change when
    the same movement is filmed closer or further away. See
    :data:`MAX_TRAVEL_PER_SHOULDER_WIDTH` for why that matters — in absolute
    pixels this check picked out no worse frames than it passed.

    Args:
        previous: Readings from the preceding frame.
        current: Readings from the frame under examination.
        limit: Override in pixels. Bypasses normalisation entirely, so use it
            only when every frame is framed identically.

    Returns:
        The implausible readings, largest travel first. Empty means the frame
        is continuous with the one before it — which is not proof it is
        correct, only that this check found nothing.
    """
    before = {reading.name: reading for reading in previous}

    if limit is not None:
        threshold = limit
    else:
        # Scale from the previous frame: the current one may be the broken
        # reading, and a broken reading gives a broken scale.
        scale = shoulder_width(previous) or shoulder_width(current)
        threshold = scale * MAX_TRAVEL_PER_SHOULDER_WIDTH if scale else FALLBACK_TRAVEL_PX

    found: list[Implausible] = []
    for reading in current:
        earlier = before.get(reading.name)
        if earlier is None:
            continue
        travel = ((reading.x - earlier.x) ** 2 + (reading.y - earlier.y) ** 2) ** 0.5
        if travel > threshold:
            found.append(
                Implausible(
                    name=reading.name,
                    travel_px=travel,
                    threshold_px=threshold,
                    reported_confidence=reading.confidence,
                )
            )

    found.sort(key=lambda i: -i.travel_px)
    return found


def continuity_tool(
    previous: list[JointReading],
    current: list[JointReading],
    *,
    limit: float | None = None,
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
