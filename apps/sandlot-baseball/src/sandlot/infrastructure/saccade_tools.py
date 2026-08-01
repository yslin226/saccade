"""Wrapping detector output as Saccade tools.

This is the door rule 2 provides. The engine may not import MediaPipe or YOLO
— an AST scan over ``src/saccade`` fails the build if it does — so domain
capability arrives through ``register_tool()`` instead, and the engine stays
usable by an application that has never heard of baseball.

Two tools, and the distinction between them is the point:

``pose_measurement`` is a measurement: it reports an angle computed from
coordinates, and ``is_measurement=True`` lets it overrule the model. If the
VLM says the elbow is extended and the geometry says 94 degrees, the geometry
wins.

``detector_disagreement`` is also a measurement, but of a different thing —
whether two independently trained detectors put the joints in the same place.
It answers "can this frame be trusted", not "what does this frame show". At
AUROC 0.638 over 890 held-out frames it is a weak signal and is labelled as
one; four single-detector geometric signals did worse, three of them worse
than chance.

What neither can say is *why* two detectors disagree. Blur, occlusion and a
lost track are identical in the numbers and need different handling, and that
is the question a VLM is for — see ``benchmarks/pose_probe/explain.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from saccade.tools import Tool, ToolResult
from sandlot.domain.kinematics import elbow_valgus, hip_shoulder_separation
from sandlot.domain.models import Frame

__all__ = [
    "PoseParams",
    "disagreement_tool",
    "pose_measurement_tool",
]


class PoseParams(BaseModel):
    """These tools measure a frame they were given, so they take no arguments."""


def pose_measurement_tool(frame: Frame) -> Tool:
    """A tool reporting the joint angles in one frame.

    The verdict is ``elbow_extended`` rather than a raw angle, because a
    boolean is what the verifier can confront a statement with — the angle
    itself rides along as context. ``answer_key`` says which is which, so a
    diagnostic figure never gets checked against the answer.
    """

    def run(image: Any, viewport: Any) -> ToolResult:
        separation = hip_shoulder_separation(frame)
        right = elbow_valgus(frame, side="R")

        if right is None:
            return ToolResult(
                value={
                    "method": "pose_geometry",
                    "note": "the arm was not fully detected in this frame",
                },
                is_measurement=False,
            )

        return ToolResult(
            value={
                "method": "pose_geometry",
                "elbow_extended": right > 160.0,
                "elbow_angle_degrees": round(right, 1),
                "hip_shoulder_separation_degrees": (
                    None if separation is None else round(separation, 1)
                ),
                "frame": frame.index,
            },
            is_measurement=True,
            answer_key="elbow_extended",
        )

    return Tool(
        name="pose_geometry",
        description="Measure the joint angles of the person in this frame",
        fn=run,
        params_schema=PoseParams,
    )


def disagreement_tool(first: Frame, second: Frame, *, threshold: float = 0.10) -> Tool:
    """A tool reporting whether two detectors placed the joints alike.

    Gaps are in torso lengths rather than pixels, for the reason an earlier
    attempt learned the hard way: an absolute distance conflates disagreement
    with how far away the subject was standing.

    Args:
        first: One detector's reading of a frame.
        second: The other's reading of the same frame.
        threshold: Mean gap above which the frame is called suspect. The
            continuous figure is the better signal and is always reported;
            this exists for callers who need a yes or no.
    """

    def run(image: Any, viewport: Any) -> ToolResult:
        gaps = _gaps(first, second)
        if not gaps:
            return ToolResult(
                value={
                    "method": "detector_disagreement",
                    "note": "the two detectors share no usable joint in this frame",
                },
                is_measurement=False,
            )

        mean_gap = sum(gaps.values()) / len(gaps)
        worst = max(gaps, key=lambda name: gaps[name])

        return ToolResult(
            value={
                "method": "detector_disagreement",
                "detectors_agree": mean_gap <= threshold,
                "mean_gap": round(mean_gap, 3),
                "max_gap": round(gaps[worst], 3),
                "worst_joint": worst,
                "units": "torso lengths",
                "frame": first.index,
            },
            is_measurement=True,
            answer_key="detectors_agree",
        )

    return Tool(
        name="detector_disagreement",
        description=(
            "Check whether two independent pose detectors placed the joints in the same place"
        ),
        fn=run,
        params_schema=PoseParams,
    )


def _gaps(first: Frame, second: Frame, *, min_confidence: float = 0.30) -> dict[str, float]:
    """How far apart the two detectors put each joint, in torso lengths.

    Joints either detector reports below ``min_confidence`` are skipped: a
    keypoint at 0.08 is a guess about where a limb might be, and the distance
    between two guesses says nothing about either.
    """
    from saccade.geometry import distance
    from sandlot.domain.kinematics import torso_length

    scale = torso_length(first) or torso_length(second)
    if not scale:
        return {}

    by_name = {reading.name: reading for reading in second.joints}

    gaps: dict[str, float] = {}
    for reading in first.joints:
        other = by_name.get(reading.name)
        if other is None:
            continue
        if reading.confidence < min_confidence or other.confidence < min_confidence:
            continue
        gaps[reading.name] = distance((reading.x, reading.y), (other.x, other.y)) / scale

    return gaps
