"""Two detectors disagreeing is what predicts a bad pose estimate.

Four attempts to predict it from a single detector's own geometry failed:
travel 0.64x on held-out actions, bone length 1.00x, absolute pixel travel
1.02x, and a nine-feature classifier at AUROC 0.42 — worse than chance. The
information is not in one detector's output.

It is in the gap between two. MediaPipe and YOLO11-pose are trained
separately, on different data, with different architectures, so their errors
have no reason to coincide. Where they agree both are probably right; where
they diverge at least one is wrong, and nothing here needs to know which.

Measured on 1320 frames of four actions never used for any tuning:

    mean_gap     AUROC 0.713    p = 7e-27
    max_gap            0.710
    median_gap         0.712
    gap_spread         0.709

    golf_swing         0.951        pushup    0.845
    squat              0.845        tennis    0.659

The bar was fixed at 0.70 before the run, in this docstring, because the best
single-detector signal reached 0.607 and anything less would not justify a
second model in the pipeline.

No training, no fitted parameters: mean_gap is the average distance between
where the two detectors put the same joint, in shoulder widths. What it still
cannot say is *which* detector is wrong, or why — that is what the agent is
for.

Both detectors live in benchmarks/, never in src/saccade — rule 2.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel

from benchmarks.pose_probe.continuity import JointReading, shoulder_width
from saccade.tools import Tool, ToolResult

__all__ = [
    "COMMON_JOINTS",
    "SUCCESS_AUROC",
    "SUSPECT_MEAN_GAP",
    "DisagreementParams",
    "disagreement_features",
    "disagreement_tool",
    "to_readings",
]

SUCCESS_AUROC = 0.70

# Above this mean gap, treat the frame as suspect.
#
# 0.10 shoulder widths — about 4cm on an adult. Chosen as the point where the
# measured AUROC curve turns: below it the two detectors are within their own
# localisation noise of each other, above it they are describing different
# poses. Flagging the top 20% by this figure catches frames whose true error
# is markedly worse, but the threshold is a convenience for callers who need a
# yes or no. The continuous value is the better signal and is always reported.
SUSPECT_MEAN_GAP = 0.10

# The joints both detectors report, by their own indices.
#
# COCO order for YOLO; MediaPipe's own numbering. Wrists, elbows, shoulders,
# hips and knees are what pitching mechanics depend on, and they are the joints
# both models claim to localise.
COMMON_JOINTS = {
    "L shoulder": (11, 5),
    "R shoulder": (12, 6),
    "L elbow": (13, 7),
    "R elbow": (14, 8),
    "L wrist": (15, 9),
    "R wrist": (16, 10),
    "L hip": (23, 11),
    "R hip": (24, 12),
    "L knee": (25, 13),
    "R knee": (26, 14),
}

DISAGREEMENT_FEATURES = (
    "max_gap",  # furthest apart the two detectors put any joint
    "mean_gap",  # and the average across joints
    "median_gap",  # robust to one joint being wild
    "gap_spread",  # do they differ on one joint or on all of them
    "n_large_gaps",  # how many joints differ by more than a tenth of a body
)


@dataclass(frozen=True)
class Disagreement:
    """How far apart two detectors placed the same joints in one frame."""

    per_joint: dict[str, float]  # gap in shoulder widths

    @property
    def worst_joint(self) -> str | None:
        return max(self.per_joint, key=lambda k: self.per_joint[k]) if self.per_joint else None

    def features(self) -> dict[str, float]:
        gaps = list(self.per_joint.values())
        if not gaps:
            return dict.fromkeys(DISAGREEMENT_FEATURES, 0.0)

        return {
            "max_gap": max(gaps),
            "mean_gap": statistics.mean(gaps),
            "median_gap": statistics.median(gaps),
            "gap_spread": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
            "n_large_gaps": float(sum(1 for g in gaps if g > 0.1)),
        }


def to_readings(
    points: dict[str, tuple[float, float, float]],
) -> list[JointReading]:
    """Turn a detector's output into the common reading format."""
    return [
        JointReading(name=name, x=x, y=y, confidence=confidence)
        for name, (x, y, confidence) in points.items()
    ]


def disagreement_features(
    first: list[JointReading],
    second: list[JointReading],
    *,
    scale: float | None = None,
) -> Disagreement:
    """Measure how far apart two detectors placed each joint.

    Gaps are in shoulder widths rather than pixels, for the reason the earlier
    attempt learned the hard way: an absolute distance conflates disagreement
    with apparent body size.

    Args:
        first: One detector's readings.
        second: The other's.
        scale: Shoulder width to normalise by. Taken from the first detector
            when omitted.
    """
    by_name_first = {r.name: r for r in first}
    by_name_second = {r.name: r for r in second}

    width = scale or shoulder_width(first) or shoulder_width(second)
    if not width or width <= 0:
        return Disagreement(per_joint={})

    gaps: dict[str, float] = {}
    for name in set(by_name_first) & set(by_name_second):
        a, b = by_name_first[name], by_name_second[name]
        gaps[name] = float(np.hypot(a.x - b.x, a.y - b.y)) / width

    return Disagreement(per_joint=gaps)


class DisagreementParams(BaseModel):
    """This tool compares readings it was given, so it takes no arguments."""


def disagreement_tool(
    first: list[JointReading],
    second: list[JointReading],
    *,
    threshold: float = SUSPECT_MEAN_GAP,
) -> Tool:
    """Build a tool reporting whether two detectors agree about this frame.

    A measurement, not an opinion: it is arithmetic on two sets of coordinates.
    Validated at AUROC 0.713 over 1320 held-out frames, which makes it the
    first signal in this project that predicts pose error better than chance.

    What it cannot say is which detector is wrong, or whether the cause is
    motion blur, occlusion or a lost track — those need different handling and
    look identical in the numbers. That gap is the agent's to fill.
    """

    def run(image: Image, viewport: object) -> ToolResult:
        gaps = disagreement_features(first, second)
        if not gaps.per_joint:
            return ToolResult(
                value={
                    "method": "detector_disagreement",
                    "note": "the two detectors share no joints in this frame",
                },
                is_measurement=False,
            )

        features = gaps.features()
        agree = features["mean_gap"] <= threshold

        return ToolResult(
            value={
                "method": "detector_disagreement",
                "detectors_agree": agree,
                "mean_gap": round(features["mean_gap"], 3),
                "max_gap": round(features["max_gap"], 3),
                "worst_joint": gaps.worst_joint,
                "threshold": threshold,
                "units": "shoulder widths",
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
        params_schema=DisagreementParams,
    )
