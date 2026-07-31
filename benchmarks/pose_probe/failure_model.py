"""Learning to predict when a pose estimate is wrong.

Three hand-set geometric thresholds failed at this — travel 0.64x on held-out
actions, bone length 1.00x, absolute pixel travel 1.02x. Each asked whether
one number crossed one line.

The approach here is different, and it is not mine. Schneider et al.
(arXiv:2603.02881, a March 2026 preprint, not peer reviewed) detect pose
failures in robotic grasping by feeding several weak alignment signals into a
small classifier rather than thresholding any one of them, reporting 80.5%
detection accuracy on real scenes. Their setting is rigid objects with depth
data; a human body has far more freedom, so the method may simply not carry
over. That is what this measures.

The features are the same quantities the failed thresholds used. The change
is that a model learns how to weigh them together, from labelled examples of
which frames were actually wrong.

Ground truth is Penn Action's hand-labelled joints (Zhang et al., ICCV 2013).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np

from benchmarks.pose_probe.continuity import JointReading, shoulder_width

__all__ = ["FEATURE_NAMES", "FrameFeatures", "extract_features", "label_frames"]

# Bones whose length anatomy fixes, used to spot a joint flying loose.
BONES = (
    ("L shoulder", "L elbow"),
    ("L elbow", "L wrist"),
    ("R shoulder", "R elbow"),
    ("R elbow", "R wrist"),
    ("L hip", "L knee"),
    ("R hip", "R knee"),
    ("L shoulder", "R shoulder"),
    ("L hip", "R hip"),
)

# Limb pairs that should mirror each other in length.
SYMMETRIC = (
    (("L shoulder", "L elbow"), ("R shoulder", "R elbow")),
    (("L elbow", "L wrist"), ("R elbow", "R wrist")),
    (("L hip", "L knee"), ("R hip", "R knee")),
)

FEATURE_NAMES = (
    "travel",  # how far the fastest joint moved since the last frame
    "mean_travel",  # how far joints moved on average
    "jerk",  # second difference: implausible acceleration
    "bone_dev",  # worst bone length deviation from this clip's median
    "mean_bone_dev",  # average deviation, so one bad bone does not dominate
    "asymmetry",  # left limb length against right
    "min_confidence",  # the detector's least confident joint
    "mean_confidence",  # and its average
    "confidence_spread",  # disagreement within its own scores
)


@dataclass
class FrameFeatures:
    """Signals for one frame, plus the error it actually had."""

    clip: str
    action: str
    frame: int
    values: dict[str, float]
    error: float  # mean joint error against the labels, in shoulder widths

    def vector(self) -> list[float]:
        return [self.values[name] for name in FEATURE_NAMES]


@dataclass
class ClipReadings:
    """A clip's detections and the labelled truth to score them against."""

    clip: str
    action: str
    readings: list[list[JointReading] | None] = field(default_factory=list)
    errors: list[float | None] = field(default_factory=list)


def _distance(a: JointReading, b: JointReading) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _bone_lengths(readings: list[JointReading]) -> dict[str, float]:
    positions = {r.name: r for r in readings}
    lengths: dict[str, float] = {}
    for first, second in BONES:
        a, b = positions.get(first), positions.get(second)
        if a is not None and b is not None:
            lengths[f"{first}|{second}"] = _distance(a, b)
    return lengths


def extract_features(clip: ClipReadings) -> list[FrameFeatures]:
    """Turn a clip's detections into per-frame feature vectors.

    Everything scale-dependent is divided by shoulder width, so a close-up and
    a wide shot of the same movement produce the same numbers. Skipping that
    step is what made the first attempt measure apparent body size instead of
    error.
    """
    present = [r for r in clip.readings if r is not None]
    if len(present) < 5:
        return []

    scale = statistics.median(
        [w for w in (shoulder_width(r) for r in present) if w is not None] or [1.0]
    )
    if scale <= 0:
        return []

    # A clip's own median bone length is the reference: people differ, and an
    # absolute expectation would just measure who is in frame.
    all_bones = [_bone_lengths(r) for r in present]
    median_bone = {
        key: statistics.median([b[key] for b in all_bones if key in b]) for key in all_bones[0]
    }

    out: list[FrameFeatures] = []
    for index, current in enumerate(clip.readings):
        error = clip.errors[index] if index < len(clip.errors) else None
        if current is None or error is None:
            continue

        previous = clip.readings[index - 1] if index >= 1 else None
        before_that = clip.readings[index - 2] if index >= 2 else None
        if previous is None:
            continue

        by_name = {r.name: r for r in current}
        prev_by_name = {r.name: r for r in previous}
        shared = set(by_name) & set(prev_by_name)
        if not shared:
            continue

        travels = [_distance(by_name[n], prev_by_name[n]) / scale for n in shared]

        jerk = 0.0
        if before_that is not None:
            older = {r.name: r for r in before_that}
            triples = shared & set(older)
            if triples:
                jerk = max(
                    (
                        abs(by_name[n].x - 2 * prev_by_name[n].x + older[n].x)
                        + abs(by_name[n].y - 2 * prev_by_name[n].y + older[n].y)
                    )
                    / scale
                    for n in triples
                )

        bones = _bone_lengths(current)
        deviations = [
            abs(bones[k] - median_bone[k]) / max(median_bone[k], 1.0)
            for k in bones
            if k in median_bone and median_bone[k] > 0
        ]

        asymmetries = []
        for left, right in SYMMETRIC:
            left_key, right_key = f"{left[0]}|{left[1]}", f"{right[0]}|{right[1]}"
            if left_key in bones and right_key in bones:
                longer = max(bones[left_key], bones[right_key], 1.0)
                asymmetries.append(abs(bones[left_key] - bones[right_key]) / longer)

        confidences = [r.confidence for r in current]

        out.append(
            FrameFeatures(
                clip=clip.clip,
                action=clip.action,
                frame=index,
                values={
                    "travel": max(travels),
                    "mean_travel": statistics.mean(travels),
                    "jerk": jerk,
                    "bone_dev": max(deviations) if deviations else 0.0,
                    "mean_bone_dev": statistics.mean(deviations) if deviations else 0.0,
                    "asymmetry": max(asymmetries) if asymmetries else 0.0,
                    "min_confidence": min(confidences),
                    "mean_confidence": statistics.mean(confidences),
                    "confidence_spread": (
                        statistics.pstdev(confidences) if len(confidences) > 1 else 0.0
                    ),
                },
                error=error / scale,
            )
        )

    return out


def label_frames(frames: list[FrameFeatures], *, percentile: float = 80.0) -> np.ndarray:
    """Which frames count as failures: the worst ``percentile`` by true error.

    A relative cut rather than an absolute one, because "wrong" has no
    universal pixel value — it depends on how large the subject appears and on
    what the measurement will be used for.
    """
    errors = np.array([f.error for f in frames])
    return errors >= np.percentile(errors, percentile)
