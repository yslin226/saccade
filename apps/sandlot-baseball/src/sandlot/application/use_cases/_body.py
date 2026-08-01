"""Internal: the metrics a pitch and a swing both have.

Hip-shoulder separation, elbow flexion, stride and chain order are properties
of a body moving, not of what it is holding — a hitter coils and a pitcher
coils, and the angle between the two lines is the same angle. Only the
bat-dependent metrics differ, so those live with their own use case and this
is imported by both.

Shared by import rather than by copy. Two files computing the same angle drift
apart the first time one of them is corrected.

Which frame a per-frame rule is applied to is decided here rather than in the
domain, because it is a procedural question: the domain says what the angle is
in a frame, this says which frame is the interesting one.
"""

from __future__ import annotations

from typing import Any

from sandlot.domain.kinematics import (
    elbow_valgus,
    hip_shoulder_separation,
    kinetic_chain_order,
    stride_length,
)
from sandlot.domain.models import Frame, Metric

__all__ = ["EXPECTED_CHAIN", "body_metrics", "peak", "sequence_score"]

# Ground-up: each segment hands energy to the next. Out of order is what
# coaches mean by "all arm".
EXPECTED_CHAIN = ("hips", "shoulders", "elbow", "wrist")


def body_metrics(frames: list[Frame]) -> list[Metric]:
    """Every body metric that could be computed, each citing its frames.

    A metric that could not be computed anywhere is absent rather than
    present with a null: rule 8 says a claim needs evidence, and "we looked
    and found nothing" is the absence of a claim.
    """
    metrics: list[Metric] = []

    peak_separation = peak(frames, hip_shoulder_separation)
    if peak_separation is not None:
        index, value = peak_separation
        metrics.append(
            Metric(
                name="hip_shoulder_separation",
                value=value,
                unit="degrees",
                frames=(frames[index].index,),
                detail={"taken_at": "maximum over the movement"},
            )
        )

        # Deliberately the same frame: a stride measured at one instant and a
        # separation at another describe two different moments, and reporting
        # them together implies they do not.
        stride = stride_length(frames[index])
        if stride is not None:
            metrics.append(
                Metric(
                    name="stride_length",
                    value=stride,
                    unit="torso lengths",
                    frames=(frames[index].index,),
                    detail={"taken_at": "frame of peak hip-shoulder separation"},
                )
            )

    for side in ("L", "R"):
        flexed = peak(frames, lambda f, s=side: elbow_valgus(f, side=s), most=min)
        if flexed is None:
            continue
        index, value = flexed
        metrics.append(
            Metric(
                name=f"elbow_flexion_{side}",
                value=value,
                unit="degrees",
                frames=(frames[index].index,),
                detail={"taken_at": "most flexed over the movement", "side": side},
            )
        )

    order = kinetic_chain_order(frames)
    if order is not None:
        # An ordering is not a number, so it rides in detail with a value
        # saying how much of the expected sequence survived.
        metrics.append(
            Metric(
                name="kinetic_chain_order",
                value=sequence_score(order),
                unit="fraction in order",
                frames=tuple(f.index for f in frames[:1]),
                detail={"order": order, "expected": list(EXPECTED_CHAIN)},
            )
        )

    return metrics


def peak(frames: list[Frame], rule: Any, *, most: Any = max) -> tuple[int, float] | None:
    """Where a per-frame rule reached its extreme, and what it read there.

    Returns the *list* index alongside the value so the caller can reach the
    frame; the frame's own number goes into the metric's evidence.
    """
    measured = [(i, rule(frame)) for i, frame in enumerate(frames)]
    usable = [(i, value) for i, value in measured if value is not None]
    return most(usable, key=lambda pair: pair[1]) if usable else None


def sequence_score(order: list[str]) -> float:
    """How much of the ground-up sequence the movement kept.

    The fraction of adjacent pairs in the expected order. 1.0 is textbook;
    0.0 is exactly backwards. A single number rather than a verdict because
    "your chain is broken" is a judgement and this is arithmetic.
    """
    ranks = {name: i for i, name in enumerate(EXPECTED_CHAIN)}

    pairs = [
        (order[i], order[i + 1])
        for i in range(len(order) - 1)
        if order[i] in ranks and order[i + 1] in ranks
    ]
    if not pairs:
        return 0.0

    forward = sum(1 for a, b in pairs if ranks[a] < ranks[b])
    return forward / len(pairs)
