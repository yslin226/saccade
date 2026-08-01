"""Measure one video.

The flow, and only the flow: decode, detect, measure, store. Every number
comes from ``domain.kinematics`` — a use case that computed an angle would
have put a rule where the domain's tests cannot reach it (spec 5.1).

Which frame each metric is taken from is itself a decision, and it is made
here because it is about the *procedure*, not about the mechanics: the
per-frame rules live in the domain, and this picks which frame to apply them
to. Two choices, both stated in the metric's evidence:

- Angles are taken at their extreme over the delivery. The interesting
  hip-shoulder separation is the largest one, not the one in whichever frame
  happened to be sampled.
- Stride is taken at the same frame as peak separation, so the two describe
  the same instant rather than two unrelated moments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sandlot.application.ports import PosePort, SessionRepoPort, VideoPort
from sandlot.domain.kinematics import (
    elbow_valgus,
    hip_shoulder_separation,
    kinetic_chain_order,
    stride_length,
)
from sandlot.domain.models import Frame, Metric, Session

__all__ = ["AnalysisFailedError", "analyze_pitch", "measure"]


class AnalysisFailedError(RuntimeError):
    """Raised when a video yielded nothing measurable.

    Distinct from a session with no metrics, which is a legitimate result:
    that means the detector ran and found no usable joints. This means the
    video could not be read at all.
    """


def analyze_pitch(
    video_path: Any,
    *,
    video: VideoPort,
    pose: PosePort,
    repo: SessionRepoPort | None = None,
    stride: int = 1,
    session_id: str | None = None,
) -> Session:
    """Decode, detect, measure, and optionally store.

    Args:
        video_path: The file to analyse.
        video: Decoder.
        pose: Joint detector.
        repo: Where to store the result. ``None`` measures without saving,
            which is what a determinism check wants — ten runs that each
            wrote a session would leave nine to clean up.
        stride: Analyse every Nth frame.
        session_id: Override the generated id. Present so a caller can make
            the whole result reproducible; the default derives from the
            video's hash and the current time, and the time is what stops
            two analyses of the same file from colliding.

    Raises:
        AnalysisFailedError: If the video decoded to no frames.
    """
    decoded = video.read(video_path, stride=stride)
    if not decoded.images:
        raise AnalysisFailedError(f"{video_path} decoded to no frames")

    frames = pose.detect(decoded.images, fps=decoded.fps)

    session = Session(
        id=session_id or _make_id(decoded.sha256),
        created_at=datetime.now(UTC),
        video_sha256=decoded.sha256,
        frame_count=len(frames),
        fps=decoded.fps,
        toolchain=pose.toolchain,
        metrics=tuple(measure(frames)),
    )

    if repo is not None:
        repo.save(session)
    return session


def measure(frames: list[Frame]) -> list[Metric]:
    """Every metric that could be computed, each citing its frames.

    Separated from :func:`analyze_pitch` so it can be tested against
    hand-written frames without a decoder or a detector in the way.

    A metric that could not be computed anywhere is absent rather than
    present with a null: rule 8 says a claim needs evidence, and "we looked
    and found nothing" is the absence of a claim.
    """
    metrics: list[Metric] = []

    peak_separation = _peak(frames, hip_shoulder_separation)
    if peak_separation is not None:
        index, value = peak_separation
        metrics.append(
            Metric(
                name="hip_shoulder_separation",
                value=value,
                unit="degrees",
                frames=(frames[index].index,),
                detail={"taken_at": "maximum over the delivery"},
            )
        )

        # Deliberately the same frame: a stride measured at one instant and a
        # separation at another describe two different moments of the
        # delivery, and reporting them together implies they do not.
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
        peak = _peak(frames, lambda f, s=side: elbow_valgus(f, side=s), most=min)
        if peak is None:
            continue
        index, value = peak
        metrics.append(
            Metric(
                name=f"elbow_flexion_{side}",
                value=value,
                unit="degrees",
                frames=(frames[index].index,),
                detail={"taken_at": "most flexed over the delivery", "side": side},
            )
        )

    order = kinetic_chain_order(frames)
    if order is not None:
        # An ordering is not a number, so it rides in detail with a value
        # that says how much of the expected sequence survived — 1.0 for
        # ground-up, lower as segments fire out of turn.
        metrics.append(
            Metric(
                name="kinetic_chain_order",
                value=_sequence_score(order),
                unit="fraction in order",
                frames=tuple(f.index for f in frames[:1]),
                detail={"order": order, "expected": ["hips", "shoulders", "elbow", "wrist"]},
            )
        )

    return metrics


def _peak(
    frames: list[Frame],
    rule: Any,
    *,
    most: Any = max,
) -> tuple[int, float] | None:
    """Where a per-frame rule reached its extreme, and what it read there.

    Returns the *list* index alongside the value so the caller can reach the
    frame; the frame's own number goes into the metric's evidence.
    """
    measured = [(i, rule(frame)) for i, frame in enumerate(frames)]
    usable = [(i, value) for i, value in measured if value is not None]
    return most(usable, key=lambda pair: pair[1]) if usable else None


def _sequence_score(order: list[str]) -> float:
    """How much of the ground-up sequence the delivery kept.

    The fraction of adjacent pairs that are in the expected order. 1.0 is
    textbook; 0.0 is exactly backwards. A single number rather than a verdict
    because "your chain is broken" is a judgement and this is arithmetic.
    """
    expected = ["hips", "shoulders", "elbow", "wrist"]
    ranks = {name: i for i, name in enumerate(expected)}

    pairs = [
        (order[i], order[i + 1])
        for i in range(len(order) - 1)
        if order[i] in ranks and order[i + 1] in ranks
    ]
    if not pairs:
        return 0.0

    forward = sum(1 for a, b in pairs if ranks[a] < ranks[b])
    return forward / len(pairs)


def _make_id(video_sha: str) -> str:
    """Short, sortable, and tied to the video it describes."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{video_sha[:8]}"
