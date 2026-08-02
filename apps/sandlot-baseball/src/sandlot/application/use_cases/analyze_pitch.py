"""Measure one pitch.

The flow, and only the flow: decode, detect, measure, store. Every number
comes from the domain — a use case that computed an angle would have put a
rule where the domain's tests cannot reach it (spec 5.1).

A pitch is measured entirely from the body, so ``measure`` here is the shared
body metrics and nothing else. What separates this from ``analyze_swing`` is
that a swing additionally needs the bat located, and the bat is what an
object detector loses exactly when the swing is most interesting.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sandlot.application.ports import PosePort, SessionRepoPort, VideoPort
from sandlot.application.use_cases._body import body_metrics
from sandlot.domain.models import Frame, Metric, Session

__all__ = ["AnalysisFailedError", "analyze_pitch", "measure", "new_session_id"]


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
        id=session_id or new_session_id(decoded.sha256),
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


def measure(frames: Sequence[Frame]) -> list[Metric]:
    """Every pitching metric that could be computed.

    All of them are body metrics. Kept as a named function rather than an
    alias so the two use cases read alike, and so a pitching-specific metric
    added later has an obvious home.
    """
    return body_metrics(frames)


def new_session_id(video_sha: str) -> str:
    """Short, sortable, and tied to the video it describes."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{video_sha[:8]}"
