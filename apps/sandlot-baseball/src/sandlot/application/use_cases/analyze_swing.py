"""Measure one swing.

Same flow as a pitch with one addition: an object detector runs alongside the
pose detector, because a swing's distinguishing metrics need the bat located
and the body cannot supply that.

The bat is also what fails. COCO knows "baseball bat" as a shape, and a bat
at contact speed is a smear — so the swing-plane metric is frequently absent,
and it is absent rather than estimated. A plane fitted through three spurious
detections is a confident number describing nothing, and this project's whole
premise is that a measurement with nothing behind it is worse than silence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sandlot.application.ports import DetectPort, PosePort, SessionRepoPort, VideoPort
from sandlot.application.use_cases._body import body_metrics
from sandlot.application.use_cases.analyze_pitch import AnalysisFailedError, new_session_id
from sandlot.domain.models import Frame, Metric, Session
from sandlot.domain.swing import swing_plane_angle, weight_transfer

__all__ = ["BAT_LABEL", "analyze_swing", "bat_boxes", "measure"]

# COCO's own class name. The detector reports names rather than indices, so
# this survives a change of weights file.
BAT_LABEL = "baseball bat"


def analyze_swing(
    video_path: Any,
    *,
    video: VideoPort,
    pose: PosePort,
    objects: DetectPort | None = None,
    repo: SessionRepoPort | None = None,
    stride: int = 1,
    session_id: str | None = None,
) -> Session:
    """Decode, detect body and bat, measure, and optionally store.

    Args:
        video_path: The file to analyse.
        video: Decoder.
        pose: Joint detector.
        objects: Bat detector. ``None`` measures the body only, which is a
            legitimate result rather than a degraded one — the swing-plane
            metric is simply absent, the same as when the detector ran and
            found no bat.
        repo: Where to store the result. ``None`` measures without saving.
        stride: Analyse every Nth frame.
        session_id: Override the generated id.

    Raises:
        AnalysisFailedError: If the video decoded to no frames.
    """
    decoded = video.read(video_path, stride=stride)
    if not decoded.images:
        raise AnalysisFailedError(f"{video_path} decoded to no frames")

    frames = pose.detect(decoded.images, fps=decoded.fps)
    boxes = bat_boxes(objects.detect(decoded.images)) if objects is not None else None

    session = Session(
        id=session_id or new_session_id(decoded.sha256),
        created_at=datetime.now(UTC),
        video_sha256=decoded.sha256,
        frame_count=len(frames),
        fps=decoded.fps,
        toolchain=pose.toolchain,
        metrics=tuple(measure(frames, bat=boxes)),
    )

    if repo is not None:
        repo.save(session)
    return session


def bat_boxes(
    detections: Sequence[Sequence[Any]],
) -> list[tuple[float, float, float, float] | None]:
    """The bat's box per frame, ``None`` where it was not found.

    When a frame holds several bat detections the most confident wins. A
    swing has one bat in it, so the others are the detector finding
    bat-shaped background, and averaging them would place the bat between a
    real one and a fencepost.

    The list keeps one entry per frame including the misses, so the caller
    can see how much of the swing was tracked rather than only what was.
    """
    boxes: list[tuple[float, float, float, float] | None] = []
    for frame_detections in detections:
        bats = [d for d in frame_detections if d.label == BAT_LABEL]
        if not bats:
            boxes.append(None)
            continue
        best = max(bats, key=lambda d: d.confidence)
        boxes.append(best.bbox)
    return boxes


def measure(
    frames: Sequence[Frame],
    *,
    bat: Sequence[tuple[float, float, float, float] | None] | None = None,
) -> list[Metric]:
    """Every hitting metric that could be computed.

    The body metrics are shared with pitching by import, not by copy — two
    files computing the same angle drift apart the first time one of them is
    corrected.
    """
    metrics = body_metrics(frames)

    transfer = weight_transfer(frames)
    if transfer is not None:
        metrics.append(
            Metric(
                name="weight_transfer",
                value=transfer,
                unit="torso lengths",
                frames=tuple(f.index for f in frames),
                detail={
                    "taken_at": "smoothed centre of mass, first to last measurable frame",
                    "why": (
                        "bat speed explains almost none of exit velocity (R-squared 0.097); "
                        "weight transfer explains a further 37.8%"
                    ),
                },
            )
        )

    if bat is not None:
        located = [i for i, box in enumerate(bat) if box is not None]
        angle = swing_plane_angle(bat)
        if angle is not None:
            metrics.append(
                Metric(
                    name="swing_plane_angle",
                    value=angle,
                    unit="degrees",
                    # Every frame the bat was actually seen in. An auditor
                    # checking this needs to know which pictures it came
                    # from, and how few they were.
                    frames=tuple(frames[i].index for i in located if i < len(frames)),
                    detail={
                        "taken_at": "first to last located bat position",
                        "frames_with_bat": len(located),
                        "frames_total": len(bat),
                    },
                )
            )

    return metrics
