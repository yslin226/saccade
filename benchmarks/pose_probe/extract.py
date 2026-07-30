"""Running MediaPipe over a video and keeping what it reported.

Separate from continuity.py so the plausibility check stays testable without
a model file, a video, or MediaPipe installed.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.pose_probe.continuity import JointReading, implausible_joints

__all__ = ["FramePose", "extract_poses", "load_poses", "suspect_frames"]

# The joints pitching mechanics depend on: shoulder-hip separation, elbow
# valgus, stride. Face and hand landmarks are irrelevant here and only add
# noise to a continuity check.
WATCHED = {
    11: "L shoulder",
    12: "R shoulder",
    13: "L elbow",
    14: "R elbow",
    15: "L wrist",
    16: "R wrist",
    23: "L hip",
    24: "R hip",
    25: "L knee",
    26: "R knee",
}


@dataclass
class FramePose:
    """What the detector reported for one frame."""

    frame: int
    detected: bool
    joints: list[JointReading]


def extract_poses(video: str | Path, model: str | Path) -> list[FramePose]:
    """Run pose detection over every frame of a video.

    Imports MediaPipe lazily so the rest of this package stays usable without
    it — the continuity check is arithmetic and needs no model.
    """
    import cv2
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses: list[FramePose] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                Image(image_format=ImageFormat.SRGB, data=rgb),
                int(index * 1000 / fps),
            )

            if not result.pose_landmarks:
                poses.append(FramePose(frame=index, detected=False, joints=[]))
            else:
                marks = result.pose_landmarks[0]
                poses.append(
                    FramePose(
                        frame=index,
                        detected=True,
                        joints=[
                            JointReading(
                                name=name,
                                x=float(marks[i].x) * width,
                                y=float(marks[i].y) * height,
                                confidence=float(marks[i].visibility),
                            )
                            for i, name in WATCHED.items()
                        ],
                    )
                )
            index += 1

    capture.release()
    return poses


def suspect_frames(
    poses: list[FramePose], *, limit: float | None = None
) -> list[tuple[FramePose, FramePose, list]]:
    """Frames whose pose could not physically follow the frame before.

    Returns:
        ``(previous, current, implausible)`` for each suspect frame.
    """
    from benchmarks.pose_probe.continuity import MAX_JOINT_TRAVEL_PX

    threshold = MAX_JOINT_TRAVEL_PX if limit is None else limit
    found = []
    for previous, current in itertools.pairwise(poses):
        if not (previous.detected and current.detected):
            continue
        if current.frame - previous.frame != 1:
            continue
        bad = implausible_joints(previous.joints, current.joints, limit=threshold)
        if bad:
            found.append((previous, current, bad))
    return found


def save_poses(poses: list[FramePose], path: str | Path) -> None:
    """Cache the detector's output so a rerun needs no MediaPipe."""
    Path(path).write_text(json.dumps([asdict(p) for p in poses]), encoding="utf-8")


def load_poses(path: str | Path) -> list[FramePose]:
    """Read cached detector output."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        FramePose(
            frame=entry["frame"],
            detected=entry["detected"],
            joints=[JointReading(**j) for j in entry["joints"]],
        )
        for entry in raw
    ]
