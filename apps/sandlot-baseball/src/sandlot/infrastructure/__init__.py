"""The outside world: detectors, storage, and the Saccade wiring.

This is the only layer that may import MediaPipe, YOLO or OpenCV. The engine
may not import them at all (rule 2, enforced by an AST scan over
``src/saccade``); this application may, because it is the application's job
to bring domain capability and hand it to the engine through
``register_tool()``.

Each module here implements a Port declared in ``application/ports``. The
dependency points inward: infrastructure knows about the port, the port
knows nothing about infrastructure.
"""

from __future__ import annotations

from sandlot.infrastructure.persistence import JsonSessionRepo
from sandlot.infrastructure.saccade_tools import disagreement_tool, pose_measurement_tool
from sandlot.infrastructure.vision import (
    Detected,
    MediaPipePose,
    OpenCVVideo,
    VideoFile,
    YOLODetector,
    file_sha256,
)

__all__ = [
    "Detected",
    "JsonSessionRepo",
    "MediaPipePose",
    "OpenCVVideo",
    "VideoFile",
    "YOLODetector",
    "disagreement_tool",
    "file_sha256",
    "pose_measurement_tool",
]
