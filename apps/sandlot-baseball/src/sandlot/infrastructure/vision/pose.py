"""MediaPipe pose detection.

``RunningMode.IMAGE`` is a contract, not a default. VIDEO mode carries state
between frames, so the same frame detected from a different starting point
gives a different answer — and M3's acceptance condition is that ten analyses
of one video agree exactly. The day-zero measurement that showed this
possible was made in IMAGE mode; changing it invalidates that measurement and
every session recorded under it.

Measured on happy/林永閎.MOV, five separate processes:

    mediapipe=1e94edf3ee6360162f634472   identical every time

The versions are pinned in pyproject.toml for the same reason, and recorded
into every session so a later comparison can refuse to subtract across them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from sandlot import __version__
from sandlot.domain.models import Frame, JointReading, Toolchain

__all__ = ["DEFAULT_MODEL", "LANDMARK_INDEX", "MediaPipePose"]

# The default model file. Heavy rather than lite: this runs offline on
# recorded video, so accuracy costs nothing a user waits for.
DEFAULT_MODEL = Path("models/pose_landmarker_heavy.task")

# MediaPipe's 33 landmarks, of which these twelve are what the metrics use.
# Its own numbering — body landmarks start at 11, the first eleven being
# face points this project has no use for.
LANDMARK_INDEX = {
    "L shoulder": 11,
    "R shoulder": 12,
    "L elbow": 13,
    "R elbow": 14,
    "L wrist": 15,
    "R wrist": 16,
    "L hip": 23,
    "R hip": 24,
    "L knee": 25,
    "R knee": 26,
    "L ankle": 27,
    "R ankle": 28,
}


class MediaPipePose:
    """Finds joints with MediaPipe. Implements ``PosePort``.

    Args:
        model_path: The ``.task`` bundle. Defaults to the heavy pose
            landmarker under ``models/``.

    Raises:
        OSError: If the model file is missing. MediaPipe's own error for this
            is a bare RuntimeError from C++ that does not name the path.
    """

    def __init__(self, model_path: Path | str = DEFAULT_MODEL) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise OSError(
                f"pose model not found at {self._model_path}. "
                f"Download pose_landmarker_heavy.task from MediaPipe's model garden."
            )

    @property
    def toolchain(self) -> Toolchain:
        """The versions producing these coordinates.

        Read at runtime rather than hardcoded: a pin in pyproject.toml states
        the intent, and this states what actually got installed. When they
        disagree it is the installed one that moved the numbers.
        """
        from ultralytics import __version__ as ultralytics_version

        return Toolchain(
            mediapipe=mediapipe.__version__,
            ultralytics=ultralytics_version,
            sandlot=__version__,
        )

    def detect(self, images: list[np.ndarray], *, fps: float) -> list[Frame]:
        """One :class:`Frame` per image, in order.

        A frame where no pose was found still appears, with no joints.
        Dropping it would renumber everything after it, and a metric citing
        "frame 47" would point at the wrong picture.

        Args:
            images: Decoded frames, BGR as OpenCV produces them.
            fps: Used only to timestamp the frames.

        Raises:
            ValueError: If ``fps`` is not positive — every timestamp would be
                infinite or negative, and a rate computed from them worse
                than absent.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            # See the module docstring. Not a default — a contract.
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )

        frames: list[Frame] = []
        with PoseLandmarker.create_from_options(options) as landmarker:
            for index, image in enumerate(images):
                frames.append(
                    Frame(
                        index=index,
                        timestamp=index / fps,
                        joints=self._joints(landmarker, image),
                    )
                )
        return frames

    def _joints(self, landmarker: Any, image: np.ndarray) -> tuple[JointReading, ...]:
        """The tracked joints in one image, in pixel coordinates.

        MediaPipe reports normalised coordinates; the metrics work in pixels,
        because a pixel threshold is what tells a collapsed detection from a
        person seen edge-on.
        """
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(
            mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
        )

        if not result.pose_landmarks:
            return ()

        landmarks = result.pose_landmarks[0]
        return tuple(
            JointReading(
                name=name,
                x=float(landmarks[i].x) * width,
                y=float(landmarks[i].y) * height,
                # MediaPipe's visibility, recorded and not trusted: on this
                # project's own data it predicts error at AUROC 0.358, worse
                # than chance. Clamped because the model occasionally reports
                # slightly outside [0, 1].
                confidence=min(1.0, max(0.0, float(landmarks[i].visibility))),
            )
            for name, i in LANDMARK_INDEX.items()
            if i < len(landmarks)
        )
