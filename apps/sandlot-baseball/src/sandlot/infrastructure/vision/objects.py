"""YOLO object detection — the bat and the ball.

COCO's own classes, no training: "baseball bat" and "sports ball" are both in
the standard 80. That is why this works offline on a phone clip, and also why
it will miss a bat mid-swing — a motion-blurred bat looks like nothing COCO
was shown.

Missing the bat is expected and is reported as an empty list rather than a
guess. A swing-plane angle fitted to a bat the detector never found would be
a number with no measurement behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BASEBALL_CLASSES", "DEFAULT_WEIGHTS", "Detected", "YOLODetector"]

# COCO class names this application cares about. Names rather than indices:
# the index depends on the weights file, the name does not.
BASEBALL_CLASSES = frozenset({"baseball bat", "baseball glove", "sports ball", "person"})

DEFAULT_WEIGHTS = "yolo11n-pose.pt"


@dataclass(frozen=True)
class Detected:
    """One object found in one frame. Implements ``Detection``."""

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float


class YOLODetector:
    """Finds objects with YOLO. Implements ``DetectPort``.

    Args:
        weights: Model weights. The default is the pose model already in the
            repo, which also carries COCO detection heads.
        min_confidence: Detections below this are dropped. A bat detected at
            0.05 confidence is the model finding a bat-shaped smear, and a
            swing plane fitted through those is worse than no swing plane.
    """

    def __init__(self, weights: str = DEFAULT_WEIGHTS, *, min_confidence: float = 0.25) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        self._weights = weights
        self._min_confidence = min_confidence
        self._model: Any | None = None

    def detect(self, images: list[np.ndarray]) -> list[list[Detected]]:
        """Detections per image, in order.

        An empty list for a frame where nothing was found, so indices keep
        lining up with the video — the same reason the pose detector emits a
        frame with no joints rather than skipping it.
        """
        if not images:
            return []

        model = self._load()
        out: list[list[Detected]] = []

        for image in images:
            # verbose=False: ultralytics prints a line per frame otherwise,
            # which buries everything else in a thousand-frame clip.
            result = model(image, verbose=False)[0]
            out.append(self._read(result))

        return out

    def _load(self) -> Any:
        """Load once. Constructing a YOLO per frame would dominate the
        runtime and, worse, re-read the weights from disk each time."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._weights)
        return self._model

    def _read(self, result: Any) -> list[Detected]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names
        found: list[Detected] = []

        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < self._min_confidence:
                continue

            label = names[int(box.cls[0])]
            if label not in BASEBALL_CLASSES:
                continue

            # ultralytics reports xyxy; the ports and saccade.models.BBox
            # both use (x, y, width, height) with x/y at the top-left.
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            found.append(
                Detected(
                    label=label,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=confidence,
                )
            )

        return found
