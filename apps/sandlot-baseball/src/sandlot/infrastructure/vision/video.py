"""Decoding video into frames, once.

Decoding happens here and nowhere else, and the frames are handed on as
arrays. That is not a convenience — it is the first half of the contract the
day-zero determinism measurement depended on. Re-decoding inside each
detection pass would put the decoder into the loop being measured, and a
decoder that drifts looks exactly like a detector that drifts.

The other half lives in :mod:`sandlot.infrastructure.vision.pose`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["OpenCVVideo", "VideoFile", "file_sha256"]

# Read the file in chunks rather than whole. A phone clip is tens of
# megabytes and a longer session could be hundreds; hashing should not depend
# on the file fitting in memory.
_HASH_CHUNK = 1 << 20


def file_sha256(path: Path) -> str:
    """Hash a file's bytes.

    This identifies a session's source rather than its name: the same
    delivery stays comparable however the file was renamed or moved, and a
    re-encode makes it correctly incomparable, since re-encoding changes the
    pixels the detectors see.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoFile:
    """Decoded frames plus what identifies and times them."""

    sha256: str
    fps: float
    images: list[np.ndarray]


class OpenCVVideo:
    """Decodes with OpenCV. Implements ``VideoPort``."""

    def read(self, path: Path | str, *, stride: int = 1) -> VideoFile:
        """Decode ``path`` into frames in memory.

        Args:
            path: The video file.
            stride: Keep every Nth frame. Frames are read in order and
                dropped after decoding rather than sought past, because
                seeking in a variable-bitrate file lands on the nearest
                keyframe and would silently return different frames on a
                different build of ffmpeg.

        Raises:
            OSError: If the file cannot be opened or contains no frames.
                An empty result would look like a video of nothing, and the
                analysis would report no metrics rather than a failure.
            ValueError: If ``stride`` is not positive.
        """
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")

        video_path = Path(path)
        if not video_path.is_file():
            raise OSError(f"no such video: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"could not open {video_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            images: list[np.ndarray] = []
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    images.append(frame)
                index += 1
        finally:
            capture.release()

        if not images:
            raise OSError(f"{video_path} contains no decodable frames")

        return VideoFile(
            sha256=file_sha256(video_path),
            # A container that reports 0 or a nonsense rate would make every
            # timestamp wrong. 30 is a guess, but a stated one, and the
            # metrics that depend on timing are per-frame rather than
            # per-second.
            fps=fps if fps > 0 else 30.0,
            images=images,
        )
