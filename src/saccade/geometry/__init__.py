"""Geometric verification.

Requires the ``geometry`` extra, because OpenCV is over 60MB and users who
only crop and zoom should not be made to download it.

The check happens here at package import so the failure names the fix,
instead of surfacing as a bare ``NameError`` deep inside a contour routine.
"""

from __future__ import annotations

try:
    import cv2 as cv2  # imported only to check availability
except ImportError as exc:  # pragma: no cover - depends on install extras
    raise ImportError(
        "Geometric verification requires opencv. Install it with:\n"
        '    pip install "saccade-vision[geometry]"\n'
        '    uv add "saccade-vision[geometry]"'
    ) from exc

from saccade.geometry.shapes import (
    Point,
    Segment,
    angle_between,
    centroid,
    circles_overlap,
    count_line_intersections,
    distance,
    segments_intersect,
    speed,
)

__all__ = [
    "Point",
    "Segment",
    "angle_between",
    "centroid",
    "circles_overlap",
    "count_line_intersections",
    "distance",
    "segments_intersect",
    "speed",
]
