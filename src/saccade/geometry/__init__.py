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
    BBox,
    Point,
    Segment,
    angle_between,
    bbox_iou,
    bearing,
    centroid,
    circles_overlap,
    count_line_intersections,
    distance,
    point_to_segment_distance,
    segments_intersect,
    smooth,
    speed,
)

__all__ = [
    "BBox",
    "Point",
    "Segment",
    "angle_between",
    "bbox_iou",
    "bearing",
    "centroid",
    "circles_overlap",
    "count_line_intersections",
    "distance",
    "point_to_segment_distance",
    "segments_intersect",
    "smooth",
    "speed",
]
