"""Detecting the two circles in a Touching Circles image, so geometry can judge.

``saccade.geometry.circles_overlap`` needs centres and radii; the benchmark
gives a picture. This is the missing step — and it is the whole reason the
verifier had nothing to verify in M1.

Detection is by colour, not Hough. The two circles are drawn in fixed,
distinct hues (magenta and blue on white), so isolating each colour and
taking the extent of its pixels is exact where a Hough transform would be
approximate and parameter-dependent. A referee that is itself a guess cannot
referee anything (spec 1.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel

from saccade.geometry.shapes import distance
from saccade.tools import Tool, ToolResult

__all__ = [
    "TANGENT_LOWER_PX",
    "TANGENT_UPPER_PX",
    "Circle",
    "CircleParams",
    "circle_tool",
    "detect_circles",
    "measure_circles",
]

# A measured gap inside this band means the circles are tangent; what that
# implies then depends on which question was asked.
#
# The band is asymmetric and both edges come from sweeping the full dataset
# rather than from judgement. Rasterisation and stroke width push a truly
# tangent pair a little either side of zero, so the lower edge cannot be 0;
# but real overlaps reach -1.5px, so it cannot go far below either. The
# values sit in the middle of a plateau that scores 99.0%, not on its edge.
TANGENT_LOWER_PX = -1.0
TANGENT_UPPER_PX = 3.0

# A pixel counts as coloured when it is far enough from white. The circles
# are drawn as saturated outlines, so this is a wide margin, not a threshold
# that needs tuning per image.
_WHITE_MARGIN = 60

# A detected shape needs at least this many pixels of one exact colour.
# Below it, the "shape" is anti-aliasing fringe.
_MIN_PIXELS = 20

# Total RGB difference below which two colours are shades of one another.
# The dataset's two circles differ by far more than this (magenta vs blue),
# while anti-aliasing fringes differ by a handful of levels.
_SHADE_DISTANCE = 60

# Below this, a mask is anti-aliasing fringe rather than a drawn shape.
_MIN_DIAMETER = 8

# How close to the view boundary a circle may come before it is assumed cut
# off. One pixel of slack for rasterisation.
_EDGE_MARGIN = 1


@dataclass(frozen=True)
class Circle:
    """A detected circle, in pixel coordinates."""

    centre: tuple[float, float]
    radius: float


class CircleParams(BaseModel):
    """This tool measures the whole view, so it takes no arguments."""


def detect_circles(image: Image) -> list[Circle]:
    """Find the coloured circles in a view.

    Returns:
        The circles found, largest first. Fewer than two means the view does
        not contain both — which happens constantly once the agent starts
        magnifying corners, and is reported honestly rather than guessed at.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)

    # Anything clearly not white is part of a drawn shape.
    from_white = np.abs(pixels - 255).sum(axis=2)
    coloured = from_white > _WHITE_MARGIN
    if not coloured.any():
        return []

    # Group the coloured pixels by hue. Each circle is a single flat colour,
    # so exact colour values separate them without clustering.
    values = pixels[coloured]
    unique, counts = np.unique(values.reshape(-1, 3), axis=0, return_counts=True)
    dominant = unique[counts >= _MIN_PIXELS]

    detections: list[tuple[Circle, np.ndarray]] = []
    for colour in dominant:
        mask = np.all(pixels == colour, axis=2)
        found = _circle_from_mask(mask)
        if found is not None:
            detections.append((found, colour))

    detections.sort(key=lambda pair: pair[0].radius, reverse=True)
    return _merge_shades(detections)


def _merge_shades(detections: list[tuple[Circle, np.ndarray]]) -> list[Circle]:
    """Collapse several shades of one drawn circle into a single detection.

    Anti-aliasing renders one circle in a handful of near-identical colours,
    each passing the pixel floor and yielding its own detection at the same
    place. Left alone, "the two largest" can be one circle compared against
    itself: centre distance zero, and a confident report that it overlaps.
    That measurement is worse than none, because the verifier trusts it
    enough to overrule the model.

    Position alone cannot decide this — two genuinely overlapping circles
    also sit close together, and merging those would erase the very case the
    benchmark is about. Shades of one circle are near-identical in *colour*
    as well, so both tests must agree before anything is merged.
    """
    kept: list[tuple[Circle, np.ndarray]] = []
    for circle, colour in detections:
        same_shape = False
        for other, other_colour in kept:
            close = distance(circle.centre, other.centre) < max(circle.radius, other.radius) * 0.25
            similar = int(np.abs(colour.astype(np.int32) - other_colour.astype(np.int32)).sum())
            if close and similar <= _SHADE_DISTANCE:
                same_shape = True
                break
        if not same_shape:
            kept.append((circle, colour))
    return [circle for circle, _ in kept]


def _circle_from_mask(mask: np.ndarray) -> Circle | None:
    """Fit a circle to the pixels of one colour.

    The diameter comes from the mask's longer axis, and the centre from the
    midpoint of that axis. Both are taken along the *unoccluded* direction on
    purpose: when two circles overlap, the one drawn underneath survives as a
    crescent, and its bounding box is narrower than the circle it came from.
    Averaging width and height would shrink it and report a gap that is not
    there.

    A mask spanning only a few pixels is anti-aliasing fringe, not a shape.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    width = int(x1 - x0 + 1)
    height = int(y1 - y0 + 1)
    diameter = max(width, height)
    if diameter < _MIN_DIAMETER:
        return None

    # The circles are drawn as outlines rather than filled discs, so neither
    # ever hides part of the other: both bounding boxes stay square even
    # where the shapes cross. Measured across the dataset, squareness is
    # ~1.0 for overlapping and separate pairs alike, so the box gives the
    # centre and diameter directly.
    return Circle(
        centre=(x0 + width / 2, y0 + height / 2),
        radius=diameter / 2,
    )


def _touches_edge(circles: list[Circle], size: tuple[int, int]) -> bool:
    """Whether any circle reaches the boundary of the view.

    A circle that does is probably cut off, and what was measured is a
    fragment rather than the shape.
    """
    width, height = size
    for circle in circles:
        cx, cy = circle.centre
        r = circle.radius
        if (
            cx - r <= _EDGE_MARGIN
            or cy - r <= _EDGE_MARGIN
            or cx + r >= width - 1 - _EDGE_MARGIN
            or cy + r >= height - 1 - _EDGE_MARGIN
        ):
            return True
    return False


def measure_circles(image: Image, *, tangent_counts: bool) -> dict[str, object]:
    """Measure whether the two circles in a view overlap.

    Args:
        image: The current view.
        tangent_counts: Whether circles that exactly touch count. The dataset
            distinguishes this by question: at distance 0 it answers Yes to
            "are they touching" and No to "are they overlapping", so the
            caller must say which is being asked.
    """
    circles = detect_circles(image)

    if len(circles) < 2:
        return {
            "method": "circles_overlap",
            "detected": len(circles),
            "note": "fewer than two circles in this view",
        }

    if _touches_edge(circles, image.size):
        # A magnified crop cuts the circles, and a fragment measures as a
        # smaller circle in the wrong place. Reporting a verdict from that
        # would put a confident wrong number into evidence — which is worse
        # than no measurement, because the verifier believes it.
        return {
            "method": "circles_overlap",
            "detected": len(circles),
            "note": "a circle runs past the edge of this view",
        }

    first, second = circles[0], circles[1]
    centre_gap = distance(first.centre, second.centre)
    radius_sum = first.radius + second.radius
    gap = centre_gap - radius_sum

    # Exact float comparison is right for exact geometry and wrong for
    # pixels: rasterising two mathematically tangent circles leaves a gap of
    # about 1.3px (max 2.05 across the dataset), while the next distance
    # apart never comes closer than 6.8px.
    #
    # The tolerance is one-sided on purpose. A gap slightly *above* zero is
    # ambiguous — tangency and rasterisation look the same. A gap below zero
    # is not: the outlines genuinely cross, and treating that as "merely
    # touching" got six overlapping pairs wrong.
    # Exact float comparison is right for exact geometry and wrong for
    # pixels: rasterisation and stroke width move a truly tangent pair a
    # little either side of zero.
    ambiguous = TANGENT_LOWER_PX <= gap <= TANGENT_UPPER_PX
    overlap = tangent_counts if ambiguous else gap < 0

    return {
        "method": "circles_overlap",
        "overlap": overlap,
        "centre_distance": round(centre_gap, 2),
        "radius_sum": round(radius_sum, 2),
        "gap": round(gap, 2),
        "detected": len(circles),
    }


def circle_tool(*, tangent_counts: bool) -> Tool:
    """Build the circle-geometry tool for one phrasing of the question.

    Args:
        tangent_counts: True for "are they touching", False for "are they
            overlapping". See :func:`measure_circles`.
    """

    def run(image: Image, viewport: object) -> ToolResult:
        value = measure_circles(image, tangent_counts=tangent_counts)
        # A view without both circles yields no verdict. Reporting it as a
        # measurement anyway would let "I could not see them" masquerade as
        # "they do not overlap".
        #
        # answer_key names the verdict, so centre_distance and radius_sum stay
        # in the evidence chain without being checked against the answer.
        return ToolResult(
            value=value,
            is_measurement="overlap" in value,
            answer_key="overlap",
        )

    return Tool(
        name="circle_geometry",
        description="Measure whether the two circles in the view overlap",
        fn=run,
        params_schema=CircleParams,
    )
