"""Counting where a blue line and a red line cross.

This is the task leading models score 56.84% on — barely above chance, and
the one the paper singles out. ``count_line_intersections`` in the engine
needs segment coordinates; the benchmark gives a plot.

The method is not Hough. These are anti-aliased curves rather than straight
segments, and fitting lines to them would introduce exactly the sort of
approximation a referee cannot afford. Instead each curve is traced as one
y-value per column, and crossings are counted where their difference
changes sign — which is what "the lines cross" means, computed rather than
estimated.
"""

from __future__ import annotations

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel

from saccade.tools import Tool, ToolResult

__all__ = ["LineParams", "count_crossings", "line_tool", "trace_curves"]

# A pixel is coloured when it is far enough from white.
_WHITE_MARGIN = 60

# How much redder than bluer (or the reverse) a pixel must be to be claimed
# by one curve. Anti-aliasing blends the two where they meet, and those
# blended pixels belong to neither.
_CHANNEL_MARGIN = 30

# Columns holding fewer than this many pixels of a colour are the ragged
# ends of a stroke rather than part of the curve.
_MIN_COLUMN_PIXELS = 1

# The curves must be separated by more than this to count as apart. Below
# it they are within stroke width of each other and effectively touching.
_CONTACT_PX = 2.0


class LineParams(BaseModel):
    """This tool measures the whole view, so it takes no arguments."""


def trace_curves(image: Image) -> tuple[np.ndarray, np.ndarray]:
    """Trace the red and blue curves as one y-value per column.

    Returns:
        Two arrays of length ``width``, holding the mean row of that colour
        in each column, or NaN where the colour is absent. NaN is deliberate:
        a column the curve does not reach must not be filled in with a guess.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    height, width = pixels.shape[:2]

    coloured = np.abs(pixels - 255).sum(axis=2) > _WHITE_MARGIN
    red = coloured & (pixels[:, :, 0] > pixels[:, :, 2] + _CHANNEL_MARGIN)
    blue = coloured & (pixels[:, :, 2] > pixels[:, :, 0] + _CHANNEL_MARGIN)

    rows = np.arange(height)[:, None]
    return _column_means(red, rows, width), _column_means(blue, rows, width)


def _column_means(mask: np.ndarray, rows: np.ndarray, width: int) -> np.ndarray:
    """Mean row index of a mask in each column, NaN where it is absent."""
    counts = mask.sum(axis=0)
    totals = (mask * rows).sum(axis=0)

    means = np.full(width, np.nan)
    present = counts >= _MIN_COLUMN_PIXELS
    means[present] = totals[present] / counts[present]
    return means


def count_crossings(red: np.ndarray, blue: np.ndarray) -> int:
    """Count sign changes in ``red - blue`` over the columns holding both.

    A crossing is where the two curves swap which one is on top. Runs of
    contact — where they are within a stroke width and the difference sits
    near zero — count once, not once per column, since a stretch of touching
    is one meeting rather than several.
    """
    both = ~np.isnan(red) & ~np.isnan(blue)
    if both.sum() < 2:
        return 0

    difference = (red - blue)[both]

    # Collapse each column to which curve is on top, treating near-zero as
    # contact rather than as a side.
    side = np.zeros(len(difference), dtype=np.int8)
    side[difference > _CONTACT_PX] = 1
    side[difference < -_CONTACT_PX] = -1

    crossings = 0
    last_side = 0
    touching = False

    for value in side:
        if value == 0:
            # In contact. One crossing per run, counted when it ends.
            touching = True
            continue

        if last_side == 0:
            last_side = value
            touching = False
            continue

        if value != last_side:
            crossings += 1
            last_side = value
        elif touching:
            # Left contact on the same side it arrived: the curves met and
            # parted without swapping over. Still a meeting.
            crossings += 1
        touching = False

    return crossings


def measure_lines(image: Image) -> dict[str, object]:
    """Measure how many times the two curves cross in a view."""
    red, blue = trace_curves(image)
    both = ~np.isnan(red) & ~np.isnan(blue)

    if both.sum() < 2:
        return {
            "method": "count_line_intersections",
            "columns_with_both": int(both.sum()),
            "note": "the two lines do not share enough of this view to compare",
        }

    return {
        "method": "count_line_intersections",
        "crossings": count_crossings(red, blue),
        "columns_with_both": int(both.sum()),
    }


def line_tool() -> Tool:
    """Build the line-crossing tool."""

    def run(image: Image, viewport: object) -> ToolResult:
        value = measure_lines(image)
        # A view that does not hold both curves yields no count. Reporting
        # zero would let "I could not see them" pass as "they never cross".
        #
        # answer_key names the verdict, so columns_with_both is read as
        # context rather than as a second number to check the answer against.
        return ToolResult(
            value=value,
            is_measurement="crossings" in value,
            answer_key="crossings",
        )

    return Tool(
        name="line_geometry",
        description="Count where the blue and red lines cross in the view",
        fn=run,
        params_schema=LineParams,
    )
