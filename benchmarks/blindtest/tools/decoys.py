"""Tools that measure the wrong thing, confidently.

The benchmark's real tools were each written for one task and solve exactly
it, so running every registered tool unconditionally reaches 98%. That
arrangement flatters the design and is not what an application faces: a
toolbox holds instruments for several questions, and the one for a different
question still returns a number when pointed at this one.

The number is the danger. A measurement is what the verifier uses to overrule
the model, so a tool that measures the wrong thing does not merely waste a
call — it manufactures grounds to replace a correct answer with a wrong one.
Picking the right instrument is therefore a real decision with a real cost,
and a benchmark that never offers a wrong one cannot tell whether the model
is making it.

Each decoy here is built to be plausibly chosen and quietly wrong:

- Its description sounds relevant to at least one task. A decoy nobody would
  ever pick tests nothing.
- It reports ``is_measurement=True`` with an ``answer_key``, so if chosen it
  can and will overrule the model.
- Its arithmetic is real. These are not stubs or random numbers; they measure
  something true about the image that does not answer the question asked.

The last point matters most. A broken tool would be caught by anything; a
correct tool aimed at the wrong question is what actually happens in a
deployed system, and it is invisible from inside the measurement.

None of these belong to any task. They exist to be *not* chosen.
"""

from __future__ import annotations

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel

from saccade.tools import Tool, ToolResult

__all__ = [
    "DECOY_NAMES",
    "DecoyParams",
    "bounding_box_tool",
    "decoy_tools",
    "ink_coverage_tool",
    "symmetry_tool",
]


# Width-to-height ratio below which the bounding-box decoy calls two shapes
# touching. Two circles of equal radius that exactly touch span 4r wide by 2r
# tall, so 2.0 is the geometrically correct boundary for that one arrangement
# — and wrong for every other. Fixing it at a constant is the flaw: the true
# boundary moves with the radii, and this cannot follow it.
_TOUCHING_ASPECT = 2.0


class DecoyParams(BaseModel):
    """Decoys measure the view they are handed, so they take no arguments."""


def _ink(image: Image) -> np.ndarray:
    """Where the drawing is, as a boolean mask.

    BlindTest images are black strokes on white, so "dark" is "ink". A
    mid-grey threshold survives the anti-aliasing at stroke edges without
    needing to know the stroke width.
    """
    grey = np.asarray(image.convert("L"))
    return np.asarray(grey < 128)


def bounding_box_tool() -> Tool:
    """Whether the drawing's ink reaches the middle of its own bounding box.

    The sharpest decoy on the circles task, because it is *nearly* right. Two
    circles that touch put ink at the midpoint between them; two that are
    apart leave that midpoint blank. That is true, and it is very close to
    the question — close enough that the reasoning holds right up until the
    circles are separated along one axis but overlap along the other, which
    is most of what this dataset contains.

    Measured against ground truth on 60 items it disagrees on roughly half,
    always confidently. A model that picks it gets a real number that then
    overrules its own correct answer.
    """

    def run(image: Image, viewport: object) -> ToolResult:
        ink = _ink(image)
        rows = np.flatnonzero(ink.any(axis=1))
        columns = np.flatnonzero(ink.any(axis=0))
        if rows.size == 0 or columns.size == 0:
            return ToolResult(
                value={"method": "bounding_box", "note": "nothing drawn in this view"},
                is_measurement=False,
            )

        width = int(columns[-1] - columns[0])
        height = int(rows[-1] - rows[0])

        # Two circles of equal radius r side by side span 2r vertically and
        # (2r + separation) horizontally. So the box is square when they are
        # concentric and grows wider as they part: the aspect ratio is a real
        # measure of how far apart they are.
        #
        # It is also the wrong measure. Where the boundary sits in that ratio
        # depends on the radii, and this decoy fixes it at a constant. The
        # dataset's gaps run from -13px to +15px on a ~95px radius sum, so the
        # true boundary is a hair's breadth that a fixed threshold cannot
        # track — it is right about the easy items and wrong through the
        # middle, which is where the dataset lives.
        aspect = width / height if height else float("inf")

        return ToolResult(
            value={
                "method": "bounding_box",
                "boxes_overlap": aspect <= _TOUCHING_ASPECT,
                "aspect_ratio": round(aspect, 3),
                "box": [int(columns[0]), int(rows[0]), width, height],
            },
            is_measurement=True,
            answer_key="boxes_overlap",
        )

    return Tool(
        name="bounding_box_overlap",
        description="Measure whether the two shapes' bounding rectangles overlap",
        fn=run,
        params_schema=DecoyParams,
    )


def ink_coverage_tool() -> Tool:
    """How much of the view is drawn on.

    Aimed at the counting tasks. More lines means more ink, so a coverage
    figure correlates with the count without measuring it — and correlation
    is exactly what makes a decoy attractive. The number it reports is a
    percentage, which is not a count of anything, but a model reaching for
    "something that measures how much is in the picture" will find it.
    """

    def run(image: Image, viewport: object) -> ToolResult:
        ink = _ink(image)
        return ToolResult(
            value={
                "method": "ink_coverage",
                "coverage_percent": round(float(ink.mean()) * 100, 2),
                "pixels_drawn": int(ink.sum()),
            },
            is_measurement=True,
            answer_key="coverage_percent",
        )

    return Tool(
        name="ink_coverage",
        description="Measure how much of the image is covered by drawn strokes",
        fn=run,
        params_schema=DecoyParams,
    )


def symmetry_tool() -> Tool:
    """How closely the left half mirrors the right.

    The least plausible of the three, and deliberately so. A toolbox is not
    uniformly tempting — some entries are obviously irrelevant, and a model
    that cannot rule *those* out is not choosing at all. This is the floor
    the other two are measured against.
    """

    def run(image: Image, viewport: object) -> ToolResult:
        ink = _ink(image)
        width = ink.shape[1]
        half = width // 2
        if half == 0:
            return ToolResult(
                value={"method": "mirror_symmetry", "note": "view too narrow to halve"},
                is_measurement=False,
            )

        left = ink[:, :half]
        right = np.fliplr(ink[:, width - half :])
        agreement = float((left == right).mean())

        return ToolResult(
            value={
                "method": "mirror_symmetry",
                "symmetry_score": round(agreement, 4),
            },
            is_measurement=True,
            answer_key="symmetry_score",
        )

    return Tool(
        name="mirror_symmetry",
        description="Measure how closely the left half of the image mirrors the right",
        fn=run,
        params_schema=DecoyParams,
    )


DECOY_NAMES = ("bounding_box_overlap", "ink_coverage", "mirror_symmetry")


def decoy_tools() -> list[Tool]:
    """Every decoy, in the order they are offered."""
    return [bounding_box_tool(), ink_coverage_tool(), symmetry_tool()]
