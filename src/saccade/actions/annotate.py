"""Annotation — drawing what the agent looked at onto a copy of the image.

Annotated frames are for the evidence chain: a person auditing a result
should be able to see which region produced which claim. Nothing here feeds
a measurement.
"""

from __future__ import annotations

from PIL import ImageDraw
from PIL.Image import Image

from saccade.models import BBox

__all__ = ["annotate"]

DEFAULT_COLOR = "#ff2d55"
DEFAULT_WIDTH = 2


def annotate(
    image: Image,
    boxes: list[BBox],
    labels: list[str] | None = None,
    *,
    color: str = DEFAULT_COLOR,
    width: int = DEFAULT_WIDTH,
) -> Image:
    """Draw boxes, and optionally labels, on a copy of ``image``.

    Args:
        image: The source image. Left unmodified — the drawing happens on a
            copy, so a caller can annotate the same frame repeatedly without
            each call inheriting the previous one's marks.
        boxes: Regions to outline.
        labels: Text for each box. Must match ``boxes`` in length when given.
        color: Outline and text colour.
        width: Outline thickness in pixels.

    Returns:
        A new annotated image.

    Raises:
        ValueError: If ``labels`` is given but its length differs from
            ``boxes`` — a silent mismatch would mislabel the evidence.

    Example:
        >>> from PIL import Image as PILImage
        >>> marked = annotate(PILImage.new("RGB", (50, 50)), [BBox(x=5, y=5, w=20, h=20)], ["a"])
        >>> marked.size
        (50, 50)
    """
    if labels is not None and len(labels) != len(boxes):
        raise ValueError(f"got {len(boxes)} box(es) but {len(labels)} label(s)")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")

    canvas = image.copy()
    if canvas.mode not in ("RGB", "RGBA"):
        canvas = canvas.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    for index, box in enumerate(boxes):
        draw.rectangle(
            (box.x, box.y, box.right - 1, box.bottom - 1),
            outline=color,
            width=width,
        )
        if labels is not None:
            # Prefer above the box; drop inside it when there is no room.
            text_y = box.y - 12 if box.y >= 12 else box.y + 2
            draw.text((box.x + 2, text_y), labels[index], fill=color)

    return canvas
