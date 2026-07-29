"""Cropping — narrowing the viewport to a region of the source image.

The returned :class:`~saccade.models.Viewport` always describes the region
in *source* coordinates, never in coordinates of the cropped result. Without
that, two crops in a row would produce an evidence chain nobody could map
back to the original picture.
"""

from __future__ import annotations

from PIL.Image import Image

from saccade.models import BBox, Viewport

__all__ = ["crop"]


def crop(image: Image, bbox: BBox) -> tuple[Image, Viewport]:
    """Cut ``bbox`` out of ``image``.

    Args:
        image: The source image. Left unmodified.
        bbox: Region to keep, in source pixel coordinates.

    Returns:
        The cropped image, and a viewport describing what it shows.

    Raises:
        ValueError: If ``bbox`` extends past the image bounds. Silently
            clamping would report a viewport the caller never asked for and
            quietly corrupt the evidence chain.

    Example:
        >>> from PIL import Image as PILImage
        >>> region, viewport = crop(PILImage.new("RGB", (100, 100)), BBox(x=10, y=10, w=40, h=40))
        >>> region.size
        (40, 40)
        >>> viewport.source_size
        (100, 100)
    """
    width, height = image.size
    if bbox.right > width or bbox.bottom > height:
        raise ValueError(
            f"bbox ({bbox.x}, {bbox.y}, {bbox.w}, {bbox.h}) extends past image size {image.size}"
        )

    region = image.crop((bbox.x, bbox.y, bbox.right, bbox.bottom))
    viewport = Viewport(bbox=bbox, zoom=1.0, source_size=(width, height))
    return region, viewport
