"""Zooming — cropping to a region and magnifying it.

This is the action the whole project is named after. When a model cannot
tell whether two circles touch, the fix is not a better prompt; it is more
pixels on the place where they might touch.

Magnification adds no information — it resamples what is already there. What
it does is stop fine detail from being destroyed by the model's own input
downscaling, which is where thin lines and near-tangencies get lost.
"""

from __future__ import annotations

from PIL.Image import Image, Resampling

from saccade.models import BBox, Viewport

__all__ = ["zoom"]

# Guard against a plausible-looking factor producing a gigabyte of pixels.
MAX_ZOOM = 16.0


def zoom(image: Image, bbox: BBox, factor: float = 2.0) -> tuple[Image, Viewport]:
    """Crop to ``bbox`` and scale the result by ``factor``.

    Args:
        image: The source image. Left unmodified.
        bbox: Region to magnify, in source pixel coordinates.
        factor: Magnification. Must be greater than 0 and at most
            :data:`MAX_ZOOM`.

    Returns:
        The magnified crop, and a viewport recording both the region and the
        magnification. ``viewport.source_size`` stays the *original* image
        size, so the region can always be located in the picture the user
        supplied.

    Raises:
        ValueError: If ``factor`` is out of range, or ``bbox`` extends past
            the image bounds.

    Example:
        >>> from PIL import Image as PILImage
        >>> view, viewport = zoom(PILImage.new("RGB", (100, 100)), BBox(x=0, y=0, w=20, h=20), 3.0)
        >>> view.size
        (60, 60)
        >>> viewport.zoom
        3.0
    """
    if factor <= 0:
        raise ValueError(f"zoom factor must be positive, got {factor}")
    if factor > MAX_ZOOM:
        raise ValueError(f"zoom factor {factor} exceeds the maximum of {MAX_ZOOM}")

    width, height = image.size
    if bbox.right > width or bbox.bottom > height:
        raise ValueError(
            f"bbox ({bbox.x}, {bbox.y}, {bbox.w}, {bbox.h}) extends past image size {image.size}"
        )

    region = image.crop((bbox.x, bbox.y, bbox.right, bbox.bottom))

    target = (max(1, round(bbox.w * factor)), max(1, round(bbox.h * factor)))
    # LANCZOS keeps edges crisp. NEAREST would produce blocky steps a model
    # could read as real structure, which is exactly the sort of artefact
    # this project must not introduce.
    magnified = region.resize(target, Resampling.LANCZOS)

    viewport = Viewport(bbox=bbox, zoom=factor, source_size=(width, height))
    return magnified, viewport
