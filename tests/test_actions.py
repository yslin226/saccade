"""Tests for the visual actions.

Two properties matter beyond the obvious geometry: the source image is never
modified, and the reported viewport is always in source coordinates. Both
are what make an evidence chain replayable.
"""

from __future__ import annotations

import pytest
from PIL import Image

from saccade.actions import annotate, crop, zoom
from saccade.actions.zoom import MAX_ZOOM
from saccade.models import BBox


def checkerboard(size: tuple[int, int] = (100, 100)) -> Image.Image:
    """An image with distinguishable regions, so crops can be told apart."""
    image = Image.new("RGB", size, color="white")
    for x in range(0, size[0], 10):
        for y in range(0, size[1], 10):
            if (x // 10 + y // 10) % 2 == 0:
                for dx in range(10):
                    for dy in range(10):
                        if x + dx < size[0] and y + dy < size[1]:
                            image.putpixel((x + dx, y + dy), (0, 0, 0))
    return image


class TestCrop:
    def test_output_size_matches_the_bbox(self) -> None:
        region, _ = crop(checkerboard(), BBox(x=10, y=20, w=40, h=30))
        assert region.size == (40, 30)

    def test_viewport_reports_source_coordinates(self) -> None:
        _, viewport = crop(checkerboard((200, 150)), BBox(x=10, y=20, w=40, h=30))
        assert viewport.bbox == BBox(x=10, y=20, w=40, h=30)
        assert viewport.source_size == (200, 150)
        assert viewport.zoom == 1.0

    def test_pixels_come_from_the_requested_region(self) -> None:
        source = Image.new("RGB", (20, 20), color="white")
        source.putpixel((15, 15), (255, 0, 0))
        region, _ = crop(source, BBox(x=10, y=10, w=10, h=10))
        assert region.getpixel((5, 5)) == (255, 0, 0)

    def test_source_image_is_not_modified(self) -> None:
        source = checkerboard()
        before = source.tobytes()
        crop(source, BBox(x=0, y=0, w=10, h=10))
        assert source.tobytes() == before
        assert source.size == (100, 100)

    def test_full_image_crop(self) -> None:
        region, viewport = crop(checkerboard(), BBox(x=0, y=0, w=100, h=100))
        assert region.size == (100, 100)
        assert viewport.covers_full_image is True

    def test_bbox_exactly_at_the_edge_is_allowed(self) -> None:
        region, _ = crop(checkerboard(), BBox(x=90, y=90, w=10, h=10))
        assert region.size == (10, 10)

    def test_bbox_past_the_edge_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="extends past image size"):
            crop(checkerboard(), BBox(x=95, y=0, w=10, h=10))

    def test_single_pixel_crop(self) -> None:
        region, _ = crop(checkerboard(), BBox(x=5, y=5, w=1, h=1))
        assert region.size == (1, 1)


class TestZoom:
    def test_magnifies_by_the_factor(self) -> None:
        view, _ = zoom(checkerboard(), BBox(x=0, y=0, w=20, h=20), 3.0)
        assert view.size == (60, 60)

    def test_default_factor_is_two(self) -> None:
        view, viewport = zoom(checkerboard(), BBox(x=0, y=0, w=20, h=20))
        assert view.size == (40, 40)
        assert viewport.zoom == 2.0

    def test_viewport_keeps_original_source_size(self) -> None:
        """The magnified image is 60x60, but the source is still 100x100."""
        _, viewport = zoom(checkerboard(), BBox(x=10, y=10, w=20, h=20), 3.0)
        assert viewport.source_size == (100, 100)
        assert viewport.bbox == BBox(x=10, y=10, w=20, h=20)
        assert viewport.zoom == 3.0

    def test_fractional_factor_shrinks(self) -> None:
        view, _ = zoom(checkerboard(), BBox(x=0, y=0, w=40, h=40), 0.5)
        assert view.size == (20, 20)

    def test_never_produces_a_zero_dimension(self) -> None:
        """A tiny region with a tiny factor must still yield a real image."""
        view, _ = zoom(checkerboard(), BBox(x=0, y=0, w=2, h=2), 0.1)
        assert view.size == (1, 1)

    def test_source_image_is_not_modified(self) -> None:
        source = checkerboard()
        before = source.tobytes()
        zoom(source, BBox(x=0, y=0, w=20, h=20), 4.0)
        assert source.tobytes() == before

    def test_zero_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            zoom(checkerboard(), BBox(x=0, y=0, w=10, h=10), 0)

    def test_negative_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            zoom(checkerboard(), BBox(x=0, y=0, w=10, h=10), -2.0)

    def test_absurd_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds the maximum"):
            zoom(checkerboard(), BBox(x=0, y=0, w=10, h=10), MAX_ZOOM + 1)

    def test_maximum_factor_allowed(self) -> None:
        view, _ = zoom(checkerboard(), BBox(x=0, y=0, w=4, h=4), MAX_ZOOM)
        assert view.size == (64, 64)

    def test_bbox_past_the_edge_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="extends past image size"):
            zoom(checkerboard(), BBox(x=0, y=95, w=10, h=10), 2.0)

    def test_magnified_content_matches_the_region(self) -> None:
        """A solid-colour region stays that colour after magnification."""
        source = Image.new("RGB", (40, 40), color="white")
        for x in range(10, 20):
            for y in range(10, 20):
                source.putpixel((x, y), (0, 128, 255))
        view, _ = zoom(source, BBox(x=10, y=10, w=10, h=10), 4.0)
        assert view.size == (40, 40)
        assert view.getpixel((20, 20)) == (0, 128, 255)


class TestAnnotate:
    def test_output_keeps_the_source_size(self) -> None:
        marked = annotate(checkerboard(), [BBox(x=10, y=10, w=20, h=20)])
        assert marked.size == (100, 100)

    def test_source_image_is_not_modified(self) -> None:
        source = Image.new("RGB", (50, 50), color="white")
        before = source.tobytes()
        annotate(source, [BBox(x=5, y=5, w=20, h=20)])
        assert source.tobytes() == before

    def test_something_is_actually_drawn(self) -> None:
        source = Image.new("RGB", (50, 50), color="white")
        marked = annotate(source, [BBox(x=5, y=5, w=20, h=20)])
        assert marked.tobytes() != source.tobytes()

    def test_repeated_annotation_does_not_accumulate(self) -> None:
        """Each call starts from the original, not the previous result."""
        source = Image.new("RGB", (50, 50), color="white")
        box = [BBox(x=5, y=5, w=20, h=20)]
        assert annotate(source, box).tobytes() == annotate(source, box).tobytes()

    def test_multiple_boxes(self) -> None:
        marked = annotate(
            checkerboard(),
            [BBox(x=5, y=5, w=10, h=10), BBox(x=50, y=50, w=20, h=20)],
        )
        assert marked.size == (100, 100)

    def test_no_boxes_returns_an_unmarked_copy(self) -> None:
        source = Image.new("RGB", (30, 30), color="white")
        marked = annotate(source, [])
        assert marked.tobytes() == source.tobytes()
        assert marked is not source

    def test_labels_are_accepted(self) -> None:
        marked = annotate(checkerboard(), [BBox(x=20, y=20, w=30, h=30)], ["circle A"])
        assert marked.size == (100, 100)

    def test_label_near_the_top_edge_still_draws(self) -> None:
        """A box at y=0 has no room above it for text."""
        marked = annotate(checkerboard(), [BBox(x=0, y=0, w=30, h=30)], ["top"])
        assert marked.size == (100, 100)

    def test_label_count_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="1 box"):
            annotate(checkerboard(), [BBox(x=0, y=0, w=10, h=10)], ["a", "b"])

    def test_zero_width_rejected(self) -> None:
        with pytest.raises(ValueError, match="width must be positive"):
            annotate(checkerboard(), [BBox(x=0, y=0, w=10, h=10)], width=0)

    def test_greyscale_input_is_converted(self) -> None:
        marked = annotate(Image.new("L", (40, 40), color=255), [BBox(x=5, y=5, w=10, h=10)])
        assert marked.mode == "RGB"

    def test_custom_colour_is_used(self) -> None:
        source = Image.new("RGB", (40, 40), color="white")
        marked = annotate(source, [BBox(x=5, y=5, w=20, h=20)], color="#00ff00", width=1)
        colours = {marked.getpixel((x, y)) for x in range(40) for y in range(40)}
        assert (0, 255, 0) in colours


class TestActionsComposeForEvidence:
    def test_zoom_after_crop_still_maps_to_the_original(self) -> None:
        """Chained actions must not lose the link back to the source."""
        source = checkerboard((200, 200))
        _, first = crop(source, BBox(x=50, y=50, w=100, h=100))
        assert first.source_size == (200, 200)

        # A second look at a sub-region, expressed in source coordinates.
        _, second = zoom(source, BBox(x=60, y=60, w=20, h=20), 4.0)
        assert second.source_size == (200, 200)
        assert second.bbox.x >= first.bbox.x
        assert second.bbox.right <= first.bbox.right
