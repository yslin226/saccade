"""Tests for the circle measurement tool.

This tool is a referee: the verifier uses its output to overrule a VLM. So
the bar is higher than "does it run" — a wrong measurement stated
confidently is worse than no measurement, because the loop trusts it.

Images here are drawn to order, so the expected geometry is known exactly
rather than eyeballed.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from benchmarks.blindtest.tools.circles import (
    TANGENT_UPPER_PX,
    circle_tool,
    detect_circles,
    measure_circles,
)

MAGENTA = (234, 0, 255)
BLUE = (0, 117, 255)


def canvas(
    circles: list[tuple[tuple[int, int], int, tuple[int, int, int]]],
    size: tuple[int, int] = (300, 300),
) -> Image.Image:
    """Draw circle outlines on white, the way the dataset renders them.

    Outlines, not filled discs. Getting this wrong sent an earlier version of
    the detector chasing occlusion handling for a case that cannot occur:
    with outlines neither circle ever hides part of the other, and both
    bounding boxes stay square however much the shapes cross.
    """
    image = Image.new("RGB", size, "white")
    pen = ImageDraw.Draw(image)
    for (cx, cy), r, colour in circles:
        pen.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=4)
    return image


class TestDetection:
    def test_finds_two_circles(self) -> None:
        image = canvas([((80, 150), 40, MAGENTA), ((220, 150), 40, BLUE)])
        assert len(detect_circles(image)) == 2

    def test_centres_are_accurate(self) -> None:
        image = canvas([((80, 150), 40, MAGENTA), ((220, 150), 40, BLUE)])
        found = sorted(detect_circles(image), key=lambda c: c.centre[0])
        assert found[0].centre == pytest.approx((80, 150), abs=2)
        assert found[1].centre == pytest.approx((220, 150), abs=2)

    def test_radii_are_accurate(self) -> None:
        image = canvas([((80, 150), 40, MAGENTA), ((220, 150), 55, BLUE)])
        found = sorted(detect_circles(image), key=lambda c: c.centre[0])
        assert found[0].radius == pytest.approx(40, abs=2)
        assert found[1].radius == pytest.approx(55, abs=2)

    def test_blank_image_finds_nothing(self) -> None:
        assert detect_circles(Image.new("RGB", (100, 100), "white")) == []

    def test_one_circle_is_reported_as_one(self) -> None:
        """Not two. A view containing half the picture must say so."""
        assert len(detect_circles(canvas([((150, 150), 50, MAGENTA)]))) == 1

    def test_one_circle_seen_in_several_shades_is_still_one(self) -> None:
        """Regression: anti-aliasing split one circle into several detections.

        Two near-identical magentas each passed the pixel floor, so "the two
        largest" compared a circle with itself — centre distance zero, and a
        confident report that it overlapped. Found on a real dataset image
        (00904.png, rendered at dpi=300), not by a test.
        """
        image = canvas([((150, 150), 60, MAGENTA)])
        pen = ImageDraw.Draw(image)
        # A near-identical shade of the same circle, as anti-aliasing gives.
        pen.ellipse([92, 92, 208, 208], outline=(233, 2, 254), width=3)

        assert len(detect_circles(image)) == 1

    def test_two_genuinely_different_circles_are_not_merged(self) -> None:
        """The counterpart risk: merging by position alone would erase the
        overlapping pairs the benchmark is entirely about."""
        image = canvas([((140, 150), 50, MAGENTA), ((160, 150), 50, BLUE)])
        assert len(detect_circles(image)) == 2


class TestOverlapVerdict:
    def test_clearly_overlapping(self) -> None:
        image = canvas([((130, 150), 50, MAGENTA), ((170, 150), 50, BLUE)])
        result = measure_circles(image, tangent_counts=False)
        assert result["overlap"] is True
        assert result["gap"] < 0

    def test_clearly_apart(self) -> None:
        image = canvas([((60, 150), 30, MAGENTA), ((240, 150), 30, BLUE)])
        result = measure_circles(image, tangent_counts=False)
        assert result["overlap"] is False
        assert result["gap"] > 0

    def test_tangent_follows_the_question(self) -> None:
        """The dataset answers Yes to "touching" and No to "overlapping"
        for the same tangent image, so the caller must say which is asked."""
        image = canvas([((100, 150), 50, MAGENTA), ((200, 150), 50, BLUE)])

        assert measure_circles(image, tangent_counts=True)["overlap"] is True
        assert measure_circles(image, tangent_counts=False)["overlap"] is False

    def test_a_small_positive_gap_counts_as_touching(self) -> None:
        """Rasterising tangent circles leaves a gap of a pixel or two."""
        image = canvas([((100, 150), 50, MAGENTA), ((202, 150), 50, BLUE)])
        result = measure_circles(image, tangent_counts=True)
        assert 0 <= result["gap"] <= TANGENT_UPPER_PX
        assert result["overlap"] is True

    def test_a_clearly_negative_gap_is_overlap_regardless_of_the_question(self) -> None:
        """Past the tolerance, the outlines genuinely cross."""
        image = canvas([((110, 150), 50, MAGENTA), ((190, 150), 50, BLUE)])
        result = measure_circles(image, tangent_counts=False)
        assert result["gap"] < -TANGENT_UPPER_PX
        assert result["overlap"] is True

    def test_measurements_are_reported_for_the_evidence_chain(self) -> None:
        image = canvas([((100, 150), 50, MAGENTA), ((200, 150), 50, BLUE)])
        result = measure_circles(image, tangent_counts=False)
        assert result["method"] == "circles_overlap"
        assert "centre_distance" in result
        assert "radius_sum" in result
        assert "gap" in result


class TestPartialViews:
    """Once the agent magnifies a corner, both circles are often gone."""

    def test_a_view_with_one_circle_gives_no_verdict(self) -> None:
        result = measure_circles(canvas([((150, 150), 50, MAGENTA)]), tangent_counts=False)
        assert "overlap" not in result
        assert result["detected"] == 1

    def test_a_view_with_no_circles_gives_no_verdict(self) -> None:
        blank = Image.new("RGB", (100, 100), "white")
        assert "overlap" not in measure_circles(blank, tangent_counts=False)


class TestToolContract:
    def test_a_verdict_is_a_measurement(self) -> None:
        tool = circle_tool(tangent_counts=False)
        image = canvas([((130, 150), 50, MAGENTA), ((170, 150), 50, BLUE)])
        result = tool.fn(image=image, viewport=None)

        assert result.is_measurement is True
        assert result.value["overlap"] is True

    def test_no_verdict_is_not_a_measurement(self) -> None:
        """ "I could not see them" must not masquerade as "they do not overlap".

        If this were flagged as a measurement, the verifier would use a
        failure to detect as grounds for overruling the model.
        """
        tool = circle_tool(tangent_counts=False)
        result = tool.fn(image=Image.new("RGB", (80, 80), "white"), viewport=None)

        assert result.is_measurement is False

    def test_the_tool_is_named_for_the_evidence_chain(self) -> None:
        tool = circle_tool(tangent_counts=False)
        assert tool.name == "circle_geometry"
        assert tool.description
