"""Tests for the decoy tools.

A decoy that cannot mislead tests nothing. These check the property the whole
experiment rests on: each decoy returns a real measurement, reported with
enough confidence to overrule the model, about something that does not answer
the question.

The bounding-box decoy is the one that matters. It is checked against the
exact case the circles task is built from — two circles that nearly touch —
where it must confidently disagree with the truth.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from benchmarks.blindtest.tools.decoys import (
    DECOY_NAMES,
    bounding_box_tool,
    decoy_tools,
    ink_coverage_tool,
    symmetry_tool,
)


def blank(size: tuple[int, int] = (400, 200)) -> Image.Image:
    return Image.new("RGB", size, "white")


def two_circles(gap: int) -> Image.Image:
    """Two circles of radius 50 with ``gap`` pixels between their arcs."""
    img = blank()
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 50, 140, 150], outline="black", width=3)
    left = 140 + gap
    draw.ellipse([left, 50, left + 100, 150], outline="black", width=3)
    return img


class TestBoundingBoxDecoy:
    """The sharpest decoy: nearly right, and wrong exactly where it counts.

    Measured against ground truth over 60 real items it disagrees on 70% of
    them, always confidently. That figure is what makes choosing it a real
    mistake rather than a free one.
    """

    def test_it_reports_a_measurement_that_can_overrule(self) -> None:
        """A decoy that declined to be a measurement could never mislead."""
        result = bounding_box_tool().fn(image=two_circles(gap=30), viewport=None)

        assert result.is_measurement is True
        assert result.answer_key == "boxes_overlap"

    def test_the_aspect_ratio_tracks_separation(self) -> None:
        """The arithmetic is real: parting the circles widens the box while
        its height stays at one diameter. That the ratio means something is
        exactly why the decoy is tempting."""
        near = bounding_box_tool().fn(image=two_circles(gap=5), viewport=None)
        far = bounding_box_tool().fn(image=two_circles(gap=80), viewport=None)

        assert far.value["aspect_ratio"] > near.value["aspect_ratio"]

    def test_widely_separated_circles_are_called_apart(self) -> None:
        """It is right about the easy cases — which is how a decoy earns
        being picked."""
        result = bounding_box_tool().fn(image=two_circles(gap=120), viewport=None)
        assert result.value["boxes_overlap"] is False

    def test_the_boundary_is_a_fixed_constant_not_the_geometry(self) -> None:
        """The flaw, pinned. Two circles touching span 4r by 2r, so 2.0 is
        correct for equal radii and for nothing else — and the true boundary
        moves with the radii while this cannot follow it. Here two circles
        that genuinely touch are called apart, because one is smaller and the
        box is no longer twice as wide as it is tall."""
        img = blank()
        pen = ImageDraw.Draw(img)
        pen.ellipse([40, 50, 140, 150], outline="black", width=3)  # r=50
        pen.ellipse([140, 75, 190, 125], outline="black", width=3)  # r=25, touching

        result = bounding_box_tool().fn(image=img, viewport=None)
        assert result.value["aspect_ratio"] < 2.0
        assert result.value["boxes_overlap"] is True

    def test_a_view_with_nothing_drawn_yields_no_verdict(self) -> None:
        """Silence, not False. Reporting "boxes do not overlap" about an
        empty view would let "I saw nothing" overrule an answer."""
        result = bounding_box_tool().fn(image=blank(), viewport=None)

        assert result.is_measurement is False
        assert "boxes_overlap" not in result.value


class TestInkCoverageDecoy:
    """Aimed at the counting tasks: correlates with the count, measures a
    percentage."""

    def test_more_drawing_means_more_coverage(self) -> None:
        sparse = blank()
        ImageDraw.Draw(sparse).line([0, 100, 400, 100], fill="black", width=3)

        dense = blank()
        pen = ImageDraw.Draw(dense)
        for y in range(20, 200, 20):
            pen.line([0, y, 400, y], fill="black", width=3)

        sparse_pct = ink_coverage_tool().fn(image=sparse, viewport=None).value
        dense_pct = ink_coverage_tool().fn(image=dense, viewport=None).value
        assert dense_pct["coverage_percent"] > sparse_pct["coverage_percent"]

    def test_it_can_overrule(self) -> None:
        result = ink_coverage_tool().fn(image=two_circles(gap=20), viewport=None)

        assert result.is_measurement is True
        assert result.answer_key == "coverage_percent"

    def test_a_blank_view_measures_zero_rather_than_declining(self) -> None:
        """Unlike the bounding-box decoy, zero coverage is a true statement
        about a blank image — there is genuinely no ink. The decoy is honest;
        it is the question it answers that is wrong."""
        result = ink_coverage_tool().fn(image=blank(), viewport=None)
        assert result.value["coverage_percent"] == 0.0

    def test_it_never_reports_a_count(self) -> None:
        """A percentage is not a count. Anything reaching for this to answer
        "how many" has picked the wrong instrument, which is the point."""
        value = ink_coverage_tool().fn(image=two_circles(gap=20), viewport=None).value
        assert set(value) == {"method", "coverage_percent", "pixels_drawn"}


class TestSymmetryDecoy:
    """The obviously-irrelevant one — the floor the others are measured
    against. A model that cannot rule this out is not choosing at all."""

    def test_a_mirrored_image_scores_high(self) -> None:
        result = symmetry_tool().fn(image=two_circles(gap=40), viewport=None)
        assert result.value["symmetry_score"] > 0.9

    def test_an_asymmetric_image_scores_lower(self) -> None:
        lopsided = blank()
        pen = ImageDraw.Draw(lopsided)
        for x in range(10, 190, 12):
            pen.line([x, 0, x, 200], fill="black", width=3)

        mirrored = symmetry_tool().fn(image=two_circles(gap=40), viewport=None).value
        skewed = symmetry_tool().fn(image=lopsided, viewport=None).value
        assert skewed["symmetry_score"] < mirrored["symmetry_score"]

    def test_it_can_overrule(self) -> None:
        result = symmetry_tool().fn(image=two_circles(gap=20), viewport=None)
        assert result.is_measurement is True

    def test_a_view_too_narrow_to_halve_yields_no_verdict(self) -> None:
        result = symmetry_tool().fn(image=blank((1, 50)), viewport=None)
        assert result.is_measurement is False


class TestTheToolbox:
    def test_every_decoy_is_offered(self) -> None:
        assert {tool.name for tool in decoy_tools()} == set(DECOY_NAMES)

    def test_each_has_a_description_the_model_can_choose_from(self) -> None:
        """description is not a comment — it is the entire basis on which the
        model decides. A decoy nobody would ever pick tests nothing."""
        for tool in decoy_tools():
            assert len(tool.description) > 20

    def test_no_decoy_claims_to_answer_a_benchmark_question(self) -> None:
        """The decoys must be tempting, not deceptive about their subject.
        One that said "measure whether the circles overlap" would be a
        mislabelled tool, which is a different failure — and one no amount of
        reasoning could see through."""
        for tool in decoy_tools():
            described = tool.description.lower()
            assert "circle" not in described
            assert "cross" not in described
            assert "intersect" not in described
