"""Tests for the line-crossing tool.

Another referee, so the same bar as the circle tool: a confident wrong count
is worse than no count, because the verifier acts on it.

Curves are drawn to order here, so the expected number of crossings is known
rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from benchmarks.blindtest.tools.lines import (
    count_crossings,
    line_tool,
    measure_lines,
    trace_curves,
)

RED = (255, 60, 60)
BLUE = (60, 60, 255)


def plot(
    red_points: list[tuple[int, int]],
    blue_points: list[tuple[int, int]],
    size: tuple[int, int] = (300, 200),
) -> Image.Image:
    """Draw two polylines the way the dataset plots them."""
    image = Image.new("RGB", size, "white")
    pen = ImageDraw.Draw(image)
    pen.line(red_points, fill=RED, width=3)
    pen.line(blue_points, fill=BLUE, width=3)
    return image


class TestTracing:
    def test_a_flat_line_traces_to_a_constant_row(self) -> None:
        image = plot([(10, 50), (290, 50)], [(10, 150), (290, 150)])
        red, blue = trace_curves(image)

        assert np.nanmean(red) == pytest.approx(50, abs=2)
        assert np.nanmean(blue) == pytest.approx(150, abs=2)

    def test_columns_without_the_curve_are_nan(self) -> None:
        """A column the curve does not reach must not be filled with a guess."""
        image = plot([(10, 50), (100, 50)], [(10, 150), (290, 150)])
        red, _ = trace_curves(image)

        assert np.isnan(red[200])
        assert not np.isnan(red[50])

    def test_the_two_colours_are_kept_apart(self) -> None:
        image = plot([(10, 50), (290, 50)], [(10, 150), (290, 150)])
        red, blue = trace_curves(image)
        assert np.nanmean(red) < np.nanmean(blue)


class TestCounting:
    def test_parallel_lines_never_cross(self) -> None:
        image = plot([(10, 50), (290, 50)], [(10, 150), (290, 150)])
        assert measure_lines(image)["crossings"] == 0

    def test_one_crossing(self) -> None:
        image = plot([(10, 50), (290, 150)], [(10, 150), (290, 50)])
        assert measure_lines(image)["crossings"] == 1

    def test_two_crossings(self) -> None:
        image = plot(
            [(10, 50), (150, 150), (290, 50)],
            [(10, 100), (150, 100), (290, 100)],
        )
        assert measure_lines(image)["crossings"] == 2

    def test_three_crossings(self) -> None:
        image = plot(
            [(10, 60), (100, 140), (200, 60), (290, 140)],
            [(10, 100), (290, 100)],
        )
        assert measure_lines(image)["crossings"] == 3

    def test_a_near_miss_is_not_a_crossing(self) -> None:
        """Approaching and retreating without swapping over."""
        image = plot(
            [(10, 40), (150, 88), (290, 40)],
            [(10, 160), (150, 112), (290, 160)],
        )
        assert measure_lines(image)["crossings"] == 0


class TestCountingFromArrays:
    """The counting rule on its own, without the drawing in the way."""

    def test_no_shared_columns_counts_nothing(self) -> None:
        red = np.array([1.0, 2.0, np.nan, np.nan])
        blue = np.array([np.nan, np.nan, 3.0, 4.0])
        assert count_crossings(red, blue) == 0

    def test_a_single_swap(self) -> None:
        red = np.array([10.0, 20.0, 30.0, 40.0])
        blue = np.array([40.0, 30.0, 20.0, 10.0])
        assert count_crossings(red, blue) == 1

    def test_staying_on_one_side(self) -> None:
        red = np.array([10.0, 10.0, 10.0])
        blue = np.array([50.0, 50.0, 50.0])
        assert count_crossings(red, blue) == 0

    def test_a_run_of_contact_counts_once(self) -> None:
        """Curves that travel together for a while met once, not once per column."""
        red = np.array([10.0, 30.0, 30.0, 30.0, 30.0, 50.0])
        blue = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
        assert count_crossings(red, blue) == 1

    def test_empty_input(self) -> None:
        assert count_crossings(np.array([]), np.array([])) == 0


class TestPartialViews:
    """Once the agent magnifies a corner, one curve is often gone."""

    def test_a_view_with_one_curve_gives_no_count(self) -> None:
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).line([(10, 100), (190, 100)], fill=RED, width=3)

        result = measure_lines(image)
        assert "crossings" not in result

    def test_a_blank_view_gives_no_count(self) -> None:
        blank = Image.new("RGB", (120, 120), "white")
        assert "crossings" not in measure_lines(blank)


class TestToolContract:
    def test_a_count_is_a_measurement(self) -> None:
        image = plot([(10, 50), (290, 150)], [(10, 150), (290, 50)])
        result = line_tool().fn(image=image, viewport=None)

        assert result.is_measurement is True
        assert result.value["crossings"] == 1

    def test_no_count_is_not_a_measurement(self) -> None:
        """Reporting zero would let "I could not see them" pass as "never"."""
        blank = Image.new("RGB", (120, 120), "white")
        result = line_tool().fn(image=blank, viewport=None)

        assert result.is_measurement is False

    def test_the_tool_is_named_for_the_evidence_chain(self) -> None:
        tool = line_tool()
        assert tool.name == "line_geometry"
        assert tool.description
