"""Tests for the geometric predicates.

These functions are the referee over what the VLM claims, so they get the
strictest treatment in the project (CLAUDE.md rule 7): normal, boundary and
degenerate cases for every function.
"""

from __future__ import annotations

import math

import pytest

from saccade.geometry.shapes import (
    circles_overlap,
    count_line_intersections,
    distance,
    segments_intersect,
)


class TestDistance:
    def test_normal_case(self) -> None:
        assert distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_identical_points_give_zero(self) -> None:
        assert distance((7, 7), (7, 7)) == 0.0

    def test_negative_coordinates(self) -> None:
        assert distance((-3, -4), (0, 0)) == pytest.approx(5.0)

    def test_symmetric(self) -> None:
        assert distance((1, 2), (9, 14)) == pytest.approx(distance((9, 14), (1, 2)))

    def test_floats(self) -> None:
        assert distance((0.0, 0.0), (1.0, 1.0)) == pytest.approx(math.sqrt(2))


class TestCirclesOverlap:
    def test_clearly_overlapping(self) -> None:
        # Centres 47px apart, radii sum to 52 — the spec's worked example.
        assert circles_overlap((0, 0), 25, (47, 0), 27) is True

    def test_clearly_separate(self) -> None:
        assert circles_overlap((0, 0), 10, (100, 0), 10) is False

    def test_one_inside_the_other(self) -> None:
        assert circles_overlap((0, 0), 50, (5, 0), 2) is True

    def test_concentric(self) -> None:
        assert circles_overlap((0, 0), 10, (0, 0), 3) is True

    def test_exactly_tangent_defaults_to_not_overlapping(self) -> None:
        """Touching is not overlapping unless the caller says so."""
        assert circles_overlap((0, 0), 10, (20, 0), 10) is False

    def test_exactly_tangent_when_tangent_counts(self) -> None:
        assert circles_overlap((0, 0), 10, (20, 0), 10, tangent_counts=True) is True

    def test_just_inside_tangency_overlaps(self) -> None:
        assert circles_overlap((0, 0), 10, (19.999, 0), 10) is True

    def test_just_outside_tangency_does_not_overlap(self) -> None:
        assert circles_overlap((0, 0), 10, (20.001, 0), 10) is False

    def test_tangency_detected_on_a_diagonal(self) -> None:
        # Centres at distance 5, radii 2 + 3.
        assert circles_overlap((0, 0), 2.0, (3.0, 4.0), 3.0) is False
        assert circles_overlap((0, 0), 2.0, (3.0, 4.0), 3.0, tangent_counts=True) is True

    def test_zero_radius_rejected(self) -> None:
        with pytest.raises(ValueError, match="radii must be positive"):
            circles_overlap((0, 0), 0, (1, 0), 1)

    def test_negative_radius_rejected(self) -> None:
        with pytest.raises(ValueError, match="radii must be positive"):
            circles_overlap((0, 0), 5, (1, 0), -1)


class TestSegmentsIntersect:
    def test_plain_crossing(self) -> None:
        assert segments_intersect(((0, 0), (10, 10)), ((0, 10), (10, 0))) is True

    def test_disjoint(self) -> None:
        assert segments_intersect(((0, 0), (1, 1)), ((5, 5), (6, 6))) is False

    def test_parallel_never_meet(self) -> None:
        assert segments_intersect(((0, 0), (10, 0)), ((0, 5), (10, 5))) is False

    def test_touching_at_an_endpoint(self) -> None:
        assert segments_intersect(((0, 0), (5, 5)), ((5, 5), (10, 0))) is True

    def test_t_junction(self) -> None:
        assert segments_intersect(((0, 0), (10, 0)), ((5, 0), (5, 10))) is True

    def test_collinear_and_overlapping(self) -> None:
        assert segments_intersect(((0, 0), (10, 0)), ((5, 0), (15, 0))) is True

    def test_collinear_but_apart(self) -> None:
        assert segments_intersect(((0, 0), (5, 0)), ((10, 0), (15, 0))) is False

    def test_collinear_touching_at_one_point(self) -> None:
        assert segments_intersect(((0, 0), (5, 0)), ((5, 0), (10, 0))) is True

    def test_second_segment_ends_on_the_first(self) -> None:
        """Only the o2 branch can decide this one."""
        assert segments_intersect(((0, 0), (10, 0)), ((20, 5), (5, 0))) is True

    def test_first_segment_starts_on_the_second(self) -> None:
        """Only the o3 branch can decide this one."""
        assert segments_intersect(((5, 5), (20, 20)), ((0, 0), (10, 10))) is True

    def test_first_segment_ends_on_the_second(self) -> None:
        """Only the o4 branch can decide this one."""
        assert segments_intersect(((20, 20), (5, 5)), ((0, 0), (10, 10))) is True

    def test_would_cross_if_extended_but_does_not(self) -> None:
        assert segments_intersect(((0, 0), (1, 1)), ((0, 10), (1, 9))) is False

    def test_vertical_and_horizontal(self) -> None:
        assert segments_intersect(((5, 0), (5, 10)), ((0, 5), (10, 5))) is True

    def test_degenerate_point_on_a_segment(self) -> None:
        assert segments_intersect(((5, 5), (5, 5)), ((0, 5), (10, 5))) is True

    def test_degenerate_point_off_a_segment(self) -> None:
        assert segments_intersect(((5, 9), (5, 9)), ((0, 5), (10, 5))) is False


class TestCountLineIntersections:
    def test_single_crossing_pair(self) -> None:
        lines = [((0, 0), (10, 10)), ((0, 10), (10, 0))]
        assert count_line_intersections(lines) == 1

    def test_three_mutually_crossing_lines(self) -> None:
        lines = [
            ((0, 0), (10, 10)),
            ((0, 10), (10, 0)),
            ((5, -5), (5, 15)),
        ]
        assert count_line_intersections(lines) == 3

    def test_parallel_lines_never_cross(self) -> None:
        lines = [((0, y), (10, y)) for y in (0, 5, 10)]
        assert count_line_intersections(lines) == 0

    def test_empty_input(self) -> None:
        assert count_line_intersections([]) == 0

    def test_single_line_cannot_cross_anything(self) -> None:
        assert count_line_intersections([((0, 0), (1, 1))]) == 0

    def test_counts_pairs_not_points(self) -> None:
        """Three lines through one shared point are three intersecting pairs."""
        lines = [
            ((-10, 0), (10, 0)),
            ((0, -10), (0, 10)),
            ((-10, -10), (10, 10)),
        ]
        assert count_line_intersections(lines) == 3

    def test_mixed_crossing_and_disjoint(self) -> None:
        lines = [
            ((0, 0), (10, 10)),
            ((0, 10), (10, 0)),
            ((100, 100), (110, 110)),
        ]
        assert count_line_intersections(lines) == 1


class TestOptionalOpenCV:
    def test_geometry_package_imports_when_opencv_is_installed(self) -> None:
        """The dev environment installs all extras, so the package must import."""
        import saccade.geometry as geometry

        assert geometry.circles_overlap is circles_overlap

    def test_missing_opencv_gives_an_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate an install without the geometry extra."""
        import builtins
        import sys

        real_import = builtins.__import__

        def fail_on_cv2(name: str, *args: object, **kwargs: object) -> object:
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.delitem(sys.modules, "saccade.geometry", raising=False)
        monkeypatch.delitem(sys.modules, "cv2", raising=False)
        monkeypatch.setattr(builtins, "__import__", fail_on_cv2)

        with pytest.raises(ImportError, match=r"saccade-vision\[geometry\]"):
            import saccade.geometry  # noqa: F401

    def test_shapes_module_works_without_opencv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pure math must not be blocked by a missing optional dependency."""
        import sys

        monkeypatch.delitem(sys.modules, "cv2", raising=False)
        from saccade.geometry.shapes import circles_overlap as pure_overlap

        assert pure_overlap((0, 0), 5, (1, 0), 5) is True
