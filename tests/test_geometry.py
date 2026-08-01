"""Tests for the geometric predicates.

These functions are the referee over what the VLM claims, so they get the
strictest treatment in the project (CLAUDE.md rule 7): normal, boundary and
degenerate cases for every function.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap

import pytest

from saccade.geometry.shapes import (
    angle_between,
    bbox_iou,
    bearing,
    centroid,
    circles_overlap,
    count_line_intersections,
    distance,
    point_to_segment_distance,
    segments_intersect,
    smooth,
    speed,
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


class TestAngleBetween:
    """Three points, angle at the middle one.

    Written for joint angles — elbow, knee, shoulder — but nothing here knows
    that. It is three coordinates.
    """

    def test_a_right_angle(self) -> None:
        assert angle_between((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)

    def test_a_straight_limb_is_180_not_0(self) -> None:
        """A fully extended arm reads 180°, which is the convention every
        biomechanics text uses and the opposite of the flexion angle."""
        assert angle_between((-1, 0), (0, 0), (1, 0)) == pytest.approx(180.0)

    def test_rays_along_the_same_direction_give_zero(self) -> None:
        assert angle_between((1, 0), (0, 0), (5, 0)) == pytest.approx(0.0)

    def test_it_is_unsigned(self) -> None:
        """Mirror images give the same angle: this measures separation, not
        direction. A caller needing handedness needs more than one angle."""
        above = angle_between((1, 0), (0, 0), (0, 1))
        below = angle_between((1, 0), (0, 0), (0, -1))
        assert above == pytest.approx(below)

    def test_distance_from_the_vertex_does_not_matter(self) -> None:
        """Only direction counts, so a subject filmed closer reads the same."""
        near = angle_between((1, 0), (0, 0), (0, 1))
        far = angle_between((100, 0), (0, 0), (0, 300))
        assert near == pytest.approx(far)

    def test_the_vertex_is_the_middle_argument(self) -> None:
        """Passing the vertex first measures a different angle and returns a
        plausible number, so the order is not a detail."""
        correct = angle_between((1, 0), (0, 0), (0, 1))
        swapped = angle_between((0, 0), (1, 0), (0, 1))
        assert correct != pytest.approx(swapped)

    def test_it_never_exceeds_180(self) -> None:
        for point in ((1, 1), (-1, 1), (-1, -1), (1, -1), (0, -5)):
            assert 0.0 <= angle_between((1, 0), (0, 0), point) <= 180.0

    def test_a_point_on_the_vertex_is_an_error(self) -> None:
        """A zero-length ray has no direction. Returning 0.0 would be a
        number a caller could compare against a threshold."""
        with pytest.raises(ValueError, match="coincides with the vertex"):
            angle_between((0, 0), (0, 0), (1, 1))

    def test_the_other_point_on_the_vertex_is_also_an_error(self) -> None:
        with pytest.raises(ValueError, match="coincides with the vertex"):
            angle_between((1, 1), (0, 0), (0, 0))

    def test_near_collinear_input_does_not_blow_up(self) -> None:
        """Rounding can push the cosine just past 1.0, where acos raises."""
        assert angle_between((1e8, 0), (0, 0), (1e8, 1e-8)) == pytest.approx(0.0, abs=1e-4)


class TestSpeed:
    def test_normal_case(self) -> None:
        assert speed((0, 0), (3, 4), 2.0) == pytest.approx(2.5)

    def test_no_movement_is_zero(self) -> None:
        assert speed((5, 5), (5, 5), 1.0) == 0.0

    def test_it_is_unsigned(self) -> None:
        """Speed, not velocity — reversing the direction changes nothing."""
        assert speed((0, 0), (10, 0), 1.0) == pytest.approx(speed((10, 0), (0, 0), 1.0))

    def test_units_come_from_the_caller(self) -> None:
        """Nothing here knows whether these are pixels or metres."""
        assert speed((0, 0), (100, 0), 0.04) == pytest.approx(2500.0)

    def test_zero_elapsed_time_is_an_error(self) -> None:
        """Infinite rather than large. Returning inf would propagate into a
        comparison that silently succeeds."""
        with pytest.raises(ValueError, match="dt must be positive"):
            speed((0, 0), (1, 1), 0.0)

    def test_negative_time_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            speed((0, 0), (1, 1), -0.5)


class TestBearing:
    """Where something is, as opposed to how far apart two things are."""

    def test_due_right_is_zero(self) -> None:
        assert bearing((0, 0), (10, 0)) == pytest.approx(0.0)

    def test_it_follows_the_pixel_grid_not_the_textbook(self) -> None:
        """y grows downward in an image, so 90 degrees is down the screen.
        The inputs are pixel coordinates; the convention follows them."""
        assert bearing((0, 0), (0, 10)) == pytest.approx(90.0)

    def test_due_left_is_180(self) -> None:
        assert bearing((0, 0), (-10, 0)) == pytest.approx(180.0)

    def test_it_wraps_rather_than_going_negative(self) -> None:
        assert bearing((0, 0), (0, -10)) == pytest.approx(270.0)

    def test_it_is_signed_unlike_angle_between(self) -> None:
        """Mirror images give different bearings. That is the whole reason
        this exists alongside angle_between, which cannot tell them apart."""
        assert bearing((0, 0), (10, 10)) != pytest.approx(bearing((0, 0), (10, -10)))

    def test_distance_does_not_change_the_bearing(self) -> None:
        assert bearing((0, 0), (1, 1)) == pytest.approx(bearing((0, 0), (500, 500)))

    def test_a_point_has_no_bearing_to_itself(self) -> None:
        """0.0 would be indistinguishable from due right."""
        with pytest.raises(ValueError, match="no bearing"):
            bearing((3, 4), (3, 4))

    def test_the_result_is_always_in_range(self) -> None:
        for target in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            assert 0.0 <= bearing((0, 0), target) < 360.0


class TestPointToSegmentDistance:
    """To the segment, not to the infinite line through it."""

    def test_a_point_beside_the_middle(self) -> None:
        assert point_to_segment_distance((5, 3), ((0, 0), (10, 0))) == pytest.approx(3.0)

    def test_a_point_on_the_segment_is_zero(self) -> None:
        assert point_to_segment_distance((5, 0), ((0, 0), (10, 0))) == pytest.approx(0.0)

    def test_a_point_past_the_end_measures_to_the_end(self) -> None:
        """The distinction from the infinite line. A point at x=20 is 10 from
        the segment's end, not 0 from the line it lies on."""
        assert point_to_segment_distance((20, 0), ((0, 0), (10, 0))) == pytest.approx(10.0)

    def test_a_point_past_the_start_measures_to_the_start(self) -> None:
        assert point_to_segment_distance((-6, 8), ((0, 0), (10, 0))) == pytest.approx(10.0)

    def test_the_endpoints_are_at_zero(self) -> None:
        segment = ((2, 3), (8, 9))
        assert point_to_segment_distance((2, 3), segment) == pytest.approx(0.0)
        assert point_to_segment_distance((8, 9), segment) == pytest.approx(0.0)

    def test_a_zero_length_segment_degenerates_to_a_point(self) -> None:
        """Not an error: the shortest distance to a point is still defined."""
        assert point_to_segment_distance((3, 4), ((0, 0), (0, 0))) == pytest.approx(5.0)

    def test_a_diagonal_segment(self) -> None:
        assert point_to_segment_distance((0, 10), ((0, 0), (10, 10))) == pytest.approx(
            math.sqrt(50)
        )


class TestBBoxIoU:
    def test_identical_boxes_score_one(self) -> None:
        box = (10.0, 10.0, 20.0, 20.0)
        assert bbox_iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self) -> None:
        assert bbox_iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0

    def test_half_overlap(self) -> None:
        """Two 10x10 boxes sharing a 5x10 strip: 50 over 150."""
        assert bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)

    def test_touching_along_an_edge_scores_zero(self) -> None:
        """A shared edge has no area, so there is no intersection."""
        assert bbox_iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0

    def test_a_box_contained_in_another(self) -> None:
        assert bbox_iou((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(16 / 100)

    def test_it_is_symmetric(self) -> None:
        a, b = (0.0, 0.0, 10.0, 10.0), (3.0, 4.0, 12.0, 6.0)
        assert bbox_iou(a, b) == pytest.approx(bbox_iou(b, a))

    def test_a_zero_area_box_is_an_error(self) -> None:
        """The union would be the other box's area, making the ratio look
        like a real answer when nothing was compared."""
        with pytest.raises(ValueError, match="positive width and height"):
            bbox_iou((0, 0, 0, 10), (0, 0, 10, 10))

    def test_a_negative_dimension_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="positive width and height"):
            bbox_iou((0, 0, 10, 10), (0, 0, 10, -5))


class TestSmooth:
    def test_a_flat_series_is_unchanged(self) -> None:
        assert smooth([5.0, 5.0, 5.0, 5.0]) == pytest.approx([5.0, 5.0, 5.0, 5.0])

    def test_a_spike_is_flattened(self) -> None:
        spiky = [1.0, 1.0, 9.0, 1.0, 1.0]
        assert max(smooth(spiky)) < 9.0

    def test_the_output_is_the_same_length(self) -> None:
        """So the index stays a frame number."""
        for n in (1, 2, 5, 17):
            assert len(smooth([float(i) for i in range(n)])) == n

    def test_the_ends_average_only_what_exists(self) -> None:
        """Truncated, not padded — the series never invents data outside its
        own range."""
        assert smooth([0.0, 3.0, 6.0], window=3)[0] == pytest.approx(1.5)

    def test_a_window_of_one_changes_nothing(self) -> None:
        values = [3.0, 1.0, 4.0, 1.0]
        assert smooth(values, window=1) == pytest.approx(values)

    def test_an_even_window_is_widened_rather_than_shifted(self) -> None:
        """A half-frame shift would misalign the result against its own
        timestamps, so window=4 behaves as 5."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert smooth(values, window=4) == pytest.approx(smooth(values, window=5))

    def test_an_empty_series_stays_empty(self) -> None:
        assert smooth([]) == []

    def test_a_window_wider_than_the_series_averages_everything(self) -> None:
        assert smooth([1.0, 2.0, 3.0], window=99) == pytest.approx([2.0, 2.0, 2.0])

    def test_a_zero_window_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="window must be at least 1"):
            smooth([1.0, 2.0], window=0)


class TestCentroid:
    def test_normal_case(self) -> None:
        assert centroid([(0, 0), (2, 0), (1, 3)]) == pytest.approx((1.0, 1.0))

    def test_one_point_is_itself(self) -> None:
        assert centroid([(4, 7)]) == pytest.approx((4.0, 7.0))

    def test_two_points_give_the_midpoint(self) -> None:
        assert centroid([(0, 0), (10, 20)]) == pytest.approx((5.0, 10.0))

    def test_negative_coordinates(self) -> None:
        assert centroid([(-2, -2), (2, 2)]) == pytest.approx((0.0, 0.0))

    def test_duplicates_are_weighted(self) -> None:
        """It is the mean of the points given, not of the distinct positions.
        A caller listing a point twice has doubled its weight, deliberately
        or otherwise."""
        assert centroid([(0, 0), (0, 0), (3, 0)]) == pytest.approx((1.0, 0.0))

    def test_an_empty_list_is_an_error(self) -> None:
        """The mean of nothing is not a position, and the origin is a
        coordinate a caller could plot."""
        with pytest.raises(ValueError, match="empty list"):
            centroid([])


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

    def test_missing_opencv_gives_an_actionable_message(self) -> None:
        """Simulate an install without the geometry extra.

        Run in a subprocess rather than by patching sys.modules: unimporting
        saccade.geometry in this process leaks into whatever test collects
        next, which is exactly the kind of order-dependent failure that only
        shows up on someone else's machine.
        """
        script = textwrap.dedent(
            """
            import sys

            # Make `import cv2` fail the way a missing extra would.
            class Blocker:
                def find_module(self, name, path=None):
                    return None
                def find_spec(self, name, path=None, target=None):
                    if name == "cv2" or name.startswith("cv2."):
                        raise ImportError("No module named 'cv2'")
                    return None

            sys.modules.pop("cv2", None)
            sys.meta_path.insert(0, Blocker())

            try:
                import saccade.geometry
            except ImportError as exc:
                print(exc)
                sys.exit(0)
            sys.exit("expected ImportError, got a successful import")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "saccade-vision[geometry]" in completed.stdout

    def test_shapes_module_works_without_opencv(self) -> None:
        """Pure math must not be blocked by a missing optional dependency.

        ``from saccade.geometry.shapes import ...`` would import the parent
        package first and hit the OpenCV gate, so this loads the module
        directly — which is what the verifier will need to do in M1 when
        only the pure predicates are wanted.
        """
        script = textwrap.dedent(
            """
            import importlib.util
            import pathlib
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name == "cv2" or name.startswith("cv2."):
                        raise ImportError("No module named 'cv2'")
                    return None

            sys.modules.pop("cv2", None)
            sys.meta_path.insert(0, Blocker())

            import saccade
            path = pathlib.Path(saccade.__file__).parent / "geometry" / "shapes.py"
            spec = importlib.util.spec_from_file_location("_shapes", path)
            shapes = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(shapes)

            assert shapes.circles_overlap((0, 0), 5, (1, 0), 5) is True
            assert shapes.count_line_intersections(
                [((0, 0), (10, 10)), ((0, 10), (10, 0))]
            ) == 1
            print("ok")
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "ok" in completed.stdout
