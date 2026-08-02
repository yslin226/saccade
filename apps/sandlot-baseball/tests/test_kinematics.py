"""Tests for the baseball metrics.

These are the referee over what a session claims, so they get the treatment
rule 7 asks for: normal, boundary and degenerate for every function.

Coordinates are hand-written rather than detected. A metric that can only be
tested against a video is a metric whose failures are invisible.
"""

from __future__ import annotations

import math

import pytest
from sandlot.domain.kinematics import (
    MIN_ELBOW_DEGREES,
    MIN_SPAN_PX,
    centre_of_mass,
    elbow_valgus,
    hip_shoulder_separation,
    kinetic_chain_order,
    stride_length,
    torso_length,
)
from sandlot.domain.models import Frame, JointReading


def joint(name: str, x: float, y: float, confidence: float = 0.9) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=confidence)


def frame(*joints: JointReading, index: int = 0, timestamp: float = 0.0) -> Frame:
    return Frame(index=index, timestamp=timestamp, joints=joints)


def upright(
    *,
    shoulder_y: float = 100.0,
    hip_y: float = 300.0,
    shoulder_tilt: float = 0.0,
    hip_tilt: float = 0.0,
) -> Frame:
    """A person facing the camera, with optional tilt on each line.

    Tilt is in pixels of vertical offset between left and right, which keeps
    the fixtures readable: a hip_tilt of 100 on a 100px-wide hip line is 45
    degrees.
    """
    return frame(
        joint("L shoulder", 200, shoulder_y),
        joint("R shoulder", 300, shoulder_y + shoulder_tilt),
        joint("L hip", 210, hip_y),
        joint("R hip", 310, hip_y + hip_tilt),
        joint("L ankle", 205, 600),
        joint("R ankle", 315, 600),
    )


class TestTorsoLength:
    def test_normal_case(self) -> None:
        assert torso_length(upright()) == pytest.approx(200.0, abs=1.0)

    def test_it_is_measured_between_midpoints(self) -> None:
        """Not between one shoulder and one hip, which would grow when the
        subject turns."""
        square = frame(
            joint("L shoulder", 0, 0),
            joint("R shoulder", 100, 0),
            joint("L hip", 0, 100),
            joint("R hip", 100, 100),
        )
        assert torso_length(square) == pytest.approx(100.0)

    def test_missing_a_hip_gives_nothing(self) -> None:
        partial = frame(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 300, 100),
            joint("L hip", 210, 300),
        )
        assert torso_length(partial) is None

    def test_a_collapsed_shoulder_line_gives_nothing(self) -> None:
        """Two shoulders at the same pixel is a failed detection, and using
        it as a scale turns every ordinary distance into a catastrophe."""
        collapsed = frame(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 202, 100),
            joint("L hip", 210, 300),
            joint("R hip", 310, 300),
        )
        assert torso_length(collapsed) is None

    def test_a_collapsed_torso_gives_nothing(self) -> None:
        """Shoulders and hips at the same height: the person is not folded,
        the detector has failed."""
        flat = frame(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 300, 100),
            joint("L hip", 200, 105),
            joint("R hip", 300, 105),
        )
        assert torso_length(flat) is None

    def test_an_empty_frame_gives_nothing(self) -> None:
        assert torso_length(frame()) is None


class TestHipShoulderSeparation:
    def test_square_shoulders_and_hips_are_zero(self) -> None:
        assert hip_shoulder_separation(upright()) == pytest.approx(0.0)

    def test_a_coiled_torso_reads_the_angle(self) -> None:
        """Hips 45 degrees open with the shoulders still square."""
        coiled = upright(hip_tilt=100.0)
        assert hip_shoulder_separation(coiled) == pytest.approx(45.0)

    def test_it_is_unsigned(self) -> None:
        """A left-hander coiling the other way reads the same number, which
        is what a comparison against your own history wants."""
        one_way = hip_shoulder_separation(upright(hip_tilt=100.0))
        other = hip_shoulder_separation(upright(hip_tilt=-100.0))
        assert one_way == pytest.approx(other)

    def test_it_never_exceeds_ninety(self) -> None:
        """A line has no head or tail: 170 degrees apart is 10 from
        parallel."""
        for tilt in (-400.0, -100.0, -10.0, 10.0, 100.0, 400.0):
            separation = hip_shoulder_separation(upright(hip_tilt=tilt))
            assert separation is not None
            assert 0.0 <= separation <= 90.0

    def test_perpendicular_lines_read_ninety(self) -> None:
        crossed = frame(
            joint("L shoulder", 0, 0),
            joint("R shoulder", 100, 0),
            joint("L hip", 50, 200),
            joint("R hip", 50, 300),
        )
        assert hip_shoulder_separation(crossed) == pytest.approx(90.0)

    def test_a_missing_line_gives_nothing(self) -> None:
        assert hip_shoulder_separation(frame(joint("L shoulder", 0, 0))) is None

    def test_a_collapsed_line_gives_nothing(self) -> None:
        collapsed = frame(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 201, 100),
            joint("L hip", 210, 300),
            joint("R hip", 310, 300),
        )
        assert hip_shoulder_separation(collapsed) is None


class TestElbowValgus:
    def test_a_straight_arm_is_180(self) -> None:
        arm = frame(
            joint("R shoulder", 100, 100),
            joint("R elbow", 200, 100),
            joint("R wrist", 300, 100),
        )
        assert elbow_valgus(arm, side="R") == pytest.approx(180.0)

    def test_a_right_angle_reads_ninety(self) -> None:
        arm = frame(
            joint("L shoulder", 100, 100),
            joint("L elbow", 200, 100),
            joint("L wrist", 200, 200),
        )
        assert elbow_valgus(arm, side="L") == pytest.approx(90.0)

    def test_each_side_is_measured_separately(self) -> None:
        both = frame(
            joint("L shoulder", 100, 100),
            joint("L elbow", 200, 100),
            joint("L wrist", 300, 100),
            joint("R shoulder", 100, 400),
            joint("R elbow", 200, 400),
            joint("R wrist", 200, 500),
        )
        assert elbow_valgus(both, side="L") == pytest.approx(180.0)
        assert elbow_valgus(both, side="R") == pytest.approx(90.0)

    def test_a_missing_wrist_gives_nothing(self) -> None:
        partial = frame(joint("R shoulder", 100, 100), joint("R elbow", 200, 100))
        assert elbow_valgus(partial, side="R") is None

    def test_a_collapsed_forearm_gives_nothing(self) -> None:
        """Elbow and wrist at the same point: no direction, so no angle. The
        primitive would raise; here it is an ordinary failed detection."""
        collapsed = frame(
            joint("R shoulder", 100, 100),
            joint("R elbow", 200, 100),
            joint("R wrist", 203, 100),
        )
        assert elbow_valgus(collapsed, side="R") is None

    def test_an_unknown_side_is_an_error(self) -> None:
        """Not None. Silently measuring the other arm returns a number that
        looks fine and describes the wrong limb."""
        with pytest.raises(ValueError, match="side must be"):
            elbow_valgus(upright(), side="left")

    def test_an_anatomically_impossible_angle_gives_nothing(self) -> None:
        """15 degrees is a hand folded past the biceps. Measured for real on
        happy/林永閎.MOV at frame 17, where MediaPipe put the wrist back
        toward the shoulder at 0.78 confidence — and since the metric takes
        an extremum, that one frame became the answer."""
        folded = frame(
            joint("R shoulder", 408, 520),
            joint("R elbow", 354, 605),
            joint("R wrist", 413, 552),
        )
        assert elbow_valgus(folded, side="R") is None

    def test_a_tight_but_possible_bend_is_kept(self) -> None:
        """The threshold sits below anatomical maximum flexion, so a
        genuinely extreme delivery is not discarded."""
        tight = frame(
            joint("R shoulder", 100, 100),
            joint("R elbow", 200, 100),
            joint("R wrist", 130, 155),
        )
        angle = elbow_valgus(tight, side="R")
        assert angle is not None
        assert MIN_ELBOW_DEGREES <= angle < 45.0

    def test_the_limit_is_a_stated_guess_not_a_sourced_figure(self) -> None:
        """25 is where this project guessed "impossible" starts, having
        measured 15.4 on a real clip. It has not been checked against a
        source, and a threshold set too high discards real deliveries
        silently — so the number is pinned here to make a change to it
        deliberate."""
        assert MIN_ELBOW_DEGREES == 25.0


class TestStrideLength:
    def test_it_is_reported_in_torso_lengths(self) -> None:
        """110px of stride over a 200px torso."""
        stride = stride_length(upright())
        assert stride is not None
        assert stride == pytest.approx(110.0 / 200.0, abs=0.02)

    def test_the_same_stride_scores_the_same_at_any_distance(self) -> None:
        """Filming from further away must not change the measurement."""
        near = frame(
            joint("L shoulder", 0, 0),
            joint("R shoulder", 200, 0),
            joint("L hip", 0, 400),
            joint("R hip", 200, 400),
            joint("L ankle", 0, 800),
            joint("R ankle", 200, 800),
        )
        far = frame(
            joint("L shoulder", 0, 0),
            joint("R shoulder", 100, 0),
            joint("L hip", 0, 200),
            joint("R hip", 100, 200),
            joint("L ankle", 0, 400),
            joint("R ankle", 100, 400),
        )
        assert stride_length(near) == pytest.approx(stride_length(far))

    def test_feet_together_is_near_zero(self) -> None:
        together = frame(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 300, 100),
            joint("L hip", 200, 300),
            joint("R hip", 300, 300),
            joint("L ankle", 248, 600),
            joint("R ankle", 252, 600),
        )
        stride = stride_length(together)
        assert stride is None or stride < 0.1

    def test_a_missing_ankle_gives_nothing(self) -> None:
        no_feet = frame(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 300, 100),
            joint("L hip", 200, 300),
            joint("R hip", 300, 300),
        )
        assert stride_length(no_feet) is None

    def test_no_torso_means_no_scale_and_no_answer(self) -> None:
        feet_only = frame(joint("L ankle", 100, 600), joint("R ankle", 300, 600))
        assert stride_length(feet_only) is None


class TestCentreOfMass:
    def test_a_symmetric_body_centres_on_the_midline(self) -> None:
        centre = centre_of_mass(upright())
        assert centre is not None
        assert centre[0] == pytest.approx(260.0, abs=15.0)

    def test_mass_pulls_the_centre(self) -> None:
        """Knees carry the most weight of the tracked joints, so moving them
        moves the centre more than moving the wrists does."""
        base = frame(
            joint("L knee", 100, 500),
            joint("R knee", 200, 500),
            joint("L wrist", 100, 300),
            joint("R wrist", 200, 300),
        )
        knees_moved = frame(
            joint("L knee", 300, 500),
            joint("R knee", 400, 500),
            joint("L wrist", 100, 300),
            joint("R wrist", 200, 300),
        )
        wrists_moved = frame(
            joint("L knee", 100, 500),
            joint("R knee", 200, 500),
            joint("L wrist", 300, 300),
            joint("R wrist", 400, 300),
        )
        start = centre_of_mass(base)
        by_knees = centre_of_mass(knees_moved)
        by_wrists = centre_of_mass(wrists_moved)
        assert start is not None and by_knees is not None and by_wrists is not None
        assert abs(by_knees[0] - start[0]) > abs(by_wrists[0] - start[0])

    def test_the_trunk_carries_weight(self) -> None:
        """It is about half of body mass and has no landmark of its own, so
        it is spread over the four points bounding it. An earlier version
        gave those zero weight and silently discarded half the body — a
        hitter who rotated their trunk without moving their feet registered
        as having transferred nothing."""
        with_trunk = frame(
            joint("L knee", 100, 500),
            joint("R knee", 200, 500),
            joint("L shoulder", 900, 100),
            joint("R shoulder", 950, 100),
        )
        without = frame(joint("L knee", 100, 500), joint("R knee", 200, 500))

        moved = centre_of_mass(with_trunk)
        legs_only = centre_of_mass(without)
        assert moved is not None and legs_only is not None
        assert moved[0] > legs_only[0] + 100

    def test_the_trunk_outweighs_the_arms(self) -> None:
        """Half the body against a few percent. A model where a wrist moved
        the centre as much as a shoulder would be describing a different
        animal."""
        by_shoulders = frame(
            joint("L knee", 100, 500),
            joint("R knee", 200, 500),
            joint("L shoulder", 900, 100),
            joint("R shoulder", 950, 100),
        )
        by_wrists = frame(
            joint("L knee", 100, 500),
            joint("R knee", 200, 500),
            joint("L wrist", 900, 100),
            joint("R wrist", 950, 100),
        )
        base = centre_of_mass(frame(joint("L knee", 100, 500), joint("R knee", 200, 500)))
        shoulders = centre_of_mass(by_shoulders)
        wrists = centre_of_mass(by_wrists)
        assert base is not None and shoulders is not None and wrists is not None
        assert abs(shoulders[0] - base[0]) > abs(wrists[0] - base[0])

    def test_a_single_tracked_joint_still_gives_a_centre(self) -> None:
        """Shifted, and honestly so: the rest of the mass is genuinely
        unaccounted for, and the caller can see how few joints were found."""
        assert centre_of_mass(frame(joint("L shoulder", 7, 9))) == pytest.approx((7.0, 9.0))

    def test_a_frame_with_no_tracked_joints_gives_nothing(self) -> None:
        assert centre_of_mass(frame(JointReading(name="nose", x=1, y=2, confidence=0.9))) is None

    def test_an_empty_frame_gives_nothing(self) -> None:
        assert centre_of_mass(frame()) is None


class TestKineticChainOrder:
    def rotating(self, angles: dict[str, list[float]]) -> list[Frame]:
        """Frames where each named line rotates through the given angles."""
        frames = []
        for i in range(len(next(iter(angles.values())))):
            joints = [
                joint("L hip", 200, 300),
                joint(
                    "R hip",
                    200 + 100 * math.cos(math.radians(angles["hips"][i])),
                    300 + 100 * math.sin(math.radians(angles["hips"][i])),
                ),
                joint("L shoulder", 200, 100),
                joint(
                    "R shoulder",
                    200 + 100 * math.cos(math.radians(angles["shoulders"][i])),
                    100 + 100 * math.sin(math.radians(angles["shoulders"][i])),
                ),
            ]
            frames.append(frame(*joints, index=i, timestamp=i / 60))
        return frames

    def test_hips_peaking_first_are_ordered_first(self) -> None:
        frames = self.rotating(
            {
                # Hips make their big move between frames 1 and 2, the
                # shoulders between 3 and 4.
                "hips": [0, 0, 60, 60, 60],
                "shoulders": [0, 0, 0, 0, 60],
            }
        )
        order = kinetic_chain_order(frames, segments=("hips", "shoulders"))
        assert order == ["hips", "shoulders"]

    def test_the_reverse_is_detected_too(self) -> None:
        """All-arm delivery: the shoulders fire before the hips."""
        frames = self.rotating(
            {
                "hips": [0, 0, 0, 0, 60],
                "shoulders": [0, 0, 60, 60, 60],
            }
        )
        order = kinetic_chain_order(frames, segments=("hips", "shoulders"))
        assert order == ["shoulders", "hips"]

    def test_two_frames_are_not_enough(self) -> None:
        """A peak needs neighbours on both sides to be a peak rather than an
        endpoint."""
        frames = self.rotating({"hips": [0, 30], "shoulders": [0, 30]})
        assert kinetic_chain_order(frames, segments=("hips", "shoulders")) is None

    def test_frames_with_nothing_measurable_give_nothing(self) -> None:
        blank = [frame(index=i) for i in range(5)]
        assert kinetic_chain_order(blank, segments=("hips", "shoulders")) is None

    def test_an_unknown_segment_is_an_error(self) -> None:
        frames = self.rotating({"hips": [0, 10, 20], "shoulders": [0, 10, 20]})
        with pytest.raises(ValueError, match="unknown segment"):
            kinetic_chain_order(frames, segments=("elbows",))

    def test_the_arm_being_timed_can_be_chosen(self) -> None:
        """It used to be hardcoded to the right. A left-hander was having the
        chain timed against the arm that barely moves, which produces an
        ordering that looks reasonable and describes the wrong limb."""
        left_arm_moves = [
            frame(
                joint("L shoulder", 100, 100),
                joint("L elbow", 200, 100),
                joint("L wrist", 300, 100 + 40 * i),
                joint("R shoulder", 100, 400),
                joint("R elbow", 200, 400),
                joint("R wrist", 300, 400),
                index=i,
            )
            for i in range(4)
        ]
        # Both sides report a peak — a still arm has one, at zero. What
        # changes is which arm the ordering describes, and the only way to
        # see that is to give the two arms different timings.
        assert kinetic_chain_order(left_arm_moves, segments=("wrist",), side="L") == ["wrist"]
        assert kinetic_chain_order(left_arm_moves, segments=("wrist",), side="R") == ["wrist"]

    def test_the_two_arms_can_order_differently(self) -> None:
        """The consequence of the previous test, and the reason the argument
        exists: the left wrist fires before the hips and the right after, so
        which arm is timed decides whether the chain reads ground-up."""
        arms = [
            frame(
                joint("L hip", 200, 300),
                joint("R hip", 300, 300 + (80 if i >= 2 else 0)),
                joint("L elbow", 100, 100),
                joint("L wrist", 200, 100 + (80 if i >= 1 else 0)),
                joint("R elbow", 400, 100),
                joint("R wrist", 500, 100 + (80 if i >= 4 else 0)),
                index=i,
            )
            for i in range(5)
        ]
        left = kinetic_chain_order(arms, segments=("hips", "wrist"), side="L")
        right = kinetic_chain_order(arms, segments=("hips", "wrist"), side="R")

        assert left == ["wrist", "hips"]
        assert right == ["hips", "wrist"]

    def test_an_unknown_side_is_an_error(self) -> None:
        """Not a silent fallback to the right arm — the same reason
        elbow_valgus raises."""
        frames = self.rotating({"hips": [0, 10, 20], "shoulders": [0, 10, 20]})
        with pytest.raises(ValueError, match="side must be"):
            kinetic_chain_order(frames, side="left")

    def test_the_elbow_and_wrist_segments_are_recognised(self) -> None:
        """Named in the default segment list, so they must not raise."""
        arm_frames = [
            frame(
                joint("R shoulder", 100, 100),
                joint("R elbow", 200, 100 + 20 * i),
                joint("R wrist", 300, 100),
                index=i,
            )
            for i in range(4)
        ]
        order = kinetic_chain_order(arm_frames, segments=("elbow", "wrist"))
        assert order is not None
        assert set(order) <= {"elbow", "wrist"}

    def test_an_occlusion_does_not_become_the_peak(self) -> None:
        """A rate computed across a gap is an average, and would place the
        peak wherever the occlusion happened to be."""
        frames = [
            frame(joint("L hip", 200, 300), joint("R hip", 300, 300), index=0),
            frame(joint("L hip", 200, 300), joint("R hip", 300, 305), index=1),
            frame(index=2),  # occluded
            frame(joint("L hip", 200, 300), joint("R hip", 200, 400), index=3),
            frame(joint("L hip", 200, 300), joint("R hip", 195, 400), index=4),
        ]
        order = kinetic_chain_order(frames, segments=("hips",))
        assert order == ["hips"]


def test_the_minimum_span_is_a_pixel_count() -> None:
    """Not a fraction. It guards against collapsed detections, and those are
    measured in pixels whatever the subject's size."""
    assert MIN_SPAN_PX > 1.0
