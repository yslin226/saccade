"""Tests for the pose continuity check.

Another referee, so the same bar: it decides whether a frame's measurements
may be trusted, and a wrong verdict propagates into every angle computed from
that frame.

The numbers in these tests come from a real clip — happy/林永閎.MOV, where
MediaPipe reported four physically impossible readings across 450 frames and
attached confidence between 0.79 and 0.99 to every one of them.
"""

from __future__ import annotations

import pytest
from PIL import Image

from benchmarks.pose_probe.continuity import (
    FALLBACK_TRAVEL_PX,
    MAX_TRAVEL_PER_SHOULDER_WIDTH,
    JointReading,
    continuity_tool,
    implausible_joints,
    shoulder_width,
)

# A subject whose shoulders are 100px apart, so the travel threshold is
# 100 * 0.55 = 55px. Stated explicitly because every expectation below
# depends on it.
SHOULDERS = 100.0
THRESHOLD = SHOULDERS * MAX_TRAVEL_PER_SHOULDER_WIDTH


def reading(name: str, x: float, y: float, confidence: float = 0.9) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=confidence)


def with_shoulders(*others: JointReading, width: float = SHOULDERS) -> list[JointReading]:
    """Readings including a shoulder pair, which sets the scale."""
    return [
        reading("L shoulder", 400, 300),
        reading("R shoulder", 400 + width, 300),
        *others,
    ]


class TestPlausibleMotion:
    def test_a_still_joint_is_plausible(self) -> None:
        before = with_shoulders(reading("R wrist", 500, 400))
        after = with_shoulders(reading("R wrist", 500, 400))
        assert implausible_joints(before, after) == []

    def test_ordinary_motion_is_plausible(self) -> None:
        """A limb moving a fifth of a shoulder width is a person moving."""
        before = with_shoulders(reading("R wrist", 500, 400))
        after = with_shoulders(reading("R wrist", 520, 385))
        assert implausible_joints(before, after) == []

    def test_fast_but_possible_motion_is_plausible(self) -> None:
        before = with_shoulders(reading("R wrist", 500, 400))
        after = with_shoulders(reading("R wrist", 500 + THRESHOLD * 0.9, 400))
        assert implausible_joints(before, after) == []


class TestImplausibleMotion:
    def test_the_real_failure_is_caught(self) -> None:
        """Frame 416 of the clip: the elbow moved 286px in one frame."""
        before = with_shoulders(reading("R elbow", 700, 500, confidence=0.90))
        after = with_shoulders(reading("R elbow", 700 + 286, 500, confidence=0.861))

        found = implausible_joints(before, after)
        assert len(found) == 1
        assert found[0].name == "R elbow"
        assert found[0].travel_px == pytest.approx(286, abs=1)

    def test_high_reported_confidence_does_not_excuse_it(self) -> None:
        """The detector's own score is not evidence about the detector.

        On the real clip it never once flagged a frame it got wrong, which is
        the whole reason this check works on coordinates instead. Measured
        over 1139 labelled frames, its confidence caught 47% of the worst
        frames against this check's 57%.
        """
        before = with_shoulders(reading("R knee", 400, 800, confidence=0.99))
        after = with_shoulders(reading("R knee", 400, 800 + 156, confidence=0.989))

        found = implausible_joints(before, after)
        assert len(found) == 1
        assert found[0].reported_confidence == pytest.approx(0.989)

    def test_several_joints_are_reported_worst_first(self) -> None:
        before = with_shoulders(reading("R elbow", 700, 500), reading("R wrist", 800, 550))
        after = with_shoulders(
            reading("R elbow", 700, 500 + 200), reading("R wrist", 800, 550 + 400)
        )

        found = implausible_joints(before, after)
        assert [f.name for f in found] == ["R wrist", "R elbow"]

    def test_the_threshold_is_the_boundary(self) -> None:
        before = with_shoulders(reading("R wrist", 0, 0))
        just_under = with_shoulders(reading("R wrist", THRESHOLD - 1, 0))
        just_over = with_shoulders(reading("R wrist", THRESHOLD + 1, 0))

        assert implausible_joints(before, just_under) == []
        assert len(implausible_joints(before, just_over)) == 1

    def test_the_reported_threshold_reflects_the_subject_size(self) -> None:
        before = with_shoulders(reading("R wrist", 0, 0))
        after = with_shoulders(reading("R wrist", 300, 0))

        found = implausible_joints(before, after)
        assert found[0].threshold_px == pytest.approx(THRESHOLD)


class TestNormalisationBySubjectSize:
    """The bug this replaced, stated as behaviour.

    An absolute pixel threshold conflates limb speed with apparent body size,
    so the same movement filmed closer reads as impossible. Measured on Penn
    Action that made the check worthless — the frames it flagged carried 1.02x
    the error of the frames it passed. Normalised by shoulder width, 2.8x.
    """

    def test_the_same_movement_scales_with_the_subject(self) -> None:
        """Twice as close means twice the pixels for the same real motion."""
        far_before = with_shoulders(reading("R wrist", 0, 0), width=100)
        far_after = with_shoulders(reading("R wrist", 40, 0), width=100)

        near_before = with_shoulders(reading("R wrist", 0, 0), width=200)
        near_after = with_shoulders(reading("R wrist", 80, 0), width=200)

        assert implausible_joints(far_before, far_after) == []
        assert implausible_joints(near_before, near_after) == []

    def test_a_close_up_is_not_flagged_for_being_close(self) -> None:
        """80px would exceed the far subject's threshold but not this one."""
        before = with_shoulders(reading("R wrist", 0, 0), width=300)
        after = with_shoulders(reading("R wrist", 120, 0), width=300)
        assert implausible_joints(before, after) == []

    def test_a_wide_shot_is_still_judged_strictly(self) -> None:
        before = with_shoulders(reading("R wrist", 0, 0), width=40)
        after = with_shoulders(reading("R wrist", 40, 0), width=40)
        assert len(implausible_joints(before, after)) == 1

    def test_the_scale_comes_from_the_previous_frame(self) -> None:
        """The current frame may be the broken reading, and a broken reading
        gives a broken scale.

        Here the shoulders collapse — a detection failure. Judged against the
        previous frame's 200px shoulders the threshold is 110px, so the wrist's
        80px move passes. The collapsed shoulder is itself flagged, which is
        right: it moved 199px and shoulders do not do that.
        """
        before = with_shoulders(reading("R wrist", 0, 0), width=200)
        after = [
            reading("L shoulder", 400, 300),
            reading("R shoulder", 401, 300),
            reading("R wrist", 80, 0),
        ]

        found = implausible_joints(before, after)
        assert [f.name for f in found] == ["R shoulder"]
        assert found[0].threshold_px == pytest.approx(200 * MAX_TRAVEL_PER_SHOULDER_WIDTH)


class TestShoulderWidth:
    def test_it_measures_the_gap(self) -> None:
        assert shoulder_width(with_shoulders(width=150)) == pytest.approx(150)

    def test_absent_shoulders_give_none(self) -> None:
        assert shoulder_width([reading("R wrist", 0, 0)]) is None

    def test_collapsed_shoulders_give_none(self) -> None:
        """Coincident shoulders are a detection failure, not a scale."""
        collapsed = [reading("L shoulder", 400, 300), reading("R shoulder", 400, 300)]
        assert shoulder_width(collapsed) is None


class TestFallback:
    def test_without_shoulders_it_falls_back_to_pixels(self) -> None:
        before = [reading("R wrist", 0, 0)]
        assert implausible_joints(before, [reading("R wrist", 50, 0)]) == []
        assert (
            len(implausible_joints(before, [reading("R wrist", FALLBACK_TRAVEL_PX + 10, 0)])) == 1
        )

    def test_an_explicit_limit_overrides_normalisation(self) -> None:
        """For callers whose every frame is framed identically.

        A 40px move sits under the normalised threshold (55px) and over an
        explicit 30px one, so the two paths give opposite verdicts on the
        same input — which is what "overrides" has to mean.
        """
        before = with_shoulders(reading("R wrist", 0, 0))
        after = with_shoulders(reading("R wrist", 40, 0))

        assert implausible_joints(before, after) == []
        assert len(implausible_joints(before, after, limit=30)) == 1


class TestMissingJoints:
    def test_a_joint_absent_from_the_previous_frame_is_skipped(self) -> None:
        """Nothing to compare against is not the same as a jump."""
        before = with_shoulders()
        after = with_shoulders(reading("R wrist", 900, 900))
        assert implausible_joints(before, after) == []

    def test_a_joint_absent_from_the_current_frame_is_skipped(self) -> None:
        before = with_shoulders(reading("R wrist", 0, 0))
        after = with_shoulders()
        assert implausible_joints(before, after) == []


class TestToolContract:
    def test_a_plausible_frame_is_a_passing_measurement(self) -> None:
        tool = continuity_tool(
            with_shoulders(reading("R wrist", 500, 400)),
            with_shoulders(reading("R wrist", 520, 410)),
        )
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)

        assert result.is_measurement is True
        assert result.value["plausible"] is True
        assert result.value["implausible_count"] == 0

    def test_an_implausible_frame_reports_the_detail(self) -> None:
        tool = continuity_tool(
            with_shoulders(reading("R elbow", 700, 500)),
            with_shoulders(reading("R elbow", 986, 500, confidence=0.861)),
        )
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)

        assert result.value["plausible"] is False
        assert result.value["worst_travel_px"] == pytest.approx(286, abs=1)
        assert "286px" in result.value["detail"][0]
        assert "0.86" in result.value["detail"][0]

    def test_the_detail_states_the_threshold_used(self) -> None:
        """An auditor needs to know what "too far" meant for this subject."""
        tool = continuity_tool(
            with_shoulders(reading("R elbow", 700, 500)),
            with_shoulders(reading("R elbow", 986, 500)),
        )
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)
        assert f"{THRESHOLD:.0f}px" in result.value["detail"][0]

    def test_the_verdict_is_the_answer_key(self) -> None:
        """So the travel figures stay context rather than being judged."""
        tool = continuity_tool(
            with_shoulders(reading("R wrist", 0, 0)),
            with_shoulders(reading("R wrist", 10, 0)),
        )
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)
        assert result.answer_key == "plausible"

    def test_it_says_nothing_about_why(self) -> None:
        """Motion blur, occlusion and a lost track are identical in the
        numbers, and they need different handling. That gap is the agent's
        job, and the tool must not pretend to fill it."""
        tool = continuity_tool(
            with_shoulders(reading("R wrist", 0, 0)),
            with_shoulders(reading("R wrist", 300, 0)),
        )
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)

        text = " ".join(result.value["detail"]).lower()
        for cause in ("blur", "occlusion", "occluded", "lost"):
            assert cause not in text
