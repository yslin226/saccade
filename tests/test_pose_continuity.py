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
    MAX_JOINT_TRAVEL_PX,
    JointReading,
    continuity_tool,
    implausible_joints,
)


def reading(name: str, x: float, y: float, confidence: float = 0.9) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=confidence)


class TestPlausibleMotion:
    def test_a_still_joint_is_plausible(self) -> None:
        before = [reading("R wrist", 500, 400)]
        after = [reading("R wrist", 500, 400)]
        assert implausible_joints(before, after) == []

    def test_ordinary_motion_is_plausible(self) -> None:
        """A limb moving 40px in 1/30s is a person moving normally."""
        before = [reading("R wrist", 500, 400)]
        after = [reading("R wrist", 530, 375)]
        assert implausible_joints(before, after) == []

    def test_fast_but_possible_motion_is_plausible(self) -> None:
        before = [reading("R wrist", 500, 400)]
        after = [reading("R wrist", 600, 480)]
        assert implausible_joints(before, after) == []


class TestImplausibleMotion:
    def test_the_real_failure_is_caught(self) -> None:
        """Frame 416 of the clip: the elbow moved 286px in one frame."""
        before = [reading("R elbow", 700, 500, confidence=0.90)]
        after = [reading("R elbow", 700 + 286, 500, confidence=0.861)]

        found = implausible_joints(before, after)
        assert len(found) == 1
        assert found[0].name == "R elbow"
        assert found[0].travel_px == pytest.approx(286, abs=1)

    def test_high_reported_confidence_does_not_excuse_it(self) -> None:
        """The detector's own score is not evidence about the detector.

        On the real clip it never once flagged a frame it got wrong, which is
        the whole reason this check works on coordinates instead.
        """
        before = [reading("R knee", 400, 800, confidence=0.99)]
        after = [reading("R knee", 400, 800 + 156, confidence=0.989)]

        found = implausible_joints(before, after)
        assert len(found) == 1
        assert found[0].reported_confidence == pytest.approx(0.989)

    def test_several_joints_are_reported_worst_first(self) -> None:
        before = [reading("R elbow", 700, 500), reading("R wrist", 800, 550)]
        after = [reading("R elbow", 700, 500 + 200), reading("R wrist", 800, 550 + 400)]

        found = implausible_joints(before, after)
        assert [f.name for f in found] == ["R wrist", "R elbow"]

    def test_the_threshold_is_the_boundary(self) -> None:
        before = [reading("R wrist", 0, 0)]
        just_under = [reading("R wrist", MAX_JOINT_TRAVEL_PX - 1, 0)]
        just_over = [reading("R wrist", MAX_JOINT_TRAVEL_PX + 1, 0)]

        assert implausible_joints(before, just_under) == []
        assert len(implausible_joints(before, just_over)) == 1

    def test_a_caller_may_set_its_own_limit(self) -> None:
        """The threshold scales with resolution and frame rate."""
        before = [reading("R wrist", 0, 0)]
        after = [reading("R wrist", 80, 0)]

        assert implausible_joints(before, after) == []
        assert len(implausible_joints(before, after, limit=50)) == 1


class TestMissingJoints:
    def test_a_joint_absent_from_the_previous_frame_is_skipped(self) -> None:
        """Nothing to compare against is not the same as a jump."""
        before: list[JointReading] = []
        after = [reading("R wrist", 900, 900)]
        assert implausible_joints(before, after) == []

    def test_a_joint_absent_from_the_current_frame_is_skipped(self) -> None:
        before = [reading("R wrist", 0, 0)]
        after: list[JointReading] = []
        assert implausible_joints(before, after) == []


class TestToolContract:
    def test_a_plausible_frame_is_a_passing_measurement(self) -> None:
        tool = continuity_tool(
            [reading("R wrist", 500, 400)],
            [reading("R wrist", 520, 410)],
        )
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)

        assert result.is_measurement is True
        assert result.value["plausible"] is True
        assert result.value["implausible_count"] == 0

    def test_an_implausible_frame_reports_the_detail(self) -> None:
        tool = continuity_tool(
            [reading("R elbow", 700, 500)],
            [reading("R elbow", 986, 500, confidence=0.861)],
        )
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)

        assert result.value["plausible"] is False
        assert result.value["worst_travel_px"] == pytest.approx(286, abs=1)
        assert "286px" in result.value["detail"][0]
        assert "0.86" in result.value["detail"][0]

    def test_the_verdict_is_the_answer_key(self) -> None:
        """So the travel figures stay context rather than being judged."""
        tool = continuity_tool([reading("R wrist", 0, 0)], [reading("R wrist", 10, 0)])
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)
        assert result.answer_key == "plausible"

    def test_it_says_nothing_about_why(self) -> None:
        """Motion blur, occlusion and a lost track are identical in the
        numbers, and they need different handling. That gap is the agent's
        job, and the tool must not pretend to fill it."""
        tool = continuity_tool([reading("R wrist", 0, 0)], [reading("R wrist", 300, 0)])
        result = tool.fn(image=Image.new("RGB", (64, 64)), viewport=None)

        text = " ".join(result.value["detail"]).lower()
        for cause in ("blur", "occlusion", "occluded", "lost"):
            assert cause not in text
