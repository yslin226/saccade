"""Tests for the detector-disagreement signal.

The first signal in this project that predicts pose error better than chance:
AUROC 0.713 over 1320 held-out frames, p = 7e-27. Four single-detector
geometric signals failed at the same task, three of them worse than chance.

So this one is load-bearing, and the tests treat it as a referee: a wrong
verdict here decides whether a frame's measurements enter a calculation.
"""

from __future__ import annotations

import pytest
from PIL import Image

from benchmarks.pose_probe.continuity import JointReading
from benchmarks.pose_probe.disagreement import (
    COMMON_JOINTS,
    SUSPECT_MEAN_GAP,
    disagreement_features,
    disagreement_tool,
)

SHOULDERS = 100.0


def reading(name: str, x: float, y: float, confidence: float = 0.9) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=confidence)


def pose(*others: JointReading, width: float = SHOULDERS) -> list[JointReading]:
    """A reading set with a shoulder pair, which sets the scale."""
    return [
        reading("L shoulder", 400, 300),
        reading("R shoulder", 400 + width, 300),
        *others,
    ]


class TestAgreement:
    def test_identical_readings_show_no_gap(self) -> None:
        first = pose(reading("R wrist", 500, 400))
        gaps = disagreement_features(first, list(first))

        assert gaps.features()["mean_gap"] == pytest.approx(0.0)
        assert gaps.features()["max_gap"] == pytest.approx(0.0)

    def test_a_small_offset_stays_small(self) -> None:
        """Two detectors never agree to the pixel; that is not disagreement."""
        first = pose(reading("R wrist", 500, 400))
        second = pose(reading("R wrist", 502, 401))

        assert disagreement_features(first, second).features()["max_gap"] < 0.05


class TestDisagreement:
    def test_one_joint_placed_elsewhere_is_measured(self) -> None:
        first = pose(reading("R wrist", 500, 400))
        second = pose(reading("R wrist", 560, 400))

        gaps = disagreement_features(first, second)
        # 60px on a 100px shoulder width.
        assert gaps.per_joint["R wrist"] == pytest.approx(0.6)
        assert gaps.worst_joint == "R wrist"

    def test_the_worst_joint_is_named(self) -> None:
        """An auditor needs to know where to look."""
        first = pose(reading("R wrist", 500, 400), reading("L knee", 300, 800))
        second = pose(reading("R wrist", 510, 400), reading("L knee", 400, 800))

        assert disagreement_features(first, second).worst_joint == "L knee"

    def test_mean_and_max_differ_when_one_joint_is_wild(self) -> None:
        first = pose(reading("R wrist", 500, 400), reading("L knee", 300, 800))
        second = pose(reading("R wrist", 500, 400), reading("L knee", 400, 800))

        features = disagreement_features(first, second).features()
        assert features["max_gap"] > features["mean_gap"]


class TestNormalisation:
    """Gaps are in shoulder widths, not pixels.

    An absolute distance conflates disagreement with apparent body size, which
    is the mistake that made an earlier signal worthless.
    """

    def test_the_same_relative_gap_scores_the_same_at_any_distance(self) -> None:
        far_first = pose(reading("R wrist", 0, 0), width=100)
        far_second = pose(reading("R wrist", 20, 0), width=100)

        near_first = pose(reading("R wrist", 0, 0), width=300)
        near_second = pose(reading("R wrist", 60, 0), width=300)

        assert disagreement_features(far_first, far_second).features()["max_gap"] == pytest.approx(
            disagreement_features(near_first, near_second).features()["max_gap"]
        )

    def test_an_explicit_scale_is_honoured(self) -> None:
        first = pose(reading("R wrist", 0, 0))
        second = pose(reading("R wrist", 50, 0))

        gaps = disagreement_features(first, second, scale=50.0)
        assert gaps.per_joint["R wrist"] == pytest.approx(1.0)

    def test_without_shoulders_there_is_no_scale(self) -> None:
        first = [reading("R wrist", 0, 0)]
        second = [reading("R wrist", 90, 0)]
        assert disagreement_features(first, second).per_joint == {}


class TestPartialOverlap:
    def test_only_shared_joints_are_compared(self) -> None:
        first = pose(reading("R wrist", 500, 400), reading("L knee", 300, 800))
        second = pose(reading("R wrist", 500, 400))

        gaps = disagreement_features(first, second)
        assert set(gaps.per_joint) == {"L shoulder", "R shoulder", "R wrist"}

    def test_no_shared_joints_gives_nothing(self) -> None:
        first = pose(reading("R wrist", 0, 0))
        second = pose(reading("L ankle", 0, 0))

        gaps = disagreement_features(first, second)
        assert "R wrist" not in gaps.per_joint


class TestToolContract:
    def test_agreement_is_a_passing_measurement(self) -> None:
        first = pose(reading("R wrist", 500, 400))
        second = pose(reading("R wrist", 503, 401))

        result = disagreement_tool(first, second).fn(
            image=Image.new("RGB", (64, 64)), viewport=None
        )

        assert result.is_measurement is True
        assert result.value["detectors_agree"] is True

    def test_disagreement_is_flagged(self) -> None:
        first = pose(reading("R wrist", 500, 400))
        second = pose(reading("R wrist", 600, 400))

        result = disagreement_tool(first, second).fn(
            image=Image.new("RGB", (64, 64)), viewport=None
        )

        assert result.value["detectors_agree"] is False
        assert result.value["worst_joint"] == "R wrist"

    def test_the_verdict_is_the_answer_key(self) -> None:
        """So the gap figures stay context rather than being judged."""
        first = pose(reading("R wrist", 0, 0))
        result = disagreement_tool(first, list(first)).fn(
            image=Image.new("RGB", (64, 64)), viewport=None
        )
        assert result.answer_key == "detectors_agree"

    def test_the_continuous_value_is_always_reported(self) -> None:
        """The threshold is a convenience; the number is the signal."""
        first = pose(reading("R wrist", 500, 400))
        second = pose(reading("R wrist", 530, 400))

        value = (
            disagreement_tool(first, second)
            .fn(image=Image.new("RGB", (64, 64)), viewport=None)
            .value
        )

        assert "mean_gap" in value
        assert "max_gap" in value
        assert value["units"] == "shoulder widths"

    def test_no_shared_joints_is_not_a_measurement(self) -> None:
        """Nothing was compared, so there is nothing to overrule a model with."""
        result = disagreement_tool([reading("R wrist", 0, 0)], []).fn(
            image=Image.new("RGB", (64, 64)), viewport=None
        )
        assert result.is_measurement is False

    def test_it_says_nothing_about_which_detector_is_wrong(self) -> None:
        """Or why. Blur, occlusion and a lost track look the same here, and
        they need different handling — that is the agent's question."""
        first = pose(reading("R wrist", 500, 400))
        second = pose(reading("R wrist", 600, 400))

        value = (
            disagreement_tool(first, second)
            .fn(image=Image.new("RGB", (64, 64)), viewport=None)
            .value
        )

        text = " ".join(str(v).lower() for v in value.values())
        for word in ("blur", "occlu", "mediapipe", "yolo"):
            assert word not in text

    def test_the_threshold_can_be_overridden(self) -> None:
        first = pose(reading("R wrist", 500, 400))
        second = pose(reading("R wrist", 515, 400))

        lenient = disagreement_tool(first, second, threshold=0.5).fn(
            image=Image.new("RGB", (64, 64)), viewport=None
        )
        strict = disagreement_tool(first, second, threshold=0.01).fn(
            image=Image.new("RGB", (64, 64)), viewport=None
        )

        assert lenient.value["detectors_agree"] is True
        assert strict.value["detectors_agree"] is False


class TestJointMapping:
    def test_both_detectors_index_the_same_joints(self) -> None:
        """A mismatched index would compare a wrist against a knee and call
        the difference disagreement."""
        assert len(COMMON_JOINTS) == 10
        for name, (mp_index, yolo_index) in COMMON_JOINTS.items():
            assert isinstance(mp_index, int)
            assert isinstance(yolo_index, int)
            # MediaPipe's body landmarks start at 11; YOLO uses COCO's 17.
            assert 11 <= mp_index <= 32, name
            assert 5 <= yolo_index <= 16, name

    def test_left_and_right_are_not_swapped(self) -> None:
        """MediaPipe and COCO both put left before right at each pair."""
        for joint in ("shoulder", "elbow", "wrist", "hip", "knee"):
            left = COMMON_JOINTS[f"L {joint}"]
            right = COMMON_JOINTS[f"R {joint}"]
            assert left[0] < right[0], joint
            assert left[1] < right[1], joint


def test_the_threshold_is_a_body_fraction_not_pixels() -> None:
    assert 0.0 < SUSPECT_MEAN_GAP < 1.0
