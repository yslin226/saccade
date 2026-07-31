"""Tests for the detector-disagreement signal.

The best signal this project has found for predicting pose error, and still
not good enough: AUROC 0.638 over 890 held-out frames, against a bar of 0.70
fixed before the run. Four single-detector geometric signals did worse, three
of them worse than chance.

An earlier measurement said 0.713 and was wrong — it was reading a bug in
shoulder_width() rather than the signal. Two tests below pin the fix, because
that bug was invisible except through ground truth.
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


class TestCollapsedScale:
    """Regression: a collapsed shoulder width made every gap enormous.

    shoulder_width() rejected only widths below 1px. A failed detection
    typically reports a few pixels, and a normal 60px gap divided by 2px
    reads as 30 body widths. One run reported 157, which is not a quantity
    that exists, and the inflated values landed on the frames that were
    already wrong — so the bug scored AUROC 0.713 while the signal scored
    0.638.
    """

    def test_a_collapsed_shoulder_width_yields_no_comparison(self) -> None:
        collapsed = [
            reading("L shoulder", 400, 300),
            reading("R shoulder", 402, 300),
            reading("R wrist", 500, 400),
        ]
        other = [
            reading("L shoulder", 400, 300),
            reading("R shoulder", 402, 300),
            reading("R wrist", 560, 400),
        ]
        assert disagreement_features(collapsed, other).per_joint == {}

    def test_no_gap_can_exceed_a_plausible_body(self) -> None:
        """Whatever the inputs, a gap of 150 body widths means a broken scale."""
        first = pose(reading("R wrist", 0, 0))
        second = pose(reading("R wrist", 470, 350))

        gaps = disagreement_features(first, second)
        assert all(gap < 10.0 for gap in gaps.per_joint.values())


class TestLowConfidenceKeypoints:
    """A keypoint at 0.08 confidence is a guess about where a limb might be.

    The distance between two guesses is not disagreement about an
    observation, and both detectors emit them routinely for limbs outside
    the frame.
    """

    def test_a_low_confidence_joint_is_excluded(self) -> None:
        first = pose(reading("R wrist", 500, 400, confidence=0.05))
        second = pose(reading("R wrist", 600, 400, confidence=0.9))

        assert "R wrist" not in disagreement_features(first, second).per_joint

    def test_low_confidence_on_either_side_excludes_it(self) -> None:
        first = pose(reading("R wrist", 500, 400, confidence=0.9))
        second = pose(reading("R wrist", 600, 400, confidence=0.05))

        assert "R wrist" not in disagreement_features(first, second).per_joint

    def test_confident_joints_are_still_compared(self) -> None:
        first = pose(reading("R wrist", 500, 400, confidence=0.8))
        second = pose(reading("R wrist", 560, 400, confidence=0.8))

        assert disagreement_features(first, second).per_joint["R wrist"] == pytest.approx(0.6)

    def test_the_threshold_can_be_relaxed(self) -> None:
        first = pose(reading("R wrist", 500, 400, confidence=0.1))
        second = pose(reading("R wrist", 560, 400, confidence=0.1))

        assert "R wrist" not in disagreement_features(first, second).per_joint
        assert "R wrist" in disagreement_features(first, second, min_confidence=0.05).per_joint


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
