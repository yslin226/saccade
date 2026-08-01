"""Tests for the hitting metrics and the swing use case.

The bat is what makes these different from pitching, and the bat is what the
detector loses — a bat at contact speed is a smear COCO was never shown. So
most of these are about what happens when it is not found, because that is
the common case rather than the edge one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from sandlot.application.use_cases import analyze_swing, bat_boxes
from sandlot.application.use_cases.analyze_swing import measure
from sandlot.domain.models import Frame, JointReading, Toolchain
from sandlot.domain.swing import (
    MIN_BAT_TRAVEL_PX,
    MIN_SWING_FRAMES,
    bat_path,
    swing_plane_angle,
    weight_transfer,
)

TOOLCHAIN = Toolchain(mediapipe="1.0.0", ultralytics="8.4.113", sandlot="0.1.0")


def joint(name: str, x: float, y: float) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=0.9)


def hitter(index: int, *, shift: float = 0.0) -> Frame:
    """A body whose mass has moved ``shift`` pixels to the right."""
    return Frame(
        index=index,
        timestamp=index / 60,
        joints=(
            joint("L shoulder", 200 + shift, 100),
            joint("R shoulder", 300 + shift, 100),
            joint("L hip", 200 + shift, 300),
            joint("R hip", 300 + shift, 300),
            joint("L knee", 200 + shift, 450),
            joint("R knee", 300 + shift, 450),
            joint("L ankle", 200 + shift, 600),
            joint("R ankle", 300 + shift, 600),
        ),
    )


def box(x: float, y: float) -> tuple[float, float, float, float]:
    """A 20x20 bat box centred on (x, y)."""
    return (x - 10, y - 10, 20.0, 20.0)


@dataclass
class FakeVideo:
    images: list[Any] = field(default_factory=lambda: [object()])
    sha256: str = "abc123"
    fps: float = 60.0

    def read(self, path: Any, *, stride: int = 1) -> FakeVideo:
        return self


@dataclass
class FakePose:
    frames: list[Frame] = field(default_factory=list)
    toolchain: Toolchain = TOOLCHAIN

    def detect(self, images: list[Any], *, fps: float) -> list[Frame]:
        return self.frames


@dataclass(frozen=True)
class FakeDetection:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class FakeObjects:
    per_frame: list[list[FakeDetection]] = field(default_factory=list)

    def detect(self, images: list[Any]) -> list[list[FakeDetection]]:
        return self.per_frame


class TestBatPath:
    def test_it_returns_box_centres(self) -> None:
        assert bat_path([box(100, 200)]) == [(100.0, 200.0)]

    def test_missing_frames_are_dropped_not_interpolated(self) -> None:
        """A bat invisible for six frames was moving fastest exactly then,
        and a straight line across the gap understates the path."""
        path = bat_path([box(0, 0), None, None, box(300, 300)])
        assert path == [(0.0, 0.0), (300.0, 300.0)]

    def test_no_detections_gives_an_empty_path(self) -> None:
        assert bat_path([None, None]) == []

    def test_an_empty_input_gives_an_empty_path(self) -> None:
        assert bat_path([]) == []


class TestSwingPlaneAngle:
    def test_a_horizontal_swing_reads_zero(self) -> None:
        boxes = [box(x, 300) for x in (100, 200, 300, 400)]
        assert swing_plane_angle(boxes) == pytest.approx(0.0)

    def test_a_downward_swing_reads_its_angle(self) -> None:
        """y grows downward in an image, following the pixel grid."""
        boxes = [box(100 + 100 * i, 100 + 100 * i) for i in range(4)]
        assert swing_plane_angle(boxes) == pytest.approx(45.0)

    def test_it_is_folded_into_half_a_turn(self) -> None:
        """A plane has no direction: a right-hander and a left-hander
        swinging on the same plane read the same number."""
        rightward = [box(100 + 100 * i, 300) for i in range(4)]
        leftward = [box(400 - 100 * i, 300) for i in range(4)]
        assert swing_plane_angle(rightward) == pytest.approx(swing_plane_angle(leftward))

    def test_the_result_stays_below_180(self) -> None:
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (0, 1), (0, -1)):
            boxes = [box(300 + 100 * dx * i, 300 + 100 * dy * i) for i in range(4)]
            angle = swing_plane_angle(boxes)
            assert angle is not None
            assert 0.0 <= angle < 180.0

    def test_too_few_located_frames_gives_nothing(self) -> None:
        """A line through three detections describes the detector's luck."""
        boxes = [box(100, 300), box(200, 300), None, None]
        assert swing_plane_angle(boxes) is None

    def test_a_bat_that_barely_moved_gives_nothing(self) -> None:
        """Detection jitter has an angle, and reporting it to one decimal
        place makes noise look like a measurement."""
        boxes = [box(100 + i, 300) for i in range(5)]
        assert swing_plane_angle(boxes) is None

    def test_no_bat_at_all_gives_nothing(self) -> None:
        assert swing_plane_angle([None] * 10) is None

    def test_the_thresholds_are_stated_not_implied(self) -> None:
        assert MIN_SWING_FRAMES >= 3
        assert MIN_BAT_TRAVEL_PX > 0


class TestWeightTransfer:
    def test_a_body_that_moved_reports_the_distance(self) -> None:
        frames = [hitter(i, shift=50 * i) for i in range(5)]
        transfer = weight_transfer(frames)
        assert transfer is not None
        assert transfer > 0.0

    def test_a_body_that_stayed_put_reports_near_zero(self) -> None:
        frames = [hitter(i) for i in range(5)]
        transfer = weight_transfer(frames)
        assert transfer is not None
        assert transfer == pytest.approx(0.0, abs=0.01)

    def test_it_is_reported_in_torso_lengths(self) -> None:
        """The same shift filmed from ten feet and from thirty must read the
        same."""
        near = [
            Frame(
                index=i,
                timestamp=i / 60,
                joints=(
                    joint("L shoulder", 0, 0),
                    joint("R shoulder", 200, 0),
                    joint("L hip", 0, 400),
                    joint("R hip", 200, 400),
                    joint("L knee", 100 * i, 600),
                    joint("R knee", 200 + 100 * i, 600),
                ),
            )
            for i in range(5)
        ]
        far = [
            Frame(
                index=i,
                timestamp=i / 60,
                joints=(
                    joint("L shoulder", 0, 0),
                    joint("R shoulder", 100, 0),
                    joint("L hip", 0, 200),
                    joint("R hip", 100, 200),
                    joint("L knee", 50 * i, 300),
                    joint("R knee", 100 + 50 * i, 300),
                ),
            )
            for i in range(5)
        ]
        assert weight_transfer(near) == pytest.approx(weight_transfer(far), rel=0.05)

    def test_a_single_bad_frame_does_not_dominate(self) -> None:
        """Smoothed first: one misplaced ankle moves the centre by more than
        the hitter did."""
        clean = [hitter(i, shift=10 * i) for i in range(6)]
        spiked = list(clean)
        spiked[3] = hitter(3, shift=900)

        assert weight_transfer(spiked) == pytest.approx(weight_transfer(clean), abs=0.5)

    def test_too_few_measurable_frames_gives_nothing(self) -> None:
        assert weight_transfer([hitter(0), hitter(1)]) is None

    def test_frames_without_a_body_give_nothing(self) -> None:
        assert weight_transfer([Frame(index=i, timestamp=0.0) for i in range(6)]) is None


class TestBatBoxes:
    def test_it_keeps_one_entry_per_frame(self) -> None:
        """Including the misses, so a caller can see how much of the swing
        was tracked rather than only what was."""
        detections = [
            [FakeDetection("baseball bat", box(100, 100), 0.9)],
            [],
            [FakeDetection("baseball bat", box(300, 300), 0.8)],
        ]
        assert bat_boxes(detections) == [box(100, 100), None, box(300, 300)]

    def test_the_most_confident_bat_wins(self) -> None:
        """A swing has one bat in it, so the others are bat-shaped
        background, and averaging would place it between a real bat and a
        fencepost."""
        detections = [
            [
                FakeDetection("baseball bat", box(100, 100), 0.3),
                FakeDetection("baseball bat", box(500, 500), 0.9),
            ]
        ]
        assert bat_boxes(detections) == [box(500, 500)]

    def test_other_objects_are_ignored(self) -> None:
        detections = [
            [
                FakeDetection("person", box(100, 100), 0.99),
                FakeDetection("sports ball", box(200, 200), 0.95),
            ]
        ]
        assert bat_boxes(detections) == [None]

    def test_no_frames_gives_no_boxes(self) -> None:
        assert bat_boxes([]) == []


class TestAnalyzeSwing:
    def swinging(self) -> list[Frame]:
        return [hitter(i, shift=20 * i) for i in range(6)]

    def bat_across(self) -> list[list[FakeDetection]]:
        return [[FakeDetection("baseball bat", box(100 + 80 * i, 300), 0.8)] for i in range(6)]

    def test_it_measures_body_and_bat(self) -> None:
        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
            objects=FakeObjects(per_frame=self.bat_across()),
        )
        names = {m.name for m in result.metrics}
        assert "swing_plane_angle" in names
        assert "weight_transfer" in names

    def test_no_detector_still_measures_the_body(self) -> None:
        """A legitimate result rather than a degraded one: the swing-plane
        metric is simply absent, the same as when the detector found no
        bat."""
        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
            objects=None,
        )
        names = {m.name for m in result.metrics}
        assert "swing_plane_angle" not in names
        assert "weight_transfer" in names

    def test_a_bat_never_found_leaves_the_metric_absent(self) -> None:
        """Absent, not estimated. A plane through three spurious detections
        is a confident number describing nothing."""
        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
            objects=FakeObjects(per_frame=[[] for _ in range(6)]),
        )
        assert "swing_plane_angle" not in {m.name for m in result.metrics}

    def test_the_swing_plane_cites_the_frames_the_bat_was_seen_in(self) -> None:
        """Rule 8: an auditor checking this needs to know which pictures it
        came from, and how few they were."""
        detections: list[list[FakeDetection]] = [[] for _ in range(6)]
        for i in (1, 2, 3, 4):
            detections[i] = [FakeDetection("baseball bat", box(100 + 100 * i, 300), 0.8)]

        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
            objects=FakeObjects(per_frame=detections),
        )
        plane = next(m for m in result.metrics if m.name == "swing_plane_angle")
        assert plane.frames == (1, 2, 3, 4)
        assert plane.detail["frames_with_bat"] == 4
        assert plane.detail["frames_total"] == 6

    def test_it_shares_the_body_metrics_with_pitching(self) -> None:
        """By import, not by copy — two files computing the same angle drift
        apart the first time one is corrected."""
        from sandlot.application.use_cases.analyze_pitch import measure as pitch_measure

        frames = self.swinging()
        pitching = {m.name: m.value for m in pitch_measure(frames)}
        hitting = {m.name: m.value for m in measure(frames)}

        for name, value in pitching.items():
            assert hitting[name] == pytest.approx(value)

    def test_the_session_records_the_toolchain(self) -> None:
        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
        )
        assert result.toolchain == TOOLCHAIN

    def test_it_stores_when_given_a_repository(self) -> None:
        @dataclass
        class Repo:
            stored: dict[str, Any] = field(default_factory=dict)

            def save(self, session: Any) -> None:
                self.stored[session.id] = session

            def get(self, session_id: str) -> Any:
                return self.stored.get(session_id)

            def list_since(self, since: datetime | None = None) -> list[Any]:
                return list(self.stored.values())

        repo = Repo()
        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
            repo=repo,
        )
        assert repo.get(result.id) == result

    def test_a_video_with_no_frames_fails_loudly(self) -> None:
        from sandlot.application.use_cases import AnalysisFailedError

        with pytest.raises(AnalysisFailedError, match="no frames"):
            analyze_swing("empty.mov", video=FakeVideo(images=[]), pose=FakePose())

    def test_an_explicit_id_is_honoured(self) -> None:
        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
            session_id="fixed",
        )
        assert result.id == "fixed"

    def test_the_weight_transfer_metric_says_why_it_matters(self) -> None:
        """A number a coach will be shown needs its provenance attached."""
        result = analyze_swing(
            "swing.mov",
            video=FakeVideo(images=[object()] * 6),
            pose=FakePose(frames=self.swinging()),
        )
        transfer = next(m for m in result.metrics if m.name == "weight_transfer")
        assert "0.097" in transfer.detail["why"]


def test_the_created_at_is_timezone_aware() -> None:
    """A naive timestamp compares wrongly against an aware one, and
    compare_with_previous does exactly that comparison."""
    result = analyze_swing(
        "swing.mov",
        video=FakeVideo(images=[object()]),
        pose=FakePose(frames=[hitter(0)]),
    )
    assert result.created_at.tzinfo is not None
    assert result.created_at.tzinfo.utcoffset(result.created_at) == UTC.utcoffset(None)
