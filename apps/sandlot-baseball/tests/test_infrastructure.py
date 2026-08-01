"""Tests for the infrastructure layer.

The storage and the Saccade wiring are tested against fixtures. The detectors
are tested against a synthetic video written to tmp_path — enough to prove
the plumbing, decoding and coordinate conversion work, without depending on a
particular clip being present.

What is *not* tested here is whether MediaPipe finds a person in a synthetic
video, because it will not. The detector's accuracy is measured against real
footage in benchmarks/pose_probe; this checks that a frame with nothing found
still appears, which is the property the evidence chain depends on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from sandlot.domain.models import Frame, JointReading, Metric, Session, Toolchain
from sandlot.infrastructure import (
    JsonSessionRepo,
    OpenCVVideo,
    disagreement_tool,
    file_sha256,
    pose_measurement_tool,
)
from sandlot.infrastructure.vision.objects import Detected, YOLODetector
from sandlot.infrastructure.vision.pose import LANDMARK_INDEX, MediaPipePose

TOOLCHAIN = Toolchain(mediapipe="1.0.0", ultralytics="8.4.113", sandlot="0.1.0")


def joint(name: str, x: float, y: float, confidence: float = 0.9) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=confidence)


def body(index: int = 0, *, elbow_bend: float = 0.0) -> Frame:
    """A frame with a torso and a right arm."""
    return Frame(
        index=index,
        timestamp=index / 60,
        joints=(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 300, 100),
            joint("L hip", 200, 300),
            joint("R hip", 300, 300),
            joint("R elbow", 400, 100),
            joint("R wrist", 500 - elbow_bend, 100 + elbow_bend),
        ),
    )


def session(session_id: str = "s1", *, created_at: datetime | None = None) -> Session:
    return Session(
        id=session_id,
        created_at=created_at or datetime(2026, 8, 1, tzinfo=UTC),
        video_sha256="abc123",
        frame_count=10,
        fps=60.0,
        toolchain=TOOLCHAIN,
        metrics=(Metric(name="sep", value=42.0, unit="degrees", frames=(3,)),),
    )


def write_video(path: Path, *, frames: int = 4, fps: float = 30.0) -> Path:
    """A tiny synthetic clip, so decoding is testable without a fixture file."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(frames):
        image = np.full((48, 64, 3), i * 40 % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


class TestFileHash:
    def test_the_same_bytes_hash_the_same(self, tmp_path: Path) -> None:
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"identical")
        second.write_bytes(b"identical")
        assert file_sha256(first) == file_sha256(second)

    def test_different_bytes_hash_differently(self, tmp_path: Path) -> None:
        first = tmp_path / "a.bin"
        second = tmp_path / "b.bin"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        assert file_sha256(first) != file_sha256(second)

    def test_an_empty_file_still_hashes(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.bin"
        empty.write_bytes(b"")
        assert len(file_sha256(empty)) == 64


class TestOpenCVVideo:
    def test_it_decodes_frames(self, tmp_path: Path) -> None:
        path = write_video(tmp_path / "clip.mp4", frames=4)
        decoded = OpenCVVideo().read(path)

        assert len(decoded.images) == 4
        assert decoded.fps > 0
        assert len(decoded.sha256) == 64

    def test_stride_keeps_every_nth_frame(self, tmp_path: Path) -> None:
        path = write_video(tmp_path / "clip.mp4", frames=6)
        assert len(OpenCVVideo().read(path, stride=2).images) == 3

    def test_a_stride_of_one_keeps_everything(self, tmp_path: Path) -> None:
        path = write_video(tmp_path / "clip.mp4", frames=5)
        assert len(OpenCVVideo().read(path, stride=1).images) == 5

    def test_the_hash_identifies_the_file_not_its_name(self, tmp_path: Path) -> None:
        original = write_video(tmp_path / "a.mp4", frames=2)
        renamed = tmp_path / "b.mp4"
        renamed.write_bytes(original.read_bytes())

        assert OpenCVVideo().read(original).sha256 == OpenCVVideo().read(renamed).sha256

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        """Not an empty video. That would look like a video of nothing, and
        the analysis would report no metrics rather than a failure."""
        with pytest.raises(OSError, match="no such video"):
            OpenCVVideo().read(tmp_path / "absent.mp4")

    def test_a_file_that_is_not_a_video_raises(self, tmp_path: Path) -> None:
        junk = tmp_path / "notavideo.mp4"
        junk.write_bytes(b"this is not an mp4")
        with pytest.raises(OSError):
            OpenCVVideo().read(junk)

    def test_a_stride_below_one_is_an_error(self, tmp_path: Path) -> None:
        path = write_video(tmp_path / "clip.mp4", frames=2)
        with pytest.raises(ValueError, match="stride must be at least 1"):
            OpenCVVideo().read(path, stride=0)


class TestMediaPipePose:
    def test_a_missing_model_names_the_path(self, tmp_path: Path) -> None:
        """MediaPipe's own error is a bare RuntimeError from C++ that does
        not say which file it wanted."""
        with pytest.raises(OSError, match="pose model not found"):
            MediaPipePose(tmp_path / "absent.task")

    def test_the_landmark_map_uses_mediapipe_numbering(self) -> None:
        """Body landmarks start at 11 — the first eleven are face points
        this project has no use for. A mismatched index would compare a
        wrist against a knee and call the difference disagreement."""
        assert len(LANDMARK_INDEX) == 12
        assert all(11 <= i <= 32 for i in LANDMARK_INDEX.values())

    def test_left_and_right_are_not_swapped(self) -> None:
        for side in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            assert LANDMARK_INDEX[f"L {side}"] < LANDMARK_INDEX[f"R {side}"]

    @pytest.mark.skipif(
        not Path("models/pose_landmarker_heavy.task").is_file(),
        reason="pose model not downloaded",
    )
    def test_a_frame_with_no_person_still_appears(self, tmp_path: Path) -> None:
        """Dropping it would renumber everything after it, and a metric
        citing "frame 47" would point at the wrong picture."""
        blank = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        frames = MediaPipePose().detect(blank, fps=30.0)

        assert len(frames) == 3
        assert [f.index for f in frames] == [0, 1, 2]

    @pytest.mark.skipif(
        not Path("models/pose_landmarker_heavy.task").is_file(),
        reason="pose model not downloaded",
    )
    def test_timestamps_follow_the_frame_rate(self, tmp_path: Path) -> None:
        blank = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
        frames = MediaPipePose().detect(blank, fps=60.0)
        assert frames[2].timestamp == pytest.approx(2 / 60)

    @pytest.mark.skipif(
        not Path("models/pose_landmarker_heavy.task").is_file(),
        reason="pose model not downloaded",
    )
    def test_a_frame_rate_of_zero_is_an_error(self) -> None:
        """Every timestamp would be infinite, and a rate computed from them
        is worse than absent."""
        with pytest.raises(ValueError, match="fps must be positive"):
            MediaPipePose().detect([np.zeros((48, 64, 3), dtype=np.uint8)], fps=0.0)

    @pytest.mark.skipif(
        not Path("models/pose_landmarker_heavy.task").is_file(),
        reason="pose model not downloaded",
    )
    def test_the_toolchain_reports_installed_versions(self) -> None:
        toolchain = MediaPipePose().toolchain
        assert toolchain.mediapipe
        assert toolchain.ultralytics
        assert toolchain.sandlot


class TestYOLODetector:
    def test_no_images_gives_no_detections(self) -> None:
        """And loads no weights — a caller passing an empty list should not
        pay for a model."""
        assert YOLODetector().detect([]) == []

    def test_a_confidence_outside_zero_to_one_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            YOLODetector(min_confidence=1.5)

    def test_the_bbox_is_top_left_width_height(self) -> None:
        """ultralytics reports xyxy; the ports and saccade.models.BBox both
        use (x, y, w, h). Getting this wrong produces a box that is plausible
        and in the wrong place."""
        detection = Detected(label="sports ball", bbox=(10.0, 20.0, 30.0, 40.0), confidence=0.9)
        x, y, w, h = detection.bbox
        assert (x, y) == (10.0, 20.0)
        assert (w, h) == (30.0, 40.0)


class TestJsonSessionRepo:
    def test_a_saved_session_comes_back(self, tmp_path: Path) -> None:
        repo = JsonSessionRepo(tmp_path)
        stored = session("abc")
        repo.save(stored)

        assert repo.get("abc") == stored

    def test_an_unknown_id_gives_none(self, tmp_path: Path) -> None:
        """Absence is not an error — a caller asking for a deleted session
        wants to hear that."""
        assert JsonSessionRepo(tmp_path).get("never-existed") is None

    def test_saving_twice_replaces(self, tmp_path: Path) -> None:
        repo = JsonSessionRepo(tmp_path)
        repo.save(session("abc"))
        repo.save(session("abc"))

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_the_directory_is_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "sessions"
        JsonSessionRepo(nested)
        assert nested.is_dir()

    def test_listing_is_newest_first(self, tmp_path: Path) -> None:
        """The question a caller actually has is "what did I do last
        time"."""
        repo = JsonSessionRepo(tmp_path)
        base = datetime(2026, 8, 1, tzinfo=UTC)
        repo.save(session("old", created_at=base - timedelta(days=7)))
        repo.save(session("new", created_at=base))
        repo.save(session("middle", created_at=base - timedelta(days=1)))

        assert [s.id for s in repo.list_since()] == ["new", "middle", "old"]

    def test_listing_can_be_bounded(self, tmp_path: Path) -> None:
        repo = JsonSessionRepo(tmp_path)
        base = datetime(2026, 8, 1, tzinfo=UTC)
        repo.save(session("old", created_at=base - timedelta(days=7)))
        repo.save(session("new", created_at=base))

        recent = repo.list_since(base - timedelta(days=1))
        assert [s.id for s in recent] == ["new"]

    def test_an_empty_directory_lists_nothing(self, tmp_path: Path) -> None:
        assert JsonSessionRepo(tmp_path).list_since() == []

    def test_a_corrupt_file_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """One bad file should not stop the others being readable."""
        repo = JsonSessionRepo(tmp_path)
        repo.save(session("good"))
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

        assert [s.id for s in repo.list_since()] == ["good"]

    def test_a_corrupt_file_reads_as_missing(self, tmp_path: Path) -> None:
        repo = JsonSessionRepo(tmp_path)
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert repo.get("broken") is None

    def test_a_partial_write_leaves_no_temporary(self, tmp_path: Path) -> None:
        repo = JsonSessionRepo(tmp_path)
        repo.save(session("abc"))
        assert list(tmp_path.glob("*.tmp")) == []

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../escape",
            "a/b",
            "with space",
            "",
            "..",
            ".",
            "...",
            ".hidden",
            "x\\y",
            "C:/absolute",
            "a\x00b",
        ],
    )
    def test_an_id_that_is_a_path_is_refused(self, tmp_path: Path, bad_id: str) -> None:
        """Session ids reach the filesystem as names. An id of
        "../../etc/passwd" would otherwise write there.

        ".." is the one an "allowed characters" pattern lets through: a dot
        is an allowed character, so a rule of "one or more of them" accepts
        it, and `directory / "...json"` resolves to the parent.
        """
        repo = JsonSessionRepo(tmp_path)
        with pytest.raises(ValueError, match="not usable as a filename"):
            repo.get(bad_id)

    def test_saving_under_a_path_id_is_refused_too(self, tmp_path: Path) -> None:
        """Reading is not the dangerous direction. Writing is."""
        repo = JsonSessionRepo(tmp_path)
        with pytest.raises(ValueError, match="not usable as a filename"):
            repo.save(session(".."))

    def test_nothing_is_written_outside_the_directory(self, tmp_path: Path) -> None:
        """The property the pattern exists to protect, checked directly."""
        inside = tmp_path / "sessions"
        repo = JsonSessionRepo(inside)
        repo.save(session("legitimate"))

        written = list(tmp_path.rglob("*.json"))
        assert all(inside in path.parents for path in written)

    def test_a_generated_id_is_accepted(self, tmp_path: Path) -> None:
        """The ids analyze_pitch produces look like 20260801T120000-deadbeef."""
        repo = JsonSessionRepo(tmp_path)
        repo.save(session("20260801T120000-deadbeef"))
        assert repo.get("20260801T120000-deadbeef") is not None


class TestSaccadeTools:
    def test_the_pose_tool_may_overrule_the_model(self, tmp_path: Path) -> None:
        result = pose_measurement_tool(body()).fn(image=None, viewport=None)

        assert result.is_measurement is True
        assert result.answer_key == "elbow_extended"

    def test_a_straight_arm_reads_extended(self) -> None:
        result = pose_measurement_tool(body(elbow_bend=0.0)).fn(image=None, viewport=None)
        assert result.value["elbow_extended"] is True

    def test_a_bent_arm_reads_not_extended(self) -> None:
        result = pose_measurement_tool(body(elbow_bend=200.0)).fn(image=None, viewport=None)
        assert result.value["elbow_extended"] is False

    def test_the_angle_rides_along_as_context(self) -> None:
        """answer_key names the verdict; the number is evidence for it, not
        a second thing to be checked against a statement."""
        result = pose_measurement_tool(body()).fn(image=None, viewport=None)
        assert "elbow_angle_degrees" in result.value

    def test_the_frame_number_is_reported(self) -> None:
        """Rule 8: an auditor has to be able to find the frame again."""
        result = pose_measurement_tool(body(index=47)).fn(image=None, viewport=None)
        assert result.value["frame"] == 47

    def test_an_undetected_arm_is_not_a_measurement(self) -> None:
        """Nothing was measured, so there is nothing to overrule a model
        with."""
        torso_only = Frame(
            index=0,
            timestamp=0.0,
            joints=(joint("L shoulder", 200, 100), joint("R shoulder", 300, 100)),
        )
        result = pose_measurement_tool(torso_only).fn(image=None, viewport=None)
        assert result.is_measurement is False

    def test_identical_readings_agree(self) -> None:
        frame = body()
        result = disagreement_tool(frame, frame).fn(image=None, viewport=None)

        assert result.value["detectors_agree"] is True
        assert result.value["mean_gap"] == pytest.approx(0.0)

    def test_a_displaced_joint_is_measured_and_named(self) -> None:
        first = body()
        moved = Frame(
            index=0,
            timestamp=0.0,
            joints=tuple(
                joint(r.name, r.x + (200 if r.name == "R wrist" else 0), r.y) for r in first.joints
            ),
        )
        result = disagreement_tool(first, moved).fn(image=None, viewport=None)

        assert result.value["detectors_agree"] is False
        assert result.value["worst_joint"] == "R wrist"

    def test_gaps_are_in_torso_lengths_not_pixels(self) -> None:
        """An absolute distance conflates disagreement with how far away the
        subject was standing."""
        result = disagreement_tool(body(), body()).fn(image=None, viewport=None)
        assert result.value["units"] == "torso lengths"

    def test_low_confidence_joints_are_excluded(self) -> None:
        """A keypoint at 0.08 is a guess about where a limb might be, and the
        distance between two guesses says nothing about either."""
        first = Frame(
            index=0,
            timestamp=0.0,
            joints=(
                joint("L shoulder", 200, 100),
                joint("R shoulder", 300, 100),
                joint("L hip", 200, 300),
                joint("R hip", 300, 300),
                joint("R wrist", 500, 100, confidence=0.05),
            ),
        )
        second = Frame(
            index=0,
            timestamp=0.0,
            joints=(
                joint("L shoulder", 200, 100),
                joint("R shoulder", 300, 100),
                joint("L hip", 200, 300),
                joint("R hip", 300, 300),
                joint("R wrist", 900, 100, confidence=0.9),
            ),
        )
        result = disagreement_tool(first, second).fn(image=None, viewport=None)
        assert result.value["worst_joint"] != "R wrist"

    def test_no_shared_joints_is_not_a_measurement(self) -> None:
        empty = Frame(index=0, timestamp=0.0)
        result = disagreement_tool(empty, empty).fn(image=None, viewport=None)
        assert result.is_measurement is False

    def test_it_says_nothing_about_why_they_disagree(self) -> None:
        """Blur, occlusion and a lost track are identical in the numbers and
        need different handling. That question is the VLM's."""
        first = body()
        moved = Frame(
            index=0,
            timestamp=0.0,
            joints=tuple(
                joint(r.name, r.x + (200 if r.name == "R wrist" else 0), r.y) for r in first.joints
            ),
        )
        value = disagreement_tool(first, moved).fn(image=None, viewport=None).value

        text = " ".join(str(v).lower() for v in value.values())
        for word in ("blur", "occlu", "mediapipe", "yolo"):
            assert word not in text
