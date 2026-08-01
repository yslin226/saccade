"""Tests for the application layer.

Every port is faked. A use case that could only be tested with MediaPipe
installed and a video on disk would be one whose orchestration is never
checked separately from the things it orchestrates — and the orchestration is
all a use case is.

The fakes are deliberately dumb: they hand back what they were given. Their
job is to prove the flow calls them in the right order with the right
arguments, not to simulate a detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sandlot.application.ports import PosePort, SessionRepoPort, VideoPort
from sandlot.application.use_cases import (
    AnalysisFailedError,
    SessionNotFoundError,
    analyze_pitch,
    compare_sessions,
    compare_with_previous,
    measure,
)
from sandlot.domain.comparison import IncomparableSessionsError
from sandlot.domain.models import Frame, JointReading, Metric, Session, Toolchain

TOOLCHAIN = Toolchain(mediapipe="1.0.0", ultralytics="8.4.113", sandlot="0.1.0")
OTHER_TOOLCHAIN = Toolchain(mediapipe="9.9.9", ultralytics="8.4.113", sandlot="0.1.0")


# --- fakes ----------------------------------------------------------------


@dataclass
class FakeVideo:
    """A decoder that returns whatever it was constructed with."""

    images: list[Any] = field(default_factory=lambda: [object()])
    sha256: str = "abc123"
    fps: float = 60.0
    reads: list[tuple[Any, int]] = field(default_factory=list)

    def read(self, path: Any, *, stride: int = 1) -> FakeVideo:
        self.reads.append((path, stride))
        return self


@dataclass
class FakePose:
    """A detector that hands back pre-built frames."""

    frames: list[Frame] = field(default_factory=list)
    toolchain: Toolchain = TOOLCHAIN
    calls: list[float] = field(default_factory=list)

    def detect(self, images: list[Any], *, fps: float) -> list[Frame]:
        self.calls.append(fps)
        return self.frames


@dataclass
class FakeRepo:
    """An in-memory session store."""

    stored: dict[str, Session] = field(default_factory=dict)

    def save(self, session: Session) -> None:
        self.stored[session.id] = session

    def get(self, session_id: str) -> Session | None:
        return self.stored.get(session_id)

    def list_since(self, since: datetime | None = None) -> list[Session]:
        sessions = [s for s in self.stored.values() if since is None or s.created_at >= since]
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)


# --- fixtures -------------------------------------------------------------


def joint(name: str, x: float, y: float) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=0.9)


def coiled(index: int, *, hip_tilt: float) -> Frame:
    """A frame with the hips rotated by roughly ``hip_tilt`` pixels."""
    return Frame(
        index=index,
        timestamp=index / 60,
        joints=(
            joint("L shoulder", 200, 100),
            joint("R shoulder", 300, 100),
            joint("L hip", 200, 300),
            joint("R hip", 300, 300 + hip_tilt),
            joint("L ankle", 190, 600),
            joint("R ankle", 310, 600),
        ),
    )


def session(
    session_id: str,
    *metrics: Metric,
    created_at: datetime | None = None,
    toolchain: Toolchain = TOOLCHAIN,
) -> Session:
    return Session(
        id=session_id,
        created_at=created_at or datetime(2026, 8, 1, tzinfo=UTC),
        video_sha256="abc",
        frame_count=10,
        fps=60.0,
        toolchain=toolchain,
        metrics=metrics,
    )


# --- tests ----------------------------------------------------------------


class TestAnalyzePitch:
    def test_it_decodes_detects_and_measures(self) -> None:
        video = FakeVideo(images=[object(), object()])
        pose = FakePose(frames=[coiled(0, hip_tilt=0), coiled(1, hip_tilt=100)])

        result = analyze_pitch("clip.mov", video=video, pose=pose)

        assert video.reads == [("clip.mov", 1)]
        assert pose.calls == [60.0]
        assert result.frame_count == 2
        assert result.metrics

    def test_the_video_hash_identifies_the_session(self) -> None:
        """Not the filename. Two analyses of the same delivery stay
        comparable however the file was renamed."""
        video = FakeVideo(sha256="deadbeef")
        result = analyze_pitch("a.mov", video=video, pose=FakePose(frames=[coiled(0, hip_tilt=0)]))

        assert result.video_sha256 == "deadbeef"
        assert "deadbeef"[:8] in result.id

    def test_the_toolchain_is_recorded(self) -> None:
        """Part of the answer: a later comparison is refused without it."""
        result = analyze_pitch(
            "a.mov", video=FakeVideo(), pose=FakePose(frames=[coiled(0, hip_tilt=0)])
        )
        assert result.toolchain == TOOLCHAIN

    def test_stride_is_passed_through_to_the_decoder(self) -> None:
        video = FakeVideo()
        analyze_pitch("a.mov", video=video, pose=FakePose(), stride=5)
        assert video.reads == [("a.mov", 5)]

    def test_it_saves_when_given_a_repository(self) -> None:
        repo = FakeRepo()
        result = analyze_pitch(
            "a.mov", video=FakeVideo(), pose=FakePose(frames=[coiled(0, hip_tilt=0)]), repo=repo
        )
        assert repo.get(result.id) == result

    def test_no_repository_means_no_write(self) -> None:
        """What a determinism check wants: ten runs that each wrote a
        session would leave nine to clean up."""
        repo = FakeRepo()
        analyze_pitch("a.mov", video=FakeVideo(), pose=FakePose())
        assert repo.stored == {}

    def test_an_explicit_id_is_honoured(self) -> None:
        result = analyze_pitch("a.mov", video=FakeVideo(), pose=FakePose(), session_id="fixed-id")
        assert result.id == "fixed-id"

    def test_a_video_with_no_frames_fails_loudly(self) -> None:
        """Not an empty session. "The file could not be read" and "the
        detector found nothing" are different, and only one is a bug."""
        with pytest.raises(AnalysisFailedError, match="no frames"):
            analyze_pitch("empty.mov", video=FakeVideo(images=[]), pose=FakePose())

    def test_frames_with_nothing_detected_give_a_session_with_no_metrics(self) -> None:
        """A legitimate result, distinct from the failure above."""
        blank = [Frame(index=i, timestamp=i / 60) for i in range(3)]
        result = analyze_pitch("a.mov", video=FakeVideo(), pose=FakePose(frames=blank))

        assert result.metrics == ()
        assert result.frame_count == 3

    def test_it_satisfies_the_ports_it_declares(self) -> None:
        assert isinstance(FakeVideo(), VideoPort)
        assert isinstance(FakePose(), PosePort)
        assert isinstance(FakeRepo(), SessionRepoPort)


class TestMeasure:
    def test_peak_separation_is_taken_over_the_whole_delivery(self) -> None:
        """Not from whichever frame happened to be sampled."""
        frames = [coiled(0, hip_tilt=0), coiled(1, hip_tilt=100), coiled(2, hip_tilt=20)]
        metrics = {m.name: m for m in measure(frames)}

        separation = metrics["hip_shoulder_separation"]
        assert separation.value == pytest.approx(45.0)
        assert separation.frames == (1,)

    def test_stride_is_taken_at_the_same_frame_as_the_peak(self) -> None:
        """Two metrics from different instants reported together imply they
        describe the same moment. They must actually do so."""
        frames = [coiled(0, hip_tilt=0), coiled(1, hip_tilt=100), coiled(2, hip_tilt=20)]
        metrics = {m.name: m for m in measure(frames)}

        assert metrics["stride_length"].frames == metrics["hip_shoulder_separation"].frames

    def test_elbow_flexion_takes_the_most_flexed_frame(self) -> None:
        """The minimum, not the maximum: a straight arm is 180 degrees and
        the interesting moment is the bend."""
        frames = [
            Frame(
                index=i,
                timestamp=i / 60,
                joints=(
                    joint("R shoulder", 100, 100),
                    joint("R elbow", 200, 100),
                    joint("R wrist", 300 - 100 * i, 100 + 100 * i),
                ),
            )
            for i in range(2)
        ]
        metrics = {m.name: m for m in measure(frames)}

        assert metrics["elbow_flexion_R"].value == pytest.approx(90.0)
        assert metrics["elbow_flexion_R"].frames == (1,)

    def test_each_metric_cites_a_frame(self) -> None:
        """Rule 8, checked on real output rather than trusted."""
        frames = [coiled(i, hip_tilt=i * 30) for i in range(4)]
        for metric in measure(frames):
            assert metric.frames

    def test_every_metric_carries_a_unit(self) -> None:
        frames = [coiled(i, hip_tilt=i * 30) for i in range(4)]
        for metric in measure(frames):
            assert metric.unit

    def test_an_unmeasurable_metric_is_absent_not_null(self) -> None:
        """ "We looked and found nothing" is the absence of a claim."""
        no_legs = [
            Frame(
                index=i,
                timestamp=i / 60,
                joints=(
                    joint("L shoulder", 200, 100),
                    joint("R shoulder", 300, 100),
                    joint("L hip", 200, 300),
                    joint("R hip", 300, 300 + 40 * i),
                ),
            )
            for i in range(3)
        ]
        names = {m.name for m in measure(no_legs)}
        assert "hip_shoulder_separation" in names
        assert "stride_length" not in names

    def test_no_frames_gives_no_metrics(self) -> None:
        assert measure([]) == []

    def test_a_ground_up_chain_scores_one(self) -> None:
        frames = [
            Frame(
                index=i,
                timestamp=i / 60,
                joints=(
                    joint("L hip", 200, 300),
                    joint("R hip", 300, 300 + (60 if i >= 2 else 0)),
                    joint("L shoulder", 200, 100),
                    joint("R shoulder", 300, 100 + (60 if i >= 4 else 0)),
                ),
            )
            for i in range(5)
        ]
        chain = next(m for m in measure(frames) if m.name == "kinetic_chain_order")
        assert chain.value == pytest.approx(1.0)
        assert chain.detail["order"] == ["hips", "shoulders"]

    def test_an_out_of_order_chain_scores_zero(self) -> None:
        """All-arm: the shoulders fire before the hips."""
        frames = [
            Frame(
                index=i,
                timestamp=i / 60,
                joints=(
                    joint("L hip", 200, 300),
                    joint("R hip", 300, 300 + (60 if i >= 4 else 0)),
                    joint("L shoulder", 200, 100),
                    joint("R shoulder", 300, 100 + (60 if i >= 2 else 0)),
                ),
            )
            for i in range(5)
        ]
        chain = next(m for m in measure(frames) if m.name == "kinetic_chain_order")
        assert chain.value == pytest.approx(0.0)

    def test_the_expected_order_is_stated_alongside_the_score(self) -> None:
        """A score with nothing to compare it against is not evidence."""
        frames = [coiled(i, hip_tilt=i * 30) for i in range(4)]
        chain = next(m for m in measure(frames) if m.name == "kinetic_chain_order")
        assert chain.detail["expected"] == ["hips", "shoulders", "elbow", "wrist"]

    def test_a_single_segment_scores_zero(self) -> None:
        """One segment has no successor, so nothing fired in order. Zero
        rather than one: an unordered sequence is not a correct one, and
        rewarding it would let a delivery where only the hips were visible
        outscore one where the whole chain was measured and went wrong."""
        hips_only = [
            Frame(
                index=i,
                timestamp=i / 60,
                joints=(
                    joint("L hip", 200, 300),
                    joint("R hip", 300, 300 + (60 if i >= 2 else 0)),
                ),
            )
            for i in range(4)
        ]
        chain = next(m for m in measure(hips_only) if m.name == "kinetic_chain_order")
        assert chain.detail["order"] == ["hips"]
        assert chain.value == pytest.approx(0.0)


class TestCompareSessions:
    def test_it_differences_two_stored_sessions(self) -> None:
        repo = FakeRepo()
        repo.save(session("a", Metric(name="sep", value=40.0, unit="deg", frames=(0,))))
        repo.save(session("b", Metric(name="sep", value=35.2, unit="deg", frames=(0,))))

        deltas = compare_sessions("a", "b", repo=repo)
        assert deltas[0].change == pytest.approx(-4.8)

    def test_an_unknown_before_id_names_itself(self) -> None:
        """A caller who mistyped needs to know which of the two they got
        wrong."""
        repo = FakeRepo()
        repo.save(session("b"))
        with pytest.raises(SessionNotFoundError, match="'missing'"):
            compare_sessions("missing", "b", repo=repo)

    def test_an_unknown_after_id_names_itself(self) -> None:
        repo = FakeRepo()
        repo.save(session("a"))
        with pytest.raises(SessionNotFoundError, match="'gone'"):
            compare_sessions("a", "gone", repo=repo)

    def test_a_toolchain_mismatch_propagates(self) -> None:
        """Not caught and turned into "no change" — that would be read as
        "you did the same thing"."""
        repo = FakeRepo()
        repo.save(session("a"))
        repo.save(session("b", toolchain=OTHER_TOOLCHAIN))
        with pytest.raises(IncomparableSessionsError):
            compare_sessions("a", "b", repo=repo)


class TestCompareWithPrevious:
    def base(self) -> datetime:
        return datetime(2026, 8, 1, tzinfo=UTC)

    def test_it_finds_the_most_recent_earlier_session(self) -> None:
        repo = FakeRepo()
        old = session(
            "old",
            Metric(name="sep", value=30.0, unit="deg", frames=(0,)),
            created_at=self.base() - timedelta(days=7),
        )
        recent = session(
            "recent",
            Metric(name="sep", value=35.0, unit="deg", frames=(0,)),
            created_at=self.base() - timedelta(days=1),
        )
        current = session(
            "current",
            Metric(name="sep", value=40.0, unit="deg", frames=(0,)),
            created_at=self.base(),
        )
        for stored in (old, recent, current):
            repo.save(stored)

        result = compare_with_previous(current, repo=repo)
        assert result is not None
        previous, deltas = result
        assert previous.id == "recent"
        assert deltas[0].change == pytest.approx(5.0)

    def test_a_first_session_has_nothing_to_compare_against(self) -> None:
        """None, not an empty list: "no previous session" is not the same as
        "compared and found nothing"."""
        repo = FakeRepo()
        only = session("only", created_at=self.base())
        repo.save(only)

        assert compare_with_previous(only, repo=repo) is None

    def test_later_sessions_are_ignored(self) -> None:
        repo = FakeRepo()
        current = session("current", created_at=self.base())
        later = session("later", created_at=self.base() + timedelta(days=1))
        repo.save(current)
        repo.save(later)

        assert compare_with_previous(current, repo=repo) is None

    def test_the_session_does_not_compare_against_itself(self) -> None:
        repo = FakeRepo()
        current = session("current", created_at=self.base())
        repo.save(current)
        assert compare_with_previous(current, repo=repo) is None

    def test_an_unsaved_session_still_compares(self) -> None:
        """analyze_pitch may return without storing, and the caller still
        wants to know what changed."""
        repo = FakeRepo()
        repo.save(
            session(
                "old",
                Metric(name="sep", value=30.0, unit="deg", frames=(0,)),
                created_at=self.base() - timedelta(days=1),
            )
        )
        fresh = session(
            "fresh",
            Metric(name="sep", value=40.0, unit="deg", frames=(0,)),
            created_at=self.base(),
        )

        result = compare_with_previous(fresh, repo=repo)
        assert result is not None
        assert result[1][0].change == pytest.approx(10.0)

    def test_a_toolchain_mismatch_is_not_swallowed(self) -> None:
        """A caller shown "no change" after an upgrade would draw the wrong
        conclusion."""
        repo = FakeRepo()
        repo.save(
            session("old", created_at=self.base() - timedelta(days=1), toolchain=OTHER_TOOLCHAIN)
        )
        current = session("current", created_at=self.base())

        with pytest.raises(IncomparableSessionsError):
            compare_with_previous(current, repo=repo)
