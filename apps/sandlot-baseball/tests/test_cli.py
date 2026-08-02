"""Tests for the command line.

The detectors are patched out. What is being checked is the translation
between arguments and use-case calls, and whether the output carries what
rule 8 requires — not whether MediaPipe works, which is measured elsewhere.

Exit codes are load-bearing: a script driving this needs to tell "the video
was unreadable" from "those two sessions cannot be compared" without parsing
English, so each is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sandlot.domain.models import Frame, JointReading, Metric, Session, Toolchain
from sandlot.interfaces.cli import (
    DEFAULT_DATA_DIR,
    INCOMPARABLE,
    INCONSISTENT,
    NOT_FOUND,
    OK,
    UNREADABLE,
    USAGE,
    build_parser,
    fingerprint,
    main,
)

TOOLCHAIN = Toolchain(mediapipe="1.0.0", ultralytics="8.4.113", sandlot="0.1.0")


def joint(name: str, x: float, y: float) -> JointReading:
    return JointReading(name=name, x=x, y=y, confidence=0.9)


def coiled(index: int, *, hip_tilt: float = 0.0) -> Frame:
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
        video_sha256="abc123def456",
        frame_count=20,
        fps=60.0,
        toolchain=toolchain,
        metrics=metrics,
    )


@dataclass
class FakeVideo:
    images: list[Any] = field(default_factory=lambda: [object()] * 4)
    sha256: str = "abc123def456"
    fps: float = 60.0

    def read(self, path: Any, *, stride: int = 1) -> FakeVideo:
        return self


@dataclass
class FakePose:
    frames: list[Frame] = field(default_factory=list)
    toolchain: Toolchain = TOOLCHAIN

    def detect(self, images: list[Any], *, fps: float) -> list[Frame]:
        return self.frames


@dataclass
class FakeObjects:
    def detect(self, images: list[Any]) -> list[list[Any]]:
        return [[] for _ in images]


@pytest.fixture
def detectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the detectors the CLI imports lazily inside _analyze."""
    frames = [coiled(i, hip_tilt=40 * i) for i in range(4)]

    import sandlot.infrastructure as infra

    monkeypatch.setattr(infra, "OpenCVVideo", lambda: FakeVideo())
    monkeypatch.setattr(infra, "MediaPipePose", lambda: FakePose(frames=frames))
    monkeypatch.setattr(infra, "YOLODetector", lambda: FakeObjects())


class TestArguments:
    def test_version_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0

    def test_no_command_prints_help_and_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == USAGE
        assert "sandlot" in capsys.readouterr().out

    def test_the_data_dir_defaults_outside_the_repo(self) -> None:
        """So analysing a video never leaves files in a checkout."""
        assert DEFAULT_DATA_DIR.is_absolute()
        assert Path.cwd() not in DEFAULT_DATA_DIR.parents

    def test_the_data_dir_is_overridable(self, tmp_path: Path) -> None:
        args = build_parser().parse_args(["--data-dir", str(tmp_path), "analyze", "v.mov"])
        assert args.data_dir == tmp_path

    def test_movement_defaults_to_pitch(self) -> None:
        assert build_parser().parse_args(["analyze", "v.mov"]).movement == "pitch"

    def test_an_unknown_movement_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["analyze", "v.mov", "--movement", "sprint"])


class TestAnalyze:
    def test_it_prints_a_metric_with_its_frame(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], detectors: None
    ) -> None:
        """Rule 8: a person reading "63.9 degrees" has to be able to open the
        video at that frame and look."""
        assert main(["--data-dir", str(tmp_path), "analyze", "clip.mov"]) == OK

        out = capsys.readouterr().out
        assert "hip_shoulder_separation" in out
        assert "frame" in out

    def test_it_names_the_detector_versions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], detectors: None
    ) -> None:
        """Part of the answer: a later comparison is refused across them."""
        main(["--data-dir", str(tmp_path), "analyze", "clip.mov"])
        assert "mediapipe 1.0.0" in capsys.readouterr().out

    def test_it_stores_by_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], detectors: None
    ) -> None:
        main(["--data-dir", str(tmp_path), "analyze", "clip.mov"])
        assert list(tmp_path.glob("*.json"))
        assert "stored as" in capsys.readouterr().out

    def test_no_save_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], detectors: None
    ) -> None:
        main(["--data-dir", str(tmp_path), "analyze", "clip.mov", "--no-save"])
        assert list(tmp_path.glob("*.json")) == []
        assert "stored as" not in capsys.readouterr().out

    def test_a_swing_uses_the_object_detector(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], detectors: None
    ) -> None:
        assert (
            main(["--data-dir", str(tmp_path), "analyze", "clip.mov", "--movement", "swing"]) == OK
        )

    def test_a_video_that_will_not_decode_reports_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit 3, not 0. A command that appears to succeed while measuring
        nothing is worse than one that refuses."""
        import sandlot.infrastructure as infra

        class Broken:
            def read(self, path: Any, *, stride: int = 1) -> Any:
                raise OSError(f"no such video: {path}")

        monkeypatch.setattr(infra, "OpenCVVideo", lambda: Broken())
        monkeypatch.setattr(infra, "MediaPipePose", lambda: FakePose())

        assert main(["--data-dir", str(tmp_path), "analyze", "gone.mov"]) == UNREADABLE
        assert "no such video" in capsys.readouterr().err

    def test_a_missing_pose_model_reports_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sandlot.infrastructure as infra

        def missing() -> Any:
            raise OSError("pose model not found at models/x.task")

        monkeypatch.setattr(infra, "OpenCVVideo", lambda: FakeVideo())
        monkeypatch.setattr(infra, "MediaPipePose", missing)

        assert main(["--data-dir", str(tmp_path), "analyze", "clip.mov"]) == UNREADABLE
        assert "pose model not found" in capsys.readouterr().err

    def test_a_video_with_nothing_detectable_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A result, not a failure — and the output has to say which."""
        import sandlot.infrastructure as infra

        blank = [Frame(index=i, timestamp=i / 60) for i in range(4)]
        monkeypatch.setattr(infra, "OpenCVVideo", lambda: FakeVideo())
        monkeypatch.setattr(infra, "MediaPipePose", lambda: FakePose(frames=blank))

        assert main(["--data-dir", str(tmp_path), "analyze", "clip.mov"]) == OK
        assert "no usable joints" in capsys.readouterr().out


class TestRepeat:
    def test_it_reports_one_fingerprint_when_the_numbers_agree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], detectors: None
    ) -> None:
        assert main(["--data-dir", str(tmp_path), "analyze", "clip.mov", "--repeat", "3"]) == OK
        assert "distinct fingerprints across 3 runs: 1" in capsys.readouterr().out

    def test_repeated_runs_store_nothing(self, tmp_path: Path, detectors: None) -> None:
        """Ten runs that each wrote a session would leave nine to clean up."""
        main(["--data-dir", str(tmp_path), "analyze", "clip.mov", "--repeat", "3"])
        assert list(tmp_path.glob("*.json")) == []

    def test_differing_numbers_fail_loudly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of --repeat. A detector that drifts must not be
        reported as a successful analysis."""
        import sandlot.infrastructure as infra

        @dataclass
        class DriftingPose:
            """A detector that gives a different answer each time it is asked.

            The drift has to be in ``detect`` rather than in construction:
            the CLI builds the detector once outside the repeat loop, which
            is what keeps a ten-run check from reloading the model ten times.
            """

            toolchain: Toolchain = TOOLCHAIN
            calls: int = 0

            def detect(self, images: list[Any], *, fps: float) -> list[Frame]:
                self.calls += 1
                tilt = 40.0 * self.calls
                return [coiled(i, hip_tilt=tilt * i) for i in range(4)]

        monkeypatch.setattr(infra, "OpenCVVideo", lambda: FakeVideo())
        monkeypatch.setattr(infra, "MediaPipePose", lambda: DriftingPose())

        assert (
            main(["--data-dir", str(tmp_path), "analyze", "clip.mov", "--repeat", "2"])
            == INCONSISTENT
        )
        assert "different numbers" in capsys.readouterr().err

    def test_a_repeat_below_one_is_a_usage_error(self, tmp_path: Path) -> None:
        assert main(["--data-dir", str(tmp_path), "analyze", "clip.mov", "--repeat", "0"]) == USAGE


class TestFingerprint:
    def test_the_same_metrics_hash_the_same(self) -> None:
        metric = Metric(name="sep", value=1.0, unit="deg", frames=(3,))
        assert fingerprint(session("a", metric)) == fingerprint(session("b", metric))

    def test_the_id_and_timestamp_are_excluded(self) -> None:
        """They must differ between runs, so what remains is exactly the
        claim being checked for stability."""
        metric = Metric(name="sep", value=1.0, unit="deg", frames=(3,))
        early = session("a", metric, created_at=datetime(2020, 1, 1, tzinfo=UTC))
        late = session("z", metric, created_at=datetime(2030, 1, 1, tzinfo=UTC))
        assert fingerprint(early) == fingerprint(late)

    def test_a_different_value_hashes_differently(self) -> None:
        first = Metric(name="sep", value=1.0, unit="deg", frames=(3,))
        second = Metric(name="sep", value=1.5, unit="deg", frames=(3,))
        assert fingerprint(session("a", first)) != fingerprint(session("a", second))

    def test_a_different_frame_hashes_differently(self) -> None:
        """The frame is part of the claim: the same angle from a different
        moment is a different finding."""
        first = Metric(name="sep", value=1.0, unit="deg", frames=(3,))
        second = Metric(name="sep", value=1.0, unit="deg", frames=(9,))
        assert fingerprint(session("a", first)) != fingerprint(session("a", second))


class TestCompare:
    def stored(self, tmp_path: Path) -> None:
        from sandlot.infrastructure import JsonSessionRepo

        repo = JsonSessionRepo(tmp_path)
        repo.save(session("take-a", Metric(name="sep", value=40.0, unit="deg", frames=(3,))))
        repo.save(session("take-b", Metric(name="sep", value=35.2, unit="deg", frames=(5,))))

    def test_it_prints_the_change(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.stored(tmp_path)
        assert main(["--data-dir", str(tmp_path), "compare", "take-a", "take-b"]) == OK

        out = capsys.readouterr().out
        assert "-4.800" in out
        assert "sep" in out

    def test_the_sign_says_which_way(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.stored(tmp_path)
        main(["--data-dir", str(tmp_path), "compare", "take-b", "take-a"])
        assert "+4.800" in capsys.readouterr().out

    def test_an_unknown_session_reports_which(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self.stored(tmp_path)
        assert main(["--data-dir", str(tmp_path), "compare", "nope", "take-b"]) == NOT_FOUND
        assert "nope" in capsys.readouterr().err

    def test_different_toolchains_are_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A caller shown "no change" after a detector upgrade would draw
        exactly the wrong conclusion."""
        from sandlot.infrastructure import JsonSessionRepo

        other = Toolchain(mediapipe="9.9.9", ultralytics="8.4.113", sandlot="0.1.0")
        repo = JsonSessionRepo(tmp_path)
        repo.save(session("old", Metric(name="sep", value=40.0, unit="deg", frames=(3,))))
        repo.save(
            session(
                "new",
                Metric(name="sep", value=35.0, unit="deg", frames=(3,)),
                toolchain=other,
            )
        )

        assert main(["--data-dir", str(tmp_path), "compare", "old", "new"]) == INCOMPARABLE
        assert "toolchains differ" in capsys.readouterr().err

    def test_a_metric_measurable_in_only_one_session_is_explained(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ "Nothing moved" and "nobody looked" are different findings, and
        the output has to distinguish them."""
        from sandlot.infrastructure import JsonSessionRepo

        repo = JsonSessionRepo(tmp_path)
        repo.save(session("full", Metric(name="stride", value=1.2, unit="torsos", frames=(3,))))
        repo.save(session("partial"))

        assert main(["--data-dir", str(tmp_path), "compare", "full", "partial"]) == OK

        out = capsys.readouterr().out
        assert "not comparable" in out
        assert "no measurement in the later session" in out

    def test_two_empty_sessions_say_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from sandlot.infrastructure import JsonSessionRepo

        repo = JsonSessionRepo(tmp_path)
        repo.save(session("a"))
        repo.save(session("b"))

        assert main(["--data-dir", str(tmp_path), "compare", "a", "b"]) == OK
        assert "nothing to compare" in capsys.readouterr().out

    def test_a_session_id_that_is_a_path_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The repository refuses it; the CLI must turn that into an exit
        code rather than a traceback."""
        assert main(["--data-dir", str(tmp_path), "compare", "..", "take-b"]) == USAGE
        assert "not usable as a filename" in capsys.readouterr().err


def test_the_exit_codes_are_distinct() -> None:
    """A script driving this tells the failures apart by number."""
    codes = [OK, USAGE, NOT_FOUND, UNREADABLE, INCOMPARABLE, INCONSISTENT]
    assert len(set(codes)) == len(codes)
