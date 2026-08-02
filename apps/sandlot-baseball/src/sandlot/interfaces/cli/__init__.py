"""Command line entry point.

``sandlot analyze <video>`` and ``sandlot compare <a> <b>``. Arguments are
translated into use-case calls here and nowhere else — this layer holds no
rules and makes no decisions about baseball. If a change here needs a change
to a metric, the rule has leaked out of ``domain``.

Every number printed carries the frame it came from. Rule 8 is not satisfied
by storing the evidence and showing the figure; a person reading "63.9
degrees" has to be able to open the video at that frame and look.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sandlot import __version__
from sandlot.application.use_cases import (
    AnalysisFailedError,
    SessionNotFoundError,
    analyze_pitch,
    analyze_swing,
    compare_sessions,
)
from sandlot.domain.comparison import IncomparableSessionsError
from sandlot.domain.models import Metric, MetricDelta, Session

__all__ = ["DEFAULT_DATA_DIR", "build_parser", "main"]

# Sessions land in the user's home rather than the repo, so analysing a video
# does not leave files in a checkout. Every entry point takes --data-dir, and
# tests pass tmp_path, so no test can write here by accident.
DEFAULT_DATA_DIR = Path.home() / ".sandlot" / "sessions"

# Exit codes. Distinct so a script can tell "the video was unreadable" from
# "the sessions cannot be compared" without parsing English.
OK = 0
USAGE = 1
NOT_FOUND = 2
UNREADABLE = 3
INCOMPARABLE = 4
INCONSISTENT = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sandlot",
        description="Measure a pitch or a swing, and compare it against your own history.",
    )
    parser.add_argument("--version", action="version", version=f"sandlot {__version__}")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"where sessions are stored (default: {DEFAULT_DATA_DIR})",
    )

    commands = parser.add_subparsers(dest="command")

    analyse = commands.add_parser("analyze", help="measure one video")
    analyse.add_argument("video", type=Path)
    analyse.add_argument(
        "--movement",
        choices=("pitch", "swing"),
        default="pitch",
        help="a swing additionally locates the bat (default: pitch)",
    )
    analyse.add_argument(
        "--stride", type=int, default=1, help="analyse every Nth frame (default: 1)"
    )
    analyse.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "run the analysis N times and report whether the numbers agree. "
            "Repeated runs are not stored — ten runs that each wrote a session "
            "would leave nine to clean up"
        ),
    )
    analyse.add_argument(
        "--no-save", action="store_true", help="measure without storing the session"
    )

    compare = commands.add_parser("compare", help="difference between two stored sessions")
    compare.add_argument("session_a")
    compare.add_argument("session_b")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return USAGE

    if args.command == "analyze":
        return _analyze(args)
    return _compare(args)


def _analyze(args: argparse.Namespace) -> int:
    # Imported here rather than at module scope so `sandlot --help` and
    # `sandlot compare` do not pay for loading MediaPipe and torch, which
    # take several seconds between them.
    from sandlot.infrastructure import JsonSessionRepo, MediaPipePose, OpenCVVideo, YOLODetector

    if args.repeat < 1:
        print("--repeat must be at least 1", file=sys.stderr)
        return USAGE

    try:
        video, pose = OpenCVVideo(), MediaPipePose()
    except OSError as error:
        print(error, file=sys.stderr)
        return UNREADABLE

    # A repeated run never stores: it is a consistency check, and each pass
    # would otherwise write a session that differs only in its id.
    storing = not args.no_save and args.repeat == 1
    repo = JsonSessionRepo(args.data_dir) if storing else None

    fingerprints: list[str] = []
    session: Session | None = None

    for attempt in range(args.repeat):
        try:
            if args.movement == "swing":
                session = analyze_swing(
                    args.video,
                    video=video,
                    pose=pose,
                    objects=YOLODetector(),
                    repo=repo,
                    stride=args.stride,
                )
            else:
                session = analyze_pitch(
                    args.video,
                    video=video,
                    pose=pose,
                    repo=repo,
                    stride=args.stride,
                )
        except AnalysisFailedError as error:
            print(error, file=sys.stderr)
            return UNREADABLE
        except OSError as error:
            print(error, file=sys.stderr)
            return UNREADABLE

        fingerprints.append(fingerprint(session))
        if args.repeat > 1:
            print(f"  run {attempt + 1:>3}  {fingerprints[-1]}")

    assert session is not None  # the loop runs at least once
    if args.repeat > 1:
        print()

    _report(session, stored=storing)

    if args.repeat > 1:
        distinct = len(set(fingerprints))
        print()
        print(f"distinct fingerprints across {args.repeat} runs: {distinct}")
        if distinct > 1:
            print("the same video produced different numbers", file=sys.stderr)
            return INCONSISTENT

    return OK


def _compare(args: argparse.Namespace) -> int:
    from sandlot.infrastructure import JsonSessionRepo

    repo = JsonSessionRepo(args.data_dir)
    try:
        deltas = compare_sessions(args.session_a, args.session_b, repo=repo)
    except SessionNotFoundError as error:
        print(error, file=sys.stderr)
        return NOT_FOUND
    except IncomparableSessionsError as error:
        print(error, file=sys.stderr)
        return INCOMPARABLE
    except ValueError as error:
        print(error, file=sys.stderr)
        return USAGE

    if not deltas:
        print("neither session recorded a metric, so there is nothing to compare")
        return OK

    _report_deltas(args.session_a, args.session_b, deltas)
    return OK


def fingerprint(session: Session) -> str:
    """A digest of everything a user would see.

    Excludes the id and the timestamp, which must differ between runs, so
    what remains is exactly the claim being checked for stability.
    """
    payload = [
        {"name": m.name, "value": m.value, "unit": m.unit, "frames": list(m.frames)}
        for m in session.metrics
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _report(session: Session, *, stored: bool) -> None:
    print(f"session   {session.id}")
    print(
        f"video     {session.video_sha256[:16]}…  {session.frame_count} frames @ {session.fps:g}fps"
    )
    print(
        f"detectors mediapipe {session.toolchain.mediapipe}, "
        f"ultralytics {session.toolchain.ultralytics}"
    )
    print()

    if not session.metrics:
        print("no metric could be computed from this video")
        print("the detector ran and found no usable joints — this is a result, not a failure")
        return

    print(f"{'metric':<28}{'value':>12}  {'frames':<18}unit")
    for metric in session.metrics:
        print(f"{metric.name:<28}{metric.value:>12.4f}  {_frames(metric):<18}{metric.unit}")

    print()
    for metric in session.metrics:
        taken = metric.detail.get("taken_at")
        if taken:
            print(f"  {metric.name}: {taken}")

    if stored:
        print()
        print(f"stored as {session.id}")


def _report_deltas(before_id: str, after_id: str, deltas: list[MetricDelta]) -> None:
    print(f"{before_id}  ->  {after_id}")
    print()
    print(f"{'metric':<28}{'before':>10}{'after':>10}{'change':>10}  unit")

    for delta in deltas:
        before = "—" if delta.before is None else f"{delta.before:.3f}"
        after = "—" if delta.after is None else f"{delta.after:.3f}"
        change = "n/a" if delta.change is None else f"{delta.change:+.3f}"
        print(f"{delta.name:<28}{before:>10}{after:>10}{change:>10}  {delta.unit}")

    incomparable = [d for d in deltas if not d.comparable]
    if incomparable:
        print()
        print("not comparable — measurable in one session but not the other:")
        for delta in incomparable:
            missing = "the earlier" if delta.before is None else "the later"
            print(f"  {delta.name}: no measurement in {missing} session")


def _frames(metric: Metric) -> str:
    """Which frames a number came from, short enough to sit in a table.

    Rule 8 wants the evidence reachable, and a metric drawn from forty frames
    would otherwise wrap the line — the count and the range say where to look
    without printing all of them.
    """
    frames = metric.frames
    if len(frames) == 1:
        return f"frame {frames[0]}"
    return f"{len(frames)} frames, {min(frames)} to {max(frames)}"


def _entry() -> Any:
    """Console-script wrapper, so pyproject can name a zero-argument callable."""
    return sys.exit(main())


if __name__ == "__main__":
    _entry()
