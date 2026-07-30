"""Does the continuity check flag real problems, or does it flag everything?

Three clips of a pitch told us MediaPipe produces impossible readings without
noticing. They could not tell us whether the check that caught them is
measuring anything: a rule that fires on every clip is noise, however
plausible its output looks.

Penn Action supplies the control. Fast actions smear a limb; slow ones do not.
If the flag rate is the same in both groups, the flags mean nothing. If it is
higher on the fast group, the check is tracking something real — and Penn
Action's hand-labelled joints let us go further and ask whether the flagged
readings are the wrong ones.

Usage:
    uv run python -m benchmarks.pose_probe.compare --clips 12
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.pose_probe.continuity import JointReading, implausible_joints
from benchmarks.pose_probe.pennaction import (
    FAST_ACTIONS,
    SLOW_ACTIONS,
    Clip,
    list_clips,
    load_clip,
)

ROOT = Path("data/PennAction/Penn_Action")
CACHE = Path("benchmarks/pose_probe/cache")
POSE_MODEL = "models/pose_landmarker_heavy.task"

# Penn Action joints that MediaPipe also reports, so the two can be compared.
SHARED = {
    "L shoulder": 11,
    "R shoulder": 12,
    "L elbow": 13,
    "R elbow": 14,
    "L wrist": 15,
    "R wrist": 16,
    "L hip": 23,
    "R hip": 24,
    "L knee": 25,
    "R knee": 26,
}


@dataclass
class ClipResult:
    """What happened on one clip."""

    name: str
    action: str
    group: str
    frames: int
    detected: int
    flagged_frames: int
    mean_error_px: float
    error_on_flagged_px: float
    error_on_clean_px: float

    @property
    def flag_rate(self) -> float:
        return self.flagged_frames / self.detected if self.detected else 0.0


def scale_to_clip(clip: Clip) -> tuple[float, float]:
    """Penn Action frames vary in size; the travel threshold is in pixels.

    Scaling it by frame width keeps "implausible" meaning the same thing
    across a 480px clip and a 1920px one.
    """
    height, width = clip.dimensions
    return float(width), float(height)


def run_clip(clip: Clip, landmarker: object) -> ClipResult | None:
    """Detect poses, compare against the labels, and count flags."""
    import cv2
    from mediapipe import Image, ImageFormat

    width, height = scale_to_clip(clip)

    readings: list[list[JointReading] | None] = []
    errors_by_frame: list[float | None] = []

    for index in range(clip.n_frames):
        path = clip.frame_path(index)
        if not path.is_file():
            readings.append(None)
            errors_by_frame.append(None)
            continue

        frame = cv2.imread(str(path))
        if frame is None:
            readings.append(None)
            errors_by_frame.append(None)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(Image(image_format=ImageFormat.SRGB, data=rgb))  # type: ignore[attr-defined]

        if not result.pose_landmarks:
            readings.append(None)
            errors_by_frame.append(None)
            continue

        marks = result.pose_landmarks[0]
        truth = clip.joints(index)

        frame_readings: list[JointReading] = []
        distances: list[float] = []
        for name, mp_index in SHARED.items():
            m = marks[mp_index]
            px, py = float(m.x) * width, float(m.y) * height
            frame_readings.append(
                JointReading(name=name, x=px, y=py, confidence=float(m.visibility))
            )

            tx, ty, visible = truth[name]
            # Only compare against joints a human could see; an invisible
            # joint has no true position to be wrong about.
            if visible:
                distances.append(float(np.hypot(px - tx, py - ty)))

        readings.append(frame_readings)
        errors_by_frame.append(statistics.mean(distances) if distances else None)

    detected = sum(1 for r in readings if r is not None)
    if detected < 2:
        return None

    flagged: set[int] = set()
    for i in range(1, len(readings)):
        before, current = readings[i - 1], readings[i]
        if before is None or current is None:
            continue
        # No limit passed: the check normalises by the subject's own shoulder
        # width, which is what makes one threshold work across clips filmed at
        # different distances.
        if implausible_joints(before, current):
            flagged.add(i)

    measured = [(i, e) for i, e in enumerate(errors_by_frame) if e is not None]
    on_flagged = [e for i, e in measured if i in flagged]
    on_clean = [e for i, e in measured if i not in flagged]

    group = "fast" if clip.action in FAST_ACTIONS else "slow"
    return ClipResult(
        name=clip.name,
        action=clip.action,
        group=group,
        frames=clip.n_frames,
        detected=detected,
        flagged_frames=len(flagged),
        mean_error_px=statistics.mean(e for _, e in measured) if measured else 0.0,
        error_on_flagged_px=statistics.mean(on_flagged) if on_flagged else 0.0,
        error_on_clean_px=statistics.mean(on_clean) if on_clean else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=10, help="clips per group")
    parser.add_argument("--pose-model", default=POSE_MODEL)
    args = parser.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    print("indexing clips...")
    fast = [n for n, a in list_clips(ROOT, FAST_ACTIONS)][: args.clips]
    slow = [n for n, a in list_clips(ROOT, SLOW_ACTIONS)][: args.clips]
    print(f"  {len(fast)} fast, {len(slow)} slow")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.pose_model),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    results: list[ClipResult] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        for group_name, names in (("fast", fast), ("slow", slow)):
            for name in names:
                clip = load_clip(ROOT, name)
                outcome = run_clip(clip, landmarker)
                if outcome is None:
                    continue
                results.append(outcome)
                print(
                    f"  [{group_name}] {name} {clip.action:<16} "
                    f"{outcome.flagged_frames:>3}/{outcome.detected:<4} flagged  "
                    f"error {outcome.mean_error_px:>5.1f}px"
                )

    print()
    print("=== does the check fire more on fast movement? ===")
    for group in ("fast", "slow"):
        rows = [r for r in results if r.group == group]
        if not rows:
            continue
        rate = statistics.mean(r.flag_rate for r in rows)
        error = statistics.mean(r.mean_error_px for r in rows)
        print(
            f"  {group:<5} n={len(rows):<3} flag rate {rate:>6.1%}   "
            f"mean joint error vs labels {error:>5.1f}px"
        )

    print()
    print("=== are the flagged frames the wrong ones? ===")
    flagged_errors = [r.error_on_flagged_px for r in results if r.error_on_flagged_px > 0]
    clean_errors = [r.error_on_clean_px for r in results if r.error_on_clean_px > 0]
    if flagged_errors and clean_errors:
        print(f"  error on flagged frames : {statistics.mean(flagged_errors):>6.1f}px")
        print(f"  error on other frames   : {statistics.mean(clean_errors):>6.1f}px")
        ratio = statistics.mean(flagged_errors) / statistics.mean(clean_errors)
        print(f"  ratio                   : {ratio:>6.2f}x")
        if ratio > 1.3:
            print("  -> flagged frames really are the worse ones")
        else:
            print("  -> the flags do not pick out worse frames; the check is noise")
    else:
        print("  not enough flagged frames to compare")

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "pennaction_comparison.json"
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8")
    print()
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
