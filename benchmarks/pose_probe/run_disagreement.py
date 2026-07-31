"""Run the two-detector disagreement experiment.

Usage:
    uv run python -m benchmarks.pose_probe.run_disagreement --clips 10

Reports AUROC on actions neither the threshold nor the classifier was ever
tuned on. The bar is 0.70, set in disagreement.py before any result was seen.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from benchmarks.pose_probe.continuity import JointReading, shoulder_width
from benchmarks.pose_probe.disagreement import (
    COMMON_JOINTS,
    DISAGREEMENT_FEATURES,
    SUCCESS_AUROC,
    disagreement_features,
)
from benchmarks.pose_probe.pennaction import list_clips, load_clip

ROOT = Path("data/PennAction/Penn_Action")
CACHE = Path("benchmarks/pose_probe/cache")

# Same held-out actions as the previous experiment, so the two are comparable.
TEST_ACTIONS = frozenset({"golf_swing", "tennis_serve", "squat", "pushup"})


def auroc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Rank-based AUROC, needing no threshold."""
    positives, negatives = truth.sum(), (~truth).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    return float(
        (negatives * positives - (ranks[truth].sum() - positives * (positives - 1) / 2))
        / (positives * negatives)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--pose-model", default="models/pose_landmarker_heavy.task")
    args = parser.parse_args(argv)

    import cv2
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    from ultralytics import YOLO

    by_action: dict[str, list[str]] = {}
    for name, action in list_clips(ROOT, TEST_ACTIONS):
        by_action.setdefault(action, []).append(name)
    clips = [
        (name, action)
        for action, names in sorted(by_action.items())
        for name in names[: args.clips]
    ]
    print(f"clips: {len(clips)}  actions: {sorted(by_action)}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.pose_model),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
    yolo = YOLO("yolo11x-pose.pt")

    rows: list[dict] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        for name, action in clips:
            clip = load_clip(ROOT, name)
            height, width = clip.dimensions

            for index in range(clip.n_frames):
                path = clip.frame_path(index)
                frame = cv2.imread(str(path)) if path.is_file() else None
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = landmarker.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
                if not mp_result.pose_landmarks:
                    continue

                yolo_result = yolo(frame, verbose=False)[0]
                if yolo_result.keypoints is None or len(yolo_result.keypoints.data) == 0:
                    continue

                marks = mp_result.pose_landmarks[0]
                kp = yolo_result.keypoints.data[0].cpu().numpy()

                mp_readings: list[JointReading] = []
                yolo_readings: list[JointReading] = []
                for joint, (mp_i, yolo_i) in COMMON_JOINTS.items():
                    mp_readings.append(
                        JointReading(
                            name=joint,
                            x=float(marks[mp_i].x) * width,
                            y=float(marks[mp_i].y) * height,
                            confidence=float(marks[mp_i].visibility),
                        )
                    )
                    if yolo_i < len(kp):
                        yolo_readings.append(
                            JointReading(
                                name=joint,
                                x=float(kp[yolo_i][0]),
                                y=float(kp[yolo_i][1]),
                                confidence=float(kp[yolo_i][2]),
                            )
                        )

                scale = shoulder_width(mp_readings)
                if not scale:
                    continue

                gaps = disagreement_features(mp_readings, yolo_readings, scale=scale)
                if not gaps.per_joint:
                    continue

                truth = clip.joints(index)
                mp_errors = [
                    float(np.hypot(r.x - truth[r.name][0], r.y - truth[r.name][1])) / scale
                    for r in mp_readings
                    if truth[r.name][2]
                ]
                if not mp_errors:
                    continue

                rows.append(
                    {
                        "action": action,
                        "error": statistics.mean(mp_errors),
                        **gaps.features(),
                    }
                )

            print(f"  {name} {action:<14} {len(rows)} rows", flush=True)

    if len(rows) < 50:
        print("not enough frames")
        return 1

    errors = np.array([r["error"] for r in rows])
    truth = errors >= np.percentile(errors, 80)

    print()
    print(f"frames {len(rows)}   failures {truth.sum()} ({truth.mean():.1%})")
    print()
    print("=== can detector disagreement predict MediaPipe's error? ===")
    print(f"    (bar set in advance: AUROC >= {SUCCESS_AUROC})")
    print()

    best = 0.0
    results = []
    for feature in DISAGREEMENT_FEATURES:
        scores = np.array([r[feature] for r in rows])
        score = auroc(scores, truth)
        best = max(best, score)
        mark = "  <-- clears the bar" if score >= SUCCESS_AUROC else ""
        print(f"  {feature:<16} AUROC {score:>5.3f}{mark}")
        results.append({"feature": feature, "auroc": score})

    print()
    for action in sorted({r["action"] for r in rows}):
        subset = [r for r in rows if r["action"] == action]
        sub_err = np.array([r["error"] for r in subset])
        sub_truth = sub_err >= np.percentile(errors, 80)
        if sub_truth.sum() and (~sub_truth).sum():
            score = auroc(np.array([r["max_gap"] for r in subset]), sub_truth)
            print(f"  {action:<14} max_gap AUROC {score:>5.3f}  (n={len(subset)})")

    print()
    print(f"best AUROC {best:.3f}   previous best (single signal) 0.607")
    print("VERDICT:", "worth a second model" if best >= SUCCESS_AUROC else "does not clear the bar")

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "disagreement_results.json").write_text(
        json.dumps({"frames": len(rows), "results": results, "best": best}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
