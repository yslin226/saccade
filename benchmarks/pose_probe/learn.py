"""Can a classifier predict which pose estimates are wrong?

Following Schneider et al. (arXiv:2603.02881): rather than thresholding one
geometric quantity, feed several weak signals to a small model and let it
learn the weighting. They report 80.5% on rigid objects with depth; this asks
whether the idea survives on monocular human video, where the body has far
more freedom and there is no depth to fall back on.

The split is fixed here, in the source, before any result is seen:

    train on   baseball_pitch, bench_press, jumping_jacks, bowl
    test on    golf_swing, tennis_serve, squat, pushup

Different actions, not different clips of the same action — the earlier
attempt scored 2.78x when tuned and tested on the same actions and 0.64x when
moved to new ones. Splitting by clip would have hidden that.

Usage:
    uv run python -m benchmarks.pose_probe.learn --clips 15
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from benchmarks.pose_probe.continuity import JointReading
from benchmarks.pose_probe.failure_model import (
    FEATURE_NAMES,
    ClipReadings,
    FrameFeatures,
    extract_features,
    label_frames,
)
from benchmarks.pose_probe.pennaction import list_clips, load_clip

ROOT = Path("data/PennAction/Penn_Action")
CACHE = Path("benchmarks/pose_probe/cache")
POSE_MODEL = "models/pose_landmarker_heavy.task"

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

# Fixed before running. Changing these after seeing a result would turn the
# held-out set into a second training set.
TRAIN_ACTIONS = frozenset({"baseball_pitch", "bench_press", "jumping_jacks", "bowl"})
TEST_ACTIONS = frozenset({"golf_swing", "tennis_serve", "squat", "pushup"})


def collect(names: list[tuple[str, str]], landmarker: object) -> list[FrameFeatures]:
    """Run the detector over clips and build feature rows."""
    import cv2
    from mediapipe import Image, ImageFormat

    frames: list[FrameFeatures] = []
    for name, action in names:
        clip = load_clip(ROOT, name)
        height, width = clip.dimensions
        readings = ClipReadings(clip=name, action=action)

        for index in range(clip.n_frames):
            path = clip.frame_path(index)
            frame = cv2.imread(str(path)) if path.is_file() else None
            if frame is None:
                readings.readings.append(None)
                readings.errors.append(None)
                continue

            result = landmarker.detect(  # type: ignore[attr-defined]
                Image(
                    image_format=ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
            )
            if not result.pose_landmarks:
                readings.readings.append(None)
                readings.errors.append(None)
                continue

            marks = result.pose_landmarks[0]
            detected = [
                JointReading(
                    name=joint,
                    x=float(marks[i].x) * width,
                    y=float(marks[i].y) * height,
                    confidence=float(marks[i].visibility),
                )
                for joint, i in SHARED.items()
            ]

            truth = clip.joints(index)
            distances = [
                float(np.hypot(r.x - truth[r.name][0], r.y - truth[r.name][1]))
                for r in detected
                if truth[r.name][2]
            ]

            readings.readings.append(detected)
            readings.errors.append(statistics.mean(distances) if distances else None)

        frames.extend(extract_features(readings))
        print(f"  {name} {action:<16} {len(frames)} rows so far", flush=True)

    return frames


def report(name: str, scores: np.ndarray, truth: np.ndarray) -> dict:
    """Score a predictor, with the baselines it has to beat."""
    order = np.argsort(-scores)

    # AUROC by rank, which needs no threshold choice.
    positives, negatives = truth.sum(), (~truth).sum()
    if positives == 0 or negatives == 0:
        return {}
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    auroc = float(
        (negatives * positives - (ranks[truth].sum() - positives * (positives - 1) / 2))
        / (positives * negatives)
    )

    # Flagging the top 20%, matching the definition of a failure frame.
    cut = max(1, int(0.2 * len(scores)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[order[:cut]] = True
    precision = float((flagged & truth).sum() / max(flagged.sum(), 1))
    recall = float((flagged & truth).sum() / max(truth.sum(), 1))

    print(f"  {name:<22} AUROC {auroc:>5.3f}   precision {precision:>5.1%}   recall {recall:>5.1%}")
    return {"name": name, "auroc": auroc, "precision": precision, "recall": recall}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=12, help="clips per action")
    parser.add_argument("--pose-model", default=POSE_MODEL)
    args = parser.parse_args(argv)

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def pick(actions: frozenset[str]) -> list[tuple[str, str]]:
        by_action: dict[str, list[str]] = {}
        for name, action in list_clips(ROOT, actions):
            by_action.setdefault(action, []).append(name)
        return [
            (name, action)
            for action, names in sorted(by_action.items())
            for name in names[: args.clips]
        ]

    train_clips, test_clips = pick(TRAIN_ACTIONS), pick(TEST_ACTIONS)
    print(f"train: {len(train_clips)} clips, actions {sorted(TRAIN_ACTIONS)}")
    print(f"test : {len(test_clips)} clips, actions {sorted(TEST_ACTIONS)}")
    print()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.pose_model),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        print("training clips:")
        train = collect(train_clips, landmarker)
        print("held-out clips:")
        test = collect(test_clips, landmarker)

    if not train or not test:
        print("not enough data")
        return 1

    x_train = np.array([f.vector() for f in train])
    x_test = np.array([f.vector() for f in test])
    y_train = label_frames(train)
    y_test = label_frames(test)

    print()
    print(f"train frames {len(train)} ({y_train.mean():.1%} failures)")
    print(f"test frames  {len(test)} ({y_test.mean():.1%} failures)")
    print()

    scaler = StandardScaler().fit(x_train)
    results = []

    print("=== on the held-out actions ===")
    # Baselines first, so the learned models have something to beat.
    for i, name in enumerate(FEATURE_NAMES):
        if name in ("travel", "min_confidence"):
            direction = -1.0 if "confidence" in name else 1.0
            results.append(report(f"{name} alone", direction * x_test[:, i], y_test))

    logistic = LogisticRegression(max_iter=2000, class_weight="balanced")
    logistic.fit(scaler.transform(x_train), y_train)
    results.append(
        report(
            "logistic regression",
            logistic.predict_proba(scaler.transform(x_test))[:, 1],
            y_test,
        )
    )

    boosted = GradientBoostingClassifier(random_state=0)
    boosted.fit(x_train, y_train)
    results.append(report("gradient boosting", boosted.predict_proba(x_test)[:, 1], y_test))

    print()
    print("=== what the linear model leaned on ===")
    for name, weight in sorted(
        zip(FEATURE_NAMES, logistic.coef_[0], strict=True),
        key=lambda pair: -abs(pair[1]),
    ):
        print(f"  {name:<20} {weight:+.3f}")

    print()
    print("  AUROC 0.5 is chance. Precision/recall of 20% is chance, since")
    print("  failures are defined as the worst 20% of frames.")

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "failure_model_results.json").write_text(
        json.dumps(
            {
                "train_actions": sorted(TRAIN_ACTIONS),
                "test_actions": sorted(TEST_ACTIONS),
                "train_frames": len(train),
                "test_frames": len(test),
                "results": [r for r in results if r],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
