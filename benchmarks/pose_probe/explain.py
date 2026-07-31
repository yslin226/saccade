"""Can a VLM say why two detectors disagree?

Detector disagreement predicts which frames are wrong (AUROC 0.713), but not
what went wrong — and the causes need different handling:

    motion blur   the limb is there but smeared; discard and interpolate
    occlusion     the limb is hidden; the position may still be inferable
    lost track    the detector is following the wrong thing; reinitialise

The numbers are identical in all three cases. So this is the question the
agent exists to answer, and the first one in the project where a VLM has
something a measurement cannot supply.

The test is not whether the VLM produces a plausible label — it will always
produce one. It is whether the labels correspond to anything:

    1. Does it agree with itself on the same frame? (asked twice)
    2. Do its labels differ between fast actions, where blur dominates, and
       slow ones, where it should not?
    3. On frames where the detectors agree, does it decline to invent a
       problem?

The third is the sharpest. A model that reports blur on clean frames is
pattern-matching the question, not reading the image.

Usage:
    uv run python -m benchmarks.pose_probe.explain --frames 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image as PILImage

from benchmarks.blindtest.models import build_vlm
from benchmarks.pose_probe.continuity import JointReading, shoulder_width
from benchmarks.pose_probe.disagreement import (
    COMMON_JOINTS,
    SUSPECT_MEAN_GAP,
    disagreement_features,
    disagreement_tool,
)
from benchmarks.pose_probe.pennaction import FAST_ACTIONS, SLOW_ACTIONS, list_clips, load_clip
from saccade import ActiveVisionAgent, VLMError
from saccade.models import BBox
from saccade.vlm import FileCache

ROOT = Path("data/PennAction/Penn_Action")
CACHE = Path("benchmarks/pose_probe/cache")
DEFAULT_MODEL = "azure:gpt-5.4"

QUESTION = (
    "Two independent pose detectors placed this person's {joint} in different "
    "places — they disagree by {gap:.2f} body widths, which usually means at "
    "least one of them is wrong.\n\n"
    "Look at the region and say which of these explains it:\n"
    "  BLUR — the limb is smeared by motion and its position is ambiguous\n"
    "  OCCLUDED — the limb is hidden behind the body or another object\n"
    "  CLEAR — the limb is plainly visible and both detectors should agree\n\n"
    "Answer with one word: BLUR, OCCLUDED, or CLEAR."
)

VERDICTS = ("BLUR", "OCCLUDED", "CLEAR")


@dataclass
class Candidate:
    """A frame to ask about, with everything needed to judge the answer."""

    clip: str
    action: str
    group: str  # "fast" or "slow"
    frame: int
    image: PILImage.Image
    region: BBox
    worst_joint: str
    mean_gap: float
    true_error: float
    mp_readings: list[JointReading] = field(default_factory=list)
    yolo_readings: list[JointReading] = field(default_factory=list)

    @property
    def disagrees(self) -> bool:
        return self.mean_gap > SUSPECT_MEAN_GAP


def region_around(joint: JointReading, size: tuple[int, int], scale: float) -> BBox:
    """A box around a joint, sized relative to the body rather than the frame."""
    width, height = size
    pad = max(40, int(scale * 1.2))
    x = max(0, int(joint.x) - pad)
    y = max(0, int(joint.y) - pad)
    return BBox(
        x=x,
        y=y,
        w=max(1, min(width - x, pad * 2)),
        h=max(1, min(height - y, pad * 2)),
    )


def collect_candidates(
    clips: list[tuple[str, str]], *, per_group: int, pose_model: str
) -> list[Candidate]:
    """Find frames where the detectors disagree, and frames where they agree."""
    import cv2
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    from ultralytics import YOLO

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
    yolo = YOLO("yolo11n-pose.pt")

    found: list[Candidate] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        for name, action in clips:
            clip = load_clip(ROOT, name)
            height, width = clip.dimensions
            group = "fast" if action in FAST_ACTIONS else "slow"

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

                mp_readings = [
                    JointReading(
                        name=joint,
                        x=float(marks[mp_i].x) * width,
                        y=float(marks[mp_i].y) * height,
                        confidence=float(marks[mp_i].visibility),
                    )
                    for joint, (mp_i, _) in COMMON_JOINTS.items()
                ]
                yolo_readings = [
                    JointReading(
                        name=joint,
                        x=float(kp[yolo_i][0]),
                        y=float(kp[yolo_i][1]),
                        confidence=float(kp[yolo_i][2]),
                    )
                    for joint, (_, yolo_i) in COMMON_JOINTS.items()
                    if yolo_i < len(kp)
                ]

                scale = shoulder_width(mp_readings)
                if not scale:
                    continue

                gaps = disagreement_features(mp_readings, yolo_readings, scale=scale)
                if not gaps.per_joint or gaps.worst_joint is None:
                    continue

                truth = clip.joints(index)
                errors = [
                    float(np.hypot(r.x - truth[r.name][0], r.y - truth[r.name][1])) / scale
                    for r in mp_readings
                    if truth[r.name][2]
                ]
                if not errors:
                    continue

                worst = next(r for r in mp_readings if r.name == gaps.worst_joint)
                found.append(
                    Candidate(
                        clip=name,
                        action=action,
                        group=group,
                        frame=index,
                        image=PILImage.fromarray(rgb),
                        region=region_around(worst, (width, height), scale),
                        worst_joint=gaps.worst_joint,
                        mean_gap=gaps.features()["mean_gap"],
                        true_error=statistics.mean(errors),
                        mp_readings=mp_readings,
                        yolo_readings=yolo_readings,
                    )
                )
            print(f"  scanned {name} {action:<14} {len(found)} frames", flush=True)

    # Take the clearest disagreements and the clearest agreements from each
    # group: the contrast is what the experiment turns on.
    picked: list[Candidate] = []
    for group in ("fast", "slow"):
        rows = [c for c in found if c.group == group]
        rows.sort(key=lambda c: -c.mean_gap)
        picked.extend(rows[:per_group])
        picked.extend(rows[-per_group:])
    return picked


async def ask(candidate: Candidate, vlm: object, cache: FileCache, *, repeat: int = 2) -> list[str]:
    """Ask the agent to explain a frame, more than once."""
    view = candidate.image.crop(
        (
            candidate.region.x,
            candidate.region.y,
            candidate.region.right,
            candidate.region.bottom,
        )
    )
    question = QUESTION.format(joint=candidate.worst_joint, gap=candidate.mean_gap)

    answers: list[str] = []
    for attempt in range(repeat):
        agent = ActiveVisionAgent(vlm, cache=cache, max_steps=1)  # type: ignore[arg-type]
        agent.register_tool(disagreement_tool(candidate.mp_readings, candidate.yolo_readings))
        # A different phrasing on the repeat, so the cache cannot answer it.
        suffix = "" if attempt == 0 else "\n\nConsider the image again."

        try:
            result = await agent.investigate_async(view, question + suffix)
        except VLMError as exc:
            # Azure's content filter rejects some crops of people mid-motion.
            # A refusal is not a verdict, and it must neither end the run nor
            # be counted as an answer.
            answers.append("REFUSED" if "content_policy" in str(exc) else "ERROR")
            continue

        text = result.answer.strip().upper()
        answers.append(next((v for v in VERDICTS if v in text), "UNREADABLE"))
    return answers


async def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=3, help="clips per action")
    parser.add_argument("--frames", type=int, default=8, help="frames per group per side")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pose-model", default="models/pose_landmarker_heavy.task")
    args = parser.parse_args(argv)

    actions = FAST_ACTIONS | SLOW_ACTIONS
    by_action: dict[str, list[str]] = {}
    for name, action in list_clips(ROOT, actions):
        by_action.setdefault(action, []).append(name)
    clips = [
        (name, action)
        for action, names in sorted(by_action.items())
        for name in names[: args.clips]
    ]

    print(f"scanning {len(clips)} clips for disagreements...")
    candidates = collect_candidates(clips, per_group=args.frames, pose_model=args.pose_model)
    if not candidates:
        print("no candidates")
        return 1

    disputed = [c for c in candidates if c.disagrees]
    agreed = [c for c in candidates if not c.disagrees]
    print()
    print(f"asking about {len(candidates)} frames ({len(disputed)} disputed, {len(agreed)} agreed)")
    print()

    vlm = build_vlm(args.model)
    cache = FileCache(str(CACHE / "vlm_explain"))

    findings = []
    for candidate in candidates:
        answers = await ask(candidate, vlm, cache)
        findings.append(
            {
                "clip": candidate.clip,
                "action": candidate.action,
                "group": candidate.group,
                "frame": candidate.frame,
                "joint": candidate.worst_joint,
                "mean_gap": round(candidate.mean_gap, 3),
                "disagrees": candidate.disagrees,
                "true_error": round(candidate.true_error, 3),
                "answers": answers,
                "consistent": len(set(answers)) == 1,
            }
        )
        state = "DISPUTED" if candidate.disagrees else "agreed  "
        print(
            f"  {state} {candidate.action:<14} gap {candidate.mean_gap:>5.2f}  "
            f"{candidate.worst_joint:<12} -> {answers}"
        )

    print()
    print("=== 1. does it agree with itself? ===")
    consistent = sum(1 for f in findings if f["consistent"])
    print(
        f"  {consistent}/{len(findings)} frames got the same answer twice "
        f"({consistent / len(findings):.0%})"
    )

    print()
    print("=== 2. does the explanation depend on the movement? ===")
    for group in ("fast", "slow"):
        rows = [f for f in findings if f["group"] == group and f["disagrees"]]
        if rows:
            counts = Counter(f["answers"][0] for f in rows)
            print(f"  {group:<5} n={len(rows):<3} {dict(counts)}")

    print()
    print("=== 3. does it invent problems on frames the detectors agree on? ===")
    for state, rows in (
        ("disputed", [f for f in findings if f["disagrees"]]),
        ("agreed", [f for f in findings if not f["disagrees"]]),
    ):
        if rows:
            counts = Counter(f["answers"][0] for f in rows)
            clear = counts.get("CLEAR", 0)
            print(
                f"  {state:<9} n={len(rows):<3} said CLEAR {clear}/{len(rows)} "
                f"({clear / len(rows):.0%})   {dict(counts)}"
            )

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "explanation_results.json"
    out.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print()
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
