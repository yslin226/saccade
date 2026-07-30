"""Can an agent say why a pose reading is impossible?

The question BlindTest could not answer. There the tool was right 99% of the
time and the loop was overhead. Here the tool can prove a frame is wrong but
cannot say which of three things went wrong, and they need different handling:

  motion blur  -> discard the frame, interpolate from neighbours
  occlusion    -> the joint is hidden but its position may be inferable
  lost track   -> reinitialise the detector

If a VLM shown the magnified region can tell these apart, the agent earns its
keep. If it cannot, the honest answer is a threshold — discard anything over
150px and skip the model entirely.

Usage:
    uv run python -m benchmarks.pose_probe.run --video happy/林永閎.MOV
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from benchmarks.blindtest.models import build_vlm
from benchmarks.pose_probe.continuity import JointReading, continuity_tool
from benchmarks.pose_probe.extract import (
    FramePose,
    extract_poses,
    load_poses,
    save_poses,
    suspect_frames,
)
from saccade import ActiveVisionAgent
from saccade.models import BBox
from saccade.vlm import FileCache

DEFAULT_MODEL = "azure:gpt-5.4"
CACHE = Path("benchmarks/pose_probe/cache")

QUESTION = (
    "A pose detector placed a joint here, but the position is physically "
    "impossible — it moved further in one frame than a limb can move.\n\n"
    "Look at this region and say which of these explains it:\n"
    "  BLUR — the limb is there but smeared by motion\n"
    "  OCCLUDED — the limb is hidden behind something\n"
    "  ABSENT — the limb is outside this region entirely\n\n"
    "Answer with one word: BLUR, OCCLUDED, or ABSENT."
)


def region_around(joint: JointReading, size: tuple[int, int], pad: int = 220) -> BBox:
    """A box around a joint, clamped to the frame.

    Generous padding on purpose: the detector's position is wrong, so the
    limb is somewhere near it rather than at it.
    """
    width, height = size
    x = max(0, int(joint.x) - pad)
    y = max(0, int(joint.y) - pad)
    w = min(width - x, pad * 2)
    h = min(height - y, pad * 2)
    return BBox(x=x, y=y, w=max(1, w), h=max(1, h))


def frame_image(video: str, index: int) -> Image.Image:
    """Read one frame as a PIL image."""
    import cv2

    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


async def investigate_frame(
    vlm: object,
    cache: FileCache,
    video: str,
    previous: FramePose,
    current: FramePose,
    implausible: list,
) -> dict:
    """Ask the agent why one frame's pose is impossible.

    A fresh agent per frame: the continuity tool is built from this frame's
    readings and the previous one's, so it cannot be shared across frames.
    """
    image = frame_image(video, current.frame)
    worst = implausible[0]

    joint = next(j for j in current.joints if j.name == worst.name)
    region = region_around(joint, image.size)

    agent = ActiveVisionAgent(vlm, cache=cache, max_steps=2)  # type: ignore[arg-type]
    agent.register_tool(continuity_tool(previous.joints, current.joints))

    view = image.crop((region.x, region.y, region.right, region.bottom))
    result = await agent.investigate_async(view, QUESTION)

    return {
        "frame": current.frame,
        "joint": worst.name,
        "travel_px": round(worst.travel_px, 1),
        "detector_confidence": round(worst.reported_confidence, 3),
        "region": [region.x, region.y, region.w, region.h],
        "verdict": result.answer.strip()[:80],
        "agent_confidence": round(result.confidence, 2),
        "converged": result.converged,
        "steps": len(result.evidence_chain),
    }


async def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pose-model", default="models/pose_landmarker_heavy.task")
    parser.add_argument("--limit", type=float, default=None, help="px travel threshold")
    args = parser.parse_args(argv)

    CACHE.mkdir(parents=True, exist_ok=True)
    pose_cache = CACHE / (Path(args.video).stem + ".poses.json")

    if pose_cache.is_file():
        poses = load_poses(pose_cache)
        print(f"loaded {len(poses)} cached frames")
    else:
        print("running pose detection over the video...")
        poses = extract_poses(args.video, args.pose_model)
        save_poses(poses, pose_cache)
        print(f"detected {len(poses)} frames, cached")

    missing = sum(1 for p in poses if not p.detected)
    suspects = suspect_frames(poses, limit=args.limit)

    print()
    print(f"frames                     : {len(poses)}")
    print(f"detector reported no pose  : {missing}")
    print(f"physically impossible poses: {len(suspects)}")

    if not suspects:
        print("\nNothing to investigate — the detector held up on this clip.")
        return 0

    unflagged = sum(1 for _, _, bad in suspects if all(b.reported_confidence >= 0.5 for b in bad))
    print(f"  of those, {unflagged} came with the detector reporting confidence >= 0.5")
    print(f"  -> {unflagged / len(suspects):.0%} of its errors were not self-reported")
    print()

    vlm = build_vlm(args.model)
    cache = FileCache(str(CACHE / "vlm"))

    findings = []
    for previous, current, bad in suspects:
        finding = await investigate_frame(vlm, cache, args.video, previous, current, bad)
        findings.append(finding)
        print(
            f"  frame {finding['frame']:>3}  {finding['joint']:<12} "
            f"{finding['travel_px']:>6.0f}px  detector said "
            f"{finding['detector_confidence']:.2f}  ->  {finding['verdict']!r}"
        )

    out = CACHE / f"findings_{Path(args.video).stem}.json"
    out.write_text(json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
