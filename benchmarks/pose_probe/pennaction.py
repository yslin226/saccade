"""Reading Penn Action, which supplies the thing a single clip cannot: ground
truth, and a control group.

2326 YouTube clips across 15 actions, every frame hand-labelled with 13 joint
positions and a visibility flag. Two properties make it the right test:

- The labels are human judgements, so a detector's output can be checked
  against what is actually there rather than against its own confidence.
- The actions split into fast movements that smear (a pitch, a golf swing)
  and slow ones that should not (squats, push-ups). If the agent flags
  problems in the slow group too, it is not detecting anything real.

Annotations are MATLAB .mat files, read with scipy.io as data — that parser
does not execute code, unlike pickle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

__all__ = [
    "FAST_ACTIONS",
    "SLOW_ACTIONS",
    "Clip",
    "list_clips",
    "load_clip",
]

# Penn Action's joint order, from its README.
JOINT_NAMES = (
    "head",
    "L shoulder",
    "R shoulder",
    "L elbow",
    "R elbow",
    "L wrist",
    "R wrist",
    "L hip",
    "R hip",
    "L knee",
    "R knee",
    "L ankle",
    "R ankle",
)

# Movements fast enough to smear a limb at ordinary frame rates. If active
# vision helps anywhere, it should help here.
FAST_ACTIONS = frozenset(
    {"baseball_pitch", "baseball_swing", "golf_swing", "tennis_serve", "tennis_forehand"}
)

# The control. A squat has no limb moving fast enough to blur, so a detector
# should hold up and the agent should find nothing to report. Flagging these
# at the same rate as the fast group would mean the flags are noise.
#
# Names taken from the dataset itself rather than guessed — it uses the
# singular "squat", "pushup", "situp", "pullup".
SLOW_ACTIONS = frozenset({"squat", "pushup", "situp", "bench_press", "pullup"})


@dataclass
class Clip:
    """One annotated video, as frames on disk plus hand-labelled joints."""

    name: str
    action: str
    frame_dir: Path
    x: np.ndarray  # (frames, 13)
    y: np.ndarray  # (frames, 13)
    visible: np.ndarray  # (frames, 13) bool
    dimensions: tuple[int, int]  # height, width

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[0])

    def frame_path(self, index: int) -> Path:
        """Penn Action names frames 000001.jpg upwards."""
        return self.frame_dir / f"{index + 1:06d}.jpg"

    def joints(self, index: int) -> dict[str, tuple[float, float, bool]]:
        """Hand-labelled joints for one frame: name -> (x, y, visible)."""
        return {
            name: (float(self.x[index, j]), float(self.y[index, j]), bool(self.visible[index, j]))
            for j, name in enumerate(JOINT_NAMES)
        }


def load_clip(root: Path, name: str) -> Clip:
    """Read one clip's annotation.

    Args:
        root: The ``Penn_Action`` directory.
        name: Clip id, e.g. ``"0001"``.
    """
    raw = loadmat(str(root / "labels" / f"{name}.mat"), simplify_cells=True)

    dimensions = raw["dimensions"]
    height, width = int(dimensions[0]), int(dimensions[1])

    return Clip(
        name=name,
        action=str(raw["action"]),
        frame_dir=root / "frames" / name,
        x=np.atleast_2d(raw["x"]),
        y=np.atleast_2d(raw["y"]),
        visible=np.atleast_2d(raw["visibility"]).astype(bool),
        dimensions=(height, width),
    )


def list_clips(root: Path, actions: frozenset[str] | None = None) -> list[tuple[str, str]]:
    """Every clip id and its action, optionally filtered.

    Reads only the action field, so scanning all 2326 labels is quick.
    """
    found: list[tuple[str, str]] = []
    for path in sorted((root / "labels").glob("*.mat")):
        raw = loadmat(str(path), simplify_cells=True, variable_names=["action"])
        action = str(raw["action"])
        if actions is None or action in actions:
            found.append((path.stem, action))
    return found
