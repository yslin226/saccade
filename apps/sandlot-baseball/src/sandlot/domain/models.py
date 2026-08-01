"""What a measurement is, and what it has to carry with it.

Rule 8: nothing leaves this system without numbers, a frame, and a source.
That is enforced by the shape of :class:`Metric` — there is no way to build
one without saying which frames produced it.

Everything here is frozen. A metric that could be edited after the fact is a
metric whose evidence chain no longer describes it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Frame",
    "JointReading",
    "Metric",
    "MetricDelta",
    "Session",
    "Toolchain",
]

# The joints both detectors report and every metric here is built from.
# MediaPipe emits 33 landmarks and COCO 17; this is the intersection that
# matters for pitching and hitting mechanics.
JOINT_NAMES = (
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


class JointReading(BaseModel):
    """One joint's position in one frame, as a detector reported it.

    ``confidence`` is the detector's own estimate and is recorded, not
    trusted: on this project's own data MediaPipe reported 0.79 to 0.99 for four
    physically impossible readings, and its confidence predicts error at
    AUROC 0.358 — worse than chance. Use it to exclude guesses, never as
    evidence a reading is right.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)


class Frame(BaseModel):
    """Every joint found in one frame of video.

    ``index`` is the frame number in the source file and ``timestamp`` is its
    position in seconds. Both are kept: the index is what an auditor uses to
    find the frame again, the timestamp is what a rate is computed against.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    joints: tuple[JointReading, ...] = ()

    def joint(self, name: str) -> JointReading | None:
        """The named reading, or None when the detector did not report it."""
        return next((j for j in self.joints if j.name == name), None)

    def position(self, name: str) -> tuple[float, float] | None:
        reading = self.joint(name)
        return None if reading is None else (reading.x, reading.y)


class Metric(BaseModel):
    """One measured quantity, with what it was measured from.

    ``frames`` is not optional. A number without the frames behind it cannot
    be checked, and rule 8 says an unbacked claim does not get to leave. The
    validator refuses rather than defaulting to an empty list, because an
    empty evidence chain that passes silently is exactly the failure the rule
    is written against.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    unit: str
    frames: tuple[int, ...]
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _must_cite_a_frame(self) -> Metric:
        if not self.frames:
            raise ValueError(f"metric {self.name!r} cites no frame; rule 8 requires evidence")
        return self


class Toolchain(BaseModel):
    """The versions that produced a session.

    Recorded because they are part of the answer. The detectors are pinned
    exactly and ten runs agree bitwise, but an upgrade can still move a
    coordinate in the last decimal place — which is invisible in a changelog
    and fatal to "you changed by 4.8 degrees since last time". Comparing two
    sessions built by different toolchains is refused rather than reported.
    """

    model_config = ConfigDict(frozen=True)

    mediapipe: str
    ultralytics: str
    sandlot: str


class Session(BaseModel):
    """One analysis of one video.

    ``video_sha256`` identifies the source rather than naming it: two
    sessions of the same file are comparable however the file was named or
    moved, and re-encoding it makes them incomparable, which is correct.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    video_sha256: str
    frame_count: int = Field(ge=0)
    fps: float = Field(gt=0)
    toolchain: Toolchain
    metrics: tuple[Metric, ...] = ()

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)


class MetricDelta(BaseModel):
    """How one metric moved between two sessions.

    ``before`` or ``after`` may be None when a metric could be computed in
    one session and not the other — an occluded ankle, a swing that left the
    frame. That is reported as a gap rather than as a change of zero, which
    would read as "nothing moved" when the truth is "nobody looked".
    """

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    before: float | None
    after: float | None

    @property
    def change(self) -> float | None:
        """After minus before, or None when either side is missing."""
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def comparable(self) -> bool:
        return self.before is not None and self.after is not None
