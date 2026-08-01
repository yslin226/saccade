"""The flow: what happens in what order, and what it needs to be given.

Use cases orchestrate domain objects. They must not contain domain knowledge
of their own (spec 5.1) — a use case that computed an angle would have put a
rule somewhere the domain tests do not reach.

Everything external arrives as a Port. That is what makes a use case
testable without a video file, a detector, or a disk.
"""

from __future__ import annotations

from sandlot.application.ports import (
    DecodedVideo,
    Detection,
    DetectPort,
    PosePort,
    SessionRepoPort,
    VideoPort,
)
from sandlot.application.use_cases import (
    AnalysisFailedError,
    SessionNotFoundError,
    analyze_pitch,
    analyze_swing,
    bat_boxes,
    compare_sessions,
    compare_with_previous,
)

__all__ = [
    "AnalysisFailedError",
    "DecodedVideo",
    "DetectPort",
    "Detection",
    "PosePort",
    "SessionNotFoundError",
    "SessionRepoPort",
    "VideoPort",
    "analyze_pitch",
    "analyze_swing",
    "bat_boxes",
    "compare_sessions",
    "compare_with_previous",
]
