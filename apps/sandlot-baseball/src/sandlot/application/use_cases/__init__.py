"""One file per use case. Each is a flow, not a rule.

``_body`` holds what a pitch and a swing both measure, imported by both
rather than copied — two files computing the same angle drift apart the first
time one of them is corrected.

``measure`` is deliberately not re-exported: both use cases define one, and a
single name here would have to pick a winner. Import from the specific module
when you want to measure pre-detected frames.
"""

from __future__ import annotations

from sandlot.application.use_cases.analyze_pitch import (
    AnalysisFailedError,
    analyze_pitch,
    new_session_id,
)
from sandlot.application.use_cases.analyze_swing import analyze_swing, bat_boxes
from sandlot.application.use_cases.compare_sessions import (
    SessionNotFoundError,
    compare_sessions,
    compare_with_previous,
)

__all__ = [
    "AnalysisFailedError",
    "SessionNotFoundError",
    "analyze_pitch",
    "analyze_swing",
    "bat_boxes",
    "compare_sessions",
    "compare_with_previous",
    "new_session_id",
]
