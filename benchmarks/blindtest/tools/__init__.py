"""Measurement tools for BlindTest, registered into the agent.

These live in benchmarks/ rather than src/ because rule 2 forbids the engine
from importing domain detectors. OpenCV circle and line detection is domain
capability; it reaches the agent through register_tool(), which is the only
door the architecture provides.
"""

from __future__ import annotations

from benchmarks.blindtest.tools.circles import circle_tool, detect_circles
from benchmarks.blindtest.tools.decoys import DECOY_NAMES, decoy_tools
from benchmarks.blindtest.tools.lines import line_tool, trace_curves

__all__ = [
    "DECOY_NAMES",
    "circle_tool",
    "decoy_tools",
    "detect_circles",
    "line_tool",
    "trace_curves",
]
