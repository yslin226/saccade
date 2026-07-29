"""Saccade — an active vision agent for VLMs.

Vision language models answer in one glance. Human eyes do not: they saccade
three to four times a second, sampling a scene from different positions until
the picture holds together. This library gives a VLM that loop at inference
time, with computed measurements acting as referee over what the model claims
to see.

Names exported here are the public contract. Modules starting with an
underscore are internal and may change in any release.
"""

from __future__ import annotations

from saccade.agent import ActiveVisionAgent
from saccade.exceptions import SaccadeError, ToolError, VLMError
from saccade.models import (
    BBox,
    EvidenceStep,
    InvestigationResult,
    Observation,
    Verification,
    Viewport,
    VLMResponse,
)
from saccade.ports import CachePort, VLMPort
from saccade.tools import Tool, ToolResult

__version__ = "0.1.0"

__all__ = [
    "ActiveVisionAgent",
    "BBox",
    "CachePort",
    "EvidenceStep",
    "InvestigationResult",
    "Observation",
    "SaccadeError",
    "Tool",
    "ToolError",
    "ToolResult",
    "VLMError",
    "VLMPort",
    "VLMResponse",
    "Verification",
    "Viewport",
]
