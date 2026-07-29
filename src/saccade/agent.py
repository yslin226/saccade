"""The agent entry point.

M0 fixes the signature; the Perceive-Verify loop lands in M1. The shape is
settled now because it is the part users write against, and every parameter
after ``vlm`` is keyword-only so M1 can add to it without breaking callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL.Image import Image

from saccade.models import EvidenceStep, InvestigationResult
from saccade.ports import CachePort, VLMPort
from saccade.tools import Tool

__all__ = ["ActiveVisionAgent"]


class ActiveVisionAgent:
    """Investigates an image by deciding where to look next.

    Args:
        vlm: A :class:`~saccade.ports.VLMPort`, or a Pydantic AI model string
            such as ``"google:gemini-2.5-flash"``.
        cache: Response cache. Strongly recommended — see rule 6 in CLAUDE.md.
        max_steps: Hard ceiling on loop iterations.
        confidence_threshold: Stop once verified confidence reaches this.
        tools: Extra tools, beyond the built-in visual actions.
        on_step: Called after each step, for tracing or progress display.
    """

    def __init__(
        self,
        vlm: VLMPort | str,
        *,
        cache: CachePort | None = None,
        max_steps: int = 8,
        confidence_threshold: float = 0.8,
        tools: list[Tool] | None = None,
        on_step: Callable[[EvidenceStep], None] | None = None,
    ) -> None:
        raise NotImplementedError("ActiveVisionAgent lands in M1")

    async def investigate_async(
        self,
        image: Image,
        question: str,
        *,
        expect: type[Any] | None = None,
    ) -> InvestigationResult:
        """Run the Perceive-Verify loop until confident or out of steps."""
        raise NotImplementedError("ActiveVisionAgent lands in M1")

    def investigate(
        self,
        image: Image,
        question: str,
        *,
        expect: type[Any] | None = None,
    ) -> InvestigationResult:
        """Synchronous wrapper around :meth:`investigate_async`."""
        raise NotImplementedError("ActiveVisionAgent lands in M1")

    def register_tool(self, tool: Tool) -> None:
        """Add a domain tool.

        This is the only way domain capability enters the engine: Saccade
        never imports MediaPipe, YOLO or any other domain package itself
        (rule 2 in CLAUDE.md).
        """
        raise NotImplementedError("ActiveVisionAgent lands in M1")
