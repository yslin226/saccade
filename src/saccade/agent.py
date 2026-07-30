"""The agent entry point — the Perceive-Verify loop.

Plan where to look, act on that decision, observe the result, confront the
observation with measurement, record what happened. Repeat until confidence
is earned or the budget runs out.

The part that is not a conventional ReAct agent is step four. Elsewhere a
tool extends the model and its output is believed; here a tool referees the
model, and only computed results may overrule what it said.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from PIL.Image import Image

from saccade._evidence import EvidenceChain
from saccade._observer import Observer
from saccade._planner import Action, Planner
from saccade._verifier import adjust_confidence, verify
from saccade.actions import crop, zoom
from saccade.exceptions import ToolError
from saccade.models import EvidenceStep, InvestigationResult, Viewport
from saccade.ports import CachePort, VLMPort
from saccade.tools import Tool, ToolResult

__all__ = ["ActiveVisionAgent"]

logger = logging.getLogger("saccade")


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

    Example:
        >>> from saccade.vlm import FakeVLM
        >>> agent = ActiveVisionAgent(FakeVLM(["two circles overlap"], exhausted="repeat_last"))
        >>> agent.max_steps
        8
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
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {max_steps}")
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be in (0, 1], got {confidence_threshold}")

        self._vlm = _resolve_vlm(vlm)
        self._cache = cache
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self._tools: dict[str, Tool] = {tool.name: tool for tool in (tools or [])}
        self._on_step = on_step

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def register_tool(self, tool: Tool) -> None:
        """Add a domain tool.

        This is the only way domain capability enters the engine: Saccade
        never imports MediaPipe, YOLO or any other domain package itself
        (rule 2 in CLAUDE.md).
        """
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    async def investigate_async(
        self,
        image: Image,
        question: str,
        *,
        expect: type[Any] | None = None,
    ) -> InvestigationResult:
        """Run the Perceive-Verify loop until confident or out of steps."""
        observer = Observer(self._vlm, self._cache)
        planner = Planner(image.size, max_steps=self.max_steps)
        chain = EvidenceChain()
        confidence = 0.0
        structured: Any | None = None
        structured_verified = False

        for _ in range(self.max_steps):
            planned = planner.plan_next(chain.steps, confidence)
            if planned.action is Action.STOP:
                logger.debug("stopping: %s", planned.reason)
                break

            view = _apply(image, planned.action, planned.viewport, planned.zoom_factor)
            planner.record(planned.viewport)

            observation, response = await observer.observe([view], question, output_type=expect)

            results = self._run_tools(view, planned.viewport)
            verification = verify(observation, results)
            confidence = adjust_confidence(confidence, verification)

            # Keep the first verified structured answer. Once a measurement
            # has backed a value, later steps are narrower crops rather than
            # better views, and letting them overwrite it discards the one
            # piece of output the tools actually stood behind.
            if response.structured is not None and (structured is None or not structured_verified):
                structured = response.structured
                structured_verified = verification.passed

            step = chain.add(
                action_name=planned.action.value,
                viewport=planned.viewport,
                observation=observation,
                verification=verification,
                reason=planned.reason,
            )
            if self._on_step is not None:
                self._on_step(step)

            if confidence >= self.confidence_threshold:
                logger.debug("converged at %.2f after %d step(s)", confidence, len(chain))
                break

        converged = confidence >= self.confidence_threshold

        return InvestigationResult(
            answer=chain.best_statement() or "no observation could be made",
            confidence=confidence,
            converged=converged,
            evidence_chain=chain.steps,
            total_tokens=observer.tokens_used,
            structured=structured,
        )

    def investigate(
        self,
        image: Image,
        question: str,
        *,
        expect: type[Any] | None = None,
    ) -> InvestigationResult:
        """Synchronous wrapper around :meth:`investigate_async`.

        Raises:
            RuntimeError: If called from inside a running event loop. Nesting
                loops is not possible, and the caller should await
                :meth:`investigate_async` instead — saying so beats
                deadlocking.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.investigate_async(image, question, expect=expect))

        raise RuntimeError(
            "investigate() cannot be called from inside a running event loop; "
            "await investigate_async() instead"
        )

    def _run_tools(self, view: Image, viewport: Viewport) -> list[ToolResult]:
        """Run every registered tool over the current view.

        Tools are invoked through ``tool.fn`` rather than ``tool(...)``: the
        latter validates arguments through ``params_schema``, and a Pydantic
        model would drop the image and viewport as unknown fields. Schema
        validation is for arguments a model chooses; these two are context
        the loop supplies.

        A tool that raises is recorded and skipped rather than aborting the
        investigation: one broken detector should not discard the evidence
        already gathered.
        """
        results: list[ToolResult] = []
        for tool in self._tools.values():
            try:
                results.append(tool.fn(image=view, viewport=viewport))
            except ToolError:
                logger.warning("tool %s failed", tool.name, exc_info=True)
            except Exception:
                logger.warning("tool %s raised unexpectedly", tool.name, exc_info=True)
        return results


def _apply(image: Image, action: Action, viewport: Viewport, factor: float) -> Image:
    """Carry out a planned visual action."""
    if action is Action.ZOOM:
        view, _ = zoom(image, viewport.bbox, factor)
        return view
    view, _ = crop(image, viewport.bbox)
    return view


def _resolve_vlm(vlm: VLMPort | str) -> VLMPort:
    """Accept either a port implementation or a model string."""
    if isinstance(vlm, str):
        from saccade.vlm.pydantic_ai import PydanticAIVLM

        return PydanticAIVLM(vlm)
    return vlm
