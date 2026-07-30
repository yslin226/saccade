"""End-to-end tests for the Perceive-Verify loop.

Everything here runs on FakeVLM: no API key, no network, no cost. That is
deliberate — a loop whose tests need credentials cannot be run in CI, and a
benchmark whose numbers depend on a live model cannot be reproduced.
"""

from __future__ import annotations

import asyncio

import pytest
from PIL import Image
from pydantic import BaseModel

from saccade import ActiveVisionAgent, EvidenceStep, InvestigationResult, Tool, ToolResult
from saccade.vlm import FakeVLM, MemoryCache


class Answer(BaseModel):
    overlapping: bool


def an_image(size: tuple[int, int] = (100, 100)) -> Image.Image:
    return Image.new("RGB", size, color="white")


def measuring_tool(value: object, *, name: str = "measure") -> Tool:
    """A tool that always reports the same measurement."""

    def run(image: object, viewport: object) -> ToolResult:
        return ToolResult(value=value, is_measurement=True)

    return Tool(name=name, description="a measurement", fn=run, params_schema=Answer)


def describing_tool(value: object) -> Tool:
    """A tool whose output is an opinion, not a measurement."""

    def run(image: object, viewport: object) -> ToolResult:
        return ToolResult(value=value, is_measurement=False)

    return Tool(name="describe", description="an opinion", fn=run, params_schema=Answer)


class TestBasicLoop:
    async def test_returns_a_result(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["two circles"], exhausted="repeat_last"))
        result = await agent.investigate_async(an_image(), "how many circles?")
        assert isinstance(result, InvestigationResult)
        assert result.answer

    async def test_evidence_chain_is_populated(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["a circle"], exhausted="repeat_last"))
        result = await agent.investigate_async(an_image(), "q")
        assert len(result.evidence_chain) >= 1
        assert result.evidence_chain[0].index == 0

    async def test_every_step_records_a_viewport_in_source_coordinates(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"))
        result = await agent.investigate_async(an_image((120, 80)), "q")
        for step in result.evidence_chain:
            assert step.viewport.source_size == (120, 80)

    async def test_step_indices_are_sequential(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"), max_steps=4)
        result = await agent.investigate_async(an_image(), "q")
        assert [s.index for s in result.evidence_chain] == list(range(len(result.evidence_chain)))

    async def test_the_first_look_is_the_whole_image(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"))
        result = await agent.investigate_async(an_image(), "q")
        assert result.evidence_chain[0].viewport.covers_full_image is True


class TestConvergence:
    async def test_agreement_with_measurements_converges(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM(["the circles overlap"], exhausted="repeat_last"),
            tools=[measuring_tool({"overlap": True})],
            confidence_threshold=0.5,
        )
        result = await agent.investigate_async(an_image(), "do they overlap?")
        assert result.converged is True
        assert result.confidence >= 0.5

    async def test_unverified_runs_do_not_converge_at_a_high_threshold(self) -> None:
        """Without measurement, confidence is capped below the threshold."""
        agent = ActiveVisionAgent(
            FakeVLM(["looks like two circles"], exhausted="repeat_last"),
            confidence_threshold=0.8,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert result.converged is False
        assert result.confidence <= 0.6

    async def test_failing_to_converge_is_not_an_error(self) -> None:
        """Spec 4.4: running out of steps is an ordinary outcome."""
        agent = ActiveVisionAgent(
            FakeVLM(["unclear"], exhausted="repeat_last"),
            max_steps=3,
            confidence_threshold=0.99,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert result.converged is False
        assert len(result.evidence_chain) >= 1, "evidence is kept even when unconverged"

    async def test_the_step_budget_is_respected(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM(["x"], exhausted="repeat_last"),
            max_steps=2,
            confidence_threshold=0.99,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert len(result.evidence_chain) <= 2

    async def test_converging_early_stops_the_loop(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM(["the circles overlap"], exhausted="repeat_last"),
            tools=[measuring_tool({"overlap": True})],
            confidence_threshold=0.2,
            max_steps=8,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert len(result.evidence_chain) == 1, "should stop as soon as it is confident"


class TestVerificationDrivesTheLoop:
    async def test_a_conflict_lowers_confidence(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM(["there are 3 people"], exhausted="repeat_last"),
            tools=[measuring_tool({"people": 2})],
            max_steps=2,
        )
        result = await agent.investigate_async(an_image(), "how many people?")
        conflicts = [s for s in result.evidence_chain if s.verification and s.verification.conflict]
        assert conflicts, "the measurement contradicted the model and should be recorded"
        assert result.confidence < 0.5

    async def test_a_conflict_redirects_the_next_look(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM(["there are 3 people"], exhausted="repeat_last"),
            tools=[measuring_tool({"people": 2})],
            max_steps=2,
            confidence_threshold=0.99,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert len(result.evidence_chain) == 2
        assert result.evidence_chain[1].viewport.bbox != result.evidence_chain[0].viewport.bbox

    async def test_opinions_cannot_drive_convergence(self) -> None:
        """The core claim: a non-measurement result may not confirm anything."""
        agent = ActiveVisionAgent(
            FakeVLM(["the circles overlap"], exhausted="repeat_last"),
            tools=[describing_tool({"overlap": True})],
            confidence_threshold=0.8,
            max_steps=5,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert result.converged is False
        assert result.confidence <= 0.6


class TestTools:
    async def test_registered_tools_are_called(self) -> None:
        calls: list[int] = []

        def run(image: object, viewport: object) -> ToolResult:
            calls.append(1)
            return ToolResult(value={"n": 1}, is_measurement=True)

        agent = ActiveVisionAgent(
            FakeVLM(["x"], exhausted="repeat_last"),
            tools=[Tool(name="counter", description="d", fn=run, params_schema=Answer)],
            max_steps=1,
        )
        await agent.investigate_async(an_image(), "q")
        assert calls

    async def test_register_tool_adds_a_tool(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"))
        agent.register_tool(measuring_tool({"a": 1}))
        assert len(agent.tools) == 1

    async def test_duplicate_tool_names_are_rejected(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"))
        agent.register_tool(measuring_tool({"a": 1}, name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            agent.register_tool(measuring_tool({"b": 2}, name="dup"))

    async def test_a_broken_tool_does_not_abort_the_investigation(self) -> None:
        """One failing detector should not discard the evidence gathered."""

        def explode(image: object, viewport: object) -> ToolResult:
            raise RuntimeError("detector crashed")

        agent = ActiveVisionAgent(
            FakeVLM(["x"], exhausted="repeat_last"),
            tools=[Tool(name="broken", description="d", fn=explode, params_schema=Answer)],
            max_steps=2,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert len(result.evidence_chain) >= 1


class TestCallbackAndAccounting:
    async def test_on_step_fires_for_every_step(self) -> None:
        seen: list[EvidenceStep] = []
        agent = ActiveVisionAgent(
            FakeVLM(["x"], exhausted="repeat_last"),
            max_steps=3,
            confidence_threshold=0.99,
            on_step=seen.append,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert len(seen) == len(result.evidence_chain)

    async def test_tokens_are_reported(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"), max_steps=2)
        result = await agent.investigate_async(an_image(), "q")
        assert result.total_tokens >= 0

    async def test_the_cache_is_used_across_investigations(self) -> None:
        cache = MemoryCache()
        vlm = FakeVLM(["consistent answer"], exhausted="repeat_last")

        agent = ActiveVisionAgent(vlm, cache=cache, max_steps=1)
        first = await agent.investigate_async(an_image(), "q")
        calls_after_first = vlm.call_count

        second = await agent.investigate_async(an_image(), "q")
        assert vlm.call_count == calls_after_first, "the repeat should be served from cache"
        assert first.answer == second.answer


class TestReproducibility:
    def test_the_same_input_gives_the_same_result_twice(self) -> None:
        """A benchmark number nobody can reproduce is not a number."""

        def run_once() -> InvestigationResult:
            agent = ActiveVisionAgent(
                FakeVLM(["the circles overlap"], exhausted="repeat_last"),
                tools=[measuring_tool({"overlap": True})],
                max_steps=4,
            )
            return asyncio.run(agent.investigate_async(an_image(), "do they overlap?"))

        first, second = run_once(), run_once()
        assert first.answer == second.answer
        assert first.confidence == second.confidence
        assert first.converged == second.converged
        assert len(first.evidence_chain) == len(second.evidence_chain)


class TestStructuredOutput:
    async def test_expect_is_forwarded_and_returned(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM([Answer(overlapping=True)], exhausted="repeat_last"), max_steps=1
        )
        result = await agent.investigate_async(an_image(), "q", expect=Answer)
        assert result.structured == Answer(overlapping=True)

    async def test_a_verified_structured_answer_survives_later_steps(self) -> None:
        """An unverified later look must not overwrite a confirmed answer.

        Later steps are magnified corners; taking the most recent structured
        value regardless would discard the one the tools actually backed.
        """
        agent = ActiveVisionAgent(
            FakeVLM(
                [Answer(overlapping=True), Answer(overlapping=False)],
                exhausted="repeat_last",
            ),
            tools=[measuring_tool({"overlap": True})],
            max_steps=2,
            confidence_threshold=0.99,
        )
        result = await agent.investigate_async(an_image(), "q", expect=Answer)
        assert result.structured == Answer(overlapping=True)


class TestEvidenceRecordsWhy:
    """Rule 8: a chain showing only coordinates cannot be argued with."""

    async def test_every_step_records_why_it_looked_there(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM(["x"], exhausted="repeat_last"), max_steps=3, confidence_threshold=0.99
        )
        result = await agent.investigate_async(an_image(), "q")

        assert all(step.reason for step in result.evidence_chain)
        assert "whole image" in result.evidence_chain[0].reason

    async def test_the_reason_is_not_stuffed_into_image_ref(self) -> None:
        """image_ref means a path or cache key, not prose."""
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"), max_steps=1)
        result = await agent.investigate_async(an_image(), "q")
        assert result.evidence_chain[0].image_ref is None

    async def test_a_conflict_is_explained_in_the_next_step(self) -> None:
        agent = ActiveVisionAgent(
            FakeVLM(["there are 3 people"], exhausted="repeat_last"),
            tools=[measuring_tool({"people": 2})],
            max_steps=2,
            confidence_threshold=0.99,
        )
        result = await agent.investigate_async(an_image(), "q")
        assert "conflict" in result.evidence_chain[1].reason.lower()


class TestSyncWrapper:
    def test_investigate_runs_the_loop(self) -> None:
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"), max_steps=1)
        result = agent.investigate(an_image(), "q")
        assert isinstance(result, InvestigationResult)

    async def test_calling_it_inside_a_loop_says_what_to_do_instead(self) -> None:
        """Deadlocking would be worse than an error naming the fix.

        This test is async, so it runs inside a live event loop — which is
        exactly the situation the guard exists for.
        """
        agent = ActiveVisionAgent(FakeVLM(["x"], exhausted="repeat_last"))
        with pytest.raises(RuntimeError, match="await investigate_async"):
            agent.investigate(an_image(), "q")


class TestConstruction:
    def test_invalid_max_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            ActiveVisionAgent(FakeVLM(["x"]), max_steps=0)

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            ActiveVisionAgent(FakeVLM(["x"]), confidence_threshold=0.0)

    def test_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            ActiveVisionAgent(FakeVLM(["x"]), confidence_threshold=1.5)

    def test_a_model_string_is_accepted_without_credentials(self) -> None:
        """Naming a model must not require an API key at construction."""
        agent = ActiveVisionAgent("google:gemini-2.5-flash")
        assert agent.max_steps == 8
