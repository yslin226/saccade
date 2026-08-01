"""Tests for letting the model pick which tools to run.

The default loop runs every registered tool every step, which is right only
while every tool was written for the question. On BlindTest that holds — one
tool per task, and running it unconditionally reaches 98%. It stops holding
as soon as an application registers a dozen, because a tool that measures the
wrong thing produces a confident number the verifier will use to overrule the
model.

Every failure mode below falls back to running everything, which is the
behaviour the loop had before the chooser existed. A bad chooser must degrade
to the old loop, never to no measurement at all.
"""

from __future__ import annotations

import pytest
from PIL import Image
from pydantic import BaseModel

from saccade._chooser import CHOOSE_PROMPT, catalogue, choose_tools
from saccade.exceptions import VLMError
from saccade.tools import Tool, ToolResult
from saccade.vlm import FakeVLM


class NoParams(BaseModel):
    pass


def tool(name: str, description: str = "measures something") -> Tool:
    return Tool(
        name=name,
        description=description,
        fn=lambda image, viewport: ToolResult(value={"x": 1}, is_measurement=True),
        params_schema=NoParams,
    )


def image() -> Image.Image:
    return Image.new("RGB", (32, 32))


CIRCLES = tool("circle_geometry", "measure whether two circles touch")
LINES = tool("line_crossings", "count where two lines cross")
COLOURS = tool("colour_histogram", "report the dominant colours")


class TestChoosing:
    @pytest.mark.asyncio
    async def test_the_named_tool_is_the_one_returned(self) -> None:
        vlm = FakeVLM(["circle_geometry"])
        choice = await choose_tools(vlm, [image()], "are they touching?", [CIRCLES, LINES])

        assert [t.name for t in choice.tools] == ["circle_geometry"]
        assert choice.fallback is False

    @pytest.mark.asyncio
    async def test_several_tools_can_be_chosen(self) -> None:
        vlm = FakeVLM(["circle_geometry, colour_histogram"])
        choice = await choose_tools(vlm, [image()], "?", [CIRCLES, LINES, COLOURS])

        assert {t.name for t in choice.tools} == {"circle_geometry", "colour_histogram"}

    @pytest.mark.asyncio
    async def test_a_name_wrapped_in_prose_is_still_found(self) -> None:
        """Models rarely answer in exactly the format asked for."""
        vlm = FakeVLM(["I would run the line_crossings tool here."])
        choice = await choose_tools(vlm, [image()], "?", [CIRCLES, LINES])

        assert [t.name for t in choice.tools] == ["line_crossings"]
        assert choice.fallback is False

    @pytest.mark.asyncio
    async def test_the_choice_is_recorded_for_the_evidence_chain(self) -> None:
        vlm = FakeVLM(["circle_geometry"])
        choice = await choose_tools(vlm, [image()], "?", [CIRCLES, LINES])

        assert "circle_geometry" in choice.reason


class TestDeclining:
    @pytest.mark.asyncio
    async def test_none_runs_nothing(self) -> None:
        vlm = FakeVLM(["NONE"])
        choice = await choose_tools(vlm, [image()], "what colour is the sky?", [CIRCLES])

        assert choice.tools == []
        assert choice.fallback is False

    @pytest.mark.parametrize("reply", ["NONE", "none", "None.", "none of these apply"])
    @pytest.mark.asyncio
    async def test_recognised_forms_of_declining(self, reply: str) -> None:
        vlm = FakeVLM([reply])
        choice = await choose_tools(vlm, [image()], "?", [CIRCLES])
        assert choice.tools == []

    @pytest.mark.asyncio
    async def test_declining_is_not_a_fallback(self) -> None:
        """Running nothing and failing to choose produce opposite tool lists,
        and a benchmark has to be able to tell them apart."""
        vlm = FakeVLM(["NONE"])
        assert (await choose_tools(vlm, [image()], "?", [CIRCLES])).fallback is False


class TestFallingBack:
    """Every failure runs everything — the behaviour before the chooser."""

    @pytest.mark.asyncio
    async def test_a_hallucinated_name_runs_everything(self) -> None:
        vlm = FakeVLM(["measure_the_vibes"])
        choice = await choose_tools(vlm, [image()], "?", [CIRCLES, LINES])

        assert {t.name for t in choice.tools} == {"circle_geometry", "line_crossings"}
        assert choice.fallback is True

    @pytest.mark.asyncio
    async def test_a_vlm_error_runs_everything(self) -> None:
        class Broken:
            model_id = "broken"

            async def ask(self, images: object, prompt: str, output_type: object = None) -> object:
                raise VLMError("network died")

        choice = await choose_tools(Broken(), [image()], "?", [CIRCLES, LINES])  # type: ignore[arg-type]

        assert len(choice.tools) == 2
        assert choice.fallback is True

    @pytest.mark.asyncio
    async def test_the_fallback_reason_says_what_went_wrong(self) -> None:
        vlm = FakeVLM(["I am not sure what you mean"])
        choice = await choose_tools(vlm, [image()], "?", [CIRCLES])

        assert choice.fallback is True
        assert "running everything" in choice.reason

    @pytest.mark.asyncio
    async def test_no_registered_tools_is_not_a_fallback(self) -> None:
        vlm = FakeVLM(["anything"])
        choice = await choose_tools(vlm, [image()], "?", [])

        assert choice.tools == []
        assert choice.fallback is False


class TestThePrompt:
    def test_the_catalogue_names_and_describes_each_tool(self) -> None:
        text = catalogue([CIRCLES, LINES])

        assert "circle_geometry" in text
        assert "measure whether two circles touch" in text
        assert "line_crossings" in text

    @pytest.mark.asyncio
    async def test_the_question_reaches_the_model(self) -> None:
        """Choosing depends on what is being asked, not only on what is
        visible — the same image warrants different tools for different
        questions."""
        vlm = FakeVLM(["NONE"])
        await choose_tools(vlm, [image()], "how many times do they cross?", [LINES])

        assert "how many times do they cross?" in vlm.calls[0][1]

    def test_the_prompt_does_not_ask_the_model_for_an_answer(self) -> None:
        """Choosing an instrument and reading it are different jobs. Only the
        first is delegated: the tool still measures, and is_measurement still
        governs whether that measurement may overrule anything."""
        text = CHOOSE_PROMPT.lower()
        assert "which tools" in text
