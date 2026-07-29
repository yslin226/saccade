"""Tests for FakeVLM."""

from __future__ import annotations

import pytest
from PIL import Image
from pydantic import BaseModel

from saccade import VLMError
from saccade.vlm import FakeVLM


class Answer(BaseModel):
    overlapping: bool
    reason: str


def an_image(size: tuple[int, int] = (8, 8)) -> Image.Image:
    return Image.new("RGB", size, color="white")


class TestResponseSequence:
    async def test_responses_come_back_in_order(self) -> None:
        vlm = FakeVLM(["first", "second", "third"])
        texts = [(await vlm.ask([an_image()], "q")).text for _ in range(3)]
        assert texts == ["first", "second", "third"]

    async def test_structured_response_populates_both_fields(self) -> None:
        answer = Answer(overlapping=True, reason="centres are 47px apart")
        vlm = FakeVLM([answer])
        response = await vlm.ask([an_image()], "do they overlap?", output_type=Answer)
        assert response.structured == answer
        assert "47px apart" in response.text

    async def test_model_id_is_reported_on_each_response(self) -> None:
        vlm = FakeVLM(["ok"], model_id="fake:v2")
        assert vlm.model_id == "fake:v2"
        assert (await vlm.ask([an_image()], "q")).model_id == "fake:v2"


class TestExhaustion:
    async def test_raises_by_default_when_out_of_responses(self) -> None:
        vlm = FakeVLM(["only one"])
        await vlm.ask([an_image()], "q")
        with pytest.raises(VLMError, match="ran out of responses"):
            await vlm.ask([an_image()], "q")

    async def test_repeat_last_keeps_returning_the_final_response(self) -> None:
        vlm = FakeVLM(["a", "b"], exhausted="repeat_last")
        results = [(await vlm.ask([an_image()], "q")).text for _ in range(4)]
        assert results == ["a", "b", "b", "b"]

    async def test_empty_response_list_raises_on_first_call(self) -> None:
        vlm = FakeVLM([], exhausted="repeat_last")
        with pytest.raises(VLMError, match="no responses"):
            await vlm.ask([an_image()], "q")

    def test_unknown_exhausted_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="exhausted must be"):
            FakeVLM(["a"], exhausted="explode")


class TestCallRecording:
    async def test_calls_record_image_count_prompt_and_output_type(self) -> None:
        vlm = FakeVLM(["a", "b"])
        await vlm.ask([an_image()], "first question")
        await vlm.ask([an_image(), an_image()], "second question", output_type=Answer)

        assert vlm.calls == [
            (1, "first question", None),
            (2, "second question", Answer),
        ]
        assert vlm.call_count == 2

    async def test_calls_are_recorded_even_when_exhausted(self) -> None:
        """A failed call still happened, and the test should be able to see it."""
        vlm = FakeVLM([])
        with pytest.raises(VLMError):
            await vlm.ask([an_image()], "q")
        assert vlm.call_count == 1

    async def test_calls_property_returns_a_copy(self) -> None:
        vlm = FakeVLM(["a"])
        await vlm.ask([an_image()], "q")
        vlm.calls.clear()
        assert vlm.call_count == 1

    async def test_reset_clears_recorded_calls_and_restarts_the_sequence(self) -> None:
        vlm = FakeVLM(["a", "b"])
        await vlm.ask([an_image()], "q")
        vlm.reset()
        assert vlm.call_count == 0
        assert (await vlm.ask([an_image()], "q")).text == "a"
