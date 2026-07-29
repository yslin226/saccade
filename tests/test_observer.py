"""Tests for the observer.

The load-bearing assertion here is the cache one: a second identical request
must not reach the model. Rule 6 exists so benchmark runs are reproducible
and cheap, and a cache that some path bypasses provides neither.
"""

from __future__ import annotations

import pytest
from PIL import Image

from saccade._observer import OBSERVE_PROMPT, Observer, _split_confidence
from saccade.models import VLMResponse
from saccade.vlm import FakeVLM, MemoryCache


def an_image(color: str = "white") -> Image.Image:
    return Image.new("RGB", (8, 8), color=color)


class TestObserve:
    async def test_returns_the_statement(self) -> None:
        vlm = FakeVLM(["two circles, clearly overlapping"])
        observation, _ = await Observer(vlm).observe([an_image()], "do they overlap?")
        assert observation.statement == "two circles, clearly overlapping"

    async def test_question_is_embedded_in_the_prompt(self) -> None:
        vlm = FakeVLM(["yes"])
        await Observer(vlm).observe([an_image()], "how many lines cross?")
        _, prompt, _ = vlm.calls[0]
        assert "how many lines cross?" in prompt

    async def test_prompt_tells_the_model_it_may_say_it_cannot_tell(self) -> None:
        """Forcing a guess is how a blind model produces confident nonsense."""
        assert "too ambiguous" in OBSERVE_PROMPT
        assert "rather than picking the more likely answer" in OBSERVE_PROMPT

    async def test_raw_response_is_returned_alongside(self) -> None:
        vlm = FakeVLM(["a circle"])
        _, response = await Observer(vlm).observe([an_image()], "q")
        assert isinstance(response, VLMResponse)
        assert response.model_id == "fake"

    async def test_multiple_images_are_forwarded(self) -> None:
        vlm = FakeVLM(["compared"])
        await Observer(vlm).observe([an_image(), an_image("black")], "q")
        image_count, _, _ = vlm.calls[0]
        assert image_count == 2

    async def test_output_type_is_forwarded(self) -> None:
        from pydantic import BaseModel

        class Answer(BaseModel):
            overlapping: bool

        vlm = FakeVLM([Answer(overlapping=True)])
        await Observer(vlm).observe([an_image()], "q", output_type=Answer)
        _, _, output_type = vlm.calls[0]
        assert output_type is Answer


class TestCaching:
    async def test_identical_request_does_not_reach_the_model_twice(self) -> None:
        vlm = FakeVLM(["first answer"])
        observer = Observer(vlm, MemoryCache())

        first, _ = await observer.observe([an_image()], "q")
        second, _ = await observer.observe([an_image()], "q")

        assert vlm.call_count == 1, "second observation should have been served from cache"
        assert first.statement == second.statement
        assert observer.cache_hits == 1
        assert observer.calls == 2

    async def test_a_different_question_is_a_different_request(self) -> None:
        vlm = FakeVLM(["a", "b"])
        observer = Observer(vlm, MemoryCache())

        await observer.observe([an_image()], "question one")
        await observer.observe([an_image()], "question two")

        assert vlm.call_count == 2
        assert observer.cache_hits == 0

    async def test_a_different_image_is_a_different_request(self) -> None:
        vlm = FakeVLM(["a", "b"])
        observer = Observer(vlm, MemoryCache())

        await observer.observe([an_image("white")], "q")
        await observer.observe([an_image("black")], "q")

        assert vlm.call_count == 2

    async def test_cache_hits_cost_no_tokens(self) -> None:
        class BillingVLM:
            model_id = "billing"

            async def ask(
                self, images: list[Image.Image], prompt: str, output_type: type | None = None
            ) -> VLMResponse:
                return VLMResponse(text="answer", tokens_used=500)

        observer = Observer(BillingVLM(), MemoryCache())

        await observer.observe([an_image()], "q")
        assert observer.tokens_used == 500

        await observer.observe([an_image()], "q")
        assert observer.tokens_used == 500, "a cache hit must not be billed"

    async def test_works_without_a_cache(self) -> None:
        vlm = FakeVLM(["a", "b"])
        observer = Observer(vlm, None)

        await observer.observe([an_image()], "q")
        await observer.observe([an_image()], "q")

        assert vlm.call_count == 2
        assert observer.cache_hits == 0

    async def test_tokens_are_accumulated_across_calls(self) -> None:
        class CountingVLM:
            model_id = "counting"

            def __init__(self) -> None:
                self.n = 0

            async def ask(
                self, images: list[Image.Image], prompt: str, output_type: type | None = None
            ) -> VLMResponse:
                self.n += 1
                return VLMResponse(text=f"reply {self.n}", tokens_used=100)

        observer = Observer(CountingVLM())
        await observer.observe([an_image()], "q1")
        await observer.observe([an_image()], "q2")
        assert observer.tokens_used == 200


class TestConfidenceParsing:
    def test_trailing_confidence_line_is_extracted(self) -> None:
        statement, confidence = _split_confidence("Two circles touch.\nCONFIDENCE: 0.8")
        assert statement == "Two circles touch."
        assert confidence == 0.8

    def test_absent_confidence_is_none(self) -> None:
        statement, confidence = _split_confidence("Two circles touch.")
        assert statement == "Two circles touch."
        assert confidence is None

    def test_case_insensitive(self) -> None:
        _, confidence = _split_confidence("x\nconfidence: 0.5")
        assert confidence == 0.5

    def test_unparseable_value_is_ignored(self) -> None:
        statement, confidence = _split_confidence("x\nCONFIDENCE: very sure")
        assert statement == "x"
        assert confidence is None

    def test_out_of_range_value_is_ignored(self) -> None:
        """A model claiming 1.5 confidence is not evidence of anything."""
        _, confidence = _split_confidence("x\nCONFIDENCE: 1.5")
        assert confidence is None

    def test_empty_text(self) -> None:
        assert _split_confidence("") == ("", None)

    def test_multiline_statement_is_preserved(self) -> None:
        statement, confidence = _split_confidence("Line one.\nLine two.\nCONFIDENCE: 0.9")
        assert statement == "Line one.\nLine two."
        assert confidence == 0.9

    async def test_parsed_confidence_reaches_the_observation(self) -> None:
        vlm = FakeVLM(["They overlap.\nCONFIDENCE: 0.75"])
        observation, _ = await Observer(vlm).observe([an_image()], "q")
        assert observation.statement == "They overlap."
        assert observation.self_confidence == 0.75

    async def test_structured_confidence_takes_precedence(self) -> None:
        class ConfidentVLM:
            model_id = "confident"

            async def ask(
                self, images: list[Image.Image], prompt: str, output_type: type | None = None
            ) -> VLMResponse:
                return VLMResponse(text="They overlap.", confidence=0.9)

        observation, _ = await Observer(ConfidentVLM()).observe([an_image()], "q")
        assert observation.self_confidence == 0.9


class TestErrorPropagation:
    async def test_vlm_errors_are_not_swallowed(self) -> None:
        from saccade import VLMError

        vlm = FakeVLM([])
        with pytest.raises(VLMError):
            await Observer(vlm).observe([an_image()], "q")
