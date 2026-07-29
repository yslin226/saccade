"""Tests for the Pydantic AI adapter.

Nothing here touches the network. Agent.run is mocked, which is also the
only way to assert the thing that matters: that PIL images are converted to
BinaryContent, and that this module is the only place that happens.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UserError

from saccade import VLMError, VLMPort
from saccade.vlm.pydantic_ai import PydanticAIVLM, _to_binary_content


class Answer(BaseModel):
    overlapping: bool
    reason: str


class FakeUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 5) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.requests = 1


class FakeResult:
    def __init__(self, output: Any, usage: FakeUsage | None = None) -> None:
        self.output = output
        self._usage = usage or FakeUsage()

    def usage(self) -> FakeUsage:
        return self._usage


def an_image(mode: str = "RGB", size: tuple[int, int] = (8, 8)) -> Image.Image:
    return Image.new(mode, size, color="white" if mode != "L" else 255)


class StubAgent:
    """Stands in for pydantic_ai.Agent so no provider is ever resolved."""

    def __init__(self, handler: Any) -> None:
        self.run = handler


def with_agent(vlm: PydanticAIVLM, handler: Any) -> PydanticAIVLM:
    """Inject a stub agent, bypassing lazy construction."""
    vlm._agent_instance = StubAgent(handler)  # type: ignore[assignment]
    return vlm


@pytest.fixture
def vlm() -> PydanticAIVLM:
    return PydanticAIVLM("google:gemini-2.5-flash")


class TestImageConversion:
    """The BinaryContent seam (spec 4.3)."""

    def test_produces_png_binary_content(self) -> None:
        content = _to_binary_content(an_image())
        assert isinstance(content, BinaryContent)
        assert content.media_type == "image/png"
        assert content.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_pixels_survive_the_round_trip(self) -> None:
        original = Image.new("RGB", (4, 4), color=(12, 34, 56))
        restored = Image.open(BytesIO(_to_binary_content(original).data))
        assert restored.size == (4, 4)
        assert restored.getpixel((0, 0)) == (12, 34, 56)

    def test_rgba_is_converted_rather_than_rejected(self) -> None:
        content = _to_binary_content(an_image(mode="RGBA"))
        assert Image.open(BytesIO(content.data)).mode == "RGB"

    def test_greyscale_is_left_alone(self) -> None:
        content = _to_binary_content(an_image(mode="L"))
        assert Image.open(BytesIO(content.data)).mode == "L"


class TestAsk:
    async def test_prompt_comes_first_then_the_images(self, vlm: PydanticAIVLM) -> None:
        captured: dict[str, Any] = {}

        async def fake_run(content: Any, **kwargs: Any) -> FakeResult:
            captured["content"] = content
            captured["kwargs"] = kwargs
            return FakeResult("two circles")

        with_agent(vlm, fake_run)
        await vlm.ask([an_image(), an_image()], "how many circles?")

        content = captured["content"]
        assert content[0] == "how many circles?"
        assert len(content) == 3
        assert all(isinstance(part, BinaryContent) for part in content[1:])

    async def test_plain_text_response(self, vlm: PydanticAIVLM) -> None:
        async def fake_run(content: Any, **kwargs: Any) -> FakeResult:
            return FakeResult("two circles, clearly overlapping")

        with_agent(vlm, fake_run)
        response = await vlm.ask([an_image()], "q")

        assert response.text == "two circles, clearly overlapping"
        assert response.structured is None
        assert response.model_id == "google:gemini-2.5-flash"

    async def test_token_usage_is_recorded(self, vlm: PydanticAIVLM) -> None:
        async def fake_run(content: Any, **kwargs: Any) -> FakeResult:
            return FakeResult("x", FakeUsage(input_tokens=120, output_tokens=30))

        with_agent(vlm, fake_run)
        response = await vlm.ask([an_image()], "q")

        assert response.tokens_used == 150
        assert response.raw["input_tokens"] == 120
        assert response.raw["output_tokens"] == 30

    async def test_structured_output_is_requested_and_returned(self, vlm: PydanticAIVLM) -> None:
        answer = Answer(overlapping=True, reason="centres 47px apart")
        captured: dict[str, Any] = {}

        async def fake_run(content: Any, **kwargs: Any) -> FakeResult:
            captured.update(kwargs)
            return FakeResult(answer)

        with_agent(vlm, fake_run)
        response = await vlm.ask([an_image()], "do they overlap?", output_type=Answer)

        assert captured["output_type"] is Answer
        assert response.structured == answer
        assert "47px apart" in response.text

    async def test_output_type_is_not_passed_when_absent(self, vlm: PydanticAIVLM) -> None:
        """Passing output_type=None is not the same as omitting it."""
        captured: dict[str, Any] = {}

        async def fake_run(content: Any, **kwargs: Any) -> FakeResult:
            captured["kwargs"] = kwargs
            return FakeResult("plain")

        with_agent(vlm, fake_run)
        await vlm.ask([an_image()], "q")

        assert "output_type" not in captured["kwargs"]

    async def test_works_with_no_images(self, vlm: PydanticAIVLM) -> None:
        async def fake_run(content: Any, **kwargs: Any) -> FakeResult:
            assert content == ["text only"]
            return FakeResult("ok")

        with_agent(vlm, fake_run)
        assert (await vlm.ask([], "text only")).text == "ok"


class TestErrorWrapping:
    @pytest.mark.parametrize(
        "error",
        [
            ModelHTTPError(status_code=429, model_name="gemini", body="quota exceeded"),
            UnexpectedModelBehavior("malformed tool call"),
            UserError("no API key configured"),
        ],
        ids=["http", "unexpected_behaviour", "user_error"],
    )
    async def test_framework_errors_become_vlm_errors(
        self, vlm: PydanticAIVLM, error: Exception
    ) -> None:
        async def fake_run(content: Any, **kwargs: Any) -> FakeResult:
            raise error

        with_agent(vlm, fake_run)
        with pytest.raises(VLMError) as caught:
            await vlm.ask([an_image()], "q")

        assert caught.value.__cause__ is error
        assert "google:gemini-2.5-flash" in str(caught.value)


class TestConstruction:
    def test_satisfies_the_port(self, vlm: PydanticAIVLM) -> None:
        assert isinstance(vlm, VLMPort)

    def test_model_string_becomes_the_model_id(self, vlm: PydanticAIVLM) -> None:
        assert vlm.model_id == "google:gemini-2.5-flash"

    def test_construction_needs_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Naming a model must not demand credentials.

        Building the Agent eagerly made this raise UserError with no key set,
        which would break every test environment and CI run.
        """
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        adapter = PydanticAIVLM("google:gemini-2.5-flash")
        assert adapter.model_id == "google:gemini-2.5-flash"

    async def test_missing_credentials_surface_as_vlm_error_on_use(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure should arrive at the call, wrapped in our own type."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        adapter = PydanticAIVLM("google:gemini-2.5-flash")
        with pytest.raises(VLMError, match="could not initialise model"):
            await adapter.ask([an_image()], "q")

    def test_model_instance_supplies_its_own_name(self) -> None:
        from saccade.vlm.pydantic_ai import _model_name

        class StubModel:
            model_name = "glm-4.6v"

        assert _model_name(StubModel()) == "glm-4.6v"

    def test_unnameable_model_falls_back_to_its_class(self) -> None:
        from saccade.vlm.pydantic_ai import _model_name

        class Mystery:
            pass

        assert _model_name(Mystery()) == "Mystery"

    def test_system_prompt_is_accepted(self) -> None:
        configured = PydanticAIVLM(
            "google:gemini-2.5-flash", system_prompt="You measure, you do not guess."
        )
        assert configured.model_id == "google:gemini-2.5-flash"
