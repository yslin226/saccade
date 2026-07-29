"""The Pydantic AI adapter.

This is the only module in the project that knows ``BinaryContent`` exists
(spec 4.3). Everything else — the public API, the visual actions, the
geometry — speaks ``PIL.Image.Image``. Keeping the conversion in one place
means swapping agent frameworks later is a rewrite of this file rather than
a search across the codebase.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL.Image import Image
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import AgentRunError, UserError

from saccade.exceptions import VLMError
from saccade.models import VLMResponse

__all__ = ["PydanticAIVLM"]

# PNG rather than JPEG: the benchmark tasks turn on thin lines and exact
# circle edges, and JPEG artefacts are precisely the kind of detail damage
# this project exists to avoid introducing.
_IMAGE_FORMAT = "PNG"
_IMAGE_MEDIA_TYPE = "image/png"


def _to_binary_content(image: Image) -> BinaryContent:
    """Convert a PIL image into what Pydantic AI actually accepts.

    ``Agent.run()`` takes no PIL objects; its multimodal input is
    ``BinaryContent(data=..., media_type=...)``. This function is the seam.
    """
    buffer = BytesIO()
    # RGBA/P images cannot always round-trip through every provider, and the
    # alpha channel carries nothing for these tasks.
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, format=_IMAGE_FORMAT)
    return BinaryContent(data=buffer.getvalue(), media_type=_IMAGE_MEDIA_TYPE)


class PydanticAIVLM:
    """A :class:`~saccade.ports.VLMPort` backed by Pydantic AI.

    Args:
        model: A model string with a provider prefix
            (``"google:gemini-flash-latest"``, ``"openai:gpt-4.1"``), or a
            constructed ``Model`` instance. Azure and other
            OpenAI-compatible endpoints — GLM, Qwen, OpenRouter, Ollama —
            need the instance form, because an endpoint and API version
            cannot be expressed in a model string:

            >>> from pydantic_ai.models.openai import OpenAIChatModel
            >>> from pydantic_ai.providers.openai import OpenAIProvider
            >>> model = OpenAIChatModel(
            ...     "glm-4.6v",
            ...     provider=OpenAIProvider(base_url="https://...", api_key="..."),
            ... )

        system_prompt: Optional instructions applied to every call.

    Example:
        >>> vlm = PydanticAIVLM("google:gemini-2.5-flash")
        >>> vlm.model_id
        'google:gemini-2.5-flash'
    """

    def __init__(self, model: str | Any, *, system_prompt: str | None = None) -> None:
        self._model = model
        self._system_prompt = system_prompt
        self._model_id = model if isinstance(model, str) else _model_name(model)
        self._agent_instance: Agent[None, Any] | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def _agent(self) -> Agent[None, Any]:
        """Build the Agent on first use, not in ``__init__``.

        Constructing it eagerly resolves the provider and demands credentials
        immediately, so merely naming a model would raise without an API key.
        That would make the adapter impossible to construct in tests or in CI,
        and would report a missing key at a point that says nothing about
        which call needed it.
        """
        if self._agent_instance is None:
            try:
                self._agent_instance = (
                    Agent(self._model, system_prompt=self._system_prompt)
                    if self._system_prompt
                    else Agent(self._model)
                )
            except (AgentRunError, UserError) as exc:
                raise VLMError(f"could not initialise model {self._model_id!r}: {exc}") from exc
        return self._agent_instance

    async def ask(
        self,
        images: list[Image],
        prompt: str,
        output_type: type | None = None,
    ) -> VLMResponse:
        """Send images and a prompt to the model.

        Raises:
            VLMError: If the call fails for any reason — network, auth,
                quota, or a reply the framework could not make sense of.
                The original exception is kept as ``__cause__``.
        """
        content: list[Any] = [prompt]
        content.extend(_to_binary_content(image) for image in images)

        try:
            if output_type is None:
                result = await self._agent.run(content)
            else:
                result = await self._agent.run(content, output_type=output_type)
        except (AgentRunError, UserError) as exc:
            raise VLMError(f"VLM call to {self._model_id} failed: {exc}") from exc

        return self._to_response(result, structured=output_type is not None)

    def _to_response(self, result: Any, *, structured: bool) -> VLMResponse:
        output = result.output
        usage = _usage_of(result)

        return VLMResponse(
            text=output if isinstance(output, str) else _render(output),
            raw={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "requests": usage.requests,
            },
            tokens_used=usage.input_tokens + usage.output_tokens,
            model_id=self._model_id,
            structured=output if structured else None,
        )


def _usage_of(result: Any) -> Any:
    """Read token usage off a run result.

    ``usage`` is a property in pydantic-ai 2.x but was a method earlier, and
    both shapes appear in the wild depending on the installed version. Accept
    either rather than crash on a working reply.
    """
    usage = result.usage
    return usage() if callable(usage) else usage


def _render(output: Any) -> str:
    """Best-effort text form of a structured output, for the evidence chain."""
    dump = getattr(output, "model_dump_json", None)
    return dump() if callable(dump) else str(output)


def _model_name(model: Any) -> str:
    """Pull an identifier out of a constructed Model instance."""
    for attribute in ("model_name", "name"):
        value = getattr(model, attribute, None)
        if isinstance(value, str):
            return value
    return type(model).__name__
