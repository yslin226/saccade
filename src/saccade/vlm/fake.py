"""A scripted VLM for tests.

Public on purpose: anyone integrating Saccade needs a way to test their own
tools and prompts without spending API calls or depending on a model that
answers differently every run.
"""

from __future__ import annotations

from typing import Any

from PIL.Image import Image
from pydantic import BaseModel

from saccade.exceptions import VLMError
from saccade.models import VLMResponse

__all__ = ["FakeVLM"]


class FakeVLM:
    """Returns canned responses in order and records how it was called.

    Args:
        responses: Replies to hand out, one per :meth:`ask`. Strings become
            ``VLMResponse.text``; ``BaseModel`` instances additionally land in
            ``VLMResponse.structured``.
        model_id: Reported identity, so cache keys look realistic.
        exhausted: What to do once ``responses`` runs out — ``"raise"``
            (default) surfaces the bug of asking more times than the test
            scripted; ``"repeat_last"`` keeps returning the final response.

    Example:
        >>> vlm = FakeVLM(["two circles, clearly overlapping"])
        >>> vlm.model_id
        'fake'
    """

    def __init__(
        self,
        responses: list[str] | list[BaseModel] | list[str | BaseModel],
        *,
        model_id: str = "fake",
        exhausted: str = "raise",
    ) -> None:
        if exhausted not in ("raise", "repeat_last"):
            raise ValueError(f"exhausted must be 'raise' or 'repeat_last', got {exhausted!r}")
        self._responses: list[str | BaseModel] = list(responses)
        self._model_id = model_id
        self._exhausted = exhausted
        self._calls: list[tuple[int, str, type | None]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def calls(self) -> list[tuple[int, str, type | None]]:
        """One ``(image_count, prompt, output_type)`` tuple per call, in order."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    async def ask(
        self,
        images: list[Image],
        prompt: str,
        output_type: type | None = None,
    ) -> VLMResponse:
        index = len(self._calls)
        self._calls.append((len(images), prompt, output_type))

        if index >= len(self._responses):
            if self._exhausted == "raise":
                raise VLMError(
                    f"FakeVLM ran out of responses: call {index + 1} was made but only "
                    f"{len(self._responses)} response(s) were scripted"
                )
            if not self._responses:
                raise VLMError("FakeVLM was constructed with no responses")
            reply: str | BaseModel = self._responses[-1]
        else:
            reply = self._responses[index]

        structured: Any | None = None
        if isinstance(reply, BaseModel):
            structured = reply
            text = reply.model_dump_json()
        else:
            text = reply

        return VLMResponse(
            text=text,
            raw={"fake": True, "call_index": index},
            tokens_used=0,
            model_id=self._model_id,
            structured=structured,
        )

    def reset(self) -> None:
        """Forget recorded calls and restart the response sequence."""
        self._calls.clear()
