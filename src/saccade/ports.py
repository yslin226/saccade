"""Ports — the interfaces callers can implement to swap out infrastructure.

Both are :class:`typing.Protocol` rather than base classes: an implementation
does not have to import or inherit from Saccade to satisfy them, which keeps
the dependency arrow pointing inwards.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL.Image import Image

from saccade.models import VLMResponse

__all__ = ["CachePort", "VLMPort"]


@runtime_checkable
class VLMPort(Protocol):
    """Anything that can look at images and answer a prompt.

    The signature takes ``PIL.Image.Image`` because that is what the visual
    actions and geometry code operate on. Providers that want something else
    convert at their own boundary — the bundled Pydantic AI adapter is the
    only place that knows about ``BinaryContent``.
    """

    @property
    def model_id(self) -> str:
        """Stable identifier for the model, used in cache keys and evidence."""
        ...

    async def ask(
        self,
        images: list[Image],
        prompt: str,
        output_type: type | None = None,
    ) -> VLMResponse:
        """Ask about ``images``. When ``output_type`` is given, the adapter
        should request structured output and populate ``VLMResponse.structured``.
        """
        ...


@runtime_checkable
class CachePort(Protocol):
    """Storage for VLM responses, keyed by image bytes + prompt + model id.

    Sync on purpose: the bundled implementation is a local filesystem cache,
    and forcing callers to await a dict lookup buys nothing.
    """

    def get(self, key: str) -> VLMResponse | None: ...

    def set(self, key: str, value: VLMResponse) -> None: ...
