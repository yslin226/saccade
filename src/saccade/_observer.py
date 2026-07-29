"""Internal: asking the VLM what it sees, with the cache in front.

Every model call in the library goes through here, which is how rule 6 is
kept honest — a cache that some code paths bypass is not a cache, it is a
source of irreproducible benchmark numbers.
"""

from __future__ import annotations

import logging

from PIL.Image import Image

from saccade.models import Observation, VLMResponse
from saccade.ports import CachePort, VLMPort
from saccade.vlm._cache import make_cache_key

__all__ = ["Observer"]

logger = logging.getLogger("saccade")

OBSERVE_PROMPT = """You are looking at a region of a larger image.

Question under investigation: {question}

Describe only what you can actually see in this view. State it plainly and
concretely — positions, counts, whether things touch or cross. Do not guess
at what falls outside the view, and do not infer what "should" be there.

If the view is too small, too blurred or too ambiguous to tell, say so
explicitly rather than picking the more likely answer.

Then give your confidence from 0.0 to 1.0 on the last line, formatted
exactly as: CONFIDENCE: <number>"""


class Observer:
    """Turns a view of an image into an :class:`~saccade.models.Observation`.

    Args:
        vlm: The model to ask.
        cache: Optional response cache. Strongly recommended.
    """

    def __init__(self, vlm: VLMPort, cache: CachePort | None = None) -> None:
        self._vlm = vlm
        self._cache = cache
        self._tokens_used = 0
        self._cache_hits = 0
        self._calls = 0

    @property
    def tokens_used(self) -> int:
        """Tokens spent on real calls. Cache hits cost nothing and count nothing."""
        return self._tokens_used

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def calls(self) -> int:
        """Number of observations requested, cached or not."""
        return self._calls

    async def observe(
        self,
        images: list[Image],
        question: str,
        *,
        output_type: type | None = None,
    ) -> tuple[Observation, VLMResponse]:
        """Ask about ``images``, returning the parsed observation and raw reply."""
        self._calls += 1
        prompt = OBSERVE_PROMPT.format(question=question)

        response = await self._fetch(images, prompt, output_type)
        return _to_observation(response), response

    async def _fetch(
        self,
        images: list[Image],
        prompt: str,
        output_type: type | None,
    ) -> VLMResponse:
        if self._cache is None:
            response = await self._vlm.ask(images, prompt, output_type)
            self._tokens_used += response.tokens_used
            return response

        key = make_cache_key(images, prompt, self._vlm.model_id)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            logger.debug("cache hit for %s", key[:12])
            return cached

        response = await self._vlm.ask(images, prompt, output_type)
        self._tokens_used += response.tokens_used
        self._cache.set(key, response)
        return response


def _to_observation(response: VLMResponse) -> Observation:
    """Extract a statement and self-reported confidence from a reply.

    The confidence is the model's own claim about itself, so it is recorded
    but never treated as evidence — only a verification against a measurement
    can raise the agent's real confidence.
    """
    if response.confidence is not None:
        return Observation(statement=response.text.strip(), self_confidence=response.confidence)

    statement, confidence = _split_confidence(response.text)
    return Observation(statement=statement, self_confidence=confidence)


def _split_confidence(text: str) -> tuple[str, float | None]:
    """Pull a trailing ``CONFIDENCE: 0.8`` line off the statement.

    Models are inconsistent about this, so a missing or unparseable line is
    normal and simply means no self-assessment — not an error.
    """
    lines = text.strip().splitlines()
    if not lines:
        return "", None

    last = lines[-1].strip()
    if not last.upper().startswith("CONFIDENCE:"):
        return text.strip(), None

    raw = last.split(":", 1)[1].strip()
    statement = "\n".join(lines[:-1]).strip()

    try:
        value = float(raw)
    except ValueError:
        logger.debug("unparseable confidence line: %r", last)
        return statement, None

    if not 0.0 <= value <= 1.0:
        logger.debug("confidence out of range: %s", value)
        return statement, None
    return statement, value
