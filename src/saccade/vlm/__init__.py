"""VLM implementations bundled with Saccade.

``PydanticAIVLM`` is the one place that knows about ``BinaryContent``;
``FakeVLM`` is the scripted stand-in for tests.
"""

from __future__ import annotations

from saccade.vlm._cache import FileCache, MemoryCache, NullCache, make_cache_key
from saccade.vlm.fake import FakeVLM
from saccade.vlm.pydantic_ai import PydanticAIVLM

__all__ = [
    "FakeVLM",
    "FileCache",
    "MemoryCache",
    "NullCache",
    "PydanticAIVLM",
    "make_cache_key",
]
