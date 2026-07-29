"""VLM implementations bundled with Saccade.

``FakeVLM`` is here in M0. The Pydantic AI adapter — the one place that knows
about ``BinaryContent`` — arrives in M1.
"""

from __future__ import annotations

from saccade.vlm._cache import FileCache, MemoryCache, make_cache_key
from saccade.vlm.fake import FakeVLM

__all__ = ["FakeVLM", "FileCache", "MemoryCache", "make_cache_key"]
