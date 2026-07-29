"""On-disk cache for VLM responses.

Why this exists at all (CLAUDE.md rule 6): the same image and prompt should
cost one API call across a whole benchmark run, and a test that re-runs the
loop must get the same answers it got last time.

File I/O lives here deliberately. Rule 3 bans I/O in the pure logic modules
(``_planner``, ``_verifier``, ``_evidence``, ``geometry/``); persistence is
this module's entire job.
"""

from __future__ import annotations

import hashlib
import logging
import os
from io import BytesIO
from pathlib import Path

from PIL.Image import Image

from saccade.models import VLMResponse

__all__ = ["FileCache", "MemoryCache", "make_cache_key"]

logger = logging.getLogger("saccade")

DEFAULT_CACHE_DIR = "./cache"


def _image_bytes(image: Image) -> bytes:
    """Serialise an image reproducibly.

    PNG at a fixed compression level: the same pixels must always produce the
    same bytes, or the cache key changes for no reason.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def make_cache_key(images: list[Image], prompt: str, model_id: str) -> str:
    """Hash the full request: every input that could change the answer.

    Image byte lengths are folded in alongside the pixels so that two images
    cannot be confused with one concatenated image.
    """
    digest = hashlib.sha256()
    for image in images:
        payload = _image_bytes(image)
        digest.update(str(len(payload)).encode("utf-8"))
        digest.update(payload)
    digest.update(b"\x00")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(model_id.encode("utf-8"))
    return digest.hexdigest()


class FileCache:
    """Stores each response as one JSON file named after its key.

    One file per entry rather than a single index: concurrent benchmark
    workers can write without clobbering each other, and a corrupt entry costs
    one cache miss instead of the whole cache.

    Args:
        directory: Where to store entries. Defaults to ``$SACCADE_CACHE_DIR``,
            then ``./cache``.
    """

    def __init__(self, directory: str | os.PathLike[str] | None = None) -> None:
        if directory is None:
            directory = os.environ.get("SACCADE_CACHE_DIR", DEFAULT_CACHE_DIR)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> VLMResponse | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return VLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A damaged entry is a miss, not a crash — the caller can re-ask.
            logger.warning("discarding unreadable cache entry %s", path)
            return None

    def set(self, key: str, value: VLMResponse) -> None:
        path = self._path(key)
        # Write to a temp file then replace, so a reader never sees half a file.
        temp = path.with_suffix(".json.tmp")
        try:
            temp.write_text(value.model_dump_json(), encoding="utf-8")
            temp.replace(path)
        except (OSError, TypeError, ValueError):
            logger.warning("failed to cache response under %s", key)
            temp.unlink(missing_ok=True)

    def clear(self) -> int:
        """Delete every entry. Returns how many files were removed."""
        removed = 0
        for path in self.directory.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))


class MemoryCache:
    """A dict-backed cache, for tests that want no filesystem at all."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> VLMResponse | None:
        payload = self._store.get(key)
        return None if payload is None else VLMResponse.model_validate_json(payload)

    def set(self, key: str, value: VLMResponse) -> None:
        self._store[key] = value.model_dump_json()

    def __len__(self) -> int:
        return len(self._store)
