"""Tests for the VLM response cache.

All filesystem tests use tmp_path so they never touch the real cache dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from saccade import VLMResponse
from saccade.vlm import FileCache, MemoryCache, make_cache_key
from saccade.vlm._cache import DEFAULT_CACHE_DIR


def solid(color: str, size: tuple[int, int] = (8, 8)) -> Image.Image:
    return Image.new("RGB", size, color=color)


class TestCacheKey:
    def test_same_inputs_give_the_same_key(self) -> None:
        key1 = make_cache_key([solid("red")], "how many circles?", "gemini")
        key2 = make_cache_key([solid("red")], "how many circles?", "gemini")
        assert key1 == key2

    def test_different_images_give_different_keys(self) -> None:
        red = make_cache_key([solid("red")], "q", "gemini")
        blue = make_cache_key([solid("blue")], "q", "gemini")
        assert red != blue

    def test_different_prompts_give_different_keys(self) -> None:
        assert make_cache_key([solid("red")], "q1", "m") != make_cache_key(
            [solid("red")], "q2", "m"
        )

    def test_different_models_give_different_keys(self) -> None:
        assert make_cache_key([solid("red")], "q", "gemini") != make_cache_key(
            [solid("red")], "q", "gpt"
        )

    def test_image_order_matters(self) -> None:
        forward = make_cache_key([solid("red"), solid("blue")], "q", "m")
        reverse = make_cache_key([solid("blue"), solid("red")], "q", "m")
        assert forward != reverse

    def test_image_count_matters(self) -> None:
        one = make_cache_key([solid("red")], "q", "m")
        two = make_cache_key([solid("red"), solid("red")], "q", "m")
        assert one != two

    def test_key_is_a_sha256_hex_digest(self) -> None:
        key = make_cache_key([solid("red")], "q", "m")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestFileCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path)
        response = VLMResponse(text="two circles", tokens_used=17, model_id="gemini")
        cache.set("abc", response)

        loaded = cache.get("abc")
        assert loaded is not None
        assert loaded.text == "two circles"
        assert loaded.tokens_used == 17
        assert loaded.model_id == "gemini"

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        assert FileCache(tmp_path).get("never-stored") is None

    def test_directory_is_created_when_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "cache"
        assert not target.exists()
        FileCache(target)
        assert target.is_dir()

    def test_survives_a_new_instance(self, tmp_path: Path) -> None:
        FileCache(tmp_path).set("k", VLMResponse(text="persisted"))
        loaded = FileCache(tmp_path).get("k")
        assert loaded is not None
        assert loaded.text == "persisted"

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path)
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert cache.get("broken") is None

    def test_no_temp_files_left_behind(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path)
        cache.set("k", VLMResponse(text="x"))
        assert list(tmp_path.glob("*.tmp")) == []

    def test_write_failure_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache that cannot write must not take the investigation down with it."""

        def explode(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        cache = FileCache(tmp_path)
        monkeypatch.setattr(Path, "write_text", explode)
        cache.set("k", VLMResponse(text="x"))  # must not raise

        assert list(tmp_path.glob("*.tmp")) == []
        assert len(cache) == 0

    def test_len_and_clear(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path)
        cache.set("a", VLMResponse(text="1"))
        cache.set("b", VLMResponse(text="2"))
        assert len(cache) == 2
        assert cache.clear() == 2
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_uses_env_var_when_no_directory_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "from-env"
        monkeypatch.setenv("SACCADE_CACHE_DIR", str(target))
        cache = FileCache()
        assert cache.directory == target
        assert target.is_dir()

    def test_default_dir_constant_matches_the_documented_default(self) -> None:
        assert DEFAULT_CACHE_DIR == "./cache"

    def test_key_and_cache_work_together(self, tmp_path: Path) -> None:
        cache = FileCache(tmp_path)
        key = make_cache_key([solid("red")], "how many?", "gemini")
        assert cache.get(key) is None
        cache.set(key, VLMResponse(text="three", model_id="gemini"))

        same_key = make_cache_key([solid("red")], "how many?", "gemini")
        hit = cache.get(same_key)
        assert hit is not None
        assert hit.text == "three"


class TestMemoryCache:
    def test_round_trip(self) -> None:
        cache = MemoryCache()
        assert cache.get("k") is None
        cache.set("k", VLMResponse(text="cached", tokens_used=3))
        loaded = cache.get("k")
        assert loaded is not None
        assert loaded.text == "cached"
        assert loaded.tokens_used == 3

    def test_len(self) -> None:
        cache = MemoryCache()
        cache.set("a", VLMResponse(text="1"))
        cache.set("b", VLMResponse(text="2"))
        assert len(cache) == 2
