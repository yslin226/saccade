"""Tests that the ports work as structural types.

The point of using Protocol is that an implementation needs no import from
Saccade and no inheritance. These tests assert exactly that.
"""

from __future__ import annotations

from PIL import Image

from saccade import CachePort, VLMPort, VLMResponse


class StandaloneVLM:
    """Implements VLMPort without inheriting from it."""

    @property
    def model_id(self) -> str:
        return "standalone"

    async def ask(
        self,
        images: list[Image.Image],
        prompt: str,
        output_type: type | None = None,
    ) -> VLMResponse:
        return VLMResponse(text=f"saw {len(images)} image(s)", model_id=self.model_id)


class StandaloneCache:
    """Implements CachePort without inheriting from it."""

    def __init__(self) -> None:
        self.store: dict[str, VLMResponse] = {}

    def get(self, key: str) -> VLMResponse | None:
        return self.store.get(key)

    def set(self, key: str, value: VLMResponse) -> None:
        self.store[key] = value


class NotAVLM:
    def hello(self) -> str:
        return "hi"


class TestVLMPort:
    def test_structural_implementation_recognised(self) -> None:
        assert isinstance(StandaloneVLM(), VLMPort)

    def test_unrelated_class_rejected(self) -> None:
        assert not isinstance(NotAVLM(), VLMPort)

    async def test_can_be_called_through_the_port(self) -> None:
        vlm: VLMPort = StandaloneVLM()
        response = await vlm.ask([Image.new("RGB", (4, 4))], "what is this?")
        assert response.text == "saw 1 image(s)"
        assert response.model_id == "standalone"

    def test_bundled_fake_satisfies_the_port(self) -> None:
        from saccade.vlm import FakeVLM

        assert isinstance(FakeVLM(["yes"]), VLMPort)


class TestCachePort:
    def test_structural_implementation_recognised(self) -> None:
        assert isinstance(StandaloneCache(), CachePort)

    def test_unrelated_class_rejected(self) -> None:
        assert not isinstance(NotAVLM(), CachePort)

    def test_round_trip_through_the_port(self) -> None:
        cache: CachePort = StandaloneCache()
        assert cache.get("missing") is None
        cache.set("k", VLMResponse(text="cached"))
        stored = cache.get("k")
        assert stored is not None
        assert stored.text == "cached"

    def test_bundled_caches_satisfy_the_port(self) -> None:
        from saccade.vlm import MemoryCache

        assert isinstance(MemoryCache(), CachePort)
