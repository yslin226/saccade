"""Pipeline tests for the BlindTest runner.

The dataset loader is stubbed and the model is FakeVLM, so these run with no
network and no API key. What they check is the accounting: that a correct
answer is counted correct, that a failed call is counted wrong rather than
dropped, and that the reported accuracy is the number it claims to be.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from benchmarks.blindtest import run as runner
from benchmarks.blindtest.dataset import BlindTestItem
from saccade.models import VLMResponse
from saccade.vlm import FakeVLM

TASK = "Touching Circles"


def item(groundtruth: str, image_id: str = "abc") -> BlindTestItem:
    """Build one item.

    Each gets visibly different pixels. Real benchmark items always differ,
    and identical images would share a cache key — the run would silently
    answer three questions with one model call.
    """
    image = Image.new("RGB", (64, 64), "white")
    for offset, char in enumerate(image_id):
        image.putpixel((ord(char) % 64, offset % 64), (0, 0, 0))

    return BlindTestItem(
        task=TASK,
        image=image,
        prompt="Are the two circles touching each other? Answer with Yes/No.",
        groundtruth=groundtruth,
        metadata={"image_id": image_id},
    )


@pytest.fixture
def stub_dataset(monkeypatch: pytest.MonkeyPatch) -> list[BlindTestItem]:
    """Replace the network-backed loader with fixed items."""
    items = [item("Yes", "i1"), item("No", "i2"), item("Yes", "i3")]

    def fake_load(task: str, **kwargs: Any) -> list[BlindTestItem]:
        limit = kwargs.get("limit")
        return items[:limit] if limit else items

    monkeypatch.setattr(runner, "load_task", fake_load)
    return items


class TestBaselineMode:
    async def test_all_correct_gives_full_accuracy(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["Yes", "No", "Yes"])
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert report.accuracy == 1.0
        assert report.n_correct == 3
        assert report.n_items == 3

    async def test_all_wrong_gives_zero(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["No", "Yes", "No"])
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert report.accuracy == 0.0
        assert report.n_correct == 0

    async def test_mixed_accuracy_is_arithmetic(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["Yes", "Yes", "Yes"])  # correct, wrong, correct
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert report.n_correct == 2
        assert report.accuracy == pytest.approx(2 / 3)

    async def test_baseline_takes_exactly_one_look_per_item(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        """One forward pass is the thing being compared against."""
        vlm = FakeVLM(["Yes", "No", "Yes"])
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert report.mean_steps == 1.0
        assert vlm.call_count == 3


class TestSaccadeMode:
    async def test_runs_the_loop_and_scores(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["Yes"], exhausted="repeat_last")
        report = await runner.run(
            "touching_circles",
            "saccade",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            max_steps=2,
            progress=False,
        )
        assert report.n_items == 3
        assert report.n_correct == 2  # items 1 and 3 are "Yes"

    async def test_saccade_may_take_several_looks(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["Yes"], exhausted="repeat_last")
        report = await runner.run(
            "touching_circles",
            "saccade",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            max_steps=3,
            progress=False,
        )
        assert report.mean_steps >= 1.0


class TestErrorAccounting:
    async def test_a_failed_call_is_counted_wrong_not_dropped(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        """Dropping errors would quietly inflate the headline number."""

        class BrokenVLM:
            model_id = "broken"

            async def ask(
                self, images: list[Image.Image], prompt: str, output_type: type | None = None
            ) -> VLMResponse:
                from saccade import VLMError

                raise VLMError("quota exceeded")

        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=BrokenVLM(),
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert report.n_items == 3, "failed items stay in the denominator"
        assert report.n_correct == 0
        assert report.n_errors == 3
        assert report.accuracy == 0.0

    async def test_errors_are_recorded_per_item(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        class SometimesBroken:
            model_id = "flaky"

            def __init__(self) -> None:
                self.n = 0

            async def ask(
                self, images: list[Image.Image], prompt: str, output_type: type | None = None
            ) -> VLMResponse:
                from saccade import VLMError

                self.n += 1
                if self.n == 2:
                    raise VLMError("transient failure")
                return VLMResponse(text="Yes", model_id="flaky")

        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=SometimesBroken(),
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert report.n_errors == 1
        assert sum(1 for o in report.outcomes if o.error) == 1


class TestReportIntegrity:
    async def test_accuracy_matches_the_outcome_list(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        """The headline figure must be derivable from the saved detail."""
        vlm = FakeVLM(["Yes", "Yes", "Yes"])
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        recomputed = sum(1 for o in report.outcomes if o.correct) / len(report.outcomes)
        assert report.accuracy == pytest.approx(recomputed)

    async def test_every_item_is_recorded_with_its_ground_truth(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["Yes", "No", "Yes"])
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert [o.groundtruth for o in report.outcomes] == ["Yes", "No", "Yes"]
        assert [o.image_id for o in report.outcomes] == ["i1", "i2", "i3"]

    async def test_report_is_saved_as_readable_json(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        import json

        vlm = FakeVLM(["Yes", "No", "Yes"])
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        path = runner.save(report, tmp_path / "results")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["accuracy"] == report.accuracy
        assert len(loaded["outcomes"]) == 3

    async def test_summary_reports_the_same_numbers(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["Yes", "No", "Yes"])
        report = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert "3/3" in report.summary()


class TestCaching:
    async def test_a_rerun_costs_nothing(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        """Rule 6: the same run twice should hit cache, not the API."""
        cache_dir = str(tmp_path / "cache")

        first_vlm = FakeVLM(["Yes", "No", "Yes"])
        first = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=first_vlm,
            cache_dir=cache_dir,
            progress=False,
        )
        assert first_vlm.call_count == 3

        second_vlm = FakeVLM(["Yes", "No", "Yes"])
        second = await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=second_vlm,
            cache_dir=cache_dir,
            progress=False,
        )
        assert second_vlm.call_count == 0, "the rerun should be served entirely from cache"
        assert second.accuracy == first.accuracy


class TestDistinctItemsDoNotShareAnswers:
    """Two different questions must never be answered by one cached reply."""

    async def test_distinct_images_each_reach_the_model(
        self, stub_dataset: list[BlindTestItem], tmp_path: Path
    ) -> None:
        vlm = FakeVLM(["Yes", "No", "Yes"])
        await runner.run(
            "touching_circles",
            "baseline",
            limit=3,
            vlm=vlm,
            cache_dir=str(tmp_path),
            progress=False,
        )
        assert vlm.call_count == 3, "each distinct item needs its own call"

    async def test_identical_items_do_share_a_cached_answer(self, tmp_path: Path) -> None:
        """The flip side, and the reason reruns are free."""
        from saccade.vlm._cache import make_cache_key

        first, second = item("Yes", "same"), item("Yes", "same")
        key_a = make_cache_key([first.image], first.prompt, "m")
        key_b = make_cache_key([second.image], second.prompt, "m")
        assert key_a == key_b

    async def test_different_items_get_different_cache_keys(self, tmp_path: Path) -> None:
        from saccade.vlm._cache import make_cache_key

        first, second = item("Yes", "i1"), item("No", "i2")
        key_a = make_cache_key([first.image], first.prompt, "m")
        key_b = make_cache_key([second.image], second.prompt, "m")
        assert key_a != key_b


class TestRateLimiting:
    """Gemini's free tier allows 5 requests/minute, so pacing is mandatory."""

    async def test_zero_rpm_disables_pacing(self) -> None:
        limiter = runner.RateLimiter(0)
        assert limiter.interval == 0.0
        await limiter.wait()  # must not block

    def test_interval_is_derived_from_the_budget(self) -> None:
        assert runner.RateLimiter(5).interval == 12.0
        assert runner.RateLimiter(60).interval == 1.0

    async def test_a_429_is_retried_after_the_requested_wait(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from saccade import VLMError

        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        attempts = {"n": 0}

        async def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise VLMError("status_code: 429 ... Please retry in 47.88s.")
            return "recovered"

        result = await runner.with_retry(flaky, runner.RateLimiter(0))
        assert result == "recovered"
        assert slept and slept[0] == pytest.approx(48.88, abs=0.1)

    async def test_non_rate_limit_errors_are_not_retried(self) -> None:
        from saccade import VLMError

        attempts = {"n": 0}

        async def broken() -> str:
            attempts["n"] += 1
            raise VLMError("status_code: 401 unauthorized")

        with pytest.raises(VLMError, match="401"):
            await runner.with_retry(broken, runner.RateLimiter(0))
        assert attempts["n"] == 1, "an auth failure will not fix itself"

    async def test_retries_eventually_give_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from saccade import VLMError

        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(runner.asyncio, "sleep", fake_sleep)

        async def always_limited() -> str:
            raise VLMError("status_code: 429 quota exceeded")

        with pytest.raises(VLMError, match="429"):
            await runner.with_retry(always_limited, runner.RateLimiter(0))


class TestArguments:
    async def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            await runner.run("touching_circles", "guessing", progress=False)

    async def test_an_empty_task_is_an_error_not_a_zero_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reporting 0% for a task that loaded nothing would be a lie."""
        monkeypatch.setattr(runner, "load_task", lambda task, **kw: [])
        with pytest.raises(RuntimeError, match="no items loaded"):
            await runner.run("touching_circles", "baseline", progress=False)
