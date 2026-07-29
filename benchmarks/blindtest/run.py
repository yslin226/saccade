"""Run BlindTest in baseline or saccade mode.

Usage:

    uv run python -m benchmarks.blindtest.run --task touching_circles --limit 50 --mode baseline
    uv run python -m benchmarks.blindtest.run --task touching_circles --limit 50 --mode saccade

Both modes share the same model, cache and scorer. The only difference is
whether the model gets one look or gets to move its attention — which is
the comparison this project exists to make.

Every run writes a JSON file with per-item answers, step counts and token
usage, so a published number can be checked rather than believed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

from benchmarks.blindtest.dataset import TASKS, BlindTestItem, load_task
from benchmarks.blindtest.scoring import score
from saccade import ActiveVisionAgent, VLMError
from saccade.ports import VLMPort
from saccade.vlm import FileCache

T = TypeVar("T")

# The design doc names gemini-2.5-flash, but Google has since retired it for
# new API keys ("no longer available to new users"). gemini-flash-latest
# tracks whatever the current free-tier Flash model is.
DEFAULT_MODEL = "google:gemini-flash-latest"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Gemini's free tier allows 5 requests per minute per model — not the
# 1500/day figure the design doc quotes. At that rate a run must pace itself
# or it spends the whole time being rejected.
DEFAULT_RPM = 5

_RETRY_AFTER = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
_MAX_RETRIES = 4


class RateLimiter:
    """Spaces out calls to stay inside a requests-per-minute budget."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        self._last = time.monotonic()


async def with_retry(call: Callable[[], Awaitable[T]], limiter: RateLimiter) -> T:
    """Run ``call``, honouring the wait the API asks for on a 429.

    Gemini states how long to wait in the error body. Obeying it is both
    faster and politer than a blind exponential backoff.
    """
    for attempt in range(_MAX_RETRIES):
        await limiter.wait()
        try:
            return await call()
        except VLMError as exc:
            if "429" not in str(exc) or attempt == _MAX_RETRIES - 1:
                raise
            match = _RETRY_AFTER.search(str(exc))
            delay = float(match.group(1)) + 1.0 if match else 60.0
            print(f"      rate limited; waiting {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable")


@dataclass
class ItemOutcome:
    """What happened on one question."""

    image_id: str
    prompt: str
    groundtruth: str
    answer: str
    correct: bool
    steps: int
    confidence: float
    converged: bool
    tokens: int
    error: str | None = None


@dataclass
class RunReport:
    """A complete run, with everything needed to reproduce it."""

    task: str
    mode: str
    model: str
    n_items: int
    n_correct: int
    n_errors: int
    accuracy: float
    total_tokens: int
    mean_steps: float
    seconds: float
    timestamp: str
    outcomes: list[ItemOutcome] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.task} / {self.mode} / {self.model}\n"
            f"  accuracy : {self.accuracy:.1%}  ({self.n_correct}/{self.n_items})\n"
            f"  errors   : {self.n_errors}\n"
            f"  tokens   : {self.total_tokens}\n"
            f"  steps    : {self.mean_steps:.2f} mean\n"
            f"  elapsed  : {self.seconds:.1f}s"
        )


async def run_baseline(
    vlm: VLMPort, item: BlindTestItem, limiter: RateLimiter | None = None
) -> ItemOutcome:
    """One look, one answer — what a VLM does today."""
    limiter = limiter or RateLimiter(0)
    try:
        response = await with_retry(lambda: vlm.ask([item.image], item.prompt), limiter)
    except VLMError as exc:
        return _failed(item, str(exc))

    return ItemOutcome(
        image_id=item.image_id,
        prompt=item.prompt,
        groundtruth=item.groundtruth,
        answer=response.text,
        correct=score(item.task, response.text, item.groundtruth),
        steps=1,
        confidence=response.confidence or 0.0,
        converged=True,
        tokens=response.tokens_used,
    )


async def run_saccade(
    agent: ActiveVisionAgent, item: BlindTestItem, limiter: RateLimiter | None = None
) -> ItemOutcome:
    """The full loop: decide where to look, verify, converge."""
    limiter = limiter or RateLimiter(0)
    try:
        result = await with_retry(lambda: agent.investigate_async(item.image, item.prompt), limiter)
    except VLMError as exc:
        return _failed(item, str(exc))

    return ItemOutcome(
        image_id=item.image_id,
        prompt=item.prompt,
        groundtruth=item.groundtruth,
        answer=result.answer,
        correct=score(item.task, result.answer, item.groundtruth),
        steps=len(result.evidence_chain),
        confidence=result.confidence,
        converged=result.converged,
        tokens=result.total_tokens,
    )


def _failed(item: BlindTestItem, error: str) -> ItemOutcome:
    """A failed call is scored wrong, not dropped.

    Dropping errors would quietly inflate accuracy — the run would report
    only the questions that happened to work.
    """
    return ItemOutcome(
        image_id=item.image_id,
        prompt=item.prompt,
        groundtruth=item.groundtruth,
        answer="",
        correct=False,
        steps=0,
        confidence=0.0,
        converged=False,
        tokens=0,
        error=error,
    )


async def run(
    task: str,
    mode: str,
    *,
    model: str = DEFAULT_MODEL,
    limit: int = 50,
    offset: int = 0,
    max_steps: int = 4,
    cache_dir: str | None = None,
    vlm: VLMPort | None = None,
    progress: bool = True,
    rpm: int = 0,
) -> RunReport:
    """Run one task in one mode.

    Args:
        task: A key of :data:`~benchmarks.blindtest.dataset.TASKS`.
        mode: ``"baseline"`` or ``"saccade"``.
        model: Pydantic AI model string. Ignored when ``vlm`` is given.
        limit: How many items to run.
        offset: Where to start, for resuming.
        max_steps: Step budget in saccade mode.
        cache_dir: Where to cache responses.
        vlm: Inject a model directly — used by the tests to run the whole
            pipeline on FakeVLM without touching the network.
        progress: Print per-item progress.
        rpm: Requests-per-minute budget. 0 disables pacing, which is what
            the tests want; live runs should pass the provider's real limit.
    """
    if mode not in ("baseline", "saccade"):
        raise ValueError(f"mode must be 'baseline' or 'saccade', got {mode!r}")

    if vlm is None:
        from saccade.vlm.pydantic_ai import PydanticAIVLM

        vlm = PydanticAIVLM(model)

    cache = FileCache(cache_dir) if cache_dir else FileCache()
    items = load_task(task, limit=limit, offset=offset)
    if not items:
        raise RuntimeError(f"no items loaded for task {task!r}")

    agent = ActiveVisionAgent(vlm, cache=cache, max_steps=max_steps)

    limiter = RateLimiter(rpm)
    started = time.monotonic()
    outcomes: list[ItemOutcome] = []
    for index, item in enumerate(items, start=1):
        if mode == "baseline":
            outcome = await _cached_baseline(vlm, cache, item, limiter)
        else:
            outcome = await run_saccade(agent, item, limiter)
        outcomes.append(outcome)

        if progress:
            mark = "OK " if outcome.correct else "XX "
            detail = repr(outcome.answer[:60])
            if outcome.error:
                # Show the error itself: a run of silent failures otherwise
                # looks identical to a run of wrong answers.
                mark, detail = "ERR", outcome.error[:100]
            print(f"  [{index}/{len(items)}] {mark} {detail}", flush=True)

    elapsed = time.monotonic() - started
    correct = sum(1 for o in outcomes if o.correct)
    errors = sum(1 for o in outcomes if o.error)

    return RunReport(
        task=items[0].task,
        mode=mode,
        model=getattr(vlm, "model_id", model),
        n_items=len(outcomes),
        n_correct=correct,
        n_errors=errors,
        accuracy=correct / len(outcomes),
        total_tokens=sum(o.tokens for o in outcomes),
        mean_steps=sum(o.steps for o in outcomes) / len(outcomes),
        seconds=elapsed,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        outcomes=outcomes,
    )


async def _cached_baseline(
    vlm: VLMPort,
    cache: FileCache,
    item: BlindTestItem,
    limiter: RateLimiter | None = None,
) -> ItemOutcome:
    """Baseline with the same cache the loop uses, so reruns are free."""
    from saccade.vlm._cache import make_cache_key

    key = make_cache_key([item.image], item.prompt, vlm.model_id)
    hit = cache.get(key)
    if hit is not None:
        return ItemOutcome(
            image_id=item.image_id,
            prompt=item.prompt,
            groundtruth=item.groundtruth,
            answer=hit.text,
            correct=score(item.task, hit.text, item.groundtruth),
            steps=1,
            confidence=hit.confidence or 0.0,
            converged=True,
            tokens=0,
        )

    outcome = await run_baseline(vlm, item, limiter)
    if outcome.error is None:
        from saccade.models import VLMResponse

        cache.set(key, VLMResponse(text=outcome.answer, model_id=vlm.model_id))
    return outcome


def save(report: RunReport, directory: Path = RESULTS_DIR) -> Path:
    """Write a run to JSON. Every published number must be checkable."""
    directory.mkdir(parents=True, exist_ok=True)
    slug = report.task.lower().replace(" ", "_")
    path = directory / f"{slug}__{report.mode}__{report.timestamp.replace(':', '-')}.json"
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="touching_circles", choices=sorted(TASKS))
    parser.add_argument("--mode", default="baseline", choices=["baseline", "saccade"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_RPM,
        help=f"requests per minute (default {DEFAULT_RPM}, Gemini's free tier); 0 disables pacing",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    if not os.environ.get("GOOGLE_API_KEY") and args.model.startswith("google:"):
        print("GOOGLE_API_KEY is not set. Put it in .env — see .env.example.", file=sys.stderr)
        return 2

    report = asyncio.run(
        run(
            args.task,
            args.mode,
            model=args.model,
            limit=args.limit,
            offset=args.offset,
            max_steps=args.max_steps,
            cache_dir=args.cache_dir,
            rpm=args.rpm,
        )
    )

    print()
    print(report.summary())

    if not args.no_save:
        print(f"\nwritten to {save(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
