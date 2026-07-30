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

from benchmarks.blindtest.dataset import (
    TASKS,
    BlindTestItem,
    load_task,
    stratified_sample,
)
from benchmarks.blindtest.models import build_vlm, default_rpm
from benchmarks.blindtest.scoring import score
from benchmarks.blindtest.tools import circle_tool, line_tool
from saccade import ActiveVisionAgent, Tool, VLMError
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

# The three-way comparison M2 exists to make. "saccade" without tools is the
# control: it isolates what extra looks are worth on their own, so any gain
# from "saccade-tools" can be attributed to verification rather than to
# simply asking the model more times.
MODES = ("baseline", "saccade", "saccade-tools")


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
    """What happened on one question.

    The verification counts are what distinguish a loop that checked itself
    from one that merely looked several times. M1 reported a respectable
    accuracy while ``verified`` would have been 0 on every item.
    """

    image_id: str
    prompt: str
    groundtruth: str
    answer: str
    correct: bool
    steps: int
    confidence: float
    converged: bool
    tokens: int
    verified: int = 0
    conflicts: int = 0
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
    n_converged: int = 0
    n_verified_steps: int = 0
    n_conflicts: int = 0
    outcomes: list[ItemOutcome] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.task} / {self.mode} / {self.model}\n"
            f"  accuracy  : {self.accuracy:.1%}  ({self.n_correct}/{self.n_items})\n"
            f"  errors    : {self.n_errors}\n"
            f"  converged : {self.n_converged}/{self.n_items}\n"
            f"  verified  : {self.n_verified_steps} step(s) confirmed by measurement\n"
            f"  conflicts : {self.n_conflicts} time(s) the tool overruled the model\n"
            f"  tokens    : {self.total_tokens}\n"
            f"  steps     : {self.mean_steps:.2f} mean\n"
            f"  elapsed   : {self.seconds:.1f}s"
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
        verified=sum(1 for s in result.evidence_chain if s.verification and s.verification.passed),
        conflicts=sum(
            1 for s in result.evidence_chain if s.verification and s.verification.conflict
        ),
    )


def _referee_for(item: BlindTestItem) -> Tool | None:
    """Build the measurement tool for one item, if the task has one.

    Returns None for tasks with no computable referee. Those still run — the
    loop just has nothing to check itself against, which is worth measuring
    separately rather than pretending the tool exists.
    """
    if item.task == "Touching Circles":
        # The dataset answers Yes to "are they touching" and No to "are they
        # overlapping" for the same tangent image, so the referee cannot be
        # configured once per run: a sample mixes both phrasings.
        return circle_tool(tangent_counts="touching" in item.prompt)

    if item.task == "Line Plot Intersections":
        return line_tool()

    return None


def _report(progress: bool, index: int, total: int, outcome: ItemOutcome) -> None:
    if not progress:
        return
    mark = "OK " if outcome.correct else "XX "
    detail = repr(outcome.answer[:60])
    if outcome.error:
        # Show the error itself: a run of silent failures otherwise looks
        # identical to a run of wrong answers.
        mark, detail = "ERR", outcome.error[:100]
    print(f"  [{index}/{total}] {mark} {detail}", flush=True)


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
    stratify: bool = False,
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
        stratify: Spread the sample across the dataset instead of taking the
            first ``limit`` items, which would all be the same difficulty.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")

    if vlm is None:
        vlm = build_vlm(model)

    cache = FileCache(cache_dir) if cache_dir else FileCache()

    if stratify:
        # Fetch a wide slice, then spread the sample across it. Taking the
        # first N items gets N nearly identical questions — see
        # stratified_sample for why that produces a meaningless number.
        pool = load_task(task, limit=max(limit * 8, 100), offset=offset)
        items = stratified_sample(pool, limit)
    else:
        items = load_task(task, limit=limit, offset=offset)

    if not items:
        raise RuntimeError(f"no items loaded for task {task!r}")

    limiter = RateLimiter(rpm)
    started = time.monotonic()
    outcomes: list[ItemOutcome] = []
    for index, item in enumerate(items, start=1):
        if mode == "baseline":
            outcome = await _cached_baseline(vlm, cache, item, limiter)
            outcomes.append(outcome)
            _report(progress, index, len(items), outcome)
            continue

        # A fresh agent per item: in tools mode the referee is configured
        # from the question, and the sample mixes both phrasings.
        agent = ActiveVisionAgent(vlm, cache=cache, max_steps=max_steps)
        if mode == "saccade-tools":
            referee = _referee_for(item)
            if referee is not None:
                agent.register_tool(referee)

        outcome = await run_saccade(agent, item, limiter)
        outcomes.append(outcome)
        _report(progress, index, len(items), outcome)

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
        n_converged=sum(1 for o in outcomes if o.converged),
        n_verified_steps=sum(o.verified for o in outcomes),
        n_conflicts=sum(o.conflicts for o in outcomes),
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
    parser.add_argument("--mode", default="baseline", choices=list(MODES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--rpm",
        type=int,
        default=0,
        help="requests per minute; 0 uses the provider default (Gemini free tier is 20)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="take the first N items instead of spreading the sample across difficulties",
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
            rpm=args.rpm or default_rpm(args.model),
            stratify=not args.sequential,
        )
    )

    print()
    print(report.summary())

    if not args.no_save:
        print(f"\nwritten to {save(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
