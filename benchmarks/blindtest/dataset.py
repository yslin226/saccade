"""Loading BlindTest items from the published dataset.

Source: https://huggingface.co/datasets/XAI/vlmsareblind (8016 rows, one
`valid` split). Fields are ``task``, ``image``, ``prompt``, ``groundtruth``
and ``metadata``.

The benchmark lives outside ``src/`` on purpose: it is not part of the
published library, and users installing saccade-vision should not receive
it.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger("benchmarks.blindtest")

__all__ = [
    "DEFAULT_CACHE_DIR",
    "MEASURABLE",
    "TASKS",
    "BlindTestItem",
    "load_task",
    "stratified_sample",
]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "data"

_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_DATASET = "XAI/vlmsareblind"

# The dataset server caps a page at 100 rows.
_PAGE = 100

_MAX_RETRIES = 6
_INITIAL_BACKOFF = 2.0
_MAX_BACKOFF = 30.0

# The BlindTest tasks, by the short name the runner takes.
#
# Two groups, and the distinction turned out to matter more than expected.
# The first two have a closed-form referee, and on those a tool alone reaches
# 98-99% — which leaves an agent nothing to add, and on one task actively
# hurts. The rest cannot be settled by geometry: reading which letter is
# circled, or following a coloured path, needs recognition. Those are where
# looking again might earn its keep, so they are the fair test of the idea.
TASKS = {
    # Measurable: geometry settles them.
    "touching_circles": "Touching Circles",
    "line_intersections": "Line Plot Intersections",
    # Not measurable: recognition, not measurement.
    "circled_letter": "Circled Letter",
    "olympic_circles": "Olympic Counting - Circles",
    "olympic_pentagons": "Olympic Counting - Pentagons",
    "nested_squares": "Nested Squares",
    "grid_blank": "Counting Grid - Blank Grids",
    "grid_words": "Counting Grid - Word Grids",
    "subway": "Subway Connections",
}

# Tasks with a computable referee. Everything else runs without one, which
# is stated rather than hidden: "saccade-tools" on an unmeasurable task is
# the same as "saccade", and the report should not imply otherwise.
MEASURABLE = frozenset({"Touching Circles", "Line Plot Intersections"})


@dataclass(frozen=True)
class BlindTestItem:
    """One benchmark question."""

    task: str
    image: Image.Image
    prompt: str
    groundtruth: str
    metadata: dict[str, Any]

    @property
    def image_id(self) -> str:
        """A stable identifier for this item.

        Not every task carries an ``image_id`` — Touching Circles records
        geometry instead — so fall back to a hash of the metadata, which is
        distinct per item and stable across runs.
        """
        recorded = self.metadata.get("image_id")
        if recorded:
            return str(recorded)

        payload = json.dumps(self.metadata, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def load_task(
    task: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    timeout: float = 120.0,
    cache_dir: Path | None = None,
) -> list[BlindTestItem]:
    """Fetch items for one task, caching them locally.

    The dataset server rate-limits anonymous callers hard, so items are
    written to disk on first fetch and read from there afterwards. Without
    that, every rerun re-downloads the same images and eventually gets a 429
    partway through — leaving an incomplete benchmark rather than a slow one.

    Args:
        task: A key of :data:`TASKS`, or a dataset task name verbatim.
        limit: Maximum items to return. None fetches everything available.
        offset: Where to start, for resuming an interrupted run.
        timeout: Per-request timeout in seconds.
        cache_dir: Where to keep downloaded items.

    Returns:
        Items in dataset order. Order is stable, so a run over the first N
        items is reproducible.
    """
    task_name = TASKS.get(task, task)
    store = _Store(cache_dir or DEFAULT_CACHE_DIR, task_name)

    cached = store.load(limit, offset)
    if cached is not None:
        return cached

    items: list[BlindTestItem] = []
    try:
        items = _download(task_name, limit, offset, timeout)
    except httpx.HTTPStatusError:
        # The dataset server refused. If anything is cached, run on that
        # rather than not running: a smaller sample is a stated limitation,
        # while no run at all answers nothing.
        partial = store.load(None, offset)
        if not partial:
            raise
        logger.warning(
            "dataset server refused; using %d cached item(s) for %s", len(partial), task_name
        )
        return partial

    store.save(items, offset)
    return items


def _download(
    task_name: str, limit: int | None, offset: int, timeout: float
) -> list[BlindTestItem]:
    """Fetch items from the dataset server."""
    items: list[BlindTestItem] = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        cursor = 0
        total = _total(client)
        seen = 0

        while limit is None or len(items) < limit:
            # Always request a full page. Asking for exactly the number of
            # items still wanted means one request per item once the tail is
            # close, which is what got this rate-limited.
            batch = _fetch_page(client, cursor, _PAGE)
            if not batch:
                break

            for row in batch:
                if row["task"] != task_name:
                    continue
                if seen < offset:
                    seen += 1
                    continue
                items.append(_to_item(client, row))
                if limit is not None and len(items) >= limit:
                    break

            cursor += len(batch)
            if cursor >= total:
                break

    return items


def stratified_sample(items: list[BlindTestItem], n: int) -> list[BlindTestItem]:
    """Take ``n`` items spread evenly across the dataset, not the first ``n``.

    The dataset is ordered by difficulty parameter, so the leading items are
    all the same kind of question. A run over the first ten Touching Circles
    items gets seven identical "clearly overlapping" cases and a model that
    always answered Yes would score 70% — a number that says nothing.

    Taking every k-th item instead covers the range, including the
    near-tangent cases where the benchmark is actually hard. Deterministic,
    so the sample is the same on every run.
    """
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []

    stride = len(items) / n
    return [items[int(i * stride)] for i in range(n)]


class _Store:
    """Keeps downloaded items on disk, one PNG plus one JSON index per task."""

    def __init__(self, directory: Path, task_name: str) -> None:
        slug = task_name.lower().replace(" ", "_")
        self.directory = Path(directory) / slug
        self.index = self.directory / "index.json"

    def load(self, limit: int | None, offset: int) -> list[BlindTestItem] | None:
        """Return cached items, or None when the cache cannot satisfy the request."""
        if not self.index.is_file():
            return None

        try:
            records = json.loads(self.index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

        available = records[offset:] if offset else records
        if limit is not None and len(available) < limit:
            return None

        wanted = available[:limit] if limit is not None else available
        items: list[BlindTestItem] = []
        for record in wanted:
            path = self.directory / record["file"]
            if not path.is_file():
                return None
            image = Image.open(path)
            image.load()
            items.append(
                BlindTestItem(
                    task=record["task"],
                    image=image.convert("RGB"),
                    prompt=record["prompt"],
                    groundtruth=record["groundtruth"],
                    metadata=record["metadata"],
                )
            )
        return items

    def save(self, items: list[BlindTestItem], offset: int) -> None:
        """Write items to disk, extending whatever is already stored."""
        if not items:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        if self.index.is_file():
            try:
                records = json.loads(self.index.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                records = []

        for position, item in enumerate(items, start=offset):
            filename = f"{position:05d}.png"
            item.image.save(self.directory / filename, format="PNG")
            record = {
                "file": filename,
                "task": item.task,
                "prompt": item.prompt,
                "groundtruth": item.groundtruth,
                "metadata": item.metadata,
            }
            if position < len(records):
                records[position] = record
            else:
                records.append(record)

        self.index.write_text(json.dumps(records, indent=2), encoding="utf-8")


def _fetch_page(client: httpx.Client, offset: int, length: int) -> list[dict[str, Any]]:
    response = _get(
        client,
        _ROWS_URL,
        {
            "dataset": _DATASET,
            "config": "default",
            "split": "valid",
            "offset": offset,
            "length": max(1, min(_PAGE, length)),
        },
    )
    return [entry["row"] for entry in response.json()["rows"]]


def _get(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """GET with backoff on rate limits.

    The dataset server rate-limits anonymous callers, and a benchmark that
    dies partway through leaves an incomplete run rather than a slow one.
    """
    delay = _INITIAL_BACKOFF
    last: httpx.HTTPStatusError | None = None

    for _ in range(_MAX_RETRIES):
        response = client.get(url, params=params)
        if response.status_code not in (429, 502, 503, 504):
            response.raise_for_status()
            return response

        last = httpx.HTTPStatusError(
            f"{response.status_code} from {url}", request=response.request, response=response
        )
        retry_after = response.headers.get("retry-after")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
        time.sleep(wait)
        delay = min(delay * 2, _MAX_BACKOFF)

    assert last is not None
    raise last


_TOTAL: int | None = None


def _total(client: httpx.Client) -> int:
    global _TOTAL
    if _TOTAL is None:
        response = _get(
            client,
            _ROWS_URL,
            {
                "dataset": _DATASET,
                "config": "default",
                "split": "valid",
                "offset": 0,
                "length": 1,
            },
        )
        _TOTAL = int(response.json()["num_rows_total"])
    return _TOTAL


def _to_item(client: httpx.Client, row: dict[str, Any]) -> BlindTestItem:
    image = _fetch_image(client, row["image"]["src"])
    metadata = row.get("metadata") or "{}"
    return BlindTestItem(
        task=row["task"],
        image=image,
        prompt=row["prompt"],
        groundtruth=str(row["groundtruth"]),
        metadata=json.loads(metadata) if isinstance(metadata, str) else metadata,
    )


def _fetch_image(client: httpx.Client, url: str) -> Image.Image:
    response = _get(client, url)
    image = Image.open(io.BytesIO(response.content))
    image.load()
    return image.convert("RGB")
