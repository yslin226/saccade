"""The flow: what happens in what order, and what it needs to be given.

Use cases orchestrate domain objects. They must not contain domain knowledge
of their own (spec 5.1) — a use case that computed an angle would have put a
rule somewhere the domain tests do not reach.

Everything external arrives as a Port. That is what makes a use case
testable without a video file, a detector, or a disk.
"""

from __future__ import annotations

__all__: list[str] = []
