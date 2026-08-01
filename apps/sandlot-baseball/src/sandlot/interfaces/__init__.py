"""How a person drives this: the CLI.

An interface adapter translates between the outside form of a request and
what a use case expects. It holds no rules and makes no decisions about
baseball — if a change here needs a change to a metric, the rule has leaked
out of ``domain``.

FastAPI lands here too, at M6. The CLI is enough to satisfy M3, and an HTTP
layer written before anything calls it would be shaped by a guess.
"""

from __future__ import annotations

__all__: list[str] = []
