"""Exception hierarchy.

Everything Saccade raises inherits from :class:`SaccadeError`, so callers can
catch the whole library with one clause.

Note what is *not* here: failing to converge is not an exception. When the
agent runs out of steps it returns
``InvestigationResult(converged=False, ...)`` with the evidence collected so
far. That is an ordinary outcome the benchmark needs to count, and turning it
into an exception would throw the evidence away.
"""

from __future__ import annotations

__all__ = ["SaccadeError", "ToolError", "VLMError"]


class SaccadeError(Exception):
    """Base class for every Saccade error."""


class VLMError(SaccadeError):
    """A VLM call failed — network, authentication, quota, or a malformed reply."""


class ToolError(SaccadeError):
    """A tool raised while executing, or returned something unusable."""
