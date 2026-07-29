"""Tools — the actions the agent can take, and what they return.

The important field here is :attr:`ToolResult.is_measurement`. It decides
whether the verifier may use a result to confront a VLM statement. Results
that came from a VLM are just another opinion and must stay ``False``:
a blind witness cannot referee another blind witness.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PIL.Image import Image
from pydantic import BaseModel

__all__ = ["Tool", "ToolResult"]


@dataclass
class ToolResult:
    """What a tool produced.

    Args:
        value: The tool's output. Measurements should be plain numbers or
            dicts of numbers so the verifier can compare them.
        is_measurement: True only when ``value`` was computed, not described.
            See :mod:`saccade.tools` module docs.
        evidence_image: Optional rendering (an annotated crop, say) to attach
            to the evidence chain.
    """

    value: Any
    is_measurement: bool
    evidence_image: Image | None = None


@dataclass(frozen=True)
class Tool:
    """A callable the agent may invoke, with the schema of its arguments.

    ``params_schema`` is a Pydantic model rather than a hand-written JSON
    schema so the same class both validates calls and generates the schema
    handed to the model.
    """

    name: str
    description: str
    fn: Callable[..., ToolResult]
    params_schema: type[BaseModel]
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool.name must not be empty")
        if not self.description:
            raise ValueError(f"Tool {self.name!r} must have a description")

    def __call__(self, **kwargs: Any) -> ToolResult:
        """Validate ``kwargs`` against ``params_schema``, then run the tool."""
        params = self.params_schema(**kwargs)
        return self.fn(**params.model_dump())
