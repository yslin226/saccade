"""Public data models.

These types form the published contract of the library: they cross the
boundary between Saccade and its callers, and they are what ends up in an
evidence chain.

This module deliberately imports nothing but ``pydantic`` and ``typing``.
Keeping it free of PIL, OpenCV and NumPy means the models stay cheap to
import and can be serialised anywhere without dragging in the imaging
stack.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BBox",
    "EvidenceStep",
    "InvestigationResult",
    "Observation",
    "VLMResponse",
    "Verification",
    "Viewport",
]


class BBox(BaseModel):
    """An axis-aligned rectangle in pixel coordinates.

    ``x``/``y`` are the top-left corner, matching PIL's coordinate system.
    """

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


class Viewport(BaseModel):
    """What the agent is currently looking at.

    A viewport is a region of the source image plus the magnification applied
    to it. ``source_size`` is kept so a viewport can always be mapped back to
    the original image, which is what makes the evidence chain replayable.
    """

    model_config = ConfigDict(frozen=True)

    bbox: BBox
    zoom: float = Field(default=1.0, gt=0)
    source_size: tuple[int, int]

    @model_validator(mode="after")
    def _bbox_within_source(self) -> Viewport:
        width, height = self.source_size
        if width <= 0 or height <= 0:
            raise ValueError(f"source_size must be positive, got {self.source_size}")
        if self.bbox.right > width or self.bbox.bottom > height:
            raise ValueError(
                f"bbox {self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h} "
                f"extends past source_size {self.source_size}"
            )
        return self

    @classmethod
    def full(cls, source_size: tuple[int, int]) -> Viewport:
        """The whole image at 1x — where an investigation starts."""
        width, height = source_size
        return cls(bbox=BBox(x=0, y=0, w=width, h=height), source_size=source_size)

    @property
    def covers_full_image(self) -> bool:
        return self.bbox == BBox(x=0, y=0, w=self.source_size[0], h=self.source_size[1])


class Observation(BaseModel):
    """What the VLM said after looking at a viewport.

    ``self_confidence`` is the model's own estimate and is treated as a claim,
    not as evidence. Only a :class:`Verification` can raise the agent's real
    confidence.
    """

    model_config = ConfigDict(frozen=True)

    statement: str
    self_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Verification(BaseModel):
    """The result of confronting an observation with a computed measurement.

    ``method`` names the tool that produced the numbers, and ``computed``
    holds them. When ``passed`` is False, ``conflict`` explains what
    disagreed — that text is what the planner uses to pick a different angle.

    ``verdict_key`` names the entry of ``computed`` the tool declared as its
    answer. It is what makes an overruling possible: knowing a statement is
    contradicted is not enough to replace it, since ``computed`` also holds
    diagnostic figures that are not answers to anything.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    method: str
    computed: dict[str, Any] = Field(default_factory=dict)
    conflict: str | None = None
    verdict_key: str | None = None

    @model_validator(mode="after")
    def _conflict_requires_failure(self) -> Verification:
        if self.passed and self.conflict is not None:
            raise ValueError("a passed verification cannot carry a conflict")
        return self

    @model_validator(mode="after")
    def _verdict_key_must_be_computed(self) -> Verification:
        if self.verdict_key is not None and self.verdict_key not in self.computed:
            raise ValueError(f"verdict_key {self.verdict_key!r} is not among the computed values")
        return self

    @property
    def verdict(self) -> Any | None:
        """The measured value that settles the question, when a tool named one.

        ``None`` means no tool claimed to answer — either none ran, or the
        ones that did reported only context. A caller must not read that as
        a measurement of ``False``: use :attr:`verdict_key` to tell the two
        apart.
        """
        return self.computed.get(self.verdict_key) if self.verdict_key is not None else None


class EvidenceStep(BaseModel):
    """One iteration of the Perceive-Verify loop, kept for audit.

    ``image_ref`` is a reference (path, cache key, URL) rather than image
    bytes: the models stay serialisable and the library stays out of the
    business of deciding where images live.

    ``reason`` records why this region was chosen. An auditor needs to see
    not just where the agent looked but what sent it there — a chain showing
    only coordinates cannot be argued with.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    action_name: str
    viewport: Viewport
    observation: Observation
    verification: Verification | None = None
    reason: str | None = None
    image_ref: str | None = None


class InvestigationResult(BaseModel):
    """The outcome of an investigation.

    ``converged=False`` is a normal result, not an error: it means the agent
    hit ``max_steps`` without reaching the confidence threshold. The evidence
    chain is populated either way so the caller can see how far it got.

    ``structured`` holds the parsed value when ``expect=`` was passed to
    :meth:`~saccade.agent.ActiveVisionAgent.investigate`. It stays ``Any``
    for M0; making the class generic is an M1 decision.
    """

    model_config = ConfigDict(frozen=True)

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    converged: bool
    evidence_chain: list[EvidenceStep] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    structured: Any | None = None

    @property
    def steps_taken(self) -> int:
        return len(self.evidence_chain)


class VLMResponse(BaseModel):
    """A single reply from a VLM, as returned by :class:`~saccade.ports.VLMPort`.

    ``raw`` keeps the provider payload so an adapter can round-trip a response
    through the cache without losing information.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    raw: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = Field(default=0, ge=0)
    model_id: str = ""
    structured: Any | None = None
