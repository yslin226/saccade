"""Tests for the public data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from saccade import (
    BBox,
    EvidenceStep,
    InvestigationResult,
    Observation,
    Verification,
    Viewport,
    VLMResponse,
)


class TestBBox:
    def test_construct_and_serialise(self) -> None:
        box = BBox(x=10, y=20, w=100, h=50)
        assert box.model_dump_json() == '{"x":10,"y":20,"w":100,"h":50}'

    def test_derived_geometry(self) -> None:
        box = BBox(x=10, y=20, w=100, h=50)
        assert box.right == 110
        assert box.bottom == 70
        assert box.area == 5000
        assert box.center == (60.0, 45.0)

    def test_zero_width_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BBox(x=0, y=0, w=0, h=10)

    def test_negative_origin_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BBox(x=-1, y=0, w=10, h=10)

    def test_frozen(self) -> None:
        box = BBox(x=0, y=0, w=10, h=10)
        with pytest.raises(ValidationError):
            box.x = 5  # type: ignore[misc]


class TestViewport:
    def test_full_covers_source(self) -> None:
        viewport = Viewport.full((640, 480))
        assert viewport.zoom == 1.0
        assert viewport.covers_full_image is True

    def test_serialise(self) -> None:
        viewport = Viewport(bbox=BBox(x=0, y=0, w=64, h=48), zoom=2.0, source_size=(640, 480))
        assert '"zoom":2.0' in viewport.model_dump_json()
        assert viewport.covers_full_image is False

    def test_bbox_beyond_source_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extends past source_size"):
            Viewport(bbox=BBox(x=600, y=0, w=100, h=10), source_size=(640, 480))

    def test_bbox_exactly_at_edge_allowed(self) -> None:
        viewport = Viewport(bbox=BBox(x=540, y=0, w=100, h=10), source_size=(640, 480))
        assert viewport.bbox.right == 640

    def test_zero_zoom_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Viewport(bbox=BBox(x=0, y=0, w=10, h=10), zoom=0.0, source_size=(640, 480))

    def test_zero_source_size_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_size must be positive"):
            Viewport(bbox=BBox(x=0, y=0, w=10, h=10), source_size=(0, 480))

    def test_negative_source_size_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_size must be positive"):
            Viewport(bbox=BBox(x=0, y=0, w=10, h=10), source_size=(640, -1))


class TestObservation:
    def test_construct_and_serialise(self) -> None:
        observation = Observation(statement="two circles overlap", self_confidence=0.7)
        assert "two circles overlap" in observation.model_dump_json()

    def test_confidence_optional(self) -> None:
        assert Observation(statement="unclear").self_confidence is None

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Observation(statement="certain", self_confidence=1.5)


class TestVerification:
    def test_passed_verification(self) -> None:
        verification = Verification(
            passed=True,
            method="circles_overlap",
            computed={"centre_distance": 47.0, "radius_sum": 52.0},
        )
        assert "circles_overlap" in verification.model_dump_json()

    def test_conflict_recorded_on_failure(self) -> None:
        verification = Verification(
            passed=False,
            method="detect_objects",
            computed={"detected": 2},
            conflict="VLM said 3 people, detector found 2",
        )
        assert verification.conflict is not None

    def test_passed_with_conflict_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot carry a conflict"):
            Verification(passed=True, method="x", conflict="but they disagreed")


class TestEvidenceStep:
    def test_construct_and_serialise(self) -> None:
        step = EvidenceStep(
            index=0,
            action_name="crop",
            viewport=Viewport.full((100, 100)),
            observation=Observation(statement="a circle"),
            verification=Verification(passed=True, method="count_contours"),
            image_ref="cache/abc123.png",
        )
        assert step.model_dump_json()
        assert step.verification is not None

    def test_verification_optional(self) -> None:
        step = EvidenceStep(
            index=1,
            action_name="zoom",
            viewport=Viewport.full((10, 10)),
            observation=Observation(statement="still unclear"),
        )
        assert step.verification is None

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceStep(
                index=-1,
                action_name="crop",
                viewport=Viewport.full((10, 10)),
                observation=Observation(statement="x"),
            )


class TestInvestigationResult:
    def test_construct_and_serialise(self) -> None:
        result = InvestigationResult(answer="yes", confidence=0.9, converged=True)
        assert result.model_dump_json()
        assert result.steps_taken == 0

    def test_unconverged_is_a_valid_result_not_an_error(self) -> None:
        """Running out of steps is an ordinary outcome (spec 4.4)."""
        step = EvidenceStep(
            index=0,
            action_name="crop",
            viewport=Viewport.full((10, 10)),
            observation=Observation(statement="cannot tell"),
        )
        result = InvestigationResult(
            answer="undetermined",
            confidence=0.3,
            converged=False,
            evidence_chain=[step],
        )
        assert result.converged is False
        assert result.steps_taken == 1

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InvestigationResult(answer="yes", confidence=1.2, converged=True)

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            InvestigationResult(answer="yes", confidence=0.5, converged=True, total_tokens=-1)


class TestVLMResponse:
    def test_construct_and_serialise(self) -> None:
        response = VLMResponse(
            text="two circles",
            confidence=0.8,
            raw={"provider": "fake"},
            tokens_used=42,
            model_id="google:gemini-2.5-flash",
        )
        assert "gemini" in response.model_dump_json()

    def test_defaults(self) -> None:
        response = VLMResponse(text="hi")
        assert response.tokens_used == 0
        assert response.raw == {}
        assert response.structured is None

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VLMResponse(text="hi", confidence=-0.1)
