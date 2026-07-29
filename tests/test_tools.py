"""Tests for Tool and ToolResult."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

from saccade import Tool, ToolResult


class CircleParams(BaseModel):
    cx: float
    cy: float
    radius: float


def measure_circle(cx: float, cy: float, radius: float) -> ToolResult:
    return ToolResult(value={"area": 3.14159 * radius**2, "cx": cx, "cy": cy}, is_measurement=True)


class TestToolResult:
    def test_measurement_flag_is_explicit(self) -> None:
        assert ToolResult(value=42, is_measurement=True).is_measurement is True
        assert ToolResult(value="looks red", is_measurement=False).is_measurement is False

    def test_evidence_image_optional(self) -> None:
        assert ToolResult(value=1, is_measurement=True).evidence_image is None
        image = Image.new("RGB", (2, 2))
        assert (
            ToolResult(value=1, is_measurement=True, evidence_image=image).evidence_image is image
        )


class TestTool:
    def test_call_validates_and_dispatches(self) -> None:
        tool = Tool(
            name="measure_circle",
            description="Compute the area of a circle",
            fn=measure_circle,
            params_schema=CircleParams,
        )
        result = tool(cx=1.0, cy=2.0, radius=10.0)
        assert result.is_measurement is True
        assert result.value["area"] == pytest.approx(314.159)

    def test_invalid_arguments_rejected_before_the_function_runs(self) -> None:
        calls: list[int] = []

        def spy(cx: float, cy: float, radius: float) -> ToolResult:
            calls.append(1)
            return ToolResult(value=None, is_measurement=True)

        tool = Tool(name="spy", description="d", fn=spy, params_schema=CircleParams)
        with pytest.raises(ValidationError):
            tool(cx="not a number", cy=0.0, radius=1.0)
        assert calls == []

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            Tool(name="", description="d", fn=measure_circle, params_schema=CircleParams)

    def test_missing_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="must have a description"):
            Tool(name="t", description="", fn=measure_circle, params_schema=CircleParams)

    def test_frozen(self) -> None:
        tool = Tool(name="t", description="d", fn=measure_circle, params_schema=CircleParams)
        with pytest.raises(FrozenInstanceError):
            tool.name = "other"  # type: ignore[misc]
