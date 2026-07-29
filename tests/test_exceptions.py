"""Tests for the exception hierarchy.

The contract callers rely on: one except clause catches everything Saccade
raises.
"""

from __future__ import annotations

import pytest

from saccade import SaccadeError, ToolError, VLMError


class TestHierarchy:
    def test_vlm_error_is_a_saccade_error(self) -> None:
        assert issubclass(VLMError, SaccadeError)

    def test_tool_error_is_a_saccade_error(self) -> None:
        assert issubclass(ToolError, SaccadeError)

    def test_saccade_error_is_an_exception(self) -> None:
        assert issubclass(SaccadeError, Exception)

    def test_subclasses_are_distinct(self) -> None:
        assert not issubclass(VLMError, ToolError)
        assert not issubclass(ToolError, VLMError)


class TestCatching:
    def test_base_catches_vlm_error(self) -> None:
        with pytest.raises(SaccadeError):
            raise VLMError("quota exceeded")

    def test_base_catches_tool_error(self) -> None:
        with pytest.raises(SaccadeError):
            raise ToolError("detector crashed")

    def test_message_is_preserved(self) -> None:
        with pytest.raises(SaccadeError, match="quota exceeded"):
            raise VLMError("quota exceeded")

    def test_specific_clause_does_not_catch_the_sibling(self) -> None:
        with pytest.raises(ToolError):
            try:
                raise ToolError("boom")
            except VLMError:  # pragma: no cover - must not match
                pytest.fail("VLMError should not catch ToolError")
