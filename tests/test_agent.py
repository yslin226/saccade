"""Tests for the ActiveVisionAgent signature.

The loop is M1 work. What M0 locks down is the shape of the public entry
point — specifically that everything after ``vlm`` is keyword-only (rule 5),
so M1 can add parameters without breaking anyone's calls.
"""

from __future__ import annotations

import inspect

import pytest

from saccade import ActiveVisionAgent
from saccade.vlm import FakeVLM


class TestNotYetImplemented:
    def test_constructor_says_which_milestone(self) -> None:
        with pytest.raises(NotImplementedError, match="M1"):
            ActiveVisionAgent(FakeVLM(["yes"]))


class TestKeywordOnlyContract:
    """CLAUDE.md rule 5: new parameters must not break positional callers."""

    def test_init_takes_only_vlm_positionally(self) -> None:
        params = list(inspect.signature(ActiveVisionAgent.__init__).parameters.values())
        positional = [
            p.name
            for p in params
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name != "self"
        ]
        assert positional == ["vlm"]

        keyword_only = {p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
        assert keyword_only == {
            "cache",
            "max_steps",
            "confidence_threshold",
            "tools",
            "on_step",
        }

    @pytest.mark.parametrize("method", ["investigate", "investigate_async"])
    def test_expect_is_keyword_only(self, method: str) -> None:
        params = inspect.signature(getattr(ActiveVisionAgent, method)).parameters
        assert params["expect"].kind is inspect.Parameter.KEYWORD_ONLY
        positional = [
            name
            for name, p in params.items()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and name != "self"
        ]
        assert positional == ["image", "question"]

    def test_documented_defaults_are_the_spec_defaults(self) -> None:
        params = inspect.signature(ActiveVisionAgent.__init__).parameters
        assert params["max_steps"].default == 8
        assert params["confidence_threshold"].default == 0.8
        assert params["cache"].default is None

    def test_investigate_async_is_a_coroutine_function(self) -> None:
        """async is the core; the sync method wraps it (spec 4.3)."""
        assert inspect.iscoroutinefunction(ActiveVisionAgent.investigate_async)
        assert not inspect.iscoroutinefunction(ActiveVisionAgent.investigate)

    def test_register_tool_is_public(self) -> None:
        assert callable(ActiveVisionAgent.register_tool)
