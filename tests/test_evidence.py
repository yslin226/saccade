"""Tests for the evidence chain."""

from __future__ import annotations

from saccade._evidence import EvidenceChain
from saccade.models import BBox, Observation, Verification, Viewport

SIZE = (100, 100)


def view(x: int = 0, y: int = 0, w: int = 100, h: int = 100) -> Viewport:
    return Viewport(bbox=BBox(x=x, y=y, w=w, h=h), source_size=SIZE)


def passed(method: str = "measurement") -> Verification:
    return Verification(passed=True, method=method, computed={"value": 1})


def conflicted(text: str = "said 3, measured 2") -> Verification:
    return Verification(passed=False, method="measurement", computed={}, conflict=text)


class TestAppending:
    def test_indices_are_assigned_in_order(self) -> None:
        chain = EvidenceChain()
        for _ in range(3):
            chain.add(
                action_name="look",
                viewport=view(),
                observation=Observation(statement="x"),
            )
        assert [step.index for step in chain.steps] == [0, 1, 2]

    def test_add_returns_the_step(self) -> None:
        chain = EvidenceChain()
        step = chain.add(
            action_name="zoom", viewport=view(), observation=Observation(statement="x")
        )
        assert step.action_name == "zoom"
        assert step.index == 0

    def test_steps_property_returns_a_copy(self) -> None:
        chain = EvidenceChain()
        chain.add(action_name="look", viewport=view(), observation=Observation(statement="x"))
        chain.steps.clear()
        assert len(chain) == 1

    def test_length_and_truthiness(self) -> None:
        chain = EvidenceChain()
        assert not chain
        assert len(chain) == 0
        chain.add(action_name="look", viewport=view(), observation=Observation(statement="x"))
        assert chain
        assert len(chain) == 1

    def test_image_ref_is_stored_as_given(self) -> None:
        chain = EvidenceChain()
        step = chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="x"),
            image_ref="cache/abc123",
        )
        assert step.image_ref == "cache/abc123"


class TestFiltering:
    def test_verified_steps_are_isolated(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look", viewport=view(), observation=Observation(statement="unverified")
        )
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="verified"),
            verification=passed(),
        )
        assert len(chain.verified_steps) == 1
        assert chain.verified_steps[0].observation.statement == "verified"

    def test_conflicts_are_isolated(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="agreed"),
            verification=passed(),
        )
        chain.add(
            action_name="zoom",
            viewport=view(),
            observation=Observation(statement="disputed"),
            verification=conflicted(),
        )
        assert len(chain.conflicts) == 1
        assert chain.conflicts[0].observation.statement == "disputed"

    def test_a_failed_verification_is_not_a_verified_step(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="x"),
            verification=conflicted(),
        )
        assert chain.verified_steps == []


class TestBestStatement:
    def test_none_when_empty(self) -> None:
        assert EvidenceChain().best_statement() is None

    def test_a_verified_statement_beats_an_unverified_one(self) -> None:
        """Even when the unverified one came later and sounded surer."""
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="measured", self_confidence=0.2),
            verification=passed(),
        )
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="just a guess", self_confidence=0.99),
        )
        assert chain.best_statement() == "measured"

    def test_the_latest_verified_statement_wins(self) -> None:
        chain = EvidenceChain()
        for text in ("first", "second"):
            chain.add(
                action_name="look",
                viewport=view(),
                observation=Observation(statement=text),
                verification=passed(),
            )
        assert chain.best_statement() == "second"

    def test_falls_back_to_the_latest_unverified_statement(self) -> None:
        chain = EvidenceChain()
        chain.add(action_name="look", viewport=view(), observation=Observation(statement="a"))
        chain.add(action_name="look", viewport=view(), observation=Observation(statement="b"))
        assert chain.best_statement() == "b"
