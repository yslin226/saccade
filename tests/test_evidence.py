"""Tests for the evidence chain."""

from __future__ import annotations

import pytest

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


class TestPartialViewsCannotAnswerWholeImageQuestions:
    """Regression from a live run on BlindTest's line-crossing task.

    Asked how many times two lines cross, a model looking at one quadrant
    answers correctly about the quadrant and wrongly about the picture.
    Taking that as the final answer dropped accuracy from 98.7% to 56.7%:
    62 items broken, none fixed, every failure an undercount.
    """

    def test_a_full_view_answer_beats_a_later_crop(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="{2}"),
        )
        chain.add(
            action_name="zoom",
            viewport=view(x=0, y=0, w=50, h=50),
            observation=Observation(statement="{1}"),
        )
        assert chain.best_statement() == "{2}"

    def test_a_verified_crop_does_not_beat_an_unverified_full_view(self) -> None:
        """Verification cannot rescue an answer to the wrong question."""
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="{2}"),
        )
        chain.add(
            action_name="zoom",
            viewport=view(x=50, y=50, w=50, h=50),
            observation=Observation(statement="{1}"),
            verification=passed(),
        )
        assert chain.best_statement() == "{2}"

    def test_a_verified_full_view_still_wins(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="{1}"),
        )
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="{2}"),
            verification=passed(),
        )
        assert chain.best_statement() == "{2}"

    def test_a_crop_is_used_when_nothing_else_answered(self) -> None:
        """Weak evidence beats none; the chain shows which viewport it came from."""
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="CANNOT TELL"),
        )
        chain.add(
            action_name="zoom",
            viewport=view(x=0, y=0, w=50, h=50),
            observation=Observation(statement="{1}"),
        )
        assert chain.best_statement() == "{1}"


class TestNonAnswersAreSkipped:
    """Regression from a live BlindTest run.

    Later steps are often magnified corners that no longer contain what the
    question is about. Taking the last statement blindly meant answering
    "I cannot tell from this view" after an earlier step had answered it.
    """

    def test_an_earlier_answer_beats_a_later_refusal(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="Yes, the circles touch."),
        )
        chain.add(
            action_name="zoom",
            viewport=view(),
            observation=Observation(statement="CANNOT TELL from this view."),
        )
        assert chain.best_statement() == "Yes, the circles touch."

    def test_a_verified_answer_beats_a_verified_refusal(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="No, they are apart."),
            verification=passed(),
        )
        chain.add(
            action_name="zoom",
            viewport=view(),
            observation=Observation(statement="Unable to determine."),
            verification=passed(),
        )
        assert chain.best_statement() == "No, they are apart."

    @pytest.mark.parametrize(
        "refusal",
        [
            "CANNOT TELL",
            "I cannot tell from this crop.",
            "It is not possible to tell.",
            "Unable to determine the answer.",
            "This view has insufficient detail.",
        ],
    )
    def test_recognised_refusal_forms(self, refusal: str) -> None:
        chain = EvidenceChain()
        chain.add(action_name="look", viewport=view(), observation=Observation(statement="Yes."))
        chain.add(action_name="zoom", viewport=view(), observation=Observation(statement=refusal))
        assert chain.best_statement() == "Yes."

    def test_all_refusals_still_report_something(self) -> None:
        """ "No answer" and "never looked" must remain distinguishable."""
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="CANNOT TELL"),
        )
        assert chain.best_statement() == "CANNOT TELL"

    def test_a_normal_answer_is_not_mistaken_for_a_refusal(self) -> None:
        chain = EvidenceChain()
        chain.add(
            action_name="look",
            viewport=view(),
            observation=Observation(statement="No, the circles do not touch."),
        )
        assert chain.best_statement() == "No, the circles do not touch."
