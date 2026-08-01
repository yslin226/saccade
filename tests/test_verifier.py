"""Tests for the verifier.

The single most important test in this file is
``test_non_measurement_results_cannot_raise_confidence``. If that ever
passes wrongly, the project's whole claim collapses: the system would be
letting one model's opinion vouch for another's.
"""

from __future__ import annotations

import pytest

from saccade._verifier import (
    UNVERIFIED_CEILING,
    adjust_confidence,
    verify,
)
from saccade.models import Observation
from saccade.tools import ToolResult


def said(statement: str, confidence: float | None = None) -> Observation:
    return Observation(statement=statement, self_confidence=confidence)


def measured(value: object) -> ToolResult:
    return ToolResult(value=value, is_measurement=True)


def described(value: object) -> ToolResult:
    return ToolResult(value=value, is_measurement=False)


class TestOnlyMeasurementsMayJudge:
    """Spec 3.3: is_measurement decides who is allowed to referee."""

    def test_non_measurement_results_are_ignored(self) -> None:
        verification = verify(said("three people"), [described({"count": 2})])
        assert verification.method == "none"
        assert verification.passed is False
        assert verification.computed == {}

    def test_non_measurement_results_cannot_raise_confidence(self) -> None:
        """A VLM-produced result is a second opinion, never evidence."""
        verification = verify(said("two circles overlap"), [described({"overlapping": True})])
        raised = adjust_confidence(0.5, verification)
        assert raised <= UNVERIFIED_CEILING

    def test_a_measurement_alongside_a_description_still_judges(self) -> None:
        verification = verify(
            said("the circles overlap"),
            [described({"note": "looks close"}), measured({"overlap": True})],
        )
        assert verification.passed is True
        assert verification.method != "none"

    def test_empty_results_are_unverified(self) -> None:
        verification = verify(said("something"), [])
        assert verification.method == "none"
        assert verification.conflict is None


class TestAgreement:
    def test_boolean_agreement_passes(self) -> None:
        verification = verify(said("the circles overlap"), [measured({"overlap": True})])
        assert verification.passed is True
        assert verification.conflict is None
        assert verification.computed == {"overlap": True}

    def test_negated_claim_agrees_with_a_false_measurement(self) -> None:
        verification = verify(said("the circles do not overlap"), [measured({"overlap": False})])
        assert verification.passed is True

    def test_matching_count_passes(self) -> None:
        verification = verify(said("the lines cross 3 times"), [measured({"crossings": 3})])
        assert verification.passed is True

    def test_measurements_are_recorded_for_the_evidence_chain(self) -> None:
        verification = verify(
            said("centres are far apart"),
            [measured({"centre_distance": 47.0, "radius_sum": 52.0})],
        )
        assert verification.computed["centre_distance"] == 47.0
        assert verification.computed["radius_sum"] == 52.0


class TestConflict:
    def test_boolean_conflict_is_detected(self) -> None:
        verification = verify(said("the circles overlap"), [measured({"overlap": False})])
        assert verification.passed is False
        assert verification.conflict is not None
        assert "overlap" in verification.conflict

    def test_negated_claim_conflicts_with_a_true_measurement(self) -> None:
        verification = verify(said("the circles do not overlap"), [measured({"overlap": True})])
        assert verification.passed is False

    def test_count_conflict_is_detected(self) -> None:
        """The spec's worked example: VLM says 3 people, detector finds 2."""
        verification = verify(said("there are 3 people"), [measured({"people": 2})])
        assert verification.passed is False
        assert verification.conflict is not None
        assert "3" in verification.conflict
        assert "2" in verification.conflict

    def test_conflict_text_is_specific_enough_to_act_on(self) -> None:
        verification = verify(said("the lines cross 5 times"), [measured({"crossings": 2})])
        assert verification.conflict is not None
        assert "crossings" in verification.conflict

    def test_a_statement_mentioning_several_numbers_needs_only_one_to_match(self) -> None:
        verification = verify(
            said("of the 4 shapes, 2 circles overlap"), [measured({"circles": 2})]
        )
        assert verification.passed is True


class TestBareVerdicts:
    """Regression: bare "Yes"/"No" answers never reached the verifier.

    _boolean_conflict required the measurement's own key word to appear in
    the statement, but benchmarks ask for exactly "Yes" or "No" and that is
    what models return. Every such answer silently passed verification, so
    the referee never once overruled the model — the defect that would have
    made M2's tool work pointless.

    Found by review, not by the tests: every existing case used prose like
    "the circles overlap", which is the shape I imagined rather than the
    shape models produce.
    """

    @pytest.mark.parametrize("answer", ["Yes", "yes", "Yes.", "YES"])
    def test_bare_yes_conflicts_with_a_false_measurement(self, answer: str) -> None:
        verification = verify(said(answer), [measured({"overlap": False})])
        assert verification.passed is False
        assert verification.conflict is not None

    @pytest.mark.parametrize("answer", ["No", "no", "No.", "NO"])
    def test_bare_no_conflicts_with_a_true_measurement(self, answer: str) -> None:
        verification = verify(said(answer), [measured({"overlap": True})])
        assert verification.passed is False

    def test_bare_yes_agrees_with_a_true_measurement(self) -> None:
        assert verify(said("Yes"), [measured({"overlap": True})]).passed is True

    def test_bare_no_agrees_with_a_false_measurement(self) -> None:
        assert verify(said("No."), [measured({"overlap": False})]).passed is True

    def test_a_verdict_then_justification_reads_as_the_verdict(self) -> None:
        statement = "No, the two circles are not touching each other."
        assert verify(said(statement), [measured({"overlap": False})]).passed is True
        assert verify(said(statement), [measured({"overlap": True})]).passed is False

    def test_the_leading_verdict_wins_over_a_later_one(self) -> None:
        """ "Yes, though no gap is visible" is a Yes."""
        statement = "Yes, though no gap is visible."
        assert verify(said(statement), [measured({"overlap": True})]).passed is True

    def test_punctuation_cannot_hide_the_verdict(self) -> None:
        assert verify(said("No,"), [measured({"overlap": True})]).passed is False

    def test_bare_verdicts_move_confidence_in_both_directions(self) -> None:
        agreed = verify(said("Yes"), [measured({"overlap": True})])
        clashed = verify(said("Yes"), [measured({"overlap": False})])
        assert adjust_confidence(0.5, agreed) > 0.5
        assert adjust_confidence(0.5, clashed) < 0.5


class TestAnswerKeyLimitsWhatIsJudged:
    """Regression: diagnostics were being checked against the answer.

    A line counter reporting {"crossings": 1, "columns_with_both": 300} had
    every correct answer rejected, because "1" was compared against 300 too.
    150 false conflicts in one 150-item run, and no verified steps at all.
    """

    def test_only_the_named_key_is_judged(self) -> None:
        result = ToolResult(
            value={"crossings": 1, "columns_with_both": 300},
            is_measurement=True,
            answer_key="crossings",
        )
        assert verify(said("{1}"), [result]).passed is True

    def test_a_genuine_disagreement_on_the_named_key_still_conflicts(self) -> None:
        result = ToolResult(
            value={"crossings": 1, "columns_with_both": 300},
            is_measurement=True,
            answer_key="crossings",
        )
        verification = verify(said("{3}"), [result])
        assert verification.passed is False
        assert verification.conflict is not None
        assert "crossings" in verification.conflict

    def test_diagnostics_still_reach_the_evidence_chain(self) -> None:
        """Not judged, but recorded — the numbers are the point of rule 8."""
        result = ToolResult(
            value={"crossings": 1, "columns_with_both": 300},
            is_measurement=True,
            answer_key="crossings",
        )
        assert verify(said("{1}"), [result]).computed["columns_with_both"] == 300

    def test_without_an_answer_key_every_number_is_judged(self) -> None:
        """The old behaviour, kept for tools that report a single figure."""
        result = ToolResult(value={"crossings": 1}, is_measurement=True)
        assert verify(said("{1}"), [result]).passed is True
        assert verify(said("{3}"), [result]).passed is False

    def test_a_boolean_answer_key_works_the_same_way(self) -> None:
        result = ToolResult(
            value={"overlap": True, "centre_distance": 47.0, "radius_sum": 52.0},
            is_measurement=True,
            answer_key="overlap",
        )
        assert verify(said("Yes"), [result]).passed is True
        assert verify(said("No"), [result]).passed is False


class TestABareVerdictCannotAnswerTwoQuestions:
    """Regression from a mixed toolbox.

    A bare "No" names no subject. Checked against one boolean measurement
    that is unambiguous; checked against two, it is being read as a claim
    about both, and it cannot be — the second tool is answering a different
    question.

    Found on BlindTest's circles task once decoy tools were added. A decoy
    reporting that two *bounding boxes* overlap contradicted a correct "No"
    about the *circles*: both statements were true, about different things.
    Items where a tool spoke scored 86.0%, against 98.2% where none did, and
    every failure ran the same way — truth No, answer Yes.
    """

    def test_a_bare_verdict_is_not_judged_by_a_second_measurement(self) -> None:
        circles = ToolResult(value={"overlap": False}, is_measurement=True, answer_key="overlap")
        boxes = ToolResult(
            value={"boxes_overlap": True}, is_measurement=True, answer_key="boxes_overlap"
        )
        assert verify(said("No"), [circles, boxes]).passed is True

    def test_measurements_that_concur_still_judge_a_bare_verdict(self) -> None:
        """Silencing the wrong tool must not silence the right one. When both
        report the same verdict there is no ambiguity about what "No"
        contradicts, whatever each was measuring."""
        circles = ToolResult(value={"overlap": True}, is_measurement=True, answer_key="overlap")
        boxes = ToolResult(
            value={"boxes_overlap": True}, is_measurement=True, answer_key="boxes_overlap"
        )
        verification = verify(said("No"), [circles, boxes])
        assert verification.passed is False
        assert verification.verdict_key in {"overlap", "boxes_overlap"}

    def test_a_correct_answer_survives_a_contradicting_decoy(self) -> None:
        """The case that cost 12 percentage points. The model is right, the
        applicable tool agrees, and a decoy measuring something else says the
        opposite — the answer must stand."""
        circles = ToolResult(value={"overlap": False}, is_measurement=True, answer_key="overlap")
        decoy = ToolResult(
            value={"boxes_overlap": True}, is_measurement=True, answer_key="boxes_overlap"
        )
        assert verify(said("No"), [circles, decoy]).passed is True

    def test_naming_the_subject_restores_the_judgement(self) -> None:
        """The rule is about ambiguity, not about the second tool. A
        statement that says which thing it means can be checked against it."""
        boxes = ToolResult(
            value={"boxes_overlap": True}, is_measurement=True, answer_key="boxes_overlap"
        )
        circles = ToolResult(value={"overlap": False}, is_measurement=True, answer_key="overlap")
        verification = verify(said("No, the boxes do not overlap"), [boxes, circles])
        assert verification.passed is False
        assert verification.verdict_key == "boxes_overlap"

    def test_one_measurement_still_judges_a_bare_verdict(self) -> None:
        """With a single tool there is no ambiguity about what "No" answers,
        and this is the common case — a benchmark asks for a bare Yes/No."""
        only = ToolResult(value={"overlap": True}, is_measurement=True, answer_key="overlap")
        assert verify(said("No"), [only]).passed is False

    def test_agreement_across_two_tools_still_passes(self) -> None:
        first = ToolResult(value={"overlap": True}, is_measurement=True, answer_key="overlap")
        second = ToolResult(
            value={"boxes_overlap": True}, is_measurement=True, answer_key="boxes_overlap"
        )
        assert verify(said("Yes"), [first, second]).passed is True

    def test_numbers_are_unaffected(self) -> None:
        """A count carries its own subject — "3" against a measured 5 is a
        disagreement whatever else is in the toolbox."""
        crossings = ToolResult(value={"crossings": 5}, is_measurement=True, answer_key="crossings")
        coverage = ToolResult(
            value={"coverage_percent": 12.0}, is_measurement=True, answer_key="coverage_percent"
        )
        assert verify(said("{3}"), [crossings, coverage]).passed is False


class TestNoFalseConflicts:
    """A conflict that is not real costs more than a missed one."""

    def test_unrelated_measurement_does_not_conflict(self) -> None:
        verification = verify(said("the image is mostly white"), [measured({"overlap": True})])
        assert verification.passed is True

    def test_descriptive_statement_with_no_numbers(self) -> None:
        verification = verify(said("two shapes near the centre"), [measured({"crossings": 4})])
        assert verification.passed is True

    def test_float_measurement_matches_a_stated_integer(self) -> None:
        verification = verify(said("3 crossings"), [measured({"crossings": 3.0})])
        assert verification.passed is True


class TestDecliningToAnswerIsNotAgreement:
    """A measurement cannot confirm a claim nobody made.

    Found by tracing a real run: once the agent magnified a corner, the model
    answered "CANNOT TELL" while the tool measured fragments of shapes. No
    verdict meant no contradiction, so the nonsense measurement was counted
    as confirmation and pushed confidence up.
    """

    @pytest.mark.parametrize(
        "refusal",
        [
            "CANNOT TELL",
            "I cannot tell from this view.",
            "Unable to determine.",
            "It is not possible to tell.",
        ],
    )
    def test_a_refusal_is_never_verified(self, refusal: str) -> None:
        verification = verify(said(refusal), [measured({"overlap": True})])
        assert verification.passed is False
        assert verification.method == "none"

    def test_a_refusal_cannot_raise_confidence_past_the_ceiling(self) -> None:
        verification = verify(said("CANNOT TELL"), [measured({"overlap": True})])
        confidence = 0.0
        for _ in range(10):
            confidence = adjust_confidence(confidence, verification)
        assert confidence <= UNVERIFIED_CEILING

    def test_the_measurement_is_still_recorded(self) -> None:
        """Unusable as proof, but it belongs in the evidence chain."""
        verification = verify(said("CANNOT TELL"), [measured({"overlap": True})])
        assert verification.computed == {"overlap": True}

    def test_a_refusal_is_not_treated_as_a_conflict_either(self) -> None:
        """Nothing was claimed, so nothing was contradicted."""
        assert verify(said("CANNOT TELL"), [measured({"overlap": True})]).conflict is None


class TestConfidenceAdjustment:
    def test_agreement_raises_confidence(self) -> None:
        verification = verify(said("the circles overlap"), [measured({"overlap": True})])
        assert adjust_confidence(0.5, verification) > 0.5

    def test_conflict_lowers_confidence(self) -> None:
        verification = verify(said("the circles overlap"), [measured({"overlap": False})])
        assert adjust_confidence(0.5, verification) < 0.5

    def test_verified_confidence_may_exceed_the_unverified_ceiling(self) -> None:
        """Confirmation by measurement is exactly what earns real certainty."""
        verification = verify(said("the circles overlap"), [measured({"overlap": True})])
        confidence = 0.5
        for _ in range(4):
            confidence = adjust_confidence(confidence, verification)
        assert confidence > UNVERIFIED_CEILING

    def test_unverified_confidence_never_passes_the_ceiling(self) -> None:
        verification = verify(said("looks like two circles"), [])
        confidence = 0.0
        for _ in range(20):
            confidence = adjust_confidence(confidence, verification)
        assert confidence <= UNVERIFIED_CEILING

    def test_confidence_is_clamped_to_one(self) -> None:
        verification = verify(said("the circles overlap"), [measured({"overlap": True})])
        confidence = 0.95
        for _ in range(5):
            confidence = adjust_confidence(confidence, verification)
        assert confidence == 1.0

    def test_confidence_is_clamped_to_zero(self) -> None:
        verification = verify(said("the circles overlap"), [measured({"overlap": False})])
        confidence = 0.1
        for _ in range(5):
            confidence = adjust_confidence(confidence, verification)
        assert confidence == 0.0

    def test_already_high_unverified_confidence_is_not_raised_further(self) -> None:
        verification = verify(said("x"), [])
        assert adjust_confidence(0.9, verification) == UNVERIFIED_CEILING


class TestMeasurementShapes:
    def test_scalar_measurement_is_recorded(self) -> None:
        verification = verify(said("some shapes"), [measured(42)])
        assert verification.computed["value"] == 42

    def test_several_scalar_measurements_are_kept_apart(self) -> None:
        verification = verify(said("some shapes"), [measured(1), measured(2)])
        assert len(verification.computed) == 2

    def test_method_name_is_taken_from_the_result_when_present(self) -> None:
        verification = verify(
            said("the circles overlap"),
            [measured({"method": "circles_overlap", "overlap": True})],
        )
        assert verification.method == "circles_overlap"

    def test_method_falls_back_when_unnamed(self) -> None:
        verification = verify(said("x"), [measured({"overlap": True})])
        assert verification.method == "measurement"


class TestVerificationModelInvariant:
    def test_a_passing_verification_never_carries_a_conflict(self) -> None:
        """Guarded by the model, but the verifier must never attempt it."""
        for statement, value in [
            ("the circles overlap", True),
            ("the circles do not overlap", False),
            ("4 crossings", 4),
        ]:
            verification = verify(said(statement), [measured({"overlap": value})])
            if verification.passed:
                assert verification.conflict is None


@pytest.mark.parametrize("ceiling_probe", [0.0, 0.3, 0.59, 0.6])
def test_ceiling_is_an_upper_bound_from_any_starting_point(ceiling_probe: float) -> None:
    verification = verify(said("unconfirmed"), [])
    assert adjust_confidence(ceiling_probe, verification) <= UNVERIFIED_CEILING
