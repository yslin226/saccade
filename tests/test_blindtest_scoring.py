"""Tests for the BlindTest scorer.

This file exists because a scoring bug does not announce itself — it just
produces a number that looks plausible and is wrong. Every published figure
depends on this code being right, so it gets tested harder than the thing it
measures.
"""

from __future__ import annotations

import pytest

from benchmarks.blindtest.scoring import (
    extract_count,
    extract_pair,
    extract_yes_no,
    score,
)


class TestExtractYesNo:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Yes", True),
            ("No", False),
            ("yes", True),
            ("NO", False),
            ("Yes.", True),
            ("The answer is yes.", True),
            ("No, they are not touching.", False),
        ],
    )
    def test_plain_answers(self, text: str, expected: bool) -> None:
        assert extract_yes_no(text) is expected

    def test_leading_answer_wins_when_the_model_hedges(self) -> None:
        """Models often answer then equivocate; the answer comes first."""
        assert extract_yes_no("Yes, though it is hard to tell — no clear gap.") is True
        assert extract_yes_no("No. Some might say yes, but they are apart.") is False

    def test_unreadable_answers_return_none(self) -> None:
        assert extract_yes_no("I cannot tell from this image.") is None
        assert extract_yes_no("") is None

    def test_substrings_do_not_count_as_answers(self) -> None:
        """'nose' contains 'no' but is not an answer."""
        assert extract_yes_no("There is a nose in the picture.") is None
        assert extract_yes_no("The shapes are anonymous.") is None


class TestExtractCount:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("{3}", 3),
            ("{0}", 0),
            ("{12}", 12),
            ("{ 5 }", 5),
            ("The lines touch {2} times.", 2),
        ],
    )
    def test_braced_answers(self, text: str, expected: int) -> None:
        assert extract_count(text) == expected

    def test_braced_form_beats_other_numbers_in_the_sentence(self) -> None:
        """The prompt asks for braces; a compliant model has told us which
        number is the answer, so incidental figures must not override it."""
        assert extract_count("Looking at the 2 lines, they cross {4} times.") == 4

    def test_the_last_braced_value_wins(self) -> None:
        """Models sometimes revise: 'maybe {2}... actually {3}'."""
        assert extract_count("Perhaps {2}, but on closer look {3}.") == 3

    def test_falls_back_to_a_bare_integer(self) -> None:
        assert extract_count("They intersect 4 times.") == 4
        assert extract_count("7") == 7

    def test_unreadable_answers_return_none(self) -> None:
        assert extract_count("I cannot count them.") is None
        assert extract_count("") is None

    def test_negative_numbers_are_read(self) -> None:
        assert extract_count("{-1}") == -1


class TestScoreTouchingCircles:
    TASK = "Touching Circles"

    def test_correct_yes(self) -> None:
        assert score(self.TASK, "Yes, they touch.", "Yes") is True

    def test_correct_no(self) -> None:
        assert score(self.TASK, "No.", "No") is True

    def test_wrong_answer(self) -> None:
        assert score(self.TASK, "Yes", "No") is False
        assert score(self.TASK, "No", "Yes") is False

    def test_unreadable_answers_score_wrong_not_skipped(self) -> None:
        """Dropping unparseable replies would quietly inflate accuracy."""
        assert score(self.TASK, "I am unable to determine this.", "Yes") is False

    def test_empty_answer_scores_wrong(self) -> None:
        assert score(self.TASK, "", "Yes") is False


class TestScoreCounting:
    TASK = "Line Plot Intersections"

    def test_correct_braced(self) -> None:
        assert score(self.TASK, "{3}", "3") is True

    def test_correct_with_prose(self) -> None:
        assert score(self.TASK, "The lines cross {2} times.", "2") is True

    def test_off_by_one_is_wrong(self) -> None:
        assert score(self.TASK, "{3}", "2") is False

    def test_zero_is_a_real_answer_not_a_missing_one(self) -> None:
        assert score(self.TASK, "{0}", "0") is True
        assert score(self.TASK, "{0}", "1") is False

    def test_unreadable_answer_scores_wrong(self) -> None:
        assert score(self.TASK, "It is hard to say.", "2") is False


class TestScoreCountingGrid:
    """Rows and columns: both numbers matter.

    extract_count returns the last integer, so scoring these through it
    compared columns against the row count — marking correct answers wrong
    and some wrong ones right.
    """

    TASK = "Counting Grid - Blank Grids"

    def test_correct_braced_pair(self) -> None:
        assert score(self.TASK, "rows={5} columns={6}", "5,6") is True

    def test_correct_parenthesised_pair(self) -> None:
        assert score(self.TASK, "(5,6)", "5,6") is True

    def test_transposed_is_wrong(self) -> None:
        """5 rows by 6 columns is not 6 by 5."""
        assert score(self.TASK, "rows={6} columns={5}", "5,6") is False

    def test_one_number_wrong_is_wrong(self) -> None:
        assert score(self.TASK, "rows={5} columns={7}", "5,6") is False

    def test_a_square_grid_is_not_scored_by_luck(self) -> None:
        """Reading only the last number made this pass for the wrong reason."""
        assert score(self.TASK, "rows={3} columns={3}", "3,3") is True
        assert score(self.TASK, "rows={3} columns={4}", "3,3") is False

    def test_a_single_number_cannot_answer(self) -> None:
        assert score(self.TASK, "{5}", "5,6") is False

    def test_unreadable_answer_scores_wrong(self) -> None:
        assert score(self.TASK, "I cannot count them.", "5,6") is False

    def test_prose_around_the_numbers_is_tolerated(self) -> None:
        assert score(self.TASK, "The table has 5 rows and 6 columns.", "5,6") is True


class TestExtractPair:
    def test_braced_pair(self) -> None:
        assert extract_pair("rows={5} columns={6}") == (5, 6)

    def test_bare_pair(self) -> None:
        assert extract_pair("5,6") == (5, 6)

    def test_order_is_preserved(self) -> None:
        assert extract_pair("(6,5)") == (6, 5)

    def test_one_number_is_not_a_pair(self) -> None:
        assert extract_pair("{5}") is None

    def test_no_numbers(self) -> None:
        assert extract_pair("cannot tell") is None

    def test_braced_values_win_over_incidental_numbers(self) -> None:
        assert extract_pair("of the 12 cells, rows={3} columns={4}") == (3, 4)


class TestScorerCannotInflateAccuracy:
    """Guards against the failure mode that would matter most."""

    def test_a_refusal_never_scores_correct(self) -> None:
        for task, truth in [("Touching Circles", "Yes"), ("Line Plot Intersections", "3")]:
            for refusal in ["I cannot tell.", "", "Unable to determine."]:
                assert score(task, refusal, truth) is False

    def test_an_answer_mentioning_every_option_does_not_pass(self) -> None:
        """A reply listing possibilities is not an answer."""
        assert score("Line Plot Intersections", "It could be 1, 2, or 3.", "1") is False
