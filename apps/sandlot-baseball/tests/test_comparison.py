"""Tests for the models and the session comparison.

The comparison is the product's first claim — "you are 4.8 degrees less
separated than last time" — so what it refuses to say matters as much as what
it says. Two refusals are pinned here: a metric that stopped being measurable
is not a change of zero, and two sessions from different detector versions
are not comparable at all.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sandlot.domain.comparison import IncomparableSessionsError, changed, difference
from sandlot.domain.models import Frame, JointReading, Metric, MetricDelta, Session, Toolchain

TOOLCHAIN = Toolchain(mediapipe="1.0.0", ultralytics="8.4.113", sandlot="0.1.0")
OTHER_TOOLCHAIN = Toolchain(mediapipe="1.0.1", ultralytics="8.4.113", sandlot="0.1.0")


def metric(name: str, value: float, unit: str = "degrees") -> Metric:
    return Metric(name=name, value=value, unit=unit, frames=(0,))


def session(*metrics: Metric, toolchain: Toolchain = TOOLCHAIN, video: str = "abc") -> Session:
    return Session(
        id="s1",
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
        video_sha256=video,
        frame_count=100,
        fps=60.0,
        toolchain=toolchain,
        metrics=metrics,
    )


class TestAMetricMustCiteItsEvidence:
    """Rule 8: no claim leaves without numbers, a frame and a source."""

    def test_a_metric_with_frames_is_accepted(self) -> None:
        assert Metric(name="x", value=1.0, unit="deg", frames=(4, 5)).frames == (4, 5)

    def test_a_metric_with_no_frames_is_refused(self) -> None:
        """Refused, not defaulted. An empty evidence chain that passes
        silently is the failure the rule is written against."""
        with pytest.raises(ValidationError, match="cites no frame"):
            Metric(name="x", value=1.0, unit="deg", frames=())

    def test_diagnostics_ride_along_without_being_the_value(self) -> None:
        detailed = Metric(
            name="separation",
            value=42.0,
            unit="degrees",
            frames=(7,),
            detail={"shoulder_line": [[0, 0], [100, 0]]},
        )
        assert detailed.value == 42.0
        assert "shoulder_line" in detailed.detail

    def test_a_metric_cannot_be_edited_afterwards(self) -> None:
        """A metric that could change is one whose evidence no longer
        describes it."""
        fixed = metric("separation", 42.0)
        with pytest.raises(ValidationError):
            fixed.value = 43.0  # type: ignore[misc]


class TestFrame:
    def test_a_named_joint_is_found(self) -> None:
        f = Frame(
            index=3,
            timestamp=0.05,
            joints=(JointReading(name="L hip", x=1.0, y=2.0, confidence=0.9),),
        )
        assert f.position("L hip") == (1.0, 2.0)

    def test_an_absent_joint_gives_none(self) -> None:
        assert Frame(index=0, timestamp=0.0).position("L hip") is None
        assert Frame(index=0, timestamp=0.0).joint("L hip") is None

    def test_confidence_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            JointReading(name="L hip", x=0, y=0, confidence=1.4)


class TestDifference:
    def test_a_metric_that_moved(self) -> None:
        deltas = difference(session(metric("sep", 40.0)), session(metric("sep", 35.2)))
        assert len(deltas) == 1
        assert deltas[0].change == pytest.approx(-4.8)

    def test_the_sign_says_which_way(self) -> None:
        deltas = difference(session(metric("sep", 35.0)), session(metric("sep", 40.0)))
        assert deltas[0].change == pytest.approx(5.0)

    def test_an_unchanged_metric_reports_zero(self) -> None:
        deltas = difference(session(metric("sep", 40.0)), session(metric("sep", 40.0)))
        assert deltas[0].change == 0.0
        assert deltas[0].comparable is True

    def test_a_metric_only_in_the_earlier_session_is_kept(self) -> None:
        """Dropping it would make two sessions look more alike than they
        are."""
        deltas = difference(session(metric("stride", 1.2)), session())
        assert len(deltas) == 1
        assert deltas[0].before == pytest.approx(1.2)
        assert deltas[0].after is None

    def test_a_metric_only_in_the_later_session_is_kept(self) -> None:
        deltas = difference(session(), session(metric("stride", 1.2)))
        assert deltas[0].before is None
        assert deltas[0].after == pytest.approx(1.2)

    def test_an_unmeasurable_metric_is_not_a_change_of_zero(self) -> None:
        """ "Nothing moved" and "nobody looked" are different findings."""
        deltas = difference(session(metric("stride", 1.2)), session())
        assert deltas[0].change is None
        assert deltas[0].comparable is False

    def test_the_unit_survives_a_one_sided_metric(self) -> None:
        deltas = difference(session(), session(metric("stride", 1.2, unit="torsos")))
        assert deltas[0].unit == "torsos"

    def test_output_is_ordered_by_name(self) -> None:
        """So two runs of the same comparison print the same thing."""
        before = session(metric("z", 1.0), metric("a", 2.0), metric("m", 3.0))
        after = session(metric("m", 4.0), metric("a", 5.0), metric("z", 6.0))
        assert [d.name for d in difference(before, after)] == ["a", "m", "z"]

    def test_two_empty_sessions_give_nothing(self) -> None:
        assert difference(session(), session()) == []

    def test_the_same_video_analysed_twice_is_comparable(self) -> None:
        """That is how the determinism check is run."""
        deltas = difference(
            session(metric("sep", 40.0), video="same"),
            session(metric("sep", 40.0), video="same"),
        )
        assert deltas[0].change == 0.0


class TestIncomparableToolchains:
    def test_different_detector_versions_are_refused(self) -> None:
        with pytest.raises(IncomparableSessionsError, match="toolchains differ"):
            difference(
                session(metric("sep", 40.0)),
                session(metric("sep", 35.0), toolchain=OTHER_TOOLCHAIN),
            )

    def test_the_message_names_both_versions(self) -> None:
        """An auditor has to be able to see which upgrade caused it."""
        with pytest.raises(IncomparableSessionsError) as raised:
            difference(session(), session(toolchain=OTHER_TOOLCHAIN))
        assert "1.0.0" in str(raised.value)
        assert "1.0.1" in str(raised.value)

    def test_matching_toolchains_are_fine(self) -> None:
        assert difference(session(), session()) == []


class TestChanged:
    def test_it_selects_what_moved(self) -> None:
        deltas = [
            MetricDelta(name="a", unit="deg", before=1.0, after=1.0),
            MetricDelta(name="b", unit="deg", before=1.0, after=9.0),
        ]
        assert [d.name for d in changed(deltas)] == ["b"]

    def test_a_threshold_filters_small_movements(self) -> None:
        deltas = [
            MetricDelta(name="a", unit="deg", before=1.0, after=1.5),
            MetricDelta(name="b", unit="deg", before=1.0, after=9.0),
        ]
        assert [d.name for d in changed(deltas, threshold=1.0)] == ["b"]

    def test_the_threshold_is_on_the_absolute_change(self) -> None:
        deltas = [MetricDelta(name="a", unit="deg", before=9.0, after=1.0)]
        assert changed(deltas, threshold=1.0) == deltas

    def test_incomparable_metrics_are_excluded_not_counted_as_unchanged(self) -> None:
        deltas = [MetricDelta(name="a", unit="deg", before=1.0, after=None)]
        assert changed(deltas) == []

    def test_a_change_exactly_at_the_threshold_is_excluded(self) -> None:
        """Strictly greater. At the boundary the caller said it was not
        interested."""
        deltas = [MetricDelta(name="a", unit="deg", before=1.0, after=2.0)]
        assert changed(deltas, threshold=1.0) == []

    def test_a_negative_threshold_is_an_error(self) -> None:
        """It would silently include everything, which is not what anyone
        passing a threshold means."""
        with pytest.raises(ValueError, match="must not be negative"):
            changed([], threshold=-1.0)

    def test_an_empty_list_gives_an_empty_list(self) -> None:
        assert changed([]) == []
