"""Tests for the planner.

Two properties carry the design: a conflict always redirects attention, and
the agent never re-examines a region it has already seen. Without the
second, the loop burns its whole step budget staring at the same place.
"""

from __future__ import annotations

from saccade._planner import Action, Planner
from saccade.models import BBox, EvidenceStep, Observation, Verification, Viewport

SIZE = (200, 200)


def step(
    index: int,
    bbox: BBox,
    *,
    conflict: str | None = None,
    verified: bool = True,
    source_size: tuple[int, int] = SIZE,
) -> EvidenceStep:
    verification = None
    if conflict is not None:
        verification = Verification(
            passed=False, method="measurement", computed={}, conflict=conflict
        )
    elif verified:
        verification = Verification(passed=True, method="measurement", computed={})

    return EvidenceStep(
        index=index,
        action_name="look",
        viewport=Viewport(bbox=bbox, source_size=source_size),
        observation=Observation(statement="something"),
        verification=verification,
    )


class TestOpeningMove:
    def test_first_step_looks_at_the_whole_image(self) -> None:
        planned = Planner(SIZE).plan_next([], confidence=0.0)
        assert planned.action is Action.LOOK
        assert planned.viewport.covers_full_image is True

    def test_opening_reason_is_recorded(self) -> None:
        planned = Planner(SIZE).plan_next([], confidence=0.0)
        assert "whole image" in planned.reason


class TestConflictRedirectsAttention:
    def test_a_conflict_triggers_a_zoom(self) -> None:
        planner = Planner(SIZE)
        evidence = [step(0, BBox(x=0, y=0, w=200, h=200), conflict="said 3, measured 2")]
        planned = planner.plan_next(evidence, confidence=0.3)
        assert planned.action is Action.ZOOM
        assert planned.zoom_factor > 1.0

    def test_the_zoom_targets_the_disputed_region(self) -> None:
        planner = Planner(SIZE)
        disputed = BBox(x=50, y=50, w=100, h=100)
        evidence = [step(0, disputed, conflict="disagreement")]
        planned = planner.plan_next(evidence, confidence=0.3)

        # Tighter than the disputed region, and still inside it.
        assert planned.viewport.bbox.area < disputed.area
        assert planned.viewport.bbox.x >= disputed.x
        assert planned.viewport.bbox.right <= disputed.right

    def test_the_reason_names_the_conflicting_step(self) -> None:
        planner = Planner(SIZE)
        evidence = [
            step(0, BBox(x=0, y=0, w=200, h=200)),
            step(1, BBox(x=0, y=0, w=100, h=100), conflict="mismatch"),
        ]
        planned = planner.plan_next(evidence, confidence=0.4)
        assert "step 1" in planned.reason

    def test_the_most_recent_conflict_wins(self) -> None:
        planner = Planner(SIZE)
        evidence = [
            step(0, BBox(x=0, y=0, w=200, h=200), conflict="old"),
            step(1, BBox(x=100, y=100, w=100, h=100), conflict="new"),
        ]
        planned = planner.plan_next(evidence, confidence=0.4)
        assert planned.viewport.bbox.x >= 100

    def test_conflict_outranks_the_step_budget(self) -> None:
        """A live disagreement is worth spending the last step on."""
        planner = Planner(SIZE, max_steps=2)
        evidence = [
            step(0, BBox(x=0, y=0, w=200, h=200)),
            step(1, BBox(x=0, y=0, w=200, h=200), conflict="mismatch"),
        ]
        assert planner.plan_next(evidence, confidence=0.3).action is Action.ZOOM

    def test_a_region_too_small_to_subdivide_falls_through_to_scanning(self) -> None:
        """Magnifying a 4x4 region would magnify noise, so look elsewhere."""
        planner = Planner(SIZE)
        evidence = [step(0, BBox(x=0, y=0, w=4, h=4), conflict="mismatch")]
        planned = planner.plan_next(evidence, confidence=0.3)

        assert planned.viewport.bbox != BBox(x=0, y=0, w=4, h=4)
        assert "unexplored" in planned.reason or planned.action is Action.STOP


class TestExploration:
    def test_low_confidence_scans_an_unexplored_region(self) -> None:
        planner = Planner(SIZE)
        planner.record(Viewport.full(SIZE))
        planned = planner.plan_next([step(0, BBox(x=0, y=0, w=200, h=200))], confidence=0.2)
        assert planned.action is Action.ZOOM
        assert "unexplored" in planned.reason

    def test_regions_are_not_examined_twice(self) -> None:
        planner = Planner(SIZE)
        evidence = [step(0, BBox(x=0, y=0, w=200, h=200))]

        seen: list[BBox] = []
        for _ in range(4):
            planned = planner.plan_next(evidence, confidence=0.2)
            if planned.action is Action.STOP:
                break
            assert planned.viewport.bbox not in seen
            seen.append(planned.viewport.bbox)
            planner.record(planned.viewport)

        assert len(seen) == len(set(seen))

    def test_scanning_stops_once_everything_is_explored(self) -> None:
        planner = Planner(SIZE)
        evidence = [step(0, BBox(x=0, y=0, w=200, h=200))]

        for _ in range(10):
            planned = planner.plan_next(evidence, confidence=0.2)
            if planned.action is Action.STOP:
                assert "every region" in planned.reason
                return
            planner.record(planned.viewport)

        raise AssertionError("planner never stopped")

    def test_the_opening_full_look_does_not_exhaust_the_image(self) -> None:
        """Regression: containment-based exploration stopped the loop dead.

        The opening move examines the whole image, which contains every
        region that follows. Treating containment as "explored" meant the
        agent concluded there was nothing left to see and stopped after one
        step — the opposite of a saccade.
        """
        planner = Planner(SIZE)
        planner.record(Viewport.full(SIZE))
        planned = planner.plan_next([step(0, BBox(x=0, y=0, w=200, h=200))], confidence=0.2)

        assert planned.action is Action.ZOOM
        assert planned.viewport.bbox != BBox(x=0, y=0, w=200, h=200)

    def test_explored_regions_are_reported(self) -> None:
        planner = Planner(SIZE)
        planner.record(Viewport(bbox=BBox(x=0, y=0, w=50, h=50), source_size=SIZE))
        assert planner.explored == [BBox(x=0, y=0, w=50, h=50)]

    def test_explored_property_returns_a_copy(self) -> None:
        planner = Planner(SIZE)
        planner.record(Viewport.full(SIZE))
        planner.explored.clear()
        assert len(planner.explored) == 1


class TestStopping:
    def test_the_step_budget_stops_the_loop(self) -> None:
        planner = Planner(SIZE, max_steps=3)
        evidence = [step(i, BBox(x=0, y=0, w=200, h=200)) for i in range(3)]
        planned = planner.plan_next(evidence, confidence=0.5)
        assert planned.action is Action.STOP
        assert "3-step budget" in planned.reason

    def test_stopping_still_reports_a_valid_viewport(self) -> None:
        """Every planned action carries a viewport, including the last one."""
        planner = Planner(SIZE, max_steps=1)
        planned = planner.plan_next([step(0, BBox(x=0, y=0, w=200, h=200))], confidence=0.5)
        assert planned.viewport.source_size == SIZE


class TestGeometryIsAlwaysValid:
    def test_planned_viewports_stay_inside_the_image(self) -> None:
        """A viewport past the edge would fail crop() at execution time."""
        for size in [(200, 200), (101, 73), (33, 400), (1, 1)]:
            planner = Planner(size)
            evidence = [
                step(
                    0,
                    BBox(x=0, y=0, w=size[0], h=size[1]),
                    conflict="mismatch",
                    source_size=size,
                )
            ]
            for _ in range(6):
                planned = planner.plan_next(evidence, confidence=0.2)
                assert planned.viewport.bbox.right <= size[0]
                assert planned.viewport.bbox.bottom <= size[1]
                if planned.action is Action.STOP:
                    break
                planner.record(planned.viewport)

    def test_odd_sized_images_are_fully_tiled(self) -> None:
        """The final tile absorbs the remainder, so no column is skipped."""
        planner = Planner((101, 101))
        evidence = [step(0, BBox(x=0, y=0, w=101, h=101), source_size=(101, 101))]
        covered: list[BBox] = []
        for _ in range(6):
            planned = planner.plan_next(evidence, confidence=0.2)
            if planned.action is Action.STOP:
                break
            covered.append(planned.viewport.bbox)
            planner.record(planned.viewport)

        assert max(box.right for box in covered) == 101
        assert max(box.bottom for box in covered) == 101
