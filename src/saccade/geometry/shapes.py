"""Geometric predicates used to confront VLM statements with numbers.

Everything here is closed-form and deterministic: the same inputs give the
same answer on every machine and every run. That is the whole point — these
functions are the referee, so they may not themselves be approximate.

This module takes coordinates, never files. It does not open, read or write
anything (CLAUDE.md rule 3).
"""

from __future__ import annotations

import math

__all__ = [
    "Point",
    "Segment",
    "angle_between",
    "centroid",
    "circles_overlap",
    "count_line_intersections",
    "distance",
    "segments_intersect",
    "speed",
]

Point = tuple[float, float]
Segment = tuple[Point, Point]

# Coordinates come from pixel measurements, so exact float equality is the
# wrong test for "touching". This tolerance is what "just touching" means.
EPSILON = 1e-9


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def angle_between(a: Point, vertex: Point, b: Point) -> float:
    """The angle ``a-vertex-b`` in degrees, in [0, 180].

    Note the argument order: the vertex is in the middle, matching how the
    angle is written. Passing the vertex first measures a different angle and
    returns a plausible number, so the order is not a detail.

    The result is unsigned. Two rays 30° apart give 30° whichever side they
    are on — a caller that needs a direction needs more than one angle.

    Args:
        a: A point on the first ray.
        vertex: Where the rays meet.
        b: A point on the second ray.

    Raises:
        ValueError: If either point coincides with the vertex. A zero-length
            ray has no direction, and the angle is undefined rather than 0.
    """
    ax, ay = a[0] - vertex[0], a[1] - vertex[1]
    bx, by = b[0] - vertex[0], b[1] - vertex[1]

    len_a = math.hypot(ax, ay)
    len_b = math.hypot(bx, by)
    if len_a <= EPSILON or len_b <= EPSILON:
        raise ValueError(f"a point coincides with the vertex {vertex}, so no angle exists")

    # Clamped because rounding can push the quotient just outside [-1, 1] on
    # collinear inputs, and math.acos raises there.
    cosine = min(1.0, max(-1.0, (ax * bx + ay * by) / (len_a * len_b)))
    return math.degrees(math.acos(cosine))


def speed(p1: Point, p2: Point, dt: float) -> float:
    """Distance covered per unit time, in the units the caller passes in.

    Unsigned, like :func:`distance` — this is speed, not velocity. Nothing
    here knows whether the coordinates are pixels or metres, so the result
    carries whatever units went in; naming them is the caller's job, and
    rule 8 says the number must reach the reader with them attached.

    Args:
        p1: Where the point was.
        p2: Where it is now.
        dt: Elapsed time. Must be positive.

    Raises:
        ValueError: If ``dt`` is not positive. Zero elapsed time makes the
            speed infinite rather than large, and returning ``inf`` would let
            it propagate into a comparison that silently succeeds.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    return distance(p1, p2) / dt


def centroid(points: list[Point]) -> Point:
    """The arithmetic mean of the points.

    This is the centre of the *points*, which is the centre of mass only when
    they are equally weighted. For anything where mass is distributed unevenly
    — body segments, for one — weight the points before calling.

    Raises:
        ValueError: If ``points`` is empty. The mean of nothing is not a
            position, and returning the origin would be a coordinate a caller
            could plot.
    """
    if not points:
        raise ValueError("centroid of an empty list is undefined")
    return (
        math.fsum(x for x, _ in points) / len(points),
        math.fsum(y for _, y in points) / len(points),
    )


def circles_overlap(
    c1: Point,
    r1: float,
    c2: Point,
    r2: float,
    *,
    tangent_counts: bool = False,
) -> bool:
    """Whether two circles overlap.

    Tangent circles — centre distance exactly equal to the sum of the radii —
    are the case VLMs get wrong most often, so the caller must say what it
    wants rather than inherit a silent default. ``tangent_counts=False``
    (the default) treats touching as *not* overlapping, i.e. overlap requires
    a region of positive area.

    Args:
        c1: Centre of the first circle.
        r1: Radius of the first circle. Must be positive.
        c2: Centre of the second circle.
        r2: Radius of the second circle. Must be positive.
        tangent_counts: Whether externally tangent circles count as overlapping.

    Raises:
        ValueError: If either radius is not positive.
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"radii must be positive, got r1={r1}, r2={r2}")

    centre_distance = distance(c1, c2)
    radius_sum = r1 + r2
    gap = centre_distance - radius_sum

    if abs(gap) <= EPSILON:
        return tangent_counts
    return gap < 0


def _orientation(a: Point, b: Point, c: Point) -> int:
    """Sign of the cross product of ``ab`` and ``ac``.

    Returns 1 for counter-clockwise, -1 for clockwise, 0 for collinear.
    """
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross > EPSILON:
        return 1
    if cross < -EPSILON:
        return -1
    return 0


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    """Whether collinear point ``p`` lies within segment ``ab``'s bounds."""
    return (
        min(a[0], b[0]) - EPSILON <= p[0] <= max(a[0], b[0]) + EPSILON
        and min(a[1], b[1]) - EPSILON <= p[1] <= max(a[1], b[1]) + EPSILON
    )


def segments_intersect(s1: Segment, s2: Segment) -> bool:
    """Whether two line segments intersect, endpoints and collinearity included.

    Uses the standard orientation test rather than solving for the crossing
    point: no division, so vertical and parallel segments need no special case.
    """
    p1, q1 = s1
    p2, q2 = s2

    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Collinear cases: an endpoint of one segment lying on the other.
    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q1, q2):
        return True
    if o3 == 0 and _on_segment(p2, q2, p1):
        return True
    return bool(o4 == 0 and _on_segment(p2, q2, q1))


def count_line_intersections(lines: list[Segment]) -> int:
    """Count how many pairs of segments intersect.

    This is the BlindTest task where the strongest models score 56.84% —
    barely above chance. Counting pairs (not crossing points) matches how the
    benchmark phrases the question: "how many times do these lines cross".

    Args:
        lines: Segments, each a ``((x1, y1), (x2, y2))`` pair.

    Returns:
        The number of intersecting pairs. Zero for fewer than two segments.
    """
    count = 0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if segments_intersect(lines[i], lines[j]):
                count += 1
    return count
