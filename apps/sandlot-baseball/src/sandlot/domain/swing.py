"""Hitting metrics that need the bat, not just the body.

Separate from :mod:`sandlot.domain.kinematics` because these depend on an
object detector finding a bat, and that fails in exactly the frames a swing
is most interesting — a bat at contact speed is a smear COCO was never shown.
Every function here returns ``None`` rather than fitting a line through
whatever was found, because a swing plane through three spurious detections
is a confident number describing nothing.

Why these two: Driveline's hitting data found bat speed explains almost none
of exit velocity (R² = 0.097) while weight transfer explains a further 37.8%.
So the plane the bat travels through and where the body's mass is when it
gets there matter more than how fast it was swung — and the plane is what a
single camera can actually measure.
"""

from __future__ import annotations

from collections.abc import Sequence

from saccade.geometry import Point, bearing, distance, smooth
from sandlot.domain.kinematics import centre_of_mass, torso_length
from sandlot.domain.models import Frame

__all__ = [
    "MIN_BAT_TRAVEL_PX",
    "MIN_SWING_FRAMES",
    "bat_path",
    "swing_plane_angle",
    "weight_transfer",
]

# A swing plane needs a swing. Below this many frames with the bat located,
# any line fitted through them describes the detector's luck rather than the
# hitter's path.
MIN_SWING_FRAMES = 4

# And the bat has to have gone somewhere. A bat that moved five pixels across
# four frames is a stationary bat with detection jitter, and the angle of that
# jitter is noise reported to one decimal place.
MIN_BAT_TRAVEL_PX = 30.0


def bat_path(detections: Sequence[tuple[float, float, float, float] | None]) -> list[Point]:
    """Bat centres, one per frame it was found in.

    Takes bounding boxes rather than a detector, so this is testable against
    hand-written coordinates. ``None`` marks a frame where the bat was not
    found — those are dropped rather than interpolated, because a bat that
    was invisible for six frames was moving fastest exactly then, and a
    straight line across that gap understates the path it took.

    Args:
        detections: ``(x, y, width, height)`` per frame, ``None`` where the
            bat was not detected.
    """
    return [
        (x + width / 2, y + height / 2)
        for box in detections
        if box is not None
        for x, y, width, height in (box,)
    ]


def swing_plane_angle(
    detections: Sequence[tuple[float, float, float, float] | None],
) -> float | None:
    """The direction the bat travelled, in degrees, in [0, 180).

    Measured between the first and last located position rather than fitted
    across all of them: a swing is an arc, and a least-squares line through an
    arc reports the chord's angle with a residual nobody reads. The chord is
    the honest quantity, and saying so is cheaper than a fit that looks more
    rigorous than it is.

    Folded into [0, 180) because a plane has no direction — a right-hander
    and a left-hander swinging on the same plane read the same number, which
    is what comparing against your own history wants.

    Returns ``None`` when fewer than :data:`MIN_SWING_FRAMES` frames located
    the bat, or when it travelled less than :data:`MIN_BAT_TRAVEL_PX`.
    """
    path = bat_path(detections)
    if len(path) < MIN_SWING_FRAMES:
        return None

    start, end = path[0], path[-1]
    if distance(start, end) < MIN_BAT_TRAVEL_PX:
        return None

    angle = bearing(start, end)
    return angle - 180.0 if angle >= 180.0 else angle


def weight_transfer(frames: Sequence[Frame]) -> float | None:
    """How far the centre of mass moved, in torso lengths.

    The quantity Driveline's data puts ahead of bat speed. Normalised by the
    torso so the same shift filmed from ten feet and from thirty reads the
    same, and smoothed before measuring because a single frame where an ankle
    was misplaced moves the centre by more than the hitter did.

    Returns ``None`` when fewer than :data:`MIN_SWING_FRAMES` frames yielded
    both a centre of mass and a scale.
    """
    measured = [
        (centre, scale)
        for frame in frames
        if (centre := centre_of_mass(frame)) is not None
        and (scale := torso_length(frame)) is not None
    ]
    if len(measured) < MIN_SWING_FRAMES:
        return None

    # Smooth each axis separately: a centre of mass is two independent
    # series, and smoothing the distance instead would average over a
    # direction change and report less movement than happened.
    xs = smooth([centre[0] for centre, _ in measured])
    ys = smooth([centre[1] for centre, _ in measured])
    scale = sum(s for _, s in measured) / len(measured)

    return distance((xs[0], ys[0]), (xs[-1], ys[-1])) / scale
