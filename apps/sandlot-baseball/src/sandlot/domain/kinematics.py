"""How each baseball metric is computed.

Arithmetic on coordinates, built from ``saccade.geometry``. Nothing here
opens a file or calls a detector — a joint angle is the same angle whoever
found the joints, which is what makes these testable against hand-written
coordinates rather than against a video.

Every function returns ``None`` when the joints it needs are missing, and
raises only when it was given something that cannot be a body. The
distinction matters: a missing ankle is an ordinary occlusion and the caller
carries on without that metric, while a hip and a shoulder at the same pixel
is a detector that has collapsed, and returning a plausible angle for it
would put a fabricated number into an evidence chain.

Which metrics, and why these: Driveline's OpenBiomechanics work found that
pitchers matched on arm speed still differ by 13 mph, with stride length and
kinetic-chain sequencing explaining much of the rest — and that for hitters,
bat speed barely predicts exit velocity (R² = 0.097) while weight transfer
dominates. So the measurements here are the ones the data says matter, not
the ones that are easy to see.
"""

from __future__ import annotations

from collections.abc import Sequence

from saccade.geometry import angle_between, centroid, distance
from sandlot.domain.models import Frame

__all__ = [
    "COM_WEIGHTS",
    "MIN_ELBOW_DEGREES",
    "MIN_SPAN_PX",
    "centre_of_mass",
    "elbow_valgus",
    "hip_shoulder_separation",
    "kinetic_chain_order",
    "stride_length",
    "torso_length",
]

# Roughly how the body's mass distributes over the points this project
# tracks. Segment fractions are the standard ones from Dempster's cadaver
# data as tabulated in Winter, *Biomechanics and Motor Control of Human
# Movement*; how they are *attributed to landmarks* is this project's own
# approximation and is where the error lives.
#
# The trunk is about half of body mass and has no landmark of its own, so it
# is split across the four points that bound it — shoulders and hips, an
# eighth each. That places the trunk's mass at the centre of those four
# points, which is close to where it actually sits and is the whole reason
# this is called an approximation.
#
# An earlier version gave those four points zero weight, on the reasoning
# that they are the endpoints of lines rather than masses. That silently
# discarded half the body: the "centre of mass" it computed was the average
# position of the limbs, and a hitter who rotated their trunk without moving
# their feet registered as having transferred no weight.
#
# Limb fractions are halved where a landmark stands for a segment shared with
# its neighbour, so the total over a full skeleton comes to roughly 1.0. The
# weights are normalised over whichever landmarks a frame actually contains,
# so a frame missing an ankle still reports a centre — shifted, and honestly
# so, since that mass is genuinely unaccounted for.
COM_WEIGHTS = {
    # Trunk (~0.50), split across the four landmarks bounding it.
    "L shoulder": 0.125,
    "R shoulder": 0.125,
    "L hip": 0.125,
    "R hip": 0.125,
    # Upper arm (~0.028 each) and forearm-plus-hand (~0.022 each).
    "L elbow": 0.028,
    "R elbow": 0.028,
    "L wrist": 0.022,
    "R wrist": 0.022,
    # Thigh (~0.100 each) and shank-plus-foot (~0.0465 each).
    "L knee": 0.100,
    "R knee": 0.100,
    "L ankle": 0.0465,
    "R ankle": 0.0465,
}

# Below this, two joints that should be a body's width apart are the same
# point, and the detector has collapsed rather than found a person edge-on.
# A shoulder line of two pixels used as a scale turns every ordinary distance
# into an apparent catastrophe — this project has seen it report a gap of 157
# body widths, which is not a quantity that exists.
MIN_SPAN_PX = 20.0

# The tightest an elbow bends. Below this the arm is not very flexed — the
# detector has misplaced a joint.
#
# What this rests on, stated rather than dressed up: on happy/林永閎.MOV the
# minimum over the clip was 15.4 degrees at frame 17, where MediaPipe placed
# the wrist back toward the shoulder at 0.78 confidence. Taking an extremum
# lets the single worst-detected frame become the metric, so an impossible
# value is not a curiosity in the tail — it is the answer.
#
# 25 is a conservative guess at where "impossible" starts, not a figure from
# a source. Maximum active elbow flexion is usually quoted as travel from
# straight rather than as the angle left between the segments, and this
# project has not checked what that converts to. If it turns out real
# deliveries reach below 25, this discards them silently — which is why the
# figure is written down here rather than left implicit in a comparison.
#
# It catches only what anatomy rules out anyway. A wrist misplaced by 40
# pixels still produces a possible angle, and finding those needs a second
# detector to disagree with the first (M4).
MIN_ELBOW_DEGREES = 25.0


def _line(frame: Frame, left: str, right: str) -> tuple[tuple[float, float], ...] | None:
    """The two endpoints of a body line, or None if either is missing or the
    line has collapsed."""
    a, b = frame.position(left), frame.position(right)
    if a is None or b is None:
        return None
    if distance(a, b) < MIN_SPAN_PX:
        return None
    return (a, b)


def torso_length(frame: Frame) -> float | None:
    """Shoulder midpoint to hip midpoint, in pixels.

    The scale everything else is normalised by. Torso rather than height,
    because the feet leave the frame constantly and the torso does not, and
    rather than shoulder width, because shoulders foreshorten as a pitcher
    turns while the torso stays roughly side-on to the camera.
    """
    shoulders = _line(frame, "L shoulder", "R shoulder")
    hips = _line(frame, "L hip", "R hip")
    if shoulders is None or hips is None:
        return None

    span = distance(centroid(list(shoulders)), centroid(list(hips)))
    return span if span >= MIN_SPAN_PX else None


def hip_shoulder_separation(frame: Frame) -> float | None:
    """Angle between the shoulder line and the hip line, in degrees.

    The classic measure of how much a hitter or pitcher has coiled: the hips
    open first and the shoulders follow, and the gap between them at the
    moment before release is where the energy is stored.

    Unsigned, so it does not distinguish a right-hander from a left-hander.
    Both read the same number for the same amount of coil, which is what a
    session-to-session comparison wants.
    """
    shoulders = _line(frame, "L shoulder", "R shoulder")
    hips = _line(frame, "L hip", "R hip")
    if shoulders is None or hips is None:
        return None

    # Move both lines to a shared origin, then measure between them: the
    # angle wanted is between the *directions* of the lines, not between
    # their positions on the image.
    (sx1, sy1), (sx2, sy2) = shoulders
    (hx1, hy1), (hx2, hy2) = hips
    shoulder_vector = (sx2 - sx1, sy2 - sy1)
    hip_vector = (hx2 - hx1, hy2 - hy1)

    separation = angle_between(shoulder_vector, (0.0, 0.0), hip_vector)
    # Two lines 170 degrees apart are 10 degrees from parallel. A line has no
    # head or tail, so the answer belongs in [0, 90].
    return 180.0 - separation if separation > 90.0 else separation


def elbow_valgus(frame: Frame, *, side: str) -> float | None:
    """Shoulder-elbow-wrist angle, in degrees.

    180 is a fully extended arm; smaller is more flexed. This is the angle
    ASMI's injury work tracks, because the elbow's valgus load peaks near
    maximum external rotation and the arm's position there is what separates
    a durable delivery from one that ends in surgery.

    A caution about what this is not: the true valgus torque needs the
    forearm's orientation in three dimensions, and a single camera cannot
    supply it. This is the planar angle, useful against your own previous
    session and not against a published threshold.

    Returns ``None`` below :data:`MIN_ELBOW_DEGREES`, because an angle a
    joint cannot reach is a detector failure rather than a very flexed arm —
    see that constant.

    Args:
        side: ``"L"`` or ``"R"``.

    Raises:
        ValueError: If ``side`` is anything else. Silently measuring the
            other arm would produce a number that looks fine and describes
            the wrong limb.
    """
    if side not in ("L", "R"):
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")

    shoulder = frame.position(f"{side} shoulder")
    elbow = frame.position(f"{side} elbow")
    wrist = frame.position(f"{side} wrist")
    if shoulder is None or elbow is None or wrist is None:
        return None

    # A zero-length segment has no direction, and angle_between raises for
    # it. Here that is an ordinary collapsed detection rather than a caller
    # error, so it becomes "no measurement" instead.
    if distance(shoulder, elbow) < MIN_SPAN_PX or distance(elbow, wrist) < MIN_SPAN_PX:
        return None

    angle = angle_between(shoulder, elbow, wrist)
    return None if angle < MIN_ELBOW_DEGREES else angle


def stride_length(frame: Frame) -> float | None:
    """Distance between the ankles, in torso lengths.

    Normalised because the same stride filmed from ten feet away and from
    thirty is the same stride, and an absolute pixel distance says more about
    where the camera was than about the delivery. Driveline's data puts
    stride length among the variables that explain velocity once arm speed is
    held constant.
    """
    ankles = _line(frame, "L ankle", "R ankle")
    scale = torso_length(frame)
    if ankles is None or scale is None:
        return None

    return distance(ankles[0], ankles[1]) / scale


def centre_of_mass(frame: Frame) -> tuple[float, float] | None:
    """Mass-weighted centre of the tracked joints, in pixels.

    Weighted by :data:`COM_WEIGHTS` over whichever joints the frame has, so a
    frame missing an ankle still reports a centre — shifted, and honestly so,
    since the mass that ankle stood for is genuinely not accounted for.

    This is the quantity Driveline's hitting work found dominant: bat speed
    explains almost none of exit velocity (R² = 0.097) while weight transfer
    explains a further 37.8%. Tracking where it goes across a swing is the
    point of measuring it at all.

    An approximation, and the approximation is in :data:`COM_WEIGHTS` rather
    than here — the trunk has no landmark of its own and is spread over the
    four points bounding it. Good enough to say a hitter's mass moved further
    this session than last; not good enough to publish as a centre of mass.
    """
    weighted: list[tuple[float, float]] = []
    for reading in frame.joints:
        weight = COM_WEIGHTS.get(reading.name)
        if not weight:
            continue
        # Repeat the point in proportion to its mass. Coarse, but exact at
        # the resolution these weights are known to, and it keeps the
        # computation inside a primitive the engine already tests.
        weighted.extend([(reading.x, reading.y)] * round(weight * 1000))

    return centroid(weighted) if weighted else None


def kinetic_chain_order(
    frames: Sequence[Frame],
    *,
    segments: tuple[str, ...] = ("hips", "shoulders", "elbow", "wrist"),
    side: str = "R",
) -> list[str] | None:
    """Which segment reached peak angular speed first, in order.

    A good delivery fires from the ground up: hips, then shoulders, then
    elbow, then wrist, each handing energy to the next. Out-of-order is what
    coaches mean by "all arm", and it is visible in the timing of the peaks
    rather than in any single frame.

    Args:
        frames: The movement, in order.
        segments: Which links to time. The default is the full chain.
        side: Which arm the ``elbow`` and ``wrist`` segments refer to. The
            default of ``"R"`` is a default, not an assumption — a
            left-hander must pass ``"L"`` or the chain will be timed against
            the arm that barely moves, which produces an ordering that looks
            reasonable and describes the wrong limb.

    Returns the segment names ordered by when each peaked, or None when
    fewer than three frames were given — a peak needs neighbours on both
    sides to be a peak rather than an endpoint.

    Raises:
        ValueError: If ``side`` is not ``"L"`` or ``"R"``, or a segment is
            not recognised.
    """
    if side not in ("L", "R"):
        raise ValueError(f"side must be 'L' or 'R', got {side!r}")
    if len(frames) < 3:
        return None

    peaks: dict[str, tuple[int, float]] = {}
    for segment in segments:
        series = _segment_angles(frames, segment, side=side)
        peak = _peak_rate(series)
        if peak is not None:
            peaks[segment] = peak

    if not peaks:
        return None

    # Ties broken by the caller's declared order rather than arbitrarily, so
    # the result is reproducible frame-for-frame.
    return sorted(peaks, key=lambda name: (peaks[name][0], segments.index(name)))


def _segment_angles(
    frames: Sequence[Frame], segment: str, *, side: str
) -> list[tuple[int, float] | None]:
    """The angle of one segment in each frame, None where unmeasurable."""
    out: list[tuple[int, float] | None] = []
    for index, frame in enumerate(frames):
        angle = _segment_angle(frame, segment, side=side)
        out.append(None if angle is None else (index, angle))
    return out


def _segment_angle(frame: Frame, segment: str, *, side: str) -> float | None:
    from saccade.geometry import bearing

    if segment == "hips":
        line = _line(frame, "L hip", "R hip")
    elif segment == "shoulders":
        line = _line(frame, "L shoulder", "R shoulder")
    elif segment == "elbow":
        return elbow_valgus(frame, side=side)
    elif segment == "wrist":
        line = _line(frame, f"{side} elbow", f"{side} wrist")
    else:
        raise ValueError(f"unknown segment {segment!r}")

    if line is None:
        return None
    return bearing(line[0], line[1])


def _peak_rate(series: Sequence[tuple[int, float] | None]) -> tuple[int, float] | None:
    """Where the angle changed fastest between consecutive measured frames.

    Gaps are skipped rather than interpolated across: a rate computed over a
    ten-frame occlusion is an average, not a peak, and would place the peak
    wherever the occlusion happened to be.
    """
    best: tuple[int, float] | None = None
    previous: tuple[int, float] | None = None

    for point in series:
        if point is None:
            previous = None
            continue
        if previous is not None and point[0] == previous[0] + 1:
            rate = abs(point[1] - previous[1])
            # Angles wrap at 360; the short way round is the real change.
            rate = min(rate, 360.0 - rate)
            if best is None or rate > best[1]:
                best = (point[0], rate)
        previous = point

    return best
