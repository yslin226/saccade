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

from saccade.geometry import angle_between, centroid, distance
from sandlot.domain.models import Frame

__all__ = [
    "COM_WEIGHTS",
    "centre_of_mass",
    "elbow_valgus",
    "hip_shoulder_separation",
    "kinetic_chain_order",
    "stride_length",
    "torso_length",
]

# Segment masses as a fraction of body mass, from Dempster's cadaver study
# (1955) as tabulated in Winter, *Biomechanics and Motor Control of Human
# Movement*. Only the joints this project tracks appear, so the weights are
# normalised over whichever of them a frame actually contains — an estimate
# of the whole body's centre from a partial skeleton, not a claim to have
# measured it.
COM_WEIGHTS = {
    "L shoulder": 0.0,
    "R shoulder": 0.0,
    "L elbow": 0.028,
    "R elbow": 0.028,
    "L wrist": 0.022,
    "R wrist": 0.022,
    "L hip": 0.0,
    "R hip": 0.0,
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

    return angle_between(shoulder, elbow, wrist)


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
    frames: list[Frame], *, segments: tuple[str, ...] = ("hips", "shoulders", "elbow", "wrist")
) -> list[str] | None:
    """Which segment reached peak angular speed first, in order.

    A good delivery fires from the ground up: hips, then shoulders, then
    elbow, then wrist, each handing energy to the next. Out-of-order is what
    coaches mean by "all arm", and it is visible in the timing of the peaks
    rather than in any single frame.

    Returns the segment names ordered by when each peaked, or None when
    fewer than three frames were given — a peak needs neighbours on both
    sides to be a peak rather than an endpoint.
    """
    if len(frames) < 3:
        return None

    peaks: dict[str, tuple[int, float]] = {}
    for segment in segments:
        series = _segment_angles(frames, segment)
        peak = _peak_rate(series)
        if peak is not None:
            peaks[segment] = peak

    if not peaks:
        return None

    # Ties broken by the caller's declared order rather than arbitrarily, so
    # the result is reproducible frame-for-frame.
    return sorted(peaks, key=lambda name: (peaks[name][0], segments.index(name)))


def _segment_angles(frames: list[Frame], segment: str) -> list[tuple[int, float] | None]:
    """The angle of one segment in each frame, None where unmeasurable."""
    out: list[tuple[int, float] | None] = []
    for index, frame in enumerate(frames):
        angle = _segment_angle(frame, segment)
        out.append(None if angle is None else (index, angle))
    return out


def _segment_angle(frame: Frame, segment: str) -> float | None:
    from saccade.geometry import bearing

    if segment == "hips":
        line = _line(frame, "L hip", "R hip")
    elif segment == "shoulders":
        line = _line(frame, "L shoulder", "R shoulder")
    elif segment == "elbow":
        return elbow_valgus(frame, side="R")
    elif segment == "wrist":
        line = _line(frame, "R elbow", "R wrist")
    else:
        raise ValueError(f"unknown segment {segment!r}")

    if line is None:
        return None
    return bearing(line[0], line[1])


def _peak_rate(series: list[tuple[int, float] | None]) -> tuple[int, float] | None:
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
