"""What counts as a change between two sessions.

The product's first claim is "you are 4.8 degrees less separated than last
time", not "you should be at 45 degrees". The first is arithmetic and cannot
be wrong; the second needs an authority behind it and being wrong about it is
worse than saying nothing. So the comparison against your own history ships
first, and this module is all of it.

Two things it refuses to do, both because the alternative is a number that
reads as meaningful and is not:

- Compare sessions built by different toolchains. A detector upgrade can move
  a coordinate in the last decimal place, and the resulting "change" would
  describe a version bump.
- Report a missing metric as a change of zero. "Nothing moved" and "nobody
  looked" are different findings.
"""

from __future__ import annotations

from sandlot.domain.models import MetricDelta, Session, Toolchain

__all__ = [
    "IncomparableSessionsError",
    "changed",
    "difference",
]


class IncomparableSessionsError(ValueError):
    """Raised when two sessions cannot be meaningfully subtracted."""


def difference(before: Session, after: Session) -> list[MetricDelta]:
    """Every metric's movement from ``before`` to ``after``.

    Metrics present in only one session appear with the missing side as
    ``None`` rather than being dropped: a metric that stopped being
    measurable is itself worth seeing, and silently omitting it makes two
    sessions look more alike than they are.

    Order is by metric name, so two runs of the same comparison produce
    identical output.

    Raises:
        IncomparableSessionsError: If the sessions were produced by different
            toolchains. Different *videos* are fine and are the entire point.
    """
    _refuse_if_incomparable(before, after)

    names = sorted({m.name for m in before.metrics} | {m.name for m in after.metrics})

    deltas = []
    for name in names:
        earlier = before.metric(name)
        later = after.metric(name)
        # Either side may be absent, but not both — the name came from one
        # of them.
        unit = (earlier or later).unit  # type: ignore[union-attr]
        deltas.append(
            MetricDelta(
                name=name,
                unit=unit,
                before=None if earlier is None else earlier.value,
                after=None if later is None else later.value,
            )
        )
    return deltas


def changed(deltas: list[MetricDelta], *, threshold: float = 0.0) -> list[MetricDelta]:
    """Those that moved by more than ``threshold``.

    Incomparable metrics are excluded rather than treated as unchanged. A
    caller wanting to show them has the full list.

    Args:
        threshold: Minimum absolute change to report. The default of 0.0
            reports everything that moved at all, which is honest but noisy;
            a caller with a sense of what is meaningful for a given metric
            should say so.

    Raises:
        ValueError: If ``threshold`` is negative. A negative threshold would
            silently include everything, which is not what any caller
            passing one means.
    """
    if threshold < 0:
        raise ValueError(f"threshold must not be negative, got {threshold}")

    return [delta for delta in deltas if delta.change is not None and abs(delta.change) > threshold]


def _refuse_if_incomparable(before: Session, after: Session) -> None:
    """The toolchain is the only hard barrier.

    Different videos are expected — comparing two deliveries is the product.
    The same video analysed twice is also fine and is how the determinism
    check is run.
    """
    if before.toolchain != after.toolchain:
        raise IncomparableSessionsError(
            f"toolchains differ: {_describe(before.toolchain)} vs {_describe(after.toolchain)}. "
            f"A detector upgrade can move coordinates, so any difference would be partly the "
            f"version and there is no way to say how much."
        )


def _describe(toolchain: Toolchain) -> str:
    return f"mediapipe {toolchain.mediapipe}, ultralytics {toolchain.ultralytics}"
