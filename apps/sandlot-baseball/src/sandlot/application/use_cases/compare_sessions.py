"""What changed between two analyses.

Fetching and ordering only. The rule for what counts as a change lives in
``domain.comparison``; this decides which two sessions to hand it.
"""

from __future__ import annotations

from sandlot.application.ports import SessionRepoPort
from sandlot.domain.comparison import difference
from sandlot.domain.models import MetricDelta, Session

__all__ = ["SessionNotFoundError", "compare_sessions", "compare_with_previous"]


class SessionNotFoundError(LookupError):
    """Raised when an id does not name a stored session."""


def compare_sessions(
    before_id: str,
    after_id: str,
    *,
    repo: SessionRepoPort,
) -> list[MetricDelta]:
    """Difference between two stored sessions, by id.

    Raises:
        SessionNotFoundError: If either id is unknown. Naming which one
            matters — a caller who mistyped needs to know which of the two
            they got wrong.
        IncomparableSessionsError: If the toolchains differ.
    """
    before = _require(repo, before_id)
    after = _require(repo, after_id)
    return difference(before, after)


def compare_with_previous(
    session: Session,
    *,
    repo: SessionRepoPort,
) -> tuple[Session, list[MetricDelta]] | None:
    """Compare a session against the most recent one before it.

    This is the question the product actually answers — "what changed since
    last time" — and it is a different one from comparing two ids, because
    the caller does not know the previous id.

    Returns the session compared against and the differences, or ``None``
    when there is no earlier session. ``None`` rather than an empty list: a
    first session has nothing to compare against, which is not the same as
    having compared and found nothing.

    Raises:
        IncomparableSessionsError: If the previous session used a different
            toolchain. Not caught here — a caller shown "no change" after an
            upgrade would draw the wrong conclusion.
    """
    earlier = [
        candidate
        for candidate in repo.list_since(None)
        if candidate.created_at < session.created_at and candidate.id != session.id
    ]
    if not earlier:
        return None

    # list_since promises newest first, but the filter above is the only
    # thing that has looked at these timestamps — take the max rather than
    # trusting the order, since a repository that got it wrong would silently
    # compare against the oldest session instead.
    previous = max(earlier, key=lambda s: s.created_at)
    return previous, difference(previous, session)


def _require(repo: SessionRepoPort, session_id: str) -> Session:
    session = repo.get(session_id)
    if session is None:
        raise SessionNotFoundError(f"no session with id {session_id!r}")
    return session
