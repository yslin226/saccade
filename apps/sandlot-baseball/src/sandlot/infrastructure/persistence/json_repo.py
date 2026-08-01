"""Sessions as JSON files, one per file.

M5 replaces this with Postgres, which RAG opens anyway. The port is already
shaped like a database, so that is a reimplementation rather than a redesign.

One file per session rather than a single index: two analyses running at once
cannot clobber each other, and a file that fails to parse costs one session
instead of all of them.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from sandlot.domain.models import Session

__all__ = ["JsonSessionRepo"]

logger = logging.getLogger("sandlot")

# Session ids reach the filesystem as names, so they may not contain a path.
# An id of "../../etc/passwd" would otherwise write there.
#
# The leading character must not be a dot, which is what rejects ".." — a
# pattern of "one or more safe characters" accepts it, since a dot is a safe
# character, and `directory / "...json"` then resolves to the parent.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")


class JsonSessionRepo:
    """Stores sessions under a directory. Implements ``SessionRepoPort``.

    Args:
        directory: Where the files go. The CLI passes ``--data-dir``,
            defaulting to ``~/.sandlot/sessions``; tests pass ``tmp_path``,
            so no test can write to a real home directory.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session) -> None:
        """Write a session, replacing any with the same id.

        Written to a temporary file and renamed, so a reader never sees half
        a session — and a crash mid-write leaves the previous version rather
        than a truncated one.

        Raises:
            ValueError: If the id is not usable as a filename.
        """
        path = self._path(session.id)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)
            raise

    def get(self, session_id: str) -> Session | None:
        """The session with that id, or None.

        A file that will not parse is also None, with a warning: one corrupt
        session should not stop the others being readable, and a caller
        asking for a specific id can see it is gone.
        """
        path = self._path(session_id)
        if not path.is_file():
            return None
        return self._load(path)

    def list_since(self, since: datetime | None = None) -> list[Session]:
        """Sessions created at or after ``since``, newest first.

        Unreadable files are skipped rather than raised on — a listing that
        fails entirely because of one bad file is less useful than one that
        is short by one.
        """
        sessions = []
        for path in sorted(self.directory.glob("*.json")):
            session = self._load(path)
            if session is None:
                continue
            if since is None or session.created_at >= since:
                sessions.append(session)

        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def _path(self, session_id: str) -> Path:
        if not _SAFE_ID.match(session_id):
            raise ValueError(
                f"session id {session_id!r} is not usable as a filename; "
                f"expected letters, digits, dot, dash or underscore"
            )
        return self.directory / f"{session_id}.json"

    def _load(self, path: Path) -> Session | None:
        try:
            return Session.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("skipping unreadable session file %s", path)
            return None
