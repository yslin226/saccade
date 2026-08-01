"""Where sessions are kept.

JSON files for M3. The port is shaped like a database — identifiers,
timestamps, queries — so that M5 can swap in Postgres when RAG opens one
anyway. Rule 9 constrains how Postgres is used, not when one has to exist.
"""

from __future__ import annotations

__all__: list[str] = []
