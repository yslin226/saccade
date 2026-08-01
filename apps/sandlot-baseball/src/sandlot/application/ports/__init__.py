"""What the use cases need from the outside world, stated as protocols.

A port names a capability without naming a provider: ``PosePort`` says
"something that turns frames into joint positions", not "MediaPipe". The
implementations live in ``infrastructure`` and are handed in at the edge.

Protocols rather than ABCs, so an implementation does not have to import
this package to satisfy it — the same reason ``saccade.ports`` uses them.
"""

from __future__ import annotations

__all__: list[str] = []
