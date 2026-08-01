"""What the use cases need from the outside world, stated as protocols.

A port names a capability without naming a provider: ``PosePort`` says
"something that turns frames into joint positions", not "MediaPipe". The
implementations live in ``infrastructure`` and are handed in at the edge.

Protocols rather than ABCs, so an implementation does not have to import
this package to satisfy it — the same reason ``saccade.ports`` uses them.

Determinism runs through these signatures. ``VideoPort.frames`` hands back
decoded arrays and ``PosePort.detect`` takes them, because the day-zero
measurement that showed ten runs can agree depended on decoding happening
once rather than inside each detection pass. A port that took a file path
would let a caller undo that without noticing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from sandlot.domain.models import Frame, Session, Toolchain

__all__ = [
    "DecodedVideo",
    "DetectPort",
    "Detection",
    "PosePort",
    "SessionRepoPort",
    "VideoPort",
]


class DecodedVideo(Protocol):
    """Frames in memory, plus what is needed to identify and time them.

    ``sha256`` is over the file's bytes: it is what makes two sessions of the
    same delivery comparable however the file was renamed, and what makes a
    re-encode correctly incomparable.
    """

    @property
    def sha256(self) -> str: ...

    @property
    def fps(self) -> float: ...

    @property
    def images(self) -> list[Any]:
        """Decoded frames, in order. Typed loosely because the domain never
        sees them — only the detectors do, and they want numpy arrays."""
        ...


@runtime_checkable
class VideoPort(Protocol):
    """Anything that can turn a video file into frames in memory."""

    def read(self, path: Any, *, stride: int = 1) -> DecodedVideo:
        """Decode ``path``.

        Args:
            path: The video file.
            stride: Take every Nth frame. 1 is every frame.

        Raises:
            OSError: If the file cannot be read. Returning an empty video
                would look like a video of nothing, and the analysis would
                report no metrics rather than a failure.
        """
        ...


@runtime_checkable
class PosePort(Protocol):
    """Anything that can find joints in decoded frames.

    Takes images rather than a path so that decoding happens once, outside
    this call — see the module docstring.
    """

    @property
    def toolchain(self) -> Toolchain:
        """The versions this detector is running, recorded into the session.

        Part of the answer rather than metadata: two sessions from different
        versions are refused a comparison, and that refusal needs something
        to compare.
        """
        ...

    def detect(self, images: list[Any], *, fps: float) -> list[Frame]:
        """One :class:`Frame` per image, in order.

        A frame where nothing was found still appears, with no joints. The
        alternative — dropping it — would renumber every frame after it and
        break the evidence chain's ability to point back at the video.
        """
        ...


class Detection(Protocol):
    """One object found in one frame: a bat, a ball, a glove."""

    @property
    def label(self) -> str: ...

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(x, y, width, height)``, matching ``saccade.models.BBox``."""
        ...

    @property
    def confidence(self) -> float: ...


@runtime_checkable
class DetectPort(Protocol):
    """Anything that can find objects — the bat and the ball."""

    def detect(self, images: list[Any]) -> list[list[Detection]]:
        """Detections per image, in order. An empty list for a frame where
        nothing was found, so indices keep lining up with the video."""
        ...


@runtime_checkable
class SessionRepoPort(Protocol):
    """Where analyses are kept.

    Shaped like a database even though M3's implementation is JSON files:
    identifiers, timestamps, a query. M5 opens Postgres for RAG anyway, and a
    port shaped like a filesystem would have to be redesigned then rather
    than reimplemented.
    """

    def save(self, session: Session) -> None:
        """Store a session. Overwrites one with the same id."""
        ...

    def get(self, session_id: str) -> Session | None:
        """The session with that id, or None. Absence is not an error — a
        caller asking for a session that was deleted wants to hear that."""
        ...

    def list_since(self, since: datetime | None = None) -> list[Session]:
        """Sessions created at or after ``since``, newest first.

        ``None`` means all of them. Newest first because the question a
        caller actually has is "what did I do last time".
        """
        ...
