"""Visual actions — the ways the agent can change what it is looking at.

These are the "saccade" half of the library: crop, zoom, annotate. All three
are pure functions that leave their input image untouched and report their
region in source coordinates, so any step in an evidence chain can be traced
back to the original picture.

They are public because a caller may well need domain-specific ways of
looking — following a bat through a swing, say — and should be able to add
them without forking the engine.
"""

from __future__ import annotations

from saccade.actions.annotate import annotate
from saccade.actions.crop import crop
from saccade.actions.zoom import zoom

__all__ = ["annotate", "crop", "zoom"]
