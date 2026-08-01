"""Pose and object detectors.

Determinism is a contract here, not a coincidence. M3's acceptance condition
is that ten analyses of one video agree exactly, and the day-zero measurement
that showed it possible depended on two things that are easy to undo by
accident: MediaPipe in ``RunningMode.IMAGE`` rather than ``VIDEO``, and
decoding separated from detection. See ``docs/plans/M3-sandlot-skeleton.md``.
"""

from __future__ import annotations

__all__: list[str] = []
