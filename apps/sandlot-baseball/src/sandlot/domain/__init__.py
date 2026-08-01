"""The rules themselves: what a metric is and how it is computed.

Pure arithmetic on coordinates. Nothing here opens a file, calls a detector,
or knows that MediaPipe exists — a joint angle is the same angle whoever
found the joints.

The boundary with ``application`` is business *rules* against business
*flow* (spec 5.1). "The hip-shoulder separation is the angle between those
two lines" belongs here; "load the video, detect, measure, save" does not.
"""

from __future__ import annotations

__all__: list[str] = []
