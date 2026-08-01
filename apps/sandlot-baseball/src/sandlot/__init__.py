"""Sandlot Baseball — movement analysis built on the Saccade engine.

Clean Architecture, four layers (spec 5.1):

    domain/          the rules themselves — how an angle is computed, what
                     counts as a change since last time
    application/     the flow that orchestrates them, and the ports it needs
    infrastructure/  MediaPipe, YOLO, storage, and the Saccade wiring
    interfaces/      the CLI

The direction of dependency is inward: ``domain`` imports nothing from the
other three, and ``application`` names what it needs as a Port rather than
importing a detector. That is what lets the pose estimator be swapped, or
faked in a test, without the metric calculations knowing.

What this package deliberately does *not* do is measure by asking a model.
Numbers come from arithmetic on coordinates; the VLM's job is to say which
frames cannot be trusted and why (rule 1).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
