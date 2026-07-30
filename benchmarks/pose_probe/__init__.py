"""Testing whether an agent earns its keep when the detector is unreliable.

BlindTest could not answer that: there the geometry is exact, the tool alone
scores 98-99%, and the loop is overhead. Video is the opposite case. MediaPipe
misplaces joints through motion blur and occlusion, reports high confidence
while doing so, and cannot say which of the three causes it hit — and they
call for different handling.

MediaPipe lives here rather than in src/saccade, per rule 2. It reaches the
agent through register_tool().
"""

from __future__ import annotations

from benchmarks.pose_probe.continuity import (
    JointReading,
    continuity_tool,
    implausible_joints,
)

__all__ = ["JointReading", "continuity_tool", "implausible_joints"]
