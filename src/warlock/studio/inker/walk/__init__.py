"""A still side-view drawing, cut into parts, walked.

Four modules and one direction of dependency: :mod:`rig` is what the user
assembles (cut-outs, joints, a ground line), :mod:`gait` turns a rig and four
numbers into a pose at a phase, :mod:`render` turns a pose into placed pixels,
and :mod:`bake` turns the cycle into an ordinary animated Inker document.

Pure: numpy, the standard library and the rest of ``inker``. No imgui, no
document-level ops until :mod:`bake` reaches for ``Document`` inside a function,
and no persistence at all -- a rig lives in the session that built it.

This is a **prototype**, scoped to one side-view biped and one in-place walk with
rigid segments. Whether that reads as a walk or as a rotating paper puppet is a
verdict a person has to take in front of a real drawing; nothing here claims it,
and the tests below it are about the motion being *correct* rather than good.
"""

from __future__ import annotations

from .bake import TAG_NAME, WALK_MAX_PIXELS, composite_frames, document, too_large
from .gait import (
    WALK_FRAMES,
    Pose,
    WalkSettings,
    cycle,
    defaults_for,
    phases,
    pose,
    reachable_stride,
    screen_angle,
    two_bone,
)
from .render import bounds, clipping, frames, place
from .rig import (
    JOINTS,
    LIMBS,
    PART_NAMES,
    PARTS,
    Part,
    PartSpec,
    Rig,
    blank,
    copy_near_to_far,
    default_ground,
    label,
    leg_length,
    missing_joints,
    missing_parts,
    part_from_plane,
    refusal,
    segment_lengths,
    set_ground,
    set_joint,
    set_order,
    set_part,
    trim,
)

__all__ = [
    "JOINTS",
    "LIMBS",
    "PARTS",
    "PART_NAMES",
    "Part",
    "PartSpec",
    "Pose",
    "Rig",
    "TAG_NAME",
    "WALK_FRAMES",
    "WALK_MAX_PIXELS",
    "WalkSettings",
    "blank",
    "bounds",
    "clipping",
    "composite_frames",
    "copy_near_to_far",
    "cycle",
    "default_ground",
    "defaults_for",
    "document",
    "frames",
    "label",
    "leg_length",
    "missing_joints",
    "missing_parts",
    "part_from_plane",
    "phases",
    "place",
    "pose",
    "reachable_stride",
    "refusal",
    "screen_angle",
    "segment_lengths",
    "set_ground",
    "set_joint",
    "set_order",
    "set_part",
    "too_large",
    "trim",
    "two_bone",
]
