"""One rigged figure, built in code, shared by every test in this directory.

A module rather than a fixture for the same reason the rest of ``tests/inker``
uses module-private ``_doc()`` helpers: the tests here want to *vary* the figure
-- a leg placed straight, a hip dragged, a part left unassigned -- and a fixture
that has to grow a parameter per variation stops being simpler than a function.

The art is blocks of flat colour. Nothing here judges whether a walk looks
right; that verdict is a human's and it is owed in ``TODO.md``. What these tests
pin is that the motion is *correct*, and a rectangle is a better subject for that
than a drawing, because its bounds are known.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.walk import rig as R

SIZE = (64, 64)

#: ``part -> (x0, y0, x1, y1, rgba)``. A near-side half-figure; the far side is
#: minted from it by ``copy_near_to_far``, which is the workflow the brief
#: describes and therefore the one worth exercising.
BLOCKS: dict[str, tuple[int, int, int, int, tuple[int, int, int, int]]] = {
    "torso": (28, 18, 36, 38, (200, 80, 80, 255)),
    "head": (27, 8, 38, 19, (220, 180, 140, 255)),
    "near_upper_arm": (31, 19, 35, 29, (180, 60, 60, 255)),
    "near_lower_arm": (31, 28, 35, 38, (170, 55, 55, 255)),
    "near_hand": (30, 37, 36, 42, (220, 180, 140, 255)),
    "near_thigh": (30, 37, 35, 48, (70, 70, 160, 255)),
    "near_shin": (30, 47, 35, 57, (60, 60, 150, 255)),
    "near_foot": (30, 56, 40, 60, (40, 40, 40, 255)),
}

#: The rest pose. The legs are drawn straight and the ankles sit on the ground,
#: which is how a figure is normally drawn standing and is the case the gait has
#: to cope with: at that height the hip is a whole leg above the ground and no
#: step at all is reachable without coming down.
JOINTS: dict[str, tuple[float, float]] = {
    "neck": (32, 18),
    "near_shoulder": (32, 20),
    "near_elbow": (32, 29),
    "near_wrist": (32, 38),
    "near_hip": (32, 38),
    "near_knee": (32, 48),
    "near_ankle": (32, 57),
    "near_toe": (39, 58),
}


def plane(box: tuple[int, int, int, int], colour: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((SIZE[1], SIZE[0], 4), dtype=np.uint8)
    out[box[1] : box[3], box[0] : box[2]] = colour
    return out


def figure(*, far: bool = True, brightness: float = 0.7) -> R.Rig:
    """The whole rig, ready to pose. ``far=False`` leaves the far side empty."""
    rig = R.blank(SIZE)
    for name, (x0, y0, x1, y1, colour) in BLOCKS.items():
        rig = R.set_part(rig, name, R.part_from_plane(plane((x0, y0, x1, y1), colour)))
    for name, point in JOINTS.items():
        rig = R.set_joint(rig, name, point)
    if far:
        rig = R.copy_near_to_far(rig, "arm", brightness=brightness)
        rig = R.copy_near_to_far(rig, "leg", brightness=brightness)
    return R.set_ground(rig, R.default_ground(rig, SIZE[1]))
