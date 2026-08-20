"""Measure humanoid joints off a mesh, instead of scaling a template to its box.

``rigging.fit_template`` is bbox-proportional and approximate by design, and the
approximation holds only while the mesh is standing in roughly the template's
own pose. The shipped humanoid template is a fairly steep **A-pose**; a user's
base mesh is as likely to be a **T-pose**, and on one the fitted arm chain runs
diagonally down through the ribcage while the actual arm geometry lies out at
full reach. Automatic weights then bind the arms to the chest, and every clip in
the library renders a figure whose arms barely move -- silently, because nothing
in the fit can tell that it missed.

This is the other option the plan names for that case: measure the limbs. It is
pure ``numpy`` over a vertex cloud so it can be tested without Blender, and it
is deliberately only *five* measurements deep -- arm centreline, leg centreline,
pelvis, chest, head -- because every extra landmark is another thing to be wrong
about a mesh nobody has seen.

Axes are the repo's: **+Z up, -Y forward**, and the mesh standing on Z=0.

Both halves are mirrored from the left, which is a decision rather than a
shortcut: a base mesh is symmetric by construction, and measuring each side
independently would let a stray vertex give a character one long arm.
"""

from __future__ import annotations

from typing import Any

__all__ = ["BONES", "measure", "payload"]

#: The chain this produces, in template order. ``.R`` is mirrored from ``.L``,
#: so only the left is measured and the list below is what the caller gets.
BONES = (
    "hips", "spine", "chest", "neck", "head",
    "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
    "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
    "thigh.L", "shin.L", "foot.L",
    "thigh.R", "shin.R", "foot.R",
)

#: Where along the arm each joint sits, as a fraction of total reach. Measured
#: on the two supplied base meshes and ordinary human proportions: the upper arm
#: and forearm are close to equal, and the hand is about a seventh.
_SHOULDER, _ELBOW, _WRIST, _FINGER = 0.30, 0.62, 0.86, 0.995

#: Where the leg is sampled, as a fraction of standing height.
_KNEE, _ANKLE, _HIP = 0.28, 0.045, 0.45

_PELVIS = 0.52


def measure(points: Any) -> dict[str, tuple[list[float], list[float]]]:
    """``bone name -> (head, tail)`` in world space, measured off ``points``.

    ``points`` is an ``(N, 3)`` array of the mesh's vertices.
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 64:
        raise ValueError("a joint fit needs an (N, 3) vertex cloud")
    x, z = pts[:, 0], pts[:, 2]
    height = float(z.max() - z.min())
    reach = float(x.max())
    if height <= 0.0 or reach <= 0.0:
        raise ValueError("that mesh has no height or no width to measure")

    # The torso's half-width, taken at the *hips* -- the one height where the
    # arms certainly are not, whatever pose the mesh is standing in.
    hip_band = pts[(z > height * 0.42) & (z < height * 0.50)]
    if not len(hip_band):
        raise ValueError("no geometry at hip height")
    torso_half = float(np.abs(hip_band[:, 0]).max())
    # A body whose widest point is its own hips has no arms held out, and the
    # centreline fit below would happily trace a *leg* and call it an arm. That
    # is the one failure worth refusing rather than reporting: a rig with the
    # shoulders bound to the hips looks perfectly well-formed in rig.json.
    # Refusing here also covers the arms-at-the-sides mesh the plan already
    # excludes -- measuring one this way would be wrong, not merely imprecise.
    if reach < torso_half * 1.5:
        raise ValueError(
            "that mesh is not standing in a T-pose or A-pose: its widest point "
            f"({reach:.3f}) is no further out than its hips ({torso_half:.3f})"
        )

    # --- arm, by fitting a line rather than by slicing.
    #
    # The outer half of the reach is unambiguously arm: the torso ends well
    # inside it whatever the pose. So the centreline is measured there and
    # extrapolated back to the shoulder. Slicing all the way in does not work --
    # around the armpit one slab holds arm and ribcage at once and their mean is
    # neither, which puts the elbow inside the chest. Fitting also makes the
    # result indifferent to the angle the arm is held at, which is the whole
    # point: T-pose and A-pose differ by exactly that angle.
    samples = []
    for frac in np.linspace(0.45, 0.99, 16):
        at = reach * frac
        span = reach * 0.03
        sel = pts[(x >= at - span) & (x < at + span)]
        if len(sel):
            samples.append(sel.mean(axis=0))
    if len(samples) < 4:
        raise ValueError("not enough arm geometry to fit a centreline")
    arm = np.array(samples)
    fit_z = np.polyfit(arm[:, 0], arm[:, 2], 1)

    def on_arm(frac: float) -> list[float]:
        at = reach * frac
        # **Y is pinned to zero along the whole arm, deliberately.** A shoulder
        # measured a few centimetres forward tilts the bone out of the XZ plane,
        # and a swing about the bone's own local X then carries the arm forward
        # as well as down -- which reads, at sprite scale, as a character
        # walking with its arms folded across its chest. A base mesh's arms are
        # front-to-back symmetric at rest, so zero is also the honest answer.
        return [float(at), 0.0, float(np.polyval(fit_z, at))]

    shoulder = on_arm(_SHOULDER)
    elbow = on_arm(_ELBOW)
    wrist = on_arm(_WRIST)
    finger = on_arm(_FINGER)

    # --- leg. Inside the torso's half-width, below the hips, left side only.
    leg_band = pts[(np.abs(x) < torso_half * 1.2) & (z < height * 0.50) & (x > 0.0)]

    def leg_at(frac: float) -> Any:
        at = height * frac
        span = height * 0.025
        sel = leg_band[(leg_band[:, 2] >= at - span) & (leg_band[:, 2] < at + span)]
        if not len(sel):
            raise ValueError(f"no leg geometry at {frac:.3f} of height")
        return sel.mean(axis=0)

    knee, ankle = leg_at(_KNEE), leg_at(_ANKLE)
    pelvis = height * _PELVIS
    hip_x = float(leg_at(_HIP)[0])
    toe = [hip_x, float(pts[:, 1].min()), 0.0]

    # --- spine. The chest sits at the shoulder joint and the head tops the box.
    chest_z = float(shoulder[2])
    neck_z = chest_z + (height - chest_z) * 0.22
    head_z = chest_z + (height - chest_z) * 0.46
    spine_z = pelvis + (chest_z - pelvis) * 0.45

    def pt(p: Any) -> list[float]:
        return [float(p[0]), float(p[1]), float(p[2])]

    bones: dict[str, tuple[list[float], list[float]]] = {
        "hips": ([0.0, 0.0, pelvis], [0.0, 0.0, spine_z]),
        "spine": ([0.0, 0.0, spine_z], [0.0, 0.0, chest_z]),
        "chest": ([0.0, 0.0, chest_z], [0.0, 0.0, neck_z]),
        "neck": ([0.0, 0.0, neck_z], [0.0, 0.0, head_z]),
        "head": ([0.0, 0.0, head_z], [0.0, 0.0, float(z.max())]),
        "shoulder.L": ([torso_half * 0.30, 0.0, chest_z], list(shoulder)),
        "upper_arm.L": (list(shoulder), list(elbow)),
        "forearm.L": (list(elbow), list(wrist)),
        "hand.L": (list(wrist), list(finger)),
        "thigh.L": ([hip_x, 0.0, pelvis], pt(knee)),
        "shin.L": (pt(knee), pt(ankle)),
        "foot.L": (pt(ankle), list(toe)),
    }
    for name in list(bones):
        if name.endswith(".L"):
            head, tail = bones[name]
            bones[name[:-2] + ".R"] = (
                [-head[0], head[1], head[2]],
                [-tail[0], tail[1], tail[2]],
            )
    return bones


def payload(points: Any) -> dict[str, Any]:
    """``measure`` in the shape ``rigging.validate_joints`` takes.

    The whole skeleton and not a patch, which is that function's own rule: a
    partial correction leaves the caller and the worker disagreeing about which
    joints were measured and which were fitted.
    """
    bones = measure(points)
    return {
        "bones": [
            {"name": name, "head": list(bones[name][0]), "tail": list(bones[name][1])}
            for name in BONES
        ]
    }
