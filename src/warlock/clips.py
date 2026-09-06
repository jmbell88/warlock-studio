"""The shipped clip library joined to Troupe's frame table.

Two functions -- one for the sheet's grid, one for an exported animation
track -- in a module of its own, for the reason ``vectors.py`` is a module
of its own: **the worker may not import ``service``**, and the door may not
reimplement the worker. Both need to turn "the humanoid template's clips" into
"the expanded pose records the resolved frame table wants", and if either one
owned it the other would have to grow a second copy that could disagree about
what a walk is.

It is not in ``pipelines.charsheet`` because that module is deliberately
filesystem-free -- it decides what cell 137 depicts and never reads a file to
do it -- and not in ``rigging`` because that module imports nothing from
``pipelines`` and this needs ``sheet.interpolate_clip``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import rigging
from .pipelines import charsheet, sheet


def expand_clips(
    template_key: str,
    layout: charsheet.LayoutSpec | Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """``animation name -> expanded pose records``, for every Troupe animation.

    A missing clip raises ``KeyError``, which is the honest answer: the library
    ships with the package, so a Troupe animation with no clip is a broken
    build and not a user mistake. A library that expands to the wrong number of
    frames raises ``ValueError`` out of ``check_frame_counts`` -- named here
    rather than left to the renderer, because a seven-frame walk laid into an
    eight-frame table renders one cell of some other animation and sends the
    user to look at the rig.
    """
    resolved = (
        layout
        if isinstance(layout, charsheet.LayoutSpec)
        else charsheet.resolve_layout(layout)
    )
    library = rigging.clip_library(template_key)
    by_name = {c["name"]: c for c in library["clips"]}
    records: dict[str, list[dict[str, Any]]] = {}
    for movement in resolved.movements:
        animation = movement.name
        clip = by_name.get(animation)
        if clip is None:
            raise KeyError(animation)
        keys = rigging.clip_keys(template_key, animation)
        records[animation] = sheet.resample_clip(
            keys,
            clip["segments"],
            movement.frames,
            closed=clip["closed"],
            easing=clip["easing"],
            space=clip["space"],
            # The animation's name *is* the row identity here. Left to
            # ``_expand``'s default it was derived from the keys' ``id``
            # fields, which a clip library's key poses do not have -- so every
            # four-key clip shared one id and ``_charsheet``'s ``(id, frame)``
            # lookup let run overwrite walk. Names are unique by construction:
            # ``by_name`` above is keyed on them.
            clip_id=animation,
        )
    charsheet.check_frame_counts(records, resolved)
    return records


#: The timebase every baked animation track is written on, in frames per
#: second. 100 is not arbitrary: ``charsheet.ANIMATIONS`` states each movement's
#: frame duration in whole milliseconds (150, 100, 60, 80, 100), so a 10 ms
#: frame divides every one of them exactly -- each clip's keyframes land on
#: integer scene frames and no clip's tempo is rounded. glTF stores sample
#: times in seconds against the scene's own rate, which is why the base has to
#: be one number for the whole file rather than one per track.
ANIMATION_FPS = 100


def animation_tracks(template_key: str) -> list[dict[str, Any]]:
    """Every authored clip of a template, resolved to frames. -> track list.

    The host half of the animated-GLB bake, and deliberately *not*
    :func:`expand_clips`: that one resamples each clip to the frame count the
    sheet's layout asks for, because a sprite sheet has a grid to fill. An
    exported animation has no grid, so this uses the clip's **own** segment
    lengths -- the animation carries the author's timing rather than Troupe's.

    Timing comes from ``charsheet.ANIMATIONS``, which is where the per-frame
    duration and the loop flag already live, once. A second copy would be one
    edit from disagreeing about how fast a walk cycle is.

    Blender does no interpolation: it receives resolved frames, which is the
    same host/worker split ``fit_template`` establishes and what keeps the
    interpolation under test with no ``bpy``.
    """
    library = rigging.clip_library(template_key)
    timing = {name: (loop, ms) for name, _frames, loop, ms in charsheet.ANIMATIONS}
    tracks: list[dict[str, Any]] = []
    for clip in library["clips"]:
        name = str(clip["name"])
        frames = sheet.interpolate_clip(
            rigging.clip_keys(template_key, name),
            clip["segments"],
            closed=bool(clip["closed"]),
            easing=str(clip["easing"]),
            space=str(clip["space"]),
            # The clip's name is its identity here, ``expand_clips``' rule and
            # for its reason: a clip library's key poses carry no ``id``.
            clip_id=name,
        )
        # A clip nothing in the table names still exports, on its own closed
        # flag and the table's most common tempo: the library is authored and
        # the table is Troupe's, and a clip should not silently vanish from an
        # export because the sheet has no row for it.
        loop, duration_ms = timing.get(name, (bool(clip["closed"]), 100))
        tracks.append(
            {
                "name": name,
                "loop": bool(loop),
                "space": str(clip["space"]),
                # Whole scene frames per animation frame. See ANIMATION_FPS.
                "step": ANIMATION_FPS * int(duration_ms) / 1000.0,
                "frames": [
                    {"bones": record["bones"]} for record in frames
                ],
            }
        )
    return tracks


def animate_spec(
    job_dir: Path, template_key: str, out_glb: Path, result_dir: Path
) -> dict[str, Any]:
    """The worker spec for baking every authored clip into one animated GLB.

    ``rigging.pose_spec``'s shape one step up: a pose is one set of bone
    rotations, and this is a named sequence of them per clip, resolved **here**
    rather than in Blender. :func:`animation_tracks` does the interpolation on the host, so
    the timing stays under test with no ``bpy`` and the worker only does the
    thing only Blender can do -- keying an armature and writing glTF animation
    samplers.

    Here rather than beside ``pose_spec`` in ``rigging`` for that module's own
    pinned reason (``tests/test_poser_imports``): it may not import
    ``pipelines``, and resolving frames needs ``sheet.interpolate_clip``. Which
    is the argument this module was created on.

    Raises ``ValueError`` for a template with no clips, before a subprocess is
    spent: an animated GLB with no animations in it is a file that answers the
    question wrongly rather than not at all.
    """
    rigging.get_template(template_key)  # fail here, not three seconds into a subprocess
    tracks = animation_tracks(template_key)
    if not tracks:
        raise ValueError(f"nothing is authored for the {template_key} rig")
    return {
        "op": "animate",
        "rig_glb": str(job_dir / "rig.glb"),
        "out_glb": str(out_glb),
        # Named per job for ``armature_spec``'s reason: ``run_worker`` unlinks
        # and then watches this path, so two bakes sharing one name would let
        # each eat the other's answer.
        "result_path": str(result_dir / ".animate_result.json"),
        "fps": ANIMATION_FPS,
        "clips": tracks,
    }
