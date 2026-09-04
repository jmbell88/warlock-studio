"""Editing a template's clip library: the door Poser's clip editor calls.

A *clip* is an ordered list of key poses plus how many frames each step holds,
and the shipped library under ``templates/clips/`` is what Troupe renders every
character sheet from. Until now it was editable only as raw JSON in the package
tree, which the Troupe programme named as the open half of its clip work -- and
which is the
wrong place twice over: an installed build replaces that tree wholesale on
upgrade, and it may not even be writable.

**So an edit goes to the user's own copy and the shipped one becomes a factory
default.** ``rigging.clip_library`` prefers ``data_dir/poser/clips/<template>.json``
when it exists and falls back to the shipped file otherwise, so:

* a user who has never opened the editor renders exactly what the build ships;
* a user who has renders their own, and an upgrade does not silently overwrite it;
* :func:`revert` is "delete my copy", which is why it needs no stored default.

**The whole file is the unit of change**, not a clip and not a key. A clip
library is internally consistent by construction -- every clip names poses the
same file carries -- and ``rigging.parse_clip_library`` refuses one that is not.
Saving a clip in isolation would let a key rename land while another clip still
pointed at the old name, and the failure would surface in the renderer. So the
editor works on a whole library, and :func:`save` validates the whole thing
through **the renderer's own parser** before a byte reaches disk: an editor that
can write a file the renderer cannot open is the one bug this design exists to
make impossible.

**Frame counts are checked here too, and that is not belt-and-braces.**
``charsheet.check_frame_counts`` is what refuses a seven-frame walk laid into an
eight-frame table -- and left to the renderer, that lands as a failed job on an
asset the user has already waited for. Checking it at the door means the editor
refuses the *edit*, pointing at the clip whose segments do not add up.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .. import poselib, rigging
from ..pipelines import sheet as sheetlib
from .core import WarlockService
from .errors import Conflict, Invalid, NotFound, invalid_from
from .files import _staged_write

log = logging.getLogger(__name__)

#: Every easing ``sheet.interpolate_clip`` knows -- imported from it rather
#: than restated, so a build that learns a fifth one cannot have an editor that
#: refuses it. Offered by name at all (rather than as a free string) so a typo
#: is refused at the door instead of falling through to linear in the renderer.
#:
#: **A note for whoever authors clips**: easing reshapes *where inside a segment*
#: the frames land, so it has nothing to do until a segment holds at least three.
#: At the shipped 1- and 2-frame segments the only sample points are t=0 and
#: t=0.5, and ``ease`` is a smoothstep whose value at 0.5 is exactly 0.5 -- so
#: ``ease`` and ``linear`` are the *same clip* there, while ``ease_in`` (0.25)
#: and ``ease_out`` (0.75) do differ. ``tests/test_clip_editing.py`` pins this
#: so it is a documented property rather than a surprise.
EASINGS = sheetlib.EASINGS

#: The frames one segment may hold. One is the floor because a segment *is* a
#: step from one key to the next and zero of them is not a shorter step, it is
#: a missing one; the ceiling is a guard on a typo turning a walk into an hour
#: of render, not a claim about what reads well.
MIN_SEGMENT = 1
MAX_SEGMENT = 64

#: The keys one clip may hold. Two is a clip's floor -- one key is a pose, and
#: the pose library is where a pose belongs.
MIN_KEYS = 2
MAX_KEYS = 64

#: How many poses one library file may carry. ``poselib.MAX_LIBRARY_POSES`` is
#: the pose library's own cap and is deliberately not reused: that one bounds a
#: directory the user browses, this one bounds a single JSON file the renderer
#: parses on every expansion.
MAX_LIBRARY_KEYS = 256


def _lock(svc: WarlockService) -> Any:
    """The one hold every clip write takes.

    The same sentinel-key idiom ``service.poses`` uses, and for the same
    reason: name uniqueness and the caps are properties of the *file*, so read,
    check and write have to be one hold or two saves both pass the check.
    """
    return svc.convert_lock("clips", "store")


def _template_or_invalid(template: str) -> str:
    key = str(template or "").strip()
    if not key:
        raise Invalid("a clip library belongs to a skeleton template", field="template")
    try:
        rigging.get_template(key)
    except Exception as exc:
        raise NotFound(f"{key!r} is not a skeleton template", field="template") from exc
    return key


def library(svc: WarlockService, template: str) -> dict[str, Any]:
    """The editable clip library for *template*, in the editor's own shape.

    ``poses`` is a **list** here where ``rigging.clip_library`` hands back a
    dict keyed by name: the editor shows an ordered list the user can read top
    to bottom, and a dict has no order to show. The name is still the identity
    -- clips reference keys by name -- so the list is a presentation of the same
    thing rather than a second model.

    ``edited`` says whether this is the user's copy or the shipped default,
    because "Revert to the shipped clips" has to be offerable only when there is
    something to revert.
    """
    key = _template_or_invalid(template)
    found = rigging.clip_library(key)
    path = poselib.clip_path(svc.config, key)
    return {
        "template": key,
        "space": found.get("space") or "node",
        "poses": [dict(row) for row in found.get("poses", {}).values()],
        "clips": [dict(clip) for clip in found.get("clips", ())],
        "edited": path.is_file(),
    }


def _check_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """Everything about a library that is wrong *before* the parser sees it.

    The parser refuses a file that cannot be read; these are the refusals that
    can name a *field*, so the editor can put the message beside the control
    that caused it rather than over the whole dialog.
    """
    poses = list(payload.get("poses") or ())
    clips = list(payload.get("clips") or ())
    if not poses:
        raise Invalid("a clip library needs at least one key pose", field="poses")
    if len(poses) > MAX_LIBRARY_KEYS:
        raise Conflict(
            f"a clip library holds at most {MAX_LIBRARY_KEYS} key poses, not {len(poses)}",
            field="poses",
        )
    names: list[str] = []
    for pose in poses:
        name = str(pose.get("name") or "").strip()
        if not name:
            raise Invalid("every key pose needs a name", field="poses")
        if name in names:
            raise Conflict(f'two key poses are both named "{name}"', field="poses")
        if not isinstance(pose.get("bones"), dict) or not pose["bones"]:
            raise Invalid(f'the key pose "{name}" has no bones', field="poses")
        names.append(name)

    if not clips:
        raise Invalid("a clip library needs at least one clip", field="clips")
    seen: list[str] = []
    for clip in clips:
        label = str(clip.get("name") or "").strip()
        if not label:
            raise Invalid("every clip needs a name", field="clips")
        if label in seen:
            raise Conflict(f'two clips are both named "{label}"', field="clips")
        seen.append(label)
        keys = [str(k) for k in (clip.get("keys") or ())]
        segments = [int(n) for n in (clip.get("segments") or ())]
        if len(keys) < MIN_KEYS:
            raise Invalid(
                f'"{label}" needs at least {MIN_KEYS} keys; one key is a pose, '
                "and the pose library is where a pose belongs",
                field="keys",
            )
        if len(keys) > MAX_KEYS:
            raise Conflict(
                f'"{label}" holds at most {MAX_KEYS} keys, not {len(keys)}', field="keys"
            )
        missing = [k for k in keys if k not in names]
        if missing:
            raise Invalid(
                f'"{label}" names key poses this library does not carry: '
                + ", ".join(sorted(set(missing))),
                field="keys",
            )
        # One segment per step, and how many steps there are depends on whether
        # the clip closes: an open clip of N keys has N-1 steps, a closed one
        # has N because the last key steps back to the first. Getting this
        # wrong is the single easiest way to author a clip whose frame count is
        # off by one, so it is refused with both numbers.
        closed = bool(clip.get("closed", False))
        wanted = len(keys) if closed else len(keys) - 1
        if len(segments) != wanted:
            raise Invalid(
                f'"{label}" is {"closed" if closed else "open"} with {len(keys)} keys, '
                f"so it needs {wanted} segment lengths, not {len(segments)}",
                field="segments",
            )
        for n in segments:
            if not MIN_SEGMENT <= n <= MAX_SEGMENT:
                raise Invalid(
                    f'"{label}" has a segment of {n} frames; each is '
                    f"{MIN_SEGMENT}-{MAX_SEGMENT}",
                    field="segments",
                )
        easing = str(clip.get("easing") or "linear")
        if easing not in EASINGS:
            raise Invalid(
                f'"{label}" uses an unknown easing {easing!r}; '
                f"this build knows {', '.join(EASINGS)}",
                field="easing",
            )
    # The rotation frame is not cosmetic: "node" is parent-relative absolute and
    # "delta" is relative to the bone's own rest, and the two are composed
    # differently by the renderer (``blender_worker.POSE_SPACES``). An unknown
    # name would be read as "node" downstream and silently mis-pose every clip.
    space = str(payload.get("space") or "node")
    if space not in sheetlib.POSE_SPACES:
        raise Invalid(
            f"unknown rotation space {space!r}; this build knows "
            f"{', '.join(sheetlib.POSE_SPACES)}",
            field="space",
        )
    return {
        "version": 2,
        "template": str(payload.get("template") or ""),
        "space": space,
        "poses": [
            {
                "name": str(p["name"]).strip(),
                "bones": p["bones"],
                **(
                    {"root_translation": [float(v) for v in p["root_translation"]]}
                    if p.get("root_translation")
                    else {}
                ),
            }
            for p in poses
        ],
        "clips": [
            {
                "name": str(c["name"]).strip(),
                "keys": [str(k) for k in c["keys"]],
                "segments": [int(n) for n in c["segments"]],
                "closed": bool(c.get("closed", False)),
                "easing": str(c.get("easing") or "linear"),
            }
            for c in clips
        ],
    }


def _check_renders(template: str) -> None:
    """Refuse a library the character sheet could not be laid out from.

    Only for the template Troupe actually renders: another template's clips are
    not laid into Troupe's frame table at all, so holding them to its frame
    counts would refuse a perfectly good fish.

    Called *after* the file has been written and the caches dropped, because
    the check runs through the ordinary read path -- which is the point: what is
    verified is the library as the renderer will see it, not as the editor
    imagines it. :func:`save` puts the previous bytes back if this raises.
    """
    from ..clips import expand_clips
    from .troupe import TROUPE_TEMPLATE

    if template != TROUPE_TEMPLATE:
        return
    try:
        expand_clips(template)
    except KeyError as exc:
        raise Invalid(
            f"a character sheet needs a clip called {exc}; this library has none",
            field="clips",
        ) from exc
    except ValueError as exc:
        raise invalid_from(exc, "These clips cannot fill a character sheet") from exc


def save(svc: WarlockService, template: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Write the user's copy of a template's clip library.

    Validated three times over, and each pass catches something the next
    cannot: :func:`_check_shape` for everything that can name a field,
    ``rigging.parse_clip_library`` because it is *the renderer's own parser* and
    an editor that can write what the renderer cannot read is the bug this
    exists to prevent, and :func:`_check_renders` for the frame table.

    The write is staged and the previous bytes are restored if the render check
    fails -- a save that leaves a library the renderer refuses is worse than a
    refused save, because the next character sheet is where the user finds out.
    """
    key = _template_or_invalid(template)
    document = _check_shape({**payload, "template": key})
    # A library's rotation space is a property of its stored *values*. Changing
    # it without rewriting every quaternion leaves a file whose poses mean
    # something other than what it says they mean, and the only place that shows
    # up is a mangled character sheet -- so it is refused by name here.
    current = library(svc, key).get("space") or "node"
    if document["space"] != current:
        raise Invalid(
            f'this library stores {current!r} rotations; saving it as '
            f'{document["space"]!r} would reinterpret every pose it holds',
            field="space",
        )
    try:
        rigging.parse_clip_library(document)
    except Exception as exc:
        raise invalid_from(exc, "That clip library cannot be saved") from exc

    path = poselib.clip_path(svc.config, key)
    with _lock(svc):
        previous = path.read_bytes() if path.is_file() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(document, indent=2).encode("utf-8")
        # ``files._staged_write``'s shape (SVC-01), not a bare ``.tmp``: a fixed
        # name and no ``finally`` stranded a visible ``<key>.json.tmp`` beside
        # the real file forever on an ENOSPC or an antivirus lock, and nothing
        # sweeps this directory.
        _staged_write(path, blob)
        rigging.invalidate_clips()
        try:
            _check_renders(key)
        except Exception:
            # Staged, like the publish above: ``rigging.clip_library`` reads this
            # same path from the render worker with no lock shared with ours, so
            # a direct ``write_bytes`` here would hand it a torn file.
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                _staged_write(path, previous)
            rigging.invalidate_clips()
            raise
    return library(svc, key)


def revert(svc: WarlockService, template: str) -> dict[str, Any]:
    """Drop the user's copy so the shipped library takes over again.

    Deleting is the whole implementation, which is why the shipped file is left
    alone rather than copied on first edit: there is no stored default to keep
    in step, and an upgrade that ships better clips reaches a reverted user
    automatically.
    """
    key = _template_or_invalid(template)
    path = poselib.clip_path(svc.config, key)
    with _lock(svc):
        if not path.is_file():
            raise NotFound(
                "this skeleton's clips have not been edited", field="template"
            )
        path.unlink()
        rigging.invalidate_clips()
    return library(svc, key)


def preview_frames(svc: WarlockService, template: str, clip: str) -> dict[str, Any]:
    """One clip expanded to the poses each of its frames shows.

    The **stored** clip, expanded. It goes through ``sheet.interpolate_clip``
    -- the renderer's own interpolator -- rather than a preview-shaped
    reimplementation of it, so what this returns is what the sheet will draw,
    including the easing and the closed-loop wrap. A preview with its own
    interpolation would be a second opinion about what a walk is, which is the
    whole thing the single clip library exists to avoid.

    **The pane deliberately does not call this**, which is worth stating
    because it looks like a service door with no reader. ``poser_mode.
    rebuild_frames`` expands the clip *being edited* -- keys inserted and not
    yet saved, a segment count mid-typing -- and a scrubber that showed the
    file instead would be a scrubber that ignores the edit in front of it. The
    two share the one thing that must not differ, the interpolator; where the
    record comes from is exactly what they are for. This is the answer for
    anything asking about a clip it has not got open: the API, and the tests
    that pin the stored path end to end.
    """
    key = _template_or_invalid(template)
    found = rigging.clip_library(key)
    record = next((c for c in found.get("clips", ()) if c["name"] == clip), None)
    if record is None:
        raise NotFound(f"{clip!r} is not a clip of this skeleton", field="clip")
    keys = [dict(found["poses"][name]) for name in record["keys"]]
    try:
        frames = sheetlib.interpolate_clip(
            keys,
            record["segments"],
            closed=record["closed"],
            easing=record["easing"],
            space=record["space"],
            clip_id=record["name"],
        )
    except ValueError as exc:
        raise invalid_from(exc, "That clip cannot be played") from exc
    return {"clip": record["name"], "frames": frames}
