"""The TexturePacker "JSON (Array)" sidecar.

Deliberately **not** ``pipelines.sheet.sidecar``. That format is Warlock's own,
it is versioned, and ``sheet`` is its sole writer so ``version: 1`` cannot come
to mean two subtly different documents -- exactly the rule ``inker.sheetout``
was written under. What Packwright emits instead is the de-facto interchange
schema every 2D engine and framework already has a loader for, which is the
whole point of the mode: an atlas nobody can read is not an export.

The schema is engine-agnostic by construction -- pixel rectangles and nothing
else. ``rotated`` is always ``false`` because :mod:`.maxrects` does not rotate,
and it is emitted anyway because a consumer reads the key and a missing one is a
schema question rather than an answer.

``pivot`` is normalized against the **trimmed** frame rectangle, exactly as
``spriteSourceSize`` is, because that is the rectangle a loader has: the sprite
in the atlas is the trimmed one, and a fraction of the untrimmed canvas would
place it wrong by however much was cut off. A sprite that carries no pivot gets
0.5/0.5 -- the documented centre this format has always emitted -- so the key is
always present and the schema never shifts under a consumer.

A ninth key, ``slices``, appears on a frame that has named rectangles and on no
other, so an atlas of ordinary sprites is byte-for-byte what it was. Those
rectangles are in **source-image** space and have no trim interaction: they
describe the picture the artist drew, and a consumer that wants them against the
trimmed frame already has ``spriteSourceSize`` to subtract.
"""

from __future__ import annotations

import json
from typing import Any

from .layout import JSON_SCHEMAS as SCHEMAS
from .layout import Layout

FORMAT = "RGBA8888"
APP = "Warlock Packwright"
SCHEMA_VERSION = "1.0"


def _rect(x: Any, y: Any, w: Any, h: Any) -> dict[str, int]:
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _pivot(frame: Any) -> dict[str, float]:
    """The pivot as a fraction of the trimmed frame, or the documented centre.

    The division is safe without a guard: a fully transparent sprite trims to a
    1x1 rectangle rather than to nothing (``trim.EMPTY_SIZE``, and for this
    reason among others), so ``frame.w`` and ``frame.h`` are at least one for
    every sprite that reaches here. Pinned rather than asserted, because an
    assertion here would turn a packing decision made two modules away into a
    crash in a sidecar writer.
    """
    if frame.pivot is None:
        return {"x": 0.5, "y": 0.5}
    return {
        "x": (float(frame.pivot[0]) - frame.trim[0]) / frame.w,
        "y": (float(frame.pivot[1]) - frame.trim[1]) / frame.h,
    }


def _slices(frame: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": one.name,
            "bounds": _rect(one.x, one.y, one.w, one.h),
            "pivot": (
                None
                if one.pivot is None
                else {"x": float(one.pivot[0]), "y": float(one.pivot[1])}
            ),
            "center": None if one.center is None else _rect(*one.center),
        }
        for one in frame.slices
    ]


def _frame_entry(frame: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        # The sprite's own name, with a .png suffix: loaders key on it, and
        # every one of them expects a filename because TexturePacker's input
        # is a directory of files.
        "filename": f"{frame.name}.png" if not frame.name.endswith(".png") else frame.name,
        "frame": {"x": frame.x, "y": frame.y, "w": frame.w, "h": frame.h},
        "rotated": False,
        "trimmed": frame.trimmed,
        # Where the trimmed rectangle sat inside the original image. This is
        # what puts a sprite back where the artist drew it rather than flush
        # against its own bounding box.
        "spriteSourceSize": {
            "x": frame.trim[0],
            "y": frame.trim[1],
            "w": frame.w,
            "h": frame.h,
        },
        "sourceSize": {"w": frame.source_w, "h": frame.source_h},
        "pivot": _pivot(frame),
    }
    # Only when there is one, so an atlas of ordinary sprites keeps exactly the
    # eight keys it has always had.
    if frame.slices:
        entry["slices"] = _slices(frame)
    return entry


def tp_json(
    layout: Layout, *, image_name: str, scale: float = 1.0, schema: str = "array"
) -> dict[str, Any]:
    """The sidecar as a plain dict, in the layout's own canonical frame order.

    ``schema`` (imported from :mod:`.layout` as ``SCHEMAS``, so
    ``PackSettings.json_schema`` validates against the same tuple this writes)
    picks the shape of ``frames`` alone; ``meta`` is identical either way,
    because it describes the atlas, not how the frames are addressed.
    Refused by name rather than falling back to the default: a caller that
    misspells a schema silently getting the array it did not ask for is a
    worse failure than one that stops it.

    ``"array"`` is what this module always wrote and stays the default -- an
    existing caller that never heard of ``schema`` gets exactly the bytes it
    always did, ``"filename"`` key included on every entry. ``"hash"`` is
    TexturePacker's other published shape: the same entries keyed by filename
    in a dict instead of listed in an array, for a loader that looks one up
    by name rather than scans for it -- and, matching the real format, each
    value drops its own ``"filename"``, since the dict key already says it
    and the published schema does not repeat it inside the value.
    """
    if schema not in SCHEMAS:
        raise ValueError(f"schema must be one of {list(SCHEMAS)}")
    entries = [_frame_entry(frame) for frame in layout.frames]
    frames: Any = entries
    if schema == "hash":
        frames = {}
        for entry in entries:
            name = entry["filename"]
            # Two sources are allowed to share a display name -- see
            # ``PackDoc``'s own rule -- and the array schema is fine with
            # that, since every frame gets its own slot regardless of what it
            # is called. A hash schema is not: a second frame under the same
            # ``filename`` key would silently overwrite the first rather than
            # sit beside it, and the sidecar would describe fewer sprites
            # than the atlas actually has. Refused rather than losing one.
            if name in frames:
                raise ValueError(
                    f"the hash schema keys frames by filename, and {name!r} "
                    "names more than one sprite in this pack -- rename one, or export "
                    "the array schema instead"
                )
            frames[name] = {k: v for k, v in entry.items() if k != "filename"}
    return {
        "frames": frames,
        "meta": {
            "app": APP,
            "version": SCHEMA_VERSION,
            "image": image_name,
            "format": FORMAT,
            "size": {"w": layout.width, "h": layout.height},
            # A string, which is what TexturePacker writes and what several
            # loaders parse rather than read as a number.
            "scale": str(scale),
        },
    }


def tp_bytes(
    layout: Layout, *, image_name: str, scale: float = 1.0, schema: str = "array"
) -> bytes:
    """The same, serialized. Stable for a given layout, which is what makes a
    re-export of an unchanged document byte-identical."""
    payload = tp_json(layout, image_name=image_name, scale=scale, schema=schema)
    return (json.dumps(payload, indent=2) + "\n").encode()
