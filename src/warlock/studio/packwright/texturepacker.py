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

``pivot`` is a constant 0.5/0.5. A per-sprite pivot is a real feature and this
is not the place to invent one: the value would have nowhere to come from, and
a made-up number is worse than a documented centre.
"""

from __future__ import annotations

import json
from typing import Any

from .layout import Layout

FORMAT = "RGBA8888"
APP = "Warlock Packwright"
SCHEMA_VERSION = "1.0"


def _frame_entry(frame: Any) -> dict[str, Any]:
    return {
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
        "pivot": {"x": 0.5, "y": 0.5},
    }


def tp_json(layout: Layout, *, image_name: str, scale: float = 1.0) -> dict[str, Any]:
    """The sidecar as a plain dict, in the layout's own canonical frame order."""
    return {
        "frames": [_frame_entry(frame) for frame in layout.frames],
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


def tp_bytes(layout: Layout, *, image_name: str, scale: float = 1.0) -> bytes:
    """The same, serialized. Stable for a given layout, which is what makes a
    re-export of an unchanged document byte-identical."""
    payload = tp_json(layout, image_name=image_name, scale=scale)
    return (json.dumps(payload, indent=2) + "\n").encode()
