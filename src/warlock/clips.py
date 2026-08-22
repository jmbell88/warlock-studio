"""The shipped clip library joined to Troupe's frame table.

One function, in a module of its own, for the reason ``vectors.py`` is a module
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
