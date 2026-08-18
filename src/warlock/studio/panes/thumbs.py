"""One picture of a job, at whatever size the caller has room for.

Extracted in the UI redesign, wave 4.3, when Home's Resume list became a grid of
thumbnails and needed exactly what a library card had been doing since H72:
find the job's ``thumb.png`` through the texture cache, draw it, and stand a
framed glyph in its place when there is not one. Two copies of that would drift
the first time either the cache key or the placeholder changed, and the failure
would be silent -- a grid of grey squares beside a library that shows pictures.

The glyph table lives here too, because the two halves are one decision: what a
job looks like when there is nothing to look at.

Nothing here reads or writes state. It takes a ``Ctx`` only for the texture
cache and the job directory, which is what makes it drawable from any pane.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, widgets
from ..state import card_kind


def thumb_glyph(job: Any) -> str:
    """The icon standing in for a missing thumbnail (H72).

    Keyed on the job's *kind* as the library computes it, so the placeholder
    says what sort of thing is coming rather than only that something is. A
    failed job gets the alert glyph regardless: its card is about the failure,
    not about what it would have been.
    """
    if job.get("status") == "error":
        return icons.CIRCLE_ALERT
    return {
        "reference": icons.IMAGE,
        "tile": icons.GRID,
        "model": icons.BOX,
        "rig": icons.BONE,
        "sheet": icons.FILM,
        "sprite": icons.LAYERS,
        # The badge's glyph, so a card's placeholder and its chip agree.
        "tilesheet": icons.LAYERS,
    }.get(card_kind(job), icons.IMAGE)


def job_thumb(ctx: Any, job: Any, side: float) -> None:
    """Draw ``job``'s picture as a ``side``-px square, or a glyph in its place.

    ``side`` is already in physical pixels -- the caller has done the ``sp``,
    because it is the one that knows how much room it has.

    The texture is asked for by job id *and* path, which is the cache's own
    contract: the id is the key it evicts by and the path is what it loads. A
    job with no ``thumb.png`` in its file list is never asked for, so a queued
    or failed row costs no disk access at all.
    """
    job_id = job["id"]
    texture = None
    if ctx.textures is not None and "thumb.png" in (job.get("files") or []):
        texture = ctx.textures.get(job_id, ctx.job_dir(job_id) / "thumb.png")
    if texture is not None:
        imgui.image(widgets.texture_ref(texture), (side, side))
    else:
        # A framed glyph rather than a hole (H72). Every queued job, every
        # failure and every rig row lacks a thumbnail, and an empty square the
        # size of one reads as a broken image instead of a pending one.
        widgets.thumb_placeholder(side, thumb_glyph(job))
