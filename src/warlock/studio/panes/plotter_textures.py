"""GL textures for Plotter, one per tileset image per open map.

The ``inker_textures`` shape and the two rules that made it work. A texture is
*registered* with the imgui backend as well as created -- an id the renderer
does not know maps to no moderngl object, and the image comes out as the font
atlas -- and it is ``forget_texture``d before release, or the backend keeps a
dead object under a GL name the driver will reuse.

What is different is that nothing here is ever re-uploaded. A tileset's pixels
are frozen at construction (``plotter.tileset.Tileset`` copies them and clears
the write flag), so the texture is uploaded once and lives until the tab closes.
That is also why the cache key is the tileset's *identity*: two maps that opened
the same ``.tsx`` hold two distinct objects, so they get two textures, and one
map's close cannot free the other's.

The canvas draws one quad per visible cell out of these, which is why there is
no composited map image anywhere in the mode -- see ``plotter/render.py`` for
the one that exists, and for why it is only ever asked for by an export.
"""

from __future__ import annotations

from typing import Any

from .. import docmodes

PREFIX = "plotter_tex:"


def _slot(uid: str, name: str) -> str:
    return f"{PREFIX}{uid}:{name}"


def tileset_texture(ctx: Any, uid: str, index: int, tileset: Any) -> Any:
    """The GL texture for one tileset of one tab, uploaded on first ask.

    ``None`` when there is no GL context, which is what the headless smoke
    suite and every state-only test run under -- the callers draw a placeholder
    rather than branching on a viewer.
    """
    if ctx.viewer is None:
        return None
    key = _slot(uid, f"ts{index}")
    stamp_key = f"{key}:id"
    texture = ctx.state.preview.get(key)
    # Identity, not index: adding a tileset shifts nothing, but *undoing* an
    # add and adding a different one reuses the index, and a stale texture
    # under it would draw the old atlas forever with nothing in the data to
    # say why.
    if texture is not None and ctx.state.preview.get(stamp_key) != id(tileset.pixels):
        docmodes.forget_texture(texture)
        texture = None
    if texture is None:
        pixels = tileset.pixels
        texture = ctx.viewer.ctx.texture(
            (int(pixels.shape[1]), int(pixels.shape[0])), 4, pixels.tobytes()
        )
        # Nearest, always: a tileset is pixel art far more often than not, and
        # a linear filter on a palette grid samples the neighbouring tile
        # across the seam.
        nearest = ctx.viewer.ctx.NEAREST
        texture.filter = (nearest, nearest)
        ctx.state.preview[key] = texture
        ctx.state.preview[stamp_key] = id(pixels)
    return texture


def image_texture(ctx: Any, uid: str, name: str, pixels: Any, stamp: Any) -> Any:
    """A texture for an array this mode *generates*, re-uploaded when it moves.

    ``tileset_texture``'s twin for the one image here that is not frozen: the
    minimap is recomputed whenever the document changes, so its texture needs a
    staleness stamp that is a value rather than an identity: a freshly computed
    array gets a new ``id`` every time, and keying on that would re-upload the
    texture on every frame that asked for it.

    Shares ``PREFIX`` so ``release_doc`` frees it with the rest of the tab's.
    """
    if ctx.viewer is None:
        return None
    key = _slot(uid, name)
    stamp_key = f"{key}:stamp"
    texture = ctx.state.preview.get(key)
    if texture is not None and ctx.state.preview.get(stamp_key) != stamp:
        docmodes.forget_texture(texture)
        texture = None
    if texture is None:
        texture = ctx.viewer.ctx.texture(
            (int(pixels.shape[1]), int(pixels.shape[0])), 4, pixels.tobytes()
        )
        nearest = ctx.viewer.ctx.NEAREST
        texture.filter = (nearest, nearest)
        ctx.state.preview[key] = texture
        ctx.state.preview[stamp_key] = stamp
    return texture


def release_doc(ctx: Any, uid: str) -> None:
    """Drop every texture belonging to one closed tab."""
    docmodes.release_prefix(ctx, f"{PREFIX}{uid}:")


def release_all(ctx: Any) -> None:
    docmodes.release_prefix(ctx, PREFIX)
