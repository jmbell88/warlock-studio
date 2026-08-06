"""GL textures for Paint mode, one set per open document.

Generalised from the inline editor's two-slot cache, with the same two rules
that made it work. Uploads are gated on the document's revision, because
re-sending a megapixel every frame to show something that did not move is
megabytes of PCIe traffic per frame. And every texture is *registered* with the
imgui backend as well as created -- an id the renderer does not know maps to no
moderngl object, and the image comes out as the font atlas.

What is new is that the upload is by dirty rectangle. ``Document.take_dirty()``
returns the region that changed, or None to mean "everything", which is what a
structural change and a freshly opened file both need.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The checkerboard behind transparency. One small texture drawn tiled, so the
# pattern costs a single quad however far the canvas is zoomed out.
_CHECKER_KEY = "inker_checker"
CHECKER_SQUARE = 8
CHECKER_LIGHT = (58, 58, 64, 255)
CHECKER_DARK = (44, 44, 50, 255)


def _slot(uid: str, name: str) -> str:
    return f"inker_tex:{uid}:{name}"


def _forget(ctx: Any, texture: Any) -> None:
    from .. import imgui_backend

    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.forget_texture(texture)
    texture.release()


def _cached(ctx: Any, key: str, size: tuple[int, int], data: Callable[[], bytes]) -> Any:
    """Create or resize a texture in a named slot.

    ``data`` is a thunk, not bytes: the common frame -- the texture exists at
    the right size and nothing changed -- must cost no pixel copy, and eagerly
    flattening a megapixel composite to pass in here was a full-canvas copy
    per frame whether or not it was used.
    """
    gl = ctx.viewer.ctx
    texture = ctx.state.preview.get(key)
    if texture is not None and texture.size != size:
        _forget(ctx, texture)
        texture = None
        ctx.state.preview.pop(f"{key}:rev", None)
    if texture is None:
        texture = gl.texture(size, 4, data())
        ctx.state.preview[key] = texture
    return texture


def composite(ctx: Any, tab: Any, *, nearest: bool) -> Any:
    """The document's composite, uploaded only where it changed."""
    if ctx.viewer is None:
        return None
    doc = tab.doc
    key = _slot(tab.uid, "composite")
    rev_key = f"{key}:rev"
    image = doc.image
    fresh = ctx.state.preview.get(key) is None
    texture = _cached(ctx, key, image.size, image.tobytes)
    region = doc.take_dirty()
    if not fresh and ctx.state.preview.get(rev_key) != doc.rev:
        if region is not None:
            x0, y0, x1, y1 = region
            texture.write(image.crop(region).tobytes(), viewport=(x0, y0, x1 - x0, y1 - y0))
        else:
            texture.write(image.tobytes())
    ctx.state.preview[rev_key] = doc.rev
    mode = ctx.viewer.ctx.NEAREST if nearest else ctx.viewer.ctx.LINEAR
    texture.filter = (mode, mode)
    return texture


def floating(ctx: Any, tab: Any, *, nearest: bool) -> Any:
    """The floating buffer, which is drawn over the composite rather than in
    it -- it is not part of any layer until it is committed."""
    if ctx.viewer is None or tab.doc.floating is None:
        return None
    buf = tab.doc.floating
    key = _slot(tab.uid, "floating")
    rev_key = f"{key}:rev"
    fresh = ctx.state.preview.get(key) is None
    texture = _cached(ctx, key, buf.size, buf.pixels.tobytes)
    if not fresh and ctx.state.preview.get(rev_key) != buf.rev:
        texture.write(buf.pixels.tobytes())
    ctx.state.preview[rev_key] = buf.rev
    mode = ctx.viewer.ctx.NEAREST if nearest else ctx.viewer.ctx.LINEAR
    texture.filter = (mode, mode)
    return texture


def layer_thumb(ctx: Any, tab: Any, index: int, size: int = 48) -> Any:
    """A small preview of one layer for the layers panel.

    Keyed by the layer's uid rather than its index, so a reorder does not show
    every layer the previous layer's picture; refreshed on the document's
    revision, which is coarse but bounded -- a handful of 48-square uploads on
    a frame where something changed.
    """
    if ctx.viewer is None or index >= len(tab.doc.stack):
        return None
    import numpy as np
    from PIL import Image

    layer = tab.doc.stack[index]
    key = _slot(tab.uid, f"thumb{layer.uid}")
    rev_key = f"{key}:rev"
    stamp = (tab.doc.rev, layer.uid)
    texture = ctx.state.preview.get(key)
    if texture is not None and ctx.state.preview.get(rev_key) == stamp:
        return texture
    small = Image.fromarray(layer.pixels, "RGBA").resize((size, size), Image.BOX)
    data = np.asarray(small, dtype=np.uint8).tobytes()
    texture = _cached(ctx, key, (size, size), lambda: data)
    texture.write(data)
    ctx.state.preview[rev_key] = stamp
    return texture


def checker(ctx: Any) -> Any:
    """A two-square-by-two-square tile, drawn repeated under the canvas."""
    if ctx.viewer is None:
        return None
    texture = ctx.state.preview.get(_CHECKER_KEY)
    if texture is not None:
        return texture
    side = CHECKER_SQUARE * 2
    data = bytearray()
    for y in range(side):
        for x in range(side):
            light = (x < CHECKER_SQUARE) == (y < CHECKER_SQUARE)
            data.extend(CHECKER_LIGHT if light else CHECKER_DARK)
    texture = ctx.viewer.ctx.texture((side, side), 4, bytes(data))
    texture.repeat_x = texture.repeat_y = True
    texture.filter = (ctx.viewer.ctx.NEAREST, ctx.viewer.ctx.NEAREST)
    ctx.state.preview[_CHECKER_KEY] = texture
    return texture


def release_doc(ctx: Any, uid: str) -> None:
    """Drop every texture belonging to one closed tab."""
    prefix = f"inker_tex:{uid}:"
    for key in [k for k in list(ctx.state.preview) if k.startswith(prefix)]:
        value = ctx.state.preview.pop(key, None)
        if value is not None and hasattr(value, "release"):
            _forget(ctx, value)


def release_all(ctx: Any) -> None:
    for key in [k for k in list(ctx.state.preview) if k.startswith("inker_tex:")]:
        value = ctx.state.preview.pop(key, None)
        if value is not None and hasattr(value, "release"):
            _forget(ctx, value)
    checker_texture = ctx.state.preview.pop(_CHECKER_KEY, None)
    if checker_texture is not None:
        _forget(ctx, checker_texture)
