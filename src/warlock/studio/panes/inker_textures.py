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

import time
from collections.abc import Callable
from typing import Any

from .. import docmodes

# How often one layer's panel thumbnail may re-render while its pixels keep
# changing (B24). During a stroke ``doc.rev`` ticks per dab, and every tick
# used to re-shrink and re-upload *every* layer's 48-square -- per frame. The
# panel draws every frame, so a throttled thumb still catches up within a
# quarter second of the stroke ending.
THUMB_REFRESH_SECONDS = 0.25

# The checkerboard behind transparency. One small texture drawn tiled, so the
# pattern costs a single quad however far the canvas is zoomed out.
_CHECKER_KEY = "inker_checker"
CHECKER_SQUARE = 8
CHECKER_LIGHT = (58, 58, 64, 255)
CHECKER_DARK = (44, 44, 50, 255)


def _slot(uid: str, name: str) -> str:
    return f"inker_tex:{uid}:{name}"


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
        docmodes.forget_texture(texture)
        texture = None
        ctx.state.preview.pop(f"{key}:rev", None)
    if texture is None:
        texture = gl.texture(size, 4, data())
        ctx.state.preview[key] = texture
    return texture


def release_dropped(ctx: Any, tab: Any) -> None:
    """Free the textures of frames the document no longer has.

    A deleted frame is simply never asked for again, so without this its
    texture lives until the tab is closed -- a clip built up and cut down over a
    session leaks one full-canvas texture per frame that was ever deleted.
    ``Document.take_dropped_frames`` is a drain, so the usual frame costs one
    empty list.

    Called from :func:`composite`, which every drawn frame of the active tab
    goes through. A background tab's deletions therefore wait until it is
    activated or closed, and ``release_doc``'s prefix sweep covers the close.
    """
    if ctx.viewer is None:
        return
    for frame_uid in tab.doc.take_dropped_frames():
        key = _slot(tab.uid, f"frame{frame_uid}")
        texture = ctx.state.preview.pop(key, None)
        ctx.state.preview.pop(f"{key}:rev", None)
        if texture is not None:
            docmodes.forget_texture(texture)


def composite(ctx: Any, tab: Any, *, nearest: bool) -> Any:
    """The document's composite, uploaded only where it changed."""
    if ctx.viewer is None:
        return None
    release_dropped(ctx, tab)
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


def frame_texture(ctx: Any, tab: Any, frame_uid: int) -> Any:
    """One animation frame's flatten, for onion skinning and playback.

    Inside the existing ``inker_tex:{uid}:`` naming on purpose, so
    ``release_doc``'s prefix sweep collects a closed tab's frames without
    knowing they exist -- a tab left open on a fifty-frame clip is fifty
    textures, which is exactly the accumulation that sweep is for.

    Keyed on the *frame's* stamp rather than on ``doc.rev``: rev moves for any
    change anywhere, so every onion-skinned neighbour would re-upload on every
    dab the user made on the frame between them.
    """
    if ctx.viewer is None:
        return None
    doc = tab.doc
    pixels = doc.frame_flat(frame_uid)
    if pixels is None:
        return None
    key = _slot(tab.uid, f"frame{frame_uid}")
    rev_key = f"{key}:rev"
    stamp = doc.frame_stamp(frame_uid)
    height, width = pixels.shape[:2]
    fresh = ctx.state.preview.get(key) is None
    texture = _cached(ctx, key, (width, height), pixels.tobytes)
    if not fresh and ctx.state.preview.get(rev_key) != stamp:
        texture.write(pixels.tobytes())
    ctx.state.preview[rev_key] = stamp
    mode = ctx.viewer.ctx.NEAREST
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
    at_key = f"{key}:at"
    stamp = (tab.doc.rev, layer.uid)
    texture = ctx.state.preview.get(key)
    if texture is not None and ctx.state.preview.get(rev_key) == stamp:
        return texture
    now = time.monotonic()
    if texture is not None and now - float(ctx.state.preview.get(at_key) or 0.0) < (
        THUMB_REFRESH_SECONDS
    ):
        # Stale but recent (B24): keep showing the last shrink and try again
        # on a later frame -- the stamp mismatch persists until we catch up.
        return texture
    small = Image.fromarray(layer.pixels, "RGBA").resize((size, size), Image.BOX)
    data = np.asarray(small, dtype=np.uint8).tobytes()
    texture = _cached(ctx, key, (size, size), lambda: data)
    texture.write(data)
    ctx.state.preview[rev_key] = stamp
    ctx.state.preview[at_key] = now
    return texture


#: How many cel thumbnails one tab may hold at once. A 36-square RGBA texture
#: is 5.2 KB, so this is about 2.6 MB per tab -- and a fifty-frame clip with
#: ten tracks is five hundred cells, which is why the cap is a number rather
#: than "all of them". Least-recently-drawn goes first, which during a scroll
#: along the timeline is exactly the column that has left the screen.
CEL_THUMB_CAP = 512


def _cel_lru(ctx: Any, uid: str) -> list[str]:
    key = f"inker_tex:{uid}:cel-lru"
    order = ctx.state.preview.get(key)
    if order is None:
        order = []
        ctx.state.preview[key] = order
    return order


def _cel_touched(ctx: Any, uid: str) -> dict[str, int]:
    """Which imgui frame each cel thumbnail was last drawn on.

    Inside the ``inker_tex:{uid}:`` naming for ``release_doc``'s reason: the
    prefix sweep drops it with the tab. Not a texture, so the sweep's
    ``hasattr(value, "release")`` leaves it alone beyond the pop.
    """
    key = f"inker_tex:{uid}:cel-touched"
    touched = ctx.state.preview.get(key)
    if touched is None:
        touched = {}
        ctx.state.preview[key] = touched
    return touched


def _frame_number() -> int | None:
    """This imgui frame's number, or ``None`` off-context.

    ``None`` means "evict immediately": with no draw list there is nothing to
    protect, which is the headless answer ``motion._clock`` gives the same way.

    The context check comes *before* the call, not as a ``try`` around it
    (``widgets._has_context``'s rule, and its exact reason): ``get_frame_count``
    with no context is not an exception, it is an access violation -- imgui's
    null check is an assert compiled out of the release build -- so wrapping it
    catches nothing and the process dies. ``get_current_context`` is the one
    entry point that is safe to call first.
    """
    try:
        from imgui_bundle import imgui

        if imgui.get_current_context() is None:
            return None
        return int(imgui.get_frame_count())
    except Exception:
        return None


def cel_thumb(ctx: Any, tab: Any, layer: Any, size: int = 36) -> Any:
    """A small preview of one cel, for a timeline cell.

    Keyed on the *layer's* uid, so two slots holding one object share one
    texture -- which is the link made visible for free, and the reason this is
    not keyed on the slot. Keyed on the layer's own stamp rather than on
    ``doc.rev`` or the frame stamp for ``frame_texture``'s reason one level
    down: those move when any cel in the document (or any track in the column)
    changes, and every cel on screen would re-shrink on a single dab.

    Rides inside the existing ``inker_tex:{uid}:`` naming so ``release_doc``'s
    prefix sweep collects a closed tab's cels without knowing they exist, and
    is bounded by ``CEL_THUMB_CAP`` because that sweep only runs at close and a
    long session on a big clip would otherwise accumulate one texture per cell
    that was ever scrolled past.
    """
    if ctx.viewer is None or layer is None:
        return None
    import numpy as np
    from PIL import Image

    doc = tab.doc
    if doc.anim is not None and doc.anim.is_placeholder(layer):
        return None
    key = _slot(tab.uid, f"cel{layer.uid}")
    rev_key = f"{key}:rev"
    at_key = f"{key}:at"
    order = _cel_lru(ctx, tab.uid)
    if key in order:
        order.remove(key)
    order.append(key)
    touched = _cel_touched(ctx, tab.uid)
    frame = _frame_number()
    if frame is not None:
        touched[key] = frame
    _evict_cel_thumbs(ctx, order, touched, frame)

    stamp = (doc.layer_stamp(layer.uid), size)
    texture = ctx.state.preview.get(key)
    if texture is not None and ctx.state.preview.get(rev_key) == stamp:
        return texture
    now = time.monotonic()
    if texture is not None and now - float(ctx.state.preview.get(at_key) or 0.0) < (
        THUMB_REFRESH_SECONDS
    ):
        # Stale but recent (B24), exactly as ``layer_thumb``: keep showing the
        # last shrink while a stroke is in flight and catch up within a quarter
        # second of it ending.
        return texture
    small = Image.fromarray(layer.pixels, "RGBA").resize((size, size), Image.BOX)
    data = np.asarray(small, dtype=np.uint8).tobytes()
    texture = _cached(ctx, key, (size, size), lambda: data)
    texture.write(data)
    texture.filter = (ctx.viewer.ctx.NEAREST, ctx.viewer.ctx.NEAREST)
    ctx.state.preview[rev_key] = stamp
    ctx.state.preview[at_key] = now
    return texture


def _evict_cel_thumbs(
    ctx: Any, order: list[str], touched: dict[str, int], frame: int | None
) -> None:
    """Drop the least recently drawn cel thumbnails past the cap.

    Never one drawn this frame -- ``ThumbnailCache._evict``'s rule: a thumbnail
    handed out during this frame's UI build already has an ``add_image`` in the
    live draw list, and releasing it now frees a texture the backend is about
    to draw. With more cells visible than the cap holds, the overshoot is
    bounded by what fits on screen and drains as soon as the timeline scrolls.
    """
    for key in list(order):
        if len(order) <= CEL_THUMB_CAP:
            return
        if frame is not None and touched.get(key) == frame:
            continue
        order.remove(key)
        touched.pop(key, None)
        texture = ctx.state.preview.pop(key, None)
        ctx.state.preview.pop(f"{key}:rev", None)
        ctx.state.preview.pop(f"{key}:at", None)
        if texture is not None:
            docmodes.forget_texture(texture)


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


#: Everything else the panes cache per tab under ``state.preview``, as key
#: prefixes. Textures are the entries that must be *released*; these merely
#: have to go, or a long session accumulates one marching-ants trace and one
#: resize form per tab that was ever opened.
_PER_TAB_KEYS = ("paint_ants:", "inker_resize:", "inker_preview:")


def release_doc(ctx: Any, uid: str) -> None:
    """Drop every texture belonging to one closed tab, and its cached state."""
    docmodes.release_prefix(ctx, f"inker_tex:{uid}:")
    for name in _PER_TAB_KEYS:
        ctx.state.preview.pop(f"{name}{uid}", None)


def release_all(ctx: Any) -> None:
    docmodes.release_prefix(ctx, "inker_tex:")
    for key in [
        k for k in list(ctx.state.preview) if k.startswith(_PER_TAB_KEYS)
    ]:
        ctx.state.preview.pop(key, None)
    checker_texture = ctx.state.preview.pop(_CHECKER_KEY, None)
    if checker_texture is not None:
        docmodes.forget_texture(checker_texture)
