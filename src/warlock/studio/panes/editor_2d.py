"""The 2D art editor: an MS-Paint-class pass over a generated reference.

It takes over the centre pane rather than opening a window, because it is a
*mode* over one asset -- both sidebars stay, so the library and the inspector
still say which job is being edited. Everything about pixels lives in
``studio/paint.py``; this file is input, layout and the two textures the canvas
is drawn with.

The document is decoded on the frame thread when the editor opens (a reference
is a small PNG, and ``ThumbnailCache`` already sets that precedent) and encoded
off it when it saves, because a save also re-measures the reference and touches
the database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from .. import dialogs, paint, theme, widgets

TOOLS = (
    ("brush", "Brush"),
    ("eraser", "Eraser"),
    ("fill", "Fill"),
    ("line", "Line"),
    ("rect", "Rect"),
    ("ellipse", "Ellipse"),
    ("select", "Select"),
    ("eyedropper", "Pick"),
)

SHAPE_TOOLS = {"line", "rect", "ellipse"}

# A small fixed palette. Not a preference and not persisted: the eyedropper is
# how a user gets the colours that are actually in their image, and this row
# only has to cover the ones that never are.
PALETTE = (
    (0, 0, 0, 255),
    (255, 255, 255, 255),
    (128, 128, 128, 255),
    (200, 40, 40, 255),
    (230, 140, 40, 255),
    (230, 210, 60, 255),
    (60, 170, 80, 255),
    (60, 130, 220, 255),
    (140, 70, 200, 255),
    (110, 70, 40, 255),
)

MIN_ZOOM = 0.25
MAX_ZOOM = 16.0

_TEXTURE_KEY = "editor_texture"
_TEXTURE_REV = "editor_texture_rev"
_SEL_TEXTURE_KEY = "editor_sel_texture"
_SEL_TEXTURE_REV = "editor_sel_rev"


def _u32(colour: int, alpha: float = 1.0) -> int:
    return imgui.get_color_u32(imgui.ImVec4(*theme.rgba(colour, alpha)))


def _to_rgba(value: Any) -> paint.RGBA:
    """imgui's float colour -> the 8-bit tuple Pillow draws with."""
    return (
        int(round(value.x * 255)),
        int(round(value.y * 255)),
        int(round(value.z * 255)),
        int(round(value.w * 255)),
    )


# --- opening and closing ----------------------------------------------------


def can_edit(ctx: Any, job: Any) -> bool:
    """Whether the Edit button should appear, from the cached row alone.

    No filesystem calls: this is asked every frame from the toolbar.
    """
    return bool(
        job
        and ctx.state.mode == "2d"
        and job.get("stage") == "reference"
        and job.get("status") == "done"
        and "input.png" in (job.get("files") or [])
    )


def open(ctx: Any, job: Any) -> None:  # noqa: A001 - the verb is the point
    from ...service import files as svc_files

    job_id = job["id"]
    path = ctx.job_dir(job_id) / "input.png"
    try:
        doc = paint.Document.load(path)
    except Exception as exc:
        ctx.toast(f"Could not open that image ({exc}).", "error")
        return
    session = paint.EditorSession(job_id=job_id, path=path, doc=doc)
    # Two stats, which is why it is allowed on the frame thread at all.
    status = svc_files.reference_edit_status(ctx.svc, job_id)
    session.has_original = bool(status.get("has_original"))
    ctx.state.editor = session


def close(ctx: Any) -> None:
    _release_textures(ctx)
    ctx.state.editor = None


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before dropping unsaved pixels. -> whether it went ahead now."""
    session = ctx.state.editor
    if session is None or not session.dirty:
        proceed()
        return True
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved edits?",
            message=f"The changes to this image will be lost if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False


def request_close(ctx: Any) -> None:
    guard(ctx, "close the editor", lambda: close(ctx))


# --- keys -------------------------------------------------------------------


def handle_key(ctx: Any, event: Any) -> bool:
    """The editor's shortcuts. -> whether the key was consumed.

    Consuming unconditionally while the editor is open is deliberate: F, W and
    S would otherwise frame and wireframe a viewport that is not on screen.
    """
    import pygame

    session = ctx.state.editor
    if session is None:
        return False
    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    doc = session.doc
    if ctrl and event.key == pygame.K_z and not shift:
        session.dirty = doc.undo() or session.dirty
    elif ctrl and (event.key == pygame.K_y or (event.key == pygame.K_z and shift)):
        session.dirty = doc.redo() or session.dirty
    elif ctrl and event.key == pygame.K_s:
        save(ctx)
    elif event.key == pygame.K_LEFTBRACKET:
        session.brush_size = paint.clamp_brush(session.brush_size - 1)
    elif event.key == pygame.K_RIGHTBRACKET:
        session.brush_size = paint.clamp_brush(session.brush_size + 1)
    elif event.key == pygame.K_DELETE:
        if doc.selection is not None:
            doc.delete_selection()
            session.dirty = True
    elif event.key == pygame.K_ESCAPE:
        # The selection first: Esc on a floating chunk means "put that back",
        # not "throw away the whole edit".
        if doc.selection is not None:
            doc.cancel_selection()
        else:
            request_close(ctx)
    return True


# --- drawing ----------------------------------------------------------------


def draw(ctx: Any) -> None:
    session = ctx.state.editor
    if session is None:
        return
    _toolbar(ctx, session)
    imgui.separator()
    _canvas(ctx, session)


def _toolbar(ctx: Any, session: paint.EditorSession) -> None:
    doc = session.doc
    for key, label in TOOLS:
        if imgui.radio_button(label, session.tool == key):
            if key != "select":
                doc.commit_selection()
            session.tool = key
        imgui.same_line()
    imgui.new_line()

    changed, rgba = imgui.color_edit4(
        "Colour",
        imgui.ImVec4(*[c / 255.0 for c in session.color]),
        imgui.ColorEditFlags_.no_inputs.value | imgui.ColorEditFlags_.alpha_bar.value,
    )
    if changed:
        session.color = _to_rgba(rgba)
    imgui.same_line()
    for index, swatch in enumerate(PALETTE):
        imgui.push_id(f"swatch{index}")
        if imgui.color_button(
            "##swatch", imgui.ImVec4(*[c / 255.0 for c in swatch]), 0, (18, 18)
        ):
            session.color = swatch
        imgui.pop_id()
        imgui.same_line()
    imgui.new_line()

    imgui.set_next_item_width(160)
    changed, size = imgui.slider_int("Size", session.brush_size, paint.MIN_BRUSH, paint.MAX_BRUSH)
    if changed:
        session.brush_size = paint.clamp_brush(size)
    if session.tool in ("rect", "ellipse"):
        imgui.same_line()
        changed, filled = imgui.checkbox("Filled", session.shape_filled)
        if changed:
            session.shape_filled = filled

    if widgets.disabled_button("Undo", doc.history.can_undo):
        session.dirty = doc.undo() or session.dirty
    imgui.same_line()
    if widgets.disabled_button("Redo", doc.history.can_redo):
        session.dirty = doc.redo() or session.dirty
    imgui.same_line()
    if imgui.button("Fit"):
        session.fitted = False
    imgui.same_line()
    widgets.muted(f"{doc.size[0]}x{doc.size[1]} - {session.zoom * 100:.0f}%")

    imgui.same_line()
    if widgets.disabled_button("Save", session.dirty and not session.saving):
        save(ctx)
    imgui.same_line()
    if widgets.disabled_button("Revert to original", session.has_original and not session.saving):
        _ask_revert(ctx, session)
    imgui.same_line()
    if imgui.button("Close"):
        request_close(ctx)
    if session.dirty:
        imgui.same_line()
        widgets.text_colored(theme.WARN, "unsaved")


def _canvas(ctx: Any, session: paint.EditorSession) -> None:
    flags = (
        imgui.WindowFlags_.no_scroll_with_mouse.value | imgui.WindowFlags_.no_scrollbar.value
    )
    if imgui.begin_child("editor-canvas", (0, 0), imgui.ChildFlags_.borders.value, flags):
        origin = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        region = (max(avail.x, 16.0), max(avail.y, 16.0))
        if not session.fitted:
            _fit(session, region)
        imgui.invisible_button("##canvas", region)
        active = imgui.is_item_active()
        hovered = imgui.is_item_hovered()
        _input(session, (origin.x, origin.y), active=active, hovered=hovered)
        _paint_canvas(ctx, session, (origin.x, origin.y))
    imgui.end_child()


def _fit(session: paint.EditorSession, region: tuple[float, float]) -> None:
    width, height = session.doc.size
    zoom = min(region[0] / width, region[1] / height)
    session.zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
    session.pan = (
        (region[0] - width * session.zoom) * 0.5,
        (region[1] - height * session.zoom) * 0.5,
    )
    session.fitted = True


def _to_image(session: paint.EditorSession, origin: tuple[float, float], sx: float, sy: float):
    return (
        int((sx - origin[0] - session.pan[0]) / session.zoom),
        int((sy - origin[1] - session.pan[1]) / session.zoom),
    )


def _to_screen(session: paint.EditorSession, origin: tuple[float, float], x: float, y: float):
    return (
        origin[0] + session.pan[0] + x * session.zoom,
        origin[1] + session.pan[1] + y * session.zoom,
    )


# --- input ------------------------------------------------------------------


def _input(
    session: paint.EditorSession,
    origin: tuple[float, float],
    *,
    active: bool,
    hovered: bool,
) -> None:
    io = imgui.get_io()
    mouse = imgui.get_mouse_pos()
    point = _to_image(session, origin, mouse.x, mouse.y)

    if hovered and io.mouse_wheel:
        _zoom_about(session, origin, (mouse.x, mouse.y), io.mouse_wheel)
    if (hovered or session.drag_kind == "pan") and imgui.is_mouse_dragging(2):
        delta = imgui.get_mouse_drag_delta(2)
        imgui.reset_mouse_drag_delta(2)
        session.drag_kind = "pan"
        session.pan = (session.pan[0] + delta.x, session.pan[1] + delta.y)
        return
    if session.drag_kind == "pan" and not imgui.is_mouse_down(2):
        session.drag_kind = ""

    if active and imgui.is_mouse_clicked(0):
        _press(session, point)
    elif session.drag_kind and imgui.is_mouse_down(0):
        _drag(session, point)
    elif session.drag_kind and not imgui.is_mouse_down(0):
        _release(session, point)


def _zoom_about(
    session: paint.EditorSession,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    wheel: float,
) -> None:
    before = session.zoom
    after = max(MIN_ZOOM, min(MAX_ZOOM, before * (1.15**wheel)))
    if after == before:
        return
    # Keep whatever pixel is under the cursor under the cursor.
    local = (mouse[0] - origin[0], mouse[1] - origin[1])
    ratio = after / before
    session.zoom = after
    session.pan = (
        local[0] - (local[0] - session.pan[0]) * ratio,
        local[1] - (local[1] - session.pan[1]) * ratio,
    )


def _press(session: paint.EditorSession, point: tuple[int, int]) -> None:
    doc = session.doc
    tool = session.tool
    session.drag_anchor = point
    session.last_point = point
    if tool == "eyedropper":
        picked = doc.eyedrop(point)
        if picked is not None:
            session.color = picked
        session.drag_kind = ""
        return
    if tool == "fill":
        doc.checkpoint()
        doc.fill(point, session.color)
        session.dirty = True
        session.drag_kind = ""
        return
    if tool == "select":
        if doc.selection is not None and _inside_selection(doc.selection, point):
            session.drag_kind = "move"
        else:
            doc.commit_selection()
            session.drag_kind = "marquee"
        return
    if tool in SHAPE_TOOLS:
        session.drag_kind = "shape"
        return
    # brush / eraser: paint immediately, so a click is a dot rather than
    # nothing at all.
    doc.commit_selection()
    doc.checkpoint()
    doc.stroke(point, point, session.color, session.brush_size, erase=tool == "eraser")
    session.dirty = True
    session.drag_kind = "paint"


def _drag(session: paint.EditorSession, point: tuple[int, int]) -> None:
    doc = session.doc
    if session.drag_kind == "paint":
        last = session.last_point or point
        doc.stroke(last, point, session.color, session.brush_size, erase=session.tool == "eraser")
        session.last_point = point
        session.dirty = True
    elif session.drag_kind == "move" and doc.selection is not None:
        last = session.last_point or point
        doc.move_selection(point[0] - last[0], point[1] - last[1])
        session.last_point = point
        session.dirty = True


def _release(session: paint.EditorSession, point: tuple[int, int]) -> None:
    doc = session.doc
    anchor = session.drag_anchor or point
    if session.drag_kind == "shape":
        doc.checkpoint()
        doc.shape(
            session.tool,
            anchor,
            point,
            session.color,
            session.brush_size,
            filled=session.shape_filled,
        )
        session.dirty = True
    elif session.drag_kind == "marquee":
        rect = paint.normalise_rect(anchor, point)
        if doc.lift_selection((rect[0], rect[1], rect[2] + 1, rect[3] + 1)):
            session.dirty = True
    session.drag_kind = ""
    session.drag_anchor = None
    session.last_point = None


def _inside_selection(selection: Any, point: tuple[int, int]) -> bool:
    x, y = point
    ox, oy = selection.origin
    width, height = selection.size
    return ox <= x < ox + width and oy <= y < oy + height


# --- the picture itself -----------------------------------------------------


def _paint_canvas(ctx: Any, session: paint.EditorSession, origin: tuple[float, float]) -> None:
    doc = session.doc
    draw_list = imgui.get_window_draw_list()
    texture = _document_texture(ctx, session)
    if texture is None:
        return
    width, height = doc.size
    top_left = _to_screen(session, origin, 0, 0)
    bottom_right = _to_screen(session, origin, width, height)
    draw_list.add_image(widgets.texture_ref(texture), top_left, bottom_right)
    draw_list.add_rect(top_left, bottom_right, _u32(theme.EDGE))

    if doc.selection is not None:
        _draw_selection(ctx, session, draw_list, origin)
    if session.drag_kind in ("shape", "marquee") and session.drag_anchor is not None:
        _draw_preview(session, draw_list, origin)


def _draw_selection(ctx: Any, session: paint.EditorSession, draw_list: Any, origin) -> None:
    selection = session.doc.selection
    texture = _selection_texture(ctx, session)
    x, y = selection.origin
    width, height = selection.size
    a = _to_screen(session, origin, x, y)
    b = _to_screen(session, origin, x + width, y + height)
    if texture is not None:
        draw_list.add_image(widgets.texture_ref(texture), a, b)
    draw_list.add_rect(a, b, _u32(theme.ACCENT))


def _draw_preview(session: paint.EditorSession, draw_list: Any, origin) -> None:
    mouse = imgui.get_mouse_pos()
    anchor = _to_screen(session, origin, *session.drag_anchor)
    colour = _u32(theme.ACCENT)
    if session.drag_kind == "marquee" or session.tool == "rect":
        draw_list.add_rect(anchor, (mouse.x, mouse.y), colour)
    elif session.tool == "line":
        draw_list.add_line(anchor, (mouse.x, mouse.y), colour)
    elif session.tool == "ellipse":
        centre = ((anchor[0] + mouse.x) * 0.5, (anchor[1] + mouse.y) * 0.5)
        radii = (abs(mouse.x - anchor[0]) * 0.5, abs(mouse.y - anchor[1]) * 0.5)
        draw_list.add_ellipse(centre, radii, colour)


# --- textures ---------------------------------------------------------------


def _document_texture(ctx: Any, session: paint.EditorSession) -> Any:
    return _sync_texture(
        ctx,
        _TEXTURE_KEY,
        _TEXTURE_REV,
        session.doc.image,
        session.doc.rev,
        nearest=session.zoom >= 1.0,
    )


def _selection_texture(ctx: Any, session: paint.EditorSession) -> Any:
    selection = session.doc.selection
    if selection is None:
        return None
    return _sync_texture(
        ctx,
        _SEL_TEXTURE_KEY,
        _SEL_TEXTURE_REV,
        selection.chunk,
        selection.rev,
        nearest=session.zoom >= 1.0,
        force=True,
    )


def _sync_texture(
    ctx: Any,
    key: str,
    rev_key: str,
    image: Any,
    rev: int,
    *,
    nearest: bool,
    force: bool = False,
) -> Any:
    """One reusable texture per slot, uploaded only when the pixels changed.

    The rev gate is the whole point: uploading a 1024x1024 RGBA image every
    frame is 4 MB of PCIe traffic to show something that did not move.
    """
    if ctx.viewer is None:
        return None
    gl = ctx.viewer.ctx
    cached = ctx.state.preview.get(key)
    if cached is not None and cached.size != image.size:
        _forget(cached)
        cached = None
        ctx.state.preview.pop(rev_key, None)
    if cached is None:
        cached = gl.texture(image.size, 4, image.tobytes())
        ctx.state.preview[key] = cached
        ctx.state.preview[rev_key] = rev
    elif force or ctx.state.preview.get(rev_key) != rev:
        cached.write(image.tobytes())
        ctx.state.preview[rev_key] = rev
    mode = gl.NEAREST if nearest else gl.LINEAR
    cached.filter = (mode, mode)
    return cached


def _forget(texture: Any) -> None:
    from .. import imgui_backend

    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.forget_texture(texture)
    texture.release()


def _release_textures(ctx: Any) -> None:
    for key, rev_key in ((_TEXTURE_KEY, _TEXTURE_REV), (_SEL_TEXTURE_KEY, _SEL_TEXTURE_REV)):
        texture = ctx.state.preview.pop(key, None)
        ctx.state.preview.pop(rev_key, None)
        if texture is not None:
            _forget(texture)


# --- save and revert --------------------------------------------------------


def save(ctx: Any) -> None:
    from ...service import files as svc_files

    session = ctx.state.editor
    if session is None or session.saving:
        return
    session.doc.commit_selection()
    # Copied on the frame thread, encoded on the task thread: the document
    # keeps being editable while the PNG is written.
    image = session.doc.image.copy()
    session.saving = True

    def run() -> str:
        import io

        buf = io.BytesIO()
        image.save(buf, "PNG")
        svc_files.save_edited_image(ctx.svc, session.job_id, buf.getvalue())
        return session.job_id

    if not ctx.submit(f"editor-save:{session.job_id}", run):
        session.saving = False


def _ask_revert(ctx: Any, session: paint.EditorSession) -> None:
    from ...service import files as svc_files

    def run() -> str:
        svc_files.revert_reference(ctx.svc, session.job_id)
        return session.job_id

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Revert to the original?",
            message="The generated image comes back and every edit to it is lost.",
            confirm_label="Revert",
            cancel_label="Keep editing",
            on_confirm=lambda: _start_revert(ctx, session, run),
        )
    )


def _start_revert(ctx: Any, session: paint.EditorSession, run: Any) -> None:
    session.saving = True
    if not ctx.submit(f"editor-revert:{session.job_id}", run):
        session.saving = False


def on_saved(ctx: Any, job_id: str, *, reverted: bool = False) -> None:
    """The task came back. Called from App._on_task_done."""
    session = ctx.state.editor
    ctx.cache.invalidate()
    if session is None or session.job_id != job_id:
        return
    session.saving = False
    if reverted:
        try:
            session.doc = paint.Document.load(session.path)
        except Exception as exc:
            ctx.toast(f"Reverted, but the image could not be reopened ({exc}).", "error")
            return
        session.has_original = False
        session.dirty = False
        session.fitted = False
        ctx.toast("Back to the original image.")
    else:
        session.dirty = False
        session.has_original = True
        ctx.toast("Saved.")
    _reload_viewer(ctx, session.path)


def _reload_viewer(ctx: Any, path: Path) -> None:
    """_sync_viewer short-circuits when the path has not changed, so an
    in-place rewrite of input.png would otherwise leave the stale texture up."""
    viewer = ctx.viewer
    if viewer is not None and viewer.path == path:
        viewer.clear()
        viewer.load_reference(path)
