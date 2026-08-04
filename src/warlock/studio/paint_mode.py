"""Paint mode's controller: opening, saving, guarding, keys, and the bridge.

Everything here is *about* documents rather than pixels -- the engine under
``paint/`` has no idea a job or a task thread exists, and this is the layer that
knows about both. The panes draw; this decides.

The one rule that shapes the whole file: **no file dialog and no encode ever
runs on the frame thread.** A native picker is modal to the OS and blocks until
dismissed, and a 4096-square ORA is a second of zlib. Both go through
``ctx.submit`` and come back through ``on_task_done``, which is why saving is a
state (``PaintDoc.saving``) rather than a function call that returns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import dialogs, paint_state
from .paint_state import PaintDoc, PaintState

log = logging.getLogger(__name__)

ORA_FILTER = ["OpenRaster (*.ora)", "*.ora"]
OPEN_FILTER = ["Images and layered files", "*.ora *.png *.jpg *.jpeg *.webp *.bmp"]
PNG_FILTER = ["PNG image (*.png)", "*.png"]

OPENABLE = (".ora", ".png", ".jpg", ".jpeg", ".webp", ".bmp")

NEW_PRESETS = ((512, 512), (1024, 1024), (2048, 2048))


def ensure(ctx: Any) -> PaintState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Paint should not pay for its
    swatches, and because ``AppState`` deliberately knows nothing about it.
    """
    state = ctx.state.paint
    if state is None:
        state = PaintState()
        stored = ctx.settings.get("paint") or {}
        state.recent = [p for p in (stored.get("recent") or []) if isinstance(p, str)]
        swatches = stored.get("swatches")
        if isinstance(swatches, list) and swatches:
            state.swatches = [
                tuple(int(c) for c in s)  # type: ignore[misc]
                for s in swatches
                if isinstance(s, list | tuple) and len(s) == 4
            ] or list(paint_state.DEFAULT_SWATCHES)
        ctx.state.paint = state
    return state


def persist(ctx: Any) -> None:
    state = ctx.state.paint
    if state is not None:
        ctx.settings.set(
            "paint",
            {"recent": state.recent, "swatches": [list(s) for s in state.swatches]},
        )


def active(ctx: Any) -> PaintDoc | None:
    state = ctx.state.paint
    return state.active if state is not None else None


# --- opening ----------------------------------------------------------------


def new_document(ctx: Any, width: int, height: int) -> PaintDoc:
    from . import paint

    state = ensure(ctx)
    doc = paint.Document.blank(int(width), int(height))
    return _adopt(ctx, state, doc, path=None, title="Untitled", file_format="ora")


def _adopt(
    ctx: Any,
    state: PaintState,
    doc: Any,
    *,
    path: Path | None,
    title: str | None = None,
    file_format: str = "png",
    job_id: str = "",
    link_kind: str = "",
    has_original: bool = False,
) -> PaintDoc:
    tab = PaintDoc(
        doc=doc,
        title=title or paint_state.title_for(path),
        path=path,
        file_format=file_format,
        saved_head=doc.history.head,
        job_id=job_id,
        link_kind=link_kind,
        has_original=has_original,
    )
    state.add(tab)
    state.remember(path)
    persist(ctx)
    return tab


def ask_open(ctx: Any) -> None:
    """The picker, on a task thread, then the decode on the same one."""
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open image", OPEN_FILTER)
        return None if path is None else _load(path)

    ctx.submit("paint-open", run)


def open_path(ctx: Any, path: Path) -> None:
    """Open a known path -- a drop, or a click in the recent list."""
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one file would race on save.
        state.activate(existing.uid)
        return
    if path.suffix.lower() not in OPENABLE:
        ctx.toast("Paint opens images and .ora files.", "error")
        return
    ctx.submit(f"paint-open:{abs(hash(str(path)))}", _load, path)


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only."""
    from . import paint

    path = Path(path)
    doc = paint.Document.load(path)
    return {"doc": doc, "path": path, "format": doc.file_format}


# --- the job bridge ---------------------------------------------------------


def can_edit_job(ctx: Any, job: Any) -> bool:
    """Whether the "Open in Paint" button belongs on this job's toolbar.

    From the cached row alone -- no filesystem calls, because the toolbar asks
    this every frame.
    """
    return bool(
        job
        and job.get("stage") == "reference"
        and job.get("status") == "done"
        and "input.png" in (job.get("files") or [])
    )


def open_job_reference(ctx: Any, job: Any) -> None:
    """Open a reference's image as a linked document.

    Prefers the layered working file when there is a fresh one, so layers
    survive between sessions; falls back to the flat input.png, which is also
    what happens after a revert or a regenerate rewrites the reference behind
    the working file's back.
    """
    state = ensure(ctx)
    job_id = job["id"]
    existing = state.find_job(job_id)
    if existing is not None:
        state.activate(existing.uid)
        ctx.state.mode = "paint"
        return
    ctx.state.mode = "paint"
    ctx.settings.set("mode", "paint")
    ctx.submit(f"paint-open:{job_id}", _load_job, ctx.svc, job_id)


def _load_job(svc: Any, job_id: str) -> dict[str, Any]:
    """Blocking; task thread only."""
    from ..service import files as svc_files
    from . import paint

    flat = svc.job_dir(job_id) / "input.png"
    working = svc_files.paint_working_path(svc, job_id)
    status = svc_files.paint_working_status(svc, job_id)
    doc = paint.Document.load(working if status["fresh"] else flat)
    # The document is *about* input.png whichever file it was decoded from:
    # the title, the dedupe and the save all key on the reference.
    doc.path = flat
    edit = svc_files.reference_edit_status(svc, job_id)
    return {
        "doc": doc,
        "path": flat,
        "format": "ora",
        "job_id": job_id,
        "link_kind": "reference-edit",
        "has_original": bool(edit.get("has_original")),
        "title": f"{job_id[:8]} reference",
    }


# --- saving -----------------------------------------------------------------


def save(ctx: Any, tab: PaintDoc | None = None) -> None:
    """Ctrl+S. Save As when the document has never been written anywhere."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.linked:
        _save_linked(ctx, tab)
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    _submit_write(ctx, tab, f"paint-save:{tab.uid}", tab.path, tab.file_format)


def save_as(ctx: Any, tab: PaintDoc | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc, rev = tab.doc, tab.doc.history.head
    suggested = tab.path.stem if tab.path else "untitled"

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Save layered document", f"{suggested}.ora", ORA_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".ora":
            dest = dest.with_suffix(".ora")
        _write(doc, dest, "ora")
        return {"path": dest, "rev": rev, "format": "ora", "retitle": True}

    _start(ctx, tab, f"paint-saveas:{tab.uid}", run)


def export_png(ctx: Any, tab: PaintDoc | None = None) -> None:
    """A flattened PNG. Not a save: it does not change what the tab points at,
    so the document stays dirty against its own file."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    suggested = tab.path.stem if tab.path else "untitled"

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Export flattened PNG", f"{suggested}.png", PNG_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        dest.write_bytes(doc.png_bytes())
        return {"exported": dest}

    _start(ctx, tab, f"paint-export:{tab.uid}", run)


def _submit_write(ctx: Any, tab: PaintDoc, key: str, path: Path, file_format: str) -> None:
    doc, rev = tab.doc, tab.doc.history.head

    def run() -> dict[str, Any]:
        _write(doc, path, file_format)
        return {"path": path, "rev": rev, "format": file_format}

    _start(ctx, tab, key, run)


def _write(doc: Any, path: Path, file_format: str) -> None:
    """Blocking; task thread only.

    Encodes the *live* document rather than a copy of it. That is safe because
    the encoders only read, and the frame thread only ever appends to a layer's
    pixels in place -- so the worst case is that the file catches a stroke that
    was mid-flight, which is precisely why the revision is captured before the
    submit rather than after the write.
    """
    from . import paint

    if file_format == "ora":
        paint.write_ora(doc, path)
    else:
        path.write_bytes(doc.png_bytes())


def _start(ctx: Any, tab: PaintDoc, key: str, run: Any) -> None:
    tab.saving = True
    if not ctx.submit(key, run):
        tab.saving = False


def _save_linked(ctx: Any, tab: PaintDoc) -> None:
    """Write both halves of a reference edit: the layers, then the flat PNG.

    The flat write goes through the untouched ``save_edited_image``, so the
    original backup, the ``hand_edited`` param, the reference re-measure and
    the staged replace all still happen exactly as they did for the old inline
    editor. The layered file is a sidecar and is written first, so a crash
    between the two leaves working state that is merely *stale* rather than
    layers that describe pixels nobody has.
    """
    from ..service import files as svc_files

    doc, rev, job_id = tab.doc, tab.doc.history.head, tab.job_id
    doc.commit_floating()

    def run() -> dict[str, Any]:
        from . import paint

        svc_files.save_paint_working(ctx.svc, job_id, paint.ora_bytes(doc))
        svc_files.save_edited_image(ctx.svc, job_id, doc.png_bytes())
        return {"rev": rev, "job_id": job_id, "linked": True}

    _start(ctx, tab, f"paint-save:{tab.uid}", run)


# --- the other direction: paint -> the pipeline ------------------------------


def save_as_reference(ctx: Any, tab: PaintDoc | None = None) -> None:
    """Mint a new reference job from what is on the canvas, and link to it."""
    from ..service import jobs as svc_jobs

    tab = tab or active(ctx)
    if tab is None or tab.saving or tab.linked:
        return
    doc, rev, title = tab.doc, tab.doc.history.head, tab.title
    doc.commit_floating()

    def run() -> dict[str, Any]:
        result = svc_jobs.import_reference(ctx.svc, doc.png_bytes(), name=title)
        job_id = result["id"]
        from . import paint

        # Linked immediately, so the next Ctrl+S saves in place rather than
        # minting a second job from the same pixels.
        svc_files_save(ctx, job_id, paint.ora_bytes(doc))
        return {"rev": rev, "job_id": job_id, "link": True}

    _start(ctx, tab, f"paint-save:{tab.uid}", run)


def svc_files_save(ctx: Any, job_id: str, data: bytes) -> None:
    from ..service import files as svc_files

    svc_files.save_paint_working(ctx.svc, job_id, data)


def send_to_3d(ctx: Any, tab: PaintDoc | None = None) -> None:
    """Take the flattened canvas into the mesh stage.

    A linked document promotes the reference it already is; an unlinked one
    becomes an ordinary image job, which is the same call the 3D pane's upload
    button makes.
    """
    from ..service import jobs as svc_jobs

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    doc.commit_floating()
    if tab.linked:
        if tab.dirty:
            ctx.toast("Save first, so the mesh is made from what you see.", "error")
            return
        _promote(ctx, tab.job_id)
        return

    def run() -> dict[str, Any]:
        return svc_jobs.create_job(
            ctx.svc, kind="image", output="model", image=doc.png_bytes()
        )

    if ctx.submit("paint-send", run):
        ctx.toast("Queued a mesh from the painted image.")


def _promote(ctx: Any, job_id: str) -> None:
    from ..service import errors as svc_errors
    from ..service import jobs as svc_jobs

    def run(force: bool = False) -> dict[str, Any]:
        return svc_jobs.promote_to_model(ctx.svc, job_id, force=force)

    def go(force: bool) -> None:
        if ctx.submit(f"paint-promote:{job_id}", run, force):
            ctx.toast("Queued a mesh from this reference.")

    # The quality gate is a heuristic about composition, not a fact, so it is
    # offered as a confirm rather than a refusal -- the same bargain the 3D
    # pane strikes.
    try:
        report = (ctx.svc.require_job(job_id).get("params") or {}).get("reference_report") or {}
    except svc_errors.ServiceError:
        report = {}
    if report.get("ok") is False:
        reasons = " ".join(report.get("reasons") or ["This image may not reconstruct well."])
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Make a mesh anyway?",
                message=reasons,
                confirm_label="Make it anyway",
                cancel_label="Keep painting",
                on_confirm=lambda: go(True),
            )
        )
        return
    go(False)


def revert(ctx: Any, tab: PaintDoc | None = None) -> None:
    """Put the generated image back, and drop the layers that described the
    edit -- they are about pixels that will no longer exist."""
    from ..service import files as svc_files

    tab = tab or active(ctx)
    if tab is None or not tab.linked or not tab.has_original or tab.saving:
        return
    job_id = tab.job_id

    def run() -> dict[str, Any]:
        svc_files.revert_reference(ctx.svc, job_id)
        svc_files.discard_paint_working(ctx.svc, job_id)
        return {"reverted": True, "job_id": job_id}

    def go() -> None:
        _start(ctx, tab, f"paint-revert:{tab.uid}", run)

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Revert to the original?",
            message="The generated image comes back and every edit to it is lost.",
            confirm_label="Revert",
            cancel_label="Keep editing",
            on_confirm=go,
        )
    )


# --- task results -----------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from App._on_task_done for every ``paint-`` key."""
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name in ("paint-open",):
        if isinstance(result, dict):
            _adopt(
                ctx,
                state,
                result["doc"],
                path=result.get("path"),
                title=result.get("title"),
                file_format=result.get("format", "png"),
                job_id=result.get("job_id", ""),
                link_kind=result.get("link_kind", ""),
                has_original=bool(result.get("has_original")),
            )
            ctx.state.mode = "paint"
        return

    if name in ("paint-send", "paint-promote"):
        ctx.cache.invalidate()
        return

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        ctx.cache.invalidate()
        return
    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported"):
        ctx.toast(f"Exported to {result['exported']}")
        return
    if result.get("reverted"):
        _reload_linked(ctx, tab)
        return

    tab.mark_saved(result.get("rev"))
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = paint_state.title_for(tab.path)
        tab.file_format = result.get("format", "ora")
        state.remember(tab.path)
        persist(ctx)
    if result.get("link"):
        tab.job_id = result["job_id"]
        tab.link_kind = "reference-edit"
        tab.has_original = False
    if tab.linked:
        tab.has_original = True
        ctx.cache.invalidate()
        _nudge_viewer(ctx, tab)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked.

    ``saving`` disables every editing control, so without this a single failed
    write makes the tab permanently read-only with no way back short of
    closing it.
    """
    state = ctx.state.paint
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False


def _reload_linked(ctx: Any, tab: PaintDoc) -> None:
    """Re-decode a linked document after a revert replaced its file."""
    from . import paint

    if tab.path is None:
        return
    try:
        tab.doc = paint.Document.load(tab.path)
    except Exception as exc:
        ctx.toast(f"Reverted, but the image could not be reopened ({exc}).", "error")
        return
    tab.saved_head = tab.doc.history.head
    tab.has_original = False
    tab.view.fitted = False
    ctx.cache.invalidate()
    ctx.toast("Back to the original image.")
    _nudge_viewer(ctx, tab)


def _nudge_viewer(ctx: Any, tab: PaintDoc) -> None:
    """``_sync_viewer`` short-circuits when the path has not changed, so an
    in-place rewrite of input.png would otherwise leave 2D mode showing the
    texture from before the edit."""
    viewer = ctx.viewer
    if viewer is not None and tab.path is not None and viewer.path == tab.path:
        viewer.clear()
        viewer.load_reference(tab.path)


# --- closing and guarding ---------------------------------------------------


def request_close(ctx: Any, tab: PaintDoc) -> None:
    state = ensure(ctx)

    def go() -> None:
        from .panes import paint_textures

        paint_textures.release_doc(ctx, tab.uid)
        state.close(tab.uid)

    if not tab.dirty:
        go()
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Close without saving?",
            message=f"The changes to {tab.title} will be lost.",
            confirm_label="Close",
            cancel_label="Keep editing",
            on_confirm=go,
        )
    )


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved pixels. -> whether it went ahead now.

    One question for all of them: ``ConfirmQueue`` holds a single pending
    question, so asking per dirty document would silently drop all but the
    first. Only quitting and closing a tab are destructive -- switching modes
    and going Home are not, because Paint is a mode rather than a takeover and
    its tabs are still there when you come back.
    """
    state = ctx.state.paint
    if state is None or not state.any_dirty:
        proceed()
        return True
    count = sum(1 for doc in state.docs if doc.dirty)
    what = "one painting has" if count == 1 else f"{count} paintings have"
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved paintings?",
            message=f"{what[0].upper()}{what[1:]} unsaved changes, which will be lost"
            f" if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False


# --- free transform ---------------------------------------------------------


def begin_transform(ctx: Any, tab: PaintDoc | None = None) -> None:
    """Ctrl+T. Lifts the selection (or the whole layer) and goes modal."""
    state = ensure(ctx)
    tab = tab or state.active
    if tab is None or tab.saving or state.transforming:
        return
    if tab.doc.begin_transform():
        state.transforming = True
        state.clear_drag()


def end_transform(ctx: Any, *, commit: bool) -> None:
    state = ensure(ctx)
    tab = state.active
    state.transforming = False
    state.clear_drag()
    if tab is None:
        return
    if commit:
        tab.doc.commit_floating()
    else:
        tab.doc.cancel_floating()


# --- the OS clipboard -------------------------------------------------------


def paste_from_os(ctx: Any, tab: PaintDoc | None = None) -> bool:
    """Try the system clipboard, then fall back to our own.

    Ours carries a mask and the OS's cannot, which is why the two are separate
    and why this only *seeds* ours: everything downstream of a paste expects to
    know which pixels were selected.

    ``grabclipboard`` is a few milliseconds for a screenshot-sized image and is
    only reached on an explicit Ctrl+V, so it stays on the frame thread rather
    than earning a task key it would spend most of its life idle in.
    """
    tab = tab or active(ctx)
    if tab is None:
        return False
    try:
        import numpy as np
        from PIL import ImageGrab

        grabbed = ImageGrab.grabclipboard()
    except Exception:
        # Unsupported platform, no display, or a clipboard holding something
        # that is not an image. None of those is an error worth a toast.
        log.debug("no image on the system clipboard", exc_info=True)
        return False
    if grabbed is None or isinstance(grabbed, list):
        # A list is a *file* copy rather than an image; opening one would be a
        # silent second way to open files, with none of the checks open_path
        # does.
        return False
    tab.doc.put_clipboard(np.asarray(grabbed.convert("RGBA"), dtype=np.uint8).copy())
    return True


# --- keys -------------------------------------------------------------------

# Aseprite's letters where they exist, because that is the muscle memory a user
# arrives with. Held here rather than in the pane so the mapping is testable.
TOOL_KEYS = {
    "b": "brush",
    "e": "eraser",
    "g": "fill",
    "u": "gradient",
    "r": "blur",
    "n": "smudge",
    "p": "line",
    "k": "rect",
    "j": "ellipse",
    "m": "select",
    "s": "select_ellipse",
    "q": "lasso",
    "w": "wand",
    "v": "move",
    "i": "eyedropper",
}


def handle_key(ctx: Any, event: Any) -> bool:
    """Paint's shortcuts. -> whether the key was consumed.

    Consumed unconditionally while a document is open, exactly as the old
    inline editor did: F, W and S would otherwise frame and wireframe a
    viewport that is not on screen.
    """
    import pygame

    state = ctx.state.paint
    if state is None or not state.docs:
        return False
    tab = state.active
    if tab is None:
        return False
    doc = tab.doc

    if event.key == pygame.K_SPACE:
        # Seen on both edges: space-to-pan is a hold, not a toggle.
        state.space_held = event.type == pygame.KEYDOWN
        return True
    if event.type != pygame.KEYDOWN:
        return True

    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key)

    if state.transforming:
        # Modal: Enter applies, Escape cancels, and nothing else may change the
        # tool out from under a half-finished transform.
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            end_transform(ctx, commit=True)
        elif event.key == pygame.K_ESCAPE or (ctrl and name == "z"):
            # Ctrl+Z during a transform means "undo the transform", which is
            # cancelling it -- not stepping back through the history behind it.
            end_transform(ctx, commit=False)
        return True

    if ctrl:
        return _ctrl_key(ctx, state, tab, doc, name, event, shift=shift)

    if name in TOOL_KEYS and not shift:
        state.tool = TOOL_KEYS[name]
    elif name == "x":
        state.swap_colours()
    elif event.key == pygame.K_LEFTBRACKET:
        if shift:
            state.hardness = max(0.0, state.hardness - 0.05)
        else:
            state.brush_size = paint_state.step_size(state.brush_size, -1)
    elif event.key == pygame.K_RIGHTBRACKET:
        if shift:
            state.hardness = min(1.0, state.hardness + 0.05)
        else:
            state.brush_size = paint_state.step_size(state.brush_size, +1)
    elif event.key == pygame.K_DELETE:
        doc.delete_selection()
    elif event.key == pygame.K_ESCAPE:
        # Never leaves the mode: Esc means "drop what I am doing", and losing a
        # workspace full of tabs to a stray keypress is not that.
        if doc.floating is not None:
            doc.cancel_floating()
        elif doc.mask is not None:
            doc.deselect()
        state.clear_drag()
    return True


def _ctrl_key(
    ctx: Any, state: PaintState, tab: PaintDoc, doc: Any, name: str, event, *, shift: bool
):
    import pygame

    if name == "z":
        doc.redo() if shift else doc.undo()
    elif name == "y":
        doc.redo()
    elif name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
    elif name == "e" and shift:
        export_png(ctx, tab)
    elif name == "n":
        new_document(ctx, 1024, 1024)
    elif name == "o":
        ask_open(ctx)
    elif name == "w":
        request_close(ctx, tab)
    elif name == "a":
        doc.select_all()
    elif name == "d":
        doc.deselect()
    elif name == "c":
        doc.copy()
    elif name == "x":
        doc.cut()
    elif name == "v":
        paste_from_os(ctx, tab)
        if shift:
            doc.paste_as_layer()
        else:
            doc.paste()
            state.tool = "move"
    elif name == "i" and shift:
        doc.invert_selection()
    elif name == "t":
        begin_transform(ctx, tab)
    elif event.key == pygame.K_TAB:
        state.cycle(-1 if shift else 1)
    elif name == "0":
        tab.view.fitted = False
    elif name == "1":
        # Applied by the canvas, which is the only thing that knows how big the
        # pane is; a keypress cannot centre on its own.
        tab.view.pending_zoom = 1.0
    return True


def release_all(ctx: Any) -> None:
    from .panes import paint_textures

    paint_textures.release_all(ctx)
