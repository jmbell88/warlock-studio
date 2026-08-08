"""Inker mode's controller: opening, saving, guarding, keys, and the bridge.

Everything here is *about* documents rather than pixels -- the engine under
``inker/`` has no idea a job or a task thread exists, and this is the layer that
knows about both. The panes draw; this decides.

The one rule that shapes the whole file: **no file dialog and no encode ever
runs on the frame thread.** A native picker is modal to the OS and blocks until
dismissed, and a 4096-square ORA is a second of zlib. Both go through
``ctx.submit`` and come back through ``on_task_done``, which is why saving is a
state (``InkerDoc.saving``) rather than a function call that returns.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ..pipelines import sheet as sheetlib
from . import dialogs, inker_state
from .inker import animation
from .inker_state import InkerDoc, InkerState

log = logging.getLogger(__name__)

ORA_FILTER = ["OpenRaster (*.ora)", "*.ora"]
OPEN_FILTER = ["Images and layered files", "*.ora *.png *.jpg *.jpeg *.webp *.bmp"]
PNG_FILTER = ["PNG image (*.png)", "*.png"]

OPENABLE = (".ora", ".png", ".jpg", ".jpeg", ".webp", ".bmp")

NEW_PRESETS = ((512, 512), (1024, 1024), (2048, 2048))


def ensure(ctx: Any) -> InkerState:
    """The mode's state, built on first use.

    Lazy because a session that never opens Paint should not pay for its
    swatches, and because ``AppState`` deliberately knows nothing about it.
    """
    state = ctx.state.inker
    if state is None:
        state = InkerState()
        stored = ctx.settings.get("inker") or {}
        state.recent = [p for p in (stored.get("recent") or []) if isinstance(p, str)]
        swatches = stored.get("swatches")
        if isinstance(swatches, list) and swatches:
            state.swatches = [
                tuple(int(c) for c in s)  # type: ignore[misc]
                for s in swatches
                if isinstance(s, list | tuple) and len(s) == 4
            ] or list(inker_state.DEFAULT_SWATCHES)
        ctx.state.inker = state
    return state


def persist(ctx: Any) -> None:
    state = ctx.state.inker
    if state is not None:
        ctx.settings.set(
            "inker",
            {"recent": state.recent, "swatches": [list(s) for s in state.swatches]},
        )


def active(ctx: Any) -> InkerDoc | None:
    state = ctx.state.inker
    return state.active if state is not None else None


# --- opening ----------------------------------------------------------------


def new_document(ctx: Any, width: int, height: int) -> InkerDoc:
    from . import inker

    state = ensure(ctx)
    doc = inker.Document.blank(int(width), int(height))
    return _adopt(ctx, state, doc, path=None, title="Untitled", file_format="ora")


def _adopt(
    ctx: Any,
    state: InkerState,
    doc: Any,
    *,
    path: Path | None,
    title: str | None = None,
    file_format: str = "png",
    job_id: str = "",
    link_kind: str = "",
    has_original: bool = False,
    saved_head: int | None = None,
) -> InkerDoc:
    tab = InkerDoc(
        doc=doc,
        title=title or inker_state.title_for(path),
        path=path,
        file_format=file_format,
        # Normally the document as opened *is* what is on disk. The one caller
        # that passes this is the matte hand-off, which opens a document with
        # an edit already applied: recording the head after that edit would
        # call the cutout saved, and closing the tab would discard it without
        # asking.
        saved_head=doc.history.head if saved_head is None else saved_head,
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

    ctx.submit("inker-open", run)


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
        ctx.toast("Inker opens images and .ora files.", "error")
        return
    ctx.submit(f"inker-open:{abs(hash(str(path)))}", _load, path)


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only."""
    from . import inker

    path = Path(path)
    doc = inker.Document.load(path)
    return {"doc": doc, "path": path, "format": doc.file_format}


# --- the job bridge ---------------------------------------------------------


def can_edit_job(ctx: Any, job: Any) -> bool:
    """Whether the "Open in Inker" button belongs on this job's toolbar.

    From the cached row alone -- no filesystem calls, because the toolbar asks
    this every frame.
    """
    return bool(
        job
        and job.get("stage") == "reference"
        and job.get("status") == "done"
        and "input.png" in (job.get("files") or [])
    )


def open_job_reference(ctx: Any, job: Any, *, matte: bool = False) -> None:
    """Open a reference's image as a linked document.

    Prefers the layered working file when there is a fresh one, so layers
    survive between sessions; falls back to the flat input.png, which is also
    what happens after a revert or a regenerate rewrites the reference behind
    the working file's back.

    ``matte`` is the promote preview's "Fix matte": the document opens with the
    host's cutout already folded into its alpha as one undoable step, so the
    eraser and the brush edit the matte directly. A tab that is *already* open
    for this job is focused rather than re-opened even then -- re-cutting a
    document the user has been editing would apply a matte measured off
    ``input.png`` to pixels that have moved on from it, and the alpha is right
    there to paint.
    """
    state = ensure(ctx)
    job_id = job["id"]
    existing = state.find_job(job_id)
    if existing is not None:
        state.activate(existing.uid)
        ctx.state.mode = "inker"
        return
    ctx.state.mode = "inker"
    ctx.submit(f"inker-open:{job_id}", _load_job, ctx.svc, job_id, matte=matte)


def _load_job(svc: Any, job_id: str, *, matte: bool = False) -> dict[str, Any]:
    """Blocking; task thread only."""
    from ..service import files as svc_files
    from . import inker

    flat = svc.job_dir(job_id) / "input.png"
    working = svc_files.inker_working_path(svc, job_id)
    status = svc_files.inker_working_status(svc, job_id)
    doc = inker.Document.load(working if status["fresh"] else flat)
    # The document is *about* input.png whichever file it was decoded from:
    # the title, the dedupe and the save all key on the reference.
    doc.path = flat
    edit = svc_files.reference_edit_status(svc, job_id)
    out = {
        "doc": doc,
        "path": flat,
        "format": "ora",
        "job_id": job_id,
        "link_kind": "reference-edit",
        "has_original": bool(edit.get("has_original")),
        "title": f"{job_id[:8]} reference",
    }
    if matte:
        # Captured before the cut, and handed back as the tab's saved head: the
        # cutout is an unsaved edit, because nothing has written it to disk.
        out["saved_head"] = doc.history.head
        _cut_matte(svc, job_id, doc)
    return out


def _cut_matte(svc: Any, job_id: str, doc: Any) -> bool:
    """Fold the host's cutout into ``doc``'s alpha. Blocking; task thread only.

    Log-and-swallow, like every other improvement the user did not ask for:
    the thing they asked for is the reference open in the editor, and BiRefNet
    failing (or a working file whose canvas has been resized away from the
    reference's) must not cost them that.
    """
    from ..service import matte as svc_matte

    try:
        alpha, _source = svc_matte.alpha_plane(svc, job_id)
        return bool(doc.apply_matte(alpha))
    except Exception:
        log.exception("could not apply the matte to %s", job_id)
        return False


# --- saving -----------------------------------------------------------------


def save(ctx: Any, tab: InkerDoc | None = None) -> None:
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
    _submit_write(ctx, tab, f"inker-save:{tab.uid}", tab.path, tab.file_format)


def save_as(ctx: Any, tab: InkerDoc | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    doc.commit_floating()  # before the head is read; see _save_linked
    rev = doc.history.head
    suggested = tab.path.stem if tab.path else "untitled"

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Save layered document", f"{suggested}.ora", ORA_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".ora":
            dest = dest.with_suffix(".ora")
        _write(doc, dest, "ora")
        return {"path": dest, "rev": rev, "format": "ora", "retitle": True}

    _start(ctx, tab, f"inker-saveas:{tab.uid}", run)


def export_png(ctx: Any, tab: InkerDoc | None = None) -> None:
    """A flattened PNG. Not a save: it does not change what the tab points at,
    so the document stays dirty against its own file."""
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    doc = tab.doc
    # Not a save, but the same rule about what is on the canvas: the composite
    # a floating buffer draws into is the pane's, not the document's, so an
    # export would otherwise be missing pixels the user is looking at.
    doc.commit_floating()
    suggested = tab.path.stem if tab.path else "untitled"

    def run() -> dict[str, Any] | None:
        dest = dialogs.save_file("Export flattened PNG", f"{suggested}.png", PNG_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        dest.write_bytes(doc.png_bytes())
        return {"exported": dest}

    _start(ctx, tab, f"inker-export:{tab.uid}", run)


def export_sheet(ctx: Any, tab: InkerDoc | None = None) -> None:
    """An animated document as a packed PNG plus its JSON sidecar.

    Mirrors ``export_png`` exactly -- gated, floating buffer committed first, one
    task under the same key so the two can never run at once, and
    ``{"exported": path}`` back so the existing completion branch toasts it
    unchanged. What differs is only what gets written.

    With one addition the other exports do not need: the frames are read off the
    document **here**, on the frame thread, and only the encode goes to the
    task. ``_write`` gets away with encoding the live document because the
    encoders only read; flattening a clip does not, since it fills and evicts
    the document's frame cache and copies track properties down onto cels --
    the same structures the onion-skin draw is walking sixty times a second.
    The cost is a flatten per frame at click time, most of which the playback
    cache has already paid for.
    """
    tab = tab or active(ctx)
    if tab is None or tab.busy or tab.doc.anim is None:
        return
    doc = tab.doc
    doc.commit_floating()
    suggested = tab.path.stem if tab.path else "untitled"
    from .inker import sheetout

    frames, durations, tags = sheetout.snapshot(doc)

    def run() -> dict[str, Any] | None:
        import json

        dest = dialogs.save_file("Export sprite sheet", f"{suggested}.png", PNG_FILTER)
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        image, plan, extra = sheetout.compose(frames, durations, tags, name=suggested)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest, "PNG")
        finally:
            image.close()
        meta = sheetlib.sidecar(
            plan,
            sheet_id=dest.stem,
            source_job=tab.job_id,
            image=dest.name,
            created=time.time(),
            name=suggested,
            trims=extra["trims"],
            animation=extra["animation"],
        )
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"exported": dest}

    _start(ctx, tab, f"inker-export:{tab.uid}", run)


def _submit_write(ctx: Any, tab: InkerDoc, key: str, path: Path, file_format: str) -> None:
    doc = tab.doc
    # A floating buffer lives in no layer, and the encoders walk the layer
    # stack -- so without this the file omits the pasted pixels while the
    # canvas still shows them, and ``mark_saved`` then calls the document
    # clean. Closing the tab discarded the paste with no prompt.
    doc.commit_floating()
    rev = doc.history.head

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
    from . import inker

    if file_format == "ora":
        inker.write_ora(doc, path)
    else:
        path.write_bytes(doc.png_bytes())


def _start(ctx: Any, tab: InkerDoc, key: str, run: Any) -> None:
    tab.saving = True
    if not ctx.submit(key, run):
        tab.saving = False


def _save_linked(ctx: Any, tab: InkerDoc) -> None:
    """Write both halves of a reference edit: the flat PNG, then the layers.

    The flat write goes through the untouched ``save_edited_image``, so the
    original backup, the ``hand_edited`` param, the reference re-measure and
    the staged replace all still happen exactly as they did for the old inline
    editor.

    The order is the whole correctness argument, and it is the opposite of what
    it looks like. Freshness is ``paint.ora`` being *newer* than ``input.png``,
    so writing the layers first guarantees the reference is newer than them the
    moment the save completes -- every save would mark its own layers stale and
    the next open would flatten them away. Writing the flat half first leaves
    the sidecar newer on success, and on a crash between the two leaves the
    layers older than the reference, which is exactly the stale verdict a
    half-finished save deserves.
    """
    from ..service import files as svc_files

    doc, job_id = tab.doc, tab.job_id
    # Committed *before* the head is read: the commit pushes a step of its own,
    # so recording the head first saves a document against a head one behind
    # it -- and dirty, being a comparison against that head, then stays true
    # forever however many times the user saves.
    doc.commit_floating()
    rev = doc.history.head

    def run() -> dict[str, Any]:
        from . import inker

        svc_files.save_edited_image(ctx.svc, job_id, doc.png_bytes())
        svc_files.save_inker_working(ctx.svc, job_id, inker.ora_bytes(doc))
        return {"rev": rev, "job_id": job_id, "linked": True}

    _start(ctx, tab, f"inker-save:{tab.uid}", run)


# --- the other direction: Inker -> the pipeline ------------------------------


def save_as_reference(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Mint a new reference job from what is on the canvas, and link to it."""
    from ..service import jobs as svc_jobs

    tab = tab or active(ctx)
    if tab is None or tab.saving or tab.linked:
        return
    doc, title = tab.doc, tab.title
    doc.commit_floating()  # before the head is read; see _save_linked
    rev = doc.history.head

    def run() -> dict[str, Any]:
        result = svc_jobs.import_reference(ctx.svc, doc.png_bytes(), name=title)
        job_id = result["id"]
        from . import inker

        # Linked immediately, so the next Ctrl+S saves in place rather than
        # minting a second job from the same pixels.
        svc_files_save(ctx, job_id, inker.ora_bytes(doc))
        return {"rev": rev, "job_id": job_id, "link": True}

    _start(ctx, tab, f"inker-save:{tab.uid}", run)


def svc_files_save(ctx: Any, job_id: str, data: bytes) -> None:
    from ..service import files as svc_files

    svc_files.save_inker_working(ctx.svc, job_id, data)


def send_to_3d(ctx: Any, tab: InkerDoc | None = None) -> None:
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

    # Gated like every other encode: ``png_bytes`` walks the layer stack on a
    # task thread, and an undo, a crop or a rotate landing mid-walk restructures
    # it underneath. Keyed by tab so ``on_task_done``/``on_task_failed`` can
    # find the document to unlock, the same shape the saves use.
    _start(ctx, tab, f"inker-send:{tab.uid}", run)
    if tab.saving:
        ctx.toast("Queued a mesh from the drawn image.")


def _promote(ctx: Any, job_id: str) -> None:
    from ..service import errors as svc_errors
    from ..service import jobs as svc_jobs

    def run(force: bool = False) -> dict[str, Any]:
        return svc_jobs.promote_to_model(ctx.svc, job_id, force=force)

    def go(force: bool) -> None:
        if ctx.submit(f"inker-promote:{job_id}", run, force):
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
                cancel_label="Keep drawing",
                on_confirm=lambda: go(True),
            )
        )
        return
    go(False)


def revert(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Put the generated image back, and drop the layers that described the
    edit -- they are about pixels that will no longer exist."""
    from ..service import files as svc_files

    tab = tab or active(ctx)
    if tab is None or not tab.linked or not tab.has_original or tab.saving:
        return
    job_id = tab.job_id

    def run() -> dict[str, Any]:
        svc_files.revert_reference(ctx.svc, job_id)
        svc_files.discard_inker_working(ctx.svc, job_id)
        return {"reverted": True, "job_id": job_id}

    def go() -> None:
        _start(ctx, tab, f"inker-revert:{tab.uid}", run)

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Revert to the original?",
            # Names the *layers* rather than "every edit" (S139). The two are
            # not the same claim: what ``run`` deletes is ``paint.ora``, so a
            # revert loses the layered document itself and not merely the
            # flattened pixels -- reopening this asset afterwards gives back one
            # layer. Stated before the button, the retarget panel's rule.
            message=(
                "The generated image comes back. The layered document is deleted "
                "with it, so reopening this asset gives back a single flat layer."
            ),
            confirm_label="Revert",
            cancel_label="Keep editing",
            on_confirm=go,
        )
    )


# --- task results -----------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from App._on_task_done for every ``inker-`` key."""
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name in ("inker-open",):
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
                saved_head=result.get("saved_head"),
            )
            ctx.state.mode = "inker"
        return

    if name == "inker-palette":
        # A list of colours, or None for a cancelled picker. Appended rather
        # than replacing: an import is "add these to what I have", and a user
        # who wanted the old ones gone can right-click them away -- where an
        # import that silently wiped a session's palette has no way back.
        if result:
            for colour in result:
                state.add_swatch(colour)
            persist(ctx)
            ctx.toast(f"Added {len(result)} colour(s).", "success")
        return

    if name in ("inker-send", "inker-promote"):
        ctx.cache.invalidate()
        # ``inker-send`` locks its tab while the flatten runs off-thread;
        # ``inker-promote`` has no tab of its own to unlock.
        sent = state.get(key.split(":", 1)[1]) if name == "inker-send" and ":" in key else None
        if sent is not None:
            sent.saving = False
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
        tab.title = inker_state.title_for(tab.path)
        tab.file_format = result.get("format", "ora")
        state.remember(tab.path)
        persist(ctx)
    linked_just_now = bool(result.get("link"))
    if linked_just_now:
        tab.job_id = result["job_id"]
        tab.link_kind = "reference-edit"
    if tab.linked:
        # "Save as reference" mints a brand-new reference from the drawn
        # pixels, so there is no input.orig.png behind it and Revert has
        # nothing to revert to. The unconditional True here overwrote the
        # False set a line earlier and offered a button that could only fail.
        # Every *other* linked save goes through save_edited_image, which
        # writes the backup.
        tab.has_original = not linked_just_now
        ctx.cache.invalidate()
        _nudge_viewer(ctx, tab)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked.

    ``saving`` disables every editing control, so without this a single failed
    write makes the tab permanently read-only with no way back short of
    closing it.
    """
    state = ctx.state.inker
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is not None:
        tab.saving = False


def _reload_linked(ctx: Any, tab: InkerDoc) -> None:
    """Re-decode a linked document after a revert replaced its file."""
    from . import inker

    if tab.path is None:
        return
    try:
        tab.doc = inker.Document.load(tab.path)
    except Exception as exc:
        ctx.toast(f"Reverted, but the image could not be reopened ({exc}).", "error")
        return
    tab.saved_head = tab.doc.history.head
    tab.has_original = False
    tab.view.fitted = False
    ctx.cache.invalidate()
    ctx.toast("Back to the original image.")
    _nudge_viewer(ctx, tab)


def _nudge_viewer(ctx: Any, tab: InkerDoc) -> None:
    """``_sync_viewer`` short-circuits when the path has not changed, so an
    in-place rewrite of input.png would otherwise leave 2D mode showing the
    texture from before the edit."""
    viewer = ctx.viewer
    if viewer is not None and tab.path is not None and viewer.path == tab.path:
        viewer.clear()
        viewer.load_reference(tab.path)


# --- closing and guarding ---------------------------------------------------


def request_close(ctx: Any, tab: InkerDoc) -> None:
    state = ensure(ctx)

    def go() -> None:
        from .panes import inker_textures

        inker_textures.release_doc(ctx, tab.uid)
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
    state = ctx.state.inker
    if state is None or not state.any_dirty:
        proceed()
        return True
    count = sum(1 for doc in state.docs if doc.dirty)
    what = "one drawing has" if count == 1 else f"{count} drawings have"
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved work?",
            message=f"{what[0].upper()}{what[1:]} unsaved changes, which will be lost"
            f" if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False


# --- free transform ---------------------------------------------------------


def begin_transform(ctx: Any, tab: InkerDoc | None = None) -> None:
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


def paste_from_os(ctx: Any, tab: InkerDoc | None = None) -> bool:
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


# --- playback ----------------------------------------------------------------

#: A tick longer than this is treated as a stall rather than as elapsed time.
#: Without the clamp, a two-second hitch (a dialog, a texture upload storm)
#: fast-forwards the clip through twenty frames at once, which reads as a
#: glitch rather than as the catch-up it technically is.
MAX_TICK_MS = 250.0


def toggle_play(ctx: Any, tab: InkerDoc | None = None) -> None:
    """Start or stop playback. Refused while a save is encoding."""
    tab = tab or active(ctx)
    if tab is None or tab.doc.anim is None or tab.saving:
        return
    if tab.playing:
        stop_play(tab)
        return
    # The float is committed *before* playback rather than when it ends: while
    # playing, the canvas draws cached frame flattens, and a floating buffer is
    # in no layer and therefore in no flatten -- it would simply vanish for the
    # duration and reappear at the end.
    tab.doc.commit_floating()
    tab.playing = True
    tab.play_index = tab.doc.anim.current
    tab.play_accum_ms = 0.0


def stop_play(tab: InkerDoc) -> None:
    """Stop, and leave the playhead where the eye last saw it."""
    if not tab.playing:
        return
    tab.playing = False
    tab.doc.set_current_frame(tab.play_index)
    tab.play_accum_ms = 0.0


def tick_playback(tab: InkerDoc, dt_ms: float) -> None:
    """One frame's worth of time.

    Deliberately does *not* call ``set_current_frame``: that re-materialises the
    stack and recomposites the whole canvas, sixty times a second, to show a
    picture the frame cache already has. The playhead on the document stays put
    and the canvas draws ``play_index``'s cached flatten instead; stopping is
    the one moment the document catches up.
    """
    anim = tab.doc.anim
    if not tab.playing or anim is None:
        return
    durations = [frame.duration_ms for frame in anim.frames]
    index, accum, playing = animation.advance(
        durations,
        tab.play_index,
        tab.play_accum_ms,
        min(float(dt_ms), MAX_TICK_MS),
        anim.loop_range(tab.play_index),
    )
    tab.play_index, tab.play_accum_ms = index, accum
    if not playing:
        stop_play(tab)


def step_frame(ctx: Any, delta: int, tab: InkerDoc | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.doc.anim is None or tab.busy:
        return
    anim = tab.doc.anim
    tab.doc.set_current_frame((anim.current + delta) % len(anim.frames))


def animate(ctx: Any, tab: InkerDoc | None = None) -> None:
    """The entry point: turn a still document into a two-frame animation."""
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    tab.doc.add_frame()


def handle_key(ctx: Any, event: Any) -> bool:
    """Inker's shortcuts. -> whether the key was consumed.

    Consumed unconditionally while a document is open, exactly as the old
    inline editor did: F, W and S would otherwise frame and wireframe a
    viewport that is not on screen.
    """
    import pygame

    state = ctx.state.inker
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
            state.brush_size = inker_state.step_size(state.brush_size, -1)
    elif event.key == pygame.K_RIGHTBRACKET:
        if shift:
            state.hardness = min(1.0, state.hardness + 0.05)
        else:
            state.brush_size = inker_state.step_size(state.brush_size, +1)
    elif doc.anim is not None and event.key in (pygame.K_COMMA, pygame.K_PERIOD):
        # Matched on the key constant, not on ``pygame.key.name``: the spelling
        # of a punctuation key comes from SDL and has changed between versions.
        step_frame(ctx, -1 if event.key == pygame.K_COMMA else 1, tab)
    elif doc.anim is not None and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        # Reachable only past the transform branch above, which consumes Enter
        # first and returns -- applying a half-finished transform must not be
        # ambiguous with starting playback.
        toggle_play(ctx, tab)
    elif event.key == pygame.K_DELETE:
        if not tab.busy:
            doc.delete_selection()
    elif event.key == pygame.K_ESCAPE:
        # Never leaves the mode: Esc means "drop what I am doing", and losing a
        # workspace full of tabs to a stray keypress is not that.
        if tab.playing:
            stop_play(tab)
        elif not tab.saving:
            if doc.floating is not None:
                doc.cancel_floating()
            elif doc.mask is not None:
                doc.deselect()
        # Always: abandoning a half-finished drag is safe mid-save, because it
        # touches the pane's own state and never the document.
        state.clear_drag()
    return True


# Ctrl-shortcuts that change the document. A save encodes the *live* document
# on a task thread; that is safe only for a stroke landing mid-write, because
# pixels are written in place. Everything here restructures the layer stack or
# moves the history head the save captured, so it waits for the save the same
# way a brush stroke on the canvas already does.
_MUTATING_CTRL = frozenset({"z", "y", "a", "d", "x", "v", "i", "t"})


def _ctrl_key(
    ctx: Any, state: InkerState, tab: InkerDoc, doc: Any, name: str, event, *, shift: bool
):
    import pygame

    # ``busy``, not ``saving``: playback is the second reason the document may
    # not be restructured, and it is the same list of keys for the same reason.
    if tab.busy and name in _MUTATING_CTRL:
        return True

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
    from .panes import inker_textures

    inker_textures.release_all(ctx)


# --- palette files ----------------------------------------------------------
#
# The picker and the file are both blocking, so both go to a task thread -- the
# rule every dialog and every encode in this module follows. The *bytes* for an
# export are built on the frame thread, for ``save_as``'s reason: they read live
# state, and doing that after an unbounded modal would write whatever the user
# changed while it was open.

GPL_FILTER = ["GIMP palette (*.gpl)", "*.gpl"]


def import_palette(ctx: Any) -> None:
    from .inker import gpl

    ensure(ctx)

    def run() -> list[tuple[int, int, int, int]] | None:
        path = dialogs.open_file("Import a palette", GPL_FILTER)
        if path is None:
            return None
        return gpl.parse(path.read_text(encoding="utf-8", errors="replace"))

    ctx.submit("inker-palette", run)


def export_palette(ctx: Any) -> None:
    from .inker import gpl

    state = ensure(ctx)
    text = gpl.dumps(list(state.swatches))

    def run() -> None:
        path = dialogs.save_file("Export the palette", "palette.gpl", GPL_FILTER)
        if path is not None:
            path.with_suffix(".gpl").write_text(text, encoding="utf-8")

    ctx.submit("inker-palette-export", run)
