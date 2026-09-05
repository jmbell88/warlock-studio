"""Packwright's controller: sources, packing, saving, exporting and keys.

The layer that knows about jobs and task threads; the engine under
``packwright/`` knows about neither. The panes draw, this decides.

**Packing is a task, and re-arming it is a flag.** MaxRects over several hundred
sprites plus a full-atlas numpy composite is not frame-thread work, so a repack
goes through ``ctx.submit`` and the result is adopted in :func:`on_task_done`,
where the texture upload belongs. What re-arms it is ``PackTab.pack_dirty``,
pumped from the preview pane's draw and cleared *only when a submit is
accepted* -- the ``findings_dirty`` lesson verbatim: ``TaskRunner.submit``
refuses a key already in flight and nothing re-arms it, so a burst of setting
changes would otherwise pack the state as it stood at the first one and drop
every edit after it.

**Composing reads only frozen data.** A ``Sprite``'s pixels are read-only and a
``Layout`` is frozen, so the pack task can safely run against them while the
frame thread draws -- which is why the snapshot taken at submit time is a list
of sprites and a settings object rather than the document.

Every task key carries the ``packwright-`` prefix, because the app claims
results by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import dialogs, docmodes, journal, packwright_io, packwright_state

# ``ensure`` and ``active`` live in :mod:`.packwright_state` -- they touch
# nothing but ``ctx.state.packwright`` -- and the file layer lives in
# :mod:`.packwright_io`. Both are re-exported here, as **plain imports rather
# than wrappers**, because every pane, every key binding and every test says
# ``packwright_mode.save(ctx)``: a wrapper would be a second object where the
# callers reach for one, and a wiring test parametrizes over ``IMAGE_FILTER``
# by identity besides.
from .packwright_io import (  # noqa: F401
    IMAGE_FILTER,
    PNG_FILTER,
    WPACK_FILTER,
    _decode,
    _load,
    _start,
    ask_open,
    edit_asset_in_packwright,
    export_files,
    export_library,
    open_path,
    save,
    save_as,
    save_to,
)
from .packwright_state import (  # noqa: F401
    PackTab,
    PackwrightState,
    active,
    ensure,
)
from .state import set_mode

log = logging.getLogger(__name__)


# The three recents wrappers every document mode carries, over the one
# list Home's Resume rows are built from (``docmodes.recents_for``).
remember_path, forget_path, recent_paths = docmodes.recents_for("packwright")


def persist(ctx: Any) -> None:
    """Nothing to write any more: the recent list moved to :mod:`.recents`,
    which persists itself on every write. Kept as a no-op because it is called
    from a dozen places after every open and save, and turning each of those
    into "call this only if the mode still has settings" is how one of them
    comes to skip a write that mattered later."""


# --- documents ----------------------------------------------------------------


def adopt(ctx: Any, doc: Any, *, path: Path | None = None, title: str | None = None) -> PackTab:
    state = ensure(ctx)
    tab = PackTab(
        doc=doc,
        title=title or packwright_state.title_for(path),
        path=path,
        saved_head=doc.history.head,
    )
    state.add(tab)
    remember_path(ctx, path)
    persist(ctx)
    return tab


def new_document(ctx: Any) -> PackTab:
    from .packwright.document import PackDoc

    return adopt(ctx, PackDoc(), title="Untitled")


# --- sources ------------------------------------------------------------------


def ask_add_sources(ctx: Any) -> None:
    """The picker and the decode on one task thread."""
    tab = active(ctx)
    if tab is None:
        docmodes.refuse(ctx, "Start or open an atlas first.")
        return
    uid = tab.uid

    from .packwright.sources import file_key

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Add an image", IMAGE_FILTER)
        if path is None:
            return None
        return {"sprites": [(file_key(path), path.stem, _decode(path))], "uid": uid}

    ctx.submit(f"packwright-add:{uid}", run)


def ask_add_tileset(ctx: Any) -> None:
    """The picker and decode for an already-made tile sheet.

    The slicing happens later, in the sources pane's popup, because a sheet
    cannot say its own tile size -- the user provides it there, with a live
    count of what each answer would keep and drop.
    """
    tab = active(ctx)
    if tab is None:
        docmodes.refuse(ctx, "Start or open an atlas first.")
        return
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Add a tile set", IMAGE_FILTER)
        if path is None:
            return None
        return {"tileset": (str(path), path.stem, _decode(path)), "uid": uid}

    ctx.submit(f"packwright-tileset:{uid}", run)


def add_rendered_sheet(ctx: Any, job_id: str, sheet_id: str, *, pixel: bool = False) -> None:
    """A rendered 8-direction sheet onto the open atlas, through the usual door.

    The house pattern is confirm-at-the-door, so this parks the sheet on the
    existing ``tileset_import``/``tileset_import_open``/``tileset_cell`` trio
    rather than importing anything: the popup opens with the cell size the
    sidecar recorded already filled in, shows what that answer keeps and drops,
    and the existing occupancy/import path does the rest. Nothing new is
    imported and nothing new is refused.
    """
    tab = active(ctx)
    if tab is None:
        # ``add_job_source``'s rule: this is offered from *outside* Packwright,
        # so refusing is an offer taken back -- and an atlas has no numbers
        # that cannot be taken back later, so one is simply made.
        tab = new_document(ctx)
        set_mode(ctx.state, "packwright")
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        import numpy as np
        from PIL import Image

        from ..service import sheets as svc_sheets
        from .inker_mode import sheet_grid

        if pixel:
            record = svc_sheets.get_pixel_sheet(ctx.svc, job_id, sheet_id)
            png = svc_sheets.sheet_pixel_png(ctx.svc, job_id, sheet_id)
        else:
            record = svc_sheets.get_sheet(ctx.svc, job_id, sheet_id)
            png = svc_sheets.sheet_png(ctx.svc, job_id, sheet_id)
        cell, _count = sheet_grid(record)
        with Image.open(png) as opened:
            opened.load()
            pixels = np.asarray(opened.convert("RGBA"), dtype=np.uint8).copy()
        name = str(record.get("name") or sheet_id)
        return {"tileset": (str(png), name, pixels), "cell": cell, "uid": uid}

    ctx.submit(f"packwright-tileset:{uid}", run)


def import_tileset(ctx: Any) -> bool:
    """Slice the pending sheet with the popup's cell size and add the tiles.

    The key carries the tile size as well as the path, so re-importing the same
    sheet cut differently adds a different set rather than being skipped as
    duplicates of the first cut.
    """
    from .packwright.sources import dedup_tiles, sprites_from_tileset

    state = ensure(ctx)
    tab = active(ctx)
    if tab is None or state.tileset_import is None:
        return False
    path, stem, pixels = state.tileset_import
    tile = state.tileset_cell
    try:
        sprites = sprites_from_tileset(
            pixels, tile=tile, prefix=f"{path}@{tile[0]}x{tile[1]}", name=stem
        )
    except ValueError as exc:
        docmodes.refuse(ctx, f"That tile set was not imported: {exc}")
        return False
    if state.tileset_dedup:
        sprites, _dropped = dedup_tiles(
            sprites, orientations=state.tileset_dedup_flips
        )
    state.tileset_import = None
    state.tileset_preview_key = None
    state.tileset_import_open = False
    ctx.toast(_added_sentence(*_add_sprites(ctx, tab, sprites), noun="tile"))
    return True


def add_source_paths(ctx: Any, paths: list[Path]) -> None:
    """Files dropped on the window.

    **Keyed on the batch, not on the tab.** pygame raises one ``DROPFILE`` per
    file, so a drag of twenty PNGs reaches this twenty times in one pump; under
    a per-tab key the runner's dedupe refused nineteen of them and the refusal
    was thrown away, so a multi-file drop silently added the first file only
    (the 2026-09-02 review, section 7). ``inker-open``'s shape: the paths are
    in the key, so every distinct drop runs and a *repeated* drop of the same
    files while one is still decoding still dedupes, which is the half of the
    key worth keeping.
    """
    tab = active(ctx)
    if tab is None:
        # A drop is the same offer from outside: minted rather than refused,
        # ``add_job_source``'s rule.
        tab = new_document(ctx)
        set_mode(ctx.state, "packwright")
    from .packwright.sources import file_key

    wanted = [Path(p) for p in paths]
    uid = tab.uid

    def run() -> dict[str, Any]:
        return {
            "sprites": [(file_key(p), p.stem, _decode(p)) for p in wanted],
            "uid": uid,
        }

    token = abs(hash(tuple(str(p) for p in wanted)))
    ctx.submit(f"packwright-add:{uid}:{token}", run)


def add_job_source(ctx: Any, job: Any) -> None:
    """A library asset's reference image as one sprite.

    No ceiling in front of the decode, unlike the ``.wpack`` path: a job's
    ``input.png`` was bounded by ``service.files`` when it was written, and a
    second door on the way back out would be a second answer to a question that
    already has one.
    """
    tab = active(ctx)
    if tab is None:
        # Start the atlas rather than refuse. The Library offers this for any
        # asset with an ``input.png`` and cannot know whether an atlas is open,
        # so the toast was an offer taken back. ``landing.start_packwright``'s
        # move, and unlike a map an atlas has no numbers that cannot be taken
        # back later, so there is nothing to ask about first.
        tab = new_document(ctx)
        # Only when one was minted here: joining an atlas the user already had
        # open is a background addition and does not move them, but a brand new
        # empty atlas they cannot see is the same dead end by another route.
        set_mode(ctx.state, "packwright")
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") or job_id) if isinstance(job, dict) else job_id
    uid = tab.uid

    def run() -> dict[str, Any]:
        from ..service import files as svc_files

        path = svc_files.job_dir_file(ctx.svc, job_id, "input.png")
        return {"sprites": [(f"job:{job_id}", str(name), _decode(Path(path)))], "uid": uid}

    # Keyed on the job for ``add_source_paths``' reason: two assets sent from
    # the library in one gesture are two adds, and a shared key would drop the
    # second without a word.
    ctx.submit(f"packwright-add:{uid}:{job_id}", run)


def add_inker_document(ctx: Any, inker_tab: Any) -> None:
    """Every frame (or layer) of an open Inker document, as sprites.

    Enumerated on the frame thread deliberately: ``frame_flat`` fills and
    evicts the document's own flatten cache and ``layers_for`` copies track
    properties down onto cels, which is exactly what the onion-skin draw is
    doing to the same dicts. That is the ``inker.sheetout`` split, and this is
    the same boundary in a different mode.
    """
    from .packwright.sources import sprites_from_document

    tab = active(ctx)
    if tab is None:
        # ``add_job_source``'s rule: this is offered from *outside* Packwright
        # (Inker's bridge, and Packwright's own sources pane before an atlas
        # exists), so refusing here is an offer taken back. An atlas has no
        # numbers that cannot be taken back later, so one is simply made.
        tab = new_document(ctx)
        if ctx.state.mode != "packwright":
            set_mode(ctx.state, "packwright")
    prefix = Path(inker_tab.title).stem or inker_tab.uid
    try:
        sprites = sprites_from_document(inker_tab.doc, prefix=prefix)
    except ValueError as exc:
        docmodes.refuse(ctx, f"Those frames were not added: {exc}")
        return
    ctx.toast(_added_sentence(*_add_sprites(ctx, tab, sprites)))


def set_pivot(
    ctx: Any, tab: PackTab, uid: int, pivot: tuple[float, float] | None
) -> None:
    """Move one sprite's anchor, and re-arm the pack.

    Through the mode rather than onto the document for ``rename_source``'s
    reason: the pivot rides in the layout and out into the exported sidecar, so
    a pivot that never reaches a pack is a pivot the sidecar does not carry.
    """

    if tab is None or tab.saving:
        return
    before = tab.doc.history.head
    tab.doc.set_pivot(int(uid), pivot)
    if tab.doc.history.head != before:
        tab.pack_dirty = True


def _add_sprites(ctx: Any, tab: PackTab, sprites: list[Any]) -> tuple[int, int]:
    """Add what is not there and refresh what is. -> ``(added, replaced)``.

    A duplicate key is *skipped* rather than refused, unlike ``PackDoc``'s own
    rule: dropping twenty files of which one is already in the atlas should add
    nineteen, not fail. The document's refusal stays the authority on what may
    coexist; this is the caller deciding what to ask for.

    **A key already here whose pixels have changed is a replacement**, not a
    skip. ``wpack``'s own docstring says what a source is -- "what the document
    records is what was packed; re-adding the source is how you pick up a
    change" -- and until 2026-09-03 re-adding an edited PNG was silently
    nothing at all, which made that sentence false and left the only way to
    pick up an edit "delete the sprite first" (the 2026-09-02 review, section
    7). Unchanged pixels are still a skip, so re-dropping a folder is not
    twenty undo steps.
    """
    added = replaced = 0
    for sprite in sprites:
        existing = next(
            (one for one in tab.doc.sources if one.key == sprite.key), None
        )
        if existing is None:
            tab.doc.add_source(sprite)
            added += 1
            continue
        before = tab.doc.history.head
        tab.doc.replace_source(existing.uid, sprite)
        if tab.doc.history.head != before:
            replaced += 1
    if added or replaced:
        tab.pack_dirty = True
    return added, replaced


def _added_sentence(added: int, replaced: int, *, noun: str = "sprite") -> str:
    """What ``_add_sprites`` just did, in one sentence naming both halves.

    Both counts or neither: "Added 19" hides the twentieth file being the
    edited one the user actually dropped this folder for, and "Updated 1" hides
    the nineteen.
    """
    parts = []
    if added:
        parts.append(f"Added {added} {noun}(s)")
    if replaced:
        parts.append(f"updated {replaced}" if parts else f"Updated {replaced} {noun}(s)")
    if not parts:
        return f"Those {noun}s are already in this atlas, unchanged."
    return ", ".join(parts) + "."


def remove_source(ctx: Any, uid: int, tab: PackTab | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    if tab.doc.source(uid) is None:
        # A uid goes stale for ordinary reasons -- an undone add is the one that
        # bit -- and Delete against one is a no-op rather than a refusal: there
        # is nothing to tell the user that they did not already see.
        return
    tab.doc.remove_source(uid)
    tab.pack_dirty = True
    state = ensure(ctx)
    if state.selected == uid:
        state.selected = None


def rename_source(ctx: Any, tab: PackTab | None, uid: int, name: str) -> None:
    """Rename one source. **The pack is re-armed**, which is the whole reason
    this exists: the pane used to call ``tab.doc.rename_source`` directly, so
    the name changed in the list and the *layout* -- which is what the
    TexturePacker sidecar's ``filename`` is written from -- kept the old one
    until something else happened to dirty the pack. An export in between
    carried a name nothing on screen still showed."""
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    try:
        tab.doc.rename_source(uid, name)
    except ValueError as exc:
        docmodes.refuse(ctx, f"That name was not applied: {exc}")
        return
    tab.pack_dirty = True


def set_settings(ctx: Any, tab: PackTab | None = None, **values: Any) -> None:
    """Every settings edit goes through here, so ``pack_dirty`` cannot be
    forgotten at one of six call sites."""
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    try:
        tab.doc.set_settings(**values)
    except (ValueError, TypeError) as exc:
        # ``TypeError`` as well: ``dataclasses.replace`` raises it for a field
        # that does not exist, which is what a stale keyword from a pane would
        # be -- and an uncaught one here is a crash rather than a refusal.
        # Framed rather than forwarded, the house rule: a bare ``str(exc)``
        # toast is library text with no subject in front of it.
        docmodes.refuse(ctx, f"That setting was not applied: {exc}")
        return
    tab.pack_dirty = True


# --- packing ------------------------------------------------------------------


def request_repack(ctx: Any, tab: PackTab | None = None) -> None:
    """Mark the atlas dirty so the next pump repacks it.

    What ``R`` does, given a name so the settings pane's Repack button is the
    same verb rather than a second one that happens to agree. It does not pack
    here: packing is a worker job the centre pane's pump owns, and a pane that
    started one would be doing frame-thread work on a full-atlas composite.
    """
    tab = tab or active(ctx)
    if tab is None:
        return
    tab.pack_dirty = True


def request_pack(ctx: Any, tab: PackTab | None = None) -> None:
    """Ask for a repack. Safe to call every frame -- that is the point."""
    tab = tab or active(ctx)
    if tab is None or not tab.pack_dirty:
        return
    if not tab.doc.sources:
        tab.layout, tab.atlas, tab.pack_dirty, tab.pack_error = None, None, False, ""
        return

    from .packwright import compose as composelib
    from .packwright import layout as laylib

    # The snapshot: frozen sprites and a frozen settings object, so the task
    # reads nothing the frame thread can be writing.
    sprites = tab.doc.sprites()
    settings = tab.doc.settings
    uid = tab.uid

    def run() -> dict[str, Any]:
        from ..service.errors import invalid_from

        try:
            result = laylib.layout(sprites, settings)
        except ValueError as exc:
            # Framed, because only a ``ServiceError``'s text survives the task
            # classifier: the engine's *remedy* sentence -- raise the max size,
            # trim them, or split the pack -- is what ``pack_error`` is for, and
            # a bare ValueError put "see the log for details" there instead.
            raise invalid_from(exc, "That pack did not work") from exc
        return {"layout": result, "atlas": composelib.compose(sprites, result), "uid": uid}

    tab.packing = True
    if ctx.submit(f"packwright-pack:{uid}", run):
        # Cleared *only* on an accepted submit. The runner refuses a key already
        # in flight, and clearing regardless would drop the edit that arrived
        # while the previous pack was running.
        tab.pack_dirty = False
    else:
        tab.packing = False


def pump(ctx: Any) -> None:
    """Called from the preview pane's draw, which is the only thing that runs
    every frame in this mode -- the ``motion.py`` idiom."""
    request_pack(ctx)


# --- task results -------------------------------------------------------------


def on_task_done(ctx: Any, done: Any) -> None:
    state = ensure(ctx)
    key, result = done.key, done.result
    name = key.split(":", 1)[0]

    if name == "packwright-open":
        if isinstance(result, dict):
            adopt(
                ctx,
                result["doc"],
                path=Path(result["path"]) if result.get("path") else None,
                title=result.get("title"),
            )
            set_mode(ctx.state, "packwright")
        return

    # ``split(":")[1]``, not ``split(":", 1)[1]``: an add carries a third
    # segment (the batch token) so that several drops can be in flight at once,
    # and a tab uid never contains a colon.
    tab = state.get(key.split(":")[1]) if ":" in key else None
    if tab is None:
        return

    if name == "packwright-pack":
        if isinstance(result, dict):
            tab.adopt_pack(result["layout"], result["atlas"])
        else:
            tab.packing = False
        return

    if name == "packwright-tileset":
        # The decode landing: the sheet parks on the state until the popup's
        # tile-size answer turns it into sprites, or a cancel drops it.
        if isinstance(result, dict):
            state.tileset_import = result["tileset"]
            state.tileset_import_open = False
            cell = result.get("cell")
            if cell is not None:
                # Only when the door knew the answer. A picked file does not,
                # and overwriting the last typed cell size for it would throw
                # away the number the user is cutting a folder of sheets with.
                state.tileset_cell = (int(cell[0]), int(cell[1]))
        return

    if name == "packwright-add":
        # ``packing`` is deliberately not touched: it belongs to the pack task
        # (set by ``request_pack``, cleared where the pack lands or fails), and
        # an add landing while a pack was in flight used to clear it here --
        # the preview then read "not packing" about a pack still running.
        if isinstance(result, dict):
            from .packwright.sources import sprite_from_image

            sprites = [
                sprite_from_image(pixels, key=key_, name=display)
                for key_, display, pixels in result.get("sprites", [])
            ]
            ctx.toast(_added_sentence(*_add_sprites(ctx, tab, sprites)))
        return

    tab.saving = False
    if not isinstance(result, dict):
        return  # a cancelled dialog

    if result.get("exported_asset"):
        ctx.cache.invalidate()
        ctx.toast("Exported to the library.")
        return
    if result.get("exported"):
        ctx.toast(f"Exported {result.get('files', 2)} file(s) to {result['exported']}")
        if result.get("tsx_skipped"):
            # A second toast rather than folded into the first: "Exported 2
            # file(s)" is a success sentence, and burying "the .tsx was not
            # one of them, because ..." inside it reads as one long message
            # nobody finishes. The PNG and JSON still exported -- only the
            # tileset that would have sliced wrong did not.
            ctx.toast(f"No .tsx written: {result['tsx_skipped']}", "warn")
        return

    tab.mark_saved(result.get("head"))
    # See ``inker_mode``: saved is the moment the crash copy stops
    # describing anything at risk (UX-05).
    journal.drop(ctx, tab)
    if result.get("retitle") and result.get("path"):
        tab.path = Path(result["path"])
        tab.title = packwright_state.title_for(tab.path)
        remember_path(ctx, tab.path)
        persist(ctx)
    ctx.toast("Saved.")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A failed save must not leave the document locked, and a failed *pack*
    must clear ``packing`` and record why -- an empty items list that looks
    like success is the worst outcome of a pack that could not fit."""
    if done.key.startswith(packwright_io.OPEN_PREFIX):
        # Before the tab lookup, because an open that failed has no tab: what it
        # has is a path that does not open, and a Resume list that keeps
        # offering one is worse than a short one. The key carries the path
        # rather than a hash of it precisely so this can be done. An
        # ``edit_asset`` key carries a job id instead, and forgetting one of
        # those is a lookup that matches nothing.
        forget_path(ctx, done.key.split(":", 1)[1])
        return
    state = ctx.state.packwright
    if state is None or ":" not in done.key:
        return
    tab = state.get(done.key.split(":", 1)[1])
    if tab is None:
        return
    tab.saving = False
    if done.key.startswith("packwright-pack"):
        tab.packing = False
        tab.pack_error = done.message or "That pack did not work."


# --- guard and keys -----------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them, the ``clay_mode.guard`` shape. Only quitting
    and closing a tab are destructive: switching modes is not, because
    Packwright is a mode rather than a takeover and its tabs are still there on
    the way back.
    """
    return docmodes.guard(ctx, "packwright", "atlas", "atlases", verb, proceed)


def close_tab(ctx: Any, uid: str) -> None:
    """``docmodes.close_tab``; what is Packwright's is the release."""
    state = ensure(ctx)

    def release(_tab: PackTab) -> None:
        from .panes import packwright_textures

        packwright_textures.release_doc(ctx, uid)

    docmodes.close_tab(ctx, state, uid, release)


def release_all(ctx: Any) -> None:
    from .panes import packwright_textures

    packwright_textures.release_all(ctx)


_MUTATING_CTRL = docmodes.WRITE_CHORDS


# --- history ------------------------------------------------------------------
#
# One call per direction, rather than two lines under the key handler, because
# the bridge panel draws the same Undo/Redo pair Inker's does. Clay, Plotter and
# Packwright each had a full undo stack and no on-screen control at all, so the
# feature existed only for a user who already knew the chord -- and every
# side effect a step has (a repack, and a selection that may name a sprite
# the step removed) belongs to *undoing*, not to the keyboard.


def undo(ctx: Any, tab: Any) -> None:
    """One step back, whichever surface asked for it."""
    tab.doc.undo()
    tab.pack_dirty = True
    _drop_stale_selection(ensure(ctx), tab)



def redo(ctx: Any, tab: Any) -> None:
    """One step forward. :func:`undo`'s twin, and its reasoning."""
    tab.doc.redo()
    tab.pack_dirty = True
    _drop_stale_selection(ensure(ctx), tab)


def step_history(ctx: Any, tab: Any, index: int) -> bool:
    """Jump to a position in the undo stack -- the history popover's door,
    carrying :func:`undo`'s side effects for :func:`undo`'s reason."""
    moved = tab.doc.history.step_to(tab.doc, index)
    tab.pack_dirty = True
    _drop_stale_selection(ensure(ctx), tab)
    return moved



def handle_key(ctx: Any, event: Any) -> bool:
    """Packwright's keyboard. Returns whether the key was consumed; the app
    returns afterwards either way, as it does for every workspace mode."""
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    state = ensure(ctx)
    tab = state.active
    # Off ``event.mod``, never ``pygame.key.get_mods()`` -- ``main._shortcut``'s
    # rule (UX-12): ``mod`` is the state when this key was pressed, and
    # ``get_mods()`` is the state now, after the event batch drained.
    mods = event.mod
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key).lower()

    if ctrl:
        if tab is not None and docmodes.blocked_while_writing(tab, name, _MUTATING_CTRL):
            return True
        return _ctrl_key(ctx, state, tab, name, shift=shift)

    if tab is None:
        return False
    if name == "r":
        request_repack(ctx, tab)
        return True
    if event.key == pygame.K_DELETE and state.selected is not None:
        remove_source(ctx, state.selected, tab)
        return True
    if event.key == pygame.K_ESCAPE:
        state.selected = None
        return True
    return False


def _drop_stale_selection(state: PackwrightState, tab: PackTab) -> None:
    """Clear the selection if the step just undone detached what it names.

    Narrower than ``plotter_mode``'s unconditional clear, and deliberately:
    Plotter's selected *object* is a position in a layer that any step can move,
    while a source's uid survives every step but the one that removes it -- so
    keeping the row selected through an undo of a rename is the right answer,
    and only a detached uid has to go. Leaving it was a crash: Delete then
    addressed a uid the document no longer holds.
    """
    if state.selected is not None and tab.doc.source(state.selected) is None:
        state.selected = None


def _ctrl_key(
    ctx: Any, state: PackwrightState, tab: PackTab | None, name: str, *, shift: bool
) -> bool:
    if name == "n":
        new_document(ctx)
        return True
    if name == "o":
        ask_open(ctx)
        return True
    if tab is None:
        return False
    if name == "w":
        close_tab(ctx, tab.uid)
        return True
    if name == "s":
        save_as(ctx, tab) if shift else save(ctx, tab)
        return True
    if name == "e":
        export_files(ctx, tab) if shift else export_library(ctx, tab)
        return True
    if name == "z":
        # Ctrl+Shift+Z redoes as well, which is what Inker, Clay and Plotter
        # accept and what a user arriving from any of them already has in their
        # hand. Ctrl+Y keeps working: this adds a spelling rather than
        # replacing one.
        redo(ctx, tab) if shift else undo(ctx, tab)
        return True
    if name == "y":
        redo(ctx, tab)
        return True
    if name == "tab":
        state.cycle(-1 if shift else 1)
        return True
    if name == "0":
        tab.view.fitted = False
        return True
    if name == "1":
        tab.view.pending_zoom = 1.0
        return True
    return False


# --- crash recovery (UX-05) ---------------------------------------------------
#
# ``clay_mode``'s four answers, for atlases. See :mod:`studio.journal`.
#
# The *layout* is deliberately not journalled, for the reason it is not saved:
# it is derived, and a re-export of an unchanged document is byte-identical
# with nothing to invalidate. A recovered atlas repacks itself, which is
# seconds and is the only answer that cannot be stale.


def _journal_slots(ctx: Any) -> list[Any]:
    state = getattr(ctx.state, "packwright", None)
    if state is None:
        return []
    return [tab for tab in state.docs if tab.dirty and not tab.busy]


def _journal_encode(tab: Any) -> bytes:
    from .packwright import wpack

    return wpack.wpack_bytes(tab.doc)


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    from .packwright import wpack

    ensure(ctx)
    try:
        doc = wpack.read_wpack(packwright_io._within_ceiling(Path(path)).read_bytes())
    except Exception:
        log.exception("could not reopen the recovered atlas at %s", path)
        journal.adopt_failed(ctx, "atlas")
        return False
    title = f"{meta.get('title') or Path(path).stem} (recovered)"
    tab = adopt(ctx, doc, path=None, title=title)
    docmodes.mark_recovered(tab, path, doc)
    return True


JOURNAL = journal.register(
    journal.Provider(
        kind="packwright",
        ext=".wpack",
        label="atlas",
        slots=_journal_slots,
        uid_of=lambda tab: tab.uid,
        title_of=lambda tab: tab.title,
        head_of=lambda tab: tab.doc.history.head,
        encode=_journal_encode,
        adopt=_journal_adopt,
    )
)
