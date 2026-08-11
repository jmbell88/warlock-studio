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

import numpy as np

from . import dialogs, filetypes, packwright_state, recents
from .packwright_state import PackTab, PackwrightState
from .state import set_mode

log = logging.getLogger(__name__)

WPACK_FILTER = ["Warlock atlas (*.wpack)", "*.wpack"]
# Label *and* patterns from ``filetypes``, which is the whole point: the label
# was hand-written and named three of the five suffixes the pattern list
# accepted, so the dialog told the user two of the formats it would have opened
# were unsupported.
IMAGE_FILTER = [filetypes.describe("Images"), *filetypes.globs()]
PNG_FILTER = ["PNG image (*.png)", "*.png"]


def ensure(ctx: Any) -> PackwrightState:
    state = ctx.state.packwright
    if state is None:
        state = PackwrightState()
        ctx.state.packwright = state
    return state


def remember_path(ctx: Any, path: Any) -> None:
    """Put ``path`` at the front of the merged recent list.

    Through :mod:`.recents` rather than onto a field of this mode's own state:
    the four document modes kept four independent ``recent`` lists, and Home's
    single Resume list cannot be built from them at all -- four bare path lists
    carry no ordering *between* them. There is one list now, and this is how
    packwright writes to it.
    """
    recents.remember(ctx.settings, "packwright", path)


def forget_path(ctx: Any, path: Any) -> None:
    """Drop a path that turned out not to open -- :mod:`.recents`' own rule,
    named here so a caller does not have to know this mode's kind string."""
    recents.forget(ctx.settings, "packwright", path)


def recent_paths(ctx: Any) -> list[str]:
    """This mode's recent files, newest first. What its own panel draws."""
    return recents.paths(ctx.settings, "packwright")


def persist(ctx: Any) -> None:
    """Nothing to write any more: the recent list moved to :mod:`.recents`,
    which persists itself on every write. Kept as a no-op because it is called
    from a dozen places after every open and save, and turning each of those
    into "call this only if the mode still has settings" is how one of them
    comes to skip a write that mattered later."""



def active(ctx: Any) -> PackTab | None:
    state = ctx.state.packwright
    return state.active if state is not None else None


def _decode(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


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


def _load(path: Path) -> dict[str, Any]:
    from .packwright import wpack

    path = Path(path)
    return {
        "doc": wpack.read_wpack(path.read_bytes()),
        "path": str(path),
        "title": packwright_state.title_for(path),
    }


def ask_open(ctx: Any) -> None:
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open an atlas document", WPACK_FILTER)
        return None if path is None else _load(path)

    ctx.submit("packwright-open", run)


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        state.activate(existing.uid)
        return
    ctx.submit(f"packwright-open:{abs(hash(str(path)))}", _load, path)


# --- sources ------------------------------------------------------------------


def ask_add_sources(ctx: Any) -> None:
    """The picker and the decode on one task thread."""
    tab = active(ctx)
    if tab is None:
        ctx.toast("Start or open an atlas first.", "error")
        return
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Add an image", IMAGE_FILTER)
        if path is None:
            return None
        return {"sprites": [(str(path), path.stem, _decode(path))], "uid": uid}

    ctx.submit(f"packwright-add:{uid}", run)


def add_source_paths(ctx: Any, paths: list[Path]) -> None:
    """Files dropped on the window."""
    tab = active(ctx)
    if tab is None:
        ctx.toast("Start or open an atlas first.", "error")
        return
    wanted = [Path(p) for p in paths]
    uid = tab.uid

    def run() -> dict[str, Any]:
        return {
            "sprites": [(str(p), p.stem, _decode(p)) for p in wanted],
            "uid": uid,
        }

    ctx.submit(f"packwright-add:{uid}", run)


def add_job_source(ctx: Any, job: Any) -> None:
    """A library asset's reference image as one sprite."""
    tab = active(ctx)
    if tab is None:
        ctx.toast("Start or open an atlas first.", "error")
        return
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") or job_id) if isinstance(job, dict) else job_id
    uid = tab.uid

    def run() -> dict[str, Any]:
        from ..service import files as svc_files

        path = svc_files.job_dir_file(ctx.svc, job_id, "input.png")
        return {"sprites": [(f"job:{job_id}", str(name), _decode(Path(path)))], "uid": uid}

    ctx.submit(f"packwright-add:{uid}", run)


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
        ctx.toast("Start or open an atlas first.", "error")
        return
    prefix = Path(inker_tab.title).stem or inker_tab.uid
    try:
        sprites = sprites_from_document(inker_tab.doc, prefix=prefix)
    except ValueError as exc:
        ctx.toast(str(exc), "error")
        return
    added = _add_sprites(ctx, tab, sprites)
    ctx.toast(f"Added {added} sprite(s)." if added else "Every frame is already in this atlas.")


def _add_sprites(ctx: Any, tab: PackTab, sprites: list[Any]) -> int:
    """Add what is not already there, and say how many that was.

    A duplicate key is *skipped* rather than refused, unlike ``PackDoc``'s own
    rule: dropping twenty files of which one is already in the atlas should add
    nineteen, not fail. The document's refusal stays the authority on what may
    coexist; this is the caller deciding what to ask for.
    """
    added = 0
    for sprite in sprites:
        if tab.doc.has_key(sprite.key):
            continue
        tab.doc.add_source(sprite)
        added += 1
    if added:
        tab.pack_dirty = True
    return added


def remove_source(ctx: Any, uid: int, tab: PackTab | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    tab.doc.remove_source(uid)
    tab.pack_dirty = True
    state = ensure(ctx)
    if state.selected == uid:
        state.selected = None


def set_settings(ctx: Any, tab: PackTab | None = None, **values: Any) -> None:
    """Every settings edit goes through here, so ``pack_dirty`` cannot be
    forgotten at one of six call sites."""
    tab = tab or active(ctx)
    if tab is None or tab.busy:
        return
    try:
        tab.doc.set_settings(**values)
    except ValueError as exc:
        ctx.toast(str(exc), "error")
        return
    tab.pack_dirty = True


# --- packing ------------------------------------------------------------------


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
        result = laylib.layout(sprites, settings)
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


# --- saving and exporting -----------------------------------------------------


def _start(ctx: Any, tab: PackTab, key: str, run: Any) -> None:
    tab.saving = True
    if not ctx.submit(key, run):
        tab.saving = False


def save_to(ctx: Any, tab: PackTab, path: Path) -> None:
    from .packwright import wpack

    path = Path(path)
    head = tab.doc.history.head
    data = wpack.wpack_bytes(tab.doc)

    def run() -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return {"head": head, "path": str(path), "retitle": True}

    _start(ctx, tab, f"packwright-save:{tab.uid}", run)


def save(ctx: Any, tab: PackTab | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    save_to(ctx, tab, tab.path)


def save_as(ctx: Any, tab: PackTab | None = None) -> None:
    from .packwright import wpack

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    head = tab.doc.history.head
    data = wpack.wpack_bytes(tab.doc)
    stem = Path(tab.title).stem or "atlas"

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save the atlas document", f"{stem}{packwright_state.WPACK_SUFFIX}", WPACK_FILTER
        )
        if path is None:
            return None
        path = path.with_suffix(packwright_state.WPACK_SUFFIX)
        path.write_bytes(data)
        return {"head": head, "path": str(path), "retitle": True}

    _start(ctx, tab, f"packwright-saveas:{tab.uid}", run)


def export_files(ctx: Any, tab: PackTab | None = None) -> None:
    """The atlas PNG, the JSON sidecar, and a ``.tsx`` when the pack is a grid.

    One picker for the PNG; the others take its stem and sit beside it, which is
    the layout every consumer expects and the only one where the sidecar's
    ``meta.image`` resolves.
    """
    from .packwright import compose as composelib
    from .packwright import texturepacker, tsxout

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.layout is None or tab.atlas is None:
        ctx.toast("Nothing packed yet.", "error")
        return
    layout, atlas = tab.layout, tab.atlas
    stem = Path(tab.title).stem or "atlas"

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file("Export the atlas", f"{stem}.png", PNG_FILTER)
        if path is None:
            return None
        path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(composelib.png_bytes(atlas))
        path.with_suffix(".json").write_bytes(
            texturepacker.tp_bytes(layout, image_name=path.name)
        )
        written = 2
        if layout.is_grid:
            path.with_suffix(".tsx").write_bytes(
                tsxout.grid_tsx(layout, atlas, name=stem, image_name=path.name)
            )
            written = 3
        return {"exported": str(path), "files": written}

    _start(ctx, tab, f"packwright-export:{tab.uid}", run)


def export_library(ctx: Any, tab: PackTab | None = None) -> None:
    """Mint an ordinary asset from the atlas, with ``pack.wpack`` beside it."""
    from .packwright import compose as composelib
    from .packwright import wpack

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.atlas is None:
        ctx.toast("Nothing packed yet.", "error")
        return
    png = composelib.png_bytes(tab.atlas)
    source = wpack.wpack_bytes(tab.doc)
    title = tab.title

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from ..service import jobs as svc_jobs

        result = svc_jobs.import_reference(
            ctx.svc, png, name=title, prompt=title, authored="packwright"
        )
        job_id = result["id"]
        svc_files.save_packwright_source(ctx.svc, job_id, source)
        return {"job_id": job_id, "exported_asset": True}

    _start(ctx, tab, f"packwright-library:{tab.uid}", run)


def edit_asset_in_packwright(ctx: Any, job: Any) -> None:
    job_id = job["id"] if isinstance(job, dict) else str(job)
    ensure(ctx)

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from .packwright import wpack

        path = svc_files.packwright_source_path(ctx.svc, job_id)
        return {
            "doc": wpack.read_wpack(Path(path).read_bytes()),
            "path": "",
            "title": "Atlas",
        }

    ctx.submit(f"packwright-open:{job_id}", run)


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

    tab = state.get(key.split(":", 1)[1]) if ":" in key else None
    if tab is None:
        return

    if name == "packwright-pack":
        if isinstance(result, dict):
            tab.adopt_pack(result["layout"], result["atlas"])
        else:
            tab.packing = False
        return

    if name == "packwright-add":
        tab.packing = False
        if isinstance(result, dict):
            from .packwright.sources import sprite_from_image

            sprites = [
                sprite_from_image(pixels, key=key_, name=display)
                for key_, display, pixels in result.get("sprites", [])
            ]
            added = _add_sprites(ctx, tab, sprites)
            ctx.toast(
                f"Added {added} sprite(s)." if added else "Those are already in this atlas."
            )
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
        return

    tab.mark_saved(result.get("head"))
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
    state = ctx.state.packwright
    if state is None or not state.any_dirty:
        proceed()
        return True
    count = sum(1 for doc in state.docs if doc.dirty)
    what = "one atlas has" if count == 1 else f"{count} atlases have"
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved work?",
            message=f"{what[0].upper()}{what[1:]} unsaved changes, which will be lost"
            f" if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False


def close_tab(ctx: Any, uid: str) -> None:
    state = ensure(ctx)
    tab = state.get(uid)
    if tab is None:
        return

    def drop() -> None:
        from .panes import packwright_textures

        packwright_textures.release_doc(ctx, uid)
        state.close(uid)

    if not tab.dirty:
        drop()
        return
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Close without saving?",
            message=f"{tab.title} has unsaved changes.",
            on_confirm=drop,
        )
    )


def release_all(ctx: Any) -> None:
    from .panes import packwright_textures

    packwright_textures.release_all(ctx)


TOOL_KEYS: dict[str, str] = {"r": "repack"}

_MUTATING_CTRL = frozenset({"z", "y"})


def handle_key(ctx: Any, event: Any) -> bool:
    """Packwright's keyboard. Returns whether the key was consumed; the app
    returns afterwards either way, as it does for every workspace mode."""
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    state = ensure(ctx)
    tab = state.active
    mods = pygame.key.get_mods()
    ctrl = bool(mods & pygame.KMOD_CTRL)
    shift = bool(mods & pygame.KMOD_SHIFT)
    name = pygame.key.name(event.key).lower()

    if ctrl:
        if tab is not None and tab.busy and name in _MUTATING_CTRL:
            return True
        return _ctrl_key(ctx, state, tab, name, shift=shift)

    if tab is None:
        return False
    if name == "r":
        tab.pack_dirty = True
        return True
    if event.key == pygame.K_DELETE and state.selected is not None:
        remove_source(ctx, state.selected, tab)
        return True
    if event.key == pygame.K_ESCAPE:
        state.selected = None
        return True
    return False


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
        tab.doc.undo()
        tab.pack_dirty = True
        return True
    if name == "y":
        tab.doc.redo()
        tab.pack_dirty = True
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
