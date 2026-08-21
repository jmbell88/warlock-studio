"""Getting a tileset onto the open map, from wherever it came from.

Four doors, one destination. A ``.tsx`` or an image from a picker, an image from
a path (a drop), a document coming back from Inker, and a library asset's own
reference image -- and every one of them ends as
``{"tileset": ..., "source": ..., "uid": ...}`` on the same
``plotter-tileset:`` task key, because ``plotter_mode.on_task_done`` already
routes and adopts one and a second key would be a second copy of that branch.
The duplicate-key refusal that comes with sharing the key is the right rule too:
a drop and an "Add from a file..." both mean "a tileset is arriving on this
tab".

Nothing here *generates* a tileset any more. The procedural ground set and the
AI-painted one were both retired on 2026-08-18 when tile sheets moved to Create:
a sheet is made there, lands in the library, and reaches a map through
:func:`use_as_tileset` like any other asset -- which is the fourth door above
and was always the one that scaled past a single map.

Split out of ``plotter_mode`` on the seam the file layer left behind: what is
here is *about a tileset*, what is in :mod:`.plotter_io` is about a document's
bytes, and what is left in the mode is about tabs, tasks and keys. The two
filesystem questions -- may this path be followed, is this file small enough --
stay in ``plotter_io`` and are imported, because a ``.tsx`` picked here names an
image the same way a ``.tmx`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import dialogs, filetypes
from .plotter_io import _decode, _resolve_source, _within_ceiling
from .plotter_state import active, ensure

# A ``.tsx`` carries its own slicing; anything else is an image sliced at the
# map's tile size. Both halves of every entry are derived, because the label
# used to read "(*.tsx *.png)" over a pattern list that also accepted .jpg,
# .jpeg, .webp and .bmp -- a dialog disclaiming four formats it would have
# opened.
#
# **Pairs**, and the pairing is the whole of it. This was one label followed by
# a splatted ``globs()`` -- seven entries -- and portable-file-dialogs reads a
# filter list two at a time, so the row the picker opens on advertised six
# formats and filtered on ``*.tsx`` alone. A folder of PNGs came up empty. The
# combined row goes first so that is the one it opens on; the narrower rows
# follow, the shape ``inker_mode.PALETTE_FILTER`` already uses.
_TILESET_SUFFIXES = (".tsx", *filetypes.IMAGE_SUFFIXES)
TILESET_FILTER = [
    filetypes.describe("Tilesets and images", _TILESET_SUFFIXES),
    filetypes.pattern(_TILESET_SUFFIXES),
    "Tiled tileset (*.tsx)",
    "*.tsx",
    filetypes.describe("Images"),
    filetypes.pattern(),
]


def add_tileset_path(
    ctx: Any, path: Path, *, tile_w: int | None = None, tile_h: int | None = None
) -> None:
    """Add a ``.tsx`` or a grid-sliced image to the open map.

    An image is sliced at the *map's* tile size by default, which is right far
    more often than not and is the only default that needs no dialog. A ``.tsx``
    carries its own slicing and ignores both arguments.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    path = Path(path)
    width = int(tile_w or tab.doc.tile_w)
    height = int(tile_h or tab.doc.tile_h)

    def run() -> dict[str, Any]:
        from ..service.errors import invalid_from
        from .plotter import tsx as tsxlib

        try:
            if path.suffix.lower() == ".tsx":
                data = _within_ceiling(path).read_bytes()
                image = tsxlib.tsx_source(data)
                tileset = tsxlib.read_tsx(data, _decode(_resolve_source(path.parent, image)))
            else:
                return _sheet_or_tileset(
                    path.stem, str(path), _decode(path), tab.uid, width, height
                )
        except ValueError as exc:
            raise invalid_from(exc, "This tileset could not be added", field="file") from exc
        return {"tileset": tileset, "source": str(path), "uid": tab.uid}

    ctx.submit(f"plotter-tileset:{tab.uid}", run)


@dataclass(frozen=True)
class SheetMismatch:
    """A sheet whose own recorded cell size is not the map's.

    The second thing that can be wrong at an import door, and the detector
    cannot see it: a sheet generated in Create is drawn on imposed rectangles
    with no separator lines, so it detects nothing and is sliced blind at the
    map's tile size -- silently mis-cutting a 32 px sheet onto a 16 px map. What
    catches it is not a measurement but a *record*: the job's ``sheet.json``
    says what the sheet was generated at.

    Parked as the third variant of the confirm popup rather than fixed silently,
    for the same reason the first one is: the user may genuinely want the blind
    slice, and only they know.
    """

    tile_w: int
    tile_h: int


@dataclass(frozen=True)
class SheetTerrain:
    """A sheet whose cells look like a complete 47-case blob terrain set.

    ``roles`` is the inference; ``tile`` is the cell size the grid was read on,
    which the import needs and the popup says out loud. Parked rather than
    applied for :class:`SheetMismatch`'s reason and one more: a silhouette
    coincidence -- a sprite sheet whose cells happen to cover every blob case --
    would otherwise be reordered into a terrain set nobody asked for.
    """

    tile: tuple[int, int]
    roles: Any


def _terrain_from(pixels: Any, tile_w: int, tile_h: int) -> Any:
    """A :class:`SheetTerrain` for this sheet, or ``None``. Task thread.

    Runs only when the grid holds at least the 47 cells a complete set needs;
    below that there is nothing to find and the analysis is pure cost.
    """
    from .tilegrid import blob
    from .tilegrid import roles as rolelib

    array = pixels
    rows = array.shape[0] // int(tile_h)
    cols = array.shape[1] // int(tile_w)
    if rows * cols < blob.TILE_COUNT:
        return None
    found = rolelib.infer_roles(array, tile_w, tile_h)
    return None if found is None else SheetTerrain((int(tile_w), int(tile_h)), found)


def _sheet_or_tileset(
    name: str,
    source: str,
    pixels: Any,
    uid: str,
    tile_w: int,
    tile_h: int,
    *,
    recorded: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Task-thread: is this image a ruled sheet, or is it already a tileset?

    The three image doors share this so that "a file the user picked" gets the
    same answer wherever it was picked from. A ruled sheet is *parked* rather
    than sliced -- :mod:`.tilegrid.slicing`'s own rule is that detection is a
    suggestion, so the popup asks and this returns nothing that has been
    decided. No grid found is today's behaviour byte-for-byte: the blind slice
    at the map's tile size, which is also the right fallback for a sheet that
    was generated on imposed rectangles and has no rules to find.

    ``recorded`` is the cell size the sheet's own sidecar declares, which only
    the library door has. Equal to the map's, it changes nothing at all --
    today's path exactly. Different, it parks a :class:`SheetMismatch` instead,
    because a blind slice at the map's size would cut every tile in half with
    nothing anywhere to say so.
    """
    from .tilegrid import slicing
    from .tilegrid.tileset import Tileset

    grid = slicing.detect_grid(pixels)
    if grid is not None:
        # Rules found: the cells are where the detector says, so a terrain guess
        # has to be made against the *recomposed* atlas rather than the ragged
        # original -- and that is the popup's Import, not this. So the separator
        # arm wins and the terrain question is asked at the plain-grid door only.
        return {"sheet": (uid, name, source, pixels, grid), "uid": uid}
    if recorded is not None and (int(recorded[0]), int(recorded[1])) != (tile_w, tile_h):
        mismatch = SheetMismatch(int(recorded[0]), int(recorded[1]))
        return {"sheet": (uid, name, source, pixels, mismatch), "uid": uid}
    terrain = _terrain_from(pixels, tile_w, tile_h)
    if terrain is not None:
        return {"sheet": (uid, name, source, pixels, terrain), "uid": uid}
    tileset = Tileset(name=name, pixels=pixels, tile_w=tile_w, tile_h=tile_h)
    return {"tileset": tileset, "source": source, "uid": uid}


def land_tileset(ctx: Any, state: Any, tab: Any, result: dict[str, Any]) -> None:
    """Adopt an arrived tileset onto ``tab``: the whole tail of the arrival.

    Extracted from ``plotter_mode.on_task_done``'s ``plotter-tileset`` arm so
    the confirm popup's Import button reaches the *same* code rather than a hand
    copy of it. The terrain hand-off in particular is easy to leave out of a
    copy, and leaving it out produces a terrain set that is in the map, is
    selected in the palette, and cannot be painted with.
    """
    tileset = result["tileset"]
    want = result.get("projection")
    if want and want != tab.doc.projection:
        # One step, not two. A tileset that arrived while the projection
        # did not would leave a map painting the wrong lattice, one
        # Ctrl+Z away from a state nobody asked for.
        #
        # No door has set ``projection`` since the ground generator was
        # deleted on 2026-08-18 -- the new-map dialog owns the lattice
        # now -- so this arm is currently reached only from tests. Kept
        # for ``set_projection``'s own reason: it is the correct shape
        # for a door that carries a lattice with its art, and re-adding
        # it later would mean re-deriving the one-undo-step argument.
        tab.doc.set_projection(want, adding=tileset, source=result.get("source", ""))
    else:
        tab.doc.add_tileset(tileset, source=result.get("source", ""))
    state.tileset_index = len(tab.doc.tilesets) - 1
    state.brush = None
    if tileset.terrains:
        # The thing that just arrived is the thing in your hand, which
        # is exactly what setting ``tileset_index`` says for a palette.
        state.terrain = (state.tileset_index, 0)
    # An isometric map's pixel extent is not width times tile width, so
    # a fit computed against the old projection is the wrong frame.
    tab.view.fitted = False
    ctx.toast("Tileset added.")


def _parked(ctx: Any) -> tuple[Any, Any, tuple[Any, ...]] | None:
    """The parked sheet and the tab it names, or ``None`` (clearing as it goes).

    **By parked uid, never by the active tab.** The popup belongs to the tab the
    sheet arrived for; answering it after switching tabs -- which the frame
    thread cannot rule out -- would otherwise add the sheet to whichever map
    happens to be in front.
    """
    state = ensure(ctx)
    parked = state.sheet_import
    if parked is None:
        return None
    tab = state.get(parked[0])
    if tab is None:
        clear_sheet_import(ctx)
        return None
    return state, tab, parked


def clear_sheet_import(ctx: Any) -> None:
    """Drop the parked sheet and close its popup. Cancel, and the escape hatch
    for every path below that decides there is nothing to import."""
    state = ensure(ctx)
    state.sheet_import = None
    state.sheet_import_open = False


def import_detected_sheet(ctx: Any) -> bool:
    """Recompose the parked sheet on its detected grid and land it.

    Frame thread, like Packwright's ``import_tileset`` and for the same reason:
    the work is a handful of index gathers over an already-decoded array --
    milliseconds -- and routing it through a task would put a second popup-shaped
    round trip between the button and the tiles.
    """
    from .tilegrid import slicing
    from .tilegrid.tileset import Tileset

    found = _parked(ctx)
    if found is None:
        return False
    state, tab, (_uid, name, _source, pixels, grid) = found
    if isinstance(grid, SheetTerrain):
        # Its own door: reordering is not recomposing, and mixing the two would
        # slice a terrain sheet on a grid nobody measured.
        return import_sheet_terrain(ctx)
    if isinstance(grid, SheetMismatch):
        # No rules to find, but the sheet's own recorded grid is known: cut on
        # *that* and redraw each cell at the map's size. Same machinery, a grid
        # that came from a record rather than from a measurement.
        grid = slicing.uniform_grid(pixels.shape[:2], grid.tile_w, grid.tile_h)
        if grid is None:
            ctx.toast("That sheet holds no whole tile at its own size.", "error")
            clear_sheet_import(ctx)
            return False
    try:
        atlas = slicing.recompose(pixels, grid, tab.doc.tile_w, tab.doc.tile_h)
        tileset = Tileset(
            name=name, pixels=atlas, tile_w=tab.doc.tile_w, tile_h=tab.doc.tile_h
        )
    except ValueError as exc:
        ctx.toast(f"That sheet was not imported: {exc}.", "error")
        clear_sheet_import(ctx)
        return False
    clear_sheet_import(ctx)
    # ``source=""`` deliberately, and this is the rule every recomposing door
    # inherits: ``TilesetRef.source`` is what a ``.tmx`` export writes as its
    # external-file reference, and the recomposed atlas is no longer the file on
    # disk -- the rules are gone and every cell has moved. Keeping the path
    # would export a map pointing at art it is not drawing.
    land_tileset(ctx, state, tab, {"tileset": tileset, "source": ""})
    return True


#: The swatch colours an inferred terrain gets. A default rather than a
#: measurement: they name the terrain in the palette list and on the minimap and
#: nothing is stored keyed on them, so the user renaming or recolouring the
#: terrain later changes only what they see.
TERRAIN_FILL = (110, 170, 90, 255)
TERRAIN_OUTLINE = (40, 80, 40, 255)


def import_sheet_terrain(ctx: Any) -> bool:
    """Reorder the parked sheet into the canonical blob layout and land it.

    **The reorder *is* the role assignment.** A terrain set's layout is
    positional -- ``Tileset.terrain_of`` and ``local_for`` are the only source of
    a cell's role, and exactly 47 columns are enforced at construction -- so
    importing without reordering would give every tile a wrong role, silently,
    and every stroke would draw the wrong picture.

    ``source=""`` for the recomposing doors' reason: a reordered atlas is no
    longer the file on disk, and a ``.tmx`` export must not reference art it is
    not drawing.
    """
    from .tilegrid import roles as rolelib
    from .tilegrid.tileset import TerrainSpec, Tileset

    found = _parked(ctx)
    if found is None:
        return False
    state, tab, (_uid, name, _source, pixels, parked) = found
    if not isinstance(parked, SheetTerrain):
        return False
    tile_w, tile_h = parked.tile
    try:
        atlas = rolelib.reorder(pixels, parked.roles, tile_w, tile_h)
        tileset = Tileset(
            name=name,
            pixels=atlas,
            tile_w=tile_w,
            tile_h=tile_h,
            terrains=(TerrainSpec(name=name or "Terrain", fill=TERRAIN_FILL,
                                  outline=TERRAIN_OUTLINE),),
        )
    except ValueError as exc:
        ctx.toast(f"That terrain set was not imported: {exc}.", "error")
        clear_sheet_import(ctx)
        return False
    clear_sheet_import(ctx)
    land_tileset(ctx, state, tab, {"tileset": tileset, "source": ""})
    return True



def import_sheet_blind(ctx: Any) -> bool:
    """Land the parked sheet the way it would have landed with no detector:
    sliced whole at the map's tile size, source kept.

    This is the mitigation for a false positive. Genuinely dark art -- a night
    scene, a silhouette sheet -- can rule itself off convincingly, and the answer
    to that is a button that does exactly what yesterday's build did.
    """
    from .tilegrid.tileset import Tileset

    found = _parked(ctx)
    if found is None:
        return False
    state, tab, (_uid, name, source, pixels, parked) = found
    # A terrain parking was read on the map's own grid already, so "plain tiles"
    # here is the very slice that would have happened without the analyzer.
    del parked
    try:
        tileset = Tileset(
            name=name, pixels=pixels, tile_w=tab.doc.tile_w, tile_h=tab.doc.tile_h
        )
    except ValueError as exc:
        ctx.toast(f"That sheet was not imported: {exc}.", "error")
        clear_sheet_import(ctx)
        return False
    clear_sheet_import(ctx)
    land_tileset(ctx, state, tab, {"tileset": tileset, "source": source})
    return True


def choose_layer_image(ctx: Any, layer_uid: int) -> None:
    """Pick a picture for an image layer, decode it, attach it.

    The picker and the decode on one task thread, ``ask_add_tileset``'s shape and
    for its reason: routing a bare path back through the frame thread only to
    submit a second job with it gives the intermediate result nowhere sensible to
    live while it waits.

    ``.wmap`` already stores an image layer's pixels (as ``images/N.png``), so
    nothing about the container changes and this needs no version bump.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        from ..service.errors import invalid_from

        # ``(label, patterns)`` pairs, and the pairing is the whole of it --
        # portable-file-dialogs reads a filter list two at a time, so a label
        # followed by a splatted glob list advertises formats it then filters
        # out. ``filetypes`` builds both halves from one suffix list.
        path = dialogs.open_file(
            "Choose an image",
            [filetypes.describe("Images"), filetypes.pattern()],
        )
        if path is None:
            return None
        try:
            pixels = _decode(path)
        except ValueError as exc:
            raise invalid_from(exc, "That image could not be opened", field="file") from exc
        return {"image": (int(layer_uid), str(path), pixels), "uid": uid}

    ctx.submit(f"plotter-layer-image:{uid}", run)


def land_layer_image(ctx: Any, tab: Any, result: dict[str, Any]) -> None:
    """Adopt a chosen picture onto its image layer. Frame thread."""
    layer_uid, source, pixels = result["image"]
    try:
        tab.doc.set_image_pixels(layer_uid, pixels, source=source)
    except (KeyError, ValueError) as exc:
        ctx.toast(f"That image was not attached: {exc}.", "error")
        return
    tab.view.fitted = False
    ctx.toast("Image attached.")


def polish_in_inker(ctx: Any, tab: Any, index: int) -> None:
    """Open a tileset's atlas as an ordinary Inker document.

    Flat and unlinked on purpose. Slicing it into cells -- which is what
    ``inker.sheetin`` does for a sprite sheet -- would be the wrong model here:
    a polish pass on a blob set is *about* keeping the outline consistent
    **across** adjacent cases, and 235 one-cell frames makes exactly that
    impossible to see.

    The way back is the Plotter side pulling the document in, which is the
    direction Packwright already takes documents from Inker.
    """
    from . import inker_mode

    if index < 0 or index >= len(tab.doc.tilesets):
        return
    ref = tab.doc.tilesets[index]
    # Already a frozen RGBA copy, so there is nothing to decode and nothing the
    # editor can write through into the tileset the map is drawing from.
    inker_mode.open_pixels(ctx, ref.tileset.pixels, title=f"{ref.tileset.name} (atlas)")
    ctx.toast("Atlas opened in Inker.")


def tileset_from_inker(ctx: Any, doc: Any, *, index: int | None = None) -> None:
    """An Inker document back onto the map, as art for an existing tileset.

    Flattened on the frame thread, for the reason Packwright flattens there:
    the composite fills and evicts the document's own cache, which the pane
    drawing it is touching.

    ``index`` names the tileset to repaint. Without one this is an ordinary
    append, which is what an unrelated drawing should be.
    """
    from .tilegrid.tileset import Tileset, repolish

    tab = active(ctx)
    if tab is None:
        return
    # ``matte=False`` because a tileset's transparency is meaningful -- the
    # checkerboard the editor draws under a document is not part of the art.
    pixels = doc.flatten(matte=False)
    if index is None:
        tab.doc.add_tileset(
            Tileset(
                name=doc.title or "Atlas",
                pixels=pixels,
                tile_w=tab.doc.tile_w,
                tile_h=tab.doc.tile_h,
            )
        )
        ctx.toast("Tileset added.")
        return
    ref = tab.doc.tilesets[index]
    try:
        tab.doc.replace_tileset(index, repolish(ref.tileset, pixels))
    except ValueError as exc:
        ctx.toast(f"The tileset was not replaced: {exc}.", "error")
        return
    ctx.toast(f"{ref.tileset.name} repainted.")


def use_inker_tileset(ctx: Any, doc: Any, uid: int) -> None:
    """A fifth door onto the tileset-arrival key: an Inker document's tileset,
    handed to the active Plotter tab exactly as it stands.

    Zero-copy and **snapshot semantics**, both by construction rather than by
    anything this function does. ``doc.tileset_slot(uid).tileset`` is the very
    frozen object ``PlotterDoc.add_tileset`` (through ``on_task_done``'s
    ``plotter-tileset`` branch, shared with the other three doors) puts into
    the map's ``TilesetRef`` -- no copy is made on the way across. And every
    ``_apply_tileset_*`` hook on the Inker side replaces ``slot.tileset``
    wholesale rather than mutating it (``_doc_tiles.py``'s own rule: the
    frozen tileset is rebuilt, never written through), so a later edit there
    mints a brand new object and leaves this one -- and the map painting from
    it -- untouched.

    ``doc`` rather than a uid into some registry of open Inker tabs: the
    caller (a future tile-picker pane, Task 8) already has the document in
    hand, and a second uid namespace here would only be one more thing that
    can point at nothing.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return

    def run() -> dict[str, Any] | None:
        try:
            slot = doc.tileset_slot(uid)
        except KeyError:
            return None
        return {"tileset": slot.tileset, "source": "", "uid": tab.uid}

    ctx.submit(f"plotter-tileset:{tab.uid}", run)


def ask_add_tileset(ctx: Any) -> None:
    """The picker and the decode on one task thread.

    One task rather than a pick task and an add task: the pane would otherwise
    have to route a bare path back through the frame thread only to submit a
    second job with it, and the intermediate result has nowhere sensible to
    live while it waits.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    width, height = int(tab.doc.tile_w), int(tab.doc.tile_h)
    uid = tab.uid

    def run() -> dict[str, Any] | None:
        from ..service.errors import invalid_from
        from .plotter import tsx as tsxlib

        path = dialogs.open_file("Add a tileset", TILESET_FILTER)
        if path is None:
            return None
        try:
            if path.suffix.lower() == ".tsx":
                data = _within_ceiling(path).read_bytes()
                image = tsxlib.tsx_source(data)
                tileset = tsxlib.read_tsx(data, _decode(_resolve_source(path.parent, image)))
            else:
                return _sheet_or_tileset(
                    path.stem, str(path), _decode(path), uid, width, height
                )
        except ValueError as exc:
            raise invalid_from(exc, "This tileset could not be added", field="file") from exc
        return {"tileset": tileset, "source": str(path), "uid": uid}

    ctx.submit(f"plotter-tileset:{uid}", run)


def _recorded_cell(ctx: Any, job_id: str) -> tuple[int, int] | None:
    """The tile size a generated sheet's ``sheet.json`` declares, or ``None``.

    ``None`` for every job that has no sidecar -- an ordinary reference image,
    a render, anything not a tile sheet -- and those doors are untouched by
    design: absence of a record is not evidence of a mismatch.
    """
    import json

    from ..service import files as svc_files

    try:
        path = svc_files.job_dir_file(ctx.svc, job_id, "sheet.json")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return int(data["tile_w"]), int(data["tile_h"])
    except (KeyError, TypeError, ValueError):
        return None


def use_as_tileset(ctx: Any, job: Any) -> None:
    """A library asset's reference image, sliced as a tileset.

    The bytes are read on the task thread: an ``input.png`` is routinely several
    megabytes and decoding one between ``new_frame`` and ``render`` is the sort
    of stall the whole task layer exists to avoid.
    """
    tab = active(ctx)
    if tab is None:
        # Start the map rather than refuse the action. This is offered from the
        # Library for any asset carrying an ``input.png`` and nothing there
        # knows whether a map is open, so refusing here made the item an offer
        # the app took back -- and left the user to walk to Plotter, make a
        # map, walk back to the Library and find the card again. Asks rather
        # than invents (``plotter_mode.ask_new_document``'s rule), because the
        # tile size a map is built with is the number this action slices on;
        # the asset waits on ``pending_job_tileset`` and the dialog's Create
        # calls back in here with a tab to work on.
        from . import plotter_mode

        plotter_mode.ask_new_document(ctx)
        plotter_mode.ensure(ctx).pending_job_tileset = job
        return
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") or job_id) if isinstance(job, dict) else job_id
    width, height = int(tab.doc.tile_w), int(tab.doc.tile_h)

    uid = tab.uid

    def run() -> dict[str, Any]:
        from ..service import files as svc_files

        path = svc_files.job_dir_file(ctx.svc, job_id, "input.png")
        # Create's own sheets are drawn on imposed rectangles and carry no
        # separator lines, so detection finds nothing here and this is today's
        # blind slice. The call is made anyway because a *user's* image can
        # reach the library too, and the door should not depend on where the
        # bytes came from. What this door *does* have that the others do not is
        # the sheet's own sidecar, which records the size it was generated at.
        return _sheet_or_tileset(
            str(name),
            "",
            _decode(Path(path)),
            uid,
            width,
            height,
            recorded=_recorded_cell(ctx, job_id),
        )

    ctx.submit(f"plotter-tileset:{tab.uid}", run)
