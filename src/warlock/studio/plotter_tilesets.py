"""Getting a tileset onto the open map, from wherever it came from.

Five doors, one destination. A ``.tsx`` or an image from a picker, an image from
a path (a drop), a procedurally generated ground set, a document coming back
from Inker, and a library asset's own reference image -- and every one of them
ends as ``{"tileset": ..., "source": ..., "uid": ...}`` on the same
``plotter-tileset:`` task key, because ``plotter_mode.on_task_done`` already
routes and adopts one and a second key would be a second copy of that branch.
The duplicate-key refusal that comes with sharing the key is the right rule too:
a generate and an "Add from a file..." both mean "a tileset is arriving on this
tab".

Split out of ``plotter_mode`` on the seam the file layer left behind: what is
here is *about a tileset*, what is in :mod:`.plotter_io` is about a document's
bytes, and what is left in the mode is about tabs, tasks and keys. The two
filesystem questions -- may this path be followed, is this file small enough --
stay in ``plotter_io`` and are imported, because a ``.tsx`` picked here names an
image the same way a ``.tmx`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import dialogs, filetypes
from .plotter_io import _decode, _resolve_source, _within_ceiling
from .plotter_state import active

# A ``.tsx`` carries its own slicing; anything else is an image sliced at the
# map's tile size. Both halves of the entry are derived, because the label used
# to read "(*.tsx *.png)" over a pattern list that also accepted .jpg, .jpeg,
# .webp and .bmp -- a dialog disclaiming four formats it would have opened.
_TILESET_SUFFIXES = (".tsx", *filetypes.IMAGE_SUFFIXES)
TILESET_FILTER = [
    filetypes.describe("Tilesets and images", _TILESET_SUFFIXES),
    *filetypes.globs(_TILESET_SUFFIXES),
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
        from .plotter.tileset import Tileset

        try:
            if path.suffix.lower() == ".tsx":
                data = _within_ceiling(path).read_bytes()
                image = tsxlib.tsx_source(data)
                tileset = tsxlib.read_tsx(data, _decode(_resolve_source(path.parent, image)))
            else:
                tileset = Tileset(
                    name=path.stem, pixels=_decode(path), tile_w=width, tile_h=height
                )
        except ValueError as exc:
            raise invalid_from(exc, "This tileset could not be added", field="file") from exc
        return {"tileset": tileset, "source": str(path), "uid": tab.uid}

    ctx.submit(f"plotter-tileset:{tab.uid}", run)


def generate_terrain_set(ctx: Any, spec: Any) -> None:
    """A procedural ground set, built on a task thread and added like any other.

    Deliberately on the existing ``plotter-tileset:`` key rather than one of its
    own: the result *is* a tileset for this tab, ``on_task_done`` already routes
    and adopts one, and a second key would be a second copy of that branch. The
    duplicate-key refusal it inherits is the right rule too -- a generate and an
    "Add from a file..." both end in "a tileset is arriving on this tab".

    The pane builds the spec, which is plain frozen data, and nothing about a
    numpy raster happens on the frame thread.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    uid = tab.uid

    def run() -> dict[str, Any]:
        from .plotter import tilegen

        return {
            "tileset": tilegen.generate(spec),
            "source": "",
            "projection": spec.projection,
            "uid": uid,
        }

    ctx.submit(f"plotter-tileset:{uid}", run)


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
    from .plotter import tilegen
    from .plotter.tileset import Tileset

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
        tab.doc.replace_tileset(index, tilegen.repolish(ref.tileset, pixels))
    except ValueError as exc:
        ctx.toast(f"The tileset was not replaced: {exc}.", "error")
        return
    ctx.toast(f"{ref.tileset.name} repainted.")


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
        from .plotter.tileset import Tileset

        path = dialogs.open_file("Add a tileset", TILESET_FILTER)
        if path is None:
            return None
        try:
            if path.suffix.lower() == ".tsx":
                data = _within_ceiling(path).read_bytes()
                image = tsxlib.tsx_source(data)
                tileset = tsxlib.read_tsx(data, _decode(_resolve_source(path.parent, image)))
            else:
                tileset = Tileset(
                    name=path.stem, pixels=_decode(path), tile_w=width, tile_h=height
                )
        except ValueError as exc:
            raise invalid_from(exc, "This tileset could not be added", field="file") from exc
        return {"tileset": tileset, "source": str(path), "uid": uid}

    ctx.submit(f"plotter-tileset:{uid}", run)


def use_as_tileset(ctx: Any, job: Any) -> None:
    """A library asset's reference image, sliced as a tileset.

    The bytes are read on the task thread: an ``input.png`` is routinely several
    megabytes and decoding one between ``new_frame`` and ``render`` is the sort
    of stall the whole task layer exists to avoid.
    """
    tab = active(ctx)
    if tab is None:
        ctx.toast("Open or start a map first.", "error")
        return
    job_id = job["id"] if isinstance(job, dict) else str(job)
    name = (job.get("name") or job_id) if isinstance(job, dict) else job_id
    width, height = int(tab.doc.tile_w), int(tab.doc.tile_h)

    def run() -> dict[str, Any]:
        from ..service import files as svc_files
        from .plotter.tileset import Tileset

        path = svc_files.job_dir_file(ctx.svc, job_id, "input.png")
        tileset = Tileset(
            name=str(name), pixels=_decode(Path(path)), tile_w=width, tile_h=height
        )
        return {"tileset": tileset, "source": "", "uid": tab.uid}

    ctx.submit(f"plotter-tileset:{tab.uid}", run)
