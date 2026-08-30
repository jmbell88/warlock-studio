"""AI tile sheets: what one may ask for, and queuing the paint.

The door onto ``kind="tile_sheet"``. Modelled on ``sprites.create_sprite_synthesis``
and for its reason: every refusal that can be made from the form belongs here,
before a row exists, because the alternative is a place in the queue and a
minute of GPU spent on a request that was never going to produce a usable
sheet.

**Three modes, and two of them are new.**

``materials``
    N material descriptions, each generated on its own as a seamless 1024px
    tile and reduced into a cell of a plain grid. The default, because it is
    the one that produces N genuinely different tiles.

``terrain``
    Two seamless materials composited into a blob-47 autotile set by
    ``pipelines.tilemask``. The model draws two surfaces and never sees an
    edge.

``grid``
    The original path: one 1024px frame painted through a canny guide and cut
    into sixty-four cells. ``docs/measurements/2026-08-18-tile-sheet-grid.md``
    measured what it produces -- every cell of the guide is identical, so there
    is no per-cell signal for variety and the model answers with one scene cut
    up or one tile repeated. It still builds, because rerunning a sheet made
    last week is not an error, but it is **refused for a new request**: see
    ``allow_grid`` on :func:`create_tile_sheet`.

**What each mode loads is a property of the mode, not of the kind.** The grid
guide *is* a ControlNet and the seamless modes never touch one, so the required
weights are :func:`rows_needed`'s answer rather than one tuple -- a host with no
canny weights can build materials and terrain, and used to be refused at the
door for a download the request would never have opened.

**The geometry rules are restated, not imported.** ``service/`` may not import
``studio/`` and a pipeline is the wrong place to look up a form's ceiling, so
the tile sizes and the projection list are literals here -- and
``tests/test_tilesheet_service.py`` and ``tests/test_tileset_service.py`` pin
each of them to ``pipelines.tilesheet`` and ``pipelines.tileatlas``, so the copy
cannot drift without a red test saying which one moved. The *seamless* modes'
own two refusals -- which views tile and which tile sizes divide a 1024px
material -- are delegated to ``pipelines.tileatlas`` rather than restated,
because each of them is a sentence explaining a fact about tiling and two
copies of such a sentence is one copy away from being wrong.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .. import models
from ..asset_workflows import (
    MAX_COLLECTION_CELLS,
    MAX_COLLECTION_LINES,
    MAX_COLLECTION_VARIANTS,
    collection_cells,
)
from .core import WarlockService
from .errors import Invalid, TooLarge
from .files import ImageTooLarge, to_png
from .validation import (
    MAX_JOB_NAME,
    MAX_UPLOAD_BYTES,
    check_base_model_weights,
    check_prompt,
    check_seed,
    check_vram,
    install_remedy,
    random_seed,
)

#: The base a tile sheet is pinned to. Not ``models.DEFAULT_BASE_MODEL``: this
#: path wants full CFG so the negative prompt actually steers, and it has to
#: keep wanting it if the default ever moves. ``sprites.py``'s constant, for
#: ``sprites.py``'s reason.
TILE_SHEET_BASE_MODEL = "sdxl_cfg"

#: What the *grid* mode additionally loads: (registry kind, registry key, the
#: form field a refusal about it names). The grid guide **is** the ControlNet --
#: without it there is no grid, only one picture -- and it is here rather than
#: in :data:`_REQUIRED_ALWAYS` because the seamless modes never open it. Making
#: it a requirement of the *kind* refused materials and terrain sheets on a host
#: with no canny weights, for a download those requests would never have used.
_REQUIRED_GRID: tuple[tuple[str, str, str], ...] = (("control", "canny", "control"),)

#: What every mode loads. The art style *is* the LoRA, in all three: a missing
#: one is not a plainer picture, it is sixty-four photographs of gravel.
_REQUIRED_ALWAYS: tuple[tuple[str, str, str], ...] = (
    ("lora", models.PIXEL_SHEET_LORA, "style_lora"),
)

#: What an *attached reference* additionally needs. Separate from the tuples
#: above because a reference is optional: making the adapter unconditional
#: would refuse the common prompt-only request on a host that has everything
#: the common request actually uses.
_REFERENCE_REQUIRED: tuple[str, str, str] = ("adapter", "plus", "ip_adapter")

MODE_MATERIALS = "materials"
MODE_TERRAIN = "terrain"
MODE_GRID = "grid"

#: The three shapes a request can take, and which one an unstated request means.
#: ``materials`` rather than ``grid``, because ``grid`` is the one the
#: measurement above says does not work; it survives for reruns.
#: ``pipelines.tileatlas.MODES`` is the first two of these, and
#: ``tests/test_tileset_service.py`` pins the pair.
TILE_MODES: tuple[str, ...] = (MODE_MATERIALS, MODE_TERRAIN, MODE_GRID)
DEFAULT_MODE = MODE_MATERIALS

#: What the pane calls each mode. Here rather than in the pane for
#: ``view_labels``' reason: a fourth mode should not need an edit in a file that
#: knows nothing about what a mode is.
MODE_LABELS: dict[str, str] = {
    MODE_MATERIALS: "Materials",
    # "Terrain set" and not "Terrain": what this mode produces is the whole
    # forty-seven case autotile *set*, and a picker entry reading "Terrain"
    # beside "Materials" reads as "one terrain" -- which is the one thing it
    # never is.
    MODE_TERRAIN: "Terrain set",
    MODE_GRID: "Grid (legacy)",
}

#: The layouts a terrain set may be laid out on. One, and it is a list anyway:
#: ``pipelines.tilemask``'s forty-seven blob cases are what
#: ``Tileset.local_for`` indexes, and a set that named any other layout would be
#: a set whose columns mean nothing.
TERRAIN_LAYOUTS: tuple[str, ...] = ("blob47",)
DEFAULT_TERRAIN_LAYOUT = "blob47"

#: How many prompt lines, how many draws of each, and the ceiling on their
#: product. **Aliases and not copies**: ``asset_workflows.collection_cells`` is
#: what actually expands the lines into cells, so a second set of numbers here
#: is a door that accepts what the expansion then refuses -- with the request
#: already past every other check. The pair is one rule with two enforcement
#: points (``pipelines.tileatlas`` holds the third, on the product alone), and
#: the aliasing is what keeps them from drifting.
MAX_MATERIALS = MAX_COLLECTION_LINES
MAX_VARIANTS = MAX_COLLECTION_VARIANTS
MAX_CELLS = MAX_COLLECTION_CELLS

#: The longest a generated terrain's palette name may be. A terrain name is what
#: the Plotter palette shows in a list beside a swatch, and a material
#: description is capped at ``MAX_PROMPT`` (1000) -- so the name is the
#: description's first words rather than the description.
MAX_TERRAIN_NAME = 40

#: The two placeholder swatches a generated terrain set is landed with, in
#: ``(inner, outer)`` order. ``studio.plotter.terrain``'s own first two defaults
#: -- Grass and Dirt -- restated because ``service/`` may not import ``studio/``.
#:
#: **Placeholders on purpose.** A generated terrain's colour is not knowable
#: from its description at the door, the pixels do not exist yet, and the swatch
#: is a palette affordance rather than a fact about the set -- so these are what
#: the palette shows until somebody recolours them, and never what anything
#: draws. The outline is the fill at three fifths, which is
#: ``plotter_tools``' own derivation.
_TERRAIN_SWATCHES: tuple[tuple[int, int, int, int], ...] = (
    (106, 153, 78, 255),
    (156, 122, 84, 255),
)

#: ``tilesheet.TILE_SIZES`` and ``tilesheet.VIEWS``, restated.
TILE_SIZES: tuple[int, ...] = (16, 32, 48, 64)
VIEWS: tuple[str, ...] = ("top_down", "three_quarter", "isometric")

#: The one stored spelling that is not in :data:`VIEWS`. Restated here for the
#: same reason the list is: a door may not import the pipeline's constants, and
#: a request carrying the old word has to be accepted rather than refused --
#: rerolling a sheet made last week is not an error.
LEGACY_VIEWS: dict[str, str] = {"orthogonal": "top_down"}

#: One palette across the whole sheet, at the size the ground run measured
#: (``docs/measurements/2026-08-17-ground-reduction.md``: occupancy at 32 was
#: saturated, 64 stayed above 95%). Not a control, for the reason the module
#: docstring gives.
#:
#: **The grid mode's value, and only that.** The measurement was taken over
#: *one* generation of *one* subject, which is exactly what a grid sheet is: one
#: 1024px frame cut into sixty-four cells of the same scene. It is not a
#: statement about sixteen unrelated materials -- see :func:`sheet_colors`.
SHEET_COLORS = 64

DEFAULT_TILE_SIZE = 32
DEFAULT_VIEW = "top_down"


def sheet_colors(cells: int) -> int:
    """How many palette entries one seamless sheet is quantized to.

    ``min(256, max(64, 32 * cells))`` -- thirty-two entries a material, floored
    at the grid mode's sixty-four and ceilinged at a byte.

    **Provisional, and deliberately labelled so.** :data:`SHEET_COLORS` is a
    *measured* constant and this is not one: it is an argument, which is that
    the measurement behind 64 was taken over one generation of one subject and
    says nothing about sixteen unrelated materials sharing one table. Sixteen
    materials in sixty-four entries is four colours each, which is not a shared
    palette, it is a posterisation. Thirty-two a material is roughly what the
    ground run found *one* material wanted; the floor keeps a one- or
    two-material request byte-identical to what it produces today; the ceiling
    is where an indexed PNG stops being indexed.

    It changes the bytes of every sheet it touches, so it is the kind of number
    this repo owes a ``docs/measurements/`` document before it moves again. What
    would move it: occupancy measured per material at 2, 4, 8 and 16 materials
    against the shared table, the way the ground run measured it at one -- and
    in particular whether the saturation point rises linearly with the material
    count at all, which is the assumption the ``32 *`` encodes and nothing has
    yet tested.

    **A named palette supersedes this entirely.** ``palette`` and this number
    are not two halves of one setting and never combine: with a palette file
    the median cut does not run, so there is no budget for it to bound and this
    function's provisionality stops mattering to that request. It is still
    computed and still stored, because it is what the row asked for and a reroll
    that later drops the palette needs an answer -- ``tilesheet.palette_record``
    is where the two meet on the way out, and its ``palette_source`` key is what
    says which one actually ran.
    """
    return min(256, max(SHEET_COLORS, 32 * int(cells)))


def rows_needed(mode: str, with_reference: bool = False) -> tuple[str, ...]:
    """Every registry row this mode needs on this host, in ``fetch.Entry``'s
    ``row_key`` spelling -- so a pane can offer "install what this needs"
    without knowing what a tile sheet is made of.

    A function of the mode rather than a constant, because the modes do not load
    the same things: the grid guide is a ControlNet and the seamless modes never
    open one. As a constant it refused a materials sheet on a host with no canny
    weights, which is a refusal about a download the request would not have
    used.
    """
    return (
        f"base:{TILE_SHEET_BASE_MODEL}",
        *(f"{kind}:{key}" for kind, key, _field in _required(mode, with_reference)),
    )


def _required(mode: str, with_reference: bool) -> tuple[tuple[str, str, str], ...]:
    """The non-base weights this mode loads, in check order."""
    rows = list(_REQUIRED_GRID) if mode == MODE_GRID else []
    rows.extend(_REQUIRED_ALWAYS)
    if with_reference:
        rows.append(_REFERENCE_REQUIRED)
    return tuple(rows)


#: The grid mode's rows, under the two names the 2D pane has always read. Kept
#: as module constants rather than folded into :func:`rows_needed` because
#: ``studio.panes.settings_2d`` reads both by name and its Sheet output is the
#: grid arm; a pane that grows a mode picker asks :func:`rows_needed` instead.
TILE_SHEET_ROWS: tuple[str, ...] = rows_needed(MODE_GRID)

#: With a reference attached. A superset, in check order.
TILE_SHEET_REFERENCE_ROWS: tuple[str, ...] = rows_needed(MODE_GRID, True)


def tile_sheet_options() -> dict[str, Any]:
    """What a tile-sheet request may ask for. One source for the form, so the
    pane never hardcodes a ceiling this module enforces.

    The grid summary comes from ``tilesheet.geometry`` rather than being written
    out again here -- ``sprite_options``' rule: the pane says "64 tiles, 8x8"
    under the control, and a second copy of that arithmetic is a label that
    goes stale the first time the grid moves.

    **The installed palettes are deliberately not in here**; they are
    :func:`tile_sheet_palettes`. This function takes no ``svc`` and reads no
    disk, and ``studio.panes.settings_2d`` caches its answer for the process
    lifetime in a one-slot list on exactly that ground -- "there is nothing for
    them to go stale against: neither reads config, disk or state". A palette is
    a *file the user dropped in a directory*, so folding the listing in here
    would make that comment false and cache a directory listing until the app
    restarts: drop in a palette, and it would not appear.
    """
    from ..pipelines import tileatlas, tilesheet

    sizes = []
    for size in TILE_SIZES:
        entries = {}
        for view in VIEWS:
            geom = tilesheet.geometry(size, view)
            entries[view] = {
                "tile_w": geom.tile_w,
                "tile_h": geom.tile_h,
                "sheet_w": geom.sheet_size[0],
                "sheet_h": geom.sheet_size[1],
            }
        sizes.append({"key": size, "views": entries})
    reference = tilesheet.geometry(DEFAULT_TILE_SIZE, DEFAULT_VIEW)
    # Which of the offered sizes a *seamless* material can be reduced to.
    # Derived by asking the pipeline rather than written out, so the pane never
    # holds a second opinion about which sizes divide a 1024px material: 48 is
    # on the grid menu and is not on this one, and the day 1024 changes this
    # list changes with it.
    seamless_sizes = [size for size in TILE_SIZES if not tileatlas.MATERIAL_PX % size]
    return {
        "tile_sizes": list(TILE_SIZES),
        "views": list(VIEWS),
        "modes": list(TILE_MODES),
        "mode_labels": dict(MODE_LABELS),
        # Per mode, because what a sheet loads is a property of the mode: the
        # pane must publish "install what this needs" for the mode that is
        # actually selected, and a single list made a materials sheet ask for a
        # ControlNet it will never open.
        "mode_rows_needed": {mode: list(rows_needed(mode)) for mode in TILE_MODES},
        "mode_reference_rows_needed": {
            mode: list(rows_needed(mode, True)) for mode in TILE_MODES
        },
        # What the seamless modes accept, which is a subset of the two menus
        # above: one view (a 3/4 tile has a visible front face and an isometric
        # tile is a diamond, so neither wraps) and only the sizes that divide a
        # material exactly.
        "seamless_views": list(tileatlas.VIEWS),
        "seamless_tile_sizes": seamless_sizes,
        "terrain_layouts": list(TERRAIN_LAYOUTS),
        "max_materials": MAX_MATERIALS,
        "max_variants": MAX_VARIANTS,
        "max_cells": MAX_CELLS,
        # The label the form draws, beside the key it submits. Here rather than
        # in the pane for ``tile_sizes``' reason: the pane hardcoded its two
        # labels and so could not have grown a third without an edit in a file
        # that knows nothing about what a view is.
        "view_labels": {
            "top_down": "Top-down",
            "three_quarter": "3/4",
            "isometric": "Isometric",
        },
        "sizes": sizes,
        "columns": reference.columns,
        "rows": reference.rows,
        "tiles": reference.tiles,
        "colors": SHEET_COLORS,
        "base_model": TILE_SHEET_BASE_MODEL,
        # The grid arm's, under the two names the 2D pane reads today. The
        # per-mode maps above are what a pane with a mode picker asks.
        "rows_needed": list(TILE_SHEET_ROWS),
        "reference_rows_needed": list(TILE_SHEET_REFERENCE_ROWS),
        "defaults": {
            "tile_size": DEFAULT_TILE_SIZE,
            "view": DEFAULT_VIEW,
            "mode": DEFAULT_MODE,
            "variants": 1,
            "terrain_layout": DEFAULT_TERRAIN_LAYOUT,
            "style_lock": False,
            "seam_erase": False,
        },
    }


def tile_sheet_palettes(svc: WarlockService) -> list[str]:
    """Every authored palette a tile sheet may be drawn on, by stem, sorted.

    Its own call rather than a key in :func:`tile_sheet_options` because it is
    the one answer on this door that can change without the process changing:
    the source is a directory the user drops files into, and the pane caches
    the options blob forever precisely because nothing in it reads disk. Asked
    per open of the form, which is what makes a palette dropped in five minutes
    ago appear -- it is one ``iterdir`` of a directory of small text files, not
    something to hold a cache over.

    Empty on a host with no palette directory, which is not an error: the whole
    feature is optional and a form that offers nothing is the correct rendering
    of "none installed" -- ``palettes.available``'s rule, not restated here
    because this delegates to it.
    """
    from . import palettes

    return palettes.available(svc.config)


def _check_weights(
    svc: WarlockService,
    *,
    mode: str = MODE_GRID,
    with_reference: bool = False,
    style_lock: bool = False,
) -> None:
    """Everything *this mode* loads, refused by name with its download line.

    ``validation.check_weights`` stays text-only on purpose -- it is keyed on
    ``params`` a text job wrote -- so this kind brings its own, exactly as
    ``sprites._check_weights`` does.

    ``mode`` defaults to the grid, which is the conservative end: it is the mode
    that needs the most, so a caller that cannot say which mode a stored row was
    made under asks for a superset rather than admitting a job whose weights are
    missing.
    """
    from .. import fetch
    from .downloads import needed_keys

    # A style lock loads the same encoder a reference does -- the first
    # material *is* the reference for the rest -- so it needs the same rows.
    wanted = rows_needed(mode, with_reference or style_lock)
    check_base_model_weights(
        svc,
        models.BASE_MODELS[TILE_SHEET_BASE_MODEL],
        rows=needed_keys(svc, wanted),
    )
    for kindname, key, field in _required(mode, with_reference or style_lock):
        entry = fetch.find(f"{kindname}:{key}")
        assert entry is not None, f"{kindname}:{key} is not a registry row"
        spec = entry.spec
        if fetch.present(svc.config, kindname, spec):
            continue
        raise Invalid(
            f"A tile sheet needs {spec.label!r}, which is not downloaded. "
            f"{install_remedy(spec.label, fetch.download_text(svc.config, kindname, spec))}",
            field=field,
            # Every row the feature is short of, not merely the one that
            # tripped: a user offered "install what this needs" wants one
            # download, not three refusals in a row.
            rows=needed_keys(svc, wanted),
        )


def _seamless_refusal(exc: ValueError, *, view: str, size: int) -> Invalid:
    """``tileatlas``' own refusal, landed on the control it is about.

    The sentence is not rewritten here. Both of the refusals that can reach this
    point -- the two views that cannot tile and the tile sizes that do not
    divide a 1024px material -- are explanations of a fact about tiling, and a
    second copy of such an explanation is one edit away from being the wrong
    one.

    Which control it names is decided from ``tileatlas``' own published list
    rather than from the text: the cell count is checked at this door *before*
    a geometry is asked for, so the view and the size are the only two things
    left that the geometry can refuse.
    """
    from ..pipelines import tileatlas

    return Invalid(str(exc), field="projection" if view not in tileatlas.VIEWS else "tile_size")


def _terrain_records(inner: str, outer: str) -> list[dict[str, Any]]:
    """The two terrains a set declares, in precedence order.

    Ordered, and the order is meaning: a terrain's position is its precedence,
    which is ``TerrainSpec``'s rule and the whole of how a cell with two
    terrains around it picks one picture. ``inner`` first, because ``inner`` is
    the one the forty-seven blob cases are pictures *of*.
    """
    out = []
    for text, fill in zip((inner, outer), _TERRAIN_SWATCHES, strict=True):
        out.append(
            {
                "name": str(text).strip()[:MAX_TERRAIN_NAME].strip(),
                "fill": list(fill),
                "outline": [*(part * 3 // 5 for part in fill[:3]), fill[3]],
            }
        )
    return out


def create_tile_sheet(
    svc: WarlockService,
    *,
    prompt: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    view: str = DEFAULT_VIEW,
    seed: int | None = None,
    negative_prompt: str | None = None,
    reference: bytes | None = None,
    asset_type: str | None = None,
    asset_intent: str | None = None,
    mode: str = DEFAULT_MODE,
    prompt_items: Any = (),
    variants: int = 1,
    inner_terrain: str = "",
    outer_terrain: str = "",
    boundary: str = "",
    terrain_layout: str = DEFAULT_TERRAIN_LAYOUT,
    style_lock: bool = False,
    seam_erase: bool = False,
    palette: str = "",
    dither: bool = False,
    outline: str | None = None,
    allow_grid: bool = False,
) -> dict[str, Any]:
    """Queue a sheet of generated tiles. Three shapes -- see the module docstring.

    ``prompt`` is the **style sentence** in the two seamless modes: the words
    every cell shares. What each cell is *of* comes from ``prompt_items``
    (materials) or from ``inner_terrain``/``outer_terrain`` (terrain). In grid
    mode it is the whole subject, as it always was.

    ``boundary`` is **context, and never an instruction**. It is appended to
    *both* terrain subjects so two independent samples come back sharing a world
    and a palette -- "a temperate coastline" gives grass and water that belong to
    the same map. It is not a description of the seam, because there is no seam
    to describe: the boundary is a scalar field computed by
    ``pipelines.tilemask``, and words asking SDXL for a shoreline would put a
    *drawn* edge inside a tile that the field then cuts across -- which is the
    one defect the composited path exists to make impossible. Anything
    "improving" this into a drawing request is reintroducing it.

    ``palette`` is the stem of an authored palette file -- ``""`` means "derive
    one", which is what every sheet before today did and still does. ``dither``
    turns on the ordered 4x4 offset in ``pixel.map_palette``. The two are
    independent: a dithered sheet with no palette named still derives its own
    table by median cut and then dithers against it.

    ``outline`` exists only to be **refused by name**. A tile sheet has no
    outline pass, and a caller that hands one over has misunderstood what this
    kind is rather than asked for something unavailable. It is forwarded to
    ``check_pixel_options`` like every other option -- ``allow_outline=False``
    is what refuses it -- and the sentence carrying the reason is passed with
    it, because the reason is a fact about tiles rather than about the checker.
    ``"none"`` is not an ask and is accepted in silence.

    ``allow_grid`` is the escape hatch on the one refusal here that is about a
    measurement rather than about an impossibility: the grid mode still builds,
    so anything rerunning a stored grid request passes it, and a *new* request
    is told which two modes replace it.

    Nothing is written until every check has passed, which is the shape
    ``create_pixel_sheet`` uses: a form that asks for an impossible tile, or
    that asks on a host with no pixel-art LoRA, costs the request rather than a
    minute of GPU and a sheet that merely came back plain.
    """
    from ..pipelines import tileatlas, tilemask, tilesheet

    mode_key = str(mode)
    if mode_key not in TILE_MODES:
        raise Invalid(
            f"{mode_key!r} is not a tile-sheet mode; this door builds "
            f"{', '.join(TILE_MODES)}",
            field="mode",
        )
    if mode_key == MODE_GRID and not allow_grid:
        raise Invalid(
            "the grid layout paints one frame through a guide whose sixty-four "
            "cells are identical, so there is no per-cell signal for variety and "
            "it comes back as one scene cut up or as one tile repeated "
            "(docs/measurements/2026-08-18-tile-sheet-grid.md). Ask for "
            "'materials' -- a list of surfaces, each generated seamlessly on its "
            "own -- or for 'terrain', two surfaces composited into an autotile "
            "set.",
            field="mode",
        )

    text = str(prompt or "").strip()
    if not text:
        raise Invalid("a tile sheet needs a prompt", field="prompt")
    check_prompt(text)

    # ``None`` means "the form said nothing", and what the absent value means is
    # the pipeline's own default -- an empty *string* is a user explicitly
    # asking for no negative prompt, and is honoured. ``grounds.py``'s rule.
    negative = (
        tilesheet.SHEET_NEGATIVE_PROMPT
        if negative_prompt is None
        else str(negative_prompt)
    )
    check_prompt(negative, field="negative_prompt")

    # Coerced inside a guard: these are form values, and a bare ``int()`` over
    # one turns a typo into an unhandled TypeError/ValueError rather than the
    # refusal-with-a-field this module exists to produce.
    try:
        size = int(tile_size)
    except (TypeError, ValueError):
        raise Invalid("tile_size must be a whole number", field="tile_size") from None
    if size not in TILE_SIZES:
        raise Invalid(
            f"{size} is not a tile size this sheet can publish; "
            f"choose one of {list(TILE_SIZES)}",
            field="tile_size",
        )
    # ``field="projection"`` and not ``"view"``: the *form field* is still
    # called that, and a refusal has to name the control the user is looking at.
    # The vocabulary moved; the field key did not.
    resolved = LEGACY_VIEWS.get(str(view), str(view))
    if resolved not in VIEWS:
        raise Invalid(
            f"{resolved!r} is not a projection this sheet can draw; "
            f"choose one of {list(VIEWS)}",
            field="projection",
        )
    expected_asset_type = {
        "top_down": "tileset_top_down",
        "three_quarter": "tileset_three_quarter",
        "isometric": "tileset_isometric",
    }[resolved]
    # ``tileset`` is the consolidated Create vocabulary. The directional
    # aliases remain accepted for old service clients and still validate the
    # view they name.
    accepted_asset_types = {"tileset", expected_asset_type}
    if asset_type is not None and asset_type not in accepted_asset_types:
        raise Invalid("asset_type does not match the tile-set view", field="asset_type")
    if asset_intent is not None and asset_intent != "tileset":
        raise Invalid("a tile set must use asset_intent='tileset'", field="asset_intent")
    if seed is not None:
        check_seed("seed", seed)
    # Drawn here rather than at the params literal below, because in the two
    # seamless modes it is an *input* to the request and not only a record: each
    # cell's own seed is derived from it, so the cells cannot be compiled until
    # this number exists.
    sheet_seed = random_seed() if seed is None else int(seed)

    # -- what this mode is a sheet of -----------------------------------------
    #
    # The geometry is derived once, here, the block's single writer -- so the
    # worker and anything reading the row later share one stored fact rather
    # than each re-deriving the table.
    materials: list[dict[str, Any]] = []
    terrains: list[dict[str, Any]] = []
    mask: dict[str, Any] | None = None
    layout = "grid"

    if mode_key == MODE_MATERIALS:
        lines = tuple(
            line for line in (str(item).strip() for item in prompt_items or ()) if line
        )
        if not lines:
            raise Invalid(
                "a materials sheet is the list of surfaces you type; describe at "
                "least one",
                field="prompt_items",
            )
        if len(lines) > MAX_MATERIALS:
            raise Invalid(
                f"{len(lines)} material lines is past the {MAX_MATERIALS} one sheet "
                f"can name",
                field="prompt_items",
            )
        for line in lines:
            check_prompt(line, field="prompt_items")
        try:
            draws = int(variants)
        except (TypeError, ValueError):
            raise Invalid("variants must be a whole number", field="variants") from None
        if not 1 <= draws <= MAX_VARIANTS:
            raise Invalid(
                f"{draws} draws of each material is outside 1..{MAX_VARIANTS}",
                field="variants",
            )
        if len(lines) * draws > MAX_CELLS:
            raise Invalid(
                f"{len(lines)} materials by {draws} variants is {len(lines) * draws} "
                f"cells, past the {MAX_CELLS} one sheet can hold; each cell is its "
                f"own full generation",
                field="variants",
            )
        # The expansion itself is ``asset_workflows``', not a second copy of it:
        # the three bounds above are that function's own, aliased rather than
        # restated, and this call is what turns the lines into cells and derives
        # each cell's seed.
        materials = [dict(cell) for cell in collection_cells(lines, draws, seed=sheet_seed)]
        try:
            atlas = tileatlas.material_geometry(size, resolved, len(materials))
        except ValueError as exc:
            raise _seamless_refusal(exc, view=resolved, size=size) from exc
        columns, rows, tile_w, tile_h, tiles = (
            atlas.columns,
            atlas.rows,
            atlas.tile_w,
            atlas.tile_h,
            atlas.tiles,
        )
        layout = atlas.layout
        colors = sheet_colors(len(materials))
    elif mode_key == MODE_TERRAIN:
        inner = str(inner_terrain or "").strip()
        outer = str(outer_terrain or "").strip()
        edge_context = str(boundary or "").strip()
        # Both halves, because both are generated. A terrain set is two seamless
        # materials composited through a coverage field; a request that
        # describes one of them has nothing to put on the other side of every
        # boundary.
        # Naming the field it is about, for the reason the pane's own copy of
        # this check records: both halves are empty on a fresh request, and a
        # refusal that says only "both have to be described" leaves the user
        # to guess which of the two the door actually stopped on.
        for name, label, field_text in (
            ("inner_terrain", "Inside", inner),
            ("outer_terrain", "Outside", outer),
        ):
            if not field_text:
                raise Invalid(
                    f"{label} is empty; a terrain set is two surfaces and both "
                    f"are generated, so both have to be described",
                    field=name,
                )
            check_prompt(field_text, field=name)
        check_prompt(edge_context, field="boundary")
        if str(terrain_layout) not in TERRAIN_LAYOUTS:
            raise Invalid(
                f"{terrain_layout!r} is not a terrain layout; "
                f"choose one of {list(TERRAIN_LAYOUTS)}",
                field="terrain_layout",
            )
        try:
            atlas = tileatlas.terrain_geometry(size, resolved)
        except ValueError as exc:
            raise _seamless_refusal(exc, view=resolved, size=size) from exc
        columns, rows, tile_w, tile_h, tiles = (
            atlas.columns,
            atlas.rows,
            atlas.tile_w,
            atlas.tile_h,
            atlas.tiles,
        )
        layout = atlas.layout
        # Two source materials, in the order ``tileatlas.terrain_subjects``
        # takes them -- **not** the forty-seven cells of the atlas, which are
        # composites of these two rather than generations of their own. Their
        # seeds come from the same derived family a materials sheet uses, so one
        # recorded seed reproduces the pair.
        seeds = tileatlas.material_seeds(sheet_seed, 2)
        materials = [
            {"index": index, "prompt": subject, "variant": 1, "seed": int(cell_seed)}
            for index, (subject, cell_seed) in enumerate(zip((inner, outer), seeds, strict=True))
        ]
        terrains = _terrain_records(inner, outer)
        # The field that will draw the boundaries, on the request's own seed.
        # ``inset``/``amplitude``/``feather`` are ``None`` rather than numbers:
        # ``tilemask`` reads absent as "the ratio", which is what keeps the
        # boundary the same *shape* at every tile size, and a door that wrote
        # today's pixels into the row would pin next year's sheets to this
        # year's arithmetic.
        mask = {
            "version": int(tilemask.MASK_VERSION),
            "seed": sheet_seed,
            "inset": None,
            "amplitude": None,
            "feather": None,
        }
        # Two materials share the table, which is the case the ground run
        # actually measured -- so this is 64, and says so through the same
        # function a sixteen-material sheet asks.
        colors = sheet_colors(len(materials))
    else:
        geom = tilesheet.geometry(size, resolved)
        columns, rows, tile_w, tile_h, tiles = (
            geom.columns,
            geom.rows,
            geom.tile_w,
            geom.tile_h,
            geom.tiles,
        )
        colors = SHEET_COLORS

    # -- the pixelisation options, through the shared door ---------------------
    #
    # ``check_pixel_options`` rather than four more refusals here, for the
    # reason that module exists: two paths refusing the same value in two
    # different sentences is the drift it was written to stop. What is actually
    # wanted from it on this path is the palette, which it **loads and throws
    # away** -- a palette file deleted, emptied or corrupted since the form
    # listed it costs the request, rather than N full generations and a sheet
    # that merely came back the wrong colours.
    #
    # Both ladders are this path's own already-validated values rather than the
    # menus they came from, and that is deliberate. A tile sheet has no
    # ``logical_size`` control -- its size is ``tile_size``, refused twenty
    # lines above under that name -- and no colour control at all, since
    # ``colors`` is computed from the cell count. Passing the real ladders would
    # let a refusal name ``logical_size``, a field this form does not draw; a
    # ladder of one already-valid value cannot refuse, which is the honest
    # encoding of "this path checks its own".
    from .pixelopts import check_pixel_options

    pixel_opts = check_pixel_options(
        svc,
        {
            "logical_size": size,
            "colors": colors,
            "palette": palette,
            "dither": dither,
            "outline": outline,
        },
        sizes=(size,),
        size_default=size,
        colors=(colors,),
        colors_default=colors,
        # No outline pass and no reduction mode on this kind. ``False`` drops
        # the key when the request said nothing -- so nothing reaches params
        # that nothing reads -- and **refuses** it when the request named a
        # mode. That refusal used to live here as a second check after this
        # call, which is what an ``outline`` parameter whose only job was to be
        # refused by name was for; the checker does it now, and this door's
        # only remaining job is to say *why*, which is a fact about tiles.
        allow_outline=False,
        allow_reduce_mode=False,
        # The reason, not a restatement of the rule: ``pixelize._edge_mask``
        # pads with ``constant_values=False``, so on a cell that is opaque edge
        # to edge -- which every tile is -- every border pixel has a
        # "transparent" neighbour and ``inner`` returns the outer ring of *each
        # cell*. An outline on a tile sheet is a grid line around all
        # sixty-four tiles, not an outline of anything in them.
        outline_refusal=(
            "a tile sheet has no outline pass: a tile is opaque edge to edge, "
            "so an outline finds the edge of every cell and draws a grid line "
            "around all of them rather than around anything in them. Gutters "
            "between tiles, if they are ever wanted, are a different feature "
            "and would be named as one."
        ),
    )

    # The sheet's identity is the pixel-art LoRA on a base that can take it, so
    # a mismatch is refused here rather than queued -- the worker would drop the
    # style and paint sixty-four photographs of gravel. Written as a guard on a
    # pair of constants deliberately, for ``create_pixel_sheet``'s reason: params
    # outlive today's UI, and the day the base becomes a parameter this refusal
    # is already at the door.
    base = models.BASE_MODELS[TILE_SHEET_BASE_MODEL]
    pixel_lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    if not models.lora_fits(base, pixel_lora):
        fitting = sorted(
            key
            for key, loras in models.loras_by_base().items()
            if models.PIXEL_SHEET_LORA in loras
        )
        raise Invalid(
            f"base_model {base.key!r} is {base.family} and the pixel-art LoRA "
            f"{pixel_lora.key!r} is fitted to {pixel_lora.family}; "
            f"pick one of {fitting}",
            field="base_model",
        )

    params: dict[str, Any] = {
        "seed": sheet_seed,
        "base_model": TILE_SHEET_BASE_MODEL,
        "style_lora": models.PIXEL_SHEET_LORA,
        "colors": colors,
        "negative_prompt": negative,
        # One nested block rather than six loose keys, because it is a
        # *document description* rather than a settings vector: anything
        # reading the sheet back compares it whole against the file on disk,
        # and VECTOR_PARAMS deliberately does not carry it.
        "sheet": {
            # 3: the block says which *shape* it describes. A version-2 block
            # (or one with no version at all) is the grid path and nothing else
            # -- the number is what lets a reader tell "a sheet from before the
            # seamless modes" from "a sheet somebody asked for the grid", which
            # a mode key alone could not, since version 2 has no mode key.
            "version": 3,
            "mode": mode_key,
            "tile_w": tile_w,
            "tile_h": tile_h,
            # The key stays ``projection`` -- see ``tilesheet.sheet_sidecar``.
            "projection": resolved,
            "columns": columns,
            "rows": rows,
            # What the positions mean. ``"blob47"`` is a promise about
            # positions: forty-seven columns in ascending ``tilemask.BLOB_MASKS``
            # order, which is what ``Tileset.local_for`` indexes.
            "layout": layout,
            "materials": materials,
            # Ordered, and the order is precedence -- see ``_terrain_records``.
            "terrains": terrains,
            "mask": mask,
            # The words both terrain materials share. Not part of either
            # description, because it is *context* rather than an instruction:
            # ``tileatlas.terrain_subjects`` appends it to both so two
            # independent samples come back sharing a world and a palette, and
            # anything asking for a seam in it is asking for a drawn edge inside
            # a cell the coverage field then cuts across.
            "boundary": str(boundary or "").strip() if mode_key == MODE_TERRAIN else "",
            "variants": int(variants) if mode_key == MODE_MATERIALS else 1,
            "style_lock": bool(style_lock),
            "seam_erase": bool(seam_erase),
        },
    }
    # Flat, beside ``colors``, and matching Troupe's spelling exactly -- the
    # worker reads ``params["palette"]`` and ``params["dither"]`` by those names
    # on every path that has them. Written only when they were asked for, the
    # ``ip_adapter`` rule two blocks down: a request that named no palette and
    # asked for no dither stores the params blob it stored yesterday, so every
    # stored vector, every reroll comparison and every sheet already on disk is
    # untouched by this feature existing.
    if pixel_opts["palette"]:
        params["palette"] = pixel_opts["palette"]
    if pixel_opts["dither"]:
        params["dither"] = True
    if mode_key == MODE_GRID:
        # Written in grid mode and nowhere else. The guide *is* the grid, and a
        # seamless request that carried this key would charge a ControlNet it
        # never opens against admission and name one in its sidecar.
        params["control"] = "canny"
    if asset_type in {
        "tileset",
        "tileset_top_down", "tileset_three_quarter", "tileset_isometric"
    }:
        params["asset_type"] = "tileset" if asset_type == "tileset" else asset_type
    if asset_intent == "tileset":
        params["asset_intent"] = asset_intent
    if reference is not None:
        # Only alongside the image it scales, so an unused adapter never
        # reaches params as a live setting -- ``settings_2d.submit_kwargs``'
        # rule for ip_scale, applied at the door.
        params["ip_adapter"] = "plus"

    # Presence as well as fit, and at the door: the family check above only asks
    # whether these weights belong together, not whether either is on this host.
    # Missing, the worker's own tolerance takes over -- it logs and paints bare
    # -- so the job would finish, look like one flat picture, and write a
    # sidecar naming a LoRA that never loaded.
    _check_weights(
        svc, mode=mode_key, with_reference=reference is not None, style_lock=bool(style_lock)
    )
    check_vram(svc, "tile_sheet", "tilesheet", params)

    normalized_ref: bytes | None = None
    if reference is not None:
        if len(reference) > MAX_UPLOAD_BYTES:
            raise TooLarge("reference upload is over 20 MB")
        try:
            normalized_ref = to_png(reference)
        except ImageTooLarge as exc:
            raise Invalid(str(exc), field="reference") from exc
        except Exception as exc:
            raise Invalid("could not decode uploaded reference", field="reference") from exc

    # The directory before the row, and the directory removed again if the row
    # write raises: ``next_queued`` can otherwise claim the job in the gap and
    # find no ``ref.png`` on disk. ``_jobs_create``'s invariant, and the reason
    # that module keeps its three doors together.
    new_id = uuid.uuid4().hex[:12]
    job_dir: Path | None = None
    try:
        if normalized_ref is not None:
            job_dir = svc.config.job_dir(new_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "ref.png").write_bytes(normalized_ref)
        svc.store.create("tile_sheet", text, params, new_id, stage="tilesheet")
    except Exception:
        if job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    # Named at creation rather than left to the library's fallback: a tile sheet
    # is not a subject, and a row called "mossy dungeon" among a list of assets
    # reads as a mesh somebody generated.
    # Truncated like every other auto-derived name (_jobs_create's two): the
    # prompt is capped at MAX_PROMPT (1000), far past what a name column holds.
    svc.store.set_meta(new_id, name=f"{text} tile sheet"[:MAX_JOB_NAME])
    svc.wake_worker()
    return {
        "id": new_id,
        "mode": mode_key,
        "tiles": tiles,
        "tile_w": tile_w,
        "tile_h": tile_h,
    }
