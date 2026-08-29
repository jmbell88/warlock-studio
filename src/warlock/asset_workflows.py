"""Pure planning and validation helpers for tilesets.

Generation itself remains in the existing queue.  These helpers make the
structural part deterministic: a model may decorate a cell, but it cannot
change which Wang/path role that cell represents.

**What used to be here and is not any more.** Six functions --
``compatible_edges``, ``validate_edge_pixels``, ``repair_atlas``,
``atlas_warnings``, ``sheet_manifest`` and ``sprite_plan`` -- were written for a
tileset path that never shipped, and every one of them had zero callers in
``src/`` and ``tests/``.  ``validate_edge_pixels`` is the one worth naming,
because it is the one somebody would be tempted to revive: it compares the guard
bands of cells that are adjacent *in an eight-wide atlas*, which is only the
same thing as "adjacent on a map" when the atlas order happens to be the map
order.  In a forty-seven column blob layout it is not -- neighbouring columns
are unrelated coverage cases -- so the check would pass on a wrong set and fail
on a right one.  A generated set is landed by its record
(``pipelines.tileatlas.atlas_sidecar``), not by measuring its pixels.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .generation import cell_dimensions

#: How many distinct *prompt lines* one collection may name, how many draws of
#: each it may ask for, and the ceiling on their product.
#:
#: Named constants rather than literals in the guard below because they are one
#: half of a pair: ``pipelines.tileatlas`` enforces the product (it is all a
#: geometry can still see by the time the lines have become cells) and
#: ``service.tilesheets`` enforces the lines and the variants (it is the last
#: place they still exist as lines).  Both read these, and
#: ``tests/test_tileset_service.py`` pins them against
#: ``tileatlas.MAX_MATERIALS``/``MAX_CELLS`` so three enforcement points cannot
#: come to disagree about one rule.
MAX_COLLECTION_LINES = 16
MAX_COLLECTION_VARIANTS = 4
MAX_COLLECTION_CELLS = 64

#: The mode words this module accepts, mapped onto the two it plans for.
#:
#: ``collection``/``terrain_transition``/``path`` are the stored spellings --
#: rows and profiles carry them and a request naming one is not an error -- and
#: ``materials``/``terrain`` are what ``pipelines.tileatlas`` calls the same two
#: shapes.  ``path`` folds onto ``terrain`` because a path *is* a terrain
#: transition: one surface laid through another, which is exactly the pair the
#: blob field composites.
TILE_MODE_ALIASES: dict[str, str] = {
    "collection": "materials",
    "materials": "materials",
    "terrain_transition": "terrain",
    "terrain": "terrain",
    "path": "terrain",
}


@dataclass(frozen=True, slots=True)
class TileRole:
    index: int
    mask: int
    name: str
    edges: tuple[str, str, str, str]


def tile_plan(
    *,
    mode: str,
    view: str = "top_down",
    prompt_items: Iterable[str] = (),
    variants: int = 1,
    inner_terrain: str = "",
    outer_terrain: str = "",
    boundary: str = "",
    ground: str = "",
    path: str = "",
    edge: str = "",
    seed: int = 0,
    target_cell_px: int | None = None,
) -> dict[str, Any]:
    """Compile a structural tileset request before any model call.

    ``mode`` is read through :data:`TILE_MODE_ALIASES`, so both vocabularies
    work: a stored row saying ``collection`` and a new request saying
    ``materials`` compile to the same plan.

    The role table below is the sixteen-corner Wang set, which is what the
    older grid path lays out.  It is **not** the blob-47 layout the seamless
    terrain path uses -- that one is ``pipelines.tilemask``'s, is forty-seven
    cases rather than sixteen, and is described by
    ``pipelines.tileatlas.atlas_sidecar`` rather than by anything here.
    """
    isometric = view == "isometric"
    working = (256, 128) if isometric else (256, 256)
    target = cell_dimensions(working, target_cell_px, isometric=isometric)
    resolved = TILE_MODE_ALIASES.get(str(mode), "")
    if resolved == "materials":
        cells = collection_cells(prompt_items, variants, seed=seed)
        roles = ()
    elif resolved == "terrain" and mode == "path":
        if not ground.strip() or not path.strip():
            raise ValueError("path sets need ground and path descriptions")
        cells = tuple({"index": r.index, "role": r.name} for r in path_roles())
        roles = path_roles()
    elif resolved == "terrain":
        if not inner_terrain.strip() or not outer_terrain.strip():
            raise ValueError("terrain sets need inner and outer terrain descriptions")
        cells = tuple({"index": r.index, "role": r.name} for r in wang_roles())
        roles = wang_roles()
    else:
        raise ValueError(
            f"unknown tileset mode {mode!r}; this plans "
            f"{', '.join(sorted(set(TILE_MODE_ALIASES)))}"
        )
    return {
        # 2: cells carry a seed, and ``materials``/``terrain`` read as mode
        # words.  Bumped because the worker consumes this block now rather than
        # merely recording it, so "which shape are these cells" is a question
        # something asks rather than a comment.
        "version": 2,
        "mode": mode,
        "resolved_mode": resolved,
        "view": view,
        "working_cell_px": list(working),
        "target_cell_px": target_cell_px,
        "output_cell_px": list(target),
        "cells": cells,
        "roles": [
            {"index": r.index, "mask": r.mask, "name": r.name, "edges": list(r.edges)}
            for r in roles
        ],
        "descriptions": {
            "inner": inner_terrain,
            "outer": outer_terrain,
            "boundary": boundary,
            "ground": ground,
            "path": path,
            "edge": edge,
        },
    }


def collection_cells(
    prompt_items: Iterable[str], variants: int = 1, *, seed: int = 0
) -> tuple[dict[str, Any], ...]:
    """N prompt lines by V draws, expanded into cells in reading order.

    Each cell carries the seed its generation runs on, derived from the one
    request seed by :func:`pipelines.tileatlas.material_seeds` rather than drawn
    here.  Derived, so re-running the request reproduces every cell and cell
    ``i`` is reproducible on its own from the pair ``(seed, i)``; and derived
    *there* rather than restated here, because two implementations of "cell
    ``i``'s seed" is how a reroll of one material comes back as a different
    picture from the one it is rerolling.
    """
    items = tuple(str(x).strip() for x in prompt_items if str(x).strip())
    count = int(variants)
    if not 1 <= len(items) <= MAX_COLLECTION_LINES:
        raise ValueError(f"a tile collection needs 1-{MAX_COLLECTION_LINES} prompt lines")
    if not 1 <= count <= MAX_COLLECTION_VARIANTS:
        raise ValueError(
            f"collection variants must be between 1 and {MAX_COLLECTION_VARIANTS}"
        )
    if len(items) * count > MAX_COLLECTION_CELLS:
        raise ValueError(f"a tile collection is capped at {MAX_COLLECTION_CELLS} cells")
    # Imported here rather than at module scope: this module is imported by the
    # service layer's door and by the worker, and only this one function needs
    # the pipeline.
    from .pipelines.tileatlas import material_seeds

    pairs = [(prompt, variant) for prompt in items for variant in range(count)]
    seeds = material_seeds(seed, len(pairs))
    return tuple(
        {"index": i, "prompt": prompt, "variant": variant + 1, "seed": int(cell_seed)}
        for i, ((prompt, variant), cell_seed) in enumerate(zip(pairs, seeds, strict=True))
    )


def wang_roles() -> tuple[TileRole, ...]:
    """The exact 16 corner cases, in stable binary-mask order."""
    names = ("none", "north", "east", "south", "west")
    out = []
    for mask in range(16):
        edges = tuple("inner" if mask & (1 << bit) else "outer" for bit in range(4))
        out.append(TileRole(mask, mask, names[0] if mask == 0 else f"wang_{mask:02d}", edges))
    return tuple(out)


def path_roles() -> tuple[TileRole, ...]:
    """The 18 canonical connectable path cases.

    Four-way masks cover the 16 edge combinations; the two extra roles are
    explicit end caps used by the existing path layout.
    """
    roles = list(wang_roles())
    roles.extend(
        (
            TileRole(16, 16, "cap_horizontal", ("path", "ground", "path", "ground")),
            TileRole(17, 17, "cap_vertical", ("ground", "path", "ground", "path")),
        )
    )
    return tuple(roles)
