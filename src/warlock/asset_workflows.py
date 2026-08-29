"""Pure planning helpers for tilesets.

Generation itself remains in the existing queue.  What is left here is the one
expansion something actually generates from: :func:`collection_cells`, which
turns N material lines by V draws into the cells ``service.tilesheets`` queues
and ``_q_tileset`` then samples one by one.

**The grid planner is gone (2026-08-29).**  ``tile_plan`` compiled a structural
description -- per-cell prompts for a materials request, or the 16/18 Wang and
path roles for a terrain one -- and its only caller was the *legacy grid* path
in ``_q_tilesheet``, which paints one 1024px frame through a canny guide and
slices it on a fixed lattice.  That image has no per-cell prompt and no role in
it, so the plan changed nothing about the generation and was merely written into
``params["tile_plan"]`` and into the sheet's sidecar as a ``workflow`` block: a
record of a structure the picture does not have, which is the defect the whole
tileset programme exists to have fixed.  The seamless modes honour their plan
for real and it is ``service.tilesheets`` that compiles theirs, into the stored
``sheet`` block, with ``pipelines.tileatlas.atlas_sidecar`` as its record.
``wang_roles``, ``path_roles`` and the ``TileRole`` dataclass went with it,
having had no other caller; ``pipelines.tilemask``'s blob-47 field is what the
shipped terrain path is laid out on, and it is forty-seven coverage cases rather
than these sixteen corners.

**What used to be here and is not any more.** Six further functions --
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
from typing import Any

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

#: The mode words a tile request may carry, mapped onto the two shapes that get
#: built.  Read by ``service.jobs``, which is what turns a request document's
#: mode into ``service.tilesheets``' own.
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
