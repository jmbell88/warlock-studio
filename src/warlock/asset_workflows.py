"""Pure planning and validation helpers for tilesets and sprite sheets.

Generation itself remains in the existing queue.  These helpers make the
structural part deterministic: a model may decorate a cell, but it cannot
change which Wang/path role that cell represents or make an untouched repair
rewrite neighboring pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .generation import (
    SPRITE_ACTIONS,
    SPRITE_FRAME_COUNTS,
    TARGET_CELL_MAX,
    TARGET_CELL_MIN,
    cell_dimensions,
)


@dataclass(frozen=True, slots=True)
class TileRole:
    index: int
    mask: int
    name: str
    edges: tuple[str, str, str, str]


def tile_plan(*, mode: str, view: str = "top_down", prompt_items: Iterable[str] = (), variants: int = 1, inner_terrain: str = "", outer_terrain: str = "", boundary: str = "", ground: str = "", path: str = "", edge: str = "", target_cell_px: int | None = None) -> dict[str, Any]:
    """Compile a structural tileset request before any model call."""
    isometric = view == "isometric"
    working = (256, 128) if isometric else (256, 256)
    target = cell_dimensions(working, target_cell_px, isometric=isometric)
    if mode == "collection":
        cells = collection_cells(prompt_items, variants)
        roles = ()
    elif mode == "terrain_transition":
        if not inner_terrain.strip() or not outer_terrain.strip():
            raise ValueError("terrain transitions need inner and outer terrain descriptions")
        cells = tuple({"index": r.index, "role": r.name} for r in wang_roles())
        roles = wang_roles()
    elif mode == "path":
        if not ground.strip() or not path.strip():
            raise ValueError("path sets need ground and path descriptions")
        cells = tuple({"index": r.index, "role": r.name} for r in path_roles())
        roles = path_roles()
    else:
        raise ValueError("unknown tileset mode")
    return {"version": 1, "mode": mode, "view": view, "working_cell_px": list(working), "target_cell_px": target_cell_px, "output_cell_px": list(target), "cells": cells, "roles": [{"index": r.index, "mask": r.mask, "name": r.name, "edges": list(r.edges)} for r in roles], "descriptions": {"inner": inner_terrain, "outer": outer_terrain, "boundary": boundary, "ground": ground, "path": path, "edge": edge}}


def collection_cells(prompt_items: Iterable[str], variants: int = 1) -> tuple[dict[str, Any], ...]:
    items = tuple(str(x).strip() for x in prompt_items if str(x).strip())
    if not 1 <= len(items) <= 16:
        raise ValueError("a tile collection needs 1–16 prompt lines")
    if not 1 <= int(variants) <= 4:
        raise ValueError("collection variants must be between 1 and 4")
    if len(items) * int(variants) > 64:
        raise ValueError("a tile collection is capped at 64 cells")
    return tuple({"index": i, "prompt": prompt, "variant": v + 1} for i, (prompt, v) in enumerate((
        (prompt, variant) for prompt in items for variant in range(int(variants))
    )))


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
    roles.extend((TileRole(16, 16, "cap_horizontal", ("path", "ground", "path", "ground")), TileRole(17, 17, "cap_vertical", ("ground", "path", "ground", "path"))))
    return tuple(roles)


def compatible_edges(roles: Iterable[TileRole]) -> tuple[tuple[int, int, str], ...]:
    """Return adjacent role pairs whose shared edge must be pixel-identical."""
    rows = tuple(roles)
    result = []
    for left in rows:
        for right in rows:
            if left is right:
                continue
            if left.edges[1] == right.edges[3]:
                result.append((left.index, right.index, "vertical"))
            if left.edges[2] == right.edges[0]:
                result.append((left.index, right.index, "horizontal"))
    return tuple(result)


def validate_edge_pixels(atlas: Any, roles: Iterable[TileRole], cell_w: int, cell_h: int) -> list[str]:
    """Check matching guard-band pixels in an atlas-shaped array."""
    import numpy as np

    pixels = np.asarray(atlas)
    if pixels.ndim < 3:
        raise ValueError("atlas must be (height, width, channels)")
    rows = tuple(roles)
    errors: list[str] = []
    # Role order is the deterministic row-major layout. Compare the right and
    # bottom guard bands of every adjacent pair; never compare unrelated cells.
    for index, left in enumerate(rows):
        col = index % 8
        row = index // 8
        if col < 7 and index + 1 < len(rows):
            right = index + 1
            a = pixels[row * cell_h : (row + 1) * cell_h, (col + 1) * cell_w - 1]
            b = pixels[(right // 8) * cell_h : (right // 8 + 1) * cell_h, (right % 8) * cell_w]
            if not np.array_equal(a, b):
                errors.append(f"edge mismatch between cells {index} and {right}")
        if row < 7 and index + 8 < len(rows):
            down = index + 8
            a = pixels[(row + 1) * cell_h - 1, col * cell_w : (col + 1) * cell_w]
            b = pixels[(down // 8) * cell_h, (down % 8) * cell_w : (down % 8 + 1) * cell_w]
            if not np.array_equal(a, b):
                errors.append(f"edge mismatch between cells {index} and {down}")
    return errors


def sheet_manifest(*, generation_type: str, mode: str, working_cell: tuple[int, int], target_cell_px: int | None, reduction: str | None, palette: Any, seed: int, roles: Iterable[TileRole] = ()) -> dict[str, Any]:
    return {
        "version": 1,
        "generation_type": generation_type,
        "mode": mode,
        "working_cell_px": list(working_cell),
        "target_cell_px": target_cell_px,
        "reduction": reduction if target_cell_px is not None else None,
        "palette": palette,
        "source_seed": int(seed),
        "roles": [{"index": r.index, "mask": r.mask, "name": r.name, "edges": list(r.edges)} for r in roles],
    }


def sprite_plan(*, mode: str = "turnaround", action: str = "idle", directions: int = 4, candidates: int = 2, target_cell_px: int | None = None) -> dict[str, Any]:
    if mode not in ("turnaround", "action"):
        raise ValueError("sprite mode must be turnaround or action")
    if mode == "action" and action not in SPRITE_ACTIONS:
        raise ValueError(f"unknown sprite action {action!r}")
    if directions not in (4, 8):
        raise ValueError("sprite sheets support 4 or 8 directions")
    if candidates not in (1, 2):
        raise ValueError("sprite sheets support one or two candidates")
    frame_count = 4 if mode == "turnaround" else SPRITE_FRAME_COUNTS[action]
    if target_cell_px is not None and not TARGET_CELL_MIN <= int(target_cell_px) <= TARGET_CELL_MAX:
        raise ValueError(f"target cell must be between {TARGET_CELL_MIN} and {TARGET_CELL_MAX}")
    return {"version": 1, "mode": mode, "action": action, "directions": directions, "frame_count": frame_count, "candidate_count": candidates, "target_cell_px": target_cell_px, "working_cell_px": 512 if mode == "turnaround" else 256}


def repair_atlas(atlas: Any, replacements: Mapping[int, Any], *, cell_w: int, cell_h: int) -> Any:
    """Replace selected cells while preserving all untouched bytes."""
    import numpy as np

    result = np.array(atlas, copy=True)
    for index, replacement in replacements.items():
        row, col = divmod(int(index), 8)
        value = np.asarray(replacement)
        if value.shape != (cell_h, cell_w, *result.shape[2:]):
            raise ValueError(f"replacement for cell {index} has shape {value.shape}, expected {(cell_h, cell_w, *result.shape[2:])}")
        result[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = value
    return result


def atlas_warnings(atlas: Any, *, cell_w: int, cell_h: int, expected_cells: int, palette: Iterable[Any] | None = None) -> list[str]:
    """Inspectable QA warnings for generated sheets."""
    import numpy as np

    pixels = np.asarray(atlas)
    warnings: list[str] = []
    if pixels.ndim < 3:
        return ["atlas does not have pixel channels"]
    rows, cols = divmod(expected_cells - 1, 8)
    if pixels.shape[0] < (rows + 1) * cell_h or pixels.shape[1] < (cols + 1) * cell_w:
        warnings.append("atlas is clipped or smaller than its declared cell layout")
    for index in range(expected_cells):
        row, col = divmod(index, 8)
        cell = pixels[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
        if cell.size == 0 or not np.any(cell[..., -1] if cell.shape[-1] == 4 else np.any(cell != 0, axis=-1)):
            warnings.append(f"cell {index} is empty")
    if palette is not None:
        allowed = {tuple(int(v) for v in colour) for colour in palette}
        colours = np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0)
        if any(tuple(int(v) for v in colour) not in allowed for colour in colours):
            warnings.append("atlas contains colours outside the shared palette")
    return warnings
