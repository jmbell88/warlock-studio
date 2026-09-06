"""Structural validation of a rendered sheet: is anything clipped, blank or missing?

Pure arithmetic over a ``sheet.Plan``, a trims map and a sidecar dict. No
filesystem, no Blender, no PIL, no ``service`` -- the same bargain ``sheet.py``
and ``charsheet.py`` make, and for the same reason: this is the half of the
answer that has to be reachable from the worker, from the queue stage and from
a test with none of those installed.

**Nothing here fails a job.** A sheet that reports findings is still on disk
and still openable; the block exists so the UI can say "the attack run is
clipped at the top" instead of the user discovering it in their engine. The one
finding that *acts* is ``clipped_cells`` on the first render, which
``_q_troupe`` answers with a single wider-margin retry.

The trims map is ``{cell index: {"x", "y", "w", "h"}}`` -- exactly what
``sheet.measure_trim`` returns per cell, cell-local, with ``None`` for a frame
whose alpha channel was empty. A ``None`` is **blank, not clipped**: nothing
reached the edge because nothing was drawn at all, and calling that a clip
would send the reframe retry after a sheet with no subject in it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "VALIDATION_VERSION",
    "blank_cells",
    "clipped_cells",
    "describe",
    "metadata_findings",
    "missing_frames",
    "validate",
]

#: Bumped when the block's *shape* changes, never when a check is added: a
#: reader that skips unknown finding kinds keeps working, and one that cannot
#: parse the block at all needs to know.
VALIDATION_VERSION = 1


def _cell_size(plan: Any) -> tuple[int, int]:
    """The rectangle one cell occupies, non-square plans included."""
    return (
        int(getattr(plan, "cell_w", plan.frame_size)),
        int(getattr(plan, "cell_h", plan.frame_size)),
    )


def clipped_cells(plan: Any, trims: Mapping[int, Mapping[str, int] | None]) -> list[int]:
    """Cells whose silhouette touches a cell edge -- i.e. was cut off.

    The trim is the alpha bounding box *within* the cell, so touching an edge
    means the subject continued past the frame the camera drew. That is the
    measurable form of "the jump apex is clipped": a pose that leaves the rest
    bounding box renders with its top row of pixels flush against ``y == 0``.

    A missing or ``None`` trim is not a clip -- see the module docstring.
    """
    out = []
    for cell in plan.cells:
        trim = trims.get(cell.index)
        if not trim:
            continue
        width, height = _cell_size(plan)
        x, y = int(trim["x"]), int(trim["y"])
        if x <= 0 or y <= 0 or x + int(trim["w"]) >= width or y + int(trim["h"]) >= height:
            out.append(cell.index)
    return out


def blank_cells(plan: Any, trims: Mapping[int, Mapping[str, int] | None]) -> list[int]:
    """Cells that rendered nothing: no trim, or one with no area.

    Distinct from :func:`missing_frames`, which is about the *map* and not the
    pixels: a blank cell was measured and had nothing in it, which is a camera
    or pose problem, where a missing one was never measured at all, which is a
    plumbing problem. Reported separately because they send a reader to
    different places.
    """
    out = []
    for cell in plan.cells:
        if cell.index not in trims:
            continue
        trim = trims.get(cell.index)
        if not trim or int(trim["w"]) <= 0 or int(trim["h"]) <= 0:
            out.append(cell.index)
    return out


def missing_frames(plan: Any, trims: Mapping[int, Mapping[str, int] | None]) -> list[int]:
    """Indices the plan contains that the trims map has no entry for at all."""
    return [cell.index for cell in plan.cells if cell.index not in trims]


def metadata_findings(meta: Mapping[str, Any] | None) -> list[str]:
    """What is wrong with the sidecar, as sentences. Empty means nothing is.

    The sidecar is the half of a sheet an engine actually reads, and every
    check here is one an importer would otherwise hit as a silent wrong answer:
    a tag range that skips a frame plays a stutter, a duplicated one plays the
    same frame twice, a pivot outside its cell puts the sprite's feet somewhere
    else, and a missing ``image`` is a sidecar pointing at nothing.
    """
    if not meta:
        return ["the sheet has no sidecar"]
    out: list[str] = []
    cells = list(meta.get("cells") or ())
    count = len(cells)

    if not str(meta.get("image") or ""):
        out.append("the sidecar does not name its atlas")

    troupe = meta.get("troupe")
    if isinstance(troupe, Mapping) and "cell_count" in troupe:
        declared = int(troupe.get("cell_count") or 0)
        if declared != count:
            out.append(
                f"the layout declares {declared} cells and the sidecar lists {count}"
            )

    animation = meta.get("animation")
    if isinstance(animation, Mapping):
        tags = list(animation.get("tags") or ())
        covered: dict[int, int] = {}
        for tag in tags:
            start, end = int(tag.get("start", 0)), int(tag.get("end", -1))
            if end < start:
                out.append(f"tag {tag.get('name')!r} ends before it starts")
                continue
            # Dense by construction -- a run is a contiguous span of cells --
            # so a gap inside one is a frame no tag names and no engine plays.
            for index in range(start, end + 1):
                covered[index] = covered.get(index, 0) + 1
        if tags:
            duplicated = sorted(i for i, n in covered.items() if n > 1)
            uncovered = sorted(set(range(count)) - set(covered))
            outside = sorted(i for i in covered if i >= count or i < 0)
            if duplicated:
                out.append(f"{len(duplicated)} cells are named by more than one tag")
            if uncovered:
                out.append(f"{len(uncovered)} cells are named by no tag")
            if outside:
                out.append(f"{len(outside)} tagged frames are not cells of this sheet")

    for cell in cells:
        px, py = float(cell.get("pivot_x", 0.0)), float(cell.get("pivot_y", 0.0))
        w, h = float(cell.get("w", 0.0)), float(cell.get("h", 0.0))
        # Cell-relative, which is what the format documents: ``sheet.sidecar``
        # defaults the field to ``(cell_w / 2, cell_h)``. A pivot recorded in
        # *atlas* pixels lands far outside a 32px cell, which is exactly the
        # defect ``charsheet.point_in_cell`` exists to prevent.
        if not (0.0 <= px <= w and 0.0 <= py <= h):
            out.append(f"cell {cell.get('index')}'s pivot is outside the cell")
            break
    return out


def validate(
    plan: Any,
    trims: Mapping[int, Mapping[str, int] | None],
    meta: Mapping[str, Any] | None = None,
    *,
    reframed: bool = False,
) -> dict[str, Any]:
    """The whole structural verdict, as the block that goes in the sidecar.

    ``ok`` is the conjunction and nothing more: it is advisory, and a sheet
    with ``ok: False`` has still been written, packed and published.
    """
    clipped = clipped_cells(plan, trims)
    blank = blank_cells(plan, trims)
    missing = missing_frames(plan, trims)
    metadata = metadata_findings(meta) if meta is not None else []
    return {
        "version": VALIDATION_VERSION,
        "ok": not (clipped or blank or missing or metadata),
        "clipped": clipped,
        "blank": blank,
        "missing": missing,
        "metadata": metadata,
        # Whether the render that produced these trims was the wider-margin
        # second attempt. Recorded rather than inferred: a reframed sheet that
        # still clips is a different story from one that never needed it.
        "reframed": bool(reframed),
    }


def describe(validation: Mapping[str, Any] | None) -> list[str]:
    """The findings as sentences a pane can list. Empty when the sheet is clean.

    Counts and a couple of example cells rather than 256 indices: the reader
    wants to know whether to re-render, and a wall of numbers does not help
    them decide.
    """
    if not validation:
        return []
    out: list[str] = []
    for key, phrase in (
        ("clipped", "clipped at the frame edge"),
        ("blank", "empty"),
        ("missing", "never rendered"),
    ):
        indices = list(validation.get(key) or ())
        if not indices:
            continue
        shown = ", ".join(str(i) for i in indices[:4])
        more = "" if len(indices) <= 4 else f" and {len(indices) - 4} more"
        out.append(f"{len(indices)} cells are {phrase}: {shown}{more}")
    out.extend(str(m) for m in (validation.get("metadata") or ()))
    if validation.get("reframed"):
        out.append("the sheet was re-rendered at a wider margin to fit its poses")
    return out
