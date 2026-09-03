"""The subset half of a re-render: which cells, and how they get composited.

Pure. No worker, no Blender, no job store -- ``charsheet`` is filesystem-free by
design and ``sheet.pack``/``compose_cells`` take paths and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import charsheet
from warlock.pipelines import sheet as sheetlib


def _runs(n=1):
    return [
        {"animation": animation, "direction": direction}
        for animation, direction, *_ in charsheet.spans()[:n]
    ]


# -- which runs -----------------------------------------------------------


def test_a_subset_normalises_to_animation_and_direction_pairs():
    wanted = charsheet.check_subset(_runs(3))
    assert len(wanted) == 3
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in wanted)


@pytest.mark.parametrize(
    ("subset", "says"),
    [
        (None, "at least one"),
        ([], "at least one"),
        ([{"animation": "walk"}], "needs an animation and a direction"),
        ([{"animation": "", "direction": "front"}], "needs an animation and a direction"),
        ([{"animation": "moonwalk", "direction": "front"}], "not an animation"),
        ([{"animation": "walk", "direction": "upwards"}], "no walk facing upwards"),
    ],
)
def test_a_subset_is_refused_by_name(subset, says):
    """``check_frame_counts``' rule one function up: a request naming something
    the sheet does not carry is a mistake about the sheet, and rendering the
    rest of it would leave the user hunting a change that never happened."""
    with pytest.raises(ValueError, match=says):
        charsheet.check_subset(subset)


def test_the_same_run_twice_is_refused():
    run = _runs(1)
    with pytest.raises(ValueError, match="named twice"):
        charsheet.check_subset(run + run)


def test_every_run_at_once_is_refused_as_a_full_render():
    """Not pedantry: it is a full render taking the slower path, paying for a
    copy step and a pinned palette it does not need. The ordinary door exists."""
    everything = [
        {"animation": animation, "direction": direction}
        for animation, direction, *_ in charsheet.spans()
    ]
    with pytest.raises(ValueError, match="every run"):
        charsheet.check_subset(everything)


# -- which cells ----------------------------------------------------------


def test_the_indices_are_exactly_the_runs_spans():
    """Expanded from ``spans`` rather than from arithmetic of its own, so this
    and ``sheetscope.runs`` come from the two copies the geometry-agreement
    test owns rather than from a third nothing owns."""
    animation, direction, start, end, _loop = charsheet.spans()[2]
    indices = charsheet.subset_indices([{"animation": animation, "direction": direction}])
    assert indices == tuple(range(start, end + 1))


def test_the_indices_of_several_runs_are_sorted_and_disjoint():
    indices = charsheet.subset_indices(_runs(4))
    assert list(indices) == sorted(indices)
    assert len(set(indices)) == len(indices)


def test_a_subset_never_names_a_cell_the_full_table_does_not_have():
    every = {cell.index for cell in charsheet.frame_table()}
    assert set(charsheet.subset_indices(_runs(5))) <= every


# -- the geometry claim the whole feature rests on ------------------------


def _plan(**kw):
    records = {
        movement.name: [{"id": movement.name, "frame": i} for i in range(movement.frames)]
        for movement in charsheet.resolve_layout().movements
    }
    return charsheet.plan(records, frame_size=16, **kw)


def test_a_subset_changes_no_cell_geometry_at_all():
    """**The claim the merge rests on.** A re-rendered cell has to land on the
    rectangle it has always had, or the composite is a smear. The plan is built
    unfiltered and the *spec list* is what gets filtered, so this is a property
    of the design rather than something to be careful about."""
    plan = _plan()
    geometry = [
        (cell.index, cell.row, cell.column, cell.x, cell.y) for cell in plan.cells
    ]
    wanted = set(charsheet.subset_indices(_runs(2)))
    subset = [cell for cell in plan.cells if cell.index in wanted]

    for cell in subset:
        assert (cell.index, cell.row, cell.column, cell.x, cell.y) in geometry


# -- packing a subset ------------------------------------------------------


def _write_cells(tmp_path, plan, indices, value):
    out = {}
    for cell in plan.cells:
        if cell.index not in indices:
            continue
        pixels = np.zeros((plan.cell_h, plan.cell_w, 4), dtype=np.uint8)
        pixels[..., 0] = value
        pixels[..., 3] = 255
        path = tmp_path / f"{cell.index:04d}.png"
        Image.fromarray(pixels, "RGBA").save(path)
        out[cell.index] = path
    return out


def test_packing_a_subset_leaves_every_other_cell_transparent(tmp_path):
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    frames = _write_cells(tmp_path, plan, wanted, 200)

    out = tmp_path / "subset.png"
    trims = sheetlib.pack(plan, frames, out, only=wanted)

    assert set(trims) == wanted, "a cell outside the subset is not measured"
    with Image.open(out) as opened:
        atlas = np.asarray(opened.convert("RGBA"))
    assert atlas.shape[:2] == (plan.height, plan.width)

    outside = next(cell for cell in plan.cells if cell.index not in wanted)
    patch = atlas[outside.y : outside.y + plan.cell_h, outside.x : outside.x + plan.cell_w]
    assert patch[..., 3].max() == 0


def test_a_named_cell_that_did_not_render_is_still_refused(tmp_path):
    """The filter must not become a relaxation. A dropped render frame and a
    subset have to stay distinguishable in the one function positioned to
    notice."""
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    frames = _write_cells(tmp_path, plan, wanted, 200)
    frames.pop(sorted(wanted)[0])

    with pytest.raises(ValueError, match="no rendered frame"):
        sheetlib.pack(plan, frames, tmp_path / "out.png", only=wanted)


# -- composing -------------------------------------------------------------


def _atlas(plan, value, path):
    pixels = np.zeros((plan.height, plan.width, 4), dtype=np.uint8)
    pixels[..., 1] = value
    pixels[..., 3] = 255
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def test_composing_replaces_only_the_named_cells(tmp_path):
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    base = _atlas(plan, 50, tmp_path / "base.png")
    overlay = _atlas(plan, 250, tmp_path / "overlay.png")

    out = tmp_path / "merged.png"
    sheetlib.compose_cells(base, overlay, plan, wanted, out)

    with Image.open(out) as opened:
        merged = np.asarray(opened.convert("RGBA"))

    inside = next(cell for cell in plan.cells if cell.index in wanted)
    outside = next(cell for cell in plan.cells if cell.index not in wanted)
    assert merged[inside.y, inside.x, 1] == 250
    assert merged[outside.y, outside.x, 1] == 50


def test_composing_twice_is_byte_identical(tmp_path):
    """**The test the outline finding is about.** ``outline`` in the shipped
    ``outer`` mode grows a silhouette by a pixel on every side, so a design that
    composed first and quantised after would fatten every copied cell once per
    re-render -- and the sheet would go a pixel thinner in the runs nobody
    touched, with nothing to say why. Composing is a paste and nothing else."""
    plan = _plan()
    wanted = set(charsheet.subset_indices(_runs(1)))
    base = _atlas(plan, 50, tmp_path / "base.png")
    overlay = _atlas(plan, 250, tmp_path / "overlay.png")

    once = tmp_path / "once.png"
    twice = tmp_path / "twice.png"
    sheetlib.compose_cells(base, overlay, plan, wanted, once)
    sheetlib.compose_cells(once, overlay, plan, wanted, twice)

    assert once.read_bytes() == twice.read_bytes()


def test_a_base_of_the_wrong_size_is_refused_by_name(tmp_path):
    plan = _plan()
    overlay = _atlas(plan, 250, tmp_path / "overlay.png")
    wrong = tmp_path / "wrong.png"
    Image.fromarray(
        np.zeros((plan.height // 2, plan.width, 4), dtype=np.uint8), "RGBA"
    ).save(wrong)

    with pytest.raises(ValueError, match="re-rendered is"):
        sheetlib.compose_cells(wrong, overlay, plan, {0}, tmp_path / "out.png")


def test_a_replaced_cell_does_not_show_the_old_silhouette_through(tmp_path):
    """Pasted, not alpha-composited. Blending would leave the previous pose
    showing wherever the new one is transparent, which is a ghost."""
    plan = _plan()
    base = _atlas(plan, 50, tmp_path / "base.png")
    clear = np.zeros((plan.height, plan.width, 4), dtype=np.uint8)
    overlay = tmp_path / "overlay.png"
    Image.fromarray(clear, "RGBA").save(overlay)

    out = tmp_path / "out.png"
    sheetlib.compose_cells(base, overlay, plan, {0}, out)

    with Image.open(out) as opened:
        merged = np.asarray(opened.convert("RGBA"))
    cell = plan.cells[0]
    patch = merged[cell.y : cell.y + plan.cell_h, cell.x : cell.x + plan.cell_w]
    assert patch[..., 3].max() == 0
