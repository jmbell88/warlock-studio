"""An Inker animation exported as a sprite sheet.

The grid arithmetic is the interesting half and it is pure, so all of it is
asserted from plain arrays with no ``Document`` and no GPU. The one thing worth
saying out loud about the format: a linked cel appearing in three frames is
three identical cells here, because a sheet is played back by an engine that
knows nothing about links.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from warlock.pipelines import sheet as sheetlib
from warlock.studio import inker
from warlock.studio.inker import sheetout
from warlock.studio.inker.animation import Tag
from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _frame(width: int, height: int, mark: tuple[int, int, int, int], at=(0, 0)) -> np.ndarray:
    plane = np.zeros((height, width, 4), dtype=np.uint8)
    plane[at[1], at[0]] = mark
    return plane


# --- the grid ----------------------------------------------------------------


def test_one_cell_per_frame_in_row_major_order():
    plan = sheetout.plan_frames(3, 40, 20)

    assert plan.columns == 3 and plan.rows == 1
    assert [(c.index, c.x, c.y, c.frame, c.yaw) for c in plan.cells] == [
        (0, 0, 0, 0, 0.0),
        (1, 40, 0, 1, 0.0),
        (2, 80, 0, 2, 0.0),
    ]
    assert (plan.width, plan.height) == (120, 20)


def test_a_non_square_plan_reports_its_real_cell_size():
    plan = sheetout.plan_frames(2, 40, 20)
    assert (plan.cell_w, plan.cell_h) == (40, 20)
    # Zero, not the width: a square-only importer must fail loudly rather than
    # slice the atlas correctly across and wrongly down.
    assert plan.frame_size == 0


def test_a_row_wraps_before_the_atlas_ceiling():
    """32 frames of a 320px canvas is 10240px across, past what an engine will
    accept as a texture at all -- so wrapping is required, not a nicety."""
    plan = sheetout.plan_frames(32, 320, 100)

    assert plan.columns == sheetlib.MAX_ATLAS_PX // 320
    assert plan.rows == -(-32 // plan.columns)
    assert max(plan.width, plan.height) <= sheetlib.MAX_ATLAS_PX
    assert [c.index for c in plan.cells] == list(range(32))


def test_a_wrapped_grid_positions_its_second_row():
    plan = sheetout.plan_frames(3, 4000, 10)
    assert plan.columns == 2
    assert [(c.row, c.column, c.x, c.y) for c in plan.cells] == [
        (0, 0, 0, 0),
        (0, 1, 4000, 0),
        (1, 0, 0, 10),
    ]


def test_a_frame_too_big_for_any_atlas_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        sheetout.plan_frames(1, sheetlib.MAX_ATLAS_PX + 1, 10)


def test_too_many_rows_is_refused_by_the_shared_guard():
    with pytest.raises(ValueError, match="8192"):
        sheetout.plan_frames(2000, sheetlib.MAX_ATLAS_PX, 100)


def test_an_empty_clip_is_refused():
    with pytest.raises(ValueError):
        sheetout.plan_frames(0, 10, 10)


# --- compositing -------------------------------------------------------------


def test_frame_i_lands_in_cell_i():
    frames = [_frame(4, 4, RED, (0, 0)), _frame(4, 4, BLUE, (1, 1))]
    image, plan, _trims, frame_cells = sheetout.build(frames, [100, 100])
    try:
        assert image.size == (plan.width, plan.height)
        assert image.getpixel((0, 0)) == RED
        assert image.getpixel((5, 1)) == BLUE
    finally:
        image.close()
    # Neither merge nor skip_empty was asked for, so there is nothing for a
    # caller to remap -- the frame/cell identity ``animation_block`` already
    # assumes by default.
    assert frame_cells is None


def test_every_frame_must_be_the_same_size():
    with pytest.raises(ValueError, match="same size"):
        sheetout.build([_frame(4, 4, RED), _frame(8, 8, RED)], [10, 10])


def test_every_frame_needs_a_duration():
    with pytest.raises(ValueError, match="duration"):
        sheetout.build([_frame(4, 4, RED)], [])


def test_trims_are_measured_per_frame():
    frames = [_frame(4, 4, RED, (2, 3)), np.zeros((4, 4, 4), dtype=np.uint8)]
    _image, _plan, trims, _frame_cells = sheetout.build(frames, [10, 10])
    assert trims[0] == {"x": 2, "y": 3, "w": 1, "h": 1}
    assert trims[1] is None


# --- merge duplicates + skip empty frames -------------------------------------
#
# Both default off, and both are decided before ``plan_frames`` ever runs: the
# grid is packed from however many cells *survive* merging/skipping, not from
# the frame count the document has -- which is also why arrange interacts with
# them (a reduced cell count is what arrange's math packs).


def test_duplicate_frames_share_a_cell_only_when_merge_is_on():
    frames = [_frame(4, 4, RED, (0, 0)), _frame(4, 4, RED, (0, 0)), _frame(4, 4, BLUE, (1, 1))]

    image, plan, _trims, frame_cells = sheetout.build(frames, [10, 10, 10])
    image.close()
    assert len(plan.cells) == 3
    assert frame_cells is None

    image, plan, _trims, frame_cells = sheetout.build(frames, [10, 10, 10], merge=True)
    image.close()
    assert len(plan.cells) == 2
    assert frame_cells == [0, 0, 1]


def test_empty_frames_get_no_cell_only_when_skip_empty_is_on():
    frames = [_frame(4, 4, RED, (0, 0)), np.zeros((4, 4, 4), dtype=np.uint8)]

    image, plan, _trims, frame_cells = sheetout.build(frames, [10, 10])
    image.close()
    assert len(plan.cells) == 2
    assert frame_cells is None

    image, plan, _trims, frame_cells = sheetout.build(frames, [10, 10], skip_empty=True)
    image.close()
    assert len(plan.cells) == 1
    assert frame_cells == [0, None]


def test_merge_and_skip_empty_compose_together():
    frames = [
        _frame(4, 4, RED, (0, 0)),
        np.zeros((4, 4, 4), dtype=np.uint8),
        _frame(4, 4, RED, (0, 0)),
        _frame(4, 4, BLUE, (1, 1)),
    ]
    image, plan, _trims, frame_cells = sheetout.build(
        frames, [10, 10, 10, 10], merge=True, skip_empty=True
    )
    image.close()
    assert len(plan.cells) == 2
    assert frame_cells == [0, None, 0, 1]


def test_a_wholly_empty_clip_with_skip_empty_is_refused_by_name():
    frames = [np.zeros((4, 4, 4), dtype=np.uint8), np.zeros((4, 4, 4), dtype=np.uint8)]
    with pytest.raises(ValueError, match="empty"):
        sheetout.build(frames, [10, 10], skip_empty=True)


def test_merge_or_skip_empty_with_a_directional_layout_is_refused_by_name():
    frames = [_frame(10, 10, RED, (0, 0)) for _ in range(4)]
    with pytest.raises(ValueError, match="directional layout"):
        sheetout.build(frames, [10] * 4, layout=_layout("turnaround"), merge=True)
    with pytest.raises(ValueError, match="directional layout"):
        sheetout.build(frames, [10] * 4, layout=_layout("turnaround"), skip_empty=True)


def test_merge_shrinks_the_cell_count_before_arrange_packs_it():
    """The grid packs the *surviving* cells: two duplicates and a unique frame
    merge down to two cells, and a horizontal arrange lays out two, not three."""
    frames = [_frame(4, 4, RED, (0, 0)), _frame(4, 4, RED, (0, 0)), _frame(4, 4, BLUE, (1, 1))]
    image, plan, _trims, _frame_cells = sheetout.build(
        frames, [10, 10, 10], merge=True, arrange="horizontal"
    )
    image.close()
    assert (plan.columns, plan.rows) == (2, 1)


def test_merged_frames_keep_their_own_durations_in_the_animation_block():
    plan = sheetout.plan_frames(1, 4, 4)
    block = sheetout.animation_block(
        plan, [40, 120], (), frame_cells=[0, 0]
    )
    assert block["frames"] == [
        {"cell_index": 0, "duration_ms": 40},
        {"cell_index": 0, "duration_ms": 120},
    ]
    assert block["merged"] is True
    assert "skipped" not in block


def test_skipped_frames_are_named_in_the_animation_block():
    plan = sheetout.plan_frames(1, 4, 4)
    block = sheetout.animation_block(
        plan, [40, 120], (), frame_cells=[0, None]
    )
    assert block["frames"] == [{"cell_index": 0, "duration_ms": 40}]
    assert block["skipped"] == [1]
    assert "merged" not in block


def test_an_ordinary_export_has_no_merged_or_skipped_keys():
    plan = sheetout.plan_frames(3, 10, 10)
    block = sheetout.animation_block(plan, [100] * 3, ())
    assert "merged" not in block
    assert "skipped" not in block


def test_merged_and_skipped_keys_ride_the_animation_block_after_tags():
    plan = sheetout.plan_frames(1, 4, 4)
    block = sheetout.animation_block(plan, [40, 120, 10], (), frame_cells=[0, 0, None])
    assert list(block) == ["frames", "tags", "merged", "skipped"]


def test_a_merged_cells_slices_are_its_representative_frames():
    """A merged cell has no single owner frame once the pixels agree, so its
    slice geometry is the *first* original frame's -- the same tie-break
    ``build`` already makes when it picks which frame's pixels become the
    cell."""
    frames = [_frame(4, 4, RED, (0, 0)), _frame(4, 4, RED, (0, 0))]
    slices = [
        {"pivot": (1.0, 1.0), "slices": []},
        {"pivot": (9.0, 9.0), "slices": []},
    ]
    _image, _plan, extra = sheetout.compose(frames, [10, 10], (), None, slices, merge=True)
    _image.close()
    assert extra["pivots"] == {0: (1.0, 1.0)}


# --- the animation block -----------------------------------------------------


def test_durations_and_tags_round_trip_in_timeline_order():
    plan = sheetout.plan_frames(3, 4, 4)
    block = sheetout.animation_block(
        plan, [40, 120, 40], [Tag(name="walk", start=0, end=2, loop=False)]
    )

    assert block["frames"] == [
        {"cell_index": 0, "duration_ms": 40},
        {"cell_index": 1, "duration_ms": 120},
        {"cell_index": 2, "duration_ms": 40},
    ]
    assert block["tags"] == [
        {"name": "walk", "start": 0, "end": 2, "loop": False, "direction": "forward"}
    ]


def test_every_cell_index_in_the_block_names_a_real_cell():
    plan = sheetout.plan_frames(5, 4000, 10)
    block = sheetout.animation_block(plan, [10] * 5, [])
    known = {c.index for c in plan.cells}
    assert {entry["cell_index"] for entry in block["frames"]} == known


# --- from a document ---------------------------------------------------------


def _animated() -> Document:
    doc = Document.blank(4, 4)
    weight = np.ones((2, 2), dtype=np.float32)
    doc.write_colour((0, 0, 2, 2), RED, weight)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    return doc


def test_a_still_document_is_refused():
    """``Export PNG`` already covers that case, and a one-frame "sprite sheet"
    is the kind of output a user has to go and check."""
    with pytest.raises(ValueError, match="not animated"):
        sheetout.from_document(Document.blank(4, 4))


def test_a_linked_cel_becomes_one_identical_cell_per_frame():
    """Right, not a bug **with merge off** (the default): an engine playing
    this back knows nothing about links, so the frames have to be there.
    ``test_linked_cels_merge_for_free_when_merge_is_on`` below is the same
    document with the option on, where the premise flips on purpose."""
    image, plan, extra = sheetout.from_document(_animated())
    try:
        assert len(plan.cells) == 3
        tiles = [
            image.crop((c.x, c.y, c.x + plan.cell_w, c.y + plan.cell_h)).tobytes()
            for c in plan.cells
        ]
        assert tiles[0] == tiles[1] == tiles[2]
        assert image.getpixel((0, 0)) == RED
    finally:
        image.close()
    assert len(extra["animation"]["frames"]) == 3


def test_linked_cels_merge_for_free_when_merge_is_on():
    """A linked cel is the same object in every frame it appears in -- same
    bytes, so it merges for free: no special-casing links, just the ordinary
    duplicate-frame path finding what a link already guaranteed."""
    image, plan, extra = sheetout.from_document(_animated(), merge=True)
    try:
        assert len(plan.cells) == 1
    finally:
        image.close()
    assert len(extra["animation"]["frames"]) == 3
    assert {f["cell_index"] for f in extra["animation"]["frames"]} == {0}
    assert extra["animation"]["merged"] is True


def test_the_sidecar_carries_the_animation_and_the_real_frame_size():
    image, plan, extra = sheetout.from_document(_animated(), name="walk")
    image.close()
    meta = sheetlib.sidecar(
        plan,
        sheet_id="s",
        source_job="",
        image="s.png",
        created=0.0,
        trims=extra["trims"],
        animation=extra["animation"],
    )

    assert meta["frame_size"] == 0
    assert (meta["frame_w"], meta["frame_h"]) == (4, 4)
    assert meta["animation"]["frames"][0]["duration_ms"] > 0
    assert all(cell["w"] == 4 and cell["h"] == 4 for cell in meta["cells"])


# --- the one pipelines reach, which is genuinely about this module -----------
#
# The rest of the package's import pin moved to ``test_inker_imports.py`` on
# 2026-08-11: it is a fact about ``studio/inker/`` and not about sprite sheets,
# and living here is how it came to carry three of the seven checks its three
# sibling package pins carry. What stays is the half whose subject is this file.


ENGINE = Path(inker.__file__).parent


def test_sheetout_reaches_for_the_format_and_nothing_else_in_pipelines():
    """``pipelines`` is a large package that runs inside worker and Blender
    processes. The allowlist above is at module granularity, so this is the
    half that says *which* module."""
    tree = ast.parse((ENGINE / "sheetout.py").read_text(encoding="utf-8"))
    reached = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("pipelines")
        for alias in node.names
    }
    assert reached == {"sheet"}


def test_the_export_is_split_so_only_the_snapshot_reads_the_document():
    """The thread split, stated where it can be checked.

    ``snapshot`` is the whole of the export that reads the document, so that it
    can be the whole of the export that runs on the frame thread: ``frame_flat``
    fills and evicts the flatten cache and ``layers_for`` copies track
    properties down onto cels, which is exactly what the onion-skin draw is
    doing to the same dicts. So ``compose`` must take no document -- a
    parameter it could reach through is a parameter a later edit will reach
    through -- and ``from_document`` must be the two back to back rather than a
    third path that could drift from either.
    """
    assert "doc" not in inspect.signature(sheetout.compose).parameters

    doc = _animated()
    image, plan, extra = sheetout.compose(*sheetout.snapshot(doc))
    image.close()
    direct_image, direct_plan, direct_extra = sheetout.from_document(_animated())
    direct_image.close()
    assert (plan, extra) == (direct_plan, direct_extra)


def test_a_snapshot_survives_the_document_recompositing_underneath_it():
    """The arrays handed out are the ones the encoder keeps. The cache replaces
    entries rather than writing into them, so a later flatten on the frame
    thread cannot rewrite a sheet that is halfway to disk."""
    doc = _animated()
    frames, _durations, _tags, _layout, _slices = sheetout.snapshot(doc)
    taken = [plane.copy() for plane in frames]

    doc.set_current_frame(1)
    doc.write_colour((0, 0, 4, 4), (0, 255, 0, 255), np.ones((4, 4), dtype=np.float32))
    for frame in doc.anim.frames:
        doc.frame_flat(frame.uid)

    assert all(np.array_equal(a, b) for a, b in zip(frames, taken, strict=True))


# --- the arrange choice -------------------------------------------------------
#
# ``arrange`` is the row-wrap's sibling, not its replacement: ``None`` is
# unchanged (the byte pin below and ``test_no_layout_is_byte_for_byte_the_
# grid_it_always_was``'s twin prove it), and the four named forms are new ways
# to pick ``columns``/``rows`` before the same cell-placement arithmetic runs.


def test_horizontal_arrange_is_one_row():
    plan = sheetout.plan_frames(5, 10, 10, arrange="horizontal")
    assert (plan.columns, plan.rows) == (5, 1)
    assert [(c.row, c.column, c.x, c.y) for c in plan.cells] == [
        (0, 0, 0, 0),
        (0, 1, 10, 0),
        (0, 2, 20, 0),
        (0, 3, 30, 0),
        (0, 4, 40, 0),
    ]


def test_vertical_arrange_is_one_column():
    plan = sheetout.plan_frames(5, 10, 10, arrange="vertical")
    assert (plan.columns, plan.rows) == (1, 5)
    assert [(c.row, c.column, c.x, c.y) for c in plan.cells] == [
        (0, 0, 0, 0),
        (1, 0, 0, 10),
        (2, 0, 0, 20),
        (3, 0, 0, 30),
        (4, 0, 0, 40),
    ]


def test_rows_arrange_fixes_the_column_count_from_wrap():
    """10 frames wrapped at 3 rows: columns = ceil(10/3) = 4, and the actual
    row count is derived from that the same way the plain row-wrap derives it
    -- so there is no dead trailing row when 3 does not divide 10 evenly."""
    plan = sheetout.plan_frames(10, 10, 10, arrange="rows", wrap=3)
    assert (plan.columns, plan.rows) == (4, 3)
    assert [c.index for c in plan.cells] == list(range(10))


def test_columns_arrange_fixes_the_column_count_directly():
    plan = sheetout.plan_frames(10, 10, 10, arrange="columns", wrap=3)
    assert (plan.columns, plan.rows) == (3, 4)


def test_arrange_none_is_byte_for_byte_the_grid_it_always_was():
    a = sheetout.plan_frames(7, 32, 32)
    b = sheetout.plan_frames(7, 32, 32, arrange=None)
    assert a == b


def test_an_arrange_the_atlas_ceiling_cannot_hold_is_refused_by_the_shared_guard():
    with pytest.raises(ValueError, match="8192"):
        sheetout.plan_frames(4000, sheetlib.MAX_ATLAS_PX, 100, arrange="horizontal")


def test_a_directional_layout_and_an_arrange_together_are_refused_by_name():
    with pytest.raises(ValueError, match="layout"):
        sheetout.plan_frames(4, 10, 10, layout=_layout("turnaround"), arrange="horizontal")


def test_a_wrap_without_a_counted_arrange_is_refused_by_name():
    with pytest.raises(ValueError, match="wrap"):
        sheetout.plan_frames(5, 10, 10, arrange="horizontal", wrap=2)
    with pytest.raises(ValueError, match="wrap"):
        sheetout.plan_frames(5, 10, 10, wrap=2)


def test_a_sub_one_wrap_is_refused_by_name():
    with pytest.raises(ValueError, match="wrap"):
        sheetout.plan_frames(5, 10, 10, arrange="rows", wrap=0)


def test_a_counted_arrange_with_no_wrap_is_refused_by_name():
    with pytest.raises(ValueError, match="wrap"):
        sheetout.plan_frames(5, 10, 10, arrange="rows")
    with pytest.raises(ValueError, match="wrap"):
        sheetout.plan_frames(5, 10, 10, arrange="columns")


def test_an_unknown_arrange_is_refused_by_name():
    with pytest.raises(ValueError, match="arrange"):
        sheetout.plan_frames(5, 10, 10, arrange="diagonal")


def test_the_arrange_rides_the_sidecars_animation_block():
    plan = sheetout.plan_frames(10, 10, 10, arrange="rows", wrap=3)
    block = sheetout.animation_block(plan, [100] * 10, (), arrange="rows", wrap=3)
    assert block["arrange"] == "rows"
    assert block["wrap"] == 3
    assert [f["cell_index"] for f in block["frames"]] == list(range(10))


def test_an_arrange_with_no_wrap_records_no_wrap_key():
    plan = sheetout.plan_frames(5, 10, 10, arrange="horizontal")
    block = sheetout.animation_block(plan, [100] * 5, (), arrange="horizontal")
    assert block["arrange"] == "horizontal"
    assert "wrap" not in block


def test_an_ordinary_export_has_no_arrange_key():
    plan = sheetout.plan_frames(3, 10, 10)
    assert "arrange" not in sheetout.animation_block(plan, [100] * 3, ())


def test_arrange_stays_inside_the_animation_block_which_stays_last():
    """``pipelines.sheet``'s square path is pinned byte-for-byte with the
    animation block last; a new key inside it must not move it, and
    ``sheet.py`` must not need to know the word "arrange" at all."""
    frames = [_frame(10, 10, RED) for _ in range(5)]
    _image, plan, extra = sheetout.compose(
        frames, [100] * 5, (), None, None, arrange="horizontal"
    )
    meta = sheetlib.sidecar(
        plan, sheet_id="s", source_job=None, image="s.png", created=1.0,
        trims=extra["trims"], animation=extra["animation"],
    )
    assert list(meta)[-1] == "animation"
    assert meta["animation"]["arrange"] == "horizontal"


# --- the default-byte-identity pin over the whole Inker export path ----------
#
# Guards every remaining Wave 4 task over this path: captured from HEAD before
# ``arrange`` existed, with ``json.dumps(sidecar(...))`` on a small non-square
# export -- the values, the key order, and (what a field-by-field check would
# miss) the absence of any new key.

DEFAULT_INKER_SIDECAR = (
    '{"version": 1, "id": "eeeeeeeeeeee", "name": "pin", "source_job": "ffffffffffff",'
    ' "created": 2.5, "image": "e.png", "frame_size": 0, "columns": 3, "rows": 1,'
    ' "width": 18, "height": 4, "elevation": 0.0, "lighting": "flat", "yaws": [0.0],'
    ' "poses": [{"id": null, "name": "pin"}], "cells": [{"index": 0, "row": 0,'
    ' "column": 0, "x": 0, "y": 0, "w": 6, "h": 4, "pose": null, "pose_name": "pin",'
    ' "yaw": 0.0, "frame": 0, "pivot_x": 3.0, "pivot_y": 4.0, "trim": {"x": 1, "y": 1,'
    ' "w": 1, "h": 1}}, {"index": 1, "row": 0, "column": 1, "x": 6, "y": 0, "w": 6,'
    ' "h": 4, "pose": null, "pose_name": "pin", "yaw": 0.0, "frame": 1, "pivot_x": 3.0,'
    ' "pivot_y": 4.0, "trim": {"x": 1, "y": 1, "w": 1, "h": 1}}, {"index": 2, "row": 0,'
    ' "column": 2, "x": 12, "y": 0, "w": 6, "h": 4, "pose": null, "pose_name": "pin",'
    ' "yaw": 0.0, "frame": 2, "pivot_x": 3.0, "pivot_y": 4.0, "trim": {"x": 1, "y": 1,'
    ' "w": 1, "h": 1}}], "frame_w": 6, "frame_h": 4, "animation": {"frames": [{"cell_index":'
    ' 0, "duration_ms": 80}, {"cell_index": 1, "duration_ms": 80}, {"cell_index": 2,'
    ' "duration_ms": 120}], "tags": [{"name": "idle", "start": 0, "end": 2, "loop": true,'
    ' "direction": "forward"}]}}'
)


def test_a_default_inker_sidecar_is_byte_for_byte_what_it_always_was():
    import json

    def _mark(width, height, mark, at=(0, 0)):
        plane = np.zeros((height, width, 4), dtype=np.uint8)
        plane[at[1], at[0]] = mark
        return plane

    frames = [_mark(6, 4, RED, (1, 1)) for _ in range(3)]
    durations = [80, 80, 120]
    tags = [Tag(name="idle", start=0, end=2, loop=True)]
    image, plan, extra = sheetout.compose(frames, durations, tags, None, None, name="pin")
    image.close()
    meta = sheetlib.sidecar(
        plan,
        sheet_id="eeeeeeeeeeee",
        source_job="ffffffffffff",
        image="e.png",
        created=2.5,
        name="pin",
        trims=extra["trims"],
        animation=extra["animation"],
        pivots=extra["pivots"],
        slices=extra["slices"],
    )
    assert json.dumps(meta) == DEFAULT_INKER_SIDECAR


# --- the directional grid ---------------------------------------------------


def _layout(kind):
    from warlock.studio.inker.animation import DirectionalLayout

    return DirectionalLayout.of(kind)


def test_a_turnaround_lays_out_two_by_two_with_its_yaws():
    plan = sheetout.plan_frames(4, 10, 10, layout=_layout("turnaround"))
    assert (plan.columns, plan.rows) == (2, 2)
    assert [(c.x, c.y) for c in plan.cells] == [(0, 0), (10, 0), (0, 10), (10, 10)]
    assert [c.pose_name for c in plan.cells] == ["front", "left", "right", "back"]
    assert [c.yaw for c in plan.cells] == [0.0, 90.0, 270.0, 180.0]
    assert [c.frame for c in plan.cells] == [0, 0, 0, 0]


def test_a_walk_lays_out_four_by_four_with_frames_restarting_per_row():
    plan = sheetout.plan_frames(16, 8, 8, layout=_layout("walk"))
    assert (plan.columns, plan.rows) == (4, 4)
    assert [c.frame for c in plan.cells] == [0, 1, 2, 3] * 4
    assert [c.pose_name for c in plan.cells[:4]] == ["front"] * 4
    assert [c.pose_name for c in plan.cells[12:]] == ["back"] * 4
    for index, cell in enumerate(plan.cells):
        assert (cell.x, cell.y) == ((index % 4) * 8, (index // 4) * 8)


def test_a_frame_count_that_does_not_fill_the_grid_is_refused():
    """Padded would be worse: a sheet with a hole where "back, frame 3" should
    be is the one outcome a game would not notice until it played it."""
    with pytest.raises(ValueError, match="16 frames and this document has 15"):
        sheetout.plan_frames(15, 8, 8, layout=_layout("walk"))


def test_the_grid_is_fixed_even_when_a_wrap_would_fit_more():
    """Without the layout, 16 eight-pixel frames wrap into one row."""
    wrapped = sheetout.plan_frames(16, 8, 8)
    assert wrapped.columns == 16
    assert sheetout.plan_frames(16, 8, 8, layout=_layout("walk")).columns == 4


def test_no_layout_is_byte_for_byte_the_grid_it_always_was():
    a = sheetout.plan_frames(7, 32, 32)
    b = sheetout.plan_frames(7, 32, 32, layout=None)
    assert a == b


def test_the_layout_rides_the_sidecars_animation_block():
    from warlock.studio.inker.animation import DIRECTION_ORDER

    layout = _layout("turnaround")
    plan = sheetout.plan_frames(4, 10, 10, layout=layout)
    block = sheetout.animation_block(plan, [100] * 4, (), layout=layout)
    assert block["layout"] == {
        "kind": "turnaround",
        "directions": list(DIRECTION_ORDER),
    }
    # Every key that was there is still there and still means the same thing.
    assert [f["cell_index"] for f in block["frames"]] == [0, 1, 2, 3]


def test_an_ordinary_export_has_no_layout_key():
    plan = sheetout.plan_frames(3, 10, 10)
    assert "layout" not in sheetout.animation_block(plan, [100] * 3, ())


def test_animation_is_still_the_last_key_of_the_sidecar():
    """``pipelines.sheet``'s square path is pinned byte-for-byte with the
    animation block last; adding a key inside it must not move it."""
    layout = _layout("turnaround")
    frames = [_frame(10, 10, RED) for _ in range(4)]
    _image, plan, extra = sheetout.compose(frames, [100] * 4, (), layout)
    meta = sheetlib.sidecar(
        plan, sheet_id="s", source_job=None, image="s.png", created=1.0,
        trims=extra["trims"], animation=extra["animation"],
    )
    assert list(meta)[-1] == "animation"
    assert meta["animation"]["layout"]["kind"] == "turnaround"


def test_compose_still_takes_a_snapshot_positionally():
    """``compose(*snapshot(doc))`` has to keep holding: a keyword-only fourth
    element would have made that spelling drop the grid silently."""
    doc = _animated()
    doc.anim.layout = _layout("turnaround")
    while len(doc.anim.frames) < 4:
        doc.add_frame()
    _image, plan, _extra = sheetout.compose(*sheetout.snapshot(doc))
    assert (plan.columns, plan.rows) == (2, 2)
