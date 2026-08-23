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


# --- tags against a shrunk frame list ----------------------------------------
#
# ``skip_empty`` drops entries from ``animation.frames``, so a tag's ``start``/
# ``end`` -- which are positions *in that list* to everyone who reads the file --
# have to be renumbered onto the survivors or a consumer indexing
# ``frames[start..end]`` plays the wrong cells.


def _skipping_block(tags):
    """Six frames, 1 and 3 dropped: positions 0,-,1,-,2,3."""
    plan = sheetout.plan_frames(4, 4, 4)
    return sheetout.animation_block(
        plan, [10] * 6, tags, frame_cells=[0, None, 1, None, 2, 3]
    )


def test_a_tag_spanning_skipped_frames_lands_on_the_surviving_entries():
    # Frames 0..2, of which 1 was dropped: the survivors are frames 0 and 2,
    # which are entries 0 and 1 of the shrunk list.
    block = _skipping_block([Tag(name="a", start=0, end=2)])
    assert block["tags"] == [
        {"name": "a", "start": 0, "end": 1, "loop": True, "direction": "forward"}
    ]
    assert block["skipped"] == [1, 3]


def test_a_tag_starting_on_a_skipped_frame_clamps_to_its_first_survivor():
    # Frames 3..5, of which 3 was dropped: frames 4 and 5 are entries 2 and 3.
    block = _skipping_block([Tag(name="c", start=3, end=5)])
    assert [(t["name"], t["start"], t["end"]) for t in block["tags"]] == [("c", 2, 3)]


def test_a_tag_wholly_inside_skipped_frames_is_dropped():
    # "b" covers frame 1 alone, which got no cell: naming it would name nothing.
    block = _skipping_block(
        [Tag(name="a", start=0, end=2), Tag(name="b", start=1, end=1)]
    )
    assert [t["name"] for t in block["tags"]] == ["a"]


def test_merge_alone_leaves_every_tag_exactly_where_it_was():
    """Nothing was dropped, so ``frames`` is still one entry per frame and the
    tags already index it correctly -- the renumbering must not fire."""
    plan = sheetout.plan_frames(1, 4, 4)
    block = sheetout.animation_block(
        plan, [10] * 3, [Tag(name="a", start=0, end=2)], frame_cells=[0, 0, 0]
    )
    assert [(t["name"], t["start"], t["end"]) for t in block["tags"]] == [("a", 0, 2)]


def test_an_ordinary_export_leaves_every_tag_exactly_where_it_was():
    plan = sheetout.plan_frames(3, 4, 4)
    block = sheetout.animation_block(
        plan, [10] * 3, [Tag(name="a", start=1, end=2)]
    )
    assert [(t["name"], t["start"], t["end"]) for t in block["tags"]] == [("a", 1, 2)]


def test_remap_tags_is_pure_and_drops_a_wholly_skipped_span():
    out = sheetout.remap_tags(
        [Tag(name="a", start=0, end=1), Tag(name="b", start=2, end=2)],
        [0, None, None, 1],
    )
    assert [(t.name, t.start, t.end) for t in out] == [("a", 0, 0)]


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


def test_a_merged_away_frames_own_slices_are_recorded_rather_than_dropped_silently():
    """Wave 4's ``slices_conflict`` item, closed.

    Dropping them is right -- a cell is one rectangle set and there is no second
    place to put another -- but the user authored geometry that is now nowhere
    in the sidecar, and until now nothing said so. ``{cell: [frames]}`` names
    exactly which frames to go and look at."""
    frames = [_frame(4, 4, RED, (0, 0))] * 2
    slices = [
        {"pivot": (1.0, 1.0), "slices": []},
        {"pivot": (9.0, 9.0), "slices": []},
    ]
    _image, _plan, extra = sheetout.compose(frames, [10, 10], (), None, slices, merge=True)
    _image.close()
    assert extra["slices_conflict"] == {0: [1]}


def test_merged_frames_that_agree_report_no_conflict():
    """The negative control, and the property the whole design rests on: the
    common case reports nothing, so the key never reaches the sidecar and every
    sheet this build wrote before it is byte-identical."""
    frames = [_frame(4, 4, RED, (0, 0))] * 3
    same = {"pivot": (1.0, 1.0), "slices": [{"name": "hit", "x": 0, "y": 0, "w": 2, "h": 2}]}
    _image, _plan, extra = sheetout.compose(
        frames, [10, 10, 10], (), None, [dict(same) for _ in range(3)], merge=True
    )
    _image.close()
    assert extra["slices_conflict"] == {}


def test_an_export_that_does_not_merge_never_reports_a_conflict():
    """With merge and skip-empty off, a cell *is* a frame -- every frame keeps
    its own slices, so there is nothing to disagree about however different
    they are."""
    frames = [_frame(4, 4, RED, (0, 0)), _frame(4, 4, RED, (0, 0))]
    slices = [{"pivot": (1.0, 1.0), "slices": []}, {"pivot": (9.0, 9.0), "slices": []}]
    _image, _plan, extra = sheetout.compose(frames, [10, 10], (), None, slices)
    _image.close()
    assert extra["slices_conflict"] == {}


def test_a_skipped_empty_frame_is_not_a_conflict():
    """A frame ``skip_empty`` left out has no cell to disagree with, and the
    user was already told it was empty by its absence. Reporting it would make
    the note fire on the ordinary use of the option."""
    frames = [_frame(4, 4, RED, (0, 0)), np.zeros((4, 4, 4), dtype=np.uint8)]
    slices = [{"pivot": (1.0, 1.0), "slices": []}, {"pivot": (9.0, 9.0), "slices": []}]
    _image, _plan, extra = sheetout.compose(
        frames, [10, 10], (), None, slices, skip_empty=True
    )
    _image.close()
    assert extra["slices_conflict"] == {}


def test_a_conflict_is_reported_per_cell_with_every_frame_that_lost_geometry():
    """Three frames onto one cell with two of them differing: both are named,
    in timeline order, under the cell they landed on."""
    frames = [_frame(4, 4, RED, (0, 0))] * 4
    slices = [
        {"pivot": (1.0, 1.0), "slices": []},
        {"pivot": (2.0, 2.0), "slices": []},
        {"pivot": (1.0, 1.0), "slices": []},
        {"pivot": (3.0, 3.0), "slices": []},
    ]
    _image, _plan, extra = sheetout.compose(
        frames, [10] * 4, (), None, slices, merge=True
    )
    _image.close()
    # Frame 2 agrees with the representative and is not named; 1 and 3 differ.
    assert extra["slices_conflict"] == {0: [1, 3]}


def test_a_conflict_in_the_rectangles_themselves_is_reported_not_only_the_pivot():
    """The pivot is the easy half. A frame whose *slice list* differs loses just
    as much geometry, and comparing only pivots would have missed the case the
    feature is named after."""
    frames = [_frame(4, 4, RED, (0, 0))] * 2
    slices = [
        {"pivot": None, "slices": [{"name": "hit", "x": 0, "y": 0, "w": 2, "h": 2}]},
        {"pivot": None, "slices": [{"name": "hit", "x": 1, "y": 1, "w": 2, "h": 2}]},
    ]
    _image, _plan, extra = sheetout.compose(frames, [10, 10], (), None, slices, merge=True)
    _image.close()
    assert extra["slices_conflict"] == {0: [1]}


def test_trim_shifts_pivots_and_slices_to_match_the_moved_art():
    """``build`` crops this frame to its trim rectangle (2, 3, 4, 2) and pastes
    the result flush at the cell's corner -- so a pivot or a slice authored
    against the *original* canvas must move by that same (-2, -3), or it names
    a point in a canvas the atlas no longer has."""
    frames = [_box(8, 8, (2, 3, 4, 2))]
    slices = [
        {
            "pivot": (3.5, 4.5),
            "slices": [
                {
                    "name": "hit",
                    "x": 3,
                    "y": 4,
                    "w": 2,
                    "h": 1,
                    "pivot": (4.0, 4.5),
                    "center": (3, 4, 2, 1),
                }
            ],
        }
    ]
    _image, _plan, extra = sheetout.compose(frames, [10], (), None, slices, trim=True)
    _image.close()
    assert extra["trims"][0] == {"x": 2, "y": 3, "w": 4, "h": 2}
    assert extra["pivots"] == {0: (1.5, 1.5)}
    assert extra["slices"][0] == [
        {
            "name": "hit",
            "bounds": {"x": 1, "y": 1, "w": 2, "h": 1},
            "pivot": {"x": 2.0, "y": 1.5},
            "center": {"x": 1, "y": 1, "w": 2, "h": 1},
        }
    ]


def test_trim_off_leaves_pivots_and_slices_in_canvas_coordinates():
    """The untouched path: with ``trim`` off (the default), nothing here
    changes -- the same numbers a caller handed in come back out."""
    frames = [_box(8, 8, (2, 3, 4, 2))]
    slices = [{"pivot": (3.5, 4.5), "slices": [{"name": "hit", "x": 3, "y": 4, "w": 2, "h": 1}]}]
    _image, _plan, extra = sheetout.compose(frames, [10], (), None, slices)
    _image.close()
    assert extra["pivots"] == {0: (3.5, 4.5)}
    assert extra["slices"][0] == [
        {
            "name": "hit",
            "bounds": {"x": 3, "y": 4, "w": 2, "h": 1},
            "pivot": None,
            "center": None,
        }
    ]


def test_trim_re_clips_a_slice_that_extended_past_the_trim_box():
    """Whatever a slice covered outside the trim rectangle was cropped out of
    the pixels themselves and is not in the atlas to describe."""
    frames = [_box(8, 8, (2, 2, 3, 3))]
    slices = [{"pivot": None, "slices": [{"name": "big", "x": 0, "y": 0, "w": 10, "h": 10}]}]
    _image, _plan, extra = sheetout.compose(frames, [10], (), None, slices, trim=True)
    _image.close()
    assert extra["slices"][0][0]["bounds"] == {"x": 0, "y": 0, "w": 3, "h": 3}


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


def test_a_sidecar_with_nothing_to_report_carries_no_slices_conflict_key():
    """The standing negative control, at the layer that actually writes files.

    ``slices_conflict`` is additive with no version bump, and what makes that
    true is that an export with nothing to say does not write the key at all --
    so every sheet this build produced before the note existed is byte-identical
    and no reader has to learn anything."""
    frames = [_frame(10, 10, RED) for _ in range(4)]
    _image, plan, extra = sheetout.compose(frames, [100] * 4)
    _image.close()
    meta = sheetlib.sidecar(
        plan, sheet_id="s", source_job=None, image="s.png", created=1.0,
        trims=extra["trims"], animation=extra["animation"],
        slices_conflict=extra["slices_conflict"],
    )
    assert "slices_conflict" not in meta
    # And passing nothing at all is the same file, which is what every caller
    # that has not learned about the key yet is doing.
    bare = sheetlib.sidecar(
        plan, sheet_id="s", source_job=None, image="s.png", created=1.0,
        trims=extra["trims"], animation=extra["animation"],
    )
    assert meta == bare


def test_a_reported_conflict_reaches_the_sidecar_as_plain_json():
    """And when there *is* something to say it lands as ints, not numpy scalars
    or tuples -- the sidecar is written with ``json.dumps`` and a key that
    cannot serialise would take the whole export down at the last step."""
    import json

    frames = [_frame(10, 10, RED) for _ in range(2)]
    slices = [{"pivot": (1.0, 1.0), "slices": []}, {"pivot": (9.0, 9.0), "slices": []}]
    _image, plan, extra = sheetout.compose(
        frames, [100, 100], (), None, slices, merge=True
    )
    _image.close()
    meta = sheetlib.sidecar(
        plan, sheet_id="s", source_job=None, image="s.png", created=1.0,
        trims=extra["trims"], animation=extra["animation"],
        slices_conflict=extra["slices_conflict"],
    )
    assert meta["slices_conflict"] == {0: [1]}
    assert json.loads(json.dumps(meta))["slices_conflict"] == {"0": [1]}
    # ``animation`` stays last: the key is appended after it, so this is the
    # one pin that would catch it being inserted in the middle instead.
    assert list(meta)[-1] == "slices_conflict"


def test_compose_still_takes_a_snapshot_positionally():
    """``compose(*snapshot(doc))`` has to keep holding: a keyword-only fourth
    element would have made that spelling drop the grid silently."""
    doc = _animated()
    doc.anim.layout = _layout("turnaround")
    while len(doc.anim.frames) < 4:
        doc.add_frame()
    _image, plan, _extra = sheetout.compose(*sheetout.snapshot(doc))
    assert (plan.columns, plan.rows) == (2, 2)


# --- subsets: one part of the stack, on its own --------------------------------
#
# Split-by-layer needs a flatten of *some* of the tracks, and the one rule it
# must not break is the frame cache: ``frame_flat`` is keyed on the frame uid
# alone, so a subset that entered it would hand the onion skin (and the next
# whole-document export) a picture with layers missing.


def _tracked() -> Document:
    """Two tracks over three frames: RED bottom-left, BLUE top-right."""
    doc = Document.blank(4, 4)
    ones = np.ones((2, 2), dtype=np.float32)
    doc.write_colour((0, 0, 2, 2), RED, ones)
    doc.add_layer("ink")
    doc.write_colour((2, 2, 4, 4), BLUE, ones)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    return doc


def test_a_subset_flatten_holds_only_the_tracks_it_was_given():
    doc = _tracked()
    uid = doc.anim.frames[0].uid
    plane = sheetout.flatten_subset(doc, uid, {doc.anim.tracks[0].uid})
    assert tuple(plane[0, 0]) == RED
    assert tuple(plane[3, 3]) == (0, 0, 0, 0)  # the ink track is not in it


def test_a_subset_flatten_equals_the_whole_one_with_the_others_hidden():
    """The oracle is the path that was already right: hiding every other track
    and flattening the whole stack has to produce exactly what asking for one
    track produces, opacity and blend mode included."""
    doc = _tracked()
    doc.set_layer_props(1, opacity=0.5, blend="multiply")
    uid = doc.anim.frames[0].uid
    subset = sheetout.flatten_subset(doc, uid, {doc.anim.tracks[1].uid})

    doc.set_layer_props(0, visible=False)
    expected = doc.frame_flat(uid)
    assert np.array_equal(subset, expected)


def test_a_hidden_track_inside_a_subset_still_contributes_nothing():
    doc = _tracked()
    doc.set_layer_props(1, visible=False)
    uid = doc.anim.frames[0].uid
    both = sheetout.flatten_subset(
        doc, uid, {doc.anim.tracks[0].uid, doc.anim.tracks[1].uid}
    )
    assert np.array_equal(both, sheetout.flatten_subset(doc, uid, {doc.anim.tracks[0].uid}))


def test_a_subset_carries_the_group_fold_of_the_rows_it_keeps():
    """The fold is per stack row, so a filtered stack needs a filtered fold --
    an unfiltered one would hang row 0's inherited opacity on row 1."""
    doc = _tracked()
    node = doc.group_layers([1], name="fx")
    assert node is not None
    doc.set_group_props(node.uid, visible=False)
    uid = doc.anim.frames[0].uid

    inside = sheetout.flatten_subset(doc, uid, {doc.anim.tracks[1].uid})
    assert not inside.any()  # the group above it is hidden
    outside = sheetout.flatten_subset(doc, uid, {doc.anim.tracks[0].uid})
    assert tuple(outside[0, 0]) == RED  # and that fold is not applied to row 0


def test_a_subset_flatten_never_enters_the_frame_cache():
    """The cache is keyed on a frame and *at most one track* -- the onion
    skin's current-layer-only ask. An arbitrary subset stored under either key
    is a whole document's flatten with layers missing, handed to the onion
    skin, to playback and to the next export, so it stays out entirely."""
    doc = _tracked()
    uid = doc.anim.frames[0].uid
    assert doc.frame_cache_bytes() == 0
    sheetout.flatten_subset(doc, uid, {doc.anim.tracks[0].uid})
    assert doc.frame_cache_bytes() == 0

    full = doc.frame_flat(uid).copy()
    filled = doc.frame_cache_bytes()
    sheetout.flatten_subset(doc, uid, {doc.anim.tracks[1].uid})
    assert doc.frame_cache_bytes() == filled
    assert np.array_equal(doc.frame_flat(uid), full)
    assert tuple(full[0, 0]) == RED and tuple(full[3, 3]) == BLUE


def test_a_subset_naming_no_live_track_is_refused():
    doc = _tracked()
    with pytest.raises(ValueError):
        sheetout.flatten_subset(doc, doc.anim.frames[0].uid, {-1})


def test_a_subset_of_a_frame_that_is_not_in_the_document_is_refused():
    doc = _tracked()
    with pytest.raises(ValueError):
        sheetout.flatten_subset(doc, 999999, {doc.anim.tracks[0].uid})


def test_frame_stack_with_no_subset_is_the_stack_it_always_was():
    """The pin on the additive parameter: None is today's call, byte for byte."""
    doc = _tracked()
    frame = doc.anim.frames[0]
    plain, again = doc.frame_stack(frame), doc.frame_stack(frame, track_uids=None)
    assert len(plain) == len(again) == 2
    assert np.array_equal(plain.flatten(), again.flatten())


def test_a_subset_snapshot_reads_every_frame_through_the_subset():
    doc = _tracked()
    frames, _durations, _tags, _layout, _slices = sheetout.snapshot(
        doc, track_uids={doc.anim.tracks[0].uid}
    )
    assert len(frames) == 3
    assert all(tuple(plane[3, 3]) == (0, 0, 0, 0) for plane in frames)
    assert doc.frame_cache_bytes() == 0


# --- what a split is split *into* ----------------------------------------------


def test_the_layer_splits_of_a_plain_document_are_its_tracks_bottom_first():
    doc = _tracked()
    splits = sheetout.layer_splits(doc)
    assert [name for name, _uids in splits] == ["Background", "ink"]
    assert [uids for _name, uids in splits] == [
        (doc.anim.tracks[0].uid,),
        (doc.anim.tracks[1].uid,),
    ]


def test_a_group_is_one_split_holding_every_layer_inside_it():
    """A group is what the outline shows as one row, and its members composite
    as a unit -- so it is one output, not one per leaf."""
    doc = _tracked()
    doc.add_layer("glow")
    node = doc.group_layers([1, 2], name="fx")
    assert node is not None
    splits = sheetout.layer_splits(doc)
    assert [name for name, _uids in splits] == ["Background", "fx"]
    assert set(splits[1][1]) == {doc.anim.tracks[1].uid, doc.anim.tracks[2].uid}


def test_a_hidden_row_is_not_a_split_of_its_own():
    doc = _tracked()
    doc.set_layer_props(1, visible=False)
    assert [name for name, _uids in sheetout.layer_splits(doc)] == ["Background"]


def test_a_hidden_group_is_not_a_split_of_its_own():
    doc = _tracked()
    node = doc.group_layers([1], name="fx")
    assert node is not None
    doc.set_group_props(node.uid, visible=False)
    assert [name for name, _uids in sheetout.layer_splits(doc)] == ["Background"]


def test_a_visible_group_whose_layers_are_all_hidden_is_not_a_split():
    """Wave 4's own "Left open / owed" item, closed.

    The group's eye is open, so the old check -- which read only the root's
    flags -- emitted the row and the export wrote a **fully transparent sheet**
    named "fx". Nothing in it, and nothing saying so. The docstring's rule was
    always "contributes no pixels", and a group is what its members contribute;
    ``skip_empty`` on the same export already raises rather than write a blank
    cell, so the two doors now agree instead of disagreeing by which one you
    reached."""
    doc = _tracked()
    doc.add_layer("glow")
    node = doc.group_layers([1, 2], name="fx")
    assert node is not None
    doc.set_layer_props(1, visible=False)
    doc.set_layer_props(2, visible=False)
    assert [name for name, _uids in sheetout.layer_splits(doc)] == ["Background"]


def test_a_group_with_one_visible_layer_left_is_still_a_split():
    """The boundary from the legal side: *any* surviving member keeps the row,
    and the row still carries every leaf -- a hidden leaf contributes nothing to
    the composite anyway, so the fix decides whether to emit the output, never
    what goes into it."""
    doc = _tracked()
    doc.add_layer("glow")
    node = doc.group_layers([1, 2], name="fx")
    assert node is not None
    doc.set_layer_props(1, visible=False)
    splits = sheetout.layer_splits(doc)
    assert [name for name, _uids in splits] == ["Background", "fx"]
    assert set(splits[1][1]) == {doc.anim.tracks[1].uid, doc.anim.tracks[2].uid}


def test_a_group_whose_layers_are_all_at_zero_opacity_is_not_a_split():
    """Opacity is the other half of "contributes no pixels", and it is a
    separate field -- a fix that read only ``visible`` would leave this case
    writing the same transparent sheet."""
    doc = _tracked()
    node = doc.group_layers([1], name="fx")
    assert node is not None
    doc.set_layer_props(1, opacity=0.0)
    assert [name for name, _uids in sheetout.layer_splits(doc)] == ["Background"]


def test_a_group_hidden_by_a_nested_subgroup_is_not_a_split():
    """Effective visibility, not the leaf's own flag. Every leaf's eye is open
    here; the subgroup between them and the root is what hides them, which is
    exactly the case a check that looked one level up would miss. ``resolve``
    is asked instead -- the same AND/multiply fold the compositor uses."""
    doc = _tracked()
    doc.add_layer("glow")
    outer = doc.group_layers([1, 2], name="fx")
    assert outer is not None
    inner = doc.group_layers([1, 2], name="inner")
    assert inner is not None
    doc.set_group_props(inner.uid, visible=False)
    assert [name for name, _uids in sheetout.layer_splits(doc)] == ["Background"]


def test_a_still_document_has_no_layer_splits():
    assert sheetout.layer_splits(Document.blank(4, 4)) == []


def test_a_tag_span_is_clamped_onto_the_frames_that_exist():
    doc = _animated()
    assert doc.add_tag("walk", 1, 99)
    assert sheetout.tag_span(doc.anim, doc.anim.tags[0]) == (1, 2)


# --- trim, padding, extrude ----------------------------------------------------
#
# All three off by default, which is what ``test_a_default_inker_sidecar_is_
# byte_for_byte_what_it_always_was`` above already proves: the untouched path
# is byte-for-byte the sheet this always packed. On, they compose in the order
# packwright's own grid mode established -- trim decides the cell size,
# padding spaces the grid out, extrude replicates each placed rectangle's own
# border into whatever gutter padding left.


def _box(width: int, height: int, rect: tuple[int, int, int, int], colour=RED) -> np.ndarray:
    """An otherwise-transparent frame with a solid filled rectangle in it."""
    plane = np.zeros((height, width, 4), dtype=np.uint8)
    x, y, w, h = rect
    plane[y : y + h, x : x + w] = colour
    return plane


def test_trim_shrinks_the_cell_to_the_largest_trimmed_frame():
    frames = [_box(8, 8, (1, 2, 3, 2)), _box(8, 8, (0, 0, 5, 1))]
    image, plan, trims, _frame_cells = sheetout.build(frames, [10, 10], trim=True)
    try:
        # cell size is the max across surviving frames: w=max(3,5)=5, h=max(2,1)=2.
        assert (plan.cell_w, plan.cell_h) == (5, 2)
        assert (plan.columns, plan.rows) == (2, 1)
        assert (plan.width, plan.height) == (10, 2)
        # Each frame's own trim rectangle is unchanged in meaning: the source
        # offset and size within the *original* (untrimmed) frame.
        assert trims[0] == {"x": 1, "y": 2, "w": 3, "h": 2}
        assert trims[1] == {"x": 0, "y": 0, "w": 5, "h": 1}
        # And each frame's own (possibly smaller) trimmed pixels are placed
        # flush at its cell's corner -- not centred, not stretched to the
        # uniform cell size.
        assert image.getpixel((0, 0)) == RED
        assert image.getpixel((2, 1)) == RED
        assert image.getpixel((3, 0)) == (0, 0, 0, 0)  # inside cell 0, past frame 0's 3px width
        assert image.getpixel((5, 0)) == RED  # cell 1 starts at x=5
        assert image.getpixel((9, 0)) == RED  # cell 1's frame is 5px wide, flush left
        assert image.getpixel((5, 1)) == (0, 0, 0, 0)  # cell 1's frame is 1px tall
    finally:
        image.close()


def test_a_frame_trimming_to_nothing_stays_a_blank_cell():
    """``skip_empty`` is off, so the empty frame still gets a cell -- just an
    empty one, sized by whatever the other frames need."""
    frames = [_box(8, 8, (0, 0, 4, 4)), np.zeros((8, 8, 4), dtype=np.uint8)]
    image, plan, trims, _frame_cells = sheetout.build(frames, [10, 10], trim=True)
    try:
        assert len(plan.cells) == 2
        assert trims[0] == {"x": 0, "y": 0, "w": 4, "h": 4}
        assert trims[1] is None
        cell1 = plan.cells[1]
        block = image.crop((cell1.x, cell1.y, cell1.x + plan.cell_w, cell1.y + plan.cell_h))
        assert block.getbbox() is None, "an empty trim pastes nothing into its cell"
    finally:
        image.close()


def test_a_wholly_empty_clip_falls_back_to_a_one_by_one_cell():
    """No ``skip_empty``, so nothing is dropped -- and no frame is non-empty to
    measure a cell size from, so the cell floors at 1x1 rather than claiming a
    size nothing here ever measured."""
    frames = [np.zeros((8, 8, 4), dtype=np.uint8), np.zeros((8, 8, 4), dtype=np.uint8)]
    image, plan, trims, _frame_cells = sheetout.build(frames, [10, 10], trim=True)
    try:
        assert (plan.cell_w, plan.cell_h) == (1, 1)
        assert trims == {0: None, 1: None}
    finally:
        image.close()


def test_padding_borders_the_atlas_and_gutters_every_cell():
    frames = [_box(4, 4, (0, 0, 4, 4)), _box(4, 4, (0, 0, 4, 4))]
    image, plan, _trims, _frame_cells = sheetout.build(
        frames, [10, 10], arrange="horizontal", padding=3
    )
    try:
        # padding + columns * (cell + padding) == 3 + 2 * (4 + 3) == 17
        assert (plan.width, plan.height) == (17, 3 + 1 * 7)
        assert [(c.x, c.y) for c in plan.cells] == [(3, 3), (10, 3)]
        assert image.size == (plan.width, plan.height)
        # The border and the gutter are transparent -- nothing was asked to
        # extrude into them.
        assert image.getpixel((0, 0)) == (0, 0, 0, 0)
        assert image.getpixel((7, 3)) == (0, 0, 0, 0)  # the gutter between cells
        assert image.getpixel((3, 3)) == RED
        assert image.getpixel((13, 3)) == RED
    finally:
        image.close()


def test_extrude_replicates_the_placed_rectangles_border_into_the_padding():
    frames = [_box(4, 4, (0, 0, 4, 4))]
    image, plan, _trims, _frame_cells = sheetout.build(
        frames, [10], padding=4, extrude=2
    )
    try:
        cell = plan.cells[0]
        grown = image.crop(
            (cell.x - 2, cell.y - 2, cell.x + plan.cell_w + 2, cell.y + plan.cell_h + 2)
        )
        assert grown.size == (8, 8)
        assert grown.getbbox() == (0, 0, 8, 8)
        assert (np.asarray(grown) == np.array(RED, np.uint8)).all(), (
            "extrude must fill every pixel, corners included"
        )
    finally:
        image.close()


def test_extrude_replicates_only_the_trimmed_frames_own_box():
    """With trim on, a smaller-than-cell frame's border is what gets
    replicated -- not the (mostly blank) uniform cell around it."""
    frames = [_box(8, 8, (0, 0, 2, 2))]
    image, plan, trims, _frame_cells = sheetout.build(
        frames, [10], trim=True, padding=4, extrude=2
    )
    try:
        assert trims[0]["w"] == trims[0]["h"] == 2
        cell = plan.cells[0]
        grown = image.crop((cell.x - 2, cell.y - 2, cell.x + 2 + 2, cell.y + 2 + 2))
        assert grown.size == (6, 6)
        assert (np.asarray(grown) == np.array(RED, np.uint8)).all()
        # Nothing bled past the extruded box into the rest of the (bigger,
        # uniform) cell.
        assert image.getpixel((cell.x + 4, cell.y)) == (0, 0, 0, 0)
    finally:
        image.close()


def test_padding_less_than_twice_extrude_is_refused_by_name():
    frames = [_box(4, 4, (0, 0, 4, 4))]
    with pytest.raises(ValueError, match="twice extrude"):
        sheetout.build(frames, [10], padding=1, extrude=1)
    with pytest.raises(ValueError, match="twice extrude"):
        sheetout.compose(frames, [10], padding=3, extrude=2)


def test_negative_padding_or_extrude_is_refused_by_name():
    frames = [_box(4, 4, (0, 0, 4, 4))]
    with pytest.raises(ValueError, match="negative"):
        sheetout.build(frames, [10], padding=-1)
    with pytest.raises(ValueError, match="negative"):
        sheetout.build(frames, [10], extrude=-1)


def test_merge_hashes_the_full_untrimmed_frame_not_the_trimmed_content():
    """Two frames whose *trimmed* content would be pixel-identical, but whose
    full frames differ (the content sits in a different place), must not
    merge: ``merge`` dedupes on the frame it was actually handed, before any
    trim runs."""
    frames = [_box(8, 8, (0, 0, 2, 2)), _box(8, 8, (4, 4, 2, 2))]
    image, plan, _trims, frame_cells = sheetout.build(
        frames, [10, 10], merge=True, trim=True
    )
    try:
        assert len(plan.cells) == 2
        assert frame_cells == [0, 1]
    finally:
        image.close()


def test_trim_shrinks_the_cell_before_arrange_packs_it():
    frames = [_box(8, 8, (0, 0, 3, 3)) for _ in range(3)]
    image, plan, _trims, _frame_cells = sheetout.build(
        frames, [10, 10, 10], trim=True, arrange="horizontal"
    )
    try:
        assert (plan.columns, plan.rows) == (3, 1)
        assert (plan.cell_w, plan.cell_h) == (3, 3)
        assert plan.width == 9
    finally:
        image.close()


def test_an_ordinary_export_has_no_trim_padding_or_extrude_keys():
    plan = sheetout.plan_frames(3, 10, 10)
    block = sheetout.animation_block(plan, [100] * 3, ())
    assert "trimmed" not in block
    assert "padding" not in block
    assert "extrude" not in block


def test_trim_padding_extrude_ride_the_animation_block_which_stays_last():
    frames = [_box(8, 8, (0, 0, 4, 4)) for _ in range(2)]
    _image, plan, extra = sheetout.compose(
        frames, [100, 100], (), None, None, trim=True, padding=3, extrude=1
    )
    meta = sheetlib.sidecar(
        plan, sheet_id="s", source_job=None, image="s.png", created=1.0,
        trims=extra["trims"], animation=extra["animation"],
    )
    assert list(meta)[-1] == "animation"
    assert meta["animation"]["trimmed"] is True
    assert meta["animation"]["padding"] == 3
    assert meta["animation"]["extrude"] == 1
    # The per-cell trim rectangle beside it is unchanged in meaning: the
    # source offset and size within the untrimmed frame.
    assert all(cell["trim"] == {"x": 0, "y": 0, "w": 4, "h": 4} for cell in meta["cells"])


def test_the_default_inker_sidecar_pin_is_untouched_by_the_new_keyword_args():
    """The byte pin above passes none of ``trim``/``padding``/``extrude`` --
    this confirms explicitly setting them to their defaults is the same call."""
    frames = [_box(6, 4, (1, 1, 1, 1)) for _ in range(3)]
    durations = [80, 80, 120]
    tags = [Tag(name="idle", start=0, end=2, loop=True)]
    a_image, a_plan, a_extra = sheetout.compose(frames, durations, tags, None, None, name="pin")
    b_image, b_plan, b_extra = sheetout.compose(
        frames, durations, tags, None, None, name="pin",
        trim=False, padding=0, extrude=0,
    )
    try:
        assert a_plan == b_plan
        assert a_extra == b_extra
        assert np.array_equal(np.asarray(a_image), np.asarray(b_image))
    finally:
        a_image.close()
        b_image.close()


# --- filename templates --------------------------------------------------------


def test_every_known_key_fills_in():
    assert (
        sheetout.filename_for(
            "{title}_{tag}_{frame}_{layer}",
            title="walk",
            tag="intro",
            frame=3,
            layer="ink",
        )
        == "walk_intro_0003_ink"
    )


def test_frame_is_always_zero_padded_to_four():
    assert sheetout.filename_for("{frame}", title="t", frame=7) == "0007"
    assert sheetout.filename_for("{frame}", title="t", frame=12345) == "12345"


def test_a_template_with_no_keys_is_the_literal_text():
    assert sheetout.filename_for("static", title="whatever") == "static"


def test_an_unknown_key_is_refused_by_name_listing_the_known_ones():
    with pytest.raises(ValueError) as excinfo:
        sheetout.filename_for("{title}_{index}", title="walk")
    message = str(excinfo.value)
    assert "{index}" in message
    for key in sheetout.FILENAME_KEYS:
        assert "{" + key + "}" in message


def test_a_tag_key_with_no_tag_is_refused_by_name():
    with pytest.raises(ValueError, match="no tag"):
        sheetout.filename_for("{title}_{tag}", title="walk")


def test_a_layer_key_with_no_layer_is_refused_by_name():
    with pytest.raises(ValueError, match="no layer"):
        sheetout.filename_for("{title}_{layer}", title="walk")


def test_a_frame_key_with_no_frame_is_refused_by_name():
    with pytest.raises(ValueError, match="no frame"):
        sheetout.filename_for("{frame}", title="walk")


def test_no_frame_is_fine_when_the_template_does_not_ask_for_one():
    assert sheetout.filename_for("{title}", title="walk", frame=None) == "walk"


def test_tag_and_layer_are_sanitised_the_same_way_split_stems_always_did():
    assert (
        sheetout.filename_for("{title}_{tag}", title="walk", tag="A/B*C? swing")
        == "walk_A-B-C-swing"
    )


def test_title_is_never_sanitised():
    assert (
        sheetout.filename_for("{title}", title="A/B*C? swing") == "A/B*C? swing"
    )


def test_a_tag_that_sanitises_to_nothing_is_refused_even_unused():
    """The eager check ``_split_stems`` always made -- a label is validated
    whether or not the template that will be attached to it happens to read
    it, because the label itself is not a name a file can be given."""
    with pytest.raises(ValueError, match="not a name a file can be given"):
        sheetout.filename_for("{title}", title="walk", tag="///")


def test_a_template_with_a_stray_brace_is_refused_not_crashed():
    with pytest.raises(ValueError):
        sheetout.filename_for("{title", title="walk")


def test_sanitize_stem_is_the_rule_filename_for_applies_to_tag_and_layer():
    assert sheetout.sanitize_stem("A/B*C? swing") == "A-B-C-swing"
    assert sheetout.sanitize_stem("///") == ""


def test_require_distinct_names_passes_a_genuinely_distinct_list():
    sheetout.require_distinct_names(["a", "b", "c"])  # does not raise


def test_require_distinct_names_refuses_a_repeat():
    with pytest.raises(ValueError, match="b"):
        sheetout.require_distinct_names(["a", "b", "b"])


def test_require_distinct_names_refuses_an_empty_name():
    with pytest.raises(ValueError, match="empty"):
        sheetout.require_distinct_names(["a", ""])


# --- the default templates, pinned ---------------------------------------------


def test_the_default_frame_template_is_title_underscore_frame():
    assert sheetout.DEFAULT_FRAME_TEMPLATE == "{title}_{frame}"


def test_the_default_split_templates_are_title_underscore_tag_or_layer():
    assert sheetout.DEFAULT_TAG_TEMPLATE == "{title}_{tag}"
    assert sheetout.DEFAULT_LAYER_TEMPLATE == "{title}_{layer}"


def test_the_default_frame_template_reproduces_the_old_index_naming():
    """``f"{stem}_{index:04d}.png"`` before templates existed."""
    for index in (0, 1, 9, 42, 4321):
        assert sheetout.filename_for(
            sheetout.DEFAULT_FRAME_TEMPLATE, title="walk", frame=index
        ) == f"walk_{index:04d}"


def test_the_default_split_templates_reproduce_the_old_split_stems():
    assert (
        sheetout.filename_for(sheetout.DEFAULT_TAG_TEMPLATE, title="walk", tag="run")
        == "walk_run"
    )
    assert (
        sheetout.filename_for(
            sheetout.DEFAULT_LAYER_TEMPLATE, title="walk", layer="ink"
        )
        == "walk_ink"
    )


def test_the_flatten_matte_never_reaches_a_sheet_cell():
    """The deliberate divergence, guarded so nobody "fixes" it.

    A flat PNG export composites ``Document.matte`` under the stack; an atlas
    must not. A sheet cell is drawn over whatever is behind it in the game, so
    baking the matte in would put an opaque square around every frame of a
    sprite opened from a photo. Same pixels with the matte on and off.
    """
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[1, 1] = RED
    doc.invalidate_all()
    doc.ensure_animation()
    uid = doc.anim.frames[0].uid

    doc.matte = None
    off_planes, *_ = sheetout.snapshot(doc)
    off_one = sheetout.flatten_one(doc, uid).copy()

    doc.matte = (255, 255, 255, 255)
    doc.invalidate_all()
    on_planes, *_ = sheetout.snapshot(doc)
    on_one = sheetout.flatten_one(doc, uid).copy()

    assert np.array_equal(off_planes[0], on_planes[0])
    assert np.array_equal(off_one, on_one)
    # And they are genuinely transparent, not merely equal to each other.
    assert int(on_planes[0][0, 0, 3]) == 0
