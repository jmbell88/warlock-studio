"""Slices in a sprite sheet's sidecar: additive, and pinned to be.

The whole shape of S4 is "nothing new happens to a document that has no
slices", so the sidecar's byte pin in ``tests/test_sheet.py`` is the negative
control and the cases here are what a document *with* slices adds: a per-cell
pivot that overrides the bottom-centre constant, a per-cell ``slices`` block in
source-image space, and a pixel-sheet rescale of both.
"""

from __future__ import annotations

import json

import pytest

from warlock.pipelines import pixelsheet
from warlock.pipelines import sheet as sheetlib
from warlock.studio.inker import sheetout
from warlock.studio.inker.document import Document
from warlock.studio.inker.slices import SliceKey


def _animated(width: int = 16, height: int = 12, frames: int = 3) -> Document:
    doc = Document.blank(width, height)
    doc.stack.active.pixels[1:4, 1:4] = (255, 0, 0, 255)
    for _ in range(frames - 1):
        doc.add_frame(copy=True)
    return doc


def _sidecar(doc: Document) -> dict:
    image, plan, extra = sheetout.from_document(doc)
    image.close()
    return sheetlib.sidecar(
        plan,
        sheet_id="s",
        source_job="j",
        image="s.png",
        created=0.0,
        trims=extra["trims"],
        animation=extra["animation"],
        pivots=extra["pivots"],
        slices=extra["slices"],
    )


# --- the snapshot -------------------------------------------------------------


def test_the_snapshot_is_plain_data_in_canvas_coordinates():
    doc = _animated()
    doc.add_slice((2, 3, 10, 9), name="body", pivot=(4.0, 6.0), center=(2, 2, 6, 4))
    snap = sheetout.slices_snapshot(doc)

    assert len(snap) == 3
    assert snap[0] == {
        "pivot": (6.0, 9.0),
        "slices": [
            {
                "name": "body",
                "x": 2, "y": 3, "w": 8, "h": 6,
                # Absolute, not relative to the bounds: the boundary converts
                # once so no consumer has to.
                "pivot": (6.0, 9.0),
                "center": (4, 5, 4, 2),
            }
        ],
    }
    assert json.loads(json.dumps(snap, default=list))


def test_a_keyed_frame_resolves_to_its_own_rectangle():
    doc = _animated()
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(6, 6, 10, 10)))
    snap = sheetout.slices_snapshot(doc)
    assert [meta["slices"][0]["x"] for meta in snap] == [0, 6, 0]


def test_the_pivot_is_the_first_slice_that_has_one():
    """One rule, and one a user can predict from the list they are looking at:
    a sprite has one pivot and a document may have several slices."""
    doc = _animated()
    doc.add_slice((0, 0, 4, 4), name="no pivot")
    doc.add_slice((8, 8, 12, 12), name="has one", pivot=(1.0, 2.0))
    doc.add_slice((0, 8, 4, 12), name="also", pivot=(3.0, 3.0))
    assert sheetout.slices_snapshot(doc)[0]["pivot"] == (9.0, 10.0)


def test_a_span_snapshot_holds_only_that_span_s_frames():
    """The block downstream is keyed by *cell* index, so a snapshot of the whole
    timeline handed to a two-cell export would hang frame 0's rectangles on the
    cell holding frame 1. Sliced by the same span as the frames, or not at all."""
    doc = _animated()
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(6, 6, 10, 10)))
    assert [meta["slices"][0]["x"] for meta in sheetout.slices_snapshot(doc, (1, 2))] == [6, 0]
    assert sheetout.slices_snapshot(doc, None) == sheetout.slices_snapshot(doc)


def test_slice_geometry_follows_an_upscaled_export():
    """An export written at 4x with a sidecar describing the canvas names the
    wrong pixels. Exact, because it is an integer count times an integer."""
    doc = _animated()
    doc.add_slice((2, 3, 10, 9), name="body", pivot=(4.0, 6.0), center=(2, 2, 6, 4))
    snap = sheetout.slices_snapshot(doc)
    scaled = sheetout.scale_slices(snap, 4)

    assert sheetout.scale_slices(snap, 1) == snap
    assert scaled[0]["pivot"] == (24.0, 36.0)
    one = scaled[0]["slices"][0]
    assert (one["x"], one["y"], one["w"], one["h"]) == (8, 12, 32, 24)
    assert one["pivot"] == (24.0, 36.0)
    assert one["center"] == (16, 20, 16, 8)
    assert one["name"] == "body"


def test_a_still_document_has_no_sheet_to_snapshot():
    doc = Document.blank(8, 8)
    doc.add_slice((0, 0, 4, 4))
    with pytest.raises(ValueError, match="not animated"):
        sheetout.slices_snapshot(doc)


# --- the sidecar --------------------------------------------------------------


def test_a_clip_with_no_slices_carries_neither_key():
    """The additive half, from the inside: the cells are exactly what they were.
    ``tests/test_sheet.py``'s square pin is the same statement from outside."""
    meta = _sidecar(_animated())
    for cell in meta["cells"]:
        assert "slices" not in cell
        assert (cell["pivot_x"], cell["pivot_y"]) == (8.0, 12.0)


def test_a_pivot_overrides_the_bottom_centre_constant_per_cell():
    doc = _animated()
    entry = doc.add_slice((2, 2, 10, 10), pivot=(4.0, 8.0))
    doc.set_slice_key(
        entry.uid,
        doc.anim.frames[2].uid,
        key=SliceKey(bounds=(2, 2, 10, 10), pivot=(0.0, 0.0)),
    )
    cells = _sidecar(doc)["cells"]
    assert [(c["pivot_x"], c["pivot_y"]) for c in cells] == [
        (6.0, 10.0),
        (6.0, 10.0),
        (2.0, 2.0),
    ]


def test_the_slice_block_lands_on_the_right_cell():
    doc = _animated()
    entry = doc.add_slice((0, 0, 4, 4), name="hit")
    doc.set_slice_key(entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(6, 6, 10, 10)))
    cells = _sidecar(doc)["cells"]
    assert [c["slices"][0]["bounds"] for c in cells] == [
        {"x": 0, "y": 0, "w": 4, "h": 4},
        {"x": 6, "y": 6, "w": 4, "h": 4},
        {"x": 0, "y": 0, "w": 4, "h": 4},
    ]
    assert [c["slices"][0]["name"] for c in cells] == ["hit"] * 3
    assert cells[0]["slices"][0]["pivot"] is None
    assert cells[0]["slices"][0]["center"] is None


def test_a_nine_slice_centre_is_absolute_in_the_block():
    doc = _animated()
    doc.add_slice((2, 3, 10, 9), name="panel", center=(2, 2, 6, 4))
    block = _sidecar(doc)["cells"][0]["slices"][0]
    assert block["center"] == {"x": 4, "y": 5, "w": 4, "h": 2}


def test_the_sidecar_is_still_plain_json():
    meta = _sidecar(_animated())
    assert json.loads(json.dumps(meta)) == meta


def test_animation_stays_the_last_key_of_the_sidecar():
    """The slice block rides the *cells*, not a sixth top-level key, so the
    order this format has always had is untouched."""
    doc = _animated()
    doc.add_slice((0, 0, 4, 4))
    meta = _sidecar(doc)
    assert list(meta)[-1] == "animation"


def test_a_directional_sheet_keys_its_block_by_timeline_position():
    """``cell.index``, not ``cell.frame``: a directional sheet restarts ``frame``
    per row, so it is the row's own numbering."""
    from warlock.studio.inker.animation import DirectionalLayout

    doc = _animated(frames=4)
    doc.anim.layout = DirectionalLayout("turnaround")
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, doc.anim.frames[3].uid, key=SliceKey(bounds=(8, 8, 12, 12)))
    cells = _sidecar(doc)["cells"]
    assert [c["frame"] for c in cells] == [0, 0, 0, 0], "the row numbering restarts"
    assert cells[3]["slices"][0]["bounds"] == {"x": 8, "y": 8, "w": 4, "h": 4}
    assert cells[0]["slices"][0]["bounds"] == {"x": 0, "y": 0, "w": 4, "h": 4}


# --- the pixel sheet ----------------------------------------------------------


def _render_meta(**cell) -> dict:
    """A rendered sheet's sidecar, square and reducible by 4."""
    base = {
        "index": 0, "row": 0, "column": 0, "x": 0, "y": 0, "w": 64, "h": 64,
        "pose": None, "pose_name": "rest", "yaw": 0.0, "frame": 0,
        "pivot_x": 32.0, "pivot_y": 64.0, "trim": None,
    }
    base.update(cell)
    return {
        "frame_size": 64, "columns": 1, "rows": 1, "id": "i", "name": "n",
        "source_job": "j", "elevation": 0.0, "lighting": "flat", "yaws": [0.0],
        "poses": [], "cells": [base],
    }


def _reduced(**cell) -> dict:
    return pixelsheet.pixel_sidecar(
        _render_meta(**cell),
        image="p.png",
        logical_size=16,
        palette=[],
        recipe={},
        created=0.0,
    )["cells"][0]


def test_a_sheet_with_no_slice_block_gets_none():
    assert "slices" not in _reduced()


def test_the_slice_block_reduces_exactly_on_divisible_sizes():
    entry = _reduced(
        slices=[
            {
                "name": "body",
                "bounds": {"x": 8, "y": 16, "w": 32, "h": 16},
                "pivot": {"x": 24.0, "y": 32.0},
                "center": {"x": 16, "y": 20, "w": 16, "h": 8},
            }
        ]
    )["slices"][0]
    assert entry["bounds"] == {"x": 2, "y": 4, "w": 8, "h": 4}
    assert entry["pivot"] == {"x": 6.0, "y": 8.0}
    assert entry["center"] == {"x": 4, "y": 5, "w": 4, "h": 2}
    assert entry["name"] == "body"


def test_an_indivisible_extent_ceils_rather_than_clipping_the_thing_it_describes():
    entry = _reduced(
        slices=[{"name": "x", "bounds": {"x": 5, "y": 5, "w": 6, "h": 6}}]
    )["slices"][0]
    assert entry["bounds"] == {"x": 1, "y": 1, "w": 2, "h": 2}
    assert entry["pivot"] is None and entry["center"] is None


def test_the_trim_rule_is_unchanged_by_the_refactor():
    """``reduced`` is now one helper serving trim, bounds and centre, so the
    rule trim already had is asserted here rather than assumed."""
    entry = _reduced(trim={"x": 5, "y": 5, "w": 6, "h": 6})
    assert entry["trim"] == {"x": 1, "y": 1, "w": 2, "h": 2}


def test_a_pixel_sheet_stays_plain_json():
    meta = pixelsheet.pixel_sidecar(
        _render_meta(slices=[{"name": "x", "bounds": {"x": 0, "y": 0, "w": 8, "h": 8}}]),
        image="p.png",
        logical_size=16,
        palette=[],
        recipe={},
        created=0.0,
    )
    assert json.loads(json.dumps(meta)) == meta
