"""The document, without a window.

The engine is pure for exactly this reason: every rule the editor has about
pixels -- a gap-free stroke, a fill that crosses an antialiased edge, an exact
selection cancel, a composite that matches what gets saved -- is a property of
an array, and asserting it against an array is the only way to know it holds.

Descended from the flat editor's suite. Where a test changed, it changed
because the model did: erasing now cuts alpha instead of painting a colour (the
colour moved to ``matte``, applied once at flatten), and history costs a dirty
rectangle instead of a whole canvas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from warlock.studio import inker
from warlock.studio.inker import ora

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
WHITE = (255, 255, 255, 255)


def _doc(size=(32, 32), colour=WHITE, budget=inker.UNDO_BYTES):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    pixels[:, :] = colour
    return inker.Document.from_pixels(pixels, budget=budget)


def _at(doc, x, y):
    return tuple(int(v) for v in doc.composite[y, x])


def _count(doc, colour):
    return int((doc.composite == np.array(colour, dtype=np.uint8)).all(axis=2).sum())


# --- loading ----------------------------------------------------------------


def test_a_document_loads_as_rgba_whatever_the_file_was(tmp_path: Path):
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    doc = inker.Document.load(path)
    assert doc.size == (8, 8)
    assert doc.composite.dtype == np.uint8
    assert _at(doc, 0, 0) == (10, 20, 30, 255)


def test_a_flat_png_opens_as_one_layer_not_as_a_blank_document(tmp_path: Path):
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    doc = inker.Document.load(path)
    assert len(doc.stack) == 1
    assert doc.stack[0].name == "Background"


def test_an_opaque_image_mattes_to_white_and_a_transparent_one_to_nothing():
    """The old ``erase_color`` question, moved to where it belongs. A photo is
    still a photo when it is saved; a sprite keeps the transparency trellis's
    background detection depends on."""
    opaque = np.full((4, 4, 4), 255, dtype=np.uint8)
    assert inker.matte_for(opaque) == inker.OPAQUE_WHITE
    matted = np.zeros((4, 4, 4), dtype=np.uint8)
    assert inker.matte_for(matted) is None


def test_png_bytes_round_trips_the_pixels():
    import io

    doc = _doc((4, 4), RED)
    with Image.open(io.BytesIO(doc.png_bytes())) as out:
        assert out.convert("RGBA").getpixel((0, 0)) == RED


def test_a_blank_document_starts_transparent_with_one_layer():
    doc = inker.Document.blank(8, 8)
    assert len(doc.stack) == 1
    assert doc.composite.max() == 0


# --- strokes ----------------------------------------------------------------


def test_a_fast_drag_leaves_no_gap_along_the_segment():
    """Spacing is walked, not sampled: a drag across the canvas in one frame
    must come out as a line rather than as two dots."""
    doc = _doc()
    doc.begin_stroke((2, 2), RED, size=3, hardness=1.0)
    doc.stroke_to((29, 29))
    doc.end_stroke()
    for t in range(4, 28):
        assert _at(doc, t, t)[0] > 200 and _at(doc, t, t)[2] < 60, t


def test_a_click_marks_rather_than_waiting_for_a_drag():
    doc = _doc()
    doc.begin_stroke((16, 16), RED, size=6, hardness=1.0)
    doc.end_stroke()
    assert _at(doc, 16, 16)[0] > 200


def test_a_stroke_bumps_the_revision_so_the_texture_resyncs():
    doc = _doc()
    before = doc.rev
    doc.begin_stroke((1, 1), RED, size=2)
    doc.stroke_to((5, 5))
    assert doc.rev > before


def test_a_stroke_off_the_canvas_is_clipped_rather_than_raising():
    doc = _doc()
    doc.begin_stroke((-40, -40), RED, size=4, hardness=1.0)
    doc.stroke_to((60, 60))
    doc.end_stroke()
    assert _count(doc, RED) > 0


def test_a_half_opacity_stroke_does_not_darken_where_it_crosses_itself():
    """Coverage accumulates with max, not with addition. Without that, every
    soft brush builds up wherever spacing put two dabs close together."""
    doc = _doc((32, 32), WHITE)
    doc.begin_stroke((4, 16), (0, 0, 0, 255), size=8, hardness=1.0, opacity=0.5)
    doc.stroke_to((28, 16))
    doc.stroke_to((4, 16))  # back over its own tracks
    doc.end_stroke()
    values = [_at(doc, x, 16)[0] for x in range(8, 25)]
    assert max(values) - min(values) <= 2
    assert 118 <= values[0] <= 138


def test_the_eraser_cuts_alpha_rather_than_painting_a_colour():
    doc = _doc((8, 8), RED)
    doc.begin_stroke((0, 0), RED, size=3, hardness=1.0, mode="erase")
    doc.stroke_to((7, 7))
    doc.end_stroke()
    assert _at(doc, 3, 3)[3] == 0


def test_a_stroke_is_one_undo_step_however_many_segments_it_had():
    doc = _doc((16, 16), WHITE)
    doc.begin_stroke((1, 1), RED, size=2)
    for i in range(2, 15):
        doc.stroke_to((i, i))
    doc.end_stroke()
    assert len(doc.history) == 1
    assert doc.undo()
    assert not doc.history.can_undo


def test_a_strokes_undo_costs_its_rectangle_not_the_canvas():
    doc = _doc((256, 256), WHITE)
    doc.begin_stroke((10, 10), RED, size=4)
    doc.stroke_to((20, 20))
    doc.end_stroke()
    assert doc.history.bytes < doc.stack[0].pixels.nbytes // 8


def test_symmetry_mirrors_the_stroke_across_the_canvas():
    doc = _doc((32, 32), WHITE)
    doc.begin_stroke((4, 8), RED, size=4, hardness=1.0, symmetry="x")
    doc.end_stroke()
    assert _at(doc, 4, 8)[0] > 200
    assert _at(doc, 27, 8)[0] > 200 and _at(doc, 27, 8)[2] < 60


def test_the_brush_size_is_clamped_at_both_ends():
    assert inker.clamp_brush(0) == inker.MIN_BRUSH
    assert inker.clamp_brush(9999) == inker.MAX_BRUSH


# --- fill -------------------------------------------------------------------


def test_a_fill_crosses_an_antialiased_edge_that_an_exact_match_would_stop_at():
    """Generated art has no flat regions, only nearly-flat ones."""
    doc = _doc((16, 16), WHITE)
    doc.stack[0].pixels[8, :] = (250, 250, 250, 255)
    doc.invalidate_all()
    doc.fill((0, 0), RED, thresh=32)
    assert _at(doc, 0, 15) == RED
    assert _at(doc, 0, 0) == RED


def test_a_tight_threshold_respects_a_hard_boundary():
    doc = _doc((16, 16), WHITE)
    doc.stack[0].pixels[8, :] = (0, 0, 0, 255)
    doc.invalidate_all()
    doc.fill((0, 0), RED, thresh=0)
    assert _at(doc, 0, 15) != RED


def test_a_fill_outside_the_canvas_does_nothing_at_all():
    doc = _doc()
    before = doc.rev
    assert not doc.fill((-1, -1), RED)
    assert doc.rev == before


def test_a_fill_stops_at_the_selection():
    doc = _doc((16, 16), WHITE)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 8, 16)))
    doc.fill((0, 0), RED)
    assert _at(doc, 2, 2) == RED
    assert _at(doc, 12, 2) == WHITE


def test_the_fill_and_the_wand_agree_about_what_similar_means():
    """One predicate, deliberately: a wand that selects more than the fill
    fills is a bug the user reads as randomness."""
    rng = np.random.default_rng(1)
    doc = inker.Document.from_pixels(rng.integers(0, 256, (24, 24, 4), dtype=np.uint8))
    region = inker.magic_wand(doc.composite, (5, 5), tolerance=90)
    before = doc.composite.copy()
    doc.fill((5, 5), RED, thresh=90)
    changed = (doc.composite != before).any(axis=2)
    assert np.array_equal(changed, region.mask >= 128)


def test_a_fill_reaches_a_disconnected_region_only_when_not_contiguous():
    """Regression: the pane's fill call site used to drop ``contiguous``
    entirely, so the wand-contiguous option was silently ignored by fill."""
    doc = _doc((16, 16), WHITE)
    doc.stack[0].pixels[8, :] = RED
    doc.invalidate_all()
    doc.fill((0, 0), BLUE, thresh=0, contiguous=True)
    assert _at(doc, 0, 0) == BLUE
    assert _at(doc, 0, 15) == WHITE

    doc = _doc((16, 16), WHITE)
    doc.stack[0].pixels[8, :] = RED
    doc.invalidate_all()
    doc.fill((0, 0), BLUE, thresh=0, contiguous=False)
    assert _at(doc, 0, 0) == BLUE
    assert _at(doc, 0, 15) == BLUE
    assert _at(doc, 0, 8) == RED


# --- shapes -----------------------------------------------------------------


@pytest.mark.parametrize("kind", inker.SHAPES)
def test_a_shape_touches_only_its_own_bounding_box(kind: str):
    doc = _doc()
    assert doc.shape(kind, (8, 8), (20, 20), RED, 1)
    for x, y in ((0, 0), (31, 31), (0, 31), (31, 0)):
        assert _at(doc, x, y) != RED, (kind, x, y)


def test_a_filled_rect_is_solid_and_an_outlined_one_is_not():
    solid = _doc()
    solid.shape("rect", (8, 8), (20, 20), RED, 1, filled=True)
    hollow = _doc()
    hollow.shape("rect", (8, 8), (20, 20), RED, 1, filled=False)
    assert _count(solid, RED) > _count(hollow, RED)
    assert _at(hollow, 14, 14) != RED


def test_an_unknown_shape_is_a_programming_error_not_a_silent_no_op():
    with pytest.raises(ValueError):
        _doc().shape("hexagon", (0, 0), (1, 1), RED, 1)


# --- eyedropper -------------------------------------------------------------


def test_the_eyedropper_reads_the_composite_not_the_active_layer():
    doc = _doc((8, 8), BLUE)
    doc.add_layer()
    doc.write_colour((0, 0, 8, 8), RED, np.ones((8, 8), dtype=np.float32))
    assert doc.eyedrop((4, 4)) == RED
    doc.stack[1].visible = False
    doc.invalidate_all()
    assert doc.eyedrop((4, 4)) == BLUE


def test_the_eyedropper_off_canvas_returns_nothing_rather_than_black():
    assert _doc().eyedrop((-1, 4)) is None
    assert _doc().eyedrop((4, 999)) is None


# --- selection and floating pixels ------------------------------------------


def test_a_lift_cuts_the_hole_it_floats_over():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (4, 4, 8, 8)))
    assert doc.lift()
    assert doc.floating.size == (4, 4)
    assert _at(doc, 5, 5)[3] == 0
    assert tuple(doc.floating.pixels[0, 0]) == RED


def test_a_lift_with_nothing_selected_is_refused():
    doc = _doc()
    assert not doc.lift()
    assert doc.floating is None


def test_lift_move_commit_puts_the_pixels_where_they_were_dragged():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 4, 4)))
    doc.lift()
    doc.move_floating(8, 8)
    assert doc.commit_floating()
    assert _at(doc, 9, 9) == RED
    assert _at(doc, 1, 1)[3] == 0
    assert doc.floating is None


def test_cancelling_a_float_restores_the_canvas_exactly():
    doc = _doc((16, 16), RED)
    before = doc.composite.copy()
    doc.select(inker.SelectionMask.from_rect((16, 16), (2, 2, 10, 10)))
    doc.lift()
    doc.move_floating(3, 3)
    assert doc.cancel_floating()
    assert np.array_equal(doc.composite, before)
    assert doc.floating is None


def test_deleting_a_float_leaves_the_hole_behind():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (2, 2, 6, 6)))
    doc.lift()
    assert doc.delete_floating()
    assert _at(doc, 3, 3)[3] == 0
    assert doc.floating is None


def test_deleting_a_selection_cuts_it_and_is_its_own_undo_step():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (4, 4, 8, 8)))
    assert doc.delete_selection()
    assert _at(doc, 5, 5)[3] == 0
    assert _at(doc, 12, 12) == RED
    assert doc.undo()
    assert _at(doc, 5, 5) == RED


def test_a_second_lift_commits_the_first_rather_than_dropping_it():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 4, 4)))
    doc.lift()
    doc.move_floating(8, 0)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 8, 4, 12)))
    doc.lift()
    assert _at(doc, 9, 1) == RED


def test_a_feathered_lift_floats_a_feathered_chunk():
    doc = _doc((32, 32), RED)
    doc.select(inker.SelectionMask.from_rect((32, 32), (8, 8, 24, 24)))
    doc.feather_selection(3.0)
    doc.lift()
    edge = doc.floating.pixels[..., 3]
    # A hard lift is 0 or 255 and nothing else; a feathered one has a ramp.
    assert int(edge.max()) >= 250
    assert 0 < int(np.median(edge[edge > 0])) <= 255
    assert np.count_nonzero((edge > 8) & (edge < 240)) > 20


def test_a_feathered_lift_partitions_alpha_exactly():
    """What floats away plus what stays behind is what was there.

    Both halves used to be computed independently -- one from the mask, one
    from 1 - the mask -- and both truncate toward zero, so wherever a * m / 255
    is not a whole number the two floors summed to a - 1. Every feathered lift
    therefore lost one level of alpha along its own edge, and a lift-and-drop
    that changed nothing else still eroded the outline a step per round trip.
    """
    doc = _doc((32, 32), RED)
    doc.select(inker.SelectionMask.from_rect((32, 32), (8, 8, 24, 24)))
    doc.feather_selection(3.0)
    layer = doc.stack.active
    before = layer.pixels[..., 3].astype(int).copy()

    doc.lift()

    x0, y0 = doc.floating.offset
    lifted = doc.floating.pixels[..., 3].astype(int)
    h, w = lifted.shape
    cut = layer.pixels[y0 : y0 + h, x0 : x0 + w, 3].astype(int)
    assert np.array_equal(lifted + cut, before[y0 : y0 + h, x0 : x0 + w])
    # And the ramp is genuinely partial somewhere, or the assertion above is
    # only about hard 0/255 pixels and would have passed before the fix.
    assert np.count_nonzero((lifted > 8) & (lifted < 240)) > 20


def test_select_all_and_deselect_are_a_round_trip():
    doc = _doc((8, 8))
    doc.select_all()
    assert doc.mask is not None and not doc.mask.is_empty
    doc.deselect()
    assert doc.mask is None


def test_inverting_an_empty_selection_selects_everything():
    doc = _doc((8, 8))
    doc.invert_selection()
    assert doc.mask is not None
    assert int(doc.mask.mask.min()) == 255


# --- clipboard --------------------------------------------------------------


def test_copy_and_paste_bring_the_pixels_back_as_a_floating_buffer():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 4, 4)))
    assert doc.copy()
    assert doc.paste((8, 8))
    assert doc.floating.offset == (8, 8)
    doc.commit_floating()
    assert _at(doc, 9, 9) == RED


def test_a_cut_copies_before_it_deletes():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 4, 4)))
    assert doc.cut()
    assert _at(doc, 1, 1)[3] == 0
    assert doc.paste((8, 8))
    doc.commit_floating()
    assert _at(doc, 9, 9) == RED


def test_an_ordinary_paste_carries_the_selection_shape_not_its_bounding_box():
    """``paste`` floats what it takes verbatim and ``commit_floating``
    composites without consulting a mask, so a rectangular clipboard crop
    pasted the whole bounding box of a lasso or ellipse -- while hit-testing,
    which does read the mask, only let the user grab the shape."""
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_ellipse((16, 16), (0, 0, 8, 8)))
    assert doc.copy()
    landed = doc.add_layer()  # onto empty pixels, so the paste is what is seen
    assert doc.paste((8, 8))
    doc.commit_floating()

    # The ellipse's centre landed; its bounding box's corner did not.
    assert int(landed.pixels[12, 12, 3]) == 255
    assert int(landed.pixels[8, 8, 3]) == 0


def test_a_feathered_copy_is_not_attenuated_twice_by_paste_as_layer():
    """The clipboard's pixels carry their mask, so the layer paste must not
    apply it a second time."""
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (4, 4, 12, 12)).feathered(2.0))
    assert doc.copy()
    taken = doc.clipboard.take()  # a copy; the clipboard is not cleared
    assert taken is not None
    assert doc.paste_as_layer()
    height, width = taken[0].shape[:2]
    placed = doc.stack.active.pixels[:height, :width]  # resize_canvas anchors top-left
    assert np.array_equal(placed[..., 3], taken[0][..., 3])
    alpha = taken[0][..., 3]
    assert ((alpha > 0) & (alpha < 255)).any()  # the feather is really there


def test_pasting_an_empty_clipboard_does_nothing():
    assert not _doc().paste()


# --- layers -----------------------------------------------------------------


def test_a_new_layer_goes_above_the_active_one_and_becomes_active():
    doc = _doc((8, 8), RED)
    doc.add_layer()
    assert len(doc.stack) == 2
    assert doc.stack.active_index == 1
    assert _at(doc, 0, 0) == RED  # a transparent layer hides nothing


def test_the_last_layer_cannot_be_removed():
    doc = _doc()
    assert not doc.remove_layer()


def test_layer_opacity_shows_up_in_the_composite_and_undoes():
    doc = _doc((8, 8), (0, 0, 0, 255))
    doc.add_layer()
    doc.write_colour((0, 0, 8, 8), WHITE, np.ones((8, 8), dtype=np.float32))
    doc.set_layer_props(opacity=0.5)
    assert 120 <= _at(doc, 0, 0)[0] <= 136
    assert doc.undo()
    assert _at(doc, 0, 0) == WHITE


def test_merging_down_keeps_what_the_composite_showed():
    doc = _doc((8, 8), (0, 0, 0, 255))
    doc.add_layer()
    doc.write_colour((0, 0, 8, 8), WHITE, np.ones((8, 8), dtype=np.float32))
    doc.set_layer_props(opacity=0.25)
    before = doc.composite.copy()
    assert doc.merge_down()
    assert len(doc.stack) == 1
    assert np.allclose(doc.composite.astype(int), before.astype(int), atol=1)


def test_merging_down_is_one_undo_step():
    doc = _doc((8, 8), RED)
    doc.add_layer()
    before = len(doc.history)
    doc.merge_down()
    assert len(doc.history) == before + 1
    assert doc.undo()
    assert len(doc.stack) == 2


def test_a_reorder_undoes_back_to_the_order_it_started_in():
    doc = _doc((8, 8), RED)
    doc.add_layer()
    uids = [layer.uid for layer in doc.stack]
    assert doc.move_layer(1, 0)
    assert [layer.uid for layer in doc.stack] == [uids[1], uids[0]]
    assert doc.undo()
    assert [layer.uid for layer in doc.stack] == uids


def test_flattening_collapses_the_stack_and_undoes_as_one_step():
    doc = _doc((8, 8), (0, 0, 0, 255))
    doc.add_layer()
    doc.write_colour((0, 0, 4, 4), RED, np.ones((4, 4), dtype=np.float32))
    doc.flatten_layers()
    assert len(doc.stack) == 1
    assert _at(doc, 1, 1) == RED
    assert doc.undo()
    assert len(doc.stack) == 2


def test_flattening_commits_a_floating_buffer_rather_than_stranding_it():
    """A live float names a layer uid; flatten replaces the whole stack, so an
    uncommitted buffer left the next undo raising KeyError out of by_uid and
    dropped the lifted pixels."""
    doc = _doc((16, 16), RED)
    doc.add_layer()
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 4, 4)))
    doc.lift()
    doc.flatten_layers()
    assert doc.floating is None
    assert len(doc.stack) == 1
    assert doc.undo()  # must not raise; reverses the flatten


# --- geometry ---------------------------------------------------------------


def test_a_flip_moves_the_pixels_and_undoes_exactly():
    doc = _doc((8, 4), WHITE)
    doc.stack[0].pixels[:, 0] = RED
    doc.invalidate_all()
    before = doc.composite.copy()
    doc.flip("horizontal")
    assert _at(doc, 7, 0) == RED
    assert doc.undo()
    assert np.array_equal(doc.composite, before)


def test_a_redo_of_a_canvas_op_replays_without_pushing_a_second_step():
    doc = _doc((8, 4), WHITE)
    doc.stack[0].pixels[:, 0] = RED
    doc.invalidate_all()
    doc.flip("horizontal")
    doc.undo()
    assert doc.redo()
    assert _at(doc, 7, 0) == RED
    assert len(doc.history) == 1


def test_a_quarter_turn_swaps_the_canvas_dimensions():
    doc = _doc((8, 4))
    doc.rotate90()
    assert doc.size == (4, 8)
    assert doc.composite.shape[:2] == (8, 4)


def test_scaling_resizes_every_layer_together():
    doc = _doc((8, 8), RED)
    doc.add_layer()
    doc.scale((16, 16))
    assert doc.size == (16, 16)
    assert all(layer.size == (16, 16) for layer in doc.stack)


def test_cropping_to_the_selection_keeps_what_was_inside_it():
    doc = _doc((16, 16), WHITE)
    doc.stack[0].pixels[4:8, 4:8] = RED
    doc.invalidate_all()
    doc.select(inker.SelectionMask.from_rect((16, 16), (4, 4, 8, 8)))
    assert doc.crop_to_selection()
    assert doc.size == (4, 4)
    assert _at(doc, 0, 0) == RED


def test_growing_the_canvas_adds_transparency_rather_than_stretching():
    doc = _doc((4, 4), RED)
    doc.resize_canvas((8, 8))
    assert doc.size == (8, 8)
    assert _at(doc, 0, 0) == RED
    assert _at(doc, 7, 7)[3] == 0


# --- the composite cache ----------------------------------------------------


def test_the_cached_composite_matches_a_composite_from_scratch():
    """The cache is the thing that can silently be wrong: the editor would show
    one image and save another, and no test of either alone would catch it."""
    rng = np.random.default_rng(11)
    layers = [
        inker.Layer(
            pixels=rng.integers(0, 256, (16, 16, 4), dtype=np.uint8),
            opacity=0.4 + 0.2 * i,
            blend=inker.BLEND_MODES[i % len(inker.BLEND_MODES)],
        )
        for i in range(3)
    ]
    doc = inker.Document(stack=inker.LayerStack(layers, 1))
    doc.begin_stroke((4, 4), RED, size=6)
    doc.stroke_to((12, 12))
    doc.end_stroke()
    assert np.array_equal(doc.composite, doc.stack.flatten())


def test_a_stroke_reports_only_the_rectangle_it_touched_as_dirty():
    doc = _doc((64, 64), WHITE)
    doc.take_dirty()
    doc.begin_stroke((10, 10), RED, size=4)
    doc.stroke_to((20, 20))
    rect = doc.take_dirty()
    assert rect is not None
    x0, y0, x1, y1 = rect
    assert 0 <= x0 <= 8 and 0 <= y0 <= 8 and x1 <= 24 and y1 <= 24


def test_a_structural_change_asks_for_a_full_upload():
    doc = _doc((16, 16), WHITE)
    doc.take_dirty()
    doc.add_layer()
    assert doc.take_dirty() is None


def test_taking_the_dirty_rectangle_clears_it():
    doc = _doc((16, 16), WHITE)
    doc.take_dirty()
    doc.write_colour((0, 0, 4, 4), RED, np.ones((4, 4), dtype=np.float32))
    assert doc.take_dirty() is not None
    assert doc.take_dirty() is None


# --- history ----------------------------------------------------------------


def test_undo_returns_the_previous_pixels_and_redo_puts_them_back():
    doc = _doc((8, 8), RED)
    doc.begin_stroke((0, 0), BLUE, size=1, hardness=1.0)
    doc.stroke_to((7, 7))
    doc.end_stroke()
    assert _at(doc, 3, 3)[2] > 200
    assert doc.undo()
    assert _at(doc, 3, 3) == RED
    assert doc.redo()
    assert _at(doc, 3, 3)[2] > 200


def test_undo_with_nothing_to_undo_reports_that_it_did_nothing():
    doc = _doc()
    assert not doc.undo()
    assert not doc.redo()


def test_a_new_edit_discards_the_redo_branch():
    doc = _doc((8, 8), RED)
    doc.write_colour((0, 0, 4, 4), BLUE, np.ones((4, 4), dtype=np.float32))
    doc.undo()
    assert doc.history.can_redo
    doc.write_colour((0, 0, 4, 4), BLUE, np.ones((4, 4), dtype=np.float32))
    assert not doc.history.can_redo


def test_a_write_that_changes_nothing_is_not_an_undo_step():
    """Clicking fill on an already-filled region must not consume a Ctrl+Z."""
    doc = _doc((8, 8), RED)
    doc.fill((0, 0), RED)
    assert not doc.history.can_undo


# --- gradients --------------------------------------------------------------


def test_a_linear_gradient_runs_from_one_colour_to_the_other():
    doc = inker.Document.blank(16, 4)
    doc.gradient((0, 0), (15, 0), (0, 0, 0, 255), WHITE)
    assert _at(doc, 0, 0)[0] < 20
    assert _at(doc, 15, 0)[0] > 235
    assert _at(doc, 0, 0)[0] < _at(doc, 8, 0)[0] < _at(doc, 15, 0)[0]


def test_a_gradient_to_transparent_fades_out_rather_than_to_black():
    doc = inker.Document.blank(16, 4)
    doc.gradient((0, 0), (15, 0), RED, (255, 0, 0, 0))
    near = _at(doc, 15, 0)
    assert near[3] < 20
    assert _at(doc, 0, 0) == RED


def test_a_radial_gradient_is_brightest_at_its_centre():
    doc = inker.Document.blank(32, 32)
    doc.gradient((16, 16), (16, 0), WHITE, (0, 0, 0, 255))
    assert _at(doc, 16, 16)[0] > _at(doc, 16, 4)[0]


def test_a_gradient_fills_only_the_selection():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 8, 16)))
    doc.gradient((0, 0), (15, 0), BLUE, BLUE)
    assert _at(doc, 1, 1) == BLUE
    assert _at(doc, 12, 1) == RED


# --- free transform ---------------------------------------------------------


def test_a_transform_with_no_selection_lifts_the_whole_layer():
    """"Rotate this layer" is the common case; requiring a select-all first
    would be busywork."""
    doc = _doc((16, 16), RED)
    assert doc.begin_transform()
    assert doc.floating is not None
    assert doc.floating.size == (16, 16)


def test_a_transform_lifts_only_the_selection_when_there_is_one():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (4, 4, 12, 12)))
    doc.begin_transform()
    assert doc.floating.size == (8, 8)


def test_scaling_a_floating_buffer_keeps_its_centre_where_it_was():
    """Scaling about the corner sends the subject off across the canvas, which
    is not what grabbing a handle means."""
    doc = _doc((64, 64), RED)
    doc.select(inker.SelectionMask.from_rect((64, 64), (16, 16, 32, 32)))
    doc.begin_transform()
    before = doc.floating.centre
    doc.transform_floating(scale=(2.0, 2.0))
    assert doc.floating.size == (32, 32)
    assert doc.floating.centre == pytest.approx(before, abs=1.0)


def test_rotating_a_floating_buffer_grows_it_and_keeps_its_centre():
    doc = _doc((64, 64), RED)
    doc.select(inker.SelectionMask.from_rect((64, 64), (16, 16, 48, 48)))
    doc.begin_transform()
    before = doc.floating.centre
    doc.rotate_floating(45.0)
    assert doc.floating.size[0] > 32
    assert doc.floating.centre == pytest.approx(before, abs=1.5)


def test_repeated_adjustments_re_render_from_the_lift_rather_than_compounding():
    """A chain of resamples turns a handle-drag into mush; each adjustment has
    to start again from the pixels that were lifted."""
    doc = _doc((64, 64), RED)
    doc.select(inker.SelectionMask.from_rect((64, 64), (16, 16, 48, 48)))
    doc.begin_transform()
    for scale in (2.0, 0.5, 3.0, 1.0):
        doc.transform_floating(scale=(scale, scale))
    assert doc.floating.size == (32, 32)
    assert np.array_equal(doc.floating.pixels, doc.floating.source)


def test_a_transform_bumps_the_buffers_revision_because_the_pixels_changed():
    """Unlike a move, which changes only where they are drawn."""
    doc = _doc((32, 32), RED)
    doc.begin_transform()
    rev = doc.floating.rev
    doc.move_floating(3, 3)
    assert doc.floating.rev == rev
    doc.transform_floating(scale=(1.5, 1.5))
    assert doc.floating.rev > rev


def test_flipping_a_floating_buffer_survives_a_later_rotation():
    """The flip is applied to the source, so re-rendering cannot undo it."""
    doc = inker.Document.blank(32, 32)
    doc.stack[0].pixels[:, :16] = RED
    doc.invalidate_all()
    doc.begin_transform()
    doc.flip_floating("horizontal")
    assert tuple(doc.floating.pixels[16, 24]) == RED
    doc.rotate_floating(90.0)
    doc.transform_floating(angle=0.0)
    assert tuple(doc.floating.pixels[16, 24]) == RED


def test_a_transform_commits_as_one_undo_step():
    doc = _doc((32, 32), RED)
    doc.select(inker.SelectionMask.from_rect((32, 32), (8, 8, 24, 24)))
    doc.begin_transform()
    steps = len(doc.history)
    doc.transform_floating(scale=(0.5, 0.5))
    doc.commit_floating()
    assert len(doc.history) == steps + 1


def test_cancelling_a_transform_puts_the_pixels_back():
    doc = _doc((32, 32), RED)
    before = doc.composite.copy()
    doc.begin_transform()
    doc.rotate_floating(30.0)
    doc.cancel_floating()
    assert np.array_equal(doc.composite, before)


def test_transforming_nothing_reports_that_it_did_nothing():
    doc = _doc()
    assert not doc.transform_floating(angle=10.0)
    assert not doc.rotate_floating(10.0)
    assert not doc.flip_floating("horizontal")


# --- pasting ----------------------------------------------------------------


def test_pasting_as_a_layer_lands_immediately_rather_than_floating():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 4, 4)))
    doc.copy()
    assert doc.paste_as_layer()
    assert doc.floating is None
    assert len(doc.stack) == 2
    assert _at(doc, 1, 1) == RED


def test_a_pasted_layer_is_undone_by_removing_it():
    doc = _doc((16, 16), RED)
    doc.select_all()
    doc.copy()
    doc.paste_as_layer()
    assert doc.undo()
    assert len(doc.stack) == 1


def test_a_pasted_layer_keeps_only_what_was_selected():
    doc = _doc((16, 16), RED)
    doc.select(inker.SelectionMask.from_rect((16, 16), (0, 0, 4, 4)))
    doc.copy()
    doc.paste_as_layer()
    assert _at(doc, 1, 1) == RED
    assert int(doc.stack[1].pixels[8, 8][3]) == 0


def test_an_image_from_outside_can_be_put_on_the_clipboard():
    """Everything pasted carries a mask; an image from elsewhere is fully
    selected by definition."""
    doc = _doc((16, 16), RED)
    incoming = np.zeros((4, 4, 4), dtype=np.uint8)
    incoming[:, :] = BLUE
    doc.put_clipboard(incoming)
    assert doc.paste_as_layer()
    assert _at(doc, 1, 1) == BLUE


def test_pasting_a_layer_from_an_empty_clipboard_does_nothing():
    assert not _doc().paste_as_layer()


def test_a_paste_bigger_than_the_canvas_is_cropped_rather_than_refused():
    doc = _doc((8, 8), RED)
    doc.put_clipboard(np.full((32, 32, 4), 255, dtype=np.uint8))
    assert doc.paste_as_layer()
    assert doc.stack[1].size == (8, 8)


# --- alpha lock and the layer eyedropper (Ink3) ------------------------------
#
# "Preserve transparency" is exactly *the alpha does not change*, which is why
# every write path restores the channel rather than pre-multiplying the weight
# by it: a weight-based version still raises the alpha of a half-transparent
# pixel, and the whole point is painting inside a shape without growing it.


def _dot(doc, x=4, y=4, colour=(255, 0, 0, 255)) -> None:
    doc.stack.active.pixels[y, x] = colour


def test_a_locked_layer_keeps_its_alpha_under_a_flat_write():
    doc = inker.Document.blank(8, 8)
    _dot(doc)
    doc.stack.active.alpha_lock = True
    before = doc.stack.active.pixels[..., 3].copy()

    doc.write_colour((0, 0, 8, 8), (0, 0, 255, 255), np.ones((8, 8), dtype=np.float32))

    after = doc.stack.active.pixels
    assert np.array_equal(after[..., 3], before), "the write must not grow the shape"
    assert tuple(after[4, 4]) == (0, 0, 255, 255), "and must still recolour what is there"
    # Outside the shape the colour channels *do* take the paint and the alpha
    # stays zero, so the pixel is invisible -- which is what makes restoring the
    # one channel enough on its own. Asserting the alpha is the honest bar;
    # asserting the RGB would be asserting an implementation detail of
    # ``paint_colour`` that nothing can see.
    assert int(after[0, 0, 3]) == 0


def test_an_unlocked_layer_is_untouched_by_any_of_this():
    doc = inker.Document.blank(8, 8)
    _dot(doc)
    doc.write_colour((0, 0, 8, 8), (0, 0, 255, 255), np.ones((8, 8), dtype=np.float32))
    assert int(doc.stack.active.pixels[0, 0, 3]) == 255


def test_a_locked_layer_keeps_its_alpha_under_a_stroke():
    doc = inker.Document.blank(16, 16)
    doc.stack.active.pixels[6:10, 6:10] = (255, 0, 0, 255)
    doc.stack.active.alpha_lock = True
    before = doc.stack.active.pixels[..., 3].copy()

    doc.begin_stroke((8.0, 8.0), (0, 0, 255, 255), size=12, hardness=1.0)
    doc.stroke_to((12.0, 12.0))
    doc.end_stroke()

    after = doc.stack.active.pixels
    assert np.array_equal(after[..., 3], before)
    assert tuple(after[8, 8])[:3] == (0, 0, 255)


def test_the_eraser_does_nothing_on_a_locked_layer():
    """The correct reading rather than a gap: erasing *is* changing alpha."""
    doc = inker.Document.blank(16, 16)
    doc.stack.active.pixels[:, :] = (255, 0, 0, 255)
    doc.stack.active.alpha_lock = True
    keep = doc.stack.active.pixels.copy()

    doc.begin_stroke((8.0, 8.0), (0, 0, 0, 255), size=8, hardness=1.0, mode="erase")
    doc.end_stroke()
    assert np.array_equal(doc.stack.active.pixels, keep)


def test_the_eyedropper_reads_the_composite_by_default_and_one_layer_on_ask():
    doc = inker.Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    top = doc.add_layer("Top")
    top.pixels[:, :] = (0, 0, 255, 255)
    top.opacity = 0.5
    doc.invalidate_all()

    blended = doc.eyedrop((1, 1))
    assert blended is not None and blended[:3] != (0, 0, 255)
    assert doc.eyedrop((1, 1), layer_only=True) == (0, 0, 255, 255)


def test_a_layer_only_pick_ignores_opacity_blend_and_visibility():
    """All three are about how a layer is *shown*, not what is painted in it."""
    doc = inker.Document.blank(4, 4)
    top = doc.add_layer("Top")
    top.pixels[:, :] = (10, 20, 30, 200)
    top.opacity = 0.1
    top.blend = "multiply"
    top.visible = False
    doc.invalidate_all()
    assert doc.eyedrop((0, 0), layer_only=True) == (10, 20, 30, 200)


def test_the_lock_travels_with_a_layer_copy_and_survives_an_ora_round_trip(tmp_path):
    doc = inker.Document.blank(8, 8)
    doc.stack.active.alpha_lock = True
    assert doc.stack.active.copy().alpha_lock is True

    path = tmp_path / "locked.ora"
    inker.write_ora(doc, path)
    assert inker.Document.load(path).stack[0].alpha_lock is True


def test_an_unlocked_document_writes_no_lock_attribute(tmp_path):
    """Written only when set, so a document nobody has locked anything in
    produces the XML this build produced before the attribute existed."""
    import zipfile
    from xml.etree import ElementTree

    path = tmp_path / "plain.ora"
    inker.write_ora(inker.Document.blank(8, 8), path)
    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    assert all(ora.LOCK_ATTR not in element.attrib for element in root.iter("layer"))


# --- the selection-op wrappers ----------------------------------------------
#
# Thin by design -- each is a guard, a call into ``SelectionMask`` and a
# ``select`` -- and the two decisions that live in the wrapper rather than in
# the mask are exactly what a thin function hides: whether the step is undoable
# and whether ``op`` is honoured. The mask's own arithmetic is asserted in
# ``test_selection.py``; what is asserted here is the plumbing around it.


def _selected(doc):
    return 0 if doc.mask is None else int((doc.mask.mask > 0).sum())


def test_grow_and_shrink_move_the_selection_edge_and_are_undoable():
    doc = _doc((32, 32))
    doc.select(inker.SelectionMask.from_rect((32, 32), (12, 12, 20, 20)))
    base = _selected(doc)

    doc.grow_selection(2)
    assert _selected(doc) > base
    doc.undo()
    assert _selected(doc) == base

    doc.shrink_selection(2)
    assert _selected(doc) < base
    doc.undo()
    assert _selected(doc) == base


def test_border_selection_keeps_the_rim_and_drops_the_middle():
    doc = _doc((32, 32))
    doc.select(inker.SelectionMask.from_rect((32, 32), (8, 8, 24, 24)))

    doc.border_selection(2)

    assert doc.mask is not None
    # The centre is no longer in it; a pixel on the old edge still is.
    assert doc.mask.mask[16, 16] == 0
    assert doc.mask.mask[8, 16] > 0


def test_the_three_edge_ops_do_nothing_without_a_selection():
    """Each is guarded on ``self.mask``, so none of them may invent one -- and
    none may spend an undo step saying so."""
    for call in ("grow_selection", "shrink_selection", "border_selection"):
        doc = _doc((16, 16))
        head = doc.history.head
        getattr(doc, call)(2)
        assert doc.mask is None, call
        assert doc.history.head == head, call


def test_the_wand_honours_its_op_rather_than_always_replacing():
    """``select_wand`` forwards ``op`` to ``select``. It is the one selection
    entry point where a caller can hold a modifier, so a wrapper that dropped
    the argument would silently turn every Shift-click into a fresh
    selection."""
    doc = _doc((16, 16), RED)
    doc.stack.active.pixels[0:8, 0:8] = BLUE
    doc.invalidate((0, 0, 16, 16))

    doc.select_wand((2, 2), tolerance=8)
    blue_only = _selected(doc)
    assert blue_only == 64

    doc.select_wand((12, 12), tolerance=8, op="add")
    assert _selected(doc) > blue_only


def test_the_wand_reads_the_composite_and_a_single_undo_reverses_it():
    doc = _doc((16, 16), RED)
    doc.select_wand((8, 8), tolerance=8)
    assert _selected(doc) == 16 * 16

    doc.undo()
    assert doc.mask is None


def test_no_private_cache_field_is_a_constructor_parameter():
    """``Document(...)`` is public. Five of the caches -- the frame stamps, the
    frame cache and its order, the dropped-frame list and the layer stamps --
    were declared without ``init=False``, so they were keyword parameters on it:
    a caller could hand a document somebody else's cache, and the signature
    advertised private state as part of the shape."""
    import dataclasses

    from warlock.studio.inker.document import Document

    on_init = [
        f.name for f in dataclasses.fields(Document) if f.name.startswith("_") and f.init
    ]
    assert on_init == []


def test_the_undo_push_sites_all_live_on_the_document_class():
    """The module docstring used to say this module is the only one that pushes
    onto the undo stack. It is the only *class* that does -- the ten ``_doc_*``
    mixins are part of ``Document`` and push at sixty sites -- and the claim as
    written stopped being true when they were split out."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio" / "inker"
    pushers = sorted(
        path.name
        for path in root.glob("*.py")
        if "history.push(" in path.read_text(encoding="utf-8")
    )
    assert pushers, "the scan has to find something or it proves nothing"
    for name in pushers:
        assert name == "document.py" or name.startswith("_doc_"), name


def test_the_engine_states_its_invariants_without_assert():
    """``assert`` is compiled out under ``python -O``, so a guard written that
    way vanishes and the failure it was catching becomes an ``AttributeError``
    on ``None`` several frames deep inside a stroke. Nothing ships optimised
    today; the installer programme is exactly the kind of change that starts."""
    import pathlib

    root = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "warlock"
        / "studio"
        / "inker"
    )
    offenders = []
    for path in sorted(root.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("assert "):
                offenders.append(f"{path.name}:{number}")
    assert offenders == [], offenders


def test_a_grid_hook_reached_without_a_grid_refuses():
    import pytest

    from warlock.studio import inker

    doc = inker.Document.blank(4, 4)
    assert doc.anim is None
    with pytest.raises(ValueError):
        doc._require_anim()


def test_require_anim_hands_back_the_grid():
    from warlock.studio import inker

    doc = inker.Document.blank(4, 4)
    anim = doc.ensure_animation()
    assert doc._require_anim() is anim
