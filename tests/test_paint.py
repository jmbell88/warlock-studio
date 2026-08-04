"""The 2D editor's document model, without a window.

paint.py is pure Pillow for exactly this reason: every rule the editor has
about pixels -- gap-free strokes, a fill that crosses an antialiased edge, an
exact selection cancel -- is a property of an image, and asserting it against
an image is the only way to know it holds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from warlock.studio import paint

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(size=(32, 32), colour=(255, 255, 255, 255), budget=paint.UNDO_BYTES):
    image = Image.new("RGBA", size, colour)
    return paint.Document(
        image=image,
        erase_color=paint.erase_color_for(image),
        history=paint.UndoStack(budget),
    )


def _pixels(doc, colour):
    counts = doc.image.getcolors(1 << 20) or []
    return sum(n for n, value in counts if value == colour)


# --- loading ----------------------------------------------------------------


def test_a_document_loads_as_rgba_whatever_the_file_was(tmp_path: Path):
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    doc = paint.Document.load(path)
    assert doc.image.mode == "RGBA"
    assert doc.size == (8, 8)


def test_an_opaque_image_erases_to_white_and_a_matted_one_to_nothing():
    """Erasing a sprite to white would break trellis's background detection;
    erasing a photo to transparent would change what the file means."""
    opaque = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
    assert paint.erase_color_for(opaque) == paint.OPAQUE_WHITE
    matted = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert paint.erase_color_for(matted) == paint.TRANSPARENT


def test_png_bytes_round_trips_the_pixels():
    import io

    doc = _doc((4, 4), RED)
    with Image.open(io.BytesIO(doc.png_bytes())) as out:
        assert out.convert("RGBA").getpixel((0, 0)) == RED


# --- strokes ----------------------------------------------------------------


def test_a_fast_drag_leaves_no_gap_along_the_segment():
    doc = _doc()
    doc.stroke((2, 2), (29, 29), RED, 3)
    # Sampled along the diagonal: a mitre-gapped line is dotted here.
    for t in range(3, 29):
        assert doc.image.getpixel((t, t)) == RED, t


def test_a_stroke_bumps_the_revision_so_the_texture_resyncs():
    doc = _doc()
    before = doc.rev
    doc.stroke((1, 1), (5, 5), RED, 2)
    assert doc.rev == before + 1


def test_a_stroke_off_the_canvas_is_clipped_rather_than_raising():
    doc = _doc()
    doc.stroke((-40, -40), (60, 60), RED, 4)
    assert _pixels(doc, RED) > 0


def test_the_eraser_paints_the_documents_erase_colour_not_the_tool_colour():
    doc = _doc((8, 8), (0, 0, 0, 0))
    doc.stroke((0, 0), (7, 7), RED, 1, erase=True)
    assert doc.image.getpixel((3, 3)) == paint.TRANSPARENT


def test_the_brush_size_is_clamped_at_both_ends():
    assert paint.clamp_brush(0) == paint.MIN_BRUSH
    assert paint.clamp_brush(999) == paint.MAX_BRUSH


# --- fill -------------------------------------------------------------------


def test_a_fill_crosses_an_antialiased_edge_that_an_exact_match_would_stop_at():
    """Generated art has no flat regions, only nearly-flat ones."""
    doc = _doc((16, 16), (255, 255, 255, 255))
    for x in range(16):
        # A one-pixel band a few levels off white: invisible, and an exact
        # fill dies on it.
        doc.image.putpixel((x, 8), (250, 250, 250, 255))
    doc.fill((0, 0), RED, thresh=32)
    assert doc.image.getpixel((0, 15)) == RED
    assert doc.image.getpixel((0, 0)) == RED


def test_a_tight_threshold_respects_a_hard_boundary():
    doc = _doc((16, 16), (255, 255, 255, 255))
    for x in range(16):
        doc.image.putpixel((x, 8), (0, 0, 0, 255))
    doc.fill((0, 0), RED, thresh=0)
    assert doc.image.getpixel((0, 15)) != RED


def test_a_fill_outside_the_canvas_does_nothing_at_all():
    doc = _doc()
    before = doc.rev
    doc.fill((-1, -1), RED)
    assert doc.rev == before


# --- shapes -----------------------------------------------------------------


@pytest.mark.parametrize("kind", paint.SHAPES)
def test_a_shape_touches_only_its_own_bounding_box(kind: str):
    doc = _doc()
    doc.shape(kind, (8, 8), (20, 20), RED, 1)
    assert doc.rev == 1
    for x, y in ((0, 0), (31, 31), (0, 31), (31, 0)):
        assert doc.image.getpixel((x, y)) != RED, (kind, x, y)
    assert _pixels(doc, RED) > 0


def test_a_filled_rect_is_solid_and_an_outlined_one_is_not():
    solid = _doc()
    solid.shape("rect", (8, 8), (20, 20), RED, 1, filled=True)
    hollow = _doc()
    hollow.shape("rect", (8, 8), (20, 20), RED, 1, filled=False)
    assert _pixels(solid, RED) > _pixels(hollow, RED)
    assert hollow.image.getpixel((14, 14)) != RED


def test_an_unknown_shape_is_a_programming_error_not_a_silent_no_op():
    with pytest.raises(ValueError):
        _doc().shape("hexagon", (0, 0), (1, 1), RED, 1)


# --- eyedropper -------------------------------------------------------------


def test_the_eyedropper_reads_the_pixel_under_it():
    doc = _doc((8, 8), BLUE)
    assert doc.eyedrop((4, 4)) == BLUE


def test_the_eyedropper_off_canvas_returns_nothing_rather_than_black():
    assert _doc().eyedrop((-1, 4)) is None
    assert _doc().eyedrop((4, 999)) is None


# --- selection --------------------------------------------------------------


def test_a_lift_cuts_the_hole_it_floats_over():
    doc = _doc((16, 16), RED)
    assert doc.lift_selection((4, 4, 8, 8))
    assert doc.selection.size == (4, 4)
    assert doc.image.getpixel((5, 5)) == doc.erase_color
    assert doc.selection.chunk.getpixel((0, 0)) == RED


def test_a_lift_of_an_empty_rectangle_is_refused():
    doc = _doc()
    assert not doc.lift_selection((5, 5, 5, 5))
    assert doc.selection is None


def test_a_lift_is_clipped_to_the_canvas():
    doc = _doc((8, 8), RED)
    assert doc.lift_selection((-10, -10, 4, 4))
    assert doc.selection.size == (4, 4)


def test_lift_move_commit_puts_the_pixels_where_they_were_dragged():
    doc = _doc((16, 16), RED)
    doc.lift_selection((0, 0, 4, 4))
    doc.move_selection(8, 8)
    assert doc.commit_selection()
    assert doc.image.getpixel((9, 9)) == RED
    assert doc.image.getpixel((1, 1)) == doc.erase_color
    assert doc.selection is None


def test_cancelling_a_selection_restores_the_canvas_exactly():
    doc = _doc((16, 16), RED)
    before = doc.image.tobytes()
    doc.lift_selection((2, 2, 10, 10))
    doc.move_selection(3, 3)
    assert doc.cancel_selection()
    assert doc.image.tobytes() == before
    assert doc.selection is None
    # And the lift's own snapshot is gone, so the next Ctrl+Z is not a no-op.
    assert not doc.history.can_undo


def test_deleting_a_selection_leaves_the_hole_behind():
    doc = _doc((16, 16), RED)
    doc.lift_selection((2, 2, 6, 6))
    assert doc.delete_selection()
    assert doc.image.getpixel((3, 3)) == doc.erase_color
    assert doc.selection is None


def test_delete_rect_erases_and_is_its_own_undo_step():
    doc = _doc((16, 16), RED)
    doc.delete_rect((4, 4, 8, 8))
    assert doc.image.getpixel((5, 5)) == doc.erase_color
    assert doc.image.getpixel((12, 12)) == RED
    assert doc.undo()
    assert doc.image.getpixel((5, 5)) == RED


def test_a_second_lift_commits_the_first_rather_than_dropping_it():
    doc = _doc((16, 16), RED)
    doc.lift_selection((0, 0, 4, 4))
    doc.move_selection(8, 0)
    doc.lift_selection((0, 8, 4, 12))
    assert doc.image.getpixel((9, 1)) == RED


# --- history ----------------------------------------------------------------


def test_undo_returns_the_previous_pixels_and_redo_puts_them_back():
    doc = _doc((8, 8), RED)
    doc.checkpoint()
    doc.stroke((0, 0), (7, 7), BLUE, 1)
    assert doc.image.getpixel((3, 3)) == BLUE
    assert doc.undo()
    assert doc.image.getpixel((3, 3)) == RED
    assert doc.redo()
    assert doc.image.getpixel((3, 3)) == BLUE


def test_undo_with_nothing_to_undo_reports_that_it_did_nothing():
    doc = _doc()
    assert not doc.undo()
    assert not doc.redo()


def test_a_new_edit_discards_the_redo_branch():
    doc = _doc((8, 8), RED)
    doc.checkpoint()
    doc.stroke((0, 0), (7, 7), BLUE, 1)
    doc.undo()
    assert doc.history.can_redo
    doc.checkpoint()
    assert not doc.history.can_redo


def test_history_evicts_the_oldest_snapshot_once_the_byte_budget_is_spent():
    # A budget small enough that the floor, not the budget, decides the depth.
    doc = _doc((64, 64), RED, budget=1)
    for _ in range(paint.UNDO_MIN_DEPTH + 5):
        doc.checkpoint()
    assert len(doc.history) == paint.UNDO_MIN_DEPTH


def test_history_never_grows_past_the_depth_cap_however_small_the_image():
    doc = _doc((1, 1), RED, budget=paint.UNDO_BYTES)
    for _ in range(paint.UNDO_MAX_DEPTH + 10):
        doc.checkpoint()
    assert len(doc.history) == paint.UNDO_MAX_DEPTH
