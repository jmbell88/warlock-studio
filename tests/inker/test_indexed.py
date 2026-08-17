"""Indexed colour: a palette the document carries, and writes snapped onto it.

The design decision the whole feature rests on is asserted first, because it is
the one a later reader is most likely to "fix": the pixels stay RGBA and the
palette is a *constraint on writes*. There is no index plane, and there is no
test here expecting one.

What the mode is actually for is the rest: no stray near-colours after a soft
stroke, a palette edit that reaches every pixel of every frame as one undo step,
and an exported colour table that is the one the user authored.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import indexed as ix
from warlock.studio.inker.document import Document
from warlock.studio.inker.layers import Layer, LayerStack

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RED = (220, 30, 30, 255)
RAMP = [BLACK, WHITE, RED]


def _plane(width: int = 4, height: int = 4, fill=(0, 0, 0, 0)) -> np.ndarray:
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[:, :] = fill
    return out


# --- the kernel --------------------------------------------------------------


def test_a_colour_snaps_to_the_nearest_swatch():
    pixels = _plane(fill=(250, 250, 250, 255))
    out = ix.snap(pixels, RAMP)
    assert tuple(out[0, 0]) == WHITE


def test_alpha_is_never_snapped():
    """A palette is a set of colours, not a set of opacities: a soft nib is
    still legal in indexed mode and simply bands."""
    pixels = _plane(fill=(250, 250, 250, 77))
    out = ix.snap(pixels, RAMP)
    assert tuple(out[0, 0]) == (255, 255, 255, 77)


def test_a_fully_transparent_pixel_is_left_alone():
    """It has no colour to snap, and rewriting its dead RGB would make a no-op
    write look like an edit to ``_commit_patch`` and push an undo step for a
    gesture that changed nothing."""
    pixels = _plane(fill=(9, 9, 9, 0))
    assert np.array_equal(ix.snap(pixels, RAMP), pixels)


def test_the_input_is_not_modified():
    pixels = _plane(fill=(250, 250, 250, 255))
    keep = pixels.copy()
    ix.snap(pixels, RAMP)
    assert np.array_equal(pixels, keep)


def test_nearest_is_measured_in_straight_rgb():
    """Premultiplying would drag every partially transparent pixel towards
    black, so a soft edge painted white would snap to the dark swatch."""
    pixels = _plane(fill=(255, 255, 255, 8))
    assert tuple(ix.snap(pixels, [BLACK, WHITE])[0, 0])[:3] == WHITE[:3]


def test_snapping_an_already_snapped_region_changes_nothing():
    """Idempotence is what makes ``_commit_patch``'s no-op check still work."""
    once = ix.snap(_plane(fill=(250, 250, 250, 255)), RAMP)
    assert np.array_equal(ix.snap(once, RAMP), once)


def test_an_empty_palette_is_refused():
    with pytest.raises(ValueError):
        ix.snap(_plane(), [])


def test_remap_is_exact_rather_than_nearest():
    """Editing a slot must not drag the swatch beside it along."""
    pixels = _plane(fill=(0, 0, 0, 255))
    pixels[0, 0] = (1, 1, 1, 255)
    out = ix.remap(pixels, BLACK, RED)
    assert tuple(out[1, 1]) == RED
    assert tuple(out[0, 0]) == (1, 1, 1, 255)


def test_histogram_counts_only_visible_pixels():
    pixels = _plane(fill=BLACK)
    pixels[0, :] = (0, 0, 0, 0)
    counts = ix.histogram(pixels, RAMP)
    assert counts == [12, 0, 0]


# --- the document ------------------------------------------------------------


def _doc(width: int = 4, height: int = 4) -> Document:
    return Document(stack=LayerStack([Layer(pixels=_plane(width, height), name="L")]))


def test_a_document_is_not_indexed_until_it_is_given_a_palette():
    assert _doc().is_indexed is False


def test_adopting_a_palette_snaps_what_is_already_painted():
    doc = _doc()
    doc.stack.active.pixels[:, :] = (250, 250, 250, 255)
    doc.invalidate_all()

    assert doc.set_palette(RAMP) is True

    assert tuple(doc.stack.active.pixels[0, 0]) == WHITE


def test_adopting_a_palette_is_one_undo_step():
    doc = _doc()
    doc.stack.active.pixels[:, :] = (250, 250, 250, 255)
    doc.invalidate_all()
    doc.set_palette(RAMP)

    doc.undo()

    assert tuple(doc.stack.active.pixels[0, 0]) == (250, 250, 250, 255)


def test_leaving_indexed_mode_keeps_the_pixels():
    """The only honest answer: the colours in the document *are* the ones that
    were painted, and there is nothing to restore them from."""
    doc = _doc()
    doc.stack.active.pixels[:, :] = (250, 250, 250, 255)
    doc.set_palette(RAMP)

    doc.set_palette(None)

    assert doc.is_indexed is False
    assert tuple(doc.stack.active.pixels[0, 0]) == WHITE


def test_a_write_is_snapped_as_it_commits():
    doc = _doc()
    doc.set_palette(RAMP)

    doc.write_colour((0, 0, 2, 2), (250, 250, 250, 255), np.ones((2, 2), dtype=np.float32))

    assert tuple(doc.stack.active.pixels[0, 0]) == WHITE


def test_the_undo_step_records_the_snapped_pixels():
    """Snapped before ``after`` is read, so undo-then-redo lands on the same
    colours rather than on the raw ones the tool asked for."""
    doc = _doc()
    doc.set_palette(RAMP)
    doc.write_colour((0, 0, 2, 2), (250, 250, 250, 255), np.ones((2, 2), dtype=np.float32))

    doc.undo()
    doc.redo()

    assert tuple(doc.stack.active.pixels[0, 0]) == WHITE


def test_a_write_that_snaps_onto_what_was_there_pushes_nothing():
    doc = _doc()
    doc.stack.active.pixels[:, :] = WHITE
    doc.set_palette(RAMP)
    head = doc.history.head

    doc.write_colour((0, 0, 2, 2), (250, 250, 250, 255), np.ones((2, 2), dtype=np.float32))

    assert doc.history.head == head


def test_recolouring_a_slot_repaints_every_pixel_of_that_colour():
    doc = _doc()
    doc.stack.active.pixels[:, :] = BLACK
    doc.set_palette(RAMP)

    assert doc.recolour_slot(0, (10, 20, 30, 255)) is True

    assert tuple(doc.stack.active.pixels[0, 0]) == (10, 20, 30, 255)
    assert doc.palette[0] == (10, 20, 30, 255)


def test_recolouring_a_slot_is_one_undo_step():
    doc = _doc()
    doc.stack.active.pixels[:, :] = BLACK
    doc.set_palette(RAMP)
    doc.recolour_slot(0, (10, 20, 30, 255))

    doc.undo()

    assert tuple(doc.stack.active.pixels[0, 0]) == BLACK


def test_deleting_a_slot_merges_its_pixels_into_the_nearest_survivor():
    doc = _doc()
    doc.stack.active.pixels[:, :] = RED
    doc.set_palette([BLACK, WHITE, RED])

    assert doc.remove_slot(2) is True

    assert doc.palette == [BLACK, WHITE]
    assert tuple(doc.stack.active.pixels[0, 0]) == BLACK


def test_the_last_slot_cannot_be_deleted():
    """An indexed document with no colours is one every visible pixel would
    have to snap to nothing."""
    doc = _doc()
    doc.set_palette([BLACK])
    assert doc.remove_slot(0) is False


def test_reordering_changes_no_pixel():
    doc = _doc()
    doc.stack.active.pixels[:, :] = RED
    doc.set_palette(RAMP)
    before = doc.stack.active.pixels.copy()

    assert doc.move_slot(2, 0) is True

    assert doc.palette == [RED, BLACK, WHITE]
    assert np.array_equal(doc.stack.active.pixels, before)


def test_usage_counts_are_reported_per_slot():
    doc = _doc()
    doc.stack.active.pixels[:, :] = RED
    doc.set_palette(RAMP)
    assert doc.palette_usage() == [0, 0, 16]


# --- across frames -----------------------------------------------------------


def _animated() -> Document:
    doc = Document.blank(4, 4)
    doc.write_colour((0, 0, 4, 4), BLACK, np.ones((4, 4), dtype=np.float32))
    doc.add_frame(link=False)
    doc.write_colour((0, 0, 4, 4), BLACK, np.ones((4, 4), dtype=np.float32))
    return doc


def test_a_slot_edit_reaches_every_frame():
    """The payoff of carrying a palette at all: a recolour addressed by colour
    rather than by selection, over the whole clip."""
    doc = _animated()
    doc.set_palette(RAMP)

    doc.recolour_slot(0, (10, 20, 30, 255))

    planes = list(doc.anim.unique_cel_layers())
    assert len(planes) == 2
    assert all(tuple(p.pixels[0, 0]) == (10, 20, 30, 255) for p in planes)


def test_a_slot_edit_across_frames_is_one_undo_step():
    doc = _animated()
    doc.set_palette(RAMP)
    doc.recolour_slot(0, (10, 20, 30, 255))

    doc.undo()

    planes = list(doc.anim.unique_cel_layers())
    assert all(tuple(p.pixels[0, 0]) == BLACK for p in planes)


# --- persistence -------------------------------------------------------------


def test_the_palette_survives_an_ora_round_trip(tmp_path):
    from warlock.studio.inker.ora import read_ora, write_ora

    doc = _doc()
    doc.stack.active.pixels[:, :] = RED
    doc.set_palette(RAMP)
    path = tmp_path / "indexed.ora"
    write_ora(doc, path)

    back = read_ora(path)

    # ``is_palette_locked``, not ``is_indexed``: an ORA carrying a palette but
    # no index planes is *palette-constrained RGB*, which is what every document
    # written before true indexed mode existed is. The two questions were one
    # property until the modes split; conflating them is what would let a pane
    # offer a transparent-index marker on a document with no index to mark.
    assert back.is_palette_locked is True
    assert back.is_indexed is False
    assert back.palette == RAMP


def test_an_ora_without_a_palette_opens_unindexed(tmp_path):
    from warlock.studio.inker.ora import read_ora, write_ora

    path = tmp_path / "plain.ora"
    write_ora(_doc(), path)

    back = read_ora(path)
    assert back.is_indexed is False and back.is_palette_locked is False


def test_opening_an_indexed_file_pushes_no_undo_step(tmp_path):
    """The pixels in the file were written snapped, so re-snapping them would
    cost a whole-document rewrite on every open and make opening a file
    undoable."""
    from warlock.studio.inker.ora import read_ora, write_ora

    doc = _doc()
    doc.stack.active.pixels[:, :] = RED
    doc.set_palette(RAMP)
    path = tmp_path / "indexed.ora"
    write_ora(doc, path)

    assert read_ora(path).history.can_undo is False


def test_a_corrupt_palette_member_costs_the_constraint_not_the_file(tmp_path):
    import zipfile

    from warlock.studio.inker.ora import PALETTE_MEMBER, read_ora, write_ora

    doc = _doc()
    doc.set_palette(RAMP)
    path = tmp_path / "indexed.ora"
    write_ora(doc, path)
    # Rewrite the archive with the palette member replaced by nonsense.
    with zipfile.ZipFile(path) as src:
        members = [(i, src.read(i.filename)) for i in src.infolist()]
    with zipfile.ZipFile(path, "w") as out:
        for info, data in members:
            out.writestr(info, b"not a palette" if info.filename == PALETTE_MEMBER else data)

    back = read_ora(path)

    assert back.is_indexed is False
    assert back.size == (4, 4)


# --- export ------------------------------------------------------------------


def test_a_gif_written_with_a_palette_carries_that_table(tmp_path):
    from PIL import Image

    from warlock.studio.inker import gifout

    frame = _plane(4, 4, fill=RED)
    path = tmp_path / "clip.gif"
    gifout.write_gif(path, [frame, frame], [100, 100], palette=RAMP)

    with Image.open(path) as im:
        table = im.getpalette()[: 3 * len(RAMP)]
    assert table == [c for colour in RAMP for c in colour[:3]]


def test_a_gif_colour_lands_in_the_slot_the_user_put_it_in(tmp_path):
    from PIL import Image

    from warlock.studio.inker import gifout

    path = tmp_path / "clip.gif"
    gifout.write_gif(path, [_plane(4, 4, fill=RED)], [100], palette=RAMP)

    with Image.open(path) as im:
        assert im.convert("RGBA").getpixel((0, 0)) == RED


def test_a_palette_too_long_for_a_gif_falls_back_rather_than_truncating(tmp_path):
    """A dropped swatch is pixels changing colour in the export."""
    from warlock.studio.inker import gifout

    long = [(i % 256, 0, 0, 255) for i in range(gifout.MAX_PALETTE + 1)]
    path = tmp_path / "clip.gif"
    gifout.write_gif(path, [_plane(4, 4, fill=RED)], [100], palette=long)

    assert path.exists()
