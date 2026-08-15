"""Exporting part of a timeline: a tag, or a marquee'd range.

The interesting half is not "fewer frames" -- it is what the *sidecar* has to
say about them. A tag numbered against the whole document names cells that are
not in a partial file; a directional layout is a claim about the whole
timeline; and the whole-clip path has to keep producing exactly the bytes it
did before spans existed, because that is the output every engine reading these
sheets is already consuming.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import sheetout
from warlock.studio.inker.animation import Tag
from warlock.studio.inker.document import Document


def _paint(doc: Document, colour: tuple[int, int, int, int]) -> None:
    weight = np.ones((2, 2), dtype=np.float32)
    assert doc.write_colour((0, 0, 2, 2), colour, weight)


def _clip(frames: int = 4) -> Document:
    doc = Document.blank(4, 4)
    for index in range(frames):
        if index:
            doc.add_frame()
        _paint(doc, (10 * index, 0, 0, 255))
    return doc


# --- rebasing tags -----------------------------------------------------------


def test_a_tag_inside_the_span_is_shifted_to_the_new_numbering():
    tags = [Tag(name="walk", start=4, end=7)]
    out = sheetout.rebase_tags(tags, 4, 9)
    assert [(t.name, t.start, t.end) for t in out] == [("walk", 0, 3)]


def test_a_tag_wholly_outside_the_span_is_dropped():
    """It would name cells the file does not contain, which is the one kind of
    sidecar error a game does not notice until it plays the animation."""
    tags = [Tag(name="intro", start=0, end=2)]
    assert sheetout.rebase_tags(tags, 5, 9) == []


def test_a_tag_straddling_an_end_is_clamped_rather_than_dropped():
    tags = [Tag(name="combat", start=2, end=8)]
    out = sheetout.rebase_tags(tags, 4, 6)
    assert [(t.start, t.end) for t in out] == [(0, 2)]


def test_rebasing_keeps_everything_else_about_a_tag():
    tags = [Tag(name="swing", start=1, end=3, loop=False, direction="pingpong", repeat=2)]
    (out,) = sheetout.rebase_tags(tags, 1, 3)
    assert (out.name, out.loop, out.direction, out.repeat) == ("swing", False, "pingpong", 2)


def test_rebasing_does_not_write_through_into_the_document():
    original = Tag(name="walk", start=4, end=7)
    sheetout.rebase_tags([original], 4, 9)
    assert (original.start, original.end) == (4, 7)


# --- reading a span off a document -------------------------------------------


def test_a_span_reads_only_its_own_frames():
    doc = _clip(5)
    everything = sheetout.frame_uids(doc)
    assert sheetout.frame_uids(doc, (1, 3)) == everything[1:4]


def test_a_reversed_or_overhanging_span_is_clamped_rather_than_refused():
    doc = _clip(4)
    assert sheetout.frame_uids(doc, (3, 1)) == sheetout.frame_uids(doc, (1, 3))
    assert sheetout.frame_uids(doc, (-4, 40)) == sheetout.frame_uids(doc)


def test_a_span_past_the_end_entirely_is_refused():
    doc = _clip(3)
    with pytest.raises(ValueError):
        sheetout.frame_uids(doc, (7, 9))


def test_timing_slices_the_durations_with_the_same_span():
    doc = _clip(4)
    doc.set_frame_duration(1, 250)
    doc.set_frame_duration(2, 300)
    durations, _tags, _layout = sheetout.timing(doc, (1, 2))
    assert durations == [250, 300]


def test_a_partial_span_renumbers_the_tags_it_keeps():
    doc = _clip(6)
    assert doc.add_tag("intro", 0, 1)
    assert doc.add_tag("walk", 2, 5)
    _durations, tags, _layout = sheetout.timing(doc, (2, 5))
    assert [(t.name, t.start, t.end) for t in tags] == [("walk", 0, 3)]


def test_a_partial_span_drops_a_directional_layout():
    """A layout says "these frames are four directions of a walk in this fixed
    grid", which is a statement about the whole timeline. Half of one is a
    clip, and the row-wrapped grid is the honest answer for a clip."""
    from warlock.studio.inker.animation import DirectionalLayout

    doc = _clip(16)
    doc.anim.layout = DirectionalLayout.of("walk")
    assert sheetout.timing(doc, (0, 3))[2] is None
    assert sheetout.timing(doc, (0, 15))[2] is not None


def test_the_whole_timeline_path_is_unchanged_by_the_span_parameter():
    """The pin: an export with no span has to be the same call it always was,
    tags and layout objects included, or every sheet already on disk is
    describing a file this build would now write differently."""
    doc = _clip(4)
    assert doc.add_tag("walk", 1, 3)
    assert sheetout.frame_uids(doc, None) == sheetout.frame_uids(doc)
    plain, spanned = sheetout.timing(doc), sheetout.timing(doc, (0, 3))
    assert plain[0] == spanned[0]
    assert [(t.name, t.start, t.end) for t in plain[1]] == [
        (t.name, t.start, t.end) for t in spanned[1]
    ]
    # The *same objects*, not rebased copies: the whole-timeline branch does no
    # work at all, which is what makes "byte-identical" true rather than likely.
    assert plain[1][0] is doc.anim.tags[0]


def test_a_composed_span_sheet_holds_only_its_own_cells():
    pytest.importorskip("PIL.Image")
    doc = _clip(5)
    frames, durations, tags, layout, slices = sheetout.snapshot(doc, (1, 3))
    assert len(slices) == len(frames)
    image, plan, extra = sheetout.compose(frames, durations, tags, layout, slices)
    try:
        assert len(plan.cells) == 3
        assert extra["animation"]["frames"] == [
            {"cell_index": 0, "duration_ms": 100},
            {"cell_index": 1, "duration_ms": 100},
            {"cell_index": 2, "duration_ms": 100},
        ]
    finally:
        image.close()
