"""The preview pane's second playhead, headless.

Every assertion here is about what ``tick_preview`` must *not* do. Advancing a
clip is arithmetic the pure ``animation.advance`` already owns and is tested to
death beside it; what earns this file is the discipline that lets a preview run
while the document is being drawn on -- it touches neither ``playing`` nor
``saving``, and it never moves the document's own playhead. Break any one of
those and the feature does not fail, it locks the canvas.
"""

from __future__ import annotations

import numpy as np

from warlock.studio import inker_mode
from warlock.studio.inker.document import Document
from warlock.studio.inker_state import InkerDoc


def _paint(doc: Document, colour: tuple[int, int, int, int]) -> None:
    weight = np.ones((2, 2), dtype=np.float32)
    assert doc.write_colour((0, 0, 2, 2), colour, weight)


def _tab(frames: int = 3) -> InkerDoc:
    doc = Document.blank(4, 4)
    for index in range(frames):
        if index:
            doc.add_frame()
        _paint(doc, (10 * index, 0, 0, 255))
    doc.set_current_frame(0)
    tab = InkerDoc(doc=doc, title="clip")
    tab.preview_playing = True
    return tab


def test_a_tick_advances_the_previews_own_index():
    tab = _tab(3)
    inker_mode.tick_preview(tab, 100.0)
    assert tab.preview_index == 1


def test_a_tick_never_moves_the_documents_playhead():
    tab = _tab(3)
    for _ in range(5):
        inker_mode.tick_preview(tab, 100.0)
    assert tab.doc.anim.current == 0


def test_a_tick_never_sets_playing_or_saving():
    """The two flags ``busy`` is made of. A preview that set either would lock
    the canvas it exists to be watched beside."""
    tab = _tab(3)
    for _ in range(5):
        inker_mode.tick_preview(tab, 100.0)
    assert not tab.playing
    assert not tab.saving
    assert not tab.busy


def test_a_stopped_preview_ticks_nothing():
    tab = _tab(3)
    tab.preview_playing = False
    inker_mode.tick_preview(tab, 1000.0)
    assert tab.preview_index == 0
    assert tab.preview_accum_ms == 0.0


def test_a_still_document_has_nothing_to_preview():
    tab = InkerDoc(doc=Document.blank(4, 4), title="still")
    tab.preview_playing = True
    inker_mode.tick_preview(tab, 100.0)
    assert tab.preview_index == 0


def test_the_whole_clip_scope_loops_past_a_non_looping_tag():
    """The scope switch has already answered "play the whole thing", so a tag
    that says otherwise must not stop it."""
    tab = _tab(3)
    assert tab.doc.add_tag("once", 0, 1, loop=False)
    tab.preview_scope = "clip"
    for _ in range(4):
        inker_mode.tick_preview(tab, 100.0)
    assert tab.preview_playing
    assert tab.preview_index == 1  # wrapped round the whole timeline


def test_the_tag_scope_stops_at_a_non_looping_tags_end():
    tab = _tab(4)
    assert tab.doc.add_tag("once", 0, 1, loop=False)
    tab.preview_scope = "tag"
    for _ in range(6):
        inker_mode.tick_preview(tab, 100.0)
    assert not tab.preview_playing
    assert tab.preview_index == 1


def test_the_tag_scope_respects_a_ping_pong():
    tab = _tab(3)
    assert tab.doc.add_tag("swing", 0, 2)
    assert tab.doc.set_tag(0, direction="pingpong")
    tab.preview_scope = "tag"
    walked = []
    for _ in range(4):
        inker_mode.tick_preview(tab, 100.0)
        walked.append(tab.preview_index)
    assert walked == [1, 2, 1, 0]


def test_the_tag_scope_stops_when_a_repeat_count_runs_out():
    tab = _tab(3)
    assert tab.doc.add_tag("twice", 0, 2)
    assert tab.doc.set_tag(0, repeat=2)
    tab.preview_scope = "tag"
    for _ in range(20):
        inker_mode.tick_preview(tab, 100.0)
        if not tab.preview_playing:
            break
    assert not tab.preview_playing
    assert tab.preview_cycles == 2
    assert tab.preview_index == 2


def test_the_speed_multiplier_scales_time_and_is_clamped():
    fast, slow = _tab(8), _tab(8)
    fast.preview_speed = 4.0
    slow.preview_speed = 0.25
    inker_mode.tick_preview(fast, 100.0)
    inker_mode.tick_preview(slow, 100.0)
    # One 100 ms tick is four 100 ms frames at x4 and a quarter of one at x0.25.
    assert fast.preview_index == 4
    assert slow.preview_index == 0

    wild = _tab(8)
    wild.preview_speed = 1000.0
    inker_mode.tick_preview(wild, 100.0)
    # Clamped to x4, i.e. the same four frames of time as ``fast`` above.
    assert wild.preview_index == 4


def test_a_stall_is_clamped_before_the_multiplier_is_applied():
    """A two-second hitch is a stall at x4 as much as at x1, so the clamp has
    to come first -- otherwise a fast preview fast-forwards eighty frames on
    the frame a dialog closed."""
    tab = _tab(60)
    tab.preview_speed = 4.0
    inker_mode.tick_preview(tab, 5000.0)
    assert tab.preview_index <= int(inker_mode.MAX_TICK_MS * 4.0 / 100.0)


def test_starting_the_preview_resets_the_leg_and_the_cycle_count():
    tab = _tab(3)
    tab.preview_playing = False
    tab.preview_forward = False
    tab.preview_cycles = 7
    inker_mode.toggle_preview(tab)
    assert tab.preview_playing
    assert tab.preview_forward
    assert tab.preview_cycles == 0
    inker_mode.toggle_preview(tab)
    assert not tab.preview_playing


def test_the_preview_index_survives_a_frame_being_deleted_under_it():
    tab = _tab(4)
    tab.preview_index = 3
    assert tab.doc.remove_range(2, 3)
    inker_mode.tick_preview(tab, 100.0)
    # Clamped at use, never at store: the index named a frame that has gone and
    # the preview carries on rather than indexing off the end.
    assert 0 <= tab.preview_index < len(tab.doc.anim.frames)
