"""Section 8 of the 2026-09-02 review, closed 2026-09-04.

The last section of that document, set aside on 2026-09-04 and built the same
day. What it named was one engine defect (row-scoped effects that every tracker
makes persistent), two invisible states an envelope marker could be dragged
into, a manifest that could name one sample twice and quietly keep the second,
and a list of verbs a FamiTracker user reaches for and did not find: note
preview, keyboard instrument selection, Home/End, Insert and shift-rows,
interpolate, and the channels past the right-hand edge of the pane.

**The split is what these assertions are for.** ``sirens_mode.py`` became
``sirens_edit``/``sirens_play``/``sirens_keys`` (T7's mechanism, its ``_MOVED``
table) *after* everything below was green, so the moves are pure code motion
over behaviour these tests pin -- which is the order T7 itself gives and the
reason it was done last. Several tests below deliberately reach through
``sirens_mode.<name>`` rather than through the file a function now lives in:
that address is the compatibility surface, and a test that named the new module
would stop noticing if the door back closed.

The six Sirens panes still cannot be driven headlessly -- this suite has no
imgui harness -- so the pattern is ``test_findings_blind_spots``': a draw
function stays covered by the smoke pass and its **decisions** are covered here,
as pure functions beside it (``sirens_patterns.first_channel``,
``sirens/envelope.marker_bounds``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from test_sirens_mode import FakeCtx, _Event, _tab

from warlock.studio import sirens_mode
from warlock.studio.panes import sirens_envelopes, sirens_patterns
from warlock.studio.sirens import document as D
from warlock.studio.sirens import envelope, notes, synth, wsng
from warlock.studio.sirens import instruments as inst

# --- the engine: an effect runs until it is cancelled -------------------------


def _put(doc: Any, row: int, **values: Any) -> None:
    cells = doc.patterns[0].cells
    for key, value in values.items():
        cells[row, 0, getattr(D, key.upper())] = value


def _song(rows: int = 8) -> Any:
    """One note on one instrument that is still sounding several rows later.

    The volume envelope is the point: the default instrument is a six-step
    pluck that finishes inside one row, so a persistence test written against
    it would compare two silences and pass whatever the engine did.
    """
    doc = D.new_song()
    doc.patterns[0].cells = D.empty_cells(rows, doc.patterns[0].channels)
    instrument = doc.add_instrument()
    doc.update_instrument(
        instrument.uid, volume=inst.Sequence(values=(15,), loop=0)
    )
    _put(doc, 0, note=60, instrument=instrument.uid)
    return doc


@pytest.mark.parametrize(
    ("effect", "param", "cancel"),
    [
        (synth.FX_SLIDE_UP, 0x40, 0x00),
        (synth.FX_SLIDE_DOWN, 0x40, 0x00),
        (synth.FX_ARPEGGIO, 0x47, 0x00),
        # ``4x0`` is the cancel, so the *speed* nibble is what stays. 4 rather
        # than 8: a half-cycle per tick is sampled at the zero crossings and
        # sounds like no vibrato at all, which would make this test pass
        # against an engine that had dropped the effect entirely.
        (synth.FX_VIBRATO, 0x4F, 0x40),
        # One step per tick, not fifteen: a fade that reaches silence inside
        # its own row cannot show whether it kept going after it.
        (synth.FX_VOLUME_SLIDE, 0x01, 0x00),
    ],
)
def test_a_voice_effect_keeps_running_after_the_row_that_set_it(effect, param, cancel):
    """The defect, stated once per effect: these were reset at the top of every
    row, so ``103`` was a slide that lasted a sixteenth note and the user who
    typed it had nothing on screen saying why."""
    persisting = _song()
    _put(persisting, 0, effect=effect, param=param)

    stopped = _song()
    _put(stopped, 0, effect=effect, param=param)
    _put(stopped, 1, effect=effect, param=cancel)

    a, _loop = synth.render(persisting)
    b, _loop = synth.render(stopped)
    assert a.shape == b.shape
    assert not np.allclose(a, b), "the effect stopped at the end of its own row"


@pytest.mark.parametrize(
    ("effect", "param"),
    [
        (synth.FX_SLIDE_UP, 0x40),
        (synth.FX_ARPEGGIO, 0x47),
        (synth.FX_VIBRATO, 0x4F),
        (synth.FX_VOLUME_SLIDE, 0x01),
    ],
)
def test_a_zero_parameter_is_what_stops_it(effect, param):
    """The other half: cancelled on row 1, the rest of the song has to be the
    audio of a song that never had the effect at all."""
    cancelled = _song()
    _put(cancelled, 0, effect=effect, param=param)
    _put(cancelled, 1, effect=effect, param=0x00)

    plain = _song()
    _put(plain, 0, effect=effect, param=param)
    _put(plain, 1, effect=effect, param=0x00)
    # Same document; what is asserted is that the *tail* of the two renders
    # agrees with a render that stops, rather than with one that keeps going.
    running = _song()
    _put(running, 0, effect=effect, param=param)

    stopped, _l = synth.render(cancelled)
    kept, _l = synth.render(running)
    assert stopped.shape == kept.shape
    tail = stopped.shape[0] // 2
    assert not np.allclose(stopped[tail:], kept[tail:])
    assert np.allclose(stopped, synth.render(plain)[0])


def test_the_player_effects_are_events_and_do_not_persist():
    """``Bxx``/``Cxx``/``Dxx`` happen on their row and nothing carries over --
    a halt that persisted would stop the song on every subsequent row too,
    which is the same thing, and a *jump* that persisted would never end."""
    doc = _song(4)
    _put(doc, 1, effect=synth.FX_HALT, param=0)
    out, _loop = synth.render(doc)
    assert out.shape[0] > 0


def test_every_effect_says_in_its_own_name_whether_it_persists():
    """The manual's rule and the tooltip's have one source. Each of the six
    voice effects names how it is turned off; a seventh added to the engine
    without that sentence fails here."""
    for effect in (
        synth.FX_ARPEGGIO,
        synth.FX_SLIDE_UP,
        synth.FX_SLIDE_DOWN,
        synth.FX_PORTAMENTO,
        synth.FX_VIBRATO,
        synth.FX_VOLUME_SLIDE,
    ):
        _letter, description = synth.EFFECT_NAMES[effect]
        assert "until" in description.lower(), description


# --- the engine: one note, previewed ------------------------------------------


def test_a_previewed_note_is_the_voice_the_channel_is():
    """The preview goes through the ordinary renderer on a scratch document, so
    a noise channel previews as noise. Two kinds, two different buffers."""
    doc = _song()
    uid = doc.instruments[0].uid
    pulse = synth.render_note(doc, uid, 60, kind="pulse")
    noise = synth.render_note(doc, uid, 60, kind="noise")
    assert pulse.size and noise.size
    assert not np.allclose(pulse[: noise.shape[0]], noise[: pulse.shape[0]])


def test_a_preview_of_a_note_with_no_instrument_is_silence_rather_than_a_crash():
    doc = _song()
    assert synth.render_note(doc, 0x7E, 60).size == 0
    assert synth.render_note(doc, doc.instruments[0].uid, notes.NOTE_OFF).size == 0


# --- the engine: a manifest that names one sample twice ------------------------


def test_two_manifest_entries_for_one_sample_key_are_refused_by_name():
    """Collapsed silently before: the second entry won, and whichever
    instrument named that key played the wrong sound with nothing saying so."""
    import json
    import zipfile

    doc = D.new_song()
    doc.set_sample("hit", np.zeros(64, dtype=np.float32))
    raw = wsng.wsng_bytes(doc)

    out = bytearray()
    import io

    source = zipfile.ZipFile(io.BytesIO(raw))
    manifest = json.loads(source.read(wsng.MANIFEST))
    entry = dict(manifest["samples"][0])
    manifest["samples"] = [entry, dict(entry)]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for member in source.namelist():
            if member == wsng.MANIFEST:
                zf.writestr(member, json.dumps(manifest))
            else:
                zf.writestr(member, source.read(member))
    out = buffer.getvalue()

    with pytest.raises(ValueError, match="twice"):
        wsng.read_wsng(out)


# --- the envelope editor's two invisible states -------------------------------


def test_a_release_marker_cannot_be_dragged_onto_step_zero():
    """``release == 0`` makes the whole sequence tail material and a held note
    silent -- a state the engine tolerates from a file and one no drag should
    be able to produce."""
    sequence = inst.Sequence(values=(15, 12, 8, 4), release=2)
    assert envelope.moved(sequence, "release", 0).release == 1
    assert envelope.moved(sequence, "release", -5).release == 1


def test_a_loop_dragged_past_the_release_stops_at_the_last_step_before_it():
    """It vanished from the graph and stayed in the document: the editor drew
    the loop only inside the held half, and the engine never reaches a loop
    point in the tail."""
    sequence = inst.Sequence(values=(15, 12, 8, 4), loop=0, release=2)
    after = envelope.moved(sequence, "loop", 3)
    assert after.loop == 1
    assert 0 <= after.loop < after.release


def test_a_marker_with_no_room_is_left_where_it_was():
    """A one-step sequence has no held half to split off, so there is nowhere
    legal for a release to land -- and nowhere for a loop to clear it."""
    one = inst.Sequence(values=(15,))
    assert envelope.moved(one, "release", 0) == one
    assert envelope.toggled(one, "release") == one
    assert envelope.moved(inst.Sequence(values=(15, 12), release=1), "loop", 5).loop == 0


def test_shortening_a_sequence_cannot_create_either_invisible_state():
    short = envelope.resized(
        inst.Sequence(values=tuple(range(20)), loop=15, release=18), 1
    )
    assert short.values == (0,)
    assert (short.loop, short.release) == (0, -1)


def test_the_pane_still_answers_at_every_moved_name():
    """The pure half moved under ``studio/sirens/``; the pane is where every
    caller and every existing test names it."""
    for name in (
        "span", "columns", "painted", "moved", "toggled", "grabbed",
        "step_at", "value_at", "marker_bounds", "resized", "MIN_STEPS",
    ):
        assert getattr(sirens_envelopes, name) is getattr(envelope, name)


def test_the_envelope_arithmetic_is_reachable_with_no_imgui_frame():
    """Which is the whole reason it moved. ``tests/sirens/test_sirens_imports``
    is what pins the package's outward set; this asserts the consequence."""
    import sys

    assert "warlock.studio.sirens.envelope" in sys.modules
    assert envelope.marker_bounds(inst.Sequence(values=(1, 2, 3)), "release") == (1, 2)


# --- rows: insert, delete, interpolate ----------------------------------------


def _grid(ctx: FakeCtx, tab: Any) -> Any:
    return tab.doc.pattern(sirens_mode.ensure(ctx).pattern).cells


def test_insert_opens_a_row_and_keeps_the_pattern_the_length_it_was():
    ctx = FakeCtx()
    tab = _tab(ctx)
    rows = tab.doc.patterns[0].rows
    tab.doc.set_cell(tab.doc.patterns[0].uid, 0, 0, D.NOTE, 60)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert sirens_mode.shift_rows(ctx, 1)
    cells = _grid(ctx, tab)
    assert cells[0, 0, D.NOTE] == notes.EMPTY
    assert cells[1, 0, D.NOTE] == 60
    assert tab.doc.patterns[0].rows == rows


def test_deleting_a_row_pulls_the_rest_up():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 1, 0, D.NOTE, 60)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert sirens_mode.shift_rows(ctx, -1)
    assert _grid(ctx, tab)[0, 0, D.NOTE] == 60


def test_a_row_shift_is_one_undo_step_and_arms_the_renderer():
    ctx = FakeCtx()
    tab = _tab(ctx)
    head = tab.doc.history.head
    tab.render_dirty = False
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    tab.doc.set_cell(tab.doc.patterns[0].uid, 0, 0, D.NOTE, 60)
    head = tab.doc.history.head
    sirens_mode.shift_rows(ctx, 1)
    assert tab.doc.history.head == head + 1
    assert tab.render_dirty


def test_interpolate_fills_the_rows_between_a_blocks_ends():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.VOLUME, 0)
    tab.doc.set_cell(uid, 4, 0, D.VOLUME, 12)
    state = sirens_mode.ensure(ctx)
    state.anchor = (0, 0)
    sirens_mode.set_caret(ctx, row=4, channel=0)
    state.anchor = (0, 0)
    assert sirens_mode.interpolate_selection(ctx)
    column = _grid(ctx, tab)[0:5, 0, D.VOLUME]
    assert list(column) == [0, 3, 6, 9, 12]


def test_interpolate_leaves_a_column_whose_ends_do_not_both_answer():
    """A ramp needs two endpoints. Inventing one from an empty cell is how a
    fade turns into notes nobody typed."""
    doc = D.new_song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 0, 0, D.NOTE, 60)
    doc.interpolate(uid, 0, 0, 5, 1)
    assert list(doc.patterns[0].cells[1:5, 0, D.NOTE]) == [notes.EMPTY] * 4


def test_interpolate_never_ramps_the_instrument_column():
    """Ids are a set with no order: a line from 01 to 07 names five slots
    nobody chose."""
    assert D.INSTRUMENT not in D.SongDoc.RAMP_COLUMNS


def test_interpolate_refuses_out_loud_with_nothing_to_ramp_between():
    ctx = FakeCtx()
    _tab(ctx)
    assert not sirens_mode.interpolate_selection(ctx)
    assert ctx.toasts and "three rows" in ctx.toasts[-1][0]


# --- the keyboard -------------------------------------------------------------


def _press(ctx: FakeCtx, name: str, mod: int = 0) -> bool:
    import pygame

    return sirens_mode.handle_key(ctx, _Event(getattr(pygame, name), mod))


def test_home_and_end_reach_the_ends_of_the_pattern():
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.set_caret(ctx, row=5, channel=0, column=D.NOTE)
    assert _press(ctx, "K_HOME")
    assert sirens_mode.ensure(ctx).row == 0
    assert _press(ctx, "K_END")
    assert sirens_mode.ensure(ctx).row == tab.doc.patterns[0].rows - 1


def test_shift_end_selects_to_the_end_rather_than_dropping_the_block():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert _press(ctx, "K_END", pygame.KMOD_SHIFT)
    row, chan, rows, chans = sirens_mode.ensure(ctx).selection()
    assert (row, chan, rows, chans) == (0, 0, tab.doc.patterns[0].rows, 1)


def test_insert_and_shift_delete_are_bound():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.NOTE, 60)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert _press(ctx, "K_INSERT")
    assert _grid(ctx, tab)[1, 0, D.NOTE] == 60
    assert _press(ctx, "K_DELETE", pygame.KMOD_SHIFT)
    assert _grid(ctx, tab)[0, 0, D.NOTE] == 60


def test_plain_delete_still_blanks_the_column_it_is_on():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.NOTE, 60)
    tab.doc.set_cell(uid, 0, 0, D.INSTRUMENT, 3)
    sirens_mode.ensure(ctx).step = 0
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.INSTRUMENT)
    assert _press(ctx, "K_DELETE")
    cells = _grid(ctx, tab)
    assert cells[0, 0, D.INSTRUMENT] == notes.EMPTY
    assert cells[0, 0, D.NOTE] == 60


def test_ctrl_up_and_down_step_the_stamped_instrument():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    uids = [one.uid for one in tab.doc.instruments]
    assert len(uids) > 1, "a new song has an instrument list to step through"
    state = sirens_mode.ensure(ctx)
    state.instrument = uids[0]
    assert _press(ctx, "K_DOWN", pygame.KMOD_CTRL)
    assert state.instrument == uids[1]
    assert _press(ctx, "K_UP", pygame.KMOD_CTRL)
    assert state.instrument == uids[0]
    # Clamped, not wrapped: a step past the end landing on the other end is a
    # stamp nobody meant.
    _press(ctx, "K_UP", pygame.KMOD_CTRL)
    assert state.instrument == uids[0]
    state.instrument = uids[-1]
    _press(ctx, "K_DOWN", pygame.KMOD_CTRL)
    assert state.instrument == uids[-1]


def test_ctrl_g_interpolates():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    uid = tab.doc.patterns[0].uid
    tab.doc.set_cell(uid, 0, 0, D.VOLUME, 0)
    tab.doc.set_cell(uid, 2, 0, D.VOLUME, 8)
    state = sirens_mode.ensure(ctx)
    sirens_mode.set_caret(ctx, row=2, channel=0)
    state.anchor = (0, 0)
    assert _press(ctx, "K_g", pygame.KMOD_CTRL)
    assert _grid(ctx, tab)[1, 0, D.VOLUME] == 4


def test_a_busy_tab_refuses_the_interpolate_chord():
    ctx = FakeCtx()
    tab = _tab(ctx)
    import pygame

    tab.saving = True
    assert "g" in sirens_mode._MUTATING_CTRL
    assert _press(ctx, "K_g", pygame.KMOD_CTRL)
    assert tab.doc.history.head == 0


# --- the note preview ---------------------------------------------------------


def test_typing_a_note_asks_for_a_preview(monkeypatch):
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.instrument = tab.doc.add_instrument().uid
    sirens_mode.set_caret(ctx, row=0, channel=0, column=D.NOTE)
    assert _press(ctx, "K_z")
    assert any(key.startswith(sirens_mode.PREVIEW_PREFIX) for key in ctx.submitted)


def test_the_song_wins_over_a_preview(monkeypatch):
    """One reserved mixer channel, so a preview would cut whatever is on it --
    and typing into bar 3 while bar 1 plays is what follow mode is for."""
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: True)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "song")
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).instrument = tab.doc.add_instrument().uid
    assert not sirens_mode.preview_note(ctx, 60)
    assert not ctx.submitted


def test_a_preview_switched_off_costs_nothing(monkeypatch):
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.instrument = tab.doc.add_instrument().uid
    state.preview = False
    assert not sirens_mode.preview_note(ctx, 60)
    assert not ctx.submitted


def test_a_preview_never_lands_on_the_songs_buffer():
    """``AUDITION_PREFIX``'s rule, third instance: its own key and its own arm,
    so a note typed during a re-render neither is refused nor replaces the
    song with a single note until the next edit."""
    assert sirens_mode.PREVIEW_PREFIX not in (
        "sirens-render:",
        sirens_mode.AUDITION_PREFIX,
        sirens_mode.PATTERN_PREFIX,
    )


# --- the channels past the pane's right-hand edge ------------------------------


def test_the_channel_window_follows_the_caret_in_both_directions():
    """They were drawn nowhere and typed into all the same."""
    assert sirens_patterns.first_channel(caret=7, count=8, fits=3, scroll=0) == 5
    assert sirens_patterns.first_channel(caret=0, count=8, fits=3, scroll=5) == 0


def test_the_channel_window_stays_put_while_the_caret_is_inside_it():
    """Recomputing "centre on the caret" every frame would slide the whole grid
    sideways on every Right."""
    assert sirens_patterns.first_channel(caret=3, count=8, fits=3, scroll=2) == 2


def test_the_channel_window_never_runs_past_the_last_channel():
    assert sirens_patterns.first_channel(caret=0, count=3, fits=5, scroll=9) == 0
    assert sirens_patterns.first_channel(caret=7, count=8, fits=8, scroll=4) == 0


# --- the channel properties the model could always do -------------------------


def test_renaming_repanning_and_re_kinding_a_channel_all_reach_the_document():
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.channels[3].uid
    assert sirens_mode.update_channel(ctx, uid, name="Snare")
    assert sirens_mode.update_channel(ctx, uid, kind="triangle")
    assert sirens_mode.update_channel(ctx, uid, pan=-0.5)
    channel = tab.doc.channel(uid)
    assert (channel.name, channel.kind, channel.pan) == ("Snare", "triangle", -0.5)
    assert tab.render_dirty


def test_a_refused_channel_change_is_a_toast_rather_than_a_traceback():
    ctx = FakeCtx()
    tab = _tab(ctx)
    assert not sirens_mode.update_channel(ctx, tab.doc.channels[0].uid, kind="banjo")
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_the_notes_written_on_a_channel_survive_a_change_of_voice():
    """The voice is how they sound, not what they are."""
    doc = D.new_song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 0, 3, D.NOTE, 60)
    doc.update_channel(doc.channels[3].uid, kind="triangle")
    assert doc.patterns[0].cells[0, 3, D.NOTE] == 60


# --- the split ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "clamp_caret", "move_caret", "jump_row", "set_caret", "write_cell",
        "write_note", "write_hex", "write_effect", "clear_cell", "transpose",
        "shift_rows", "interpolate_selection", "copy_selection", "paste",
        "set_sequence", "adopt_sample", "undo", "redo",
        "request_render", "pump", "play", "play_pattern", "play_from_caret",
        "toggle_play", "playhead_row", "follow_playhead", "audition",
        "preview_note", "handle_key", "release_all", "PIANO_KEYS",
        "AUDITION_PREFIX", "ENVELOPE_FIELDS",
    ],
)
def test_every_moved_name_still_answers_at_its_old_address(name):
    """The door back. ``sirens_mode.<name>`` is what the panes, the app's key
    routing and most of this mode's tests say."""
    assert getattr(sirens_mode, name) is not None


def test_the_moved_table_names_each_thing_once_and_no_ghosts():
    """A name in the table that its module does not define is a door onto
    nothing, and one that is *also* still defined here is two places to keep
    in step."""
    from importlib import import_module

    for name, module in sirens_mode._MOVED.items():
        assert name not in vars(sirens_mode), f"{name} is in both places"
        assert hasattr(import_module(f"warlock.studio.{module}"), name)


def test_dir_still_finds_the_moved_names():
    assert "handle_key" in dir(sirens_mode)


# --- the rest of section 8's "Tests" paragraph --------------------------------


def test_the_playhead_reads_a_multi_entry_order_rather_than_one_long_pattern():
    """The estimate this replaced was wrong the moment a song had two patterns.
    Asserted here against the *order*: the second entry's rows have to come
    back as rows of the pattern that entry names, not as rows past the end of
    the first."""
    doc = D.new_song()
    first = doc.patterns[0]
    second = doc.add_pattern(rows=4)
    doc.set_order([first.uid, second.uid])
    _samples, _loop, marks = synth.render_marked(doc)
    entries = {index for _offset, index, _uid, _row in marks}
    assert entries == {0, 1}
    for _offset, index, uid, row in marks:
        pattern = first if index == 0 else second
        assert uid == pattern.uid
        assert 0 <= row < pattern.rows


def test_a_paste_wider_than_the_remaining_channels_is_clipped_not_refused():
    """``set_cells``' rule, reached through the clipboard: what fits goes in."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    pattern = tab.doc.patterns[0]
    tab.doc.set_cell(pattern.uid, 0, 0, D.NOTE, 60)
    tab.doc.set_cell(pattern.uid, 0, 1, D.NOTE, 62)
    sirens_mode.set_caret(ctx, row=0, channel=0)
    state.anchor = (0, 0)
    sirens_mode.set_caret(ctx, row=0, channel=1)
    state.anchor = (0, 0)
    assert sirens_mode.copy_selection(ctx)
    state.anchor = None
    sirens_mode.set_caret(ctx, row=2, channel=pattern.channels - 1)
    assert sirens_mode.paste(ctx)
    cells = tab.doc.pattern(pattern.uid).cells
    assert cells[2, pattern.channels - 1, D.NOTE] == 60


def test_a_click_takes_the_column_it_landed_on_including_the_gap_after_it():
    """"Click on ``Fxx``, type, get a note" was the defect. The gap after a
    column belongs to the value on its left, the way a tracker's does."""
    widths = [30.0, 20.0, 20.0, 10.0, 20.0]
    assert sirens_patterns.column_at(0.0, widths, 6.0) == 0
    assert sirens_patterns.column_at(33.0, widths, 6.0) == 0
    assert sirens_patterns.column_at(95.0, widths, 6.0) == 3
    assert sirens_patterns.column_at(9999.0, widths, 6.0) == 4


def test_the_loop_point_follows_an_entry_that_moves_under_it():
    """``sirens_orders.moved_loop``, the order list's own pure half."""
    from warlock.studio.panes import sirens_orders

    assert sirens_orders.moved_loop(2, 2, 0) == 0
    assert sirens_orders.moved_loop(0, 2, 0) == 1
