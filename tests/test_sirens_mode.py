"""Sirens' controller: documents, the caret, rendering, saving and keys.

Three things carry this file. The save rules every editor here shares -- a
failed save clears the lock, and the head a save records is the one the encode
wrote. **The render flag**, which is ``PackTab.pack_dirty``'s lesson a second
time: ``TaskRunner.submit`` refuses a key already in flight and nothing re-arms
it, so a flag cleared regardless of whether the submit was accepted silently
drops every note typed while a render was running. And the caret, which is the
one piece of state here that can be left pointing at something the document no
longer has.

Nothing in this file touches a sound device: playback goes through
``sirens_audio``, whose no-device path is ``test_sirens_audio.py``'s subject.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warlock.studio import sirens_io, sirens_mode
from warlock.studio.sirens import document as D
from warlock.studio.sirens import notes, wsng


class FakeCtx:
    def __init__(self, *, accept: bool = True) -> None:
        self.svc = None
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.confirms = _Confirms()
        self.viewer = None
        self.accept = accept
        self.result: Any = None
        self.tag: Any = None

        self.busy_keys: set[str] = set()

    def busy(self, key: str) -> bool:
        return key in self.busy_keys

    def submit(self, key: str, run: Any, *args: Any, tag: Any = None) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        # ``tag`` is kept, not dropped: it is how a completion says which
        # request it belongs to (S1), and a fake that swallowed it would make
        # the freshness check untestable.
        self.tag = tag
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", **_extra: Any) -> None:
        self.toasts.append((message, kind))

    def toast_once(self, message: str, kind: str = "info", **_extra: Any) -> bool:
        """The real one coalesces against the toasts still on screen; here
        nothing expires, so it coalesces against all of them."""
        if (message, kind) in self.toasts:
            return False
        self.toasts.append((message, kind))
        return True


class _AppState:
    def __init__(self) -> None:
        self.sirens = None
        # The Song file panel draws Muse's *Closeness* beside Compose (W1), so
        # the slot the real ``AppState`` carries has to be here too.
        self.muse = None
        self.mode = "home"
        self.preview: dict[str, Any] = {}


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm


class _Done:
    def __init__(
        self, key: str, result: Any = None, message: str = "", tag: Any = None
    ) -> None:
        self.key = key
        self.result = result
        self.message = message
        # Which request this completion belongs to; ``TaskRunner`` carries the
        # value ``submit`` was given (S1).
        self.tag = tag


class _Event:
    """A pygame KEYDOWN, without a display."""

    def __init__(self, key: int, mod: int = 0) -> None:
        import pygame

        self.type = pygame.KEYDOWN
        self.key = key
        self.mod = mod


def _tab(ctx: FakeCtx) -> Any:
    tab = sirens_mode.new_document(ctx)
    tab.doc.mark_saved()
    tab.render_dirty = True
    return tab


def _render(ctx: FakeCtx, tab: Any) -> None:
    sirens_mode.request_render(ctx, tab)
    sirens_mode.on_task_done(ctx, _Done(f"sirens-render:{tab.uid}", ctx.result))


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    """No test in this file may reach the mixer, on a box that has one or not.

    The mode's *behaviour* around a missing device is asserted here; whether
    the device exists is ``sirens_audio``'s question and CI's answer to it is
    not something these tests should depend on either way.
    """
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)


@pytest.fixture(autouse=True)
def _no_pygame_display(monkeypatch):
    import pygame

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


# --- documents ----------------------------------------------------------------


def test_a_new_song_can_be_played_the_moment_it_exists():
    """``new_song`` rather than a bare document: an empty order list is a song
    where Space does nothing and there is no way to find out why."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    assert tab.doc.order and tab.doc.patterns and tab.doc.instruments


def test_two_tabs_over_one_path_are_one_tab():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.path = Path("/songs/a.wsng")
    sirens_mode.open_path(ctx, Path("/songs/a.wsng"))
    assert len(sirens_mode.ensure(ctx).docs) == 1
    assert not ctx.submitted


# --- rendering ----------------------------------------------------------------


def test_a_render_lands_and_bumps_the_generation_exactly_once():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _render(ctx, tab)
    assert tab.pcm is not None and tab.pcm.shape[1] == 2
    assert tab.render_generation == 1
    assert not tab.rendering and not tab.render_dirty


def test_the_dirty_flag_is_cleared_at_the_submit_and_never_at_the_adoption():
    """The rule ``PackTab.pack_dirty`` states: an edit made while a render was
    in flight is not in the buffer landing now, and clearing the flag when it
    lands would drop that edit for good."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.request_render(ctx, tab)
    assert not tab.render_dirty
    result = ctx.result
    # A note typed while the render was running.
    tab.render_dirty = True
    sirens_mode.on_task_done(ctx, _Done(f"sirens-render:{tab.uid}", result))
    assert tab.render_dirty, "the adoption cleared a flag it does not own"


def test_a_refused_submit_leaves_the_flag_armed():
    """The runner refuses a key already in flight; nothing else re-arms it."""
    ctx = FakeCtx(accept=False)
    tab = _tab(ctx)
    sirens_mode.request_render(ctx, tab)
    assert tab.render_dirty and not tab.rendering


def test_a_song_with_an_empty_order_renders_nothing_and_stops_asking():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.doc.set_order([])
    tab.render_dirty = True
    sirens_mode.request_render(ctx, tab)
    assert tab.pcm is None and not tab.render_dirty and not ctx.submitted


def test_a_failed_render_clears_the_flag_and_records_why():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.rendering = True
    sirens_mode.on_task_failed(
        ctx, _Done(f"sirens-render:{tab.uid}", message="that song is too long")
    )
    assert not tab.rendering and tab.render_error == "that song is too long"


def test_a_render_already_in_flight_is_not_re_serialised_every_frame(monkeypatch):
    """``wsng_bytes`` DEFLATEs every pattern and encodes every sample, on the
    frame thread. ``submit`` refuses a key already in flight and the dirty flag
    deliberately stays armed when it does -- so the whole song was serialised
    and thrown away once per frame for as long as the render took."""
    from warlock.studio.sirens import wsng as wsng_mod

    ctx = FakeCtx()
    tab = _tab(ctx)
    calls: list[int] = []
    real = wsng_mod.wsng_bytes
    monkeypatch.setattr(
        wsng_mod, "wsng_bytes", lambda doc: (calls.append(1), real(doc))[1]
    )

    ctx.busy_keys.add(f"sirens-render:{tab.uid}")
    sirens_mode.request_render(ctx, tab)
    assert calls == [] and ctx.submitted == []
    assert tab.render_dirty, "still armed -- it is asked again when the render lands"

    ctx.busy_keys.clear()
    sirens_mode.request_render(ctx, tab)
    assert len(calls) == 1 and not tab.render_dirty


def test_the_render_task_reads_a_snapshot_rather_than_the_document():
    """The frame thread keeps editing while the task runs, so a task holding a
    numpy view would be reading an array the caret is writing into."""
    import inspect

    source = inspect.getsource(sirens_mode.request_render)
    assert "wsng_bytes" in source and "read_wsng" in source


# --- the caret ----------------------------------------------------------------


def test_arrows_move_the_caret_and_rows_wrap():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    rows = tab.doc.pattern(state.pattern).rows
    sirens_mode.move_caret(ctx, drow=-1)
    assert state.row == rows - 1
    sirens_mode.move_caret(ctx, drow=1)
    assert state.row == 0


def test_channels_and_columns_clamp_rather_than_wrap():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    for _ in range(20):
        sirens_mode.move_caret(ctx, dchannel=1, dcolumn=1)
    assert state.channel == tab.doc.pattern(state.pattern).channels - 1
    assert state.column == D.COLUMNS - 1


def test_a_shrunk_pattern_pulls_the_caret_back_inside_it():
    """Left outside, the next keystroke writes at a row that no longer exists
    -- which ``set_cells`` clips to nothing, so the key does nothing and there
    is no way to see why."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.row = 60
    tab.doc.resize_pattern(state.pattern, 8)
    sirens_mode.clamp_caret(ctx, tab)
    assert state.row == 7


def test_a_deleted_pattern_moves_the_caret_rather_than_leaving_it_dangling():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    extra = tab.doc.add_pattern()
    sirens_mode.set_caret(ctx, pattern=extra.uid)
    tab.doc.remove_pattern(extra.uid)
    sirens_mode.clamp_caret(ctx, tab)
    assert state.pattern == tab.doc.patterns[0].uid


def test_a_tab_switch_does_not_leave_the_caret_in_the_other_song():
    """A uid from another document is not merely stale: ``set_cell`` raises
    ``MISSING_PATTERN`` for it, so the next keystroke is a refusal about a
    pattern the user cannot see."""
    ctx = FakeCtx()
    first = _tab(ctx)
    second = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    assert state.pattern == second.doc.patterns[0].uid
    state.activate(first.uid)
    assert state.pattern == first.doc.patterns[0].uid


# --- editing ------------------------------------------------------------------


def test_a_typed_note_carries_the_selected_instrument():
    """A note with no instrument plays nothing, and the user who typed it has
    no reason to suspect a second column they never touched."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.octave = 4
    sirens_mode.write_note(ctx, 0)
    cells = tab.doc.pattern(state.pattern).cells
    assert cells[0, 0, D.NOTE] == 48
    assert cells[0, 0, D.INSTRUMENT] == state.instrument


def test_a_typed_note_steps_the_caret_by_the_edit_step():
    ctx = FakeCtx()
    _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.step = 4
    sirens_mode.write_note(ctx, 0)
    assert state.row == 4


def test_every_edit_arms_the_renderer():
    """An edit you cannot hear is indistinguishable from one that did not
    happen, so no mutator may forget the flag."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    for call in (
        lambda: sirens_mode.write_note(ctx, 0),
        lambda: sirens_mode.transpose(ctx, 1),
        lambda: sirens_mode.clear_selection(ctx),
        lambda: sirens_mode.cut_selection(ctx),
        # Pasted away from the cell it was cut from: putting the same block
        # back where it already is is a no-op, and a no-op arms nothing.
        lambda: (sirens_mode.set_caret(ctx, row=4), sirens_mode.paste(ctx)),
    ):
        tab.render_dirty = False
        sirens_mode.set_caret(ctx, row=0)
        # Something under the caret for cut to take and paste to put down
        # somewhere new -- a no-op is not an edit and arms nothing.
        state = sirens_mode.ensure(ctx)
        tab.doc.set_cell(state.pattern, 0, 0, 0, 12)
        sirens_mode.set_caret(ctx, row=0)
        call()
        assert tab.render_dirty, call


def test_shift_arrow_anchors_one_rectangle_rather_than_re_anchoring():
    ctx = FakeCtx()
    _tab(ctx)
    state = sirens_mode.ensure(ctx)
    sirens_mode.move_caret(ctx, drow=1, select=True)
    sirens_mode.move_caret(ctx, drow=1, select=True)
    assert state.anchor == (0, 0)
    assert state.selection() == (0, 0, 3, 1)


def test_an_undo_clamps_the_caret_and_re_renders():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    tab.doc.resize_pattern(state.pattern, 8)
    state.row = 7
    tab.render_dirty = False
    sirens_mode.undo(ctx, tab)
    assert tab.render_dirty and state.row == 7
    sirens_mode.redo(ctx, tab)
    assert state.row == 7


# --- keys ---------------------------------------------------------------------


def test_the_piano_row_only_types_in_the_note_column():
    """``e`` in the effect column is the letter of an effect, not an E-natural
    -- a piano row that fired everywhere would make four columns untypable.

    The key is consumed rather than ignored, because it is now *answered*: a
    piano key outside the note column is the one rejected key this mode says
    something about. What must stay true is that it writes nothing.
    """
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.column = D.EFFECT
    assert sirens_mode.handle_key(ctx, _Event(pygame.K_z))
    assert tab.doc.pattern(state.pattern).cells[0, 0, D.NOTE] == notes.EMPTY
    assert tab.doc.pattern(state.pattern).cells[0, 0, D.EFFECT] == notes.EMPTY
    assert ctx.toasts and "Effect column" in ctx.toasts[0][0]


def test_space_is_bound_even_with_no_device():
    """Consumed either way: a Space that fell through would step imgui's focus
    ring through the transport buttons."""
    import pygame

    ctx = FakeCtx()
    _tab(ctx)
    assert sirens_mode.handle_key(ctx, _Event(pygame.K_SPACE))
    assert ctx.toasts and "playback is unavailable" in ctx.toasts[0][0]


def test_delete_clears_the_block_under_the_caret():
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    sirens_mode.write_note(ctx, 0)
    sirens_mode.set_caret(ctx, row=0)
    assert sirens_mode.handle_key(ctx, _Event(pygame.K_DELETE))
    assert tab.doc.pattern(state.pattern).cells[0, 0, D.NOTE] == notes.EMPTY


def test_a_busy_tab_refuses_undo_without_falling_through():
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    head = tab.doc.history.head
    assert sirens_mode.handle_key(ctx, _Event(pygame.K_z, pygame.KMOD_CTRL))
    assert tab.doc.history.head == head


# --- saving -------------------------------------------------------------------


def test_a_save_records_the_head_the_encode_wrote(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.write_note(ctx, 0)
    head = tab.doc.history.head
    sirens_mode.save_to(ctx, tab, tmp_path / "song.wsng")
    sirens_mode.on_task_done(ctx, _Done(f"sirens-save:{tab.uid}", ctx.result))
    assert not tab.dirty and tab.doc.saved_head == head
    assert (tmp_path / "song.wsng").exists()


def test_a_failed_save_unlocks_the_tab():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    sirens_mode.on_task_failed(ctx, _Done(f"sirens-save:{tab.uid}", message="disk full"))
    assert not tab.saving


def test_an_open_that_failed_drops_the_path_off_the_recent_list():
    ctx = FakeCtx()
    sirens_mode.ensure(ctx)
    sirens_mode.remember_path(ctx, "/songs/gone.wsng")
    assert "/songs/gone.wsng" in sirens_mode.recent_paths(ctx)
    sirens_mode.on_task_failed(ctx, _Done("sirens-open:/songs/gone.wsng"))
    assert "/songs/gone.wsng" not in sirens_mode.recent_paths(ctx)


def test_the_song_filter_pairs_a_label_with_its_patterns():
    """portable-file-dialogs reads a filter list two at a time, so a third
    entry would become the *next row's label*."""
    assert len(sirens_mode.SONG_FILTER) == 2
    assert sirens_mode.SONG_FILTER[1] == "*.wsng"


# --- the guard ----------------------------------------------------------------


def test_a_clean_session_quits_without_a_question():
    ctx = FakeCtx()
    tab = _tab(ctx)
    assert not tab.dirty
    went = []
    assert sirens_mode.guard(ctx, "quit", lambda: went.append(1))
    assert went and ctx.confirms.pending is None


def test_one_question_covers_every_dirty_song():
    ctx = FakeCtx()
    _tab(ctx)
    _tab(ctx)
    sirens_mode.write_note(ctx, 0)
    went = []
    assert not sirens_mode.guard(ctx, "quit", lambda: went.append(1))
    assert not went and ctx.confirms.pending is not None


def test_closing_a_dirty_tab_asks_first():
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.write_note(ctx, 0)
    sirens_mode.close_tab(ctx, tab.uid)
    assert sirens_mode.ensure(ctx).docs, "the tab went without a question"
    ctx.confirms.pending.on_confirm()
    assert not sirens_mode.ensure(ctx).docs


# --- crash recovery -----------------------------------------------------------


def test_the_journal_provider_round_trips_a_song(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.write_note(ctx, 0)
    assert [slot.uid for slot in sirens_mode._journal_slots(ctx)] == [tab.uid]
    path = tmp_path / "copy.wsng"
    path.write_bytes(sirens_mode._journal_encode(tab))

    fresh = FakeCtx()
    assert sirens_mode._journal_adopt(fresh, path, {"title": "copy"})
    recovered = sirens_mode.ensure(fresh).docs[-1]
    assert "(recovered)" in recovered.title
    assert recovered.doc.pattern(
        recovered.doc.patterns[0].uid
    ).cells[0, 0, D.NOTE] == 48


def test_a_recovered_song_reads_dirty_until_it_is_saved_somewhere(tmp_path):
    """A clean recovered tab closes without a confirm, taking the journal copy
    -- the only surviving copy of the work -- with it."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    path = tmp_path / "copy.wsng"
    path.write_bytes(sirens_mode._journal_encode(tab))
    fresh = FakeCtx()
    sirens_mode._journal_adopt(fresh, path, {})
    assert sirens_mode.ensure(fresh).docs[-1].dirty


def test_a_busy_tab_is_not_journalled():
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.write_note(ctx, 0)
    tab.saving = True
    assert sirens_mode._journal_slots(ctx) == []


def test_the_provider_is_registered_under_its_own_kind():
    from warlock.studio import journal

    journal.ensure_providers()
    provider = journal.provider_for("sirens")
    assert provider is not None and provider.ext == wsng.SUFFIX


# --- registration -------------------------------------------------------------


def test_sirens_is_a_workspace_mode_that_binds_the_arrows():
    from warlock.studio import modes

    assert "sirens" in modes.WORKSPACE_MODES
    assert "sirens" in modes.WORK_MODES
    assert "sirens" in modes.NAV_KEY_MODES
    # Not a viewport mode: it draws no mesh, so ``_sync_viewer`` has nothing to
    # do for it and loading the library selection under a tracker would be
    # work for a picture nobody sees.
    assert "sirens" not in modes.VIEWPORT_MODES
    assert modes.RAIL_GROUPS[1][-1] == "sirens"


def test_the_workspace_has_a_skeleton_with_declared_share_keys():
    from warlock.studio import skeletons

    ctx = FakeCtx()
    columns = skeletons.for_mode(ctx, "sirens")
    ids = [slot.id for column in columns.values() for slot in column.slots]
    assert ids == [
        "sirens-transport",
        "sirens-orders",
        "sirens-instruments",
        "sirens-envelopes",
        "sirens-effects",
        "sirens-bridge",
    ]


# --- samples ------------------------------------------------------------------


def _wav(path: Path, *, seconds: float = 0.05, rate: int = 44100) -> Path:
    """A short real WAV on disk. ``wavout`` writes it, because a second encoder
    here would be a second answer to what this build reads back."""
    import numpy as np

    from warlock.studio.sirens import wavout

    count = max(1, int(rate * seconds))
    tone = np.sin(np.linspace(0.0, 40.0, count, dtype=np.float64)).astype(np.float32)
    wavout.write(path, tone, rate)
    return path


def _import(ctx: FakeCtx, tab: Any, path: Path, instrument: int | None = None) -> str:
    sirens_mode.import_sample(ctx, tab, path, instrument)
    sirens_mode.on_task_done(ctx, _Done(f"sirens-sample:{tab.uid}", ctx.result))
    return ctx.result.get("name", "") if isinstance(ctx.result, dict) else ""


def test_a_dropped_wav_lands_in_the_sample_table(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    _import(ctx, tab, _wav(tmp_path / "kick.wav"))
    assert ctx.submitted == [f"sirens-sample:{tab.uid}"]
    assert "kick" in tab.doc.samples
    assert tab.doc.samples["kick"].dtype.name == "float32"


def test_the_switch_flag_moves_the_window_only_once_the_sample_is_in(tmp_path, monkeypatch):
    """Muse's Open-in-Sirens leg. The flag rides the task rather than being a
    ``set_mode`` beside the press, so the mode changes after the adopt."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    switched: list[str] = []
    monkeypatch.setattr(sirens_mode, "set_mode", lambda state, mode: switched.append(mode))

    sirens_mode.import_sample(ctx, tab, _wav(tmp_path / "track.wav"), switch=True)
    assert switched == []
    sirens_mode.on_task_done(ctx, _Done(f"sirens-sample:{tab.uid}", ctx.result))
    assert switched == ["sirens"]
    assert "track" in tab.doc.samples


def test_a_sample_the_document_refused_does_not_move_the_window(tmp_path, monkeypatch):
    """``adopt_sample`` returns "" when the table would not take it -- a full
    song. Moving for that would put the refusal in a mode nobody asked for."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    switched: list[str] = []
    monkeypatch.setattr(sirens_mode, "set_mode", lambda state, mode: switched.append(mode))

    from warlock.studio import sirens_edit

    monkeypatch.setattr(sirens_edit, "adopt_sample", lambda c, t, result: "")
    sirens_mode.import_sample(ctx, tab, _wav(tmp_path / "track.wav"), switch=True)
    sirens_mode.on_task_done(ctx, _Done(f"sirens-sample:{tab.uid}", ctx.result))
    assert switched == []


def test_a_dropped_wav_does_not_move_the_window(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx)
    switched: list[str] = []
    monkeypatch.setattr(sirens_mode, "set_mode", lambda state, mode: switched.append(mode))
    _import(ctx, tab, _wav(tmp_path / "kick.wav"))
    assert switched == []


def test_a_file_past_the_byte_ceiling_is_refused_before_it_is_read(tmp_path, monkeypatch):
    """The door in front of the decoder: ``read_wav``'s frame count is in a
    header the file has to be *read* to reach, so what the file weighs is
    answered first, off the same constant."""
    from warlock.service.errors import ServiceError
    from warlock.studio.sirens import wavout

    monkeypatch.setattr(wavout, "MAX_SAMPLE_FRAMES", 1)
    path = _wav(tmp_path / "album.wav")
    with pytest.raises(ServiceError, match="past the 8 bytes"):
        sirens_io._decode_sample(path, None)


def test_an_imported_sample_is_resampled_to_the_render_rate(tmp_path):
    """``read_wav`` is the whole conversion and the only one: the synth advances
    a sample's phase in output samples, so a 22 kHz source would otherwise play
    an octave out."""
    from warlock.studio.sirens import synth

    ctx = FakeCtx()
    tab = _tab(ctx)
    _import(ctx, tab, _wav(tmp_path / "hat.wav", seconds=0.1, rate=22050))
    frames = tab.doc.samples["hat"].size
    assert abs(frames - int(0.1 * synth.SAMPLE_RATE)) <= 2


def test_a_sample_instrument_makes_a_sound_once_it_has_one(tmp_path):
    import numpy as np

    from warlock.studio.sirens import document as D
    from warlock.studio.sirens import synth

    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    channel = next(i for i, one in enumerate(doc.channels) if one.kind == "sample")
    instrument = next(one for one in doc.instruments if one.kind == "sample")
    pattern = doc.patterns[0]
    doc.set_cell(pattern.uid, 0, channel, D.NOTE, notes.SAMPLE_BASE_NOTE)
    doc.set_cell(pattern.uid, 0, channel, D.INSTRUMENT, instrument.uid)

    silent = synth.render_pattern(doc, pattern.uid)
    assert np.abs(silent).max() < 1e-6, "a sample instrument with no sample is silent"

    key = _import(ctx, tab, _wav(tmp_path / "snare.wav"), instrument.uid)
    assert doc.instrument(instrument.uid).sample == key
    assert np.abs(synth.render_pattern(doc, pattern.uid)).max() > 0.01


def test_the_sample_and_the_instrument_that_asked_for_it_are_one_step(tmp_path):
    """A picker opened from an instrument's sample field is one action; an undo
    that took the assignment back and left the sample behind would be a second
    press to finish one mistake."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    instrument = next(one for one in tab.doc.instruments if one.kind == "sample")
    depth = len(tab.doc.history)
    key = _import(ctx, tab, _wav(tmp_path / "clap.wav"), instrument.uid)
    assert len(tab.doc.history) == depth + 1
    tab.doc.undo()
    assert key not in tab.doc.samples
    assert tab.doc.instrument(instrument.uid).sample == ""


def test_two_files_with_one_name_are_two_samples(tmp_path):
    """Landing the second on the first's key would silently retune every note
    that used it."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    _import(ctx, tab, _wav(first / "kick.wav"))
    _import(ctx, tab, _wav(second / "kick.wav", seconds=0.08))
    assert len(tab.doc.samples) == 2


def test_removing_a_sample_leaves_the_instruments_that_named_it(tmp_path):
    """``remove_instrument``'s rule one level down: an instrument pointing at a
    key nothing answers to is silent and can be put back by an undo, while
    rewriting every instrument that used it cannot."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    instrument = next(one for one in tab.doc.instruments if one.kind == "sample")
    key = _import(ctx, tab, _wav(tmp_path / "tom.wav"), instrument.uid)
    assert sirens_mode.remove_sample(ctx, tab, key)
    assert key not in tab.doc.samples
    assert tab.doc.instrument(instrument.uid).sample == key
    assert tab.render_dirty


def test_a_sample_import_re_arms_the_renderer(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.render_dirty = False
    _import(ctx, tab, _wav(tmp_path / "rim.wav"))
    assert tab.render_dirty


def test_a_file_that_is_not_a_wav_is_refused_by_name(tmp_path):
    from warlock.service.errors import ServiceError

    path = tmp_path / "notes.wav"
    path.write_bytes(b"this is not a RIFF file")
    with pytest.raises(ServiceError, match="This sample could not be loaded"):
        sirens_io._decode_sample(path, None)


def test_a_refused_sample_does_not_unlock_a_save_running_beside_it():
    """The tab was never locked for a decode, and clearing ``saving`` here
    would hand the editing controls back mid-write."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    sirens_mode.on_task_failed(
        ctx, _Done(f"sirens-sample:{tab.uid}", message="that is not a WAV")
    )
    assert tab.saving


def test_a_sample_is_not_imported_into_a_tab_that_is_being_written(tmp_path):
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    sirens_mode.import_sample(ctx, tab, _wav(tmp_path / "shk.wav"))
    assert not ctx.submitted


def test_the_sample_filter_is_the_one_the_drop_router_advertises():
    """One list, so the picker and the drop cannot disagree about formats."""
    assert sirens_mode.SAMPLE_FILTER is sirens_io.SAMPLE_FILTER
    assert "*.wav" in sirens_mode.SAMPLE_FILTER[1]


def test_the_drop_router_imports_a_wav_rather_than_saying_it_cannot():
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._on_drop)
    sirens_branch = source.split('ctx.state.mode == "sirens"', 1)[1]
    branch = sirens_branch.split('ctx.state.mode in ("poser"', 1)[0]
    assert "sirens_mode.import_sample" in branch
    assert "not built yet" not in branch


# --- sound effects ------------------------------------------------------------
#
# A one-shot has been part of the document since Phase 1; what is asserted here
# is the *mode*'s half of it -- auditioning, and the selection that points the
# grid at an effect's pattern rather than at the song's.


def _oneshot(tab: Any, name: str = "coin") -> Any:
    doc = tab.doc
    one = doc.add_oneshot(name, rows=2)
    doc.set_cell(one.pattern, 0, 0, D.NOTE, 60)
    doc.set_cell(one.pattern, 0, 0, D.INSTRUMENT, doc.instruments[0].uid)
    return one


def test_the_caret_label_says_which_of_the_two_kinds_the_grid_is_on():
    """Adding an effect repoints the grid, and this is what says so.

    The panel that did it is in another column and can be scrolled away, so
    without a readout above the grid there is nothing on the editing surface
    itself distinguishing a song pattern from a sound effect -- and which one
    is loaded decides what every keystroke changes.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)

    song = sirens_mode.caret_pattern_label(ctx)
    assert song
    assert "sound effect" not in song

    one = _oneshot(tab, "coin")
    # ``add_oneshot`` moved the caret onto the effect's own pattern; the label
    # follows the caret rather than being told separately.
    sirens_mode.set_caret(ctx, pattern=one.pattern)
    assert sirens_mode.caret_pattern_label(ctx) == "coin - sound effect"

    # Back to the song's pattern and the label goes back with it.
    state.pattern = tab.doc.patterns[0].uid
    assert "sound effect" not in sirens_mode.caret_pattern_label(ctx)


def test_the_caret_label_is_empty_rather_than_wrong_with_nothing_open():
    """A pattern the document no longer has is 'nothing', not a traceback.

    Deleting a pattern out from under the caret is the ordinary way this
    happens, and the toolbar draws every frame.
    """
    ctx = FakeCtx()
    assert sirens_mode.caret_pattern_label(ctx) == ""
    _tab(ctx)
    sirens_mode.ensure(ctx).pattern = 999999
    assert sirens_mode.caret_pattern_label(ctx) == ""


@pytest.fixture
def _device(monkeypatch):
    """A mixer that answers, and remembers what it was handed."""
    from warlock.studio import sirens_audio

    played: list[Any] = []
    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(sirens_audio, "play", lambda pcm, *_a, **_k: played.append(pcm) or True)
    return played


def test_auditioning_an_effect_renders_that_effect_rather_than_the_song(_device):
    """The one thing an Audition button must not do is play the music."""
    from warlock.studio.sirens import synth, wavout

    ctx = FakeCtx()
    tab = _tab(ctx)
    one = _oneshot(tab)
    assert sirens_mode.audition(ctx, tab, one.uid)
    sirens_mode.on_task_done(ctx, _Done(f"{sirens_mode.AUDITION_PREFIX}{tab.uid}", ctx.result))
    expected = wavout.to_int16(synth.render_oneshot(tab.doc, one.uid))
    assert _device and (_device[0] == expected).all()


def test_an_audition_never_lands_on_the_songs_buffer(_device):
    """``sirens-render:`` is the song's key and its arm writes ``tab.pcm``, which
    is what Space plays. An effect adopted there would replace the song until
    the next edit re-armed the renderer."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _render(ctx, tab)
    song = tab.pcm
    generation = tab.render_generation
    one = _oneshot(tab)
    sirens_mode.audition(ctx, tab, one.uid)
    sirens_mode.on_task_done(ctx, _Done(f"{sirens_mode.AUDITION_PREFIX}{tab.uid}", ctx.result))
    assert tab.pcm is song and tab.render_generation == generation


def test_an_audition_with_no_device_says_so_rather_than_rendering():
    ctx = FakeCtx()
    tab = _tab(ctx)
    one = _oneshot(tab)
    assert not sirens_mode.audition(ctx, tab, one.uid)
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "warn"


def test_an_effect_that_is_not_in_the_song_is_not_auditioned(_device):
    ctx = FakeCtx()
    tab = _tab(ctx)
    assert not sirens_mode.audition(ctx, tab, 9999)
    assert ctx.submitted == []


def test_a_failed_audition_does_not_unlock_a_save_running_beside_it():
    """``sirens-sample``'s clause, for the same reason: the tab was never locked
    for an audition, so clearing ``saving`` here would unlock a save."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    sirens_mode.on_task_failed(
        ctx, _Done(f"{sirens_mode.AUDITION_PREFIX}{tab.uid}", message="nope")
    )
    assert tab.saving


def test_an_effect_removed_under_the_selection_clears_it_rather_than_moving_it():
    """Unlike the instrument, which needs *some* answer for the next note: a
    selected effect is what the grid is editing, and silently switching the grid
    to a different one after an undo is the caret bug ``clamp_caret`` exists to
    prevent, one level up."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    one = _oneshot(tab)
    state = sirens_mode.ensure(ctx)
    state.oneshot = one.uid
    tab.doc.remove_oneshot(one.uid)
    sirens_mode.clamp_caret(ctx, tab)
    assert state.oneshot is None


def test_a_tab_switch_drops_the_effect_selection():
    ctx = FakeCtx()
    first = _tab(ctx)
    one = _oneshot(first)
    state = sirens_mode.ensure(ctx)
    state.oneshot = one.uid
    sirens_mode.new_document(ctx)
    assert state.oneshot is None


# --- the playhead: a bisect of what the renderer did --------------------------
#
# The estimate this replaced was seconds over the document's own seconds-per-row,
# which describes one imaginary pattern of unbounded length: right for a
# one-pattern song and wrong for every other, and wrong again after every
# ``Fxx``, ``Bxx`` and ``Dxx`` (the 2026-09-02 review, section 8).


def _two_pattern_song(ctx: FakeCtx) -> Any:
    """A song whose order list is two different patterns, both rendered."""
    tab = _tab(ctx)
    doc = tab.doc
    first = doc.patterns[0].uid
    second = doc.add_pattern().uid
    doc.set_order(list(doc.order) + [second])
    _render(ctx, tab)
    return tab, first, second


def _sounding(monkeypatch, tab: Any, seconds: float, *, anchor: int = 0) -> None:
    """Put this tab's own render on the (fake) channel at ``seconds``.

    The snapshot is what the playhead bisects since S2 -- the live ``marks`` may
    already belong to a render the mixer has never heard -- so a test that only
    faked the device would be asking about audio that, as far as the tab is
    concerned, was never started.
    """
    from warlock.studio import sirens_audio
    from warlock.studio.sirens_state import Sounding

    tab.sounding = Sounding(
        marks=tab.marks, anchor=anchor, generation=tab.render_generation
    )
    monkeypatch.setattr(sirens_audio, "tag", lambda: tab.uid)
    monkeypatch.setattr(sirens_audio, "position", lambda: seconds)


def test_a_render_carries_a_row_map(monkeypatch):
    ctx = FakeCtx()
    tab, first, second = _two_pattern_song(ctx)

    assert tab.marks, "the render says which row it played where"
    offsets = [mark[0] for mark in tab.marks]
    assert offsets == sorted(offsets), "ascending, so a bisect is legal"
    assert {mark[2] for mark in tab.marks} == {first, second}


def test_the_playhead_is_none_while_the_song_is_in_another_pattern(monkeypatch):
    """The half the estimate could not express at all: it answered with a row
    number whatever was playing, so a two-pattern song highlighted a row of the
    pattern on screen because a different pattern had reached it."""
    ctx = FakeCtx()
    tab, first, second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    state.follow = False
    state.pattern = first
    # A moment inside the *second* entry, by asking the map itself where that is.
    at = next(mark[0] for mark in tab.marks if mark[2] == second)
    _sounding(monkeypatch, tab, at / 44100.0)

    assert sirens_mode.playhead_mark(ctx, tab)[1] == second
    assert sirens_mode.playhead_row(ctx, tab) is None

    state.pattern = second
    assert sirens_mode.playhead_row(ctx, tab) == 0


def test_the_playhead_is_the_row_the_renderer_was_on(monkeypatch):
    ctx = FakeCtx()
    tab, first, _second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    state.follow = False
    state.pattern = first
    wanted = next(mark for mark in tab.marks if mark[2] == first and mark[3] == 3)
    _sounding(monkeypatch, tab, (wanted[0] + 1) / 44100.0)

    assert sirens_mode.playhead_row(ctx, tab) == 3


def test_a_sound_effect_on_the_channel_is_not_the_songs_playhead(monkeypatch):
    """One channel: an audition replaces the song on it, and bisecting the
    song's map against an effect's clock walks rows nothing is playing."""
    from warlock.studio import sirens_audio

    ctx = FakeCtx()
    tab, _first, _second = _two_pattern_song(ctx)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "")
    monkeypatch.setattr(sirens_audio, "position", lambda: 0.5)

    assert sirens_mode.playhead_mark(ctx, tab) is None


def test_following_moves_the_caret_onto_the_sounding_row(monkeypatch):
    """Follow used to scroll the view and leave the caret behind, so the row
    under the highlight was not the row a keystroke wrote to -- and on a
    two-pattern song the highlight was not even in the pattern being edited."""
    ctx = FakeCtx()
    tab, first, second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    state.follow = True
    state.pattern = first
    state.row = 0
    state.digit = 1
    at = next(mark for mark in tab.marks if mark[2] == second and mark[3] == 2)
    _sounding(monkeypatch, tab, (at[0] + 1) / 44100.0)

    assert sirens_mode.follow_playhead(ctx) is True
    assert (state.pattern, state.row) == (second, 2)
    assert state.digit == 0, "everything that moves the caret clears the nibble"
    assert sirens_mode.playhead_row(ctx, tab) == 2


def test_following_leaves_the_caret_alone_when_it_is_off(monkeypatch):
    ctx = FakeCtx()
    tab, first, second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    state.follow = False
    state.pattern = first
    state.row = 7
    at = next(mark for mark in tab.marks if mark[2] == second)
    _sounding(monkeypatch, tab, (at[0] + 1) / 44100.0)

    assert sirens_mode.follow_playhead(ctx) is False
    assert (state.pattern, state.row) == (first, 7)


def test_nothing_playing_is_no_playhead_and_no_follow():
    ctx = FakeCtx()
    tab, _first, _second = _two_pattern_song(ctx)
    assert sirens_mode.playhead_mark(ctx, tab) is None
    assert sirens_mode.playhead_row(ctx, tab) is None
    assert sirens_mode.follow_playhead(ctx) is False


# --- the order list's own doors -------------------------------------------------


def test_the_caret_on_a_sound_effect_is_named_so_the_order_button_can_refuse():
    """Adding an effect mints a pattern of its own and points the grid at it,
    so "+ To order" used to append a coin pickup into the middle of the song
    and say nothing (the 2026-09-02 review, section 8)."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    assert sirens_mode.oneshot_name_for_caret(ctx, tab) == ""

    effect = tab.doc.add_oneshot(name="coin")
    state.pattern = effect.pattern
    assert sirens_mode.oneshot_name_for_caret(ctx, tab) == "coin"

    state.pattern = tab.doc.patterns[0].uid
    assert sirens_mode.oneshot_name_for_caret(ctx, tab) == ""


def test_deleting_an_unused_pattern_asks_nothing():
    ctx = FakeCtx()
    tab = _tab(ctx)
    spare = tab.doc.add_pattern().uid

    sirens_mode.confirm_remove_pattern(ctx, tab, spare, used=0)

    assert ctx.confirms.pending is None
    assert tab.doc.pattern(spare) is None


def test_deleting_a_pattern_the_order_plays_asks_first_and_says_what_goes():
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    first = doc.patterns[0].uid
    second = doc.add_pattern().uid
    doc.set_order([first, second, second])

    sirens_mode.confirm_remove_pattern(ctx, tab, second, used=2)

    pending = ctx.confirms.pending
    assert pending is not None and "2 time(s)" in pending.message
    assert doc.pattern(second) is not None, "nothing happens before the answer"

    pending.on_confirm()
    assert doc.pattern(second) is None
    assert doc.order == [first]


def test_moving_an_entry_carries_the_loop_point_with_it():
    """The loop is an index into the order list, so moving entries under it
    repoints it at whatever landed there -- a song looping from somewhere the
    user never chose."""
    from warlock.studio.panes.sirens_orders import moved_loop

    # The moved entry takes its own loop with it.
    assert moved_loop(2, 2, 0) == 0
    # An entry the move stepped over shifts the other way.
    assert moved_loop(0, 2, 0) == 1
    assert moved_loop(3, 1, 3) == 2
    # And one the move did not reach does not move.
    assert moved_loop(5, 1, 3) == 5
    assert moved_loop(-1, 1, 3) == -1
    assert moved_loop(2, 1, 1) == 2


# --- mute and solo --------------------------------------------------------------
#
# View state, not the song: a mute is how a person listens to what they are
# writing, and a ``.wsng`` remembering one would hand somebody else a song with
# a missing part. It reaches the mix through the render.


def test_muting_a_channel_takes_it_out_of_the_mix_and_re_renders():
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    _render(ctx, tab)
    assert tab.render_dirty is False

    second = doc.channels[1].uid
    assert sirens_mode.toggle_mute(ctx, second, tab) is True
    assert tab.render_dirty is True, "a mute the mix has not heard yet is not a mute"
    assert sirens_mode.audible_channels(doc, tab) == (0, 2, 3, 4)

    assert sirens_mode.toggle_mute(ctx, second, tab) is False
    assert sirens_mode.audible_channels(doc, tab) == (0, 1, 2, 3, 4)


def test_solo_wins_over_every_mute():
    """Soloing to check a bass line and then unsoloing must not have to undo
    four mutes."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    sirens_mode.toggle_mute(ctx, doc.channels[0].uid, tab)
    sirens_mode.toggle_mute(ctx, doc.channels[2].uid, tab)

    assert sirens_mode.toggle_solo(ctx, doc.channels[2].uid, tab) == doc.channels[2].uid
    assert sirens_mode.audible_channels(doc, tab) == (2,)

    assert sirens_mode.toggle_solo(ctx, doc.channels[2].uid, tab) == -1
    assert sirens_mode.audible_channels(doc, tab) == (1, 3, 4), "the mutes stood"


def test_a_solo_on_a_channel_the_song_no_longer_has_is_not_silence():
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    tab.solo = 999_999

    assert sirens_mode.audible_channels(doc, tab) == tuple(range(len(doc.channels)))


def test_the_header_can_tell_muted_from_merely_unheard():
    """Two different facts: a channel that is not the soloed one is silent
    without being muted, and one button lighting for both would lie about what
    a click undoes."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    first, second = doc.channels[0].uid, doc.channels[1].uid
    sirens_mode.toggle_mute(ctx, first, tab)
    assert sirens_mode.channel_state(ctx, first) == (True, False, False)

    sirens_mode.toggle_solo(ctx, second, tab)
    assert sirens_mode.channel_state(ctx, first) == (True, False, False)
    assert sirens_mode.channel_state(ctx, second) == (False, True, True)
    third = doc.channels[2].uid
    assert sirens_mode.channel_state(ctx, third) == (False, False, False)


def test_a_muted_render_keeps_the_row_map_the_full_one_has():
    """The effect column survives a mute, so the mix stays sample-aligned and
    the playhead does not move when a channel is silenced."""
    from warlock.studio.sirens import synth

    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    doc.set_cell(doc.patterns[0].uid, 0, 0, 0, 60)
    _whole, _loop, marks = synth.render_marked(doc)
    _muted, _loop2, muted_marks = synth.render_only(doc, {1})

    assert muted_marks == marks


def test_a_mute_does_not_touch_the_document():
    ctx = FakeCtx()
    tab = _tab(ctx)
    before = len(tab.doc.history)
    sirens_mode.toggle_mute(ctx, tab.doc.channels[0].uid, tab)
    assert len(tab.doc.history) == before, "a mute is not an edit"
    assert not tab.doc.dirty


# --- playing less than the whole song from the top ------------------------------


def _audible(monkeypatch):
    from warlock.studio import sirens_audio

    played: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(
        sirens_audio, "play", lambda pcm, **kw: played.append((pcm, kw)) or True
    )
    return played


def test_playing_from_the_caret_slices_the_buffer_at_that_row(monkeypatch):
    """The row map's second job: writing bar 40 of a three-minute song and
    having to hear the first two minutes to check it is the complaint."""
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab, _first, second = _two_pattern_song(ctx)
    wanted = next(mark for mark in tab.marks if mark[2] == second and mark[3] == 2)
    sirens_mode.set_caret(ctx, pattern=second, row=2)

    assert sirens_mode.play_from_caret(ctx, tab) is True

    pcm, kwargs = played[-1]
    assert len(pcm) == len(tab.pcm) - wanted[0]
    assert kwargs["tag"] == tab.uid
    assert tab.sounding.anchor == wanted[0]


def test_the_playhead_is_still_the_songs_row_after_playing_from_the_caret(monkeypatch):
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab, _first, second = _two_pattern_song(ctx)
    sirens_mode.set_caret(ctx, pattern=second, row=2)
    sirens_mode.play_from_caret(ctx, tab)
    assert played

    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "tag", lambda: tab.uid)
    monkeypatch.setattr(sirens_audio, "position", lambda: 0.0)
    assert sirens_mode.playhead_row(ctx, tab) == 2


def test_a_row_the_order_never_reaches_says_so_rather_than_playing_the_top(monkeypatch):
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    spare = tab.doc.add_pattern().uid
    _render(ctx, tab)
    sirens_mode.set_caret(ctx, pattern=spare, row=0)

    assert sirens_mode.play_from_caret(ctx, tab) is False
    assert played == []
    assert any("never reaches" in message for message, _kind in ctx.toasts)


def test_playing_one_pattern_never_touches_the_songs_buffer(monkeypatch):
    """``AUDITION_PREFIX``'s rule: an effect -- or a pattern -- landing on
    ``tab.pcm`` would replace the song until the next edit re-armed the
    renderer."""
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    _render(ctx, tab)
    song = tab.pcm
    sirens_mode.set_caret(ctx, pattern=tab.doc.patterns[0].uid)

    assert sirens_mode.play_pattern(ctx, tab) is True
    assert ctx.submitted[-1] == f"{sirens_mode.PATTERN_PREFIX}{tab.uid}"
    sirens_mode.on_task_done(
        ctx, _Done(f"{sirens_mode.PATTERN_PREFIX}{tab.uid}", ctx.result)
    )

    assert tab.pcm is song, "the song is still the song"
    assert played[-1][1]["tag"].startswith("pattern:"), "and not the song's playhead"


def test_loop_playback_asks_the_mixer_to_repeat(monkeypatch):
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    _render(ctx, tab)
    state = sirens_mode.ensure(ctx)

    assert sirens_mode.play(ctx, tab) is True
    assert played[-1][1]["loops"] == 0

    state.loop_playback = True
    assert sirens_mode.play(ctx, tab) is True
    assert played[-1][1]["loops"] == -1


# --- the 2026-09-05 playback defects (S1-S6) ---------------------------------
#
# Six defects found by a code read of a mode nobody has heard yet (TODO P14),
# all of them green under the suite as it stood. Each test below is the claim
# its name makes, and each fails against the code as it was.


def test_a_pattern_that_finishes_rendering_after_stop_is_not_played(monkeypatch):
    """S1. ``on_task_done`` handed every successful audition straight to the
    mixer with no freshness check at all, so pressing a key and then Stop
    played the sound anyway the moment its render landed. Against the unfixed
    code this records one ``play`` call after an explicit Stop.
    """
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    _render(ctx, tab)
    state = sirens_mode.ensure(ctx)
    state.pattern = tab.doc.patterns[0].uid

    assert sirens_mode.play_pattern(ctx, tab) is True
    requested, result = ctx.tag, ctx.result
    sirens_mode.stop(ctx)  # the user changes their mind while it renders

    sirens_mode.on_task_done(
        ctx, _Done(f"sirens-pattern:{tab.uid}", result, tag=requested)
    )
    assert played == [], "a withdrawn request does not sound"


def test_the_audition_the_user_is_still_waiting_for_does_play(monkeypatch):
    """S1's other half: the freshness check must not swallow the ordinary
    case, or Sirens would simply go silent."""
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    _render(ctx, tab)
    state = sirens_mode.ensure(ctx)
    state.pattern = tab.doc.patterns[0].uid

    assert sirens_mode.play_pattern(ctx, tab) is True
    sirens_mode.on_task_done(
        ctx, _Done(f"sirens-pattern:{tab.uid}", ctx.result, tag=ctx.tag)
    )
    assert len(played) == 1


def test_the_playhead_bisects_the_render_the_mixer_is_actually_playing(monkeypatch):
    """S2. ``adopt_render`` swaps ``pcm``, ``loop`` and ``marks`` while the
    device is still playing a ``Sound`` built from the *previous* buffer, and
    the playhead used to bisect the live fields -- passing its tag check and
    then reading the new map against the old audio, which with follow mode on
    walks the caret to a row nothing is playing. Against the unfixed code this
    reports the replacement map's row 7 instead of the sounding render's.
    """
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab, _first, second = _two_pattern_song(ctx)
    assert sirens_mode.play(ctx, tab) is True
    assert played

    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "tag", lambda: tab.uid)
    monkeypatch.setattr(sirens_audio, "position", lambda: 0.0)
    sounding_now = sirens_mode.playhead_mark(ctx, tab)
    assert sounding_now is not None and sounding_now[2] != 7

    # A re-render lands: same tab, a map that says something else entirely.
    tab.adopt_render(tab.pcm, None, ((0, 9, second, 7),))
    assert sirens_mode.playhead_mark(ctx, tab) == sounding_now


def test_playing_from_the_caret_starts_at_the_order_entry_the_caret_is_in(monkeypatch):
    """S3. A pattern used at two places in the order list is one uid, and the
    lookup broke on the first mark whose *pattern* matched -- so a chorus at
    entries 00 and 02 always played from 00, however far down the song the
    user was working. Against the unfixed code this starts at entry 00's
    offset, two thirds of the song early.
    """
    _audible(monkeypatch)
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    chorus = doc.patterns[0].uid
    verse = doc.add_pattern().uid
    doc.set_order([chorus, verse, chorus])
    _render(ctx, tab)

    sirens_mode.set_caret(ctx, pattern=chorus, row=2, order_index=2)
    assert sirens_mode.play_from_caret(ctx, tab) is True

    wanted = next(
        mark for mark in tab.marks if mark[1] == 2 and mark[2] == chorus and mark[3] == 2
    )
    first_time = next(
        mark for mark in tab.marks if mark[1] == 0 and mark[2] == chorus and mark[3] == 2
    )
    assert wanted[0] != first_time[0], "the two occurrences are at different offsets"
    assert tab.sounding.anchor == wanted[0]


def test_the_highlight_names_the_order_entry_and_not_merely_the_pattern(monkeypatch):
    """S3, the half a listener sees. ``playhead_row`` dropped the order index
    too, so while the song was in entry 00 the grid lit up for a caret sitting
    in entry 02 of the same pattern. Against the unfixed code this returns row
    0 rather than ``None``.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    doc = tab.doc
    chorus = doc.patterns[0].uid
    verse = doc.add_pattern().uid
    doc.set_order([chorus, verse, chorus])
    _render(ctx, tab)
    sirens_mode.set_caret(ctx, pattern=chorus, row=0, order_index=2)

    _sounding(monkeypatch, tab, 0.0)  # the song is at the top: entry 00
    assert sirens_mode.playhead_mark(ctx, tab)[0] == 0
    assert sirens_mode.playhead_row(ctx, tab) is None, "the caret is in entry 02"


def test_from_the_caret_with_loop_playback_repeats_the_song_not_its_tail(monkeypatch):
    """S4, which is M10's Muse bug still live in Sirens. ``loops=-1`` on the
    slice ``pcm[offset:]`` repeats only what is left of the song from that row
    onward and never comes back to bar 1. Against the unfixed code the buffer
    handed to the mixer is shorter than the song by exactly the caret's offset.
    """
    played = _audible(monkeypatch)
    ctx = FakeCtx()
    tab, _first, second = _two_pattern_song(ctx)
    state = sirens_mode.ensure(ctx)
    state.loop_playback = True
    wanted = next(mark for mark in tab.marks if mark[2] == second and mark[3] == 2)
    sirens_mode.set_caret(ctx, pattern=second, row=2)

    assert sirens_mode.play_from_caret(ctx, tab) is True
    pcm, kwargs = played[-1]
    assert kwargs["loops"] == -1
    assert len(pcm) == len(tab.pcm), "the whole song repeats, rotated"
    assert tab.sounding.wrap == len(tab.pcm)

    # And the rotation is unwound: at the instant it starts, the playhead is on
    # the row the caret was on, not on row 0 of the song.
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "tag", lambda: tab.uid)
    monkeypatch.setattr(sirens_audio, "position", lambda: 0.0)
    assert sirens_mode.playhead_mark(ctx, tab) == (wanted[1], wanted[2], wanted[3])


def test_a_mute_in_one_song_leaves_the_other_song_alone():
    """S5. ``muted``/``solo`` lived on ``SirensState``, shared by every tab,
    while channels are identified by uid -- and ``document.reserve_uid`` starts
    each document's count over, so two songs carry the *same* channel uids.
    Against the unfixed code muting channel 1 of the first song silences
    channel 1 of the second, which reads as a channel that went quiet on its
    own.
    """
    ctx = FakeCtx()
    # The same file, opened twice -- which is what a person does when they want
    # to compare a change against the version on disk. Channel uids come out of
    # the file, so the two tabs carry identical ones.
    data = wsng.wsng_bytes(_tab(ctx).doc)
    first = sirens_mode.adopt(ctx, wsng.read_wsng(data), title="one")
    second = sirens_mode.adopt(ctx, wsng.read_wsng(data), title="two")
    assert [one.uid for one in first.doc.channels] == [
        one.uid for one in second.doc.channels
    ], "the same file really does give two tabs the same channel uids"

    victim = first.doc.channels[1].uid
    assert sirens_mode.toggle_mute(ctx, victim, first) is True

    assert sirens_mode.audible_channels(first.doc, first) == (0, 2, 3, 4)
    assert sirens_mode.audible_channels(second.doc, second) == (0, 1, 2, 3, 4)


def test_switching_tabs_stops_the_song_that_was_playing(monkeypatch):
    """S6. There is one mixer channel, so tab A went on sounding under tab B --
    and because the transport read the *global* ``playing()``, B showed a Stop
    button that silenced A. Against the unfixed code nothing stops and A's
    buffer is still on the channel.
    """
    stopped: list[bool] = []
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "stop", lambda: stopped.append(True))
    ctx = FakeCtx()
    first = _tab(ctx)
    second = _tab(ctx)

    state = sirens_mode.ensure(ctx)
    state.activate(first.uid)
    assert stopped, "the device is silenced on the way out"
    assert first.sounding is None and second.sounding is None
