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

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", **_extra: Any) -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.sirens = None
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
    def __init__(self, key: str, result: Any = None, message: str = "") -> None:
        self.key = key
        self.result = result
        self.message = message


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
    ):
        tab.render_dirty = False
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
    -- a piano row that fired everywhere would make four columns untypable."""
    import pygame

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.column = D.EFFECT
    assert not sirens_mode.handle_key(ctx, _Event(pygame.K_z))
    assert tab.doc.pattern(state.pattern).cells[0, 0, D.NOTE] == notes.EMPTY


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
