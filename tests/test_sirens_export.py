"""Sirens' export: ``song.wav``, ``stems/`` and ``sfx/`` into a chosen folder.

**The invariant this file is here for.** A ``.wsng`` is the composition and every
WAV is a pure function of it (``docs/INVARIANTS.md``), which is a claim with two
halves: the bytes must not depend on when they were written, and the *names* must
not depend on anything but the document either. So the byte-identity assertion
below is not a nicety about diffs -- it is the statement that an export is
reproducible, which is what lets a build script check one in and a reviewer
diff two.

**Filenames come from user text**, which is the one shape ``wsng.py`` refused
outright: it numbers its archive members precisely so that ``../`` and ``CON``
cannot reach a filesystem through a name somebody typed. An export cannot take
that way out -- ``sfx/coin.wav`` is the point of the folder -- so the names come
through sanitised, and the tests for the two hostile shapes are here.

Nothing in this file opens a picker or touches a sound device. ``export_plan``
is pure and takes a directory; ``export_to`` is the door a caller with a
destination already in hand comes through, and it is what the pane's button
reaches after the picker has answered.
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_sirens_mode import FakeCtx, _Done, _tab

from warlock.studio import sirens_io, sirens_mode
from warlock.studio.sirens import document as D
from warlock.studio.sirens import notes, synth, wavout


def _chunks(raw: bytes) -> dict[bytes, bytes]:
    """The file's chunks, read with ``struct`` rather than with the writer's own
    assumptions -- ``tests/sirens/test_wavout.py``'s rule, for its reason."""
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    out: dict[bytes, bytes] = {}
    at = 12
    while at + 8 <= len(raw):
        tag = raw[at : at + 4]
        size = struct.unpack("<I", raw[at + 4 : at + 8])[0]
        out[tag] = raw[at + 8 : at + 8 + size]
        at += 8 + size + (size % 2)
    return out


def _frames(raw: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(raw)) as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")


def _song(ctx: FakeCtx, *, rows: int = 4) -> Any:
    """A short song with a note on two channels. Short deliberately: an export
    renders the whole thing once per channel, and the default 64-row pattern
    turns every test in this file into most of a second."""
    tab = _tab(ctx)
    doc = tab.doc
    pattern = doc.patterns[0]
    doc.resize_pattern(pattern.uid, rows)
    instrument = doc.instruments[0].uid
    for channel, note in ((0, 48), (1, 55)):
        doc.set_cell(pattern.uid, 0, channel, D.NOTE, note)
        doc.set_cell(pattern.uid, 0, channel, D.INSTRUMENT, instrument)
    return tab


def _effect(tab: Any, name: str) -> Any:
    """One sound effect with an audible note in it."""
    doc = tab.doc
    one = doc.add_oneshot(name, rows=2)
    doc.set_cell(one.pattern, 0, 0, D.NOTE, 60)
    doc.set_cell(one.pattern, 0, 0, D.INSTRUMENT, doc.instruments[0].uid)
    return one


def _export(ctx: FakeCtx, tab: Any, directory: Path) -> dict[str, Any]:
    """Run the export the way the app does: submit, then adopt the result."""
    sirens_mode.export_to(ctx, tab, directory)
    result = ctx.result
    sirens_mode.on_task_done(ctx, _Done(f"{sirens_io.EXPORT_PREFIX}{tab.uid}", result))
    return result


def _written(directory: Path) -> set[str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob("*")
        if path.is_file()
    }


# --- what lands ---------------------------------------------------------------


def test_every_promised_file_lands_in_the_chosen_folder(tmp_path):
    ctx = FakeCtx()
    tab = _song(ctx)
    _effect(tab, "coin")
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    assert _written(out) == {
        "song.wav",
        *(f"stems/{name}.wav" for name in ("Pulse-1", "Pulse-2", "Triangle", "Noise", "Sample")),
        "sfx/coin.wav",
    }


def test_a_song_with_no_effects_writes_no_sfx_folder(tmp_path):
    """An empty ``sfx/`` would be a folder a build script has to special-case."""
    ctx = FakeCtx()
    tab = _song(ctx)
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    assert not (out / sirens_io.SFX_DIR).exists()


def test_the_report_names_the_folder_and_counts_the_files(tmp_path):
    ctx = FakeCtx()
    tab = _song(ctx)
    _effect(tab, "coin")
    out = tmp_path / "audio"
    result = _export(ctx, tab, out)
    assert result == {"directory": str(out), "files": 7}
    assert any("7 file(s)" in message and str(out) in message for message, _ in ctx.toasts)


def test_an_export_does_not_mark_the_song_saved(tmp_path):
    """It writes files *derived* from the document and leaves the document
    exactly as dirty as it was. Falling through to the save arm would call
    ``mark_saved`` and drop the crash copy of work that is still only in
    memory."""
    ctx = FakeCtx()
    tab = _song(ctx)
    assert tab.dirty
    _export(ctx, tab, tmp_path / "audio")
    assert tab.dirty and not tab.saving


def test_the_export_task_reads_a_snapshot_rather_than_the_document(tmp_path):
    """``wsng.wsng_bytes`` is taken on the frame thread and the task re-reads a
    document from it, so an edit made while an export is running cannot tear a
    numpy view out from under it -- ``request_render``'s rule."""
    ctx = FakeCtx(accept=False)
    tab = _song(ctx)
    sirens_mode.export_to(ctx, tab, tmp_path / "audio")
    assert ctx.submitted == [f"{sirens_io.EXPORT_PREFIX}{tab.uid}"]
    assert not (tmp_path / "audio").exists()


# --- the loop -----------------------------------------------------------------


def test_the_loop_points_survive_into_the_songs_smpl_chunk(tmp_path):
    """The whole reason ``wavout`` is hand-rolled. A soundtrack whose loop lives
    in a sidecar is a soundtrack that does not loop, because the engine reading
    the WAV never opens the sidecar."""
    ctx = FakeCtx()
    tab = _song(ctx)
    tab.doc.set_song(loop_order=0)
    out = tmp_path / "audio"
    _export(ctx, tab, out)

    _pcm, loop = synth.render(tab.doc)
    assert loop is not None
    smpl = _chunks((out / sirens_io.SONG_NAME).read_bytes())[b"smpl"]
    assert struct.unpack("<I", smpl[28:32])[0] == 1
    _cue, kind, start, end, _frac, _plays = struct.unpack("<6I", smpl[36:60])
    assert kind == wavout.LOOP_FORWARD
    assert (start, end) == (loop[0], loop[1] - 1)


def test_a_song_that_does_not_loop_writes_no_smpl_chunk(tmp_path):
    ctx = FakeCtx()
    tab = _song(ctx)
    assert tab.doc.loop_order < 0
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    assert b"smpl" not in _chunks((out / sirens_io.SONG_NAME).read_bytes())


def test_every_stem_carries_the_songs_loop_points(tmp_path):
    """A stem whose loop disagreed with the mix's would be unusable for the one
    thing stems are for -- muting a layer at runtime and looping the rest."""
    ctx = FakeCtx()
    tab = _song(ctx)
    tab.doc.set_song(loop_order=0)
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    song = _chunks((out / sirens_io.SONG_NAME).read_bytes())[b"smpl"]
    for path in sorted((out / sirens_io.STEM_DIR).glob("*.wav")):
        assert _chunks(path.read_bytes())[b"smpl"] == song


# --- the stems ----------------------------------------------------------------


def test_a_stem_holds_only_its_own_channel(tmp_path):
    """Asserted against a render of the same document with the *other* channel's
    notes taken out by hand, rather than against an amplitude somebody eyeballed
    -- so it is a claim about which notes are in the file, not about how loud it
    is."""
    ctx = FakeCtx()
    tab = _song(ctx)
    out = tmp_path / "audio"
    _export(ctx, tab, out)

    doc = tab.doc
    pattern = doc.patterns[0]
    for column in (D.NOTE, D.INSTRUMENT):
        doc.set_cell(pattern.uid, 0, 1, column, notes.EMPTY)
    soloed, _loop = synth.render(doc)
    assert (
        (out / sirens_io.STEM_DIR / "Pulse-1.wav").read_bytes()
        == wavout.wav_bytes(soloed, synth.SAMPLE_RATE)
    )


def test_a_silent_channels_stem_is_silent(tmp_path):
    """The other direction, and the one a wrongly-indexed solo would pass the
    first test while failing: a channel nobody wrote a note on exports silence,
    not the neighbour's part."""
    ctx = FakeCtx()
    tab = _song(ctx)
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    assert not _frames((out / sirens_io.STEM_DIR / "Triangle.wav").read_bytes()).any()


def test_a_stem_keeps_a_tempo_effect_written_on_another_channel(tmp_path):
    """``Bxx``, ``Cxx``, ``Dxx`` and ``Fxx`` are the *player's* rather than a
    voice's, and any channel may carry them. A stem rendered from a grid with
    the other channels wiped clean would run at a different tempo than the mix
    it is supposed to line up with -- so the effect column survives the solo and
    only the note, instrument and volume columns are blanked."""
    ctx = FakeCtx()
    tab = _song(ctx, rows=8)
    doc = tab.doc
    pattern = doc.patterns[0]
    doc.set_cell(pattern.uid, 0, 1, D.EFFECT, synth.FX_TEMPO)
    doc.set_cell(pattern.uid, 0, 1, D.PARAM, D.MAX_TEMPO)
    out = tmp_path / "audio"
    _export(ctx, tab, out)

    song = _frames((out / sirens_io.SONG_NAME).read_bytes())
    stem = _frames((out / sirens_io.STEM_DIR / "Pulse-1.wav").read_bytes())
    assert stem.size == song.size


# --- reproducibility ----------------------------------------------------------


def test_re_exporting_an_unchanged_song_writes_the_same_bytes(tmp_path):
    """The invariant. Two exports of a document nobody has touched are the same
    files, byte for byte -- no timestamp in the RIFF and no name decided by
    anything but the document."""
    ctx = FakeCtx()
    tab = _song(ctx)
    _effect(tab, "coin")
    first, second = tmp_path / "one", tmp_path / "two"
    _export(ctx, tab, first)
    _export(ctx, tab, second)
    assert _written(first) == _written(second)
    for name in _written(first):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_an_export_over_an_earlier_one_replaces_every_file(tmp_path):
    """"Export it again over the last one" is the ordinary case, and every file
    goes through ``atomic.staged_set`` -- staged first, replaced after."""
    ctx = FakeCtx()
    tab = _song(ctx)
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    before = (out / sirens_io.SONG_NAME).read_bytes()
    sirens_mode.write_note(ctx, 7)
    _export(ctx, tab, out)
    assert (out / sirens_io.SONG_NAME).read_bytes() != before
    assert not list(out.glob(".*.tmp"))


# --- refusals -----------------------------------------------------------------


def test_a_failed_encode_leaves_no_partial_folder_behind(monkeypatch, tmp_path):
    """Every file is encoded before any of them is written, which is what makes
    a refusal halfway through leave nothing at all -- rather than a ``stems/``
    holding three of five WAVs that a user would reasonably believe was an
    export."""
    from warlock.service.errors import ServiceError

    ctx = FakeCtx()
    tab = _song(ctx)
    _effect(tab, "coin")

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("this effect is longer than this build renders")

    monkeypatch.setattr(synth, "render_oneshot", boom)
    out = tmp_path / "audio"
    with pytest.raises(ServiceError):
        sirens_mode.export_to(ctx, tab, out)
    assert not out.exists()


def test_a_failed_export_unlocks_the_tab():
    ctx = FakeCtx()
    tab = _song(ctx)
    tab.saving = True
    sirens_mode.on_task_failed(
        ctx, _Done(f"{sirens_io.EXPORT_PREFIX}{tab.uid}", message="disk full")
    )
    assert not tab.saving


def test_a_cancelled_folder_picker_unlocks_the_tab(monkeypatch, tmp_path):
    """``None`` out of the picker is a cancel and nothing else, so the tab has
    to come back rather than reading as busy for the rest of the session."""
    from warlock.studio import dialogs

    ctx = FakeCtx()
    tab = _song(ctx)
    monkeypatch.setattr(dialogs, "select_folder", lambda *_a, **_k: None)
    sirens_mode.export_files(ctx, tab)
    sirens_mode.on_task_done(ctx, _Done(f"{sirens_io.EXPORT_PREFIX}{tab.uid}", ctx.result))
    assert ctx.result is None
    assert not tab.saving
    assert not list(tmp_path.iterdir())


def test_a_song_with_nothing_to_render_is_refused_before_the_picker(monkeypatch):
    """A brand-new document would otherwise open a folder picker and then write
    a folder of empty WAVs into whatever the user picked."""
    from warlock.studio import dialogs

    ctx = FakeCtx()
    tab = _song(ctx)
    tab.doc.set_order([])
    monkeypatch.setattr(
        dialogs, "select_folder", lambda *_a, **_k: pytest.fail("the picker opened")
    )
    sirens_mode.export_files(ctx, tab)
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[-1][1] == "error"


def test_a_busy_tab_does_not_export():
    ctx = FakeCtx()
    tab = _song(ctx)
    tab.saving = True
    sirens_mode.export_files(ctx, tab)
    assert ctx.submitted == []


# --- names --------------------------------------------------------------------


def test_an_effect_named_dot_dot_cannot_write_outside_the_folder(tmp_path):
    """``../evil`` is the shape ``wsng.py`` refuses to let anywhere near a path.
    Here the name has to survive into a filename, so it is sanitised instead --
    and what lands is one file, inside the folder the user picked."""
    ctx = FakeCtx()
    tab = _song(ctx)
    _effect(tab, "../evil")
    out = tmp_path / "audio" / "nested"
    _export(ctx, tab, out)
    written = [path for path in tmp_path.rglob("*.wav")]
    assert written and all(path.is_relative_to(out) for path in written)
    assert _written(out) & {"sfx/evil.wav"}


def test_a_windows_reserved_name_falls_back_rather_than_taking_the_export_down(tmp_path):
    """``CON`` survives sanitising intact and then fails at ``open`` with an
    errno nobody can map back to the effect they named. ``sheetout`` refuses one
    because an Inker template is the user's to fix; here it is one row of forty
    in a document where nothing else is wrong, so it falls back to its
    position."""
    ctx = FakeCtx()
    tab = _song(ctx)
    _effect(tab, "coin")
    _effect(tab, "CON")
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    assert {name for name in _written(out) if name.startswith("sfx/")} == {
        "sfx/coin.wav",
        "sfx/effect2.wav",
    }


def test_two_effects_with_one_name_are_two_files(tmp_path):
    """Letting the second land on the first's name is one export silently
    overwriting another -- and ``coin/1`` and ``coin-1`` sanitise to one name
    without either of them looking like a duplicate on screen."""
    ctx = FakeCtx()
    tab = _song(ctx)
    _effect(tab, "coin")
    _effect(tab, "coin")
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    assert {name for name in _written(out) if name.startswith("sfx/")} == {
        "sfx/coin.wav",
        "sfx/coin-2.wav",
    }


def test_an_unnamed_channel_is_named_by_its_position(tmp_path):
    ctx = FakeCtx()
    tab = _song(ctx)
    tab.doc.update_channel(tab.doc.channels[0].uid, name="")
    out = tmp_path / "audio"
    _export(ctx, tab, out)
    assert "stems/channel1.wav" in _written(out)


def test_the_path_guard_refuses_an_escape_the_sanitiser_let_through(tmp_path):
    """Belt and braces, asserted directly: :func:`sirens_io.safe_stem` decides
    what a name *becomes* and :func:`sirens_io._under` checks where the result
    *lands*, and the second exists because the first is one regex away from
    letting a separator through."""
    with pytest.raises(ValueError):
        sirens_io._under(tmp_path, "..", "evil.wav")
    with pytest.raises(ValueError):
        sirens_io._under(tmp_path, sirens_io.SFX_DIR, "../../evil.wav")


def test_a_name_that_sanitises_to_nothing_falls_back():
    assert sirens_io.safe_stem("...", "effect1") == "effect1"
    assert sirens_io.safe_stem("", "effect1") == "effect1"
    assert sirens_io.safe_stem("   ", "effect1") == "effect1"
    assert sirens_io.safe_stem("coin pickup", "effect1") == "coin-pickup"
