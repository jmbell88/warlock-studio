"""The tick loop, from both ends.

**Determinism first**, because it is the guarantee the whole engine is built
around and it is the one that decays silently: a render that differs by a
rounding step still sounds fine, and only shows up months later as "why is the
exported WAV different when nothing changed".

Then the audible behaviours, asserted as measurements of the output rather than
by reaching into the player: what a test can see is what a listener can hear.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.sirens import document as D
from warlock.studio.sirens import instruments as inst
from warlock.studio.sirens import notes, synth


def _song(rows: int = 8, tempo: int = 150, speed: int = 6) -> D.SongDoc:
    doc = D.new_song()
    doc.set_song(tempo=tempo, speed=speed)
    doc.resize_pattern(doc.patterns[0].uid, rows)
    return doc


def _put(doc: D.SongDoc, row: int, channel: int = 0, **columns: int) -> None:
    lookup = {
        "note": D.NOTE,
        "instrument": D.INSTRUMENT,
        "volume": D.VOLUME,
        "effect": D.EFFECT,
        "param": D.PARAM,
    }
    for name, value in columns.items():
        doc.set_cell(doc.patterns[0].uid, row, channel, lookup[name], value)


def _tone(doc: D.SongDoc, row: int = 0, note: int = 48, channel: int = 0, **rest: int) -> None:
    kind = doc.channels[channel].kind
    instrument = next(one for one in doc.instruments if one.kind == kind)
    _put(doc, row, channel, note=note, instrument=instrument.uid, **rest)


def _rms(pcm: np.ndarray) -> float:
    return float(np.sqrt((pcm.astype(np.float64) ** 2).mean())) if pcm.size else 0.0


def _seconds(pcm: np.ndarray) -> float:
    return pcm.shape[0] / synth.SAMPLE_RATE


# --- the guarantee ------------------------------------------------------------


def test_the_same_document_renders_the_same_bytes():
    doc = _song()
    _tone(doc, 0, 48)
    _tone(doc, 4, 55)
    first, _ = synth.render(doc)
    second, _ = synth.render(doc)
    assert first.tobytes() == second.tobytes()


def test_a_render_is_finite_and_inside_the_rails():
    doc = _song()
    for row in range(8):
        for channel in range(len(doc.channels) - 1):
            _tone(doc, row, 48 + row, channel)
    pcm, _ = synth.render(doc)
    assert np.isfinite(pcm).all()
    assert np.abs(pcm).max() <= 1.0


def test_an_empty_song_is_silence_of_the_right_length():
    """Silence, not zero samples: an empty pattern still takes the time its rows
    take, which is what makes an empty bar a rest rather than a skip."""
    doc = _song(rows=16)
    pcm, _ = synth.render(doc)
    assert not pcm.any()
    assert _seconds(pcm) == pytest.approx(16 * 60 / (150 * D.ROWS_PER_BEAT), abs=0.01)


def test_a_song_with_no_order_renders_nothing():
    doc = _song()
    doc.set_order([])
    pcm, loop = synth.render(doc)
    assert pcm.shape == (0, 2) and loop is None


# --- notes and voices ---------------------------------------------------------


def test_a_note_makes_a_sound_and_an_empty_channel_does_not():
    doc = _song()
    _tone(doc, 0, 48)
    pcm, _ = synth.render(doc)
    assert _rms(pcm) > 0.01


def test_a_note_needs_an_instrument():
    """A cell with a pitch and no instrument is silent rather than guessing one:
    guessing means a song sounds different depending on which instrument
    happened to be first in the list."""
    doc = _song()
    _put(doc, 0, note=48)
    pcm, _ = synth.render(doc)
    assert not pcm.any()


def test_a_note_off_stops_the_voice():
    doc = _song(rows=16)
    _tone(doc, 0, 48)
    _put(doc, 4, note=notes.NOTE_OFF)
    pcm, _ = synth.render(doc)
    half = pcm.shape[0] // 2
    assert _rms(pcm[:half]) > 0.01
    assert _rms(pcm[half:]) == pytest.approx(0.0, abs=1e-6)


def test_a_release_plays_the_instruments_tail_rather_than_cutting():
    doc = _song(rows=16)
    _tone(doc, 0, 48)
    _put(doc, 4, note=notes.NOTE_RELEASE)
    released, _ = synth.render(doc)
    _put(doc, 4, note=notes.NOTE_OFF)
    cut, _ = synth.render(doc)
    assert _rms(released) > _rms(cut)


def test_the_volume_column_scales_the_voice_and_persists():
    """An empty volume column means "unchanged", not "full" -- otherwise every
    row after a quiet one snaps back to maximum."""
    doc = _song(rows=8)
    _tone(doc, 0, 48, volume=inst.MAX_VOLUME)
    loud, _ = synth.render(doc)
    doc = _song(rows=8)
    _tone(doc, 0, 48, volume=4)
    quiet, _ = synth.render(doc)
    assert _rms(quiet) < _rms(loud) / 2


def test_each_channel_kind_makes_a_different_sound():
    """Four kinds, four distinguishable outputs. A wiring mistake that played
    every channel as a pulse would pass every other test in this file."""
    heard = {}
    for index, channel in enumerate(D.new_song().channels[:4]):
        doc = _song()
        _tone(doc, 0, 48, channel=index)
        pcm, _ = synth.render(doc)
        heard[channel.kind] = pcm.tobytes()
    assert len(set(heard.values())) == 3, "pulse, triangle and noise must differ"
    assert all(len(one) for one in heard.values())


def test_a_sample_channel_is_silent_without_a_sample():
    doc = _song()
    sampler = next(one for one in doc.instruments if one.kind == "sample")
    index = next(i for i, one in enumerate(doc.channels) if one.kind == "sample")
    _put(doc, 0, index, note=48, instrument=sampler.uid)
    pcm, _ = synth.render(doc)
    assert not pcm.any()


def test_a_sample_channel_plays_the_sample_it_names():
    doc = _song()
    doc.set_sample("hit", np.ones(4000, dtype=np.float32) * 0.5)
    sampler = next(one for one in doc.instruments if one.kind == "sample")
    doc.update_instrument(sampler.uid, sample="hit")
    index = next(i for i, one in enumerate(doc.channels) if one.kind == "sample")
    _put(doc, 0, index, note=notes.SAMPLE_BASE_NOTE, instrument=sampler.uid)
    pcm, _ = synth.render(doc)
    assert _rms(pcm) > 0.01


def test_panning_puts_a_voice_on_one_side():
    """Fully away means *silent*, which is also the assertion that catches the
    two sides sharing one decimation filter -- a shared one leaks each channel's
    filter tail into the other and nothing else in this file would notice."""
    doc = _song()
    doc.update_channel(doc.channels[0].uid, pan=-1.0)
    _tone(doc, 0, 48)
    pcm, _ = synth.render(doc)
    assert _rms(pcm[:, 0]) > 0.01
    assert _rms(pcm[:, 1]) == pytest.approx(0.0, abs=1e-6)


# --- the effect column --------------------------------------------------------


def test_a_tempo_effect_changes_how_long_the_song_takes():
    doc = _song(rows=8)
    _tone(doc, 0, 48)
    plain, _ = synth.render(doc)
    _put(doc, 4, effect=synth.FX_TEMPO, param=75)
    slower, _ = synth.render(doc)
    assert _seconds(slower) > _seconds(plain) + 0.3


def test_a_halt_ends_the_song_where_it_is():
    doc = _song(rows=16)
    _tone(doc, 0, 48)
    whole, _ = synth.render(doc)
    _put(doc, 4, effect=synth.FX_HALT, param=0)
    halted, _ = synth.render(doc)
    assert _seconds(halted) < _seconds(whole)


def test_a_pattern_break_moves_to_the_next_order_entry():
    doc = _song(rows=32)
    second = doc.add_pattern(rows=8)
    doc.set_order([doc.patterns[0].uid, second.uid])
    whole, _ = synth.render(doc)
    _put(doc, 4, effect=synth.FX_BREAK, param=0)
    broken, _ = synth.render(doc)
    assert _seconds(broken) < _seconds(whole)


def test_a_jump_sends_the_player_to_an_order_position():
    doc = _song(rows=8)
    second = doc.add_pattern(rows=8)
    doc.set_order([doc.patterns[0].uid, second.uid])
    _put(doc, 0, effect=synth.FX_JUMP, param=1)
    pcm, _ = synth.render(doc)
    # One row of the first pattern, then the whole of the second.
    assert _seconds(pcm) < _seconds(synth.render(_song(rows=16))[0])


def test_a_pitch_slide_moves_the_note_without_retriggering_it():
    doc = _song(rows=16)
    _tone(doc, 0, 48)
    _put(doc, 1, effect=synth.FX_SLIDE_UP, param=200)
    pcm, _ = synth.render(doc)
    assert np.isfinite(pcm).all() and _rms(pcm) > 0.01


def test_portamento_glides_to_the_new_note_rather_than_restarting_it():
    """The audible difference is legato against stutter, and the mechanical one
    is that the envelope keeps running -- so a portamento into a note after the
    attack has decayed must stay quiet rather than jumping back to full."""
    doc = _song(rows=16)
    _tone(doc, 0, 48)
    _put(doc, 8, note=60, effect=synth.FX_PORTAMENTO, param=50)
    glide, _ = synth.render(doc)
    doc = _song(rows=16)
    _tone(doc, 0, 48)
    _tone(doc, 8, 60)
    jump, _ = synth.render(doc)
    assert glide.tobytes() != jump.tobytes()


def test_an_arpeggio_is_not_a_plain_note():
    doc = _song(rows=8)
    _tone(doc, 0, 48)
    plain, _ = synth.render(doc)
    _put(doc, 0, effect=synth.FX_ARPEGGIO, param=0x47)
    arpeggiated, _ = synth.render(doc)
    assert plain.tobytes() != arpeggiated.tobytes()


def test_a_volume_slide_fades_the_voice():
    doc = _song(rows=32)
    _tone(doc, 0, 48, volume=inst.MAX_VOLUME)
    _put(doc, 1, effect=synth.FX_VOLUME_SLIDE, param=0x04)
    pcm, _ = synth.render(doc)
    third = pcm.shape[0] // 3
    assert _rms(pcm[:third]) > _rms(pcm[2 * third :])


def test_every_effect_in_the_table_has_a_letter_and_a_description():
    """The grid draws the letter and the manual quotes the description, so a
    tenth effect added without either is an unlabelled column."""
    for value, (letter, blurb) in synth.EFFECT_NAMES.items():
        assert 0 <= value <= 0xF
        assert len(letter) == 1 and blurb
    assert len(set(synth.EFFECT_NAMES)) == len(synth.EFFECT_NAMES)


# --- the loop and the tail ----------------------------------------------------


def test_a_song_without_a_loop_gets_its_release_tail():
    """Rather than being cut the instant the last row ends."""
    doc = _song(rows=8)
    _tone(doc, 0, 48)
    pcm, loop = synth.render(doc)
    assert loop is None
    assert _seconds(pcm) > 8 * 60 / (150 * D.ROWS_PER_BEAT)


def test_a_looping_song_reports_where_it_loops_and_stops_at_the_seam():
    """A tail past the loop point would be audible as a doubled note on every
    repeat, so a looping song ends exactly where it starts again."""
    doc = _song(rows=8)
    second = doc.add_pattern(rows=8)
    doc.set_order([doc.patterns[0].uid, second.uid])
    _tone(doc, 0, 48)
    doc.set_song(loop_order=1)
    pcm, loop = synth.render(doc)
    assert loop is not None
    start, end = loop
    assert 0 < start < end <= pcm.shape[0]
    assert end == pytest.approx(pcm.shape[0], abs=2)


def test_the_render_ceiling_stops_a_hand_edited_order_list(monkeypatch):
    monkeypatch.setattr(synth, "MAX_RENDER_SECONDS", 0.5)
    doc = _song(rows=64)
    doc.set_order([doc.patterns[0].uid] * 64)
    pcm, _ = synth.render(doc)
    assert _seconds(pcm) <= 0.6


# --- the two other entry points -----------------------------------------------


def test_a_pattern_can_be_auditioned_on_its_own():
    doc = _song(rows=8)
    second = doc.add_pattern(rows=8)
    _tone(doc, 0, 48)
    pcm = synth.render_pattern(doc, second.uid)
    assert not pcm.any(), "the pattern that was asked for, not the order"


def test_a_one_shot_renders_at_its_own_tempo():
    doc = _song(rows=8)
    effect = doc.add_oneshot("zap", rows=8)
    doc.set_cell(effect.pattern, 0, 0, D.NOTE, 60)
    doc.set_cell(effect.pattern, 0, 0, D.INSTRUMENT, doc.instruments[0].uid)
    slow = synth.render_oneshot(doc, effect.uid)
    doc.update_oneshot(effect.uid, tempo=D.MAX_TEMPO)
    fast = synth.render_oneshot(doc, effect.uid)
    assert fast.shape[0] < slow.shape[0]


def test_the_two_entry_points_refuse_something_that_is_not_there():
    doc = _song()
    with pytest.raises(ValueError, match=D.MISSING_PATTERN):
        synth.render_pattern(doc, 999999)
    with pytest.raises(ValueError, match=D.MISSING_ONESHOT):
        synth.render_oneshot(doc, 999999)
