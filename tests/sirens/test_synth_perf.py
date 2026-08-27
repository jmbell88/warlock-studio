"""The render budget. ``uv run pytest -m perf -n 0``.

Sirens re-renders on every edit -- that is what "render-then-play" costs, and it
is the trade taken instead of a fourth thread with hard real-time deadlines. The
trade only holds while a render is much faster than the audio it produces: at
parity the editor would stutter, and below it playback could never keep up at
all. So the budget is stated as a *ratio to realtime* rather than as a wall
clock, which is the only form of it that means anything on a machine other than
the one it was written on.

Excluded from the parallel run for the reason every ``perf`` case is: a budget
measured while fifteen sibling workers fight for the same cores is a measurement
of the scheduler.
"""

from __future__ import annotations

import time

import pytest

from warlock.studio.sirens import document as D
from warlock.studio.sirens import synth

#: Measured at roughly 30x on the development machine with the default five
#: channels. A third of that is the floor: slow enough not to fail on a busy or
#: modest CI box, fast enough that halving the renderer's speed fails here
#: rather than in somebody's ears.
MIN_REALTIME_RATIO = 8.0


def _busy_song(bars: int = 16) -> D.SongDoc:
    """Something like a real arrangement: every channel playing, all the time.

    A song of one note is not a benchmark of anything -- the tick loop's cost is
    per voice per tick, so the case worth budgeting is the one where no voice is
    ever idle.
    """
    doc = D.new_song()
    pattern = doc.patterns[0]
    doc.resize_pattern(pattern.uid, 64)
    instruments = {one.kind: one.uid for one in doc.instruments}
    for row in range(64):
        for index, channel in enumerate(doc.channels):
            if channel.kind == "sample":
                continue
            doc.set_cell(pattern.uid, row, index, D.NOTE, 36 + (row * (index + 1)) % 36)
            doc.set_cell(pattern.uid, row, index, D.INSTRUMENT, instruments[channel.kind])
            doc.set_cell(pattern.uid, row, index, D.EFFECT, synth.FX_VIBRATO)
            doc.set_cell(pattern.uid, row, index, D.PARAM, 0x44)
    doc.set_order([pattern.uid] * bars)
    return doc


@pytest.mark.perf
def test_a_song_renders_much_faster_than_it_plays():
    doc = _busy_song()
    start = time.perf_counter()
    pcm, _loop = synth.render(doc)
    elapsed = time.perf_counter() - start
    audio = pcm.shape[0] / synth.SAMPLE_RATE
    ratio = audio / elapsed
    assert audio > 30.0, "the benchmark has to be long enough to measure"
    assert ratio >= MIN_REALTIME_RATIO, (
        f"rendered {audio:.1f}s of audio in {elapsed:.2f}s ({ratio:.1f}x realtime),"
        f" under the {MIN_REALTIME_RATIO}x floor"
    )
