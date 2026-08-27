"""Every Sirens pane, drawn -- on a machine with no GPU and no sound card.

The other pane smoke tests in this suite build a *renderer* over a real GL
context and skip where there is none, which is most CI and every remote shell.
That skip is what let a real bug ship: ``sirens_patterns`` drew its caret with
``add_rect``'s thickness and flags the wrong way round, which type-errors on
every frame that has a grid on screen -- and nothing here drew a pane, so
nothing noticed.

Nothing in this file presents anything, so no GL is needed. imgui hands its font
atlas to a backend and will not finish a frame until one claims it; declaring
``renderer_has_textures`` is that claim, and with nothing to draw into it costs
a texture upload that never happens. What the frame still does is run every
widget call, every draw-list call and every layout pass for real, which is
exactly the layer where a wrong argument order lives.

The panes are drawn into a window with a **stated size**. That is not cosmetic:
the grid skips a channel whose column starts past the content region, so a
default-sized window drew nothing at all and passed while the bug was there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from test_sirens_mode import FakeCtx, _tab

from warlock.studio import sirens_mode
from warlock.studio.panes import (
    sirens_bridge,
    sirens_effects,
    sirens_envelopes,
    sirens_instruments,
    sirens_orders,
    sirens_patterns,
    sirens_transport,
)
from warlock.studio.sirens import document as D
from warlock.studio.sirens import instruments as inst

PANES = (
    ("sirens-patterns", sirens_patterns),
    ("sirens-transport", sirens_transport),
    ("sirens-orders", sirens_orders),
    ("sirens-instruments", sirens_instruments),
    ("sirens-envelopes", sirens_envelopes),
    ("sirens-effects", sirens_effects),
    ("sirens-bridge", sirens_bridge),
)

#: Wide and tall enough that the grid draws every channel and the envelope
#: editor draws four graphs rather than four lines. A sidebar is 300 design
#: pixels and the centre column is the rest of the window.
WINDOW = (760.0, 900.0)


@pytest.fixture
def frames():
    """A bare imgui context, built and destroyed around this file.

    The save-and-restore is ``test_pane_guard``'s discipline for its reason: at
    most one imgui context may exist at a time, and a file that wants one builds
    and destroys it rather than relying on collection order.
    """
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)

    def draw(build: Any) -> None:
        imgui.new_frame()
        imgui.set_next_window_size(WINDOW)
        imgui.begin("smoke")
        try:
            build()
        finally:
            imgui.end()
            imgui.end_frame()
            imgui.render()

    yield draw
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


@pytest.fixture(autouse=True)
def _no_device(monkeypatch):
    """No pane in this file may reach the mixer. CI has no card and a box that
    has one is not something a drawing test should depend on."""
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "available", lambda: False)
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)


def _loaded(ctx: FakeCtx) -> Any:
    """A song with something in every pane: notes, a selection, a sample, and
    an instrument whose four sequences are all non-empty."""
    tab = _tab(ctx)
    doc = tab.doc
    pattern = doc.patterns[0]
    for row, note in enumerate((48, 52, 55)):
        doc.set_cell(pattern.uid, row * 4, 0, D.NOTE, note)
        doc.set_cell(pattern.uid, row * 4, 0, D.INSTRUMENT, doc.instruments[0].uid)
    doc.update_instrument(
        doc.instruments[0].uid,
        volume=inst.Sequence(values=(15, 12, 8, 4, 0), loop=0, release=2),
        arpeggio=inst.Sequence(values=(0, 4, 7), loop=0),
        pitch=inst.Sequence(values=(-40, 0, 40)),
        duty=inst.Sequence(values=(0, 1, 2, 3), loop=0),
    )
    effect = doc.add_oneshot("coin", rows=2)
    doc.set_cell(effect.pattern, 0, 0, D.NOTE, 60)
    doc.set_cell(effect.pattern, 0, 0, D.INSTRUMENT, doc.instruments[0].uid)
    doc.set_sample("kick", np.zeros(128, dtype=np.float32))
    sample = next(one for one in doc.instruments if one.kind == "sample")
    doc.update_instrument(sample.uid, sample="kick")
    state = sirens_mode.ensure(ctx)
    state.oneshot = effect.uid
    state.anchor = (0, 0)
    state.row, state.channel = 4, 1
    return tab


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_with_a_song_open(name, pane, frames):
    ctx = FakeCtx()
    _loaded(ctx)
    frames(lambda: pane.draw(ctx))


@pytest.mark.parametrize("name,pane", PANES, ids=[name for name, _ in PANES])
def test_every_pane_draws_with_nothing_open(name, pane, frames):
    """The empty state is a frame too, and it is the first one a user sees."""
    ctx = FakeCtx()
    frames(lambda: pane.draw(ctx))


def test_the_grid_draws_its_caret(frames):
    """The bug this file was written for. The caret rectangle is drawn only
    where the caret is, so a window too small to reach that column passed while
    the call itself could not run."""
    ctx = FakeCtx()
    _loaded(ctx)
    sirens_mode.set_caret(ctx, row=0, channel=0, column=0)
    frames(lambda: sirens_patterns.draw(ctx))


def test_the_envelope_editor_draws_an_instrument_with_no_sequences(frames):
    """``default`` gives a new instrument a volume curve and nothing else, and
    three of the four graphs are then empty -- which is the ordinary case, not
    an edge one."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    bare = tab.doc.add_instrument(kind="triangle")
    tab.doc.update_instrument(bare.uid, volume=inst.Sequence())
    sirens_mode.ensure(ctx).instrument = bare.uid
    frames(lambda: sirens_envelopes.draw(ctx))


def test_the_envelope_editor_draws_a_sequence_at_the_engines_ceiling(frames):
    """256 steps is the widest a graph is ever asked to draw, and the column
    width it works out to is a fraction of a pixel."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    uid = tab.doc.instruments[0].uid
    tab.doc.update_instrument(
        uid,
        volume=inst.Sequence(
            values=tuple(range(inst.MAX_SEQUENCE_LEN)), loop=8, release=200
        ),
    )
    sirens_mode.ensure(ctx).instrument = uid
    frames(lambda: sirens_envelopes.draw(ctx))


def test_the_effects_pane_draws_an_effect_whose_pattern_is_gone(frames):
    """Unreachable through the app -- ``add_oneshot`` mints the pattern and the
    pair is one undo step -- but a hand-edited ``.wsng`` can carry it, and a row
    that renders as an exception is worse than one that says what is wrong."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    one = tab.doc.add_oneshot("orphan", rows=2)
    tab.doc.remove_pattern(one.pattern)
    sirens_mode.ensure(ctx).oneshot = one.uid
    frames(lambda: sirens_effects.draw(ctx))
