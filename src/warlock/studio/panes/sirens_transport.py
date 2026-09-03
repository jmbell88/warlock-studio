"""Sirens' left-top pane: play, stop, tempo, and whether there is a device.

**The no-device state is drawn, not hidden.** A Play button greyed out with no
sentence beside it reads as a bug in the app; the same button greyed out over
"No audio device: playback is unavailable on this machine. Writing, saving and
exporting all still work." reads as a fact about the machine, and it is the one
the user can act on. ``sirens_audio.unavailable_reason`` owns the sentence so
this pane and the mode cannot say two different things.

**Tempo and speed are the song's, not the transport's.** They go through
``doc.set_song``, which is undoable and which re-arms the renderer -- a tempo
change you cannot undo is the one edit in a tracker that people make by
accident, because the control is a slider next to a Play button.
"""

from __future__ import annotations

from typing import Any

from .. import anchors, controls, icons, sirens_audio, sirens_mode, widgets
from ..manual import render as manual_render
from ..sirens import document as D


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    anchors.mark_window("sirens/transport")
    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Transport")
    manual_render.help_button(ctx, "sirens-transport")

    if tab is None:
        # The heading and nothing else. One voice for one empty state: the
        # grid's ``nothing_open`` is it, and four panels each repeating it
        # reads as four separate problems.
        return

    device = sirens_audio.available()
    playing = sirens_audio.playing()
    width = widgets.grid_width(2)
    label = f"{icons.SQUARE} Stop" if playing else f"{icons.PLAY} Play"
    if widgets.disabled_button(
        f"{label} (Space)", device, (width, 0), reason=sirens_audio.unavailable_reason()
    ):
        sirens_mode.toggle_play(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button(
        "Re-render", not tab.rendering, (width, 0), reason="A render is already running."
    ):
        sirens_mode.request_rerender(ctx, tab)

    if not device:
        widgets.muted_wrapped(sirens_audio.unavailable_reason())
    elif tab.render_error:
        # The error rather than the spinner: a render that failed is not a
        # render that is slow, and showing "Rendering..." forever is the one
        # state a user waits through instead of reading.
        widgets.muted_wrapped(tab.render_error)
    elif tab.rendering or tab.render_dirty:
        widgets.muted("Rendering...")
    elif tab.pcm is None:
        widgets.muted("Nothing in the order list to play yet.")
    else:
        seconds = tab.pcm.shape[0] / float(sirens_audio.RATE)
        widgets.muted(f"{seconds:0.1f}s rendered")

    imgui.dummy((0, 8))
    doc = tab.doc
    editable = not tab.busy
    imgui.set_next_item_width(-1)
    changed, value = controls.slider_int(
        "Tempo", doc.tempo, D.MIN_TEMPO, D.MAX_TEMPO, enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed and doc.set_song(tempo=int(value)):
        sirens_mode.request_rerender(ctx, tab)
    imgui.set_next_item_width(-1)
    changed, value = controls.slider_int(
        "Speed", doc.speed, D.MIN_SPEED, D.MAX_SPEED, enabled=editable
    )
    controls.fold_undo(doc.history)
    if changed and doc.set_song(speed=int(value)):
        sirens_mode.request_rerender(ctx, tab)
    # Ticks per row and beats per minute do not combine into anything a
    # musician can read, so the pane says the one number they can check against
    # a metronome rather than leaving them to derive it.
    widgets.muted(f"{doc.speed / doc.tick_rate * 1000:0.0f} ms per row")
