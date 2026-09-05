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
    # **This tab's sound, not the machine's (S6, 2026-09-05).** The global
    # ``playing()`` is true while any tab -- or a sound effect -- is on the one
    # mixer channel, so switching to a silent tab B while A played showed B a
    # Stop button, and pressing it stopped A. The tag names whose buffer is
    # actually on the channel. Belt and braces with ``SirensState.activate``,
    # which now stops the device on the switch: both, because every door in
    # this app is held twice.
    playing = sirens_audio.tag() == tab.uid
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

    # The two ways to hear less than the whole song from the top, which is what
    # writing one actually involves: bar 40 of a three-minute track, and one
    # pattern before it is anywhere in the order list. Both existed under the
    # pane -- the render's row map, and ``synth.render_pattern``, which had no
    # caller at all (the 2026-09-02 review, section 8).
    if widgets.disabled_button(
        f"{icons.PLAY} From the caret",
        device,
        (width, 0),
        reason=sirens_audio.unavailable_reason(),
        tooltip="Play from the row the caret is on, in the song's own timing.",
    ):
        sirens_mode.play_from_caret(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.PLAY} This pattern",
        device,
        (width, 0),
        reason=sirens_audio.unavailable_reason(),
        tooltip=(
            "Play the pattern the grid is editing, once, whether or not the "
            "order list reaches it. The song's own buffer is untouched."
        ),
    ):
        sirens_mode.play_pattern(ctx, tab)
    changed, value = controls.checkbox("Loop playback", state.loop_playback)
    if changed:
        state.loop_playback = bool(value)
        if playing:
            # Applied to what is sounding rather than to the next press: a
            # toggle that only takes effect after a stop reads as a dead
            # control, which is the thing this pane exists not to draw.
            sirens_mode.play(ctx, tab)
    widgets.muted_wrapped(
        "Repeats the rendered song. The loop *point* in the order list is what "
        "an exported WAV tells a game engine; this is for listening."
    )

    # **The device's level, and Muse's player draws the same one.** There is one
    # reserved channel, so there is one volume: two per-mode sliders writing it
    # would be a control that disagrees with itself the moment the other mode
    # set it. ``sirens_audio`` owns the number for the reason it owns the
    # channel -- see ``set_volume``. Two views, one value.
    imgui.set_next_item_width(-1)
    changed, level = controls.slider_float(
        "##sirens-volume", sirens_audio.volume(), 0.0, 1.0, "Volume %.2f"
    )
    if changed:
        sirens_audio.set_volume(level)

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
