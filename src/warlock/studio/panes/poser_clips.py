"""Poser's clip editor: authoring the keyframes a character sheet animates.

The Troupe programme's clip-authoring half. The clip *format* and its expansion
shipped with Troupe; what did not was any way to change a clip that is not
editing JSON in the package tree, and authoring those twenty-two keyframes is
the most important art task in the programme.

**The armature is the editor.** Poser already has a skeleton, gizmos and a pose
editor, so this pane adds no second posing surface: picking a key loads it onto
the preview armature, you drag joints with the controls on the right exactly as
you would for a library pose, and *Update key* puts it back. Everything a clip
adds on top of that is timing -- which keys, in what order, how many frames
apart -- and that is what the list here is.

**Scrubbing is not editing, and the pane says so.** A scrubbed frame is
interpolated between two keys and has nowhere to store an edit, so while the
scrubber is engaged the pane shows the frame number rather than a key and
*Update key* refuses by name. Coming back to a key is one click and re-applies
the authored pose.

Drawn in the left sidebar under the pose library rather than in a mode of its
own: a clip is a *library* of the same kind of thing the poses above it are,
and the right sidebar has to stay free for the joint controls that are the
actual editing surface.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, forms, poser_mode, theme, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    if not widgets.header("Clips", persist_key="poser-clips"):
        return
    manual_render.help_button(ctx, "poser-clips")
    if not ctx.rigging_available:
        widgets.muted("Editing clips needs Blender, which is not installed.")
        return
    poser_mode.clips_pump(ctx)
    if state.clips_loading and not state.clips:
        widgets.muted("Reading the clip library...")
        return
    if not state.clips.get("clips"):
        widgets.muted_wrapped(
            "This skeleton ships no clips. Clips are what a character sheet "
            "animates; only the humanoid template has them today."
        )
        return

    with forms.Form("poser-clips"):
        _picker(ctx, state)
        _keys(ctx, state)
        _timing(ctx, state)
        _scrubber(ctx, state)
        _save(ctx, state)


def _picker(ctx: Any, state: Any) -> None:
    names = [str(c.get("name") or "") for c in state.clips.get("clips") or ()]
    # ``labeled_combo`` and not a named ``combo``: imgui draws a combo's name
    # to its *right*, and a full-width one in a sidebar puts that name past the
    # content region, where ``same_line`` clips rather than wrapping -- so the
    # name is simply never drawn. ``widgets.combo``'s docstring states the rule
    # and ``test_inker_ux`` enforces it.
    # ``labeled_combo`` answers with the key alone, not a (changed, key) pair
    # -- ``select_clip`` is a no-op on an unchanged name, so the comparison is
    # the whole of the change test.
    picked = widgets.labeled_combo(
        "Clip",
        state.clip,
        [(n, n) for n in names],
        help_text="Which animation's keyframes this list is showing.",
    )
    if picked != state.clip:
        poser_mode.select_clip(ctx, picked)
    if state.clips.get("edited"):
        widgets.muted("edited - this skeleton is using your clips, not the shipped ones")


def _keys(ctx: Any, state: Any) -> None:
    record = state.open_clip()
    if record is None:
        return
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    widgets.section("Keyframes")
    for index, name in enumerate(keys):
        selected = index == state.key_index and state.frame < 0
        # The frames *out of* this key, which is what a segment is -- shown on
        # the row it belongs to rather than in a second list, so reordering a
        # key visibly carries its timing.
        span = segments[index] if index < len(segments) else None
        label = f"{index + 1}. {name}" + (f"   {span}f" if span is not None else "")
        if controls.selectable_row(f"poser-clip-key/{index}", label, selected=selected):
            poser_mode.select_key(ctx, index)

    changed, onion = controls.checkbox(
        "Onion skin",
        state.onion,
        tooltip=(
            "Ghost the keys either side of this one in the viewport. A looping "
            "clip wraps, so the first key's neighbour is the last."
        ),
    )
    if changed:
        poser_mode.set_onion(ctx, onion)

    imgui.dummy((0, 4))
    if widgets.disabled_button(
        "Up",
        state.key_index > 0,
        reason="This is already the first key.",
        tooltip="Move this key earlier, carrying its own timing with it.",
    ):
        poser_mode.move_key(ctx, state.key_index, -1)
    imgui.same_line()
    if widgets.disabled_button(
        "Down",
        state.key_index < len(keys) - 1,
        reason="This is already the last key.",
        tooltip="Move this key later, carrying its own timing with it.",
    ):
        poser_mode.move_key(ctx, state.key_index, 1)
    imgui.same_line()
    if widgets.disabled_button(
        "Remove",
        len(keys) > 2,
        reason="A clip needs at least two keys; one key is a pose.",
        tooltip=(
            "Take this key out of the clip. The pose itself stays in the "
            "library -- another clip may use it."
        ),
    ):
        poser_mode.remove_key(ctx, state.key_index)

    viewer = poser_mode.viewer_of(ctx)
    posing = viewer is not None and viewer.pose_mode
    if widgets.disabled_button(
        "Update key from pose",
        posing and state.frame < 0,
        (-1, 0),
        reason=(
            "That is an in-between frame, not a key. Pick a key first."
            if state.frame >= 0
            else "The skeleton preview is still loading."
        ),
        tooltip="Store the joints as they are now into the selected key.",
    ):
        poser_mode.capture_key(ctx)
    if widgets.disabled_button(
        "New key from pose...",
        posing,
        (-1, 0),
        reason="The skeleton preview is still loading.",
        tooltip=(
            "Add the joints as they are now as a brand-new key pose, after "
            "the selected one."
        ),
    ):
        _ask_new_key(ctx)
    _insert_existing(ctx, state)


def _insert_existing(ctx: Any, state: Any) -> None:
    """Add a key the library already holds, after the selected one.

    ``poser_mode.insert_key`` and ``PoserState.key_names`` were both written
    for this and neither had a caller: the pane could author a *new* key pose
    and could not reuse one, so building a walk out of four poses meant
    authoring each of them twice. The combo lists what the library has, minus
    nothing -- a clip may legitimately visit the same key twice, which is how
    a there-and-back cycle is written without a pingpong.
    """
    names = [name for name in state.key_names() if name]
    if not names:
        return
    slot = "poser-insert-key"
    current = str(ctx.state.preview.get(slot) or names[0])
    if current not in names:
        current = names[0]
    picked = widgets.labeled_combo(
        "Existing key", current, [(name, name) for name in names]
    )
    if picked != current:
        ctx.state.preview[slot] = picked
        current = picked
    if widgets.disabled_button(
        "Insert this key",
        bool(state.open_clip()),
        (-1, 0),
        reason="Open a clip first.",
        tooltip="Put a key the library already holds after the selected one.",
    ):
        poser_mode.insert_key(ctx, current)


def _ask_new_key(ctx: Any) -> None:
    from .. import dialogs

    ctx.prompts.ask(
        dialogs.Prompt(
            title="Name this key",
            label="Name",
            on_accept=lambda name: poser_mode.new_key(ctx, name),
        )
    )


def _timing(ctx: Any, state: Any) -> None:
    record = state.open_clip()
    if record is None:
        return
    from ...service import clips as svc_clips

    widgets.section("Timing")
    segments = list(record.get("segments") or ())
    index = min(state.key_index, len(segments) - 1) if segments else -1
    if index >= 0:
        changed, value = controls.input_int(
            "Frames after this key", int(segments[index])
        )
        if changed:
            poser_mode.set_segment(ctx, index, value)

    changed, closed = controls.checkbox(
        "Loops",
        bool(record.get("closed")),
        tooltip=(
            "A looping clip steps from its last key back to its first, so it "
            "needs one more segment than an open one. Changing this resizes "
            "the timing list to match."
        ),
    )
    if changed:
        poser_mode.set_closed(ctx, closed)

    easing = widgets.labeled_combo(
        "Easing",
        str(record.get("easing") or "linear"),
        [(name, name) for name in svc_clips.EASINGS],
        help_text="How the frames are spaced inside each step.",
    )
    poser_mode.set_easing(ctx, easing)
    if easing == "ease" and max(segments or [0]) < 3:
        # Stated rather than left to be discovered, and stated only for the one
        # easing it is true of. ``ease`` is a smoothstep whose value at a
        # two-frame segment's single interior sample is exactly one half, so it
        # renders identically to ``linear`` there. ``ease_in`` and ``ease_out``
        # sample 0.25 and 0.75 and genuinely differ at that length -- telling an
        # author otherwise sends them away from the option that would have
        # worked.
        widgets.muted_wrapped(
            '"ease" needs at least three frames in a step to do anything; '
            "every step here is shorter than that. Try ease_in or ease_out, "
            "which do act on a two-frame step."
        )

    total = sum(segments)
    widgets.muted(f"{total} frames  -  {len(record.get('keys') or ())} keys")


def _scrubber(ctx: Any, state: Any) -> None:
    widgets.section("Play")
    if state.clips_error:
        # Mid-edit inconsistency rather than a refusal: the clip momentarily
        # does not expand, the scrubber empties, and Save is what actually says
        # no. Coloured as a warning rather than an error for that reason.
        widgets.wrapped(theme.WARN, state.clips_error)
        return
    if not state.frames:
        widgets.muted("This clip has no frames yet.")
        return
    count = len(state.frames)
    current = state.frame if state.frame >= 0 else 0
    changed, value = controls.slider_int(
        f"Frame 1-{count}", int(current), 0, count - 1
    )
    if changed:
        poser_mode.scrub(ctx, value)
    if state.frame >= 0:
        widgets.text_colored(
            theme.ACCENT, f"frame {state.frame + 1} of {count} - in-between, not a key"
        )
        if controls.button(
            "Back to key",
            tooltip="Leave the scrubber and re-apply the selected key's own pose.",
        ):
            poser_mode.apply_key(ctx)


def _save(ctx: Any, state: Any) -> None:
    imgui.dummy((0, 8))
    busy = ctx.busy(poser_mode.CLIPS_SAVE_KEY)
    if state.clips_unsaved:
        widgets.text_colored(theme.ACCENT, "unsaved clip changes")
    if widgets.disabled_button(
        "Save clips",
        state.clips_unsaved and not busy,
        (-1, 0),
        reason="Nothing has changed." if not state.clips_unsaved else "Still saving.",
        tooltip=(
            "Write these clips as your own copy. The clips the build ships are "
            "left alone, so an update cannot overwrite your work."
        ),
    ):
        poser_mode.save_clips(ctx)
    if widgets.disabled_button(
        "Revert to shipped clips",
        (state.clips_unsaved or bool(state.clips.get("edited"))) and not busy,
        (-1, 0),
        reason="These are already the clips the build ships.",
        tooltip="Throw away every change to this skeleton's clips. Asks first.",
    ):
        poser_mode.revert_clips(ctx)
