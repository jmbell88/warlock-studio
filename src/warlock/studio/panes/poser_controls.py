"""Poser's right sidebar: the selected joint, the root, saving.

The mirror of ``pose_panel``'s pose-mode half, over the Poser viewer instead
of the shared one -- which is the whole reason it can exist beside an open
inspector pose session without either discarding the other.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, forms, poser_mode, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    if not widgets.header("Pose"):
        return
    manual_render.help_button(ctx, "poser-controls")
    if not ctx.rigging_available:
        widgets.muted("Posing needs Blender, which is not installed.")
        return
    viewer = poser_mode.viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        # Silently: the viewport beside this pane is already a centred empty
        # state saying the armature is being built, and this was the same fact
        # in different words a column away.
        return

    with forms.Form("poser-controls"):
        _banner(state, viewer)
        _joint(ctx, viewer)
        _root(viewer)
        _save(ctx, viewer)


def _banner(state: Any, viewer: Any) -> None:
    record = state.find(viewer.editor.current)
    label = str(record.get("name")) if record else "New pose"
    if viewer.editor.has_unsaved_edits():
        label += " - unsaved changes"
    widgets.text_colored(theme.ACCENT, label)


def _joint(ctx: Any, viewer: Any) -> None:
    selected = viewer.selected_bone
    widgets.muted(selected or "Click a joint to rotate it.")
    if selected is None:
        # The whole of what this pane said about itself was the line above.
        # Two more sentences, inline rather than in a tooltip: there is room,
        # and a hover is a poor place to put the one rule of the mode that is
        # not guessable -- that there is no undo.
        widgets.muted_wrapped(
            "Drag the ring gizmo on a joint to rotate it. Every joint you "
            "touch is recorded in the pose; the ones you do not are left at "
            "rest, so a wave is two joints rather than a whole skeleton."
        )
        # It used to say "There is no undo here", which stopped being true
        # when Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y were bound over
        # ``viewer.pose.undo()``: the pane was telling the user a working key
        # did not exist.
        widgets.muted_wrapped(
            "Ctrl+Z undoes a rotation. Reset joint puts one joint back and "
            "Reset all puts every joint back; both ask first if there is "
            "anything unsaved."
        )
    if widgets.disabled_button(
        "Reset joint",
        selected is not None,
        reason="Click a joint first.",
        tooltip="Put this one joint back to rest, leaving the rest of the pose.",
    ):
        viewer.reset_bone()
    imgui.same_line()
    # Behind the guard, exactly as ``poser_mode.new_pose`` is -- and it is the
    # same act: both put every joint back to rest, so both throw away whatever
    # was being authored. Bare, this was the one control in the mode that could
    # discard an unsaved pose with no way back, in the mode whose whole output
    # is the shared library every asset poses from.
    if controls.button(
        "Reset all",
        tooltip="Put every joint back to rest. There is no undo, so this asks first.",
    ):
        poser_mode.guard(ctx, "reset every joint", viewer.reset_all)
    if viewer.editor.mirror_pairs:
        # Hidden for a serpent or a fish, the pose_panel rule: a skeleton with
        # no mirror pairs has nothing to mirror.
        imgui.same_line()
        if controls.button(
            "Mirror",
            tooltip="Swap every left joint's rotation with its right one.",
        ):
            # Behind the guard for Reset all's reason: mirroring rewrites every
            # rotation, this mode has no undo at all, and it was the last bare
            # route to losing an unsaved pose. It does prompt mid-authoring --
            # accepted, because the alternative is an undo stack.
            poser_mode.guard(ctx, "mirror the pose", viewer.mirror)


def _root(viewer: Any) -> None:
    editor = viewer.editor
    if editor.root is None:
        return
    offset = editor.root_translation()
    if editor.selected == editor.root:
        # Shown only while the root is selected: the toggle changes what the
        # gizmo on that joint does, and drawing it against any other joint
        # would claim a capability the selection does not have.
        changed, value = controls.checkbox(
            "Move root",
            editor.root_translate,
            tooltip=(
                "Turns the root joint's gizmo from a rotator into arrows, so "
                "the whole pose can be offset rather than turned."
            ),
        )
        if changed:
            editor.root_translate = bool(value)
        if editor.root_translate:
            widgets.muted_wrapped(
                "Drag the arrows to offset the whole pose. Units are character "
                "heights; the bake scales them onto each asset's own rig."
            )
    if any(offset):
        widgets.muted(
            f"root offset  x {offset[0]:+.2f}  y {offset[1]:+.2f}  z {offset[2]:+.2f}"
        )


def _save(ctx: Any, viewer: Any) -> None:
    imgui.dummy((0, sp(tokens.SP_2)))
    busy = ctx.busy(poser_mode.SAVE_KEY)
    editing = viewer.editor.current is not None
    if editing and widgets.disabled_button(
        "Save",
        not busy,
        (-1, 0),
        tooltip="Overwrite the pose named above, in the shared library.",
    ):
        poser_mode.save(ctx)
    if widgets.disabled_button(
        "Save as...",
        not busy,
        (-1, 0),
        tooltip="Add a new pose to the library every asset on this skeleton can use.",
    ):
        poser_mode.save_as(ctx)
