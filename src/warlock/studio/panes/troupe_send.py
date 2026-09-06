"""The question both silent Troupe doors used to skip.

``service.troupe.send_to_troupe`` has always accepted a sprite size and a rig
template, and the picker *inside* Troupe passes the mode's form so both apply.
The two doors a user actually reaches for -- the library's right-click item and
the inspector's button -- called it with no form at all, so ``logical_size``
arrived None and fell back to 32, and the skeleton was pinned to ``humanoid``.
A user who wanted 64 px sprites had to know to enter Troupe first and open a
collapsed sub-header; a user with a quadruped got human walk cycles, and the
manual's own advice was to go and re-rig it from Create.

So the doors ask. One modal, enqueued from anywhere and drawn at top level --
the ``dialogs.ConfirmQueue`` shape, because the library's item is inside an
imgui context popup and ``imgui.open_popup`` cannot be called there -- with the
single slot ``matte_preview`` uses, since at most one send is in flight.

**The answers are written back into the mode's form**, not kept per door, so
the size chosen at the library is the size the inspector opens on and Troupe's
own pane shows the same numbers. Two doors remembering separately would be two
defaults for one request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

from .. import controls, tokens, troupe_mode, widgets
from ..tokens import sp

TITLE = "Send to Troupe"

#: The floor ``widgets.modal_bounds`` is given. The combos are full-width, so
#: this is what decides how wide the dialog reads.
DIALOG_W = 420.0


@dataclass
class TroupeSend:
    """The mesh being sent, and the answers so far.

    The answers are held here rather than edited straight into the mode's form
    because Cancel has to mean cancel: writing live would leave a dismissed
    dialog's choices behind as the next send's defaults.
    """

    job_id: str = ""
    label: str = ""
    #: Whether the mesh already carries ``rig.glb``. Read off the row's cached
    #: ``files`` list -- the rig's *template* is a fact about a file on disk,
    #: and reading it here is the disk read ``can_send_to_troupe`` deliberately
    #: does not do on the frame thread.
    rigged: bool = False
    template: str = ""
    logical_size: int = 32
    camera: str = ""
    outline: str = ""
    colors: int = 64
    palette: str = ""
    # ``imgui.open_popup`` must be called exactly once per question, and the
    # overlay redraws every frame: ``dialogs.Confirm._open``'s idiom.
    _open: bool = False


def ask(ctx: Any, job: dict[str, Any] | None) -> bool:
    """Put the question in front of the send. -> whether it was asked.

    Draws nothing itself; the overlay picks it up on the next frame, which is
    what lets the library's context menu call it from inside a popup.
    """
    job_id = str((job or {}).get("id") or "")
    if not job_id:
        return False
    form = troupe_mode.form(ctx)
    ctx.state.troupe_send = TroupeSend(
        job_id=job_id,
        label=str((job or {}).get("prompt") or (job or {}).get("name") or "")[:48],
        rigged="rig.glb" in ((job or {}).get("files") or []),
        template=str(form.get("template") or ""),
        logical_size=int(form.get("logical_size") or 32),
        camera=str(form.get("camera") or ""),
        outline=str(form.get("outline") or ""),
        colors=int(form.get("colors") or 64),
        palette=str(form.get("palette") or ""),
    )
    return True


def close(ctx: Any) -> None:
    state = getattr(getattr(ctx, "state", None), "troupe_send", None)
    if state is not None:
        ctx.state.troupe_send = None


def is_open(ctx: Any) -> bool:
    """Whether the dialog is up. Tolerant of a partial ctx.

    ``getattr`` rather than attribute access, ``matte_preview.is_open``'s rule
    and here for its reason: ``App._modal_open`` asks this on every key press
    and must not require a state object the caller has never built.
    """
    state = getattr(getattr(ctx, "state", None), "troupe_send", None)
    return state is not None and bool(state.job_id)


def draw(ctx: Any) -> None:
    """The modal. Beside the confirms, because it is one."""
    state = getattr(ctx.state, "troupe_send", None)
    if state is None or not state.job_id:
        return
    appearing = not state._open
    if appearing:
        imgui.open_popup(TITLE)
        state._open = True
    alpha, rise = widgets.popover_enter("troupe-send", appearing)
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    widgets.modal_bounds(sp(DIALOG_W))
    opened, _ = imgui.begin_popup_modal(
        TITLE, None, imgui.WindowFlags_.always_auto_resize.value
    )
    widgets.pop_surface_rounding()
    if not opened:
        # Escape dismisses a modal without going through either button, and
        # imgui will not reopen a popup whose id it thinks is already open.
        imgui.pop_style_var()
        close(ctx)
        return
    widgets.window_shadow("overlay", radius=radius)
    if frosted:
        widgets.window_backdrop(radius=radius)
    if rise > 0.0:
        imgui.dummy((0, rise))
    _body(ctx, state)
    imgui.end_popup()
    imgui.pop_style_var()


def _body(ctx: Any, state: TroupeSend) -> None:
    options = troupe_mode.options(ctx)
    form = troupe_mode.form(ctx)
    # The body scrolls and the action row does not (INVARIANTS: a bounded
    # modal puts its body in ``modal_body`` and draws the buttons after it).
    with widgets.modal_body("troupe-send-body"):
        if state.label:
            widgets.muted(state.label)
        _skeleton(state, options)
        state.logical_size = int(
            widgets.labeled_combo(
                "Sprite size",
                str(state.logical_size),
                [(str(s), f"{s} px") for s in options.get("logical_sizes") or ()],
            )
        )
        presets = options.get("camera_presets") or {}
        state.camera = widgets.labeled_combo(
            "Camera",
            state.camera,
            [(key, str(entry.get("label") or key)) for key, entry in presets.items()],
        )
        helper = _camera_helper(presets, state.camera)
        if helper:
            widgets.muted(helper)
        state.outline = widgets.labeled_combo(
            "Outline",
            state.outline,
            [(m, m) for m in options.get("outline_modes") or ()],
        )
        # Shown only when no palette is named, mirroring ``troupe_settings``:
        # the budget is what a *derived* palette gets, so offering it beside a
        # named one would be a control whose value is silently ignored.
        if state.palette:
            widgets.muted(f"Palette: {state.palette}")
        else:
            state.colors = int(
                widgets.labeled_combo(
                    "Colours",
                    str(state.colors),
                    [(str(n), f"{n} colours") for n in options.get("colors") or ()],
                )
            )
    imgui.dummy((0, sp(6)))
    _actions(ctx, state, form)


def _skeleton(state: TroupeSend, options: dict[str, Any]) -> None:
    """Which rig an unrigged mesh is built on, when there is a choice.

    A rigged mesh is not asked: the skeleton is already on disk and the service
    reads it off ``rig.json``, so a picker here would be a control whose value
    that branch discards.
    """
    if state.rigged:
        widgets.muted("This mesh is already rigged; its own skeleton is used.")
        return
    choices = [
        (str(row.get("key")), str(row.get("label") or row.get("key")))
        for row in options.get("clip_templates") or ()
    ]
    if not choices:
        return
    if state.template not in {key for key, _label in choices}:
        state.template = choices[0][0]
    state.template = widgets.labeled_combo(
        "Skeleton",
        state.template,
        choices,
        help_text=(
            "The rig this mesh is built on, and the clip library its sheet is "
            "animated from. Only the skeletons with clips authored for them "
            "are offered."
        ),
    )


def _camera_helper(presets: dict[str, Any], key: str) -> str:
    """``troupe_settings._camera_helper``'s sentence, for the same reason."""
    entry = presets.get(key) or {}
    if "elevation" not in entry:
        return ""
    return f"{float(entry['elevation']):g} degrees above the horizon"


def _actions(ctx: Any, state: TroupeSend, form: dict[str, Any]) -> None:
    from . import troupe_settings

    count = troupe_settings.cell_count(form)
    note = f"{count} cells are rendered at {state.logical_size} px."
    if not state.rigged:
        note = f"A mesh that is not rigged is rigged first. Then {note}"
    widgets.cost_note(note)
    imgui.dummy((0, sp(tokens.SP_1)))
    if controls.button("Send", (sp(150), 0), role=controls.ButtonRole.PRIMARY):
        # The popup is closed here and not in ``_send``: that function is the
        # whole decision -- write back, then submit -- and a headless caller
        # (a test, the exerciser) must be able to run it with no imgui frame.
        imgui.close_current_popup()
        _send(ctx, state, form)
        return
    imgui.same_line()
    if controls.button("Cancel", (sp(110), 0)):
        imgui.close_current_popup()
        close(ctx)


def _send(ctx: Any, state: TroupeSend, form: dict[str, Any]) -> None:
    """Write the answers back, then submit with the form the pane also draws.

    No imgui: see ``_actions``.
    """
    form["logical_size"] = int(state.logical_size)
    form["camera"] = state.camera
    form["outline"] = state.outline
    if not state.palette:
        form["colors"] = int(state.colors)
    if not state.rigged and state.template:
        form["template"] = state.template
    job_id = state.job_id
    close(ctx)
    troupe_mode.send_to_troupe(ctx, {"id": job_id}, form)
