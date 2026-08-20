"""Troupe's left-bottom pane: the form that starts a character.

It submits the *first* link of the chain and nothing else: one cheap T-pose
reference. The gate is the whole point of the shape -- the user approves the
drawing in Create, and only then is the reconstruction spent -- so this pane
deliberately has no "and then build everything" button. What it does have is the
sheet's options, because they ride along on the reference and are validated at
its door: a bad palette found an hour later, on a row the worker minted, would
be a refusal the user never submitted.
"""

from __future__ import annotations

from typing import Any

from ...service import troupe as svc_troupe
from .. import forms, troupe_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = troupe_mode.ensure(ctx)
    widgets.section("New character")
    manual_render.help_button(ctx, "troupe-settings")

    options = _options(ctx)
    form = _form(state, options)
    with forms.Form("troupe-settings") as form_ui:
        _changed, form["prompt"] = form_ui.text("prompt", "Describe them", form["prompt"])
        _changed, form["variant"] = form_ui.combo(
            "variant",
            "Build",
            form["variant"],
            [(v, v) for v in options.get("variants") or ()],
        )
        _size(form, form_ui, options)
        _palette(ctx, form, form_ui, options)
    imgui.dummy((0, 4))
    _submit(ctx, form)


def _options(ctx: Any) -> dict[str, Any]:
    """The door's own answer about what may be asked for, read once.

    Cached on the frame state rather than called per draw: it walks the palette
    directory, and a directory walk sixty times a second is a cost with no
    reader. Keyed on nothing, because the only thing that changes it is a file
    the user dropped in -- which the Refresh below re-reads.
    """
    cached = ctx.state.preview.get("troupe_options")
    if cached is None:
        cached = svc_troupe.troupe_options(ctx.svc)
        ctx.state.preview["troupe_options"] = cached
    return cached


def _form(state: Any, options: dict[str, Any]) -> dict[str, Any]:
    """The request, kept on the mode's own state so a trip to Create and back
    does not lose what was typed."""
    if not state.form:
        defaults = options.get("defaults") or {}
        state.form = {
            "prompt": "",
            "variant": str(defaults.get("variant") or "male"),
            "logical_size": int(defaults.get("logical_size") or 32),
            "colors": int(defaults.get("colors") or 64),
            "outline": str(defaults.get("outline") or "outer"),
            "dither": False,
            "palette": "",
        }
    return state.form


def _size(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    _changed, size = form_ui.combo(
        "logical_size",
        "Sprite size",
        str(form["logical_size"]),
        [(str(s), f"{s} px") for s in options.get("logical_sizes") or ()],
    )
    form["logical_size"] = int(size)
    _changed, outline = form_ui.combo(
        "outline",
        "Outline",
        form["outline"],
        [(m, m) for m in options.get("outline_modes") or ()],
    )
    form["outline"] = outline


def _palette(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]
) -> None:
    """A designed palette if one is installed, a colour budget otherwise.

    Two controls rather than one because they are two different answers: a
    named palette is the artist's decision and the budget is the machine's --
    a median cut over the atlas, which is the fallback and says so.
    """
    installed = list(options.get("palettes") or ())
    choices = [("", "Derived from the render")] + [(name, name) for name in installed]
    _changed, palette = form_ui.combo("palette", "Palette", form["palette"], choices)
    form["palette"] = palette
    if not palette:
        _changed, colors = form_ui.combo(
            "colors",
            "Colours",
            str(form["colors"]),
            [(str(n), f"{n} colours") for n in options.get("colors") or ()],
        )
        form["colors"] = int(colors)
    _changed, form["dither"] = form_ui.checkbox("dither", "Dither", bool(form["dither"]))


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    from imgui_bundle import imgui

    busy = ctx.busy("troupe-start")
    ready = bool(form["prompt"].strip())
    if busy:
        widgets.spinner()
        imgui.same_line()
    if widgets.disabled_button(
        "Draw the reference",
        not busy and ready,
        (-1, 0),
        reason="Describe the character first."
        if not ready
        else "A reference is already being queued.",
    ):
        troupe_mode.start_character(ctx, form)
    widgets.cost_note(
        "One image, and then it stops. You approve the T-pose drawing in "
        "Create; the mesh, the rig and 256 rendered frames follow from there."
    )
