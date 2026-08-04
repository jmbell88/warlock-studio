"""The landing chooser: what the app opens on.

The workspace assumes you already know which of two pipelines you are in and
what you are looking at; the first thing after a launch is neither. So the
frame starts here instead -- start a 2D reference, start a 3D asset, open
something already made, or manage the style profiles the 2D pane draws from --
and the Home button in the top bar comes back to it, so it is a chooser rather
than a splash screen.

Nothing here is persisted: ``AppState.landing`` defaults True, which is what
makes this appear on every launch rather than only the first ever.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import profiles, theme, widgets
from ..state import DEFAULT_FORM_3D, default_form_2d
from . import library, profiles_panel

BUTTON = (300.0, 64.0)


def draw(ctx: Any) -> None:
    view = ctx.state.landing_view
    if view == "open":
        _open(ctx)
    elif view == "profiles":
        _profiles(ctx)
    else:
        _choose(ctx)


# --- the four choices -------------------------------------------------------


def _choose(ctx: Any) -> None:
    imgui.dummy((0, 24))
    imgui.text("Warlock Studio")
    widgets.muted("A prompt becomes a reference image; a reference becomes a mesh.")
    imgui.dummy((0, 16))

    if imgui.button("New 2D image", BUTTON):
        start_2d(ctx)
    widgets.muted("Compose a prompt and generate a reference.")
    imgui.dummy((0, 8))

    if imgui.button("New 3D model", BUTTON):
        start_3d(ctx)
    widgets.muted("Start from a finished reference, or an image you drop in.")
    imgui.dummy((0, 8))

    if imgui.button("Paint", BUTTON):
        start_paint(ctx)
    widgets.muted("A canvas, or an image you already have.")
    imgui.dummy((0, 8))

    if imgui.button("Open existing", BUTTON):
        ctx.state.landing_view = "open"
    widgets.muted("Everything already generated.")
    imgui.dummy((0, 8))

    if imgui.button("Profiles", BUTTON):
        ctx.state.landing_view = "profiles"
    active = profiles.get_active(ctx.settings)
    widgets.muted(
        f"Saved style settings. Active: {active}." if active else "Saved style settings."
    )
    if ctx.state.last_error:
        imgui.dummy((0, 16))
        widgets.text_colored(theme.ERR, ctx.state.last_error)


def start_2d(ctx: Any) -> None:
    """A clean 2D form, wearing the active profile.

    ``default_form_2d`` rolls its own seed, so this is genuinely a fresh start
    rather than last session's form with the prompt cleared.
    """
    ctx.state.form_2d = profiles.apply(default_form_2d(), profiles.active_fields(ctx.settings))
    ctx.state.select(None)
    ctx.state.mode = "2d"
    _leave(ctx)


def start_3d(ctx: Any) -> None:
    ctx.state.form_3d = dict(DEFAULT_FORM_3D)
    ctx.state.select(None)
    ctx.state.mode = "3d"
    _leave(ctx)


def start_paint(ctx: Any) -> None:
    """Paint keeps whatever was open: unlike the two generate panes, there is
    no "fresh form" here -- the documents *are* the work."""
    ctx.state.mode = "paint"
    _leave(ctx)


def _leave(ctx: Any) -> None:
    ctx.state.landing = False
    ctx.state.landing_view = "choose"
    ctx.settings.set("mode", ctx.state.mode)


def _back(ctx: Any) -> None:
    if imgui.button("Back"):
        ctx.state.landing_view = "choose"


# --- open existing ----------------------------------------------------------


def _open(ctx: Any) -> None:
    _back(ctx)
    imgui.same_line()
    job = ctx.cache.get(ctx.state.selected)
    if widgets.disabled_button("Continue", job is not None):
        _continue(ctx, job)
    imgui.separator()
    # The library verbatim rather than a second card list: the filters, the
    # cards and the primary-action ladder are the same question here as in the
    # sidebar, and two implementations of it would drift.
    library.draw(ctx)


def _continue(ctx: Any, job: dict[str, Any]) -> None:
    # The same reference/model split the library filter uses: a job that stops
    # at an image opens in the pane that made it, anything else in 3D.
    ctx.state.mode = "2d" if job.get("stage") == "reference" else "3d"
    _leave(ctx)


# --- profiles ---------------------------------------------------------------


def _profiles(ctx: Any) -> None:
    if ctx.state.profile_draft is None:
        _back(ctx)
        imgui.separator()
    profiles_panel.draw(ctx)
