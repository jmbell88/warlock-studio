"""The Home screen: what the app opens on.

The workspace assumes you already know which of two pipelines you are in and
what you are looking at; the first thing after a launch is neither. So the
frame starts here instead -- start a 2D reference, start a 3D asset, open
something already made, or manage the style profiles the 2D pane draws from --
and the Home entry in the mode switch comes back to it, so it is a chooser
rather than a splash screen.

Nothing here is persisted: ``AppState.mode`` defaults to ``"home"``, which is
what makes this appear on every launch rather than only the first ever.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import fonts, icons, profiles, theme, widgets
from ..state import DEFAULT_FORM_3D, default_form_2d
from ..tokens import sp
from . import library, profiles_panel


def draw(ctx: Any) -> None:
    view = ctx.state.landing_view
    if view == "open":
        _open(ctx)
    elif view == "profiles":
        _profiles(ctx)
    else:
        _choose(ctx)


# --- the four choices -------------------------------------------------------


def _tile(ctx: Any, key: str, icon: str, name: str, caption: str) -> bool:
    """A centred, clickable card: icon left, name and caption right."""
    width, height = sp(380), sp(64)
    imgui.set_cursor_pos_x(max((imgui.get_window_width() - width) * 0.5, 0))
    clicked = False
    with widgets.card(f"landing/{key}", (width, height)):
        imgui.dummy((0, sp(4)))
        with fonts.title(imgui):
            widgets.text_colored(theme.ACCENT, icon)
        imgui.same_line(sp(48))
        imgui.begin_group()
        with fonts.label(imgui):
            imgui.text(name)
        with fonts.small(imgui):
            widgets.muted(caption)
        imgui.end_group()
    if imgui.is_item_clicked():
        clicked = True
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
    return clicked


def _choose(ctx: Any) -> None:
    avail = imgui.get_content_region_avail()
    # Five tiles plus the title block; centre the stack in the upper half.
    stack = sp(64 + 8) * 5 + sp(110)
    imgui.dummy((0, max((avail.y - stack) * 0.4, sp(24))))

    def centred(text: str, colour: int | None = None) -> None:
        width = imgui.calc_text_size(text).x
        imgui.set_cursor_pos_x(max((imgui.get_window_width() - width) * 0.5, 0))
        if colour is None:
            imgui.text(text)
        else:
            widgets.text_colored(colour, text)

    with fonts.title(imgui):
        centred("Warlock Studio")
    with fonts.small(imgui):
        centred("A prompt becomes a reference image; a reference becomes a mesh.", theme.MUTED)
    imgui.dummy((0, sp(20)))

    if _tile(ctx, "2d", icons.IMAGE, "New 2D image", "Compose a prompt and generate a reference."):
        start_2d(ctx)
    imgui.dummy((0, sp(8)))
    if _tile(
        ctx, "3d", icons.BOX, "New 3D model", "Start from a finished reference, or drop an image."
    ):
        start_3d(ctx)
    imgui.dummy((0, sp(8)))
    if _tile(ctx, "inker", icons.BRUSH, "Inker", "A canvas, or an image you already have."):
        start_inker(ctx)
    imgui.dummy((0, sp(8)))
    if _tile(ctx, "open", icons.FOLDER_OPEN, "Open existing", "Everything already generated."):
        ctx.state.landing_view = "open"
    imgui.dummy((0, sp(8)))
    active = profiles.get_active(ctx.settings)
    caption = f"Saved style settings. Active: {active}." if active else "Saved style settings."
    if _tile(ctx, "profiles", icons.SLIDERS, "Profiles", caption):
        ctx.state.landing_view = "profiles"

    if ctx.state.last_error:
        imgui.dummy((0, sp(16)))
        centred(ctx.state.last_error, theme.ERR)


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


def start_inker(ctx: Any) -> None:
    """Inker keeps whatever was open: unlike the two generate panes, there is
    no "fresh form" here -- the documents *are* the work."""
    ctx.state.mode = "inker"
    _leave(ctx)


def _leave(ctx: Any) -> None:
    """Every caller has just set a work mode, so the persist is unconditional."""
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
