"""The Home screen: what the app opens on.

The workspace assumes you already know which of two pipelines you are in and
what you are looking at; the first thing after a launch is neither. So the
frame starts here instead -- one tile per place there is work to do, plus
opening something already made and managing the style profiles the 2D pane
draws from -- and the Home entry in the mode switch comes back to it, so it is
a chooser rather than a splash screen.

Nothing here is persisted: ``AppState.mode`` defaults to ``"home"``, which is
what makes this appear on every launch rather than only the first ever.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import fonts, icons, modes, profiles, theme, tokens, widgets
from ..manual import render as manual_render
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


# --- the choices ------------------------------------------------------------


# What each tile is *called*, in the order they are drawn (M107). Data, so the
# arrow keys and the click path agree about what the third tile *is* -- a
# hand-written keyboard index over a hand-written column of calls is two
# orderings, and they drift the first time a tile is inserted.
#
# The names are editorial and deliberately not ``modes.MODES``' labels: a
# chooser says "New 2D Image" where a switch says "2D". The *set* is not
# editorial, and that is the half that went wrong (F76): it stopped at six while
# the app grew to ten modes, so Review, Plotter and Packwright had no way in
# from the screen the app opens on. Every ``modes.WORK_MODES`` entry has a tile,
# and ``tests/test_panes_home_tiles.py`` fails if one does not -- the argument
# ``studio/palette.py`` already makes, that a second index of everything the app
# can do drifts unless something checks it.
_NAMES: tuple[tuple[str, str], ...] = (
    ("2d", "New 2D Image"),
    ("3d", "New 3D Model"),
    ("inker", "Inker"),
    ("clay", "Clay"),
    ("plotter", "Plotter"),
    ("packwright", "Packwright"),
    ("review", "Review"),
    ("open", "Open Existing"),
    ("profiles", "Profiles"),
)

# The two tiles that are not modes: sub-views of Home itself, so they have no
# entry in ``modes.MODES`` to take an icon from.
_SUBVIEW_ICONS = {"open": icons.FOLDER_OPEN, "profiles": icons.SLIDERS}

# ``(key, icon, name)``. The icon comes from ``modes.MODES`` where there is one:
# it is already stated there, and a second copy is how Home comes to draw a mode
# under a glyph the switch does not use.
TILES: tuple[tuple[str, str, str], ...] = tuple(
    (key, {k: i for k, _label, i in modes.MODES}.get(key) or _SUBVIEW_ICONS[key], name)
    for key, name in _NAMES
)

_CAPTIONS = {
    "2d": "Compose a prompt and generate a reference.",
    "3d": "Start from a finished reference, or drop an image.",
    "inker": "A canvas, or an image you already have.",
    "clay": "Block a shape out from primitives, by hand.",
    "plotter": "Lay a tilemap out over a tileset.",
    "packwright": "Pack sprites into an atlas and its sidecar.",
    "review": "Judge a sweep's meshes, or label references.",
    "open": "Everything already generated.",
}


def tiles(ctx: Any) -> list[tuple[str, str, str, str]]:
    """``(key, icon, name, caption)`` for each tile, in drawing order.

    A function rather than a fourth column on ``TILES`` because one caption
    names the active profile; the action is looked up by key in ``activate``
    rather than stored here, so the table stays importable data.
    """
    out = []
    for key, icon, name in TILES:
        if key == "profiles":
            active = profiles.get_active(ctx.settings)
            caption = (
                f"Saved style settings. Active: {active}."
                if active
                else "Saved style settings."
            )
        else:
            caption = _CAPTIONS[key]
        out.append((key, icon, name, caption))
    return out


def activate(ctx: Any, index: int) -> None:
    """Do what the ``index``-th tile does. Shared by the click and by Enter."""
    key = TILES[index % len(TILES)][0]
    if key == "2d":
        start_2d(ctx)
    elif key == "3d":
        start_3d(ctx)
    elif key == "inker":
        start_inker(ctx)
    elif key == "clay":
        start_clay(ctx)
    elif key in ("open", "profiles"):
        # Sub-views of Home rather than modes.
        ctx.state.landing_view = key
    else:
        # Plotter, Packwright and Review: a plain switch, because each already
        # draws its own empty state with a way in ("New map", "Add a sprite",
        # a run to pick). Clay is the exception above rather than the rule --
        # it had no such state, which is why its tile mints a document.
        ctx.state.mode = key
        _leave(ctx)


def move(ctx: Any, delta: int) -> None:
    """Up/Down between the tiles. Wraps, unlike the library's arrow keys: a
    fixed ring of choices is a menu, where a two-hundred-row list is not."""
    ctx.state.home_index = (ctx.state.home_index + delta) % len(TILES)


def _tile(ctx: Any, key: str, icon: str, name: str, caption: str, *, focused: bool = False) -> bool:
    """A centred, clickable card: icon left, name and caption right."""
    width, height = sp(380), sp(64)
    _centre(width)
    clicked = False
    origin = imgui.get_cursor_screen_pos()
    with widgets.card(f"landing/{key}", (width, height)):
        imgui.dummy((0, sp(tokens.SP_1)))
        # The icon is one line and the text beside it is two, so drawing both
        # from the same Y aligns the glyph with the *name* rather than with the
        # block as a whole. Half the difference is the optical centre.
        with fonts.title(imgui):
            icon_h = imgui.get_text_line_height()
        with fonts.label(imgui):
            name_h = imgui.get_text_line_height()
        with fonts.small(imgui):
            caption_h = imgui.get_text_line_height()
        group_h = name_h + caption_h + imgui.get_style().item_spacing.y
        top = imgui.get_cursor_pos_y()
        imgui.set_cursor_pos_y(top + max((group_h - icon_h) * 0.5, 0.0))
        with fonts.title(imgui):
            widgets.text_colored(theme.ACCENT, icon)
        # A column position, not a gap: the text block starts here so that
        # every tile's name lines up whatever its glyph advances to. Left a
        # literal deliberately -- SP_8 + SP_4 would be arithmetic pretending to
        # be a rhythm, and the number is a measurement of one layout.
        imgui.same_line(sp(48))
        imgui.set_cursor_pos_y(top)
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
        # Hovering moves the keyboard cursor too, so the two never disagree
        # about which tile Enter would take.
        index = next((i for i, t in enumerate(TILES) if t[0] == key), None)
        if index is not None:
            ctx.state.home_index = index
    if focused:
        widgets.ring(
            origin, imgui.ImVec2(origin.x + width, origin.y + height), theme.ACCENT, 0.9
        )
    return clicked


def _centre(width: float) -> None:
    """Put the cursor where a ``width``-wide thing is centred in what is left.

    ``get_window_width`` was the wrong measure: it ignores the host window's
    own padding, so everything sat half a gutter off centre.
    """
    avail = imgui.get_content_region_avail().x
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max((avail - width) * 0.5, 0.0))


def _choose(ctx: Any) -> None:
    avail = imgui.get_content_region_avail()
    # The tiles plus the title block; centre the stack in the upper half. Off
    # ``len(TILES)`` rather than a literal, or adding a tile silently pushes the
    # bottom of the stack off screen.
    # The title block's own height rides on the hero's size, so it is written
    # as the old figure plus what the hero grew by rather than as a new
    # constant: the number was measured against a 16 px title, and a display
    # ramp that changes again must not silently push the bottom tile off screen.
    stack = sp(64 + 8) * len(TILES) + sp(110 + (tokens.TEXT_DISPLAY - tokens.TEXT_TITLE))
    imgui.dummy((0, max((avail.y - stack) * 0.4, sp(tokens.SP_6))))

    def centred(text: str, colour: int | None = None) -> None:
        _centre(imgui.calc_text_size(text).x)
        if colour is None:
            imgui.text(text)
        else:
            widgets.text_colored(colour, text)

    with fonts.display(imgui):
        centred("Warlock Studio")
    with fonts.small(imgui):
        centred("A prompt becomes a reference image; a reference becomes a mesh.", theme.MUTED)
    # On the subtitle's line, which is where ``help_button`` puts it: it
    # right-aligns onto whatever line is current, and the tile stack below is a
    # column of full-width cards with no line to share.
    manual_render.help_button(ctx, "home")
    imgui.dummy((0, sp(tokens.SP_5)))

    focus = ctx.state.home_index % len(TILES)
    for index, (key, icon, name, caption) in enumerate(tiles(ctx)):
        if index:
            imgui.dummy((0, sp(tokens.SP_2)))
        if _tile(ctx, key, icon, name, caption, focused=index == focus):
            activate(ctx, index)

    imgui.dummy((0, sp(tokens.SP_2)))
    _setup_entry(ctx)

    if ctx.state.errors:
        imgui.dummy((0, sp(tokens.SP_4)))
        for message in ctx.state.errors:
            centred(message, theme.ERR)


def _setup_entry(ctx: Any) -> None:
    """"Diagnostics / Set up models", on the screen the app opens on (F56).

    A first run reaches Home with nothing downloaded, presses New 2D Image, and
    is refused at the door -- correctly, and now with the download command in
    the message (F55) -- but Home itself offered every way to start work and no
    way to find out whether this install can do any of it. The row is small and
    sits under the tiles rather than beside them: it is not another thing to do,
    it is the thing to do when one of the tiles did not work.

    It counts what is failing rather than saying "Diagnostics", because the
    number is the whole reason to press it, and it is drawn in the same colours
    the header dot uses so the two obviously refer to one answer.
    """
    checks = list(getattr(ctx.runtime, "checks", []) or [])
    failing = [c for c in checks if not c.ok]
    fatal = [c for c in failing if c.fatal]
    if not checks:
        label = "Diagnostics - still checking"
        colour = theme.MUTED
    elif not failing:
        label = "Diagnostics - everything checks out"
        colour = theme.MUTED
    else:
        what = (
            "1 thing needs attention"
            if len(failing) == 1
            else f"{len(failing)} things need attention"
        )
        label = f"Diagnostics / Set up models - {what}"
        colour = theme.ERR if fatal else theme.WARN
    _centre(imgui.calc_text_size(label).x + imgui.get_style().frame_padding.x * 2)
    imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(colour)))
    clicked = imgui.small_button(f"{label}##landing-doctor")
    imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
        imgui.set_tooltip(
            "What this install can and cannot do, and the command to download "
            "anything that is missing."
        )
    if clicked:
        # App-Settings rather than the header popup: the popup is a read-only
        # list, and the model rows with their Download buttons are what someone
        # pressing this actually needs. The popup is still one click away from
        # the dot, which is where "what is wrong right now" belongs.
        ctx.state.mode = "settings"
        _leave(ctx)


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


def start_clay(ctx: Any) -> None:
    """Clay keeps whatever was open, as Inker does: the documents *are* the
    work, and there is no form to reset.

    The one addition is an empty Clay: the tile says "model something", so
    arriving at a mode with nothing in it and no obvious way to begin is a dead
    end. A document is minted only when there are none -- opening one over
    existing work would break the keeps-whatever-was-open contract.
    """
    from .. import clay_mode

    ctx.state.mode = "clay"
    if not clay_mode.ensure(ctx).docs:
        clay_mode.new_document(ctx)
    _leave(ctx)


def _leave(ctx: Any) -> None:
    """The tile has already set the mode; this only resets the sub-view.

    Nothing is persisted: the app opens on Home every launch, so a stored mode
    would have no reader.
    """
    ctx.state.landing_view = "choose"


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
    # The same reference/tile/model split the library filter uses: a job that
    # stops at an image opens in the pane that made it, anything else in 3D.
    ctx.state.mode = "2d" if job.get("stage") in ("reference", "tile") else "3d"
    _leave(ctx)


# --- profiles ---------------------------------------------------------------


def _profiles(ctx: Any) -> None:
    if ctx.state.profile_draft is None:
        _back(ctx)
        imgui.separator()
    profiles_panel.draw(ctx)
