"""Section K/M/L: forms, panes and layout.

Most of these are properties of source rather than of pixels -- a hardcoded
pixel size, a control with no widget, a palette bound at import -- which is
exactly the class the smoke tests cannot see and a screenshot would not either.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from warlock.studio import layout as layout_mod
from warlock.studio import theme, tokens
from warlock.studio.panes import app_settings, landing, library, settings_2d, settings_3d

PANES = Path(inspect.getfile(library)).resolve().parent


# --- K97: design pixels ------------------------------------------------------

# The files the sweep covered. A raw two-number tuple in one of these is a size
# that stays put while the monitor's scale grows around it.
SP_SWEPT = (
    "../dialogs.py",
    "profiles_panel.py",
    "inker_canvas.py",
    "inker_bridge.py",
    "settings_2d.py",
    "settings_3d.py",
    "inker_colors.py",
)


def _literal_sizes(path: Path) -> list[tuple[int, str]]:
    """Every ``(w, h)`` tuple and ``set_next_item_width(n)`` of a bare number.

    Parsed rather than grepped, so a number inside a comment or a docstring --
    of which these files have many, most of them *about* pixel sizes -- cannot
    be mistaken for a call site.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            nums = [e for e in node.elts if isinstance(e, ast.Constant)]
            if (
                len(nums) == 2
                and all(isinstance(e.value, int | float) for e in nums)
                and any(abs(float(e.value)) >= 16 for e in nums)
            ):
                found.append((node.lineno, ast.unparse(node)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("set_next_item_width", "same_line")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, int | float)
            and abs(float(node.args[0].value)) >= 16
        ):
            found.append((node.lineno, ast.unparse(node)))
    return found


@pytest.mark.parametrize("name", SP_SWEPT)
def test_no_pane_hardcodes_a_pixel_size(name):
    """K97. ``sp()`` is what keeps a measurement meaning "this wide" on a 150%
    monitor rather than drifting; a literal 150 px button at 200% scale holds
    text drawn at 300% and the label runs off the end of it."""
    path = (PANES / name).resolve()
    assert not _literal_sizes(path), _literal_sizes(path)


def test_the_swatch_grid_scales_with_the_display():
    from warlock.studio.panes import inker_colors

    source = inspect.getsource(inker_colors._swatches)
    assert "sp(SWATCH)" in source


# --- K92 / K98: measured rather than guessed ---------------------------------


def test_the_2d_form_scrolls_under_a_fixed_submit():
    """K92. The pane is twelve collapsible sections tall and the one control
    every visit ends with sat at the bottom of all of them -- as did the
    refusals saying why it could not be pressed."""
    source = inspect.getsource(settings_2d.draw)
    assert 'begin_child("2d-form"' in source
    assert source.index("end_child") < source.index("_submit(ctx, form)")


def test_the_library_footer_reservation_is_measured():
    """K98. The two constants were guessed against a bar whose height is a
    function of the theme, the scale and how many buttons the current *view*
    draws -- the trash's bulk bar has two where the workshop's has three."""
    source = inspect.getsource(library.draw)
    assert "_footer_reserve()" in source
    assert "_measure_footer" in source


def test_a_measured_footer_is_stored_in_design_pixels():
    """Or a scale change would compound it: the value is measured in physical
    pixels and handed back out through ``sp``."""
    assert "tokens.SCALE" in inspect.getsource(library._measure_footer)
    before = library._footer_px[0]
    try:
        library._footer_px[0] = 50.0
        tokens.set_scale(2.0)
        assert library._footer_reserve() == pytest.approx(100.0)
    finally:
        tokens.set_scale(1.0)
        library._footer_px[0] = before


# --- K94 / K95 / K96: the 3D form's three controls ---------------------------


def test_the_one_option_budget_is_disabled_with_a_reason():
    """K94. A combo with a single entry looks broken: the user opens it, finds
    nothing else, and cannot tell a missing feature from a missing file."""
    source = inspect.getsource(settings_3d._budget)
    assert "begin_disabled(single)" in source
    assert "qualified" in source
    # And the reason is no longer the one that stopped being true when
    # gltfpack was vendored.
    assert "not installed" not in source


def test_custom_triangles_finally_has_a_widget():
    """K95. The field was submitted, validated and recorded with no way to set
    it -- a form field that existed only for the API."""
    source = inspect.getsource(settings_3d._budget)
    assert 'form["profile"] == "custom"' in source
    assert "custom_triangles" in source


def test_the_size_field_is_an_unbounded_drag_that_carries_its_unit():
    """K96. A slider needs a maximum and there is no largest asset -- a wall
    section is legitimately 8 m."""
    assert settings_3d.SIZE_NO_BOUND[0] >= settings_3d.SIZE_NO_BOUND[1]
    source = inspect.getsource(settings_3d._size)
    assert "drag_float" in source
    assert "%.2f m" in source
    # 0 says what it means rather than showing a measurement of zero metres.
    assert "unset" in source


# --- K99: the atlas ----------------------------------------------------------


def test_the_atlas_is_rebuilt_between_frames_and_never_inside_one():
    """Rebuilding invalidates every ImFont handle, and those are pushed and
    popped all through ``_build_ui``."""
    from warlock.studio import fonts, main

    source = inspect.getsource(main.App.frame)
    assert "fonts.reload" in source
    assert source.index("fonts.reload") < source.index("imgui.new_frame()")
    assert "clear_fonts" in inspect.getsource(fonts.reload)


def test_the_scale_slider_only_asks_for_a_rebake_on_release():
    """Per mouse-move it would be a font rebuild sixty times a second."""
    source = inspect.getsource(app_settings._interface)
    guard = source.split("is_item_deactivated_after_edit", 1)[1]
    assert "fonts_dirty = True" in guard


# --- K100: one configuration table -------------------------------------------


def test_the_config_table_is_one_function_drawn_in_two_places():
    from warlock.studio import main

    assert "config_table" in inspect.getsource(main.App._effective_config_section)
    assert "config_table" in inspect.getsource(app_settings._config)
    # And the data source is still the one doctor prints.
    assert "effective(" in inspect.getsource(app_settings.config_table)


# --- M106: the sidebar -------------------------------------------------------


def test_the_sidebar_option_is_three_names_and_the_width_is_module_state():
    assert set(layout_mod.SIDEBAR_WIDTHS) == {"narrow", "default", "wide"}
    try:
        layout_mod.set_sidebar("narrow")
        assert layout_mod.SIDEBAR_WIDTHS["narrow"] == layout_mod.SIDEBAR_W
    finally:
        layout_mod.set_sidebar("default")


# --- M107: the Home tiles ----------------------------------------------------


def test_the_tiles_are_data_so_the_keys_and_the_clicks_agree():
    """A hand-written keyboard index over a hand-written column of calls is two
    orderings, and they drift the first time a tile is inserted."""
    keys = [key for key, _icon, _name in landing.TILES]
    assert keys == ["2d", "3d", "inker", "clay", "open", "profiles"]
    assert len(set(keys)) == len(keys)


def test_every_tile_has_an_icon_the_atlas_carries_and_a_caption():
    from types import SimpleNamespace

    from warlock.studio import icons

    known = {v for k, v in vars(icons).items() if k.isupper()}
    ctx = SimpleNamespace(settings=SimpleNamespace(get=lambda *_a: None))
    for key, icon, name, caption in landing.tiles(ctx):
        assert icon in known, key
        assert name and caption
        assert all(ord(c) < 0x100 for c in name + caption), key


def test_the_tile_cursor_wraps():
    """Six fixed choices in a ring is a menu, where a two-hundred-row list is
    not -- which is why the library's arrows clamp and these do not."""
    from types import SimpleNamespace

    ctx = SimpleNamespace(state=SimpleNamespace(home_index=0))
    landing.move(ctx, -1)
    assert ctx.state.home_index == len(landing.TILES) - 1
    landing.move(ctx, 1)
    assert ctx.state.home_index == 0


def test_home_takes_the_arrows_and_enter_only_on_the_chooser():
    """Its other two views are lists with their own controls, and a tile cursor
    moving invisibly behind them would fire on the next Enter."""
    from warlock.studio import main

    source = inspect.getsource(main.App._shortcut)
    assert 'landing_view == "choose"' in source
    assert "landing.activate" in source


# --- M105: the theme hook ----------------------------------------------------


def test_both_palettes_define_exactly_the_same_names():
    """A name missing from one is an AttributeError on the frame somebody
    switches, in whichever pane happens to read it first."""
    names = [set(p) for p in tokens.PALETTES.values()]
    assert all(n == names[0] for n in names)
    assert frozenset(names[0]) == tokens.COLOUR_NAMES


def test_a_palette_name_is_a_live_lookup_and_not_an_import_time_constant():
    """The reason the switch needed no edit at any of the dozens of
    ``theme.ACCENT`` call sites -- and the reason a constant would have left
    every hand-drawn rect on the old colours."""
    try:
        dark = theme.ACCENT
        tokens.set_theme("light")
        assert dark != theme.ACCENT
        assert tokens.PALETTES["light"]["ACCENT"] == theme.ACCENT
    finally:
        tokens.set_theme("dark")


def test_theme_still_raises_for_something_that_is_not_a_colour():
    with pytest.raises(AttributeError):
        _ = theme.NOT_A_COLOUR


def test_an_unknown_theme_name_falls_back_rather_than_raising():
    """It comes out of a settings file, and a value written by a build with a
    third palette must not stop the window opening."""
    try:
        assert tokens.set_theme("solarized") == "dark"
    finally:
        tokens.set_theme("dark")


def test_the_status_map_holds_names_rather_than_resolved_colours():
    assert set(theme.STATUS_COLORS.values()) <= tokens.COLOUR_NAMES
    try:
        tokens.set_theme("light")
        assert theme.status_color("done") == theme.rgba(tokens.PALETTES["light"]["OK"])
    finally:
        tokens.set_theme("dark")


def test_the_light_palette_keeps_the_roles_rather_than_inverting_the_numbers():
    """PANEL is still "the surface a form sits on" and the elevation steps are
    still steps *away from* the floor -- which on a light ground means darker."""
    light = tokens.PALETTES["light"]

    def lum(value: int) -> float:
        return ((value >> 16 & 0xFF) + (value >> 8 & 0xFF) + (value & 0xFF)) / 3

    assert lum(light["PANEL"]) > lum(light["BG"])
    assert lum(light["ELEV_1"]) > lum(light["ELEV_2"]) > lum(light["EDGE"])
    assert lum(light["TEXT"]) < lum(light["MUTED"]) < lum(light["BG"])


# --- L104 / K93 --------------------------------------------------------------


def test_the_strip_progress_comes_from_the_renderer():
    """A tally kept by the pane would describe the *previous* run after a
    cancel-and-restart."""
    from warlock.studio.viewer_embed import Viewer

    source = inspect.getsource(Viewer.strip_progress.fget)
    assert "self._strip" in source
    assert "len(strip.yaws)" in source


DENSE_PANES = (
    "inspector.py",
    "landing.py",
    "app_settings.py",
    "profiles_panel.py",
    "pose_panel.py",
    "retarget_panel.py",
    "texture_panel.py",
    "inker_layers.py",
    "clay_props.py",
    "clay_outliner.py",
    "sheet_panel.py",
)


@pytest.mark.parametrize("name", DENSE_PANES)
def test_every_dense_pane_explains_at_least_one_of_its_controls(name):
    """K93, as a floor rather than a count. Seven of these ten had *no*
    tooltip at all, which is the state worth failing on: a pane whose controls
    are named but never explained sends the reader to the manual for every one
    of them."""
    source = (PANES / name).read_text(encoding="utf-8")
    assert "help_marker" in source or "set_tooltip" in source
