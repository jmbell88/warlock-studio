"""Every work mode is reachable from the screen the app opens on.

Home's tile table stopped at six while the app grew to ten modes, so Review,
Plotter and Packwright had no entry on it -- reachable only from the mode switch
or from Alt+7/9/0, neither of which a first run knows about. It is the argument
``studio/palette.py`` already makes and the reason its mode commands are derived
rather than typed: a second index of everything the app can do drifts unless
something checks it.

The tiles stay hand-ordered and hand-captioned, because both are editorial -- a
chooser says "New 2D Image" where the switch says "2D" -- so what is asserted
here is the *set*, plus the two facts a hand-written table gets wrong first: an
icon the atlas does not carry, and a key ``activate`` has no branch for.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import icons, modes
from warlock.studio.panes import landing
from warlock.studio.state import AppState


def _ctx():
    return SimpleNamespace(
        state=AppState(),
        settings=SimpleNamespace(get=lambda *_a: None, set=lambda *_a: None),
    )


def test_every_work_mode_has_a_tile():
    """The half that is not editorial. A mode with no tile is a mode a first run
    cannot find."""
    keys = {key for key, _icon, _name in landing.TILES}
    assert keys >= modes.WORK_MODES, modes.WORK_MODES - keys


def test_the_tiles_that_are_not_modes_are_the_two_sub_views():
    """Home's own views rather than places in the switch, which is why they are
    the only tiles ``activate`` sends to ``landing_view``."""
    keys = [key for key, _icon, _name in landing.TILES]
    assert set(keys) - modes.WORK_MODES == {"open", "profiles"}
    assert len(set(keys)) == len(keys)


def test_no_pass_through_mode_gets_a_tile():
    """Home, the Manual and Settings have no form to submit and no viewport to
    frame; a tile for Home on Home is a button that does nothing."""
    keys = {key for key, _icon, _name in landing.TILES}
    assert keys & {"home", "manual", "settings"} == set()


def test_the_icons_come_from_the_mode_table_where_there_is_one():
    """A second copy is how Home comes to draw a mode under a glyph the switch
    does not use."""
    from_modes = {key: icon for key, _label, icon in modes.MODES}
    for key, icon, _name in landing.TILES:
        if key in from_modes:
            assert icon == from_modes[key], key


def test_every_tile_has_an_icon_the_atlas_carries_a_name_and_a_caption():
    """The atlas is a pinned lucide subset, so a codepoint it does not hold
    renders as the missing-glyph box; and every string here goes through imgui's
    default Basic-Latin+Latin-1 range."""
    known = {value for name, value in vars(icons).items() if name.isupper()}
    for key, icon, name, caption in landing.tiles(_ctx()):
        assert icon in known, key
        assert name and caption, key
        assert all(ord(c) < 0x100 for c in name + caption), key


def test_every_tile_key_is_one_activate_has_an_answer_for():
    """The click path and the keyboard path share ``activate``, so a tile with
    no branch is a card that silently does nothing under both."""
    for index, (key, _icon, _name) in enumerate(landing.TILES):
        ctx = _ctx()
        landing.activate(ctx, index)
        if key in ("open", "profiles"):
            assert ctx.state.landing_view == key
        elif key == "clay":
            # Clay's tile mints a document when there are none, which needs a
            # fuller ctx than this one; tests/test_clay_mode.py covers it.
            continue
        else:
            assert ctx.state.mode == key
            assert ctx.state.landing_view == "choose"


def test_the_tile_cursor_wraps_over_every_card_drawn():
    """Over ``rows`` rather than ``TILES``: with a Continue card in front of the
    table, a cursor that wrapped over the table would skip the last tile and
    land on Continue twice."""
    ctx = _ctx()
    landing.move(ctx, -1)
    assert ctx.state.home_index == len(landing.rows(ctx)) - 1 == len(landing.TILES) - 1
    landing.move(ctx, 1)
    assert ctx.state.home_index == 0


def test_the_stack_height_is_measured_off_the_cards_it_draws():
    """A literal card count is how adding one pushes the bottom of the stack off
    screen with nothing to say so.

    It is measured off ``rows`` rather than off ``TILES`` since UX.md Phase 4:
    the drawn list has one optional card in front of the table (Continue), so
    the table is no longer what is on screen.
    """
    import inspect

    source = inspect.getsource(landing._choose)
    assert "len(drawn)" in source and "rows(ctx)" in source
    assert "len(TILES)" not in source
