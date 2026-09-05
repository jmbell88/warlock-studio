"""The same setting, verb and mode read the same in every workspace.

Three findings of the 2026-09-05 consistency review, each checked against
the v0.0.34 captures: Inker's toolbar said ``opa 1.00`` where everything
else said "Opacity"; the cross-workspace verbs were worded five ways; and
Poser and Troupe drew the same standing figure in a rail whose default is
icons only.
"""

from __future__ import annotations

import inspect
import re

from warlock.studio import icons, inker_ops, modes, state, verbs
from warlock.studio.panes import (
    clay_bridge,
    inker_context,
    inspector,
    library,
    muse_results,
    packwright_bridge,
    plotter_layers,
    plotter_menu,
    sheet_panel,
    sprite_panel,
    troupe_bridge,
    troupe_settings,
)

# --- property labels --------------------------------------------------------


def test_no_toolbar_slider_abbreviates_its_own_name():
    source = inspect.getsource(inker_context)
    assert not re.search(r'"(opa|hard) %', source)
    assert "label[:5]" not in source
    assert "percent_slider(" in source


def test_plotter_opacity_reads_as_a_percentage_in_both_panes():
    source = inspect.getsource(plotter_layers)
    assert source.count('"%.0f%%"') >= 2
    assert '"##layer-opacity", float(layer.opacity), 0.0, 1.0' not in source


# --- verbs ------------------------------------------------------------------


def test_verbs_are_spelt_from_the_mode_table():
    assert verbs.open_in("inker") == "Open in Inker"
    assert verbs.add_to("packwright") == "Add to Packwright"
    assert verbs.add_to("plotter", "as a tileset") == "Add to Plotter as a tileset"
    assert verbs.send_to("troupe") == "Send to Troupe"
    assert verbs.EXPORT_TO_LIBRARY == "Export to the library"


def test_no_pane_spells_a_cross_workspace_verb_by_hand():
    panes = (
        clay_bridge, inspector, library, muse_results, packwright_bridge,
        plotter_menu, sheet_panel, sprite_panel, troupe_bridge, troupe_settings,
        inker_ops,
    )
    literal = re.compile(r'"(Open in|Send to|Add to) [A-Z]|Export to (the )?library"')
    for module in panes:
        for line in inspect.getsource(module).splitlines():
            code = line.split("#", 1)[0]
            if '"""' in line or line.strip().startswith(("#", "*", "-")):
                continue
            assert not literal.search(code), (module.__name__, line.strip())


def test_the_card_action_ladder_uses_the_same_words():
    assert state.ACTIONS["inker"] == verbs.open_in("inker")
    assert state.ACTIONS["clay"] == verbs.open_in("clay")
    assert state.ACTIONS["plotter"] == verbs.add_to("plotter")


# --- rail icons -------------------------------------------------------------


def test_every_rail_icon_is_unique():
    glyphs = [icon for _key, _label, icon in modes.MODES]
    assert len(glyphs) == len(set(glyphs)), glyphs


def test_troupe_wears_the_same_glyph_on_every_surface():
    from warlock.studio import troupe_mode

    icon = dict((k, i) for k, _l, i in modes.MODES)["troupe"]
    assert icon == troupe_mode.ICON == icons.FILM
