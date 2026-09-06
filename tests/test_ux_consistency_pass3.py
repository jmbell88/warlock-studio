"""The third consistency pass (2026-09-05): one vocabulary per shared surface.

Where pass 2 found divergent *behaviour* (the wheel, the tab bar, the save
gate), this pass finds surfaces that say the same thing in more than one
register. Each test here is one of those stated as a claim.

Sectioned by item, because later waves of the same pass add to this file.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

# --- item 2: one empty-state system, with actions ---------------------------
#
# Two tables say what an empty screen says and there is no third:
#   * ``widgets.nothing_open``  -- no document open at all (the workspace
#     screen: a hint, a primary, ghosts, recents).
#   * ``overlay.PLACEHOLDERS`` + ``overlay.centred_empty`` -- a document *is*
#     open and its viewport has nothing in it.
# Packwright's preview drew a third: one muted sentence in the top-left.


def _pane_sources() -> dict[str, str]:
    from warlock.studio import panes

    root = Path(panes.__file__).parent
    return {path.name: path.read_text(encoding="utf-8") for path in root.glob("*.py")}


def test_no_pane_answers_an_empty_viewport_with_a_muted_sentence():
    """Packwright's preview said "Add a sprite to see the atlas." in muted
    body text where the other nine viewports drew the icon-title-hint form."""
    from warlock.studio.panes import overlay, packwright_preview

    source = inspect.getsource(packwright_preview)
    assert "Add a sprite to see the atlas." not in source
    assert "overlay.centred_empty(" in source
    assert 'overlay.PLACEHOLDERS["packwright"]' in source

    # And no pane re-draws one of the table's own sentences in the muted
    # register: that is how the third spelling got in the first time.
    spellings = {text for entry in overlay.PLACEHOLDERS.values() for text in entry[1:]}
    for name, text in _pane_sources().items():
        for sentence in spellings:
            assert f'widgets.muted("{sentence}")' not in text, (name, sentence)


#: Hints whose first word is one of these are telling the reader to *do*
#: something, so the app has to offer the doing.
_IMPERATIVES = ("Describe", "Add", "Rig", "Start", "Choose", "Draw", "Write")

#: The documented exemption: a hint beginning "Pick ..." points at a list the
#: user works in another pane, and "Describe the music you want above" points
#: at a control already on screen above it. A button repeating a pointer is a
#: second way to do one thing, which is the divergence this pass closes.
_POINTERS = {"create/mesh", "create/rig", "create/export", "poser", "review", "troupe", "muse"}


def test_every_imperative_placeholder_offers_the_thing_it_asks_for():
    from warlock.studio.panes import overlay

    for key, (_icon, _title, hint) in overlay.PLACEHOLDERS.items():
        if key in _POINTERS:
            continue
        first = hint.split()[0].rstrip(",")
        if first not in _IMPERATIVES:
            continue
        assert key in overlay.ACTIONS, f"{key} tells the reader to {first.lower()}, with no button"


def test_every_pointer_hint_really_is_a_pointer():
    """The exemption list is not a place to park work: an exempt entry must
    actually be pointing somewhere else."""
    from warlock.studio.panes import overlay

    for key in _POINTERS:
        hint = overlay.PLACEHOLDERS[key][2]
        assert hint.startswith("Pick ") or "above" in hint or "on the left" in hint, (key, hint)


def test_the_placeholder_table_stays_data():
    """The action is resolved at draw time. A callable in the table would drag
    every mode module in behind an import of the sentences."""
    from warlock.studio.panes import overlay

    for key, entry in overlay.PLACEHOLDERS.items():
        assert len(entry) == 3, key
        assert all(isinstance(part, str) for part in entry), key


def test_centred_empty_forwards_its_action_to_the_one_empty_state():
    from warlock.studio import widgets
    from warlock.studio.panes import overlay

    assert "action" in inspect.signature(overlay.centred_empty).parameters
    assert "action=action" in inspect.getsource(overlay.centred_empty)
    assert "action" in inspect.signature(widgets.empty_state).parameters


def test_action_for_binds_the_ctx_and_is_none_where_there_is_nothing_to_do():
    from warlock.studio.panes import overlay

    ctx = SimpleNamespace(state=SimpleNamespace(focus_key={}, focus_moved=False))
    label, run = overlay.action_for(ctx, "create/reference")
    assert label == "Write a brief"
    run()
    assert ctx.state.focus_key["brief"] == "prompt"
    assert ctx.state.focus_moved is True
    assert overlay.action_for(ctx, "review") is None


def test_the_packwright_button_opens_the_picker_that_exists():
    """``ask_add_image`` does not exist; ``ask_add_sources`` is the picker."""
    from warlock.studio import packwright_mode
    from warlock.studio.panes import overlay

    assert callable(packwright_mode.ask_add_sources)
    assert "packwright_mode.ask_add_sources(ctx)" in inspect.getsource(overlay._packwright_add)


def test_the_clay_button_never_names_a_generator():
    """The registry is data (``clay_props``' rule), and the button that adds a
    primitive lives under the same rule."""
    from warlock.studio.clay import primitives as bp
    from warlock.studio.panes import overlay

    source = inspect.getsource(overlay._clay_box)
    for name in bp.GENERATORS:
        assert f'"{name}"' not in source, f"{name} is hardcoded in overlay"


# --- item 2: Clay's properties pane tells the truth about a multi-selection --


def _clay_props_ctx(monkeypatch, selection: set[str]):
    """``clay_props._body`` with everything but the empty-state stubbed out.

    Cheaper than a real imgui frame, and the claim is about *which sentence*
    is drawn rather than about the frame surviving -- the smoke suite owns
    that half.
    """
    from warlock.studio.panes import clay_props

    drawn: list[tuple[str, str]] = []
    doc = SimpleNamespace(
        selection=selection,
        element_mode="object",
        by_uid=lambda uid: None,
    )
    tab = SimpleNamespace(doc=doc, saving=False)
    state = SimpleNamespace(active=tab)
    monkeypatch.setattr(clay_props.clay_mode, "ensure", lambda ctx: state)
    monkeypatch.setattr(clay_props.widgets, "section", lambda *a, **k: None)
    monkeypatch.setattr(clay_props.manual_render, "help_button", lambda *a, **k: None)
    monkeypatch.setattr(
        clay_props.widgets,
        "empty_state",
        lambda icon, title, hint="", **k: drawn.append((title, hint)),
    )
    clay_props._body(SimpleNamespace())
    return drawn


def test_clay_says_how_many_objects_are_selected_instead_of_nothing_selected(monkeypatch):
    """With sixteen objects lit up the pane said "Nothing selected", which the
    viewport plainly contradicts."""
    drawn = _clay_props_ctx(monkeypatch, {"a", "b"})
    assert drawn == [("2 objects selected", "Select one to edit it.")]


def test_clay_still_says_nothing_selected_when_nothing_is(monkeypatch):
    drawn = _clay_props_ctx(monkeypatch, set())
    assert drawn == [("Nothing selected", "Click an object in the viewport.")]


def test_the_multi_selection_refusal_itself_is_unchanged(monkeypatch):
    """Only the sentence changed: ``_selected`` still refuses to edit one of
    many, which is the whole reason the branch exists."""
    from warlock.studio.panes import clay_props

    doc = SimpleNamespace(selection={"a", "b"}, by_uid=lambda uid: "an object")
    assert clay_props._selected(doc) is None
    one = SimpleNamespace(selection={"a"}, by_uid=lambda uid: "an object")
    assert clay_props._selected(one) == "an object"


@pytest.mark.parametrize("count", [3, 16])
def test_the_count_is_the_real_one(monkeypatch, count):
    drawn = _clay_props_ctx(monkeypatch, {str(index) for index in range(count)})
    assert drawn[0][0] == f"{count} objects selected"
    assert re.fullmatch(r"\d+ objects selected", drawn[0][0])


# --- item 6: Settings speaks in one register --------------------------------
#
# ``panes/app_settings.py`` was drawing three registers on one page: a
# ``forms.Form`` field (small caps through ``widgets.field_label``), a raw
# ``controls.*`` call carrying its own trailing sentence-case label, and a
# bare ``imgui.text`` used as a name column. The decision of 2026-09-05 is
# that ``field_label`` is the one field face, so the raw labelled controls
# move onto the form and the bare text moves onto a widgets wrapper.
#
# This is pass 2's ``test_a_form_field_is_labelled_in_the_one_field_face``
# (tests/test_ux_consistency_pass2.py) extended to the call sites: that test
# pins the face inside ``Form._label``, and these pin that Settings actually
# goes through it.


def _app_settings_source() -> str:
    return _pane_sources()["app_settings.py"]


#: Raw ``controls.*`` calls take a *label* as their first argument, and imgui
#: draws it beside the control. A quoted first argument that is not an
#: ``##``-hidden id is therefore a second field face on the page.
_LABELLED_RAW = re.compile(
    r"controls\.(slider_float|slider_int|checkbox|input_text|input_float|combo|drag_int)"
    r"\(\s*\n?\s*\"(?!##)([^\"]+)\"",
)


def test_settings_labels_no_field_outside_the_form():
    """``controls.slider_float("UI scale", ...)`` and
    ``controls.checkbox("Licensed for commercial use", ...)`` drew sentence
    case in the body face two rungs from a small-caps ``field_label``."""
    offenders = [match.group(0) for match in _LABELLED_RAW.finditer(_app_settings_source())]
    assert offenders == [], offenders


def test_settings_ui_scale_and_the_licence_box_are_form_fields():
    """The two migrated call sites, named, so a revert is a failure rather
    than a silently absent assertion."""
    source = _app_settings_source()
    assert 'form_ui.slider(\n        "ui_scale",\n        "UI scale",' in source
    assert 'form_ui.switch(\n            "commercial", "Licensed for commercial use"' in source


def test_the_licence_box_is_a_switch_because_the_form_has_no_checkbox():
    """A Boolean in the field grid is submitted as ``##field``; an unlabelled
    imgui checkbox is ELEV_1 on ELEV_1 and vanishes when off. ``forms.Form``
    records that where it declines to grow a ``checkbox`` method, and this is
    the claim that the record still holds."""
    from warlock.studio import forms

    assert not hasattr(forms.Form, "checkbox")
    assert callable(forms.Form.switch) and callable(forms.Form.slider)


def test_settings_draws_no_bare_imgui_text_as_a_name_column():
    """The health list ran ``imgui.text(row.name)`` under a coloured glyph,
    beside a ``muted_wrapped`` detail -- body face in a pane whose every other
    line goes through ``widgets``. The wrapped *prose* at the two
    ``imgui.text_wrapped`` sites is not a field and stays."""
    source = _app_settings_source()
    assert "imgui.text(row.name)" not in source
    assert "widgets.muted(row.name)" in source
    # Only ``##``-hidden ids may reach raw ``imgui.text``-family label calls.
    bare = re.findall(r"^\s*imgui\.text\((?!_wrapped)", source, re.M)
    assert bare == [], bare
