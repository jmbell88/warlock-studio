"""UX.md Phases 2 and 3: the parts a screenshot cannot check.

The screenshot pass (``scripts/screenshot_modes.py``) is this repo's definition
of "somebody looked at it", and it is the right instrument for the half of
these phases that is about pixels. It cannot see a rule -- that the shadow
bands tile, that a status glyph is a shape rather than two letters, that a
refusal's address survives from ``ServiceError.field`` to a ring round the
control it names -- and those are what this file pins.
"""

from __future__ import annotations

import inspect

from warlock.studio import focus, theme, tokens, widgets
from warlock.studio.panes import settings_2d
from warlock.studio.state import AppState


def _code(obj) -> str:
    """``obj``'s source with its docstring and comments taken out.

    Every assertion in this file is about what the code *does*, and this
    module's own prose quotes the strings it is asserting the absence of --
    ``"(?)"``, "separator", "platform detail" -- because a comment saying what
    a thing used to be is exactly how these changes are documented. Scanning
    raw source would make every one of those comments fail its own test.

    A scan for triple quotes rather than ``source.replace(obj.__doc__, "")``:
    Python 3.13 dedents docstrings at compile time, so the attribute no longer
    appears verbatim in the source it came from and the replace was a silent
    no-op -- which is how the first draft of this file passed by reading prose.
    """
    kept: list[str] = []
    in_doc = False
    for line in inspect.getsource(obj).splitlines():
        stripped = line.strip()
        if in_doc:
            if stripped.endswith(('"""', "'''")):
                in_doc = False
            continue
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            # A one-liner opens and closes on the same line.
            in_doc = not (len(stripped) >= 6 and stripped.endswith(quote))
            continue
        if stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


# --- Phase 2: depth ---------------------------------------------------------


def test_the_shadow_bands_tile_with_no_gap_between_them():
    """A gap between two concentric bands reads as a ring, which is the one
    thing a shadow must not look like. Each band's outer bound is the next
    one's inner, which is what makes the union a falloff."""
    steps = tokens.SHADOW_STEPS
    assert len(steps) >= 4, "two bands is the hard-edged recipe this replaced"
    for (_, outer, _), (inner, _, _) in zip(steps, steps[1:], strict=False):
        assert outer == inner
    # Monotonically fainter outward, which is what a Gaussian's tail does.
    alphas = [alpha for _i, _o, alpha in steps]
    assert alphas == sorted(alphas, reverse=True)
    assert steps[0][0] == 0.0, "the innermost band starts at the surface's edge"


def test_the_elevation_ramp_is_three_named_steps_that_increase():
    """One physical story: a card rests, a popup floats, a modal blocks."""
    ramp = tokens.ELEVATION
    assert list(ramp) == ["resting", "raised", "overlay"]
    assert ramp["resting"] < ramp["raised"] < ramp["overlay"]
    # An unknown name is a call site's mistake and not worth an exception.
    assert ramp.get("nonesuch", 1.0) == 1.0


def test_every_floating_surface_draws_the_one_shadow():
    """Three call sites, one recipe. A pane inventing its own pair of alphas is
    the state ``widgets.shadow`` was extracted to end."""
    from warlock.studio import dialogs
    from warlock.studio.panes import palette as palette_pane

    assert "shadow(" in inspect.getsource(widgets.card)
    assert "window_shadow" in inspect.getsource(dialogs.ConfirmQueue.draw)
    assert "window_shadow" in inspect.getsource(dialogs.PromptQueue.draw)
    assert "window_shadow" in inspect.getsource(palette_pane.draw)
    assert "window_shadow" in inspect.getsource(widgets.toasts)


def test_workflow_modals_use_the_same_overlay_recipe():
    """Generation and document setup must not fall outside dialog chrome."""
    from warlock.studio import dialogs
    from warlock.studio.panes import plotter_canvas, settings_3d

    owners = (
        dialogs.ConfirmQueue.draw,
        dialogs.PromptQueue.draw,
        settings_3d.matte_modal,
        plotter_canvas.setup_popup,
    )
    for owner in owners:
        source = inspect.getsource(owner)
        assert "popover_enter" in source, owner
        assert "push_surface_rounding" in source, owner
        assert 'window_shadow("overlay"' in source, owner
        assert "window_backdrop" in source, owner


def test_plain_popups_have_one_named_raised_surface_recipe():
    source = inspect.getsource(widgets.popup_chrome)
    assert 'window_shadow("raised"' in source
    assert "window_backdrop" not in source, "plain popups do not block the app"


def test_the_surface_radius_arrived_with_its_readers():
    """``RADIUS_L`` was deleted once for having none, and Phase 0 deliberately
    declined to re-add it ahead of them."""
    assert tokens.RADIUS_L > tokens.RADIUS_M
    assert "RADIUS_L" in inspect.getsource(widgets.card)
    assert "RADIUS_L" in inspect.getsource(widgets.push_surface_rounding)


# --- Phase 2: type and rhythm -----------------------------------------------


def test_a_section_heading_breathes_instead_of_ruling():
    """A rule is a line drawn where the rhythm failed."""
    source = _code(widgets.section)
    assert "separator" not in source
    assert "SP_6" in source
    # Sentence case, not the small-caps register the *field* labels keep: two
    # registers drawn the same way is the flattened hierarchy this undid.
    assert ".upper()" not in source
    assert ".upper()" in _code(widgets.field_label)


def test_the_pane_title_is_a_rung_above_a_section():
    assert "heading" in inspect.getsource(widgets.pane_title)
    assert tokens.TEXT_HEADING > tokens.TEXT_BODY


# --- Phase 2: iconography ---------------------------------------------------


def test_every_status_glyph_is_a_shape_rather_than_ascii():
    """"OK" in green beside "!" in red differ by two letters and a hue. The
    double encoding is the point and it only pays if the second channel is
    something somebody can tell apart at 11 px."""
    glyphs = list(theme.STATUS_GLYPHS.values())
    assert len(set(glyphs)) == len(glyphs), "two statuses drawn identically"
    for glyph in glyphs:
        assert len(glyph) == 1 and ord(glyph) > 0xE000, glyph


def test_the_help_marker_and_the_manual_button_use_one_glyph():
    """They mean "there is more about this" and "there is a chapter about
    this"; drawn as three ASCII characters and a circled i, nothing about them
    said they were relatives."""
    from warlock.studio import icons
    from warlock.studio.manual import render as manual_render

    assert "icons.INFO" in _code(widgets.help_marker)
    assert "icons.INFO" in _code(manual_render.help_button)
    assert '"(?)"' not in _code(widgets.help_marker)
    assert icons.INFO


# --- Phase 2: the switch ----------------------------------------------------


def test_the_switch_splits_its_track_at_every_group_break():
    """Splitting rather than merely spacing: a gap inside one continuous track
    reads as a missing segment, where two tracks read as two groups."""
    source = inspect.getsource(widgets.segmented_control)
    assert "breaks" in _code(widgets.segmented_control)
    assert "one track per group" in source.lower()


# --- Phase 3: field-level refusals ------------------------------------------


def test_a_refusal_that_names_no_control_records_nothing():
    """It is not one of these, and it must keep going to the toast."""
    state = AppState()
    assert state.note_field_error("", "boom") is False
    assert state.note_field_error("prompt", "") is False
    assert state.field_errors == {}


def test_a_named_refusal_is_remembered_until_the_field_changes():
    state = AppState()
    assert state.note_field_error("style_lora", "not downloaded") is True
    assert state.field_errors["style_lora"] == "not downloaded"
    state.clear_field_error("style_lora")
    assert state.field_errors == {}


def test_a_new_submit_is_judged_on_its_own():
    state = AppState()
    state.note_field_error("prompt", "too long")
    state.clear_field_errors()
    assert state.field_errors == {}
    assert "clear_field_errors" in inspect.getsource(settings_2d.generate)


def test_the_frame_loop_carries_the_refusals_address_to_the_state():
    """``ServiceError.field`` has been carried since the class was written and
    read by nothing. This is the one place every task failure passes through."""
    from warlock.studio import main

    source = inspect.getsource(main.App._collect_tasks)
    assert "note_field_error" in source
    # And the toast still goes up: the ring says *which control*, not *that
    # something happened*, and the pane may not even be on screen.
    assert "ctx.toast(" in source


def test_the_service_still_names_the_controls_the_panes_ring():
    """The two halves are wired by *name*, so a rename on either side is a ring
    that never appears. These are the fields both sides agree on."""
    import importlib
    import pkgutil

    from warlock import guidance
    from warlock import service as svc_pkg
    from warlock.service import jobs as svc_jobs
    from warlock.service import validation

    # ``jobs`` is a facade over ``_jobs_*.py`` siblings, so the refusals it used
    # to raise are raised from those -- scanned here so the claim this test
    # makes about the *service* stays a claim about the service rather than
    # about one file's current contents.
    siblings = [
        importlib.import_module(f"warlock.service.{m.name}")
        for m in pkgutil.iter_modules(svc_pkg.__path__)
        if m.name.startswith("_jobs_")
    ]
    assert siblings, "the jobs siblings are not being scanned"

    named = set()
    for module in (svc_jobs, validation, guidance, *siblings):
        for line in inspect.getsource(module).splitlines():
            if 'field="' in line:
                named.add(line.split('field="', 1)[1].split('"', 1)[0])
    panes = inspect.getsource(settings_2d) + inspect.getsource(
        __import__(
            "warlock.studio.panes.settings_3d", fromlist=["settings_3d"]
        )
    )
    for key in ("prompt", "base_model", "count", "bg_removal", "profile"):
        assert key in named, f"nothing raises a refusal naming {key}"
    # ``check_weights`` names its three optional selections from a loop
    # variable, so the literal scan above cannot see them; the table it walks
    # is the statement of intent.
    assert '("style_lora", "lora"' in inspect.getsource(validation.check_weights)
    for key in ("prompt", "base_model", "style_lora", "count", "platform"):
        assert f'field_error(ctx.state, "{key}")' in panes


# --- Phase 3: the common path -----------------------------------------------


def test_the_fold_is_derived_from_the_taxonomy_rather_than_written_out():
    """A field added to ``GUIDANCE_GROUPS`` is folded by having been added."""
    folded = settings_2d.folded_fields({})
    for _title, fields in settings_2d.GUIDANCE_GROUPS:
        for name in fields:
            assert name in folded
    assert "base_model" in folded and "ref_path" in folded
    # And the common path is not in it: those are the controls a first job
    # needs, which is the whole point of the split.
    for name in ("prompt", "count", "seed", "output"):
        assert name not in folded


def test_the_closed_fold_says_how_much_is_behind_it():
    """A form restored with a style, a genre and a conditioning image set looks
    identical to an empty one with the fold shut."""
    from warlock.studio.state import default_form_2d

    empty = default_form_2d()
    assert "nothing set" in settings_2d.more_summary(empty)
    filled = {**empty, "genre": "fantasy", "art_style": "ps2"}
    assert "2 set" in settings_2d.more_summary(filled)


def test_a_refusal_behind_the_fold_opens_it():
    """A ring round a control nobody can see is not a pointer."""
    source = inspect.getsource(settings_2d._more)
    assert "request_open" in source
    # The which-folds decision moved into ``folds_to_open`` (which consults
    # ``folded_fields`` and adds the nested Advanced header when the refusal
    # names one of its fields); the pane is pinned to route through it.
    assert "folds_to_open" in source


def test_the_first_screen_is_the_common_path():
    """Prompt, preset, output and Run above the fold; everything else behind
    one reveal."""
    source = inspect.getsource(settings_2d.draw)
    for call in ("_presets", "_output", "_prompt", "_run_controls", "_more"):
        assert call in source
    assert source.index("_run_controls") < source.index("_more(")
    # The three that moved are reached through the fold and from nowhere else.
    body = inspect.getsource(settings_2d._more)
    for call in ("_guidance", "_reference", "_advanced"):
        assert call in body
        assert f"        {call}(ctx, form)" not in source


# --- Phase 3: the focus ring ------------------------------------------------


class _FakeState:
    def __init__(self) -> None:
        self.focus_key: dict[str, str] = {}
        self.focus_order: dict[str, list[str]] = {}
        self.focus_moved = False


def _record(state, pane, keys):
    focus.begin(state, pane)
    state.focus_order[pane] = list(keys)


def test_the_ring_wraps_in_both_directions():
    """A form is a fixed ring of controls, where a two-hundred-row list is
    not -- which is why Home's tiles wrap and the library's arrows clamp."""
    state = _FakeState()
    _record(state, "2d", ["prompt", "count", "generate"])
    assert focus.move(state, "2d", 1)
    assert state.focus_key["2d"] == "prompt"
    focus.move(state, "2d", -1)
    assert state.focus_key["2d"] == "generate"
    focus.move(state, "2d", 1)
    assert state.focus_key["2d"] == "prompt"


def test_a_cursor_on_a_control_that_is_gone_restarts_rather_than_guessing():
    state = _FakeState()
    _record(state, "2d", ["prompt", "count"])
    state.focus_key["2d"] = "style_lora"
    focus.move(state, "2d", 1)
    assert state.focus_key["2d"] == "prompt"


def test_moving_asks_for_imgui_focus_exactly_once():
    """``set_keyboard_focus_here`` is a command, not a state: issuing it every
    frame resets a text cursor mid-word."""
    source = inspect.getsource(focus.item)
    assert "state.focus_moved = False" in source
    assert source.index("state.focus_moved = False") < source.index(
        "imgui.set_keyboard_focus_here()"
    )


def test_the_ring_is_pumped_before_the_order_is_rebuilt():
    """``begin`` clears the order and ``move`` needs the one the last frame
    recorded, which is also the only order describing what is on screen."""
    for pane in (settings_2d.draw,):
        source = inspect.getsource(pane)
        assert source.index("focus.pump") < source.index("focus.begin")


def test_tab_is_read_from_imgui_rather_than_from_the_event_loop():
    """``App._shortcut`` is skipped entirely while a text field has the
    keyboard, and tabbing out of the prompt box is the whole job."""
    from warlock.studio import main

    assert "want_text_input" in inspect.getsource(main.App._events)
    source = inspect.getsource(focus.pump)
    assert "imgui.is_key_pressed" in source
    assert "is_popup_open" in source


def test_both_generate_panes_carry_a_ring_that_ends_on_the_button():
    from warlock.studio.panes import settings_3d

    assert 'focus.item(ctx.state, FOCUS_PANE, "generate")' in inspect.getsource(
        settings_2d._submit
    )
    assert 'focus.item(ctx.state, FOCUS_PANE, "make3d")' in inspect.getsource(
        settings_3d._submit
    )


# --- Phase 3: undo as forgiveness -------------------------------------------


def test_the_toast_vocabulary_grew_undo():
    assert widgets.TOAST_ACTIONS["undo"] == "Undo"


def test_trashing_offers_the_undo_it_was_already_relying_on():
    """The card's Delete asks nothing on the grounds that the trash *is* the
    confirmation -- which was true and invisible."""
    from warlock.studio import main
    from warlock.studio.panes import library

    source = inspect.getsource(library.delete_asset)
    assert '"undo"' in source
    handler = inspect.getsource(main.App._toast_action)
    assert 'name == "undo"' in handler
    assert "library.restore_asset" in handler


def test_the_confirms_that_stay_are_the_irreversible_ones():
    """Prune deletes from disk and empty-trash is the trash; neither has an
    undo to offer, so both keep their question."""
    from warlock.studio.panes import library

    source = inspect.getsource(library)
    assert "empty_trash" in source
    assert "dialogs.Confirm(" in source


# --- naming -----------------------------------------------------------------


def test_the_two_detail_controls_have_real_and_different_names():
    """"Platform detail" and "Detail" were one collision papered over with a
    tooltip on each apologising for the other."""
    from warlock.studio.panes import settings_3d

    two_d = _code(settings_2d._guidance)
    three_d = _code(settings_3d.draw)
    assert 'field_label("detail brief")' in two_d
    assert '"Mesh resolution"' in three_d
    assert '"platform detail"' not in two_d
    assert 'labeled_combo(\n            "Detail"' not in three_d


def test_the_manual_calls_them_what_the_app_calls_them():
    from pathlib import Path

    import warlock

    docs = Path(warlock.__file__).resolve().parents[2] / "docs" / "manual"
    if not docs.is_dir():  # a wheel install has no docs tree
        return
    body = "\n".join(p.read_text(encoding="utf-8") for p in docs.glob("*.md"))
    assert "detail brief" in body
    assert "Mesh resolution" in body
    assert "platform detail" not in body


# --- the navigation control --------------------------------------------------


def test_the_navigation_control_carries_navigation_and_nothing_else():
    """Phase 2 took Quit out of the switch and put it in a header strip beside
    a resource readout and a shortcuts button. The UI redesign, wave 3 took the whole
    header: the readout was developer chrome on screen permanently (F10 still
    raises the meter, which was always the real tool), and a destructive control
    in the shell was only less bad in the corner than it had been in the switch.

    The rail is the navigation control now, and what it draws is modes, a health
    badge when something is failing, the manual and its own collapse toggle.
    """
    from warlock.studio import main, rail

    source = inspect.getsource(rail.draw)
    assert "modes.RAIL_GROUPS" in source
    assert "quit" not in source.lower()
    assert "status_readout" not in source
    assert not hasattr(main.App, "_mode_switch")
