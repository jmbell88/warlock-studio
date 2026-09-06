"""A refusal's *address*, from the service to the ring on the control.

``ServiceError.field`` is recorded centrally in ``main._collect_tasks`` and has
been since UX.md Phase 3. What was missing is the other end: a pane that draws
the named control has to read ``state.field_errors``, or the address is
recorded and thrown away and the user gets the same red toast in the corner
whichever control was at fault.

Two invariants here, and they are the two halves of that.

* **A form that draws fields takes ``errors``.** Not every ``forms.Form`` does
  -- several are used for their label grid alone -- so the rule is conditional
  on the form actually having typed fields on it.
* **Editing the control clears its ring.** ``settings_2d`` said this by hand at
  some thirty call sites; ``Form.on_edit`` says it once, which is what fixed
  the three panes that already passed ``errors`` and cleared nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from warlock.studio import forms

#: The panes whose forms carry fields a refusal can name. Each entry is the
#: module path and the ``forms.Form`` id it opens.
FIELD_FORMS = (
    ("panes/settings_2d.py", "create-2d"),
    ("panes/settings_3d.py", "create-3d"),
    ("panes/app_settings.py", "lora-import"),
    ("panes/app_settings.py", "lora-train"),
    ("panes/remesh_panel.py", "remesh-settings"),
    ("panes/sheet_panel.py", "sheet-settings"),
    ("panes/sprite_panel.py", "sprite-settings"),
    ("panes/troupe_settings.py", "troupe-settings"),
    ("panes/retarget_panel.py", "retarget-settings"),
)


def _source(rel: str) -> str:
    from warlock import studio

    return (Path(studio.__file__).parent / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize(("rel", "form_id"), FIELD_FORMS)
def test_a_form_with_fields_reads_the_recorded_refusal(rel, form_id):
    source = _source(rel)
    opened = re.search(rf'forms\.Form\(\s*"{re.escape(form_id)}"(.*?)\)\s*as ', source, re.S)
    assert opened is not None, f"{rel}: no `forms.Form({form_id!r}) as ...`"
    assert "errors=ctx.state.field_errors" in opened.group(1), rel


@pytest.mark.parametrize(("rel", "form_id"), FIELD_FORMS)
def test_and_clears_it_when_the_control_is_edited(rel, form_id):
    """A ring that outlived the edit would be an app arguing about a value that
    is no longer there."""
    source = _source(rel)
    opened = re.search(rf'forms\.Form\(\s*"{re.escape(form_id)}"(.*?)\)\s*as ', source, re.S)
    assert opened is not None
    block = opened.group(1)
    # ``settings_2d`` and ``settings_3d`` predate ``on_edit`` and clear by hand
    # beside each control they draw outside the form; either spelling satisfies
    # the rule, and a new form should use ``on_edit``.
    assert "on_edit=ctx.state.clear_field_error" in block or (
        "clear_field_error(" in source
    ), rel


def test_the_retarget_field_ids_are_the_refusals_own_names():
    """The field id *is* the address. ``optimize_job`` refuses with
    ``remesh_profile`` and ``custom_faces`` -- which is what ``remesh_panel``,
    the other pane over the same call, already names its fields -- while this
    one called them ``profile`` and ``custom_triangles``, so even with
    ``errors`` wired the ring would have had nothing to land on."""
    source = _source("panes/retarget_panel.py")
    assert '"remesh_profile"' in source and '"custom_faces"' in source
    # The form *dict* keys are the door's parameter names and are a different
    # vocabulary; they stay.
    assert 'profile=form["profile"]' in source


def test_every_submit_that_can_be_refused_by_name_drops_last_times_rings():
    """A new submit is judged on its own."""
    for rel in (
        "panes/settings_2d.py",
        "panes/settings_3d.py",
        "panes/sheet_panel.py",
        "panes/sprite_panel.py",
        "panes/retarget_panel.py",
        "panes/remesh_panel.py",
        "panes/app_settings.py",
        "troupe_mode.py",
    ):
        assert "clear_field_errors()" in _source(rel), rel


#: The other ``forms.Form`` sites, and why each takes no ``errors``. Listed
#: rather than left out, so a field added to one of them fails here instead of
#: silently joining the class this file exists to close.
LAYOUT_ONLY = {
    # Its own settings never leave the process, and the nested call is a
    # re-entry of the same draw with the parent's form context.
    "application-settings/interface": "no service call behind it",
    # Read-only: one ``form_ui.readonly`` row per recorded parameter.
    "generation-settings": "readonly rows",
    # The label grid alone -- these three draw no typed field at all.
    "pose-controls": "layout only",
    "poser-controls": "layout only",
    "poser-clips": "layout only",
    # ``review_mode.record`` is inline on the frame thread on purpose and
    # catches its own ``ServiceError``, so nothing here ever reaches
    # ``main._collect_tasks`` to be recorded. Its two refusals are unreachable
    # from the buttons besides -- ``grade_buttons`` cannot emit an out-of-range
    # grade, and ``tag_toggles`` cannot emit an unknown tag.
    "review-verdict": "refused inline, never recorded",
}


def test_every_form_is_either_wired_or_listed():
    """The durable half: a new form with fields and no ``errors`` is the defect
    this file closes, and it must not be possible to add one quietly."""
    from warlock import studio

    root = Path(studio.__file__).parent
    known = {form_id for _rel, form_id in FIELD_FORMS} | set(LAYOUT_ONLY)
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        for form_id in re.findall(r'forms\.Form\(\s*"([^"]+)"', path.read_text(encoding="utf-8")):
            found.add(form_id)
    assert found <= known, f"unlisted forms: {sorted(found - known)}"
    assert known <= found, f"listed but gone: {sorted(known - found)}"


# --- the address really is an address -----------------------------------------
#
# The half a wiring check cannot see: a pane can read ``state.field_errors``
# faithfully and still light nothing up, because the name the service refuses
# with is not a name the pane draws. That is exactly what ``retarget_panel``
# was doing -- ``profile`` on the form, ``remesh_profile`` in the refusal.

#: Refusal addresses that deliberately have no control on the pane that
#: submitted them, and why. These reach the user as the toast (and, for the two
#: weights rows, as ``model_gate``'s Install offer beside it).
ELSEWHERE = {
    "base_model": "chosen in Create's recipe column, not here",
    "style_lora": "chosen in Create's recipe column, not here",
    "texture_size": "remesh_panel's own field; not on the retarget form",
    "custom_faces": "only on remesh_panel and retarget_panel's custom branch",
    "remesh_profile": "the budget combo, under its own name on both panes",
    "prompt": "the rework prompt, drawn by the rework panel",
    "strength": "the rework strength, drawn by its own panel",
    "control": "the rework conditioning, drawn by its own panel",
    "control_scale": "the rework conditioning, drawn by its own panel",
    "candidates": "derived from the two combos; not a control of its own",
}


# --- Create's Character arm: two vocabularies, one address --------------------
#
# ``Recipe.from_dict`` refuses in the *recipe's* words -- ``logical_size``,
# ``appearance``, ``animations`` -- because those are the keys it validates, and
# ``service.errors.invalid_from`` passes that address through untouched. The
# controls are named for the form keys they persist under (``character_pixel``,
# ``character_body``, ``character_actions``). That is exactly the shape of drift
# this file caught in ``retarget_panel`` -- ``profile`` on the form,
# ``remesh_profile`` in the refusal -- so the map between them is asserted in
# both directions rather than trusted.


def test_the_character_panes_own_refusals_name_its_own_controls():
    """Every ``Problem`` the pane raises has a control at the other end.

    The pane's problems reach the user through ``settings_2d.refuse``, which
    records each one against ``problem.field`` and rings whatever answers to
    that name -- so a field the pane does not draw is an address recorded and
    thrown away.
    """
    import inspect

    from warlock.studio.panes import settings_character

    drawn = _form_field_ids("panes/settings_character.py")
    source = inspect.getsource(settings_character.problems)
    source += inspect.getsource(settings_character._no_species)
    fields = set(re.findall(r'widgets\.Problem\([^)]*?,\s*"([a-z_]+)"\s*\)', source, re.S))
    assert fields, "the extraction found nothing, which is not an answer"
    for name in sorted(fields):
        assert name in drawn or name in ELSEWHERE, (
            f"settings_character refuses with {name!r} and nothing on that pane "
            "answers to the name"
        )


def test_a_recipe_refusal_is_re_filed_under_the_control_it_is_about():
    """The other direction, and the one a call-site scan cannot see.

    ``mirror_errors`` is what turns the door's ``logical_size`` into the pane's
    ``character_pixel``. Every alias it maps has to be a name ``Recipe`` really
    refuses with, and every control it maps onto has to be one the pane really
    draws -- otherwise the map is a comment rather than a mechanism.
    """
    from types import SimpleNamespace

    from warlock.characters.errors import CharacterError
    from warlock.characters.recipe import DEFAULT_RECIPE, Recipe
    from warlock.studio.panes import settings_character

    #: One request per address, each wrong in exactly the way that address
    #: names. Provoked rather than pattern-matched out of the source: several
    #: of these refusals go through ``_on_ladder`` and ``_integer``, which take
    #: the field name positionally, so a regex over ``field="..."`` calls four
    #: real addresses imaginary.
    provoke = {
        "family": {"family": "not-a-species"},
        "family_version": {"family_version": 99},
        "theme": {"theme": "not-a-look"},
        "camera": {"camera": "nowhere"},
        "elevation": {"elevation": 200.0},
        "animations": {"animations": {"flying": 4}},
        "directions": {"directions": 3},
        "logical_size": {"logical_size": 17},
        "colors": {"colors": 7},
        "appearance": {"appearance": {"not-a-channel": 1.0}},
        "name": {"name": "x" * 500},
    }
    drawn = _form_field_ids("panes/settings_character.py")
    mapped = {alias for aliases in settings_character.RECIPE_FIELDS.values() for alias in aliases}
    assert mapped == set(provoke), "every address this map claims, and no other"
    for control, aliases in settings_character.RECIPE_FIELDS.items():
        assert control in drawn, f"{control} is mapped but nothing draws it"
        for alias in aliases:
            with pytest.raises(CharacterError) as excinfo:
                Recipe.from_dict({**DEFAULT_RECIPE.as_dict(), **provoke[alias]})
            assert excinfo.value.field == alias, (
                f"{alias!r} is mapped onto {control!r} and the recipe refuses "
                f"it under {excinfo.value.field!r} instead"
            )

    # And it actually moves one. A door refusal about the colour count must
    # come back rung on the Colours control.
    state = SimpleNamespace(field_errors={"colors": "colours must be one of [8, 16, 32, 64]"})
    settings_character.mirror_errors(SimpleNamespace(state=state))
    assert state.field_errors["character_colors"] == state.field_errors["colors"]
    # An address the pane already holds is never overwritten: the pane's own
    # sentence is about this frame's form, and the door's is about the last
    # submitted one.
    state.field_errors["character_pixel"] = "mine"
    state.field_errors["logical_size"] = "theirs"
    settings_character.mirror_errors(SimpleNamespace(state=state))
    assert state.field_errors["character_pixel"] == "mine"


def _field_names(fn) -> set[str]:
    """Every literal ``field="..."`` in one function's own source."""
    import inspect

    return set(re.findall(r'field="([a-z_]+)"', inspect.getsource(fn)))


def _form_field_ids(rel: str) -> set[str]:
    """Every field id the pane names -- on its form, or by hand to
    ``widgets.field_error``/``clear_field_error``."""
    source = _source(rel)
    # Hyphens allowed in the pattern deliberately, so an id that *cannot* be an
    # address is still collected and compared -- ``sheet_panel``'s Name field
    # was ``sheet-name`` while the refusal said ``name``, and a pattern that
    # skipped it would have read as "the pane draws no such control" either
    # way, which is the right verdict for the wrong reason.
    ids = set(re.findall(r'form_ui\.[a-z_]+\(\s*(?:#[^\n]*\n\s*)*"([a-z_-]+)"', source))
    ids |= set(re.findall(r'field_error\(\s*ctx\.state,\s*"([a-z_]+)"', source))
    ids |= set(re.findall(r'clear_field_error\(\s*"([a-z_]+)"', source))
    # ``Form.note`` -- an address that names a composition rather than a
    # control, drawn above the block it is about.
    ids |= set(re.findall(r'\.note\(\s*"([a-z_]+)"', source))
    # A loop that draws several fields under one name -- ``sprite_panel``'s two
    # seeds. The id is a variable there, so the pattern above cannot see it;
    # the loop's own tuple is where the names actually are.
    if re.search(r"form_ui\.[a-z_]+\(\s*field\b", source):
        for group in re.findall(r"for field in \(([^)]*)\)", source):
            ids |= set(re.findall(r'"([a-z_]+)"', group))
    return ids


def test_every_refusal_a_pane_can_provoke_names_something_that_pane_draws():
    from warlock.service import _jobs_rework, sheets, sprites, troupe

    cases = (
        ("panes/sheet_panel.py", sheets.create_sheet),
        ("panes/sheet_panel.py", sheets.create_pixel_sheet),
        ("panes/sprite_panel.py", sprites.create_sprite_synthesis),
        ("panes/retarget_panel.py", _jobs_rework.optimize_job),
        ("panes/remesh_panel.py", _jobs_rework.remesh_job),
        ("panes/troupe_settings.py", troupe.check_troupe),
    )
    for rel, fn in cases:
        drawn = _form_field_ids(rel)
        for name in sorted(_field_names(fn)):
            assert name in drawn or name in ELSEWHERE, (
                f"{rel} submits {fn.__name__}, which refuses with {name!r} -- "
                "and nothing on that pane answers to the name"
            )


# --- the mechanism itself -----------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def __call__(self, field: str) -> None:
        self.cleared.append(field)


def test_a_form_reports_an_edit_and_only_an_edit():
    """``_answer`` is on every typed helper, so no helper can be added without
    it -- and it fires on ``changed``, never on a redraw of the same value."""
    seen = _Recorder()
    form = forms.Form("probe", errors={"seed": "too big"}, on_edit=seen)

    assert form._answer("seed", (True, 7)) == (True, 7)
    assert seen.cleared == ["seed"]
    assert form._answer("seed", (False, 7)) == (False, 7)
    assert seen.cleared == ["seed"], "a redraw is not an edit"


def test_a_form_without_an_owner_still_hands_the_answer_back():
    """``on_edit`` is optional: the layout-only forms pass neither."""
    form = forms.Form("probe")
    assert form._answer("seed", (True, 7)) == (True, 7)


def test_note_is_silent_when_there_is_nothing_recorded():
    """It is called unconditionally from a draw, so the ordinary case -- no
    refusal outstanding -- must add nothing to the pane. Asserted through the
    return value rather than a frame, since the drawing half needs a context."""
    assert forms.Form("probe").note("layout") is False


def test_the_error_map_is_a_snapshot():
    """Taken at ``__enter__`` time, so a clear partway down the form cannot
    change what the rest of the frame is showing."""
    live = {"seed": "too big"}
    form = forms.Form("probe", errors=live)
    live.clear()
    assert form._error("seed") == "too big"


# --- the Rig stage's bare combo, outside forms.Form ----------------------------
#
# ``stage_rig.py`` and the identical combo in ``settings_3d.py`` draw
# ``widgets.field_error(ctx.state, "rig_template")`` with a bare
# ``widgets.labeled_combo`` -- no ``forms.Form``, so neither is reachable from
# ``FIELD_FORMS`` above. That made the gap invisible to this file's own sweep:
# ``create_rig`` and ``service/poses.py`` both refuse with
# ``field="rig_template"``, and once they did, the ring stayed lit on the Rig
# stage forever -- arguing about a value the user had long since changed --
# until some unrelated pane's submit happened to call the global
# ``clear_field_errors()``.


def test_changing_the_rig_stage_skeleton_clears_its_field_error_ring():
    """The 2026-09-05 audit, finding create-05.

    ``remesh_panel``/``retarget_panel`` clear a field's ring the moment its
    control is edited (via ``forms.Form(on_edit=...)``) and again before a
    fresh submit is judged on its own. ``stage_rig._skeleton_picker`` bypasses
    ``forms.Form`` entirely and did neither -- and ``settings_3d._rig`` draws
    the same combo the same way.
    """
    import inspect

    from warlock.studio.panes import settings_3d, stage_rig

    picker_src = inspect.getsource(stage_rig._skeleton_picker)
    assert 'clear_field_error("rig_template")' in picker_src, (
        "stage_rig._skeleton_picker never clears rig_template's ring when the "
        "skeleton combo is changed"
    )

    draw_src = inspect.getsource(stage_rig.draw)
    assert "clear_field_errors()" in draw_src, (
        "stage_rig.draw submits a fresh rig without clearing last time's rings first"
    )

    rig_src = inspect.getsource(settings_3d._rig)
    assert 'clear_field_error("rig_template")' in rig_src, (
        "settings_3d._rig has the identical gap: its rig_template combo never "
        "clears the ring either"
    )
