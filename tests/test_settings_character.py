"""Create's Character column: the override model, the refusals, and the door.

Two promises this file exists to keep, and both of them are about *not* being
helpful behind the user's back.

**The prompt fills the form and the form is never its prisoner.** A brief edit
writes the fields nobody has touched and leaves the rest; touching a control
claims it; "Reset to prompt" gives it back. Without that the two directions
fight -- either a typed prompt cannot fill anything after the first frame, or
every keystroke silently undoes the species the user just picked.

**Nothing substitutes a species.** ``characters.resolve`` never returns one for
a creature this program does not model, and the pane keeps that promise at the
other end: ``character_family`` stays empty, Generate is refused in
``resolve.offer_sentence``'s exact words, and the substitution happens in
exactly one place -- :func:`settings_character.apply_offer`, reachable only from
a button the user presses.

The species-dependent cases are parameterised over the registry rather than
written about the ogre: a sibling adding a species must not be able to make a
test about "the default character" pass for the wrong reason.
"""

from __future__ import annotations

import inspect

import pytest

from warlock.characters import resolve as resolve_mod
from warlock.characters.family import families
from warlock.characters.recipe import Recipe
from warlock.studio import create_assets, generation_workspace
from warlock.studio.panes import settings_2d, settings_character
from warlock.studio.state import AppState

# --- the harness --------------------------------------------------------------


class _Ctx:
    """A headless ``ctx``: real ``AppState``, recorded submits and toasts."""

    def __init__(self, svc, *, rigging: bool = True) -> None:
        self.svc = svc
        self.state = AppState()
        self.rigging_available = rigging
        self.model_rows: list[dict] = []
        self.submitted: list[tuple[str, object]] = []
        self.toasts: list[str] = []
        self.mode_set: list[str] = []
        self.accept = True

    # ``TaskRunner.submit``'s contract, minus the thread: the callable is run
    # here so a test can see what the door was handed.
    def submit(self, key, fn, *args, **kwargs):
        if not self.accept:
            return False
        self.submitted.append((key, fn(*args, **kwargs)))
        return True

    def busy(self, _key: str) -> bool:
        return False

    def toast(self, message, *_a, **_k) -> None:
        self.toasts.append(str(message))


@pytest.fixture
def ctx(svc):
    return _Ctx(svc)


def _form(prompt: str = "") -> dict:
    form = AppState().form_2d
    form["asset_type"] = "character"
    form["prompt"] = prompt
    create_assets.sync_legacy_fields(form)
    return form


#: One species per body plan, named by key. Parameterising over all thirty-one
#: would be thirty-one copies of the same assertion; one of each plan is what
#: actually varies -- the channel set, the clip library and the skeleton all
#: come off the archetype.
def _one_per_archetype() -> list[str]:
    seen: dict[str, str] = {}
    for key, fam in families().items():
        seen.setdefault(fam.archetype, key)
    return sorted(seen.values())


ONE_PER_ARCHETYPE = _one_per_archetype()


# --- the override model -------------------------------------------------------


@pytest.mark.parametrize("species", ONE_PER_ARCHETYPE)
def test_a_prompt_edit_fills_only_the_fields_nobody_has_touched(species):
    """The whole point of the override list, in one pass.

    The user picks a size by hand; the brief then names a different species. The
    species has to move -- it is what the words say -- and the size must not,
    because it is a decision somebody made and a prompt edit is not a request to
    undo it.
    """
    fam = families()[species]
    form = _form("a wizard")
    settings_character.sync_from_prompt(form)
    assert form["character_family"] == "wizard"

    form["character_pixel"] = "128"
    settings_character.touched(form, "character_pixel")

    form["prompt"] = f"a {fam.aliases[0]}"
    assert settings_character.sync_from_prompt(form) is True
    assert form["character_family"] == species
    assert form["character_pixel"] == "128", "a claimed control is the user's"


def test_a_field_the_new_brief_says_nothing_about_goes_back_to_its_default():
    """Not to whatever the *previous* brief left in it.

    "a wolf" after "an attacking fire ogre" has to produce a wolf with the
    default actions and no fire, or the form accumulates a character out of two
    briefs the user never wrote together.
    """
    form = _form("an attacking fire ogre")
    settings_character.sync_from_prompt(form)
    assert (form["character_theme"], form["character_actions"]) == ("fire", "attack")

    form["prompt"] = "a wolf"
    settings_character.sync_from_prompt(form)
    assert form["character_family"] == "wolf"
    assert form["character_theme"] == settings_character.THEME_UNSET
    assert form["character_actions"] == AppState().form_2d["character_actions"]


def test_the_scan_runs_on_a_prompt_change_and_on_nothing_else():
    """The cache is what makes calling this from a draw affordable."""
    form = _form("a fire ogre")
    assert settings_character.sync_from_prompt(form) is True
    assert settings_character.sync_from_prompt(form) is False
    form["prompt"] = "a fire ogre "
    assert settings_character.sync_from_prompt(form) is True


def test_an_explicit_change_is_recorded_as_an_override():
    form = _form("a goblin")
    settings_character.sync_from_prompt(form)
    assert settings_character.overrides_of(form) == []
    settings_character.touched(form, "character_family")
    assert settings_character.overrides_of(form) == ["character_family"]
    # And recorded once, however many times the control is moved.
    settings_character.touched(form, "character_family")
    assert settings_character.overrides_of(form) == ["character_family"]


def test_reset_to_prompt_forgets_every_override_and_reads_the_brief_again():
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    form["character_family"] = "wolf"
    form["character_theme"] = settings_character.THEME_UNSET
    settings_character.touched(form, "character_family")
    settings_character.touched(form, "character_theme")

    settings_character.reset_to_prompt(form)

    assert settings_character.overrides_of(form) == []
    assert (form["character_family"], form["character_theme"]) == ("ogre", "fire")


# --- the promise: never a silent substitution ---------------------------------


UNSUPPORTED = ("a manticore", "a kraken", "a phoenix", "a chimera")


@pytest.mark.parametrize("prompt", UNSUPPORTED)
def test_an_unsupported_creature_is_a_problem_on_the_prompt_and_never_a_species(
    ctx, prompt
):
    """The heart of it. Three things at once, because they are one thing.

    The form holds no species; the refusal is on ``prompt``; and its words are
    ``resolve.offer_sentence``'s, character for character, because that function
    is the one home for a sentence that names a substitution -- three copies of
    it are three chances to word one of them as though the substitution had
    already happened.
    """
    form = _form(prompt)
    settings_character.sync_from_prompt(form)
    resolution = settings_character.resolution_of(form)
    sentence = resolve_mod.offer_sentence(resolution)
    assert sentence is not None

    assert form["character_family"] == "", "the resolver never substitutes"
    assert resolution.offer, "and it does have something to propose"

    problems = settings_character.problems(ctx, form)
    named = [p for p in problems if getattr(p, "field", "") == "prompt"]
    assert len(named) == 1
    assert str(named[0]) == sentence


@pytest.mark.parametrize("prompt", UNSUPPORTED)
def test_the_offer_is_applied_only_by_the_explicit_fix_action(ctx, prompt):
    """Reading the form, drawing it and submitting it all leave it empty.

    ``apply_offer`` is the one door, and it is reachable only from the button
    under the refusal.
    """
    form = _form(prompt)
    settings_character.sync_from_prompt(form)
    settings_character.problems(ctx, form)
    settings_character.recipe_kwargs(form, settings_character.options(ctx))
    settings_2d.generate(ctx, form)
    assert form["character_family"] == ""
    assert ctx.submitted == [], "a refused brief queues nothing"

    applied = settings_character.apply_offer(form, settings_character.options(ctx))
    expected = settings_character.resolution_of(form).offer[0]
    assert applied == expected
    assert form["character_family"] == expected
    # And it is the user's from now on: the next prompt edit leaves it.
    assert "character_family" in settings_character.overrides_of(form)


def test_only_one_place_in_the_program_writes_a_species_the_prompt_did_not_name():
    """The structural half of the promise.

    ``_fill`` writes ``resolution.family or ""`` -- never an offer -- and
    ``apply_offer`` is the only function that reads ``resolution.offer`` and
    assigns it. A second reader would be a second chance to make the
    substitution automatic, which is the failure the wording exists to prevent.
    """
    source = inspect.getsource(settings_character)
    writers = [
        name
        for name, fn in vars(settings_character).items()
        if callable(fn)
        and getattr(fn, "__module__", "") == settings_character.__name__
        and ".offer" in inspect.getsource(fn)
        and "character_family" in inspect.getsource(fn)
        and 'form["character_family"] =' in inspect.getsource(fn)
    ]
    assert writers == ["apply_offer"], writers
    assert 'resolution.family or ""' in source, "the fill never reaches for an offer"


def test_a_brief_that_names_no_creature_is_refused_in_the_same_register(ctx):
    """No offer to make, and still a sentence somebody can act on."""
    form = _form("something cool")
    settings_character.sync_from_prompt(form)
    problems = settings_character.problems(ctx, form)
    named = [p for p in problems if getattr(p, "field", "") == "prompt"]
    assert len(named) == 1
    message = str(named[0])
    assert str(len(families())) in message, "it says how many species there are"
    assert "Species" in message, "and where to pick one"


# --- one sentence, wherever it is met -----------------------------------------


def test_the_ring_the_footer_and_the_toast_all_say_the_same_sentence(ctx):
    """``problems_for`` is what makes this true, and it is why it is cached.

    The command bar asks whether Generate is live, the plan footer lists what is
    wrong, and a Ctrl+Enter refusal toasts the first problem. Three readers of
    one evaluation; three evaluations only *tend* to agree.
    """
    form = _form("a manticore")
    settings_character.sync_from_prompt(form)
    sentence = resolve_mod.offer_sentence(settings_character.resolution_of(form))

    footer = settings_2d.problems_for(ctx, form)
    assert str(footer[0]) == sentence

    settings_2d.generate(ctx, form)
    assert ctx.toasts == [sentence]
    assert ctx.state.field_errors["prompt"] == sentence
    assert ctx.submitted == []


# --- the escape routes --------------------------------------------------------


def test_the_escape_routes_keep_the_brief(ctx):
    """Both of them. A user who came here for a manticore still wants one, and
    the question is only which surface can draw it -- so a route that rewrote
    the prompt on the way would send a different request than the one they were
    refused for."""
    form = _form("a fierce manticore")
    settings_character.sync_from_prompt(form)

    settings_character.switch_to_sprite_sheet(form)
    assert form["prompt"] == "a fierce manticore"
    assert create_assets.selected(form).key == "sprite_sheet"
    assert form["output"] == "sheet"

    other = _form("a fierce manticore")
    settings_character.hand_to_troupe(ctx, other)
    assert other["prompt"] == "a fierce manticore"
    from warlock.studio import troupe_mode

    assert troupe_mode.form(ctx)["prompt"] == "a fierce manticore"
    assert ctx.state.mode == "troupe"


def test_troupes_form_has_one_construction_and_both_callers_use_it():
    """``troupe_settings._form`` moved to ``troupe_mode.form`` for this: the
    hand-off has to reach the form the pane will draw, and two constructions of
    one request are two defaults."""
    from warlock.studio import troupe_mode
    from warlock.studio.panes import troupe_settings

    assert "troupe_mode.form(ctx)" in inspect.getsource(troupe_settings.draw)
    assert callable(troupe_mode.form)


# --- Blender ------------------------------------------------------------------


def test_without_blender_the_press_is_refused_in_the_rig_stages_own_words(svc):
    """One wording for "this needs Blender" wherever it is met.

    The Rig stage says it, ``settings_3d`` hides its section over it, and
    ``create_character`` raises the identical sentence at the door -- because a
    body whose skeleton will never exist is the half-built asset that door's
    ordering exists to prevent.
    """
    from warlock.studio.panes import stage_rig

    sentence = "Rigging needs Blender, which is not installed."
    assert sentence in inspect.getsource(stage_rig.draw)
    from warlock.service import characters as svc_characters

    assert sentence in inspect.getsource(svc_characters.create_character)

    ctx = _Ctx(svc, rigging=False)
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    problems = settings_character.problems(ctx, form)
    assert sentence in [str(p) for p in problems]
    # No field: it is a fact about the install, not about a control, so it goes
    # to the toast and the plan block rather than ringing an arbitrary widget.
    blender = next(p for p in problems if str(p) == sentence)
    assert getattr(blender, "field", "") == ""


# --- what is submitted --------------------------------------------------------


@pytest.mark.parametrize("species", ONE_PER_ARCHETYPE)
def test_recipe_kwargs_round_trips_through_the_door_it_is_built_for(ctx, species):
    """The compilation of the request, checked against the thing that refuses it.

    Every archetype, because the appearance block is the one part of a recipe
    whose *shape* comes off the body plan -- a fixed set of sliders would be
    accepted for an ogre and refused for a wolf.
    """
    opts = settings_character.options(ctx)
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    form["character_family"] = species
    form["character_theme"] = settings_character.THEME_UNSET
    form["character_body"] = "{}"

    kwargs = settings_character.recipe_kwargs(form, opts)
    recipe = Recipe.from_dict(kwargs)

    assert recipe.family == species
    assert recipe.cell_count == settings_character.cell_count(form)
    assert recipe.directions == settings_character.DIRECTIONS
    # The camera and its angle are one fact, taken from ``troupe_options``.
    preset = opts["troupe"]["camera_presets"][recipe.camera]
    assert recipe.elevation == float(preset["elevation"])
    # And the sliders are the species' own defaults when nobody moved one.
    assert set(recipe.appearance) == {c["key"] for c in opts["channels"][species]}


def test_the_sentinel_look_means_the_species_own_and_is_never_sent(ctx):
    """``"none"`` is not a theme key; sending it would be refused by name."""
    opts = settings_character.options(ctx)
    form = _form("a wolf")
    settings_character.sync_from_prompt(form)
    assert form["character_theme"] == settings_character.THEME_UNSET
    kwargs = settings_character.recipe_kwargs(form, opts)
    assert "theme" not in kwargs
    assert Recipe.from_dict(kwargs).theme == families()["wolf"].themes[0].key


def test_a_slider_from_another_body_plan_is_dropped_rather_than_submitted(ctx):
    """A restored form can carry one; the door refuses an unknown channel by
    name, and a refusal about a control that is no longer on screen is the dead
    end this pane's whole override model exists to avoid."""
    opts = settings_character.options(ctx)
    form = _form("a wolf")
    settings_character.sync_from_prompt(form)
    form["character_body"] = '{"not-a-channel": 0.5}'
    kwargs = settings_character.recipe_kwargs(form, opts)
    assert kwargs["appearance"] == {}
    Recipe.from_dict(kwargs)


def test_the_press_builds_a_character_and_never_reaches_create_job(ctx, monkeypatch):
    """``create_job``'s allowlist refuses ``asset_type="character"`` outright,
    which is what makes this structural rather than a convention -- but the
    pane must not get that far, because the refusal would arrive as a toast
    about a field the user never set."""
    from warlock.service import characters as svc_characters
    from warlock.service import jobs as svc_jobs

    def _never(*_a, **_k):
        raise AssertionError("the character arm reached create_job")

    seen: list[dict] = []

    def _build(_svc, recipe, **kwargs):
        seen.append({"recipe": dict(recipe), **kwargs})
        return {"id": "CHAR", "rig": "RIG", "kind": "character"}

    monkeypatch.setattr(svc_jobs, "create_job", _never)
    monkeypatch.setattr(svc_characters, "create_character", _build)

    form = _form("an attacking fire ogre")
    settings_character.sync_from_prompt(form)
    settings_2d.generate(ctx, form)

    assert [key for key, _result in ctx.submitted] == ["submit"]
    assert ctx.submitted[0][1] == {"id": "CHAR", "rig": "RIG", "kind": "character"}
    assert seen[0]["prompt"] == "an attacking fire ogre"
    assert seen[0]["recipe"]["family"] == "ogre"
    # The resolution rides along, so the finished row knows what was asked for.
    assert seen[0]["resolution"]["family"] == "ogre"


def test_the_toast_names_the_species_and_the_cell_count(ctx):
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    message = settings_character.toast_for(form, settings_character.options(ctx))
    assert "ogre" in message
    assert f"{settings_character.cell_count(form)}-cell sheet" in message
    assert "Troupe" in message


def test_a_preview_has_its_own_key_so_it_can_never_swallow_a_press():
    assert settings_character.PREVIEW_KEY != "submit"
    assert settings_character.PREVIEW_KEY == "character-preview"
    source = inspect.getsource(settings_character.preview)
    assert "PREVIEW_KEY" in source
    assert "preview_character" in source


# --- the plan ------------------------------------------------------------------


@pytest.mark.parametrize("species", ONE_PER_ARCHETYPE)
def test_the_plan_names_the_species_the_cells_and_no_gpu(species):
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    form["character_family"] = species

    plan = generation_workspace.plan_for(form)

    assert plan.generations == 0, "a character draws no images at all"
    assert families()[species].label.lower() in plan.stages
    assert f"{settings_character.cell_count(form)}-cell sheet" in plan.stages
    assert "no GPU needed" in plan.stages
    assert plan.duration.startswith("about ")
    assert "image model" in plan.recipe


def test_the_plan_footer_prints_no_count_line_when_there_is_nothing_to_count():
    """"1 candidate - 0 image generations" reads as a bug rather than a fact."""
    source = inspect.getsource(settings_2d._generation_plan)
    assert "if plan.generations > 0:" in source
    assert "widgets.muted(plan.duration)" in source


def test_the_default_trio_is_the_recipes_own_and_makes_the_stated_sheet():
    """Idle, walk and attack -- 18 frames over 8 directions, 144 cells. The
    number the tooltip, the plan and the toast all print, taken from the
    recipe's table rather than typed in three places."""
    form = _form("a fire ogre")
    assert settings_character.actions_of(form) == ("idle", "walk", "attack")
    assert settings_character.cell_count(form) == 144


# --- the ladders ---------------------------------------------------------------


def test_a_restored_value_off_the_ladder_is_named_rather_than_snapped(ctx):
    """Both are persisted and the ladders can move between releases, so the
    segmented control is not on its own a gate."""
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    form["character_pixel"] = "17"
    form["character_colors"] = "7"
    fields = {getattr(p, "field", "") for p in settings_character.problems(ctx, form)}
    assert {"character_pixel", "character_colors"} <= fields
    assert form["character_pixel"] == "17", "named, never quietly snapped"


def test_the_size_and_colour_ladders_are_the_doors_own(ctx):
    """Read from ``troupe_options``, which is what ``create_character`` puts the
    request through -- a second list in the pane is a form that accepts what the
    door then refuses."""
    from warlock.service import troupe as svc_troupe

    opts = settings_character.options(ctx)
    assert opts["troupe"]["logical_sizes"] == list(svc_troupe.TROUPE_LOGICAL_SIZES)
    assert opts["troupe"]["colors"] == list(svc_troupe.TROUPE_COLOR_CHOICES)


# --- the column, and what it is not --------------------------------------------


def test_the_character_column_draws_none_of_the_sdxl_recipe(ctx):
    """No checkpoint, no LoRA, no negative prompt, no history, no conditioning:
    a character runs no text encoder, so every one of those would be a control
    whose only outcome is that it does nothing."""
    source = inspect.getsource(settings_character)
    for absent in ("base_model", "style_lora", "negative_prompt", "ref_path", "ip_adapter"):
        assert absent not in source, absent
    # And the shared halves really are shared.
    block = inspect.getsource(settings_character.draw_block)
    assert "settings_2d._seed_row" in block
    draw = inspect.getsource(settings_2d.draw)
    assert "settings_character.draw_block" in draw
    assert "_reset_row(ctx)" in draw
    assert "_plan_footer(ctx, form)" in draw


def test_the_form_checks_that_do_not_apply_are_skipped_for_a_character(ctx):
    """The tileset precedent. A character reads no checkpoint and no LoRA, so a
    disabled Generate reading "Choose a recognised image model" over a run that
    opens none would be a refusal about somebody else's job."""
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    form["base_model"] = "not-a-model"
    form["style_lora"] = "not-a-lora"
    form["control"] = "canny"
    assert settings_2d.validate(form) == []
    assert settings_2d.weights_problem(ctx, form) is None


def test_the_family_picker_is_a_real_picker_grouped_by_body_plan(ctx):
    """``_locked_sheet_recipe``'s rule is that a choice with one answer is drawn
    as a statement. Thirty-one species across four body plans is the opposite
    situation, so this is a combo -- and grouped, because a flat alphabetical
    list of thirty-one nouns is a list nobody can find a wolf in."""
    opts = settings_character.options(ctx)
    entries = settings_character.family_options(opts, "ogre")
    keys = [key for key, _label in entries]
    assert set(keys) == set(families())
    assert len(keys) == len(families()) >= 31

    labels = {row["key"]: row["label"] for row in opts["archetypes"]}
    order = [
        next(f["archetype"] for f in opts["families"] if f["key"] == key) for key in keys
    ]
    assert order == sorted(order, key=list(labels).index), "grouped by body plan"
    for key, label in entries:
        archetype = next(f["archetype"] for f in opts["families"] if f["key"] == key)
        assert label.startswith(labels[archetype] + ": ")


def test_an_empty_species_is_listed_rather_than_falling_back_to_entry_zero(ctx):
    """``controls.combo`` shows entry zero for a value it cannot find, so
    without this the picker would draw "Human" over a form that is refusing to
    submit -- the value keeping Generate off would be the one thing not on
    screen."""
    entries = settings_character.family_options(settings_character.options(ctx), "")
    assert entries[0][0] == ""
    # And a stored key from another build is marked, never dropped.
    marked = settings_character.family_options(settings_character.options(ctx), "gorgon")
    assert ("gorgon", "gorgon - not a species this build ships") in marked


@pytest.mark.parametrize("species", ONE_PER_ARCHETYPE)
def test_there_is_one_slider_per_channel_the_body_plan_declares(ctx, species):
    """Never a fixed column: a wolf has none of an ogre's channels, and a form
    that drew a fixed group would offer four controls of which three are
    refusals."""
    opts = settings_character.options(ctx)
    form = _form("")
    form["character_family"] = species
    channels = settings_character.channels_of(form, opts)
    assert channels
    assert {c["key"] for c in channels} == set(families()[species].appearance_defaults())


# --- it actually draws ---------------------------------------------------------
#
# Everything above is pure. The block is not, and the failures a pure test
# cannot see are the ones that matter most in this file: an unbalanced group, a
# ``same_line`` past the pane edge, a binding that rejects its arguments. So the
# whole block is drawn once, in a real imgui context with no GL, and the control
# census is read back -- ``troupe_preview``'s scorecard test is the precedent,
# and its bug (the centre pane dying on a ring nobody had ever drawn in a test)
# is exactly this shape.


@pytest.fixture
def ui(monkeypatch):
    from _ui_context import imgui_context

    with imgui_context(monkeypatch) as imgui:
        yield imgui


def _draw_block(ui, ctx, form):
    from warlock.studio import forms, probe, widgets

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        with widgets.section_blocks(), forms.Form(
            "create-2d", errors=ctx.state.field_errors
        ) as form_ui:
            settings_character.draw_block(ctx, form, form_ui)
    finally:
        ui.end()
        ui.end_frame()
    return probe.census()


@pytest.mark.parametrize("species", ONE_PER_ARCHETYPE)
def test_the_whole_block_draws_for_every_body_plan(ui, ctx, species):
    """One pass per archetype, because the slider column is the part whose
    *shape* comes off the body plan -- a block that draws for an ogre and
    raises for a wolf is exactly the failure this cannot be a pure test about.
    """
    form = _form("a fire ogre")
    settings_character.sync_from_prompt(form)
    form["character_family"] = species
    form["character_theme"] = settings_character.THEME_UNSET

    seen = _draw_block(ui, ctx, form)

    labels = [c.label for c in seen]
    assert any("character_family" in label for label in labels), labels
    assert any("character_pixel" in label for label in labels), labels
    # One switch per movement, one slider per channel.
    switches = [c for c in seen if c.kind == "switch"]
    assert len(switches) >= len(settings_character.MOVEMENTS)
    channels = settings_character.channels_of(form, settings_character.options(ctx))
    sliders = [c for c in seen if "character_body_" in c.label]
    assert len(sliders) == len(channels), sorted({c.kind for c in seen})


def test_the_block_draws_with_no_species_and_offers_no_slider_column(ui, ctx):
    """The refused state is the one a first-time user sees, and it has to draw.

    No species means no channels and no look to choose between, so both of
    those disappear rather than being greyed -- a control with nothing to act
    on is a control that cannot do anything.
    """
    form = _form("a manticore")
    seen = _draw_block(ui, ctx, form)
    assert seen, "the block still draws"
    assert not [c for c in seen if c.kind == "slider"]
    labels = [c.label for c in seen]
    assert not any("character_theme" in label for label in labels), labels


def test_the_refusal_and_its_three_repairs_draw_under_the_plan(ui, ctx):
    """``_preflight_fix`` is where the substitution is offered, and it is the
    one path in this file that is a *press*. It has to draw."""
    from warlock.studio import probe, widgets

    form = _form("a manticore")
    problems = settings_2d.problems_for(ctx, form)
    problem = next(p for p in problems if getattr(p, "field", "") == "prompt")

    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        with widgets.section_blocks():
            settings_2d._preflight_fix(ctx, form, problem)
    finally:
        ui.end()
        ui.end_frame()
    labels = [c.label for c in probe.census()]
    assert any("character-offer" in label for label in labels), labels
    assert any("character-sprite" in label for label in labels), labels
    assert any("character-troupe" in label for label in labels), labels
