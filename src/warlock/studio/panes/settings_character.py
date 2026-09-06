"""The Character type's own column in Create. **Not part of ``settings_2d``.**

``settings_2d`` is "everything that composes the SDXL prompt, and Generate", and
a character composes none of it: no checkpoint, no LoRA, no negative prompt, no
conditioning image, no history of prompts that were sent to a text encoder. Put
here rather than as a sixth branch inside that 2600-line module because the two
have nothing in common except the column they are drawn in and the button they
are refused by -- ``_reset_row`` and ``_plan_footer`` stay shared, and
``settings_2d.draw`` calls :func:`draw_block` instead of the Recipe section.

**The prompt fills the form; the form is never the prompt's prisoner.** Typing
in the command bar re-resolves the brief and writes every field the user has not
touched (:func:`sync_from_prompt`). Touching a control records it in
``character_overrides``, and from then on that control is the user's -- a prompt
edit stops writing it. "Reset to prompt" empties the list. Without the override
list the two directions fight: either the prompt cannot fill anything after the
first frame, or every keystroke silently undoes a species the user just picked.

**Nothing here ever substitutes a species.** ``characters.resolve`` refuses to,
by construction (``Resolution.family`` is ``None`` for a creature we do not
make), and this pane keeps that promise at the other end: ``character_family``
stays ``""``, Generate is refused in :func:`resolve.offer_sentence`'s exact
words, and the substitution is offered as a *button the user presses* -- see
:func:`preflight_fix`. A form that quietly filled in "wyvern" for the word
"dragon" would produce a character nobody asked for, which is the whole failure
that sentence exists to prevent.
"""

from __future__ import annotations

import json
from typing import Any

from imgui_bundle import imgui

from ...characters import recipe as recipe_mod
from ...characters import resolve as resolve_mod
from ...service import characters as svc_characters
from .. import controls, forms, widgets
from ..manual import render as manual_render

#: ``character_theme``'s "the species' own look" value. Not a theme key: no
#: species declares a ``none`` theme, which is what makes it safe as a sentinel
#: and what makes switching species keep the choice honest -- a fire palette
#: carried onto a creature with no fire would be a look nobody chose.
THEME_UNSET = "none"

#: The movements a character sheet carries, and how many frames each is drawn
#: in. ``recipe.DEFAULT_ANIMATIONS`` rather than the five-row legacy
#: ``charsheet.ANIMATIONS`` table, and the recipe module says why: these three
#: are what a character needs to read as alive in a top-down game. Read from
#: there rather than restated, so the "144 cells" this pane prints and the cells
#: the door plans are one arithmetic.
MOVEMENTS: tuple[tuple[str, int], ...] = tuple(recipe_mod.DEFAULT_ANIMATIONS.items())

#: How many ways each movement is drawn. The recipe's own default, and stated on
#: the camera helper rather than offered as a control: eight is what every
#: preset in ``charsheet.CAMERA_PRESETS`` is laid out for, and a fourth combo
#: for a value with one sensible answer is a control that cannot be operated.
DIRECTIONS = 8

#: Where :func:`options` caches the door's answer for the life of the process.
#: ``troupe_settings._options``' slot and its reason: ``character_options``
#: walks the palette directory, and a directory walk sixty times a second is a
#: cost with no reader.
OPTIONS_SLOT = "character_options"

#: Which recipe field each control answers to. The door refuses in the
#: *recipe's* vocabulary (``logical_size``, ``appearance``, ``animations``)
#: because that is what ``Recipe.from_dict`` validates, and the controls are
#: named for the form keys they persist under -- so without this map a refusal
#: about a colour count would be recorded against a name nothing on the pane
#: draws, and the ring would land nowhere. :func:`mirror_errors` is what walks
#: it; ``tests/test_field_error_wiring.py`` is the standing guard on the drift.
RECIPE_FIELDS: dict[str, tuple[str, ...]] = {
    "character_family": ("family", "family_version"),
    "character_theme": ("theme",),
    "character_camera": ("camera", "elevation"),
    "character_actions": ("animations", "directions"),
    "character_pixel": ("logical_size",),
    "character_colors": ("colors",),
    "character_body": ("appearance",),
    "character_name": ("name",),
}

#: The one task key a preview runs under. Its own rather than ``"submit"``, and
#: that is the whole point: a preview is a look at the form and must never make
#: the Generate button busy -- ``TaskRunner.submit`` refuses a key already in
#: flight, so sharing the key would mean a dragged slider could swallow a press.
PREVIEW_KEY = "character-preview"

#: Where the toast for an accepted press waits until the door answers.
#: Composed at submit time rather than at landing time because that is where
#: the form and the registry are both to hand -- ``create_character`` returns
#: two ids and a kind, which is all a door should have to know about a sentence.
TOAST_SLOT = "character_toast"


# --- what the form is asking for ---------------------------------------------


def options(ctx: Any) -> dict[str, Any]:
    """``service.characters.character_options``, read once per process."""
    cached = ctx.state.preview.get(OPTIONS_SLOT)
    if cached is None:
        cached = svc_characters.character_options(ctx.svc)
        ctx.state.preview[OPTIONS_SLOT] = cached
    return cached


def resolution_of(form: dict[str, Any]) -> resolve_mod.Resolution:
    """The cached resolution of the brief, or an empty one.

    Never re-resolves: :func:`sync_from_prompt` owns that, and a scan run from
    a getter would run it per control per frame.
    """
    raw = str(form.get("character_resolution") or "")
    if not raw:
        return resolve_mod.Resolution()
    try:
        return resolve_mod.Resolution.from_dict(json.loads(raw))
    except (ValueError, TypeError, AttributeError):
        # A hand-edited settings file. An empty resolution is the honest
        # answer -- the pane then says the prompt named no species, which is
        # true of a resolution nobody can read.
        return resolve_mod.Resolution()


def overrides_of(form: dict[str, Any]) -> list[str]:
    raw = form.get("character_overrides")
    return [str(v) for v in raw] if isinstance(raw, list) else []


def touched(form: dict[str, Any], key: str) -> None:
    """Record that the user set this control themselves.

    From here on a prompt edit leaves it alone. Recorded on *every* change
    rather than on a change away from the resolved value, because "I typed the
    species the prompt already said" is still a decision, and a form that
    forgot it would move that field the next time the prompt changed.
    """
    overrides = overrides_of(form)
    if key not in overrides:
        overrides.append(key)
    form["character_overrides"] = overrides


def sync_from_prompt(form: dict[str, Any]) -> bool:
    """Re-resolve the brief and fill what the user has not touched. -> did it.

    Called at the top of :func:`draw_block` *and* at the top of
    ``settings_2d.generate``'s character arm, which are the two doors: the
    keyboard ones (Ctrl+Enter, the command palette) never draw this pane, so a
    form filled only from the draw would submit the species of the *previous*
    prompt for anyone who typed and pressed Ctrl+Enter in one motion.

    Cheap on the common frame: the scan runs only when the prompt differs from
    the one the cached resolution was made from.
    """
    prompt = str(form.get("prompt") or "")
    if form.get("character_resolution") and prompt == str(
        form.get("character_resolution_prompt") or ""
    ):
        return False
    resolution = resolve_mod.resolve(prompt)
    form["character_resolution"] = json.dumps(resolution.to_dict())
    form["character_resolution_prompt"] = prompt
    _fill(form, resolution)
    return True


def reset_to_prompt(form: dict[str, Any]) -> None:
    """Forget every override and take the brief's answer again."""
    form["character_overrides"] = []
    form["character_resolution"] = ""
    form["character_resolution_prompt"] = ""
    sync_from_prompt(form)


def _fill(form: dict[str, Any], resolution: resolve_mod.Resolution) -> None:
    """Write the resolved brief into the fields the user has not claimed.

    **A field the prompt says nothing about goes back to its default**, not to
    whatever the last prompt left in it. "a wolf" after "an attacking fire ogre"
    has to produce a wolf with the default actions and no fire, or the form
    accumulates a character out of two briefs the user never wrote together.
    """
    from ..state import default_form_2d

    overrides = set(overrides_of(form))
    defaults = default_form_2d()
    actions = tuple(a for a in resolution.actions if a in dict(MOVEMENTS))
    values = {
        # Never a substitution: ``resolution.family`` is None for a creature
        # this program does not make, and "" is what that means here.
        "character_family": resolution.family or "",
        "character_theme": resolution.theme or THEME_UNSET,
        "character_camera": resolution.camera_preset or "",
        "character_actions": (
            ",".join(actions) if actions else str(defaults["character_actions"])
        ),
    }
    for key, value in values.items():
        if key not in overrides:
            form[key] = value
    # The sliders belong to the species, so a change of species drops them --
    # a quadruped has no ``tusk`` channel, and an appearance block carrying one
    # is a request ``Recipe.from_dict`` refuses by name.
    if "character_body" not in overrides:
        form["character_body"] = "{}"


# --- reading the form ---------------------------------------------------------


def actions_of(form: dict[str, Any]) -> tuple[str, ...]:
    """The movements this sheet carries, in :data:`MOVEMENTS` order.

    The stored order is not trusted: two forms that named the same movements in
    different orders would otherwise plan two different cell layouts, which is
    ``resolve._ACTION_ORDER``'s rule applied at the other end of the same trip.
    """
    stored = {
        part.strip()
        for part in str(form.get("character_actions") or "").split(",")
        if part.strip()
    }
    return tuple(name for name, _frames in MOVEMENTS if name in stored)


def animations_of(form: dict[str, Any]) -> dict[str, int]:
    frames = dict(MOVEMENTS)
    return {name: frames[name] for name in actions_of(form)}


def cell_count(form: dict[str, Any]) -> int:
    return sum(animations_of(form).values()) * DIRECTIONS


def family_of(form: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any] | None:
    """The chosen species' row, or None when the form names none we ship."""
    key = str(form.get("character_family") or "")
    return next((row for row in opts["families"] if row["key"] == key), None)


def body_of(form: dict[str, Any], opts: dict[str, Any]) -> dict[str, float]:
    """The appearance block, filtered to the chosen species' own channels.

    Filtered on the way *out* rather than cleared on the way in, so a form
    restored with a slider from another body plan is simply not sent -- the
    door refuses an unknown channel by name, and a request refused over a
    control that is no longer on screen is the dead end the pane's whole
    override model exists to avoid.
    """
    try:
        stored = json.loads(str(form.get("character_body") or "{}"))
    except (ValueError, TypeError):
        stored = {}
    if not isinstance(stored, dict):
        return {}
    channels = {c["key"] for c in channels_of(form, opts)}
    out: dict[str, float] = {}
    for key, value in stored.items():
        if str(key) in channels:
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def channels_of(form: dict[str, Any], opts: dict[str, Any]) -> list[dict[str, Any]]:
    """The sliders the chosen species declares, with *its* defaults on them."""
    key = str(form.get("character_family") or "")
    return list(opts["channels"].get(key) or ())


def set_channel(form: dict[str, Any], key: str, value: float) -> None:
    body = {}
    try:
        raw = json.loads(str(form.get("character_body") or "{}"))
        if isinstance(raw, dict):
            body = {str(k): v for k, v in raw.items()}
    except (ValueError, TypeError):
        body = {}
    body[key] = float(value)
    form["character_body"] = json.dumps(body, sort_keys=True)


def camera_of(form: dict[str, Any], opts: dict[str, Any]) -> str:
    """The preset this form will be rendered at. Empty means the door's own."""
    presets = opts["troupe"]["camera_presets"]
    chosen = str(form.get("character_camera") or "")
    if chosen in presets:
        return chosen
    return str(opts["troupe"]["defaults"]["camera"])


# --- the options each picker offers -------------------------------------------


def family_options(opts: dict[str, Any], current: str) -> tuple[tuple[str, str], ...]:
    """Every species, grouped by body plan. **A real picker, not a label.**

    ``_locked_sheet_recipe``'s rule is that a choice with one answer is drawn as
    a statement rather than as a combo nobody can operate. Thirty-one species
    across four body plans is the opposite situation, so this is a combo -- and
    it is grouped, because a flat alphabetical list of thirty-one nouns is a
    list nobody can find a wolf in. The archetype is carried in the label rather
    than as a header row: ``controls.combo`` draws one selectable per entry and
    a header would be a row that answers a click by doing nothing.

    A form naming no species at all keeps a first entry saying so -- the combo
    falls back to entry zero for a value it cannot find, so without it the
    picker would draw "Human" over a form that is refusing to submit.
    """
    labels = {row["key"]: row["label"] for row in opts["archetypes"]}
    out: list[tuple[str, str]] = []
    if not current:
        out.append(("", "Not chosen yet"))
    for archetype in labels:
        for row in opts["families"]:
            if row["archetype"] == archetype:
                out.append((row["key"], f"{labels[archetype]}: {row['label']}"))
    if current and current not in {key for key, _label in out}:
        # ``palette_options``' rule: a stored value the menu does not carry is
        # listed and marked rather than dropped, because dropping it makes the
        # one thing keeping Generate off the one thing not on screen.
        out.append((current, f"{current} - not a species this build ships"))
    return tuple(out)


def theme_options(opts: dict[str, Any], family: str) -> tuple[tuple[str, str], ...]:
    """The looks this species paints, plus "its own"."""
    row = next((f for f in opts["families"] if f["key"] == family), None)
    themes = tuple((t["key"], t["label"]) for t in (row or {}).get("themes", ()))
    return ((THEME_UNSET, "The species' own"), *themes)


def camera_options(opts: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    presets = opts["troupe"]["camera_presets"]
    return tuple((key, str(entry["label"])) for key, entry in presets.items())


def camera_helper(opts: dict[str, Any], camera: str) -> str:
    """The angle and the direction count, in the numbers that transfer.

    ``troupe_settings._camera_helper``'s argument: a user matching these sprites
    to a Plotter map knows what elevation that map is drawn at, and a preset's
    name does not answer that while its angle does.
    """
    entry = opts["troupe"]["camera_presets"].get(camera) or {}
    if "elevation" not in entry:
        return ""
    return f"{float(entry['elevation']):g} degrees elevation, {DIRECTIONS} directions"


# --- what would be refused ----------------------------------------------------


def problems(ctx: Any, form: dict[str, Any]) -> list[widgets.Problem]:
    """Everything stopping a character press, each naming its own control.

    Appended by ``settings_2d.problems_for``, which is what makes the ring, the
    plan footer, the disabled Generate's tooltip and the Ctrl+Enter toast one
    sentence rather than four -- the property that module's cache exists for.
    """
    # Before anything is judged, because the *bar* asks this question first.
    # ``main._build_ui`` draws the command bar above this column, so a sync that
    # happened only in ``draw_block`` would leave the frame's cached verdict --
    # the ring, the disabled Generate and the footer all read it -- describing
    # the previous prompt while the block below showed the new species. Cheap:
    # the scan runs only when the prompt has actually changed.
    sync_from_prompt(form)
    opts = options(ctx)
    out: list[widgets.Problem] = []
    resolution = resolution_of(form)
    family = str(form.get("character_family") or "")
    if not family:
        out.append(widgets.Problem(_no_species(resolution, opts), "prompt"))
    elif family_of(form, opts) is None:
        out.append(
            widgets.Problem(
                f"{family!r} is not a species this build ships. Pick one from "
                f"Species.",
                "character_family",
            )
        )
    if not actions_of(form):
        out.append(
            widgets.Problem(
                "A character sheet is at least one movement. Turn on idle, "
                "walk or attack.",
                "character_actions",
            )
        )
    if not getattr(ctx, "rigging_available", False):
        # The Rig segment's own sentence, verbatim -- and ``create_character``
        # raises the identical one at the door. One wording for "this needs
        # Blender" wherever it is met.
        #
        # No ``field``, deliberately: this is a fact about the *install*, not
        # about a control, and ``note_field_error`` refuses an empty address
        # precisely so a machine-shaped refusal reaches the toast and the plan
        # block instead of ringing an arbitrary widget.
        out.append(widgets.Problem("Rigging needs Blender, which is not installed."))
    # Reachable from a *restored* form rather than from this frame's controls:
    # both values are persisted and the ladders can move between releases, so a
    # segmented control offering three sizes is not on its own a gate.
    sizes = list(opts["troupe"]["logical_sizes"])
    if _int(form.get("character_pixel"), -1) not in sizes:
        out.append(widgets.Problem(f"Sprite size must be one of {sizes}.", "character_pixel"))
    colors = list(opts["troupe"]["colors"])
    if _int(form.get("character_colors"), -1) not in colors:
        out.append(widgets.Problem(f"Colours must be one of {colors}.", "character_colors"))
    return out


def _no_species(resolution: resolve_mod.Resolution, opts: dict[str, Any]) -> str:
    """Why this brief names nothing we can build. **One home for the wording.**

    The creature case is :func:`resolve.offer_sentence` verbatim and is never
    re-worded here: that function exists precisely because the Create surface, a
    tooltip and a toast all say it, and a second copy is the one that eventually
    gets written as though the substitution had already happened.
    """
    offer = resolve_mod.offer_sentence(resolution)
    if offer is not None:
        return offer
    count = len(opts["families"])
    if resolution.creature_words:
        # A creature word we know and have nothing at all to offer for. Not
        # reachable from today's registry -- every body plan ships species --
        # and still said in the same register rather than left to fall through
        # to the "names none of them" sentence, which would be untrue.
        return (
            f"Warlock has no {resolution.creature_words[0]} yet, and nothing "
            f"close enough to offer. Pick one of its {count} species from "
            f"Species."
        )
    return (
        f"Warlock builds {count} species across four body plans, and this brief "
        f"names none of them. Say what to make, or pick one from Species."
    )


# --- what is submitted --------------------------------------------------------


def recipe_kwargs(form: dict[str, Any], opts: dict[str, Any]) -> dict[str, Any]:
    """The form as ``characters.recipe.Recipe.from_dict`` takes it.

    ``settings_2d.submit_kwargs``' opposite number on this arm, and split out
    for that function's reason: the compilation of the request is the one part
    of a press no test can reach while it lives inside a closure.

    ``elevation`` is sent as the *number* rather than left to the recipe's
    default, which is ``troupe_settings``' arrangement and its argument: a
    preset is only a name for an angle, and a recipe carrying a camera whose
    elevation is somebody else's default would be framed at an angle nobody
    picked. Both come from ``troupe_options``, so the pane holds no second copy
    of the table (``tests/troupe/test_camera_presets.py``).
    """
    camera = camera_of(form, opts)
    presets = opts["troupe"]["camera_presets"]
    kwargs: dict[str, Any] = {
        "family": str(form.get("character_family") or ""),
        "camera": camera,
        "elevation": float(presets[camera]["elevation"]),
        "animations": animations_of(form),
        "directions": DIRECTIONS,
        "logical_size": _int(form.get("character_pixel"), 64),
        "colors": _int(form.get("character_colors"), 32),
        "appearance": body_of(form, opts),
        "seed": max(0, _int(form.get("seed"), 0)),
    }
    theme = str(form.get("character_theme") or THEME_UNSET)
    if theme != THEME_UNSET:
        # Omitted rather than sent as the sentinel: absent is what
        # ``Recipe.from_dict`` reads as "this species' own first look", and a
        # literal "none" is a theme key it would refuse by name.
        kwargs["theme"] = theme
    name = str(form.get("character_name") or "").strip()
    if name:
        kwargs["name"] = name
    return kwargs


def submit(ctx: Any, form: dict[str, Any]) -> bool:
    """Build the character. -> whether the press was taken.

    On the shared ``"submit"`` key, the same one every other Create output
    uses: it is one form and one Generate button, so two submits in flight from
    it is exactly what that key exists to prevent.
    """
    opts = options(ctx)
    kwargs = recipe_kwargs(form, opts)
    prompt = str(form.get("prompt") or "")
    resolution = resolution_of(form).to_dict()
    name = str(form.get("character_name") or "").strip() or None

    def run():
        return svc_characters.create_character(
            ctx.svc, kwargs, name=name, prompt=prompt, resolution=resolution
        )

    from . import settings_2d

    taken = settings_2d.submit_job(ctx, run)
    if taken:
        ctx.state.preview[TOAST_SLOT] = toast_for(form, opts)
    return taken


def toast_for(form: dict[str, Any], opts: dict[str, Any]) -> str:
    """What the app says once the press lands. Named species, counted cells."""
    row = family_of(form, opts)
    label = str((row or {}).get("label") or "character").lower()
    return (
        f"Building the {label}: mesh -> rig -> {cell_count(form)}-cell sheet. "
        f"Watch it here, then in Troupe."
    )


# --- the block ----------------------------------------------------------------


def mirror_errors(ctx: Any) -> None:
    """Re-file a door refusal under the name of the control it is about.

    ``Recipe.from_dict`` refuses in the recipe's vocabulary -- ``logical_size``,
    ``appearance``, ``animations`` -- because that is what it validates, and
    ``service.errors.invalid_from`` passes that address straight through to
    ``main._collect_tasks``. The controls here are named for the form keys they
    persist under, so without this the address is recorded and thrown away: the
    ring lands nowhere and the user gets a red toast in the corner with no idea
    which control was at fault. That is the exact defect
    ``tests/test_field_error_wiring.py`` was written for.

    Run from ``settings_2d.draw`` **before** the ``forms.Form`` is built,
    because that class snapshots the error map at construction -- a mirror
    written afterwards would not be seen until the next frame.
    """
    errors = getattr(ctx.state, "field_errors", None)
    if not errors:
        return
    for control, aliases in RECIPE_FIELDS.items():
        if control in errors:
            continue
        for alias in aliases:
            message = str(errors.get(alias) or "")
            if message:
                errors[control] = message
                break


def draw_block(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """The whole Character column, inside ``settings_2d``'s form and child."""
    from . import settings_2d

    sync_from_prompt(form)
    opts = options(ctx)
    widgets.section("Character")
    manual_render.help_button(ctx, "settings-character")
    _family(ctx, form, form_ui, opts)
    _theme(ctx, form, form_ui, opts)
    _camera(ctx, form, form_ui, opts)
    _actions(ctx, form, form_ui)
    _pixels(ctx, form, form_ui, opts)
    _appearance(ctx, form, form_ui, opts)
    settings_2d._seed_row(ctx, form, form_ui)
    _name(ctx, form, form_ui)
    _unrecognised(form)
    _footer(ctx, form)


def _family(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    current = str(form.get("character_family") or "")
    changed, picked = form_ui.combo(
        "character_family",
        "Species",
        current,
        family_options(opts, current),
        help_text=(
            "What to build. Grouped by body plan, because the plan decides the "
            "skeleton, the clips and which sliders this character has."
        ),
    )
    if changed and picked != current:
        form["character_family"] = picked
        touched(form, "character_family")
        # The sliders and the look belong to the species that had them.
        form["character_body"] = "{}"
        if not _theme_offered(opts, picked, str(form.get("character_theme") or "")):
            form["character_theme"] = THEME_UNSET
        ctx.state.clear_field_error("character_family")


def _theme_offered(opts: dict[str, Any], family: str, theme: str) -> bool:
    return theme in {key for key, _label in theme_options(opts, family)}


def _theme(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    family = str(form.get("character_family") or "")
    choices = theme_options(opts, family)
    if len(choices) < 2:
        # No species chosen, or one that paints a single look: a combo with one
        # entry is a control that answers every click with the answer it had.
        return
    current = str(form.get("character_theme") or THEME_UNSET)
    changed, picked = form_ui.combo(
        "character_theme",
        "Look",
        current,
        choices,
        help_text="The palette this species is painted in.",
    )
    if changed:
        form["character_theme"] = picked
        touched(form, "character_theme")
        ctx.state.clear_field_error("character_theme")


def _camera(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    current = camera_of(form, opts)
    changed, picked = form_ui.combo(
        "character_camera",
        "Camera",
        current,
        camera_options(opts),
        help_text="Where the eye is while the frames are rendered.",
        helper=camera_helper(opts, current),
    )
    if changed:
        form["character_camera"] = picked
        touched(form, "character_camera")
        ctx.state.clear_field_error("character_camera")


def _actions(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """One switch per movement, each carrying the frames it costs."""
    form_ui.note("character_actions")
    live = set(actions_of(form))
    for name, frames in MOVEMENTS:
        changed, on = form_ui.switch(
            f"character_action_{name}",
            name.title(),
            name in live,
            help_text=f"{frames} frames, drawn {DIRECTIONS} ways.",
            helper=f"{frames} frames x {DIRECTIONS} = {frames * DIRECTIONS} cells",
        )
        if changed:
            if on:
                live.add(name)
            else:
                live.discard(name)
            form["character_actions"] = ",".join(
                key for key, _f in MOVEMENTS if key in live
            )
            touched(form, "character_actions")
            ctx.state.clear_field_error("character_actions")
            live = set(actions_of(form))
    widgets.muted(f"{cell_count(form)} cells")


def _pixels(ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]) -> None:
    """The two ladders the door enforces, read from the door.

    Two calls rather than a loop over a table of two, deliberately: the field id
    is the *address* a refusal is rung on, and
    ``tests/test_field_error_wiring.py`` reads those ids out of this file's
    source. An id built from a loop variable is an address no scan can see, and
    an address no scan can see is how ``retarget_panel`` came to draw a control
    a refusal could never reach.
    """
    changed, picked = form_ui.segmented_choice(
        "character_pixel",
        "Sprite size",
        str(form.get("character_pixel") or ""),
        tuple((str(v), f"{v} px") for v in opts["troupe"]["logical_sizes"]),
        compact=True,
        help_text="How many pixels across one rendered frame is.",
    )
    if changed:
        form["character_pixel"] = picked
        touched(form, "character_pixel")
        ctx.state.clear_field_error("character_pixel")
    changed, picked = form_ui.segmented_choice(
        "character_colors",
        "Colours",
        str(form.get("character_colors") or ""),
        tuple((str(v), str(v)) for v in opts["troupe"]["colors"]),
        compact=True,
        help_text="How many colours the finished sheet is reduced to.",
    )
    if changed:
        form["character_colors"] = picked
        touched(form, "character_colors")
        ctx.state.clear_field_error("character_colors")


def _appearance(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, opts: dict[str, Any]
) -> None:
    """One slider per channel the *species' archetype* declares.

    Never a fixed column of sliders: the channel set belongs to the body plan,
    so a wolf has none of an ogre's and a form that drew a fixed group would
    offer four controls of which three are refusals.
    """
    channels = channels_of(form, opts)
    if not channels:
        return
    # Above the sliders rather than rung onto one of them: the door refuses the
    # ``appearance`` *block*, and a ring on an arbitrary channel would point at
    # the wrong control. ``troupe_settings``' handling of ``layout``, which is
    # the same shape of address.
    form_ui.note("character_body")
    body = body_of(form, opts)
    for channel in channels:
        key = str(channel["key"])
        changed, value = form_ui.slider(
            f"character_body_{key}",
            str(channel["label"]),
            float(body.get(key, channel["default"])),
            float(channel["lo"]),
            float(channel["hi"]),
            fmt="%.2f",
        )
        if changed:
            set_channel(form, key, value)
            touched(form, "character_body")
            ctx.state.clear_field_error("character_body")


def _name(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    changed, text = form_ui.text(
        "character_name",
        "Name",
        str(form.get("character_name") or ""),
        hint="optional",
        max_length=recipe_mod.MAX_NAME,
        helper="Shown in the library. The species' name is used when this is blank.",
    )
    if changed:
        form["character_name"] = text
        touched(form, "character_name")
        ctx.state.clear_field_error("character_name")


def _unrecognised(form: dict[str, Any]) -> None:
    """What the brief said that this form did nothing with.

    Said out loud rather than dropped: a user who typed "a fire ogre with a
    greataxe" is owed the fact that the axe was not understood, or they will
    look for it on the sheet.
    """
    words = resolution_of(form).unrecognised
    if words:
        widgets.muted_wrapped("Not interpreted: " + ", ".join(words))


def _footer(ctx: Any, form: dict[str, Any]) -> None:
    """The two ghost buttons under the block, on one row when both are drawn.

    ``same_line`` is placed by whoever drew first rather than at the top of the
    second control, because a bare continuation on a frame where the first was
    skipped attaches the button to whatever the block happened to end with --
    the bug that once orphaned "Recent prompts" against a strength slider.
    """
    if overrides_of(form) and controls.button(
        "Reset to prompt##character-reset", role=controls.ButtonRole.GHOST
    ):
        reset_to_prompt(form)
        ctx.state.clear_field_errors()
    # Read *after* the press, not before: Reset empties the list, so the row it
    # was on has one button left on the frame it is pressed and the
    # continuation must not reach for a control that is no longer there.
    drew = bool(overrides_of(form))
    if not str(form.get("character_family") or ""):
        return
    if drew:
        imgui.same_line()
    # **Never gated on the form's problems**, and it has its own task key: a
    # preview is a look at the body, which is exactly what a user with a
    # refused brief wants while deciding what to change -- and a preview that
    # shared ``"submit"`` would let a dragged slider swallow a press.
    busy = ctx.busy(PREVIEW_KEY)
    if widgets.disabled_button(
        "Preview character##character-preview",
        not busy,
        reason="Still building the last preview.",
    ):
        preview(ctx, form)


def preview(ctx: Any, form: dict[str, Any]) -> bool:
    opts = options(ctx)
    kwargs = recipe_kwargs(form, opts)
    return bool(
        ctx.submit(PREVIEW_KEY, svc_characters.preview_character, ctx.svc, kwargs)
    )


# --- the repairs offered under a refusal --------------------------------------


def preflight_fix(ctx: Any, form: dict[str, Any], problem: widgets.Problem) -> bool:
    """The one-click ways out of a character refusal. -> whether it drew any.

    Called first from ``settings_2d._preflight_fix``, which owns the SDXL arm's
    repairs and knows nothing about a species.
    """
    field = getattr(problem, "field", "")
    message = str(problem)
    if field == "prompt" and message.startswith("Warlock has no "):
        _offer_fixes(ctx, form)
        return True
    if "needs Blender" in message:
        if controls.button(
            "Open dependency packs##character-blender", role=controls.ButtonRole.GHOST
        ):
            from ..state import set_mode
            from . import app_settings

            ctx.state.preview[app_settings.CATEGORY_SLOT] = "packs"
            set_mode(ctx.state, "settings")
        return True
    return False


def _offer_fixes(ctx: Any, form: dict[str, Any]) -> None:
    """Three real ways forward from "we do not make that", and no fourth.

    **The substitution is one of them, and it is a press.** Applying the offer
    is the *only* place in this program where a species the user did not name
    becomes the species that is built, and it happens because somebody read the
    sentence and clicked the button that repeats it. Nothing on any automatic
    path may do this -- see the module docstring.

    The other two keep the brief and change the deliverable, which is why both
    leave ``form["prompt"]`` alone: a user who came here for a manticore still
    wants a manticore, and the question is only which surface can draw one.
    """
    opts = options(ctx)
    resolution = resolution_of(form)
    offer = resolution.offer[0] if resolution.offer else ""
    row = next((f for f in opts["families"] if f["key"] == offer), None)
    if row is not None and controls.button(
        f"Make it a {row['label'].lower()}##character-offer",
        role=controls.ButtonRole.GHOST,
    ):
        apply_offer(form, opts)
        ctx.state.clear_field_error("prompt")
    if controls.button(
        "Sprite sheet (experimental)##character-sprite", role=controls.ButtonRole.GHOST
    ):
        switch_to_sprite_sheet(form)
        ctx.state.clear_field_errors()
    if controls.button("Draw it in Troupe##character-troupe", role=controls.ButtonRole.GHOST):
        hand_to_troupe(ctx, form)


def apply_offer(form: dict[str, Any], opts: dict[str, Any]) -> str:
    """Take the offered species. -> the key applied, or ``""``.

    **The only substitution in the program**, and it is here rather than in
    ``sync_from_prompt`` or in ``resolve`` because it is a thing a person does:
    they read "Warlock has no phoenix yet; the closest it makes is a dragon"
    and pressed the button that repeats it. Recorded as an override for the
    same reason -- it is the user's choice now, and the next prompt edit must
    not quietly take it back.
    """
    resolution = resolution_of(form)
    offer = resolution.offer[0] if resolution.offer else ""
    if not offer or not any(f["key"] == offer for f in opts["families"]):
        return ""
    form["character_family"] = offer
    touched(form, "character_family")
    # The sliders and the look belonged to whatever the form said before.
    form["character_body"] = "{}"
    if not _theme_offered(opts, offer, str(form.get("character_theme") or "")):
        form["character_theme"] = THEME_UNSET
    return offer


def switch_to_sprite_sheet(form: dict[str, Any]) -> None:
    """The other deliverable for the same brief. **The prompt is untouched.**

    SDXL draws what the registry does not model, which is the whole reason this
    is an escape route rather than a consolation prize -- and a route that
    rewrote the brief on the way would send a different request than the one
    the user was refused for.
    """
    from .. import create_assets

    form["asset_type"] = "sprite_sheet"
    form["generation_type"] = "sprite_sheet"
    create_assets.sync_legacy_fields(form)


def hand_to_troupe(ctx: Any, form: dict[str, Any]) -> None:
    """The third route: a generated reference and a reconstruction.

    The brief goes into ``troupe_mode.form`` -- *the* form that mode's pane
    draws, not a copy -- and then the mode opens. The prompt here is left
    exactly as it was, ``switch_to_sprite_sheet``'s rule and its reason.
    """
    from .. import troupe_mode
    from ..state import set_mode

    troupe_mode.form(ctx)["prompt"] = str(form.get("prompt") or "")
    set_mode(ctx.state, "troupe")


def _int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
