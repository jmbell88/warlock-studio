"""A recipe is a request that either says what it means or says why not.

Two claims run through the whole file. **It refuses by name** -- every rejection
carries the ``field`` the control lives at, because a service door turns that
into a highlighted slider and a refusal with no address is a refusal the user
cannot act on. And **it never clamps**: a value off the ladder comes back as a
sentence, not as the nearest legal value silently substituted.
"""

from __future__ import annotations

import pytest

from warlock.characters import DEFAULT_RECIPE, CharacterError, Recipe
from warlock.characters import recipe as recipelib
from warlock.pipelines import charsheet


def test_the_default_recipe_is_the_brief():
    """The example the whole programme was specified against, in one assertion:
    a fire ogre seen 3/4 top-down at 35 degrees, idle/walk/attack across eight
    directions, 64px logical cells, 32 colours, an outer outline. 18 frames by 8
    directions is 144 cells, and that number is the budget every later decision
    about render time was made against."""
    r = DEFAULT_RECIPE
    assert (r.family, r.archetype, r.theme) == ("ogre", "humanoid", "fire")
    assert (r.camera, r.elevation) == ("three_quarter_top_down", 35.0)
    assert dict(r.animations) == {"idle": 4, "walk": 8, "attack": 6}
    assert (r.directions, r.logical_size, r.colors, r.outline) == (8, 64, 32, "outer")
    assert r.cell_count == 144
    # Through ``charsheet`` and not through this module's arithmetic: the frame
    # table is what the renderer and the sidecar agree on.
    assert r.layout_dict()["cell_count"] == 144
    assert len(r.layout_dict()["runs"]) == 3 * 8


def test_the_default_recipe_already_looks_like_an_ogre():
    """A recipe that named a species but carried neutral sliders would render a
    large human. The species' channel defaults are the species."""
    assert DEFAULT_RECIPE.appearance["bulk"] == pytest.approx(0.9)
    assert DEFAULT_RECIPE.appearance["shoulder_width"] == pytest.approx(0.75)


def test_a_recipe_round_trips_through_its_own_dict():
    back = Recipe.from_dict(DEFAULT_RECIPE.as_dict())
    assert back == DEFAULT_RECIPE
    assert Recipe.from_dict(back.as_dict()) == back


def test_changing_species_never_silently_carries_a_palette_across():
    """``replace`` revalidates the whole recipe rather than patching one key.

    The ogre's ``fire`` look does not exist for an elf, and the two ways to
    handle that -- fall back to the elf's first theme, or refuse -- are the same
    choice the unsupported-creature rule makes one level up. It refuses, and
    names the control, so the caller picks a look on purpose.
    """
    with pytest.raises(CharacterError) as excinfo:
        DEFAULT_RECIPE.replace(family="elf")
    assert excinfo.value.field == "theme"
    assert DEFAULT_RECIPE.replace(family="elf", theme="natural").family == "elf"


def test_an_unknown_species_is_refused_and_offers_the_ones_we_have():
    # ``manticore`` and not ``dragon``: the winged archetype ships a real
    # dragon, so it stopped being an example of a species we do not make.
    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "manticore"})
    assert excinfo.value.field == "family"
    assert "ogre" in str(excinfo.value)


def test_a_newer_family_version_is_refused_by_name():
    """A recipe written by a later build describes a mesh this one does not
    have. Rebuilding it from the older asset would hand back a character that is
    visibly not the saved one, with nothing on screen saying so."""
    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "ogre", "family_version": 99})
    assert excinfo.value.field == "family_version"
    assert "99" in str(excinfo.value)


@pytest.mark.parametrize(
    "payload, field",
    [
        ({"family": "nope"}, "family"),
        ({"family_version": 7}, "family_version"),
        ({"appearance": {"wingspan": 0.2}}, "appearance"),
        ({"appearance": {"bulk": 4.0}}, "appearance"),
        ({"appearance": {"bulk": "wide"}}, "appearance"),
        ({"appearance": [1, 2]}, "appearance"),
        ({"theme": "neon"}, "theme"),
        ({"camera": "worm"}, "camera"),
        ({"elevation": 120.0}, "elevation"),
        ({"elevation": float("nan")}, "elevation"),
        ({"animations": {"backflip": 4}}, "animations"),
        ({"animations": {"walk": 0}}, "animations"),
        ({"animations": {}}, "animations"),
        ({"directions": 5}, "directions"),
        ({"logical_size": 100}, "logical_size"),
        ({"colors": 7}, "colors"),
        ({"outline": "glow"}, "outline"),
        ({"reduce_mode": "lanczos"}, "reduce_mode"),
        ({"seed": -1}, "seed"),
        ({"name": "x" * 65}, "name"),
    ],
)
def test_every_refusal_names_the_control_it_came_from(payload, field):
    """``service.errors.invalid_from`` reads ``.field`` off whatever it wraps and
    hands it to the form. A refusal with no address reaches the user as a
    sentence with no control to fix."""
    with pytest.raises(CharacterError) as excinfo:
        Recipe.from_dict({"family": "ogre", **payload})
    assert excinfo.value.field == field


@pytest.mark.parametrize(
    "payload",
    [
        {"appearance": {"bulk": 4.0}},
        {"colors": 7},
        {"logical_size": 100},
        {"elevation": 120.0},
    ],
)
def test_an_out_of_range_value_is_never_quietly_clamped(payload):
    """The ``service/jobs`` rule about tile size, applied here: "make me 96px
    tiles" answered with 32px tiles and nobody told is the failure. A slider that
    snaps back on save is the same failure in nicer clothes."""
    with pytest.raises(CharacterError):
        Recipe.from_dict({"family": "ogre", **payload})


def test_a_recipe_carries_no_archetype_of_its_own():
    """Derived from the species, never stored: a stored copy would be one edit
    away from a request whose archetype and species name different skeletons."""
    assert "archetype" not in DEFAULT_RECIPE.as_dict()
    assert DEFAULT_RECIPE.archetype == DEFAULT_RECIPE.spec.archetype


# --- the two copies of the pixel vocabulary ---------------------------------
#
# ``recipe`` restates the ladders because ``characters`` may not import
# ``service``. This file is the sole owner of the agreement between the copies,
# the arrangement ``pipelines.charsheet`` and ``studio.troupe.spec`` already
# have: a change to one is a change to both plus these four assertions.


def test_the_size_and_direction_ladders_are_charsheets():
    assert recipelib.LOGICAL_SIZES == charsheet.SIZES
    assert set(recipelib.DIRECTION_CHOICES) == set(charsheet.DIRECTION_PRESETS)


def test_the_colour_ladder_is_troupes():
    from warlock.service import troupe

    assert recipelib.COLOR_CHOICES == troupe.TROUPE_COLOR_CHOICES


def test_the_outline_and_reduce_modes_are_pixelizes():
    from warlock.pipelines import pixelize

    assert set(recipelib.OUTLINE_MODES) == set(pixelize.OUTLINE_MODES)
    assert set(recipelib.REDUCE_MODES) == set(pixelize.REDUCE_MODES)


def test_every_camera_a_recipe_accepts_is_a_real_preset():
    keys = {key for key, _label, _elev in charsheet.CAMERA_PRESETS}
    for camera in keys:
        assert Recipe.from_dict({"family": "ogre", "camera": camera}).camera == camera


def test_the_default_animations_leave_the_legacy_table_alone():
    """Three animations, not five. Not a subset chosen for brevity -- idle, walk
    and attack are what a character needs to read as alive top-down -- and the
    five-row ``charsheet.ANIMATIONS`` table is deliberately untouched so every
    sheet Troupe has already rendered still means what it did."""
    assert set(DEFAULT_RECIPE.animations) == {"idle", "walk", "attack"}
    assert "run" not in DEFAULT_RECIPE.animations
    assert "jump" not in DEFAULT_RECIPE.animations
    assert [name for name, *_rest in charsheet.ANIMATIONS] == [
        "idle", "walk", "run", "attack", "jump"
    ]
    # And run and jump are still *askable*: leaving them out of the default is
    # not the same as removing them.
    assert Recipe.from_dict(
        {"family": "ogre", "animations": {"run": 8, "jump": 6}}
    ).cell_count == 112


# --- the registries ---------------------------------------------------------


def test_every_species_offers_a_look_and_every_look_paints_every_region():
    from warlock.characters import families

    for key, fam in families().items():
        assert fam.themes, key
        for theme in fam.themes:
            missing = set(fam.regions) - set(theme.materials)
            assert not missing, f"{key}/{theme.key} does not paint {sorted(missing)}"
            for region, colour in theme.materials.items():
                assert colour.startswith("#") and len(colour) == 7, f"{key}/{region}"


def test_every_nearest_hint_names_a_real_species():
    """A resolver offers ``nearest`` when the creature asked for is not one we
    make. A hint naming nothing would turn "we don't do dragons, want a troll?"
    into a dead end."""
    from warlock.characters import families

    known = set(families())
    for key, fam in families().items():
        assert set(fam.nearest) <= known, key
        assert key not in fam.nearest, f"{key} is its own nearest neighbour"


def test_every_species_default_is_inside_its_own_channel_range():
    from warlock.characters import families

    for key, fam in families().items():
        for channel in fam.channels:
            assert channel.lo <= channel.default <= channel.hi, f"{key}/{channel.key}"


def test_no_two_species_share_an_alias():
    """A resolver keyed on aliases has to have one answer per word. Two species
    claiming "orc" would make which one you got depend on dict order."""
    from warlock.characters import families

    seen: dict[str, str] = {}
    for key, fam in families().items():
        for alias in fam.aliases:
            assert alias not in seen, f"{key} and {seen[alias]} both claim {alias!r}"
            seen[alias] = key


def test_a_species_reads_its_rig_off_its_archetype_and_not_off_itself():
    from warlock.characters import families, get_family

    for fam in families().values():
        assert fam.template == fam.arch.template
        assert fam.regions == fam.arch.regions
    assert get_family("ogre").template == "humanoid"
    assert get_family("ogre").clip_library == "humanoid"
