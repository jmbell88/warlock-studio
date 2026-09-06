"""What the prompt resolver claims, over whatever the registry currently ships.

Every claim that could be phrased against the registry is parameterised over it
rather than written against a list of species: a sibling adding the quadruped's
rows must be able to add them without editing this file, and the two tests that
matter most -- ``test_every_species_is_reachable_by_at_least_one_alias`` and
``test_every_alias_resolves_to_its_own_species`` -- exist precisely to fail when
a species arrives that nobody can ask for.

Where a claim needs a species the registry does not ship yet (a winged one, an
amorphous one), the test builds a synthetic registry and passes it in. That is
not a mock of the resolver: ``resolve(..., families=...)`` is the documented
seam, and it is the only way to make "a winged creature is not offered a slime"
a real assertion today rather than a promise for the increment that lands them.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from warlock.characters import family as family_mod
from warlock.characters.family import Family
from warlock.characters.resolve import (
    ACTION_WORDS,
    CAMERA_WORDS,
    KNOWN_CREATURES,
    Resolution,
    Span,
    offer_sentence,
    resolve,
    vocabulary,
)
from warlock.pipelines import charsheet

FAMILIES = family_mod.families()
VOCAB = vocabulary()

#: Creature nouns that are *not* a species alias -- the ones an offer is for.
UNSUPPORTED = sorted(set(KNOWN_CREATURES) - set(VOCAB["families"]))


def _fake(key: str, archetype: str, *aliases: str) -> Family:
    """A species row good enough for the resolver: it reads aliases, label,
    archetype, themes and nearest, and nothing else."""
    return Family(
        key=key,
        version=1,
        label=key.capitalize(),
        archetype=archetype,
        silhouette=key,
        aliases=aliases or (key,),
        height_m=1.0,
        channel_defaults={},
        themes=(),
    )


def _with(*extra: Family) -> dict[str, Family]:
    registry = dict(FAMILIES)
    registry.update({f.key: f for f in extra})
    return registry


def _only(*extra: Family) -> dict[str, Family]:
    """A registry of **exactly** these species, with the real one shut out.

    ``_with`` layers fakes over the shipped table, which is right for a test
    asking what happens *in addition* to what we ship -- and wrong for every
    test below about ranking. Those were written when the registry shipped
    twelve humanoids and nothing else, so "a phoenix is offered the winged
    species" could be asserted against a fake winged row that had no real
    competition. The moment the quadruped, winged and amorphous archetypes
    landed, ``resolve("dragon")`` stopped being an unsupported creature at all
    and four of them failed -- not because the ranking was wrong but because
    the fixtures were reading the shipped table through the fakes.

    A ranking claim must not depend on which species happen to exist, so it
    gets a registry containing only what the claim is about.
    """
    return {f.key: f for f in extra}


# --- the brief -----------------------------------------------------------------


def test_the_brief_example_resolves_to_a_fire_ogre_seen_three_quarter_top_down():
    """The sentence the whole layer was specified from. Nothing in it is left
    over: "sprite sheet" is noise, not an unknown word."""
    got = resolve("fire ogre, 3/4 top down sprite sheet")
    assert got.family == "ogre"
    assert got.theme == "fire"
    assert got.camera_preset == "three_quarter_top_down"
    assert got.unrecognised == ()
    assert got.archetype == FAMILIES["ogre"].archetype


def test_an_empty_prompt_resolves_to_nothing_rather_than_raising():
    got = resolve("")
    assert got == Resolution()


# --- longest alias first, then the tie order -----------------------------------


@pytest.mark.parametrize(
    ("text", "camera"),
    [
        ("three quarter top down", "three_quarter_top_down"),
        ("3/4 view", "three_quarter_top_down"),
        ("side view", "side"),
        ("top down view", "top_down"),
    ],
)
def test_a_longer_alias_beats_the_prefix_it_starts_with(text, camera):
    """"side view" is the camera, not "side" plus a stray word; "top down view"
    is the camera even though the same three words are also in the noise table,
    because a same-length collision goes to the higher category."""
    got = resolve(text)
    assert got.camera_preset == camera
    assert got.unrecognised == ()


def test_birds_eye_view_is_a_camera_and_not_the_bird_species():
    """The collision the tie order and the longest-first scan exist for. Run
    against a registry that *does* ship a bird, so the claim is about the scan
    and not about the bird being absent."""
    registry = _with(_fake("bird", "winged", "bird", "birds"))
    got = resolve("bird's eye view sprite sheet", families=registry)
    assert got.camera_preset == "top_down"
    assert got.family is None
    assert got.unrecognised == ()
    # ...and the species is still reachable when it is what was asked for.
    assert resolve("bird sprite sheet", families=registry).family == "bird"


# --- spelling ------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_a_species_survives_shouting_and_punctuation(key):
    alias = FAMILIES[key].aliases[0]
    assert resolve(alias.upper()).family == key
    assert resolve(f"  {alias.title()}!!  ").family == key


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_a_species_survives_being_pluralised(key):
    """A prompt says "goblins" as often as "goblin"."""
    alias = FAMILIES[key].aliases[0]
    if alias.endswith("s"):
        pytest.skip(f"{alias!r} is already plural")
    assert resolve(f"{alias}s").family == key


@pytest.mark.parametrize(
    ("text", "action"),
    [("walking", "walk"), ("attacking", "attack"), ("running", "run"), ("jumping", "jump")],
)
def test_an_inflected_action_is_the_action(text, action):
    assert resolve(text).actions == (action,)


def test_an_inflected_theme_word_is_the_theme():
    assert resolve("flames").theme == "fire"
    assert resolve("flaming ogre").theme == "fire"


# --- coverage ------------------------------------------------------------------


def test_noise_is_consumed_silently():
    got = resolve("sprite sheet, pixel art, 8 directions, game asset, 2d")
    assert got.unrecognised == ()
    assert {s.kind for s in got.spans} == {"noise"}
    assert got.family is None


def test_unknown_words_come_back_in_their_original_spelling_and_order():
    """The UI lists them back at the user, so "Frobnitz" must not arrive as
    "frobnitz" -- a word the user did not type is a word they cannot find."""
    got = resolve("Xyzzy ogre Frobnitz banana")
    assert got.unrecognised == ("Xyzzy", "Frobnitz", "banana")
    assert got.family == "ogre"


def test_a_prompt_with_no_species_in_it_gets_no_species():
    """No default. A Create surface that filled in "human" here would be
    answering a question the user has not been asked."""
    got = resolve("3/4 top down sprite sheet, walk and attack")
    assert got.family is None
    assert got.archetype is None
    assert got.actions == ("walk", "attack")


def test_the_second_species_named_loses_and_is_reported():
    got = resolve("orc and goblin")
    assert got.family == "orc"
    assert got.unrecognised == ("goblin",)
    assert [(s.key, s.applied) for s in got.spans if s.kind == "family"] == [
        ("orc", True),
        ("goblin", False),
    ]


def test_every_token_is_either_covered_by_a_span_or_reported():
    """The invariant the UI's underlining rests on: no token vanishes."""
    text = "fire ogre, 3/4 top down sprite sheet, walking, wibble"
    got = resolve(text)
    covered = {i for s in got.spans for i in range(s.start, s.end)}
    tokens = text.replace(",", " ").split()
    assert len(covered) + len(got.unrecognised) <= len(tokens)
    assert "wibble" in got.unrecognised


def test_actions_come_back_in_the_frame_tables_order():
    """Two prompts naming the same movements must plan the same sheet."""
    assert resolve("attack idle walk").actions == resolve("walk idle attack").actions
    assert resolve("attack idle walk").actions == ("idle", "walk", "attack")


# --- the vocabulary agrees with the pipeline and the registry -------------------


@pytest.mark.parametrize("key", sorted(CAMERA_WORDS))
def test_every_camera_key_the_vocabulary_emits_is_a_real_preset(key):
    """A preset renamed in charsheet and not here would resolve to a camera the
    sheet planner has never heard of."""
    assert key in {k for k, _label, _elev in charsheet.CAMERA_PRESETS}


def test_every_camera_preset_can_be_asked_for_in_words():
    """The other direction: a preset nobody can name is a control the prompt bar
    cannot reach."""
    assert {k for k, _label, _elev in charsheet.CAMERA_PRESETS} <= set(CAMERA_WORDS)


@pytest.mark.parametrize("key", sorted(ACTION_WORDS))
def test_every_action_key_the_vocabulary_emits_is_a_real_animation(key):
    assert key in {name for name, *_rest in charsheet.ANIMATIONS}


def test_the_action_order_is_the_frame_tables_order():
    from warlock.characters.resolve import _ACTION_ORDER

    assert tuple(name for name, *_rest in charsheet.ANIMATIONS) == _ACTION_ORDER


@pytest.mark.parametrize("alias", sorted(VOCAB["families"]))
def test_every_family_alias_maps_to_a_species_that_exists(alias):
    assert VOCAB["families"][alias] in FAMILIES


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_every_species_is_reachable_by_at_least_one_alias(key):
    """The test that catches a species added with no way to ask for it -- or
    with aliases the noise table eats."""
    assert key in set(VOCAB["families"].values())


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_every_alias_resolves_to_its_own_species(key):
    for alias in FAMILIES[key].aliases:
        assert resolve(alias).family == key, alias


@pytest.mark.parametrize("theme", sorted(set(VOCAB["themes"].values())))
def test_every_theme_key_the_vocabulary_emits_is_declared_by_a_species(theme):
    """A word that resolved to a look no species paints would be a control the
    recipe layer could only ever refuse."""
    assert any(t.key == theme for fam in FAMILIES.values() for t in fam.themes)


def test_a_theme_no_species_declares_is_not_in_the_vocabulary():
    """The filter is real, not decorative: build a registry whose only species
    declares nothing, and no theme word survives."""
    vocab = vocabulary({"nobody": _fake("nobody", "humanoid")})
    assert vocab["themes"] == {}


# --- round trip ----------------------------------------------------------------


def test_a_resolution_survives_a_round_trip_through_a_dict():
    got = resolve("fire ogre, 3/4 top down sprite sheet, walking")
    assert Resolution.from_dict(got.to_dict()) == got


def test_from_dict_tolerates_a_record_written_before_a_field_existed():
    assert Resolution.from_dict({"family": "ogre"}) == Resolution(family="ogre")


def test_a_span_survives_a_round_trip():
    span = Span(0, 2, "camera", "top_down", "top down", False)
    assert Span.from_dict(span.to_dict()) == span


# --- the offer -----------------------------------------------------------------


def test_an_unsupported_creature_is_never_silently_swapped():
    got = resolve("kraken sprite sheet")
    assert got.family is None
    assert got.creature_words == ("kraken",)
    assert got.offer
    sentence = offer_sentence(got)
    assert sentence is not None
    assert "kraken" in sentence
    assert FAMILIES[got.offer[0]].label.lower() in sentence


@pytest.mark.parametrize("word", UNSUPPORTED)
def test_every_known_creature_we_do_not_make_gets_an_offer_and_a_sentence(word):
    got = resolve(word)
    assert got.family is None, f"{word} matched a species"
    assert got.creature_words == (word,)
    assert got.offer, f"{word} was recognised but offered nothing"
    assert all(k in FAMILIES for k in got.offer)
    sentence = offer_sentence(got)
    assert sentence is not None
    assert word in sentence
    assert FAMILIES[got.offer[0]].label.lower() in sentence


def test_the_sentence_names_the_dictionary_form_of_a_pluralised_creature():
    assert offer_sentence(resolve("chimeras")) == offer_sentence(resolve("chimera"))
    assert "chimera " in offer_sentence(resolve("chimeras"))


def test_a_supported_creature_gets_no_offer_sentence():
    got = resolve("ogre")
    assert got.family == "ogre"
    assert got.offer == ()
    assert offer_sentence(got) is None


def test_the_sentence_reads_the_way_the_ui_says_it():
    """One home for the wording, so a tooltip and a toast cannot drift apart --
    and so nothing anywhere phrases it as though the swap had happened."""
    # ``manticore``, not ``dragon``: the winged archetype ships a real dragon
    # now, so asking for one is no longer asking for something we cannot make.
    registry = _only(_fake("wyvern", "winged"))
    got = resolve("manticore", families=registry)
    assert offer_sentence(got, families=registry) == (
        "Warlock has no manticore yet. The closest it makes is a wyvern."
    )
    assert got.family is None


def test_the_article_agrees_with_the_species_it_offers():
    registry = _only(_fake("aurochs", "quadruped"))
    got = resolve("bison", families=registry) if "bison" in KNOWN_CREATURES else resolve(
        "chimera", families=registry
    )
    assert offer_sentence(got, families=registry).endswith("is an aurochs.")


@pytest.mark.parametrize("word", UNSUPPORTED)
def test_an_offer_never_mixes_body_plans(word):
    """Whatever else the ranking does, everything offered is shaped alike."""
    offer = resolve(word).offer
    assert len({FAMILIES[k].archetype for k in offer}) <= 1


@pytest.mark.parametrize("word", UNSUPPORTED)
def test_an_offer_stays_inside_the_creatures_own_body_plan_when_we_ship_one(word):
    hint = KNOWN_CREATURES[word].archetype
    if hint is None or not family_mod.families_of(hint):
        pytest.skip(f"{hint!r} ships no species yet")
    assert all(FAMILIES[k].archetype == hint for k in resolve(word).offer)


def test_a_winged_creature_is_not_offered_a_slime():
    """Body plan beats alphabetical order. Against a registry that ships both a
    winged species and an amorphous one, a phoenix gets the winged one."""
    registry = _only(_fake("wyvern", "winged"), _fake("slime", "amorphous"))
    got = resolve("phoenix", families=registry)
    assert got.offer[0] == "wyvern"
    assert "slime" not in got.offer


def test_a_creature_falls_back_across_body_plans_only_when_its_own_ships_nothing():
    """A phoenix would rather be a four-legged thing than a puddle, but it is
    only offered either when nothing winged exists."""
    registry = _only(_fake("wolf", "quadruped"), _fake("slime", "amorphous"))
    assert resolve("phoenix", families=registry).offer[0] == "wolf"


def test_a_creature_of_unknown_shape_may_be_offered_anything():
    got = resolve("monster")
    assert got.archetype is None
    assert got.offer


def test_the_offer_prefers_the_kin_the_creature_names():
    """"cyclops" is an ogre before it is any other humanoid, and the ranking is
    data on the creature rather than whichever species sorted first."""
    got = resolve("cyclops")
    assert got.offer[0] == "ogre"
    assert got.offer[1] in FAMILIES["ogre"].nearest


def test_a_creature_word_is_reported_even_when_it_is_supported():
    """The UI says "you asked for an ogre" whether or not we make one."""
    assert resolve("ogre walking").creature_words == ("ogre",)


# --- the module's dependencies -------------------------------------------------


def test_the_resolver_is_standard_library_plus_the_registry():
    """The door imports this to answer "what can we make" before a window
    exists. Anything heavier here is on the cold-start path of the whole app --
    and the camera keys are literals for exactly this reason."""
    source = Path(family_mod.__file__).with_name("resolve.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.module == "family", node.module
            else:
                root = (node.module or "").split(".")[0]
                assert root in sys.stdlib_module_names, node.module
