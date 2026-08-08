from __future__ import annotations

from pathlib import Path

import pytest

from warlock.pipelines import prompt

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sdxl-turbo"

# Deliberately *not* a module-level pytestmark. Only chunk()/count() need a
# real CLIP tokenizer; build(), pad_pair() and everything about TILE_TEMPLATE
# and TILE_FIELDS is pure string assembly. A module-level skip made a checkout
# without the sdxl-turbo weights -- a worktree, say -- silently stop asserting
# the tile field partition, which is exactly the regression those tests exist
# to catch. The transformers import lives in the fixture for the same reason:
# prompt.py imports it inside load_tokenizers(), so the module is importable
# without it.
needs_tokenizer = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer" / "vocab.json").exists(),
    reason="local sdxl-turbo tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def tokenizers():
    pytest.importorskip("transformers")
    return prompt.load_tokenizers(MODEL_DIR)


@needs_tokenizer
def test_short_prompt_yields_exactly_one_chunk(tokenizers):
    chunks = prompt.chunk("a wooden crate, worn condition", tokenizers)
    assert chunks == ["a wooden crate, worn condition"]


@needs_tokenizer
def test_a_long_prompt_yields_more_than_one_chunk(tokenizers):
    fragment = "an ornate medieval fantasy weapon with intricate engravings"
    long_text = ", ".join([fragment] * 8)
    chunks = prompt.chunk(long_text, tokenizers)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 77


@needs_tokenizer
def test_chunking_never_splits_a_phrase(tokenizers):
    text = "alpha one, beta two, gamma three"
    chunks = prompt.chunk(text, tokenizers, limit=5)
    for phrase in ("alpha one", "beta two", "gamma three"):
        assert any(phrase in c for c in chunks)


@needs_tokenizer
def test_a_lone_overlong_phrase_falls_back_to_whitespace_split(tokenizers):
    single_phrase = " ".join(["overlong"] * 40)
    chunks = prompt.chunk(single_phrase, tokenizers, limit=10)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 12


@needs_tokenizer
def test_a_single_unsplittable_atom_is_hard_split_not_truncated(tokenizers):
    # One whitespace-free "word" (a pasted URL, say) that alone exceeds the
    # limit: chunk() must slice it rather than emit an over-limit chunk the
    # encoder would silently truncate.
    atom = "x/" * 40
    chunks = prompt.chunk(atom, tokenizers, limit=10)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 12
    # No characters were lost in the split.
    assert "".join(chunks) == atom


@needs_tokenizer
def test_empty_prompt_yields_one_empty_chunk(tokenizers):
    assert prompt.chunk("", tokenizers) == [""]


def test_pad_pair_equalises_chunk_counts():
    a, b = prompt.pad_pair(["x", "y", "z"], ["only"])
    assert len(a) == len(b) == 3
    assert b == ["only", "", ""]

    a2, b2 = prompt.pad_pair(["x"], ["x"])
    assert a2 == ["x"]
    assert b2 == ["x"]


def test_build_reproduces_the_trigger_and_template_order():
    # Against the template constant, not a copy of its opening words: this test
    # is about the *order* the three pieces are assembled in, and a literal made
    # it fail for a wording change that left that order exactly as it was.
    text = prompt.build("a barrel", {}, trigger="3d style, 3d render")
    body = prompt.PROMPT_TEMPLATE.format(prompt="a barrel", view=prompt.view_clause())
    assert text == f"3d style, 3d render, {body}"


def test_build_with_no_trigger_has_no_leading_comma():
    text = prompt.build("a barrel", {})
    assert text == prompt.PROMPT_TEMPLATE.format(prompt="a barrel", view=prompt.view_clause())
    assert not text.startswith(",")


def test_a_tile_prompt_does_not_ask_for_a_single_centred_object():
    from warlock.pipelines import prompt as prompt_mod

    out = prompt_mod.build("mossy cobblestone", {}, tile=True)
    assert "single subject" not in out
    assert "seamless" in out and "tileable" in out


def test_a_tile_prompt_keeps_the_users_words_first():
    from warlock.pipelines import prompt as prompt_mod

    out = prompt_mod.build("mossy cobblestone", {}, tile=True)
    assert out.startswith("mossy cobblestone")


def test_a_tile_uses_only_the_surface_half_of_the_taxonomy():
    from warlock import guidance
    from warlock.pipelines import prompt as prompt_mod

    params = guidance.normalize(
        {"material": "stone", "category": "weapon", "silhouette": "elongated"}
    )
    out = prompt_mod.build("cobblestone", params, tile=True)
    assert guidance.MATERIALS["stone"].prompt in out
    assert guidance.CATEGORIES["weapon"].prompt not in out


def test_the_object_prompt_is_unchanged_by_the_tile_addition():
    # The default path must be byte-identical: every recipe on disk was
    # recorded against it.
    from warlock.pipelines import prompt as prompt_mod

    assert prompt_mod.build("a barrel", {}) == prompt_mod.PROMPT_TEMPLATE.format(
        prompt="a barrel", view=prompt_mod.view_clause()
    )


def test_the_tile_field_list_is_a_real_subset_of_the_taxonomy():
    from warlock import guidance
    from warlock.pipelines import prompt as prompt_mod

    # Against _PROMPT_FIELDS, not form_fields(). form_fields() is tuple(_TABLES)
    # and so includes base_model, style_lora, ip_adapter and control -- fields
    # that contribute no prompt fragment at all, so a TILE_FIELDS entry naming
    # one would satisfy that weaker check while adding nothing to a tile prompt.
    assert set(prompt_mod.TILE_FIELDS) < set(guidance._PROMPT_FIELDS)


def test_the_object_template_does_not_ask_for_concept_art():
    """The template rewrite. Every one of the 17 refusals in the 2026-08-07 sweep
    was the multi-object rule, and the family was concept-art layouts: character
    sheets, turnarounds, multi-view plates. "game asset concept art" is a
    request for exactly that -- a sheet is the canonical form of the genre --
    and it sat in the template that wraps every object prompt.
    """
    from warlock.pipelines import prompt as prompt_mod

    assert "concept art" not in prompt_mod.PROMPT_TEMPLATE


def test_the_object_template_asks_for_one_subject_and_nothing_beside_it():
    """The positive half of the template rewrite, and positive on purpose: a user is
    free to empty ``negative_prompt``, so a constraint that lives only there is
    one the composed prompt can lose."""
    from warlock.pipelines import prompt as prompt_mod

    text = prompt_mod.build("a rogue", {})
    assert "single subject" in text
    assert "no other objects" in text


def test_the_sheet_template_still_asks_for_a_grid():
    """The isolation clause must not leak across. SHEET_TEMPLATE restyles a
    contact sheet of eight real renders, so "one subject, nothing beside it" is
    the opposite of what it needs -- the two templates fight by design."""
    from warlock.pipelines import prompt as prompt_mod

    assert "grid of separate character poses" in prompt_mod.SHEET_TEMPLATE
    assert "single subject" not in prompt_mod.SHEET_TEMPLATE
    assert "no other objects" not in prompt_mod.SHEET_TEMPLATE


def test_no_object_field_leaks_into_the_tile_subset():
    # The partition pinned in the direction that actually goes wrong. Asserting
    # only that a category fragment is absent from one composed prompt leaves
    # "someone adds mood or rarity to TILE_FIELDS" -- the exact regression that
    # turns a "cobblestone" tile into a picture of a cobblestone -- passing.
    from warlock.pipelines import prompt as prompt_mod

    assert set(prompt_mod.TILE_FIELDS).isdisjoint(
        {"category", "silhouette", "rarity", "emissive", "mood", "platform"}
    )


# --- framing: the view clause is a field, and the default is byte-identical ---

# The exact object prompt this compiler produced before ``framing`` existed,
# for the subject "a barrel" and no guidance at all. A literal on purpose --
# every other prompt test here is deliberately written against the template
# constant so a wording change does not break it, and this one is the exact
# opposite: it is the whole safety argument for leaving PROMPT_VERSION at 4
# through a refactor that moved the view clause out of the template and into a
# taxonomy field. A test written against the constant cannot make that
# argument, because it would follow the constant wherever it went.
DEFAULT_COMPOSITION = (
    "a barrel, a single subject centered on a plain light gray background, "
    "no other objects, 3/4 perspective view, studio lighting, game asset render, "
    "full object in frame, no cropping, no text, no watermark"
)


def test_the_default_composition_is_byte_identical_to_the_pre_framing_one():
    from warlock.pipelines import prompt as prompt_mod

    assert prompt_mod.build("a barrel", {}) == DEFAULT_COMPOSITION
    # And the three ways of saying "the default": absent, empty, explicit.
    assert prompt_mod.build("a barrel", {"framing": ""}) == DEFAULT_COMPOSITION
    assert prompt_mod.build("a barrel", {"framing": "three_quarter"}) == DEFAULT_COMPOSITION
    # A value no table carries is skipped exactly as compose_prompt skips one,
    # so a job row written by a future spelling still composes something.
    assert prompt_mod.build("a barrel", {"framing": "nonsense"}) == DEFAULT_COMPOSITION
    assert prompt_mod.PROMPT_VERSION == 4


def test_front_ortho_composes_a_different_view_clause():
    from warlock.pipelines import prompt as prompt_mod

    text = prompt_mod.build("a barrel", {"framing": "front_ortho"})
    assert text != DEFAULT_COMPOSITION
    assert "3/4 perspective view" not in text
    assert "front orthographic view" in text
    # Only the view clause moves: everything the template says on either side
    # of it is what it was.
    head, _, tail = DEFAULT_COMPOSITION.partition(", 3/4 perspective view,")
    assert text.startswith(head)
    assert text.endswith(tail)


def test_front_ortho_does_not_fight_the_character_pose_fragment():
    """``category=character`` already asks for "T-pose neutral stance", and a
    front orthographic camera over a T-pose is the canonical reference plate --
    which is also the concept-art layout that caused all 17 refusals of the
    2026-08-07 rogue sweep. So the framing clause is about the *camera* only
    (it names no pose, and therefore cannot contradict one) and carries its own
    single-view guard."""
    from warlock import guidance
    from warlock.pipelines import prompt as prompt_mod

    clause = guidance.FRAMINGS["front_ortho"].prompt
    for word in ("pose", "stance", "sheet", "turnaround", "views"):
        assert word not in clause.lower(), clause

    text = prompt_mod.build("a knight", {"category": "character", "framing": "front_ortho"})
    assert "T-pose neutral stance" in text
    assert "one view only" in text
    assert "single subject" in text


def test_a_tile_prompt_takes_no_view_clause():
    """TILE_TEMPLATE has its own flat top-down framing, so the field is inert
    there rather than composing two cameras into one prompt."""
    from warlock.pipelines import prompt as prompt_mod

    plain = prompt_mod.build("mossy cobblestone", {}, tile=True)
    assert prompt_mod.build("mossy cobblestone", {"framing": "front_ortho"}, tile=True) == plain
    assert "orthographic" in plain  # the tile's own, not the field's
