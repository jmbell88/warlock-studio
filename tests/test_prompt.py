from __future__ import annotations

from pathlib import Path

import pytest

from warlock.pipelines import prompt

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sdxl-turbo"

# Deliberately *not* a module-level pytestmark. Only chunk()/count() need a
# real CLIP tokenizer; build(), pad_pair() and everything about the templates
# is pure string assembly. A module-level skip made a checkout without the
# sdxl-turbo weights -- a worktree, say -- silently stop asserting the
# template properties. The transformers import lives in the fixture for the
# same reason: prompt.py imports it inside load_tokenizers(), so the module is
# importable without it.
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
    # is about the *order* the pieces are assembled in, and a literal made it
    # fail for a wording change that left that order exactly as it was.
    text = prompt.build("a barrel", {}, trigger="3d style, 3d render")
    body = prompt.PROMPT_TEMPLATE.format(prompt="a barrel")
    assert text == f"3d style, 3d render, {body}"


def test_build_with_no_trigger_has_no_leading_comma():
    text = prompt.build("a barrel", {})
    assert text == prompt.PROMPT_TEMPLATE.format(prompt="a barrel")
    assert not text.startswith(",")


def test_a_tile_prompt_does_not_ask_for_a_single_centred_object():
    out = prompt.build("mossy cobblestone", {}, tile=True)
    assert "single subject" not in out
    assert "seamless" in out and "tileable" in out


def test_a_tile_prompt_keeps_the_users_words_first():
    out = prompt.build("mossy cobblestone", {}, tile=True)
    assert out.startswith("mossy cobblestone")


def test_the_object_template_does_not_ask_for_concept_art():
    """The template rewrite. Every one of the 17 refusals in the 2026-08-07 sweep
    was the multi-object rule, and the family was concept-art layouts: character
    sheets, turnarounds, multi-view plates. "game asset concept art" is a
    request for exactly that -- a sheet is the canonical form of the genre --
    and it sat in the template that wraps every object prompt.
    """
    assert "concept art" not in prompt.PROMPT_TEMPLATE


def test_the_object_template_asks_for_one_subject_and_nothing_beside_it():
    """The positive half of the template rewrite, and positive on purpose: a user is
    free to empty ``negative_prompt``, so a constraint that lives only there is
    one the composed prompt can lose."""
    text = prompt.build("a rogue", {})
    assert "single subject" in text
    assert "no other objects" in text


def test_the_sheet_template_still_asks_for_a_grid():
    """The isolation clause must not leak across. SHEET_TEMPLATE restyles a
    contact sheet of eight real renders, so "one subject, nothing beside it" is
    the opposite of what it needs -- the two templates fight by design."""
    assert "grid of separate character poses" in prompt.SHEET_TEMPLATE
    assert "single subject" not in prompt.SHEET_TEMPLATE
    assert "no other objects" not in prompt.SHEET_TEMPLATE


# --- version 5: the taxonomy retirement ---------------------------------------

# The exact object prompt this compiler produces for the subject "a barrel"
# and no guidance at all. A literal on purpose: it is the safety argument that
# the empty-params composition is byte-identical across the taxonomy
# retirement (PROMPT_VERSION 4 -> 5) -- the view clause was re-inlined as
# exactly the fragment the deleted ``{view}`` slot used to receive. A test
# written against the template constant cannot make that argument, because it
# would follow the constant wherever it went.
DEFAULT_COMPOSITION = (
    "a barrel, a single subject centered on a plain light gray background, "
    "no other objects, 3/4 perspective view, studio lighting, game asset render, "
    "full object in frame, no cropping, no text, no watermark"
)


def test_the_default_composition_is_byte_identical_across_the_retirement():
    assert prompt.build("a barrel", {}) == DEFAULT_COMPOSITION
    assert prompt.PROMPT_VERSION == 5


def test_stale_taxonomy_params_compose_the_default():
    """A job row written before the retirement still composes -- its fragments
    are simply gone, which is the tolerance the retirement rests on."""
    old = {"framing": "front_ortho", "art_style": "nes", "category": "weapon"}
    assert prompt.build("a barrel", old) == DEFAULT_COMPOSITION


def test_a_stale_framing_is_inert_on_a_tile_too():
    plain = prompt.build("mossy cobblestone", {}, tile=True)
    assert prompt.build("mossy cobblestone", {"framing": "front_ortho"}, tile=True) == plain
    assert "orthographic" in plain  # the tile's own clause


def test_the_prompt_field_list_is_empty():
    """No stored field composes into the prompt any more; a future entry here
    is a deliberate re-opening, not an accident."""
    from warlock import guidance

    assert guidance._PROMPT_FIELDS == ()
