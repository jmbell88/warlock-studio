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
    text = prompt.build("a barrel", {}, trigger="3d style, 3d render")
    assert text.startswith("3d style, 3d render, a barrel, single object centered")
    assert text.endswith("no cropping, no text, no watermark")


def test_build_with_no_trigger_has_no_leading_comma():
    text = prompt.build("a barrel", {})
    assert text.startswith("a barrel, single object centered")


def test_a_tile_prompt_does_not_ask_for_a_single_centred_object():
    from warlock.pipelines import prompt as prompt_mod

    out = prompt_mod.build("mossy cobblestone", {}, tile=True)
    assert "single object" not in out
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
        prompt="a barrel"
    )


def test_the_tile_field_list_is_a_real_subset_of_the_taxonomy():
    from warlock import guidance
    from warlock.pipelines import prompt as prompt_mod

    # Against _PROMPT_FIELDS, not form_fields(). form_fields() is tuple(_TABLES)
    # and so includes base_model, style_lora, ip_adapter and control -- fields
    # that contribute no prompt fragment at all, so a TILE_FIELDS entry naming
    # one would satisfy that weaker check while adding nothing to a tile prompt.
    assert set(prompt_mod.TILE_FIELDS) < set(guidance._PROMPT_FIELDS)


def test_no_object_field_leaks_into_the_tile_subset():
    # The partition pinned in the direction that actually goes wrong. Asserting
    # only that a category fragment is absent from one composed prompt leaves
    # "someone adds mood or rarity to TILE_FIELDS" -- the exact regression that
    # turns a "cobblestone" tile into a picture of a cobblestone -- passing.
    from warlock.pipelines import prompt as prompt_mod

    assert set(prompt_mod.TILE_FIELDS).isdisjoint(
        {"category", "silhouette", "rarity", "emissive", "mood", "platform"}
    )
