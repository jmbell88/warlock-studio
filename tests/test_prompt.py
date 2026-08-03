from __future__ import annotations

from pathlib import Path

import pytest

from warlock.pipelines import prompt

pytest.importorskip("transformers")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sdxl-turbo"
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer" / "vocab.json").exists(),
    reason="local sdxl-turbo tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def tokenizers():
    return prompt.load_tokenizers(MODEL_DIR)


def test_short_prompt_yields_exactly_one_chunk(tokenizers):
    chunks = prompt.chunk("a wooden crate, worn condition", tokenizers)
    assert chunks == ["a wooden crate, worn condition"]


def test_a_long_prompt_yields_more_than_one_chunk(tokenizers):
    fragment = "an ornate medieval fantasy weapon with intricate engravings"
    long_text = ", ".join([fragment] * 8)
    chunks = prompt.chunk(long_text, tokenizers)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 77


def test_chunking_never_splits_a_phrase(tokenizers):
    text = "alpha one, beta two, gamma three"
    chunks = prompt.chunk(text, tokenizers, limit=5)
    for phrase in ("alpha one", "beta two", "gamma three"):
        assert any(phrase in c for c in chunks)


def test_a_lone_overlong_phrase_falls_back_to_whitespace_split(tokenizers):
    single_phrase = " ".join(["overlong"] * 40)
    chunks = prompt.chunk(single_phrase, tokenizers, limit=10)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 12


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
