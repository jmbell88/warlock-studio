"""The chunked encoder's bit-identity against diffusers, for one chunk.

The whole argument for ``_encode_long_prompt`` is that a single-chunk prompt --
the common case, every short prompt anyone types -- comes out *bit-identical*
to the direct-string path diffusers takes, so chunking removes a ceiling and
changes nothing else. That was asserted nowhere: the CPU tests cover
``prompt.chunk``'s splitting, and the encoder itself was only ever exercised by
running a whole generation.

Needs the sdxl-turbo text encoders on a real card, so it carries the gpu marker
and is deselected by default (`-m "not gpu"`).

Run with: uv run pytest tests/test_prompt_encode_gpu.py -m gpu
"""

from __future__ import annotations

import pytest

from warlock import models
from warlock.config import get_config
from warlock.pipelines import prompt as prompt_mod
from warlock.pipelines.text2image import Text2Image, _encode_long_prompt

pytestmark = pytest.mark.gpu

SHORT = "a wooden crate, worn condition, game asset, neutral background"


@pytest.fixture(scope="module")
def t2i():
    config = get_config()
    spec = models.BASE_MODELS[config.t2i_model]
    pipe = Text2Image(spec, config.t2i_model_root)
    if not (pipe.model_dir / "model_index.json").exists():
        pytest.skip(f"{spec.label} weights not downloaded")
    pipe.load()
    yield pipe
    pipe.unload()


def test_a_single_chunk_encodes_bit_identically_to_diffusers(t2i):
    """Not "close": identical. A rounding difference here moves every pixel of
    every image against the recorded corpus, and the corpus is keyed on the
    prompt version rather than on the encoder."""
    import torch

    tokenizers = prompt_mod.load_tokenizers(t2i.model_dir)
    chunks = prompt_mod.chunk(SHORT, tokenizers)
    assert len(chunks) == 1, "the premise: this prompt fits in one chunk"

    ours, ours_pooled = _encode_long_prompt(t2i._pipe, chunks)
    theirs, _neg, theirs_pooled, _neg_pooled = t2i._pipe.encode_prompt(
        prompt=SHORT, device=t2i._pipe._execution_device, num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    assert ours.shape == theirs.shape
    assert torch.equal(ours, theirs)
    assert torch.equal(ours_pooled, theirs_pooled)


def test_two_chunks_extend_the_sequence_and_keep_the_first_chunks_pooled(t2i):
    """The other half of the contract: chunks concatenate on the *sequence*
    axis (the UNet's cross-attention has no length limit), and pooled is a
    single vector that cannot be concatenated -- it comes from the first chunk
    only, matching encode_prompt's own rule."""
    import torch

    one, one_pooled = _encode_long_prompt(t2i._pipe, [SHORT])
    two, two_pooled = _encode_long_prompt(t2i._pipe, [SHORT, "a second phrase"])

    assert two.shape[1] == one.shape[1] * 2
    assert two.shape[2] == one.shape[2]
    assert torch.equal(two[:, : one.shape[1]], one)
    assert torch.equal(two_pooled, one_pooled)
