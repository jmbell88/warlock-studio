"""Training a style LoRA, for real. The claim `test_loras.py` cannot make.

``test_loras.py`` covers the doors thoroughly -- the spec is strings and
numbers, the import registers an adapter, the training door copies the images
and queues a row, every refusal names its field -- and it covers them with the
trainer stubbed. That is the right shape for 19 fast tests, and it leaves one
thing untested: **whether the training loop converges to a file SDXL can
load**.

That is not a small gap. ``lora_train_worker`` is a from-scratch DreamBooth
loop -- attention-projection LoRA on the UNet, gradient checkpointing, a
diffusers-layout ``safetensors`` write -- and every way it can be wrong (a
mis-shaped adapter, a key naming convention ``load_lora_weights`` rejects, an
optimiser that produces NaN) ends with a file that exists, has a plausible
size, and does nothing or raises when a user selects it. The stub asserts the
child was spawned and the marker was read; it cannot assert the weights are
weights.

So this file trains a real adapter at ``MIN_STEPS`` and then **loads it into a
real pipeline**, which is the only assertion that distinguishes a trained
adapter from a well-formed file.

``MIN_STEPS`` and not ``DEFAULT_STEPS``: 800 steps is the setting that makes a
*good* adapter and it is 20-40 minutes on a 5090. 100 steps makes a bad one in
a couple of minutes, and a bad adapter loads exactly as a good one does. This
lane is asking whether the mechanism works, never whether the art is good --
that judgement needs a person and a set of the user's own images.

Run with: uv run pytest tests/test_lora_train_gpu.py -m gpu -n 0
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock import fetch, models
from warlock.config import get_config
from warlock.pipelines import lora_train

#: One checkpoint load plus a hundred training steps at 1024 with gradient
#: checkpointing, then a second load to verify. Well inside ``lora_train``'s
#: own four-hour child deadline, and a hang net rather than a budget.
pytestmark = [pytest.mark.gpu, pytest.mark.timeout(3600)]

#: The floor the door enforces. Three is what ``MIN_IMAGES`` documents as the
#: point a rank-16 adapter learns a style rather than memorising a picture --
#: and three synthetic squares learn neither, which is fine: see the docstring.
IMAGES = 3

TRIGGER = "wlktest style"


@pytest.fixture(scope="module")
def base_dir():
    """The shipped default checkpoint's directory, or a skip that names it."""
    config = get_config()
    spec = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    # ``fetch.base_model_dir`` and not a hand-built path: it is what
    # ``_q_lora`` hands to ``train_spec``, so this trains against the same
    # directory a real job would.
    path = fetch.base_model_dir(config, spec)
    if not (path / "model_index.json").exists():
        pytest.skip(f"{spec.label} weights not downloaded")
    return path


@pytest.fixture(scope="module")
def training_images(tmp_path_factory):
    """Three deterministic, *distinguishable* images.

    Not noise: a loop that silently trains on nothing would converge just as
    happily on noise, and three flat colours give the loss something it can
    actually reduce. Deterministic so a re-run of this lane trains on the same
    pictures and any difference in outcome is the code's.

    Written under ``tmp_path_factory`` and nowhere near the user's library.
    **This lane sees the real ``~/.warlock``** -- it is exempt from
    ``conftest.py``'s ``WARLOCK_HOME`` pinning because it has to resolve real
    weights -- so every path this file writes is a temporary one.
    """
    scratch = tmp_path_factory.mktemp("lora-train-images")
    paths = []
    for index, colour in enumerate(((200, 60, 40), (40, 160, 200), (230, 200, 70))):
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        canvas[:, :] = colour
        # One off-colour band, so the three are not degenerate single-value
        # tensors -- a VAE encode of a perfectly flat image is a corner the
        # loop should not be measured in.
        canvas[128:384, 128:384] = tuple(255 - c for c in colour)
        path = scratch / f"train-{index:02d}.png"
        Image.fromarray(canvas).save(path)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def trained(base_dir, training_images, tmp_path_factory):
    """One real training run. -> (the adapter directory, the result payload)."""
    from warlock import rigging

    work = tmp_path_factory.mktemp("lora-train-out")
    out_dir = work / "adapter"
    spec = lora_train.train_spec(
        base_dir,
        training_images,
        out_dir,
        work,
        trigger=TRIGGER,
        steps=lora_train.MIN_STEPS,
        seed=0,
    )
    result = rigging.run_worker(
        spec,
        module="warlock.pipelines.lora_train_worker",
        marker=lora_train.MARKER,
        name="LoRA trainer",
        timeout=lora_train.TIMEOUT,
    )
    return out_dir, result


def test_the_child_trains_and_reports_ok(trained):
    """The contract end to end: spec on stdin, result JSON at ``result_path``.

    Everything that can go wrong before the first step -- a missing dependency
    in the child's environment, an unreadable checkpoint, a spec key the worker
    does not know -- surfaces here, and surfaces on a user's card as a job that
    fails after the weights have already loaded.
    """
    _out_dir, result = trained
    assert result["ok"] is True


def test_the_weights_land_under_the_name_the_loader_expects(trained):
    """``WEIGHTS_NAME`` in diffusers' own layout, so no conversion step exists.

    The module's argument for that filename is that ``load_lora_weights`` reads
    it with no conversion. A file written under any other name is a file the
    picker registers and the pipeline cannot open.
    """
    out_dir, _result = trained
    weights = out_dir / lora_train.WEIGHTS_NAME
    assert weights.is_file(), f"no {lora_train.WEIGHTS_NAME} under {out_dir}"
    assert weights.stat().st_size > 0


def test_the_adapter_is_real_weights_and_not_merely_a_file(trained):
    """Every tensor finite, and the rank is the one that was asked for.

    A diverged run writes NaN or inf and the file is otherwise perfect: right
    name, right layout, plausible size. ``TRAINED_WEIGHT`` is 1.0 precisely
    *because* rank 16 with ``lora_alpha == rank`` makes the adapter's strength
    known by construction -- which is only true if the rank on disk is 16.
    """
    safetensors = pytest.importorskip("safetensors.torch")

    out_dir, _result = trained
    tensors = safetensors.load_file(out_dir / lora_train.WEIGHTS_NAME)
    assert tensors, "the adapter file holds no tensors"

    import torch

    for name, tensor in tensors.items():
        assert torch.isfinite(tensor).all(), f"{name} carries NaN or inf"

    # ``lora.down``/``lora.up`` and not PEFT's ``lora_A``/``lora_B``: the worker
    # writes through ``StableDiffusionXLPipeline.save_lora_weights``, whose whole
    # job is to convert the PEFT names it is handed into the diffusers ones, and
    # ``load_lora_weights`` converts them back on the way in. Asserting the PEFT
    # spelling here would be asserting that the worker skipped the conversion
    # every other diffusers consumer expects -- see the load below, which is
    # what actually settles whether the names are the right ones.
    ranks = {
        tuple(t.shape)[0] for name, t in tensors.items() if name.endswith("lora.down.weight")
    }
    assert ranks, (
        f"no lora.down tensors: this is not a LoRA in diffusers' layout. "
        f"Keys present: {sorted(tensors)[:3]}"
    )
    assert ranks == {lora_train.RANK}, f"rank on disk is {ranks}, asked for {lora_train.RANK}"


def test_a_real_pipeline_loads_it(trained, base_dir):
    """The assertion the whole file exists for.

    Loading is the only check that spans every convention at once -- key
    naming, module targeting, shapes against *this* UNet. It is also exactly
    what happens on the user's next generation after they finish training, so a
    failure here is a failure they would have hit first.

    No image is sampled: a picture would need a human to say whether the style
    took, which is `TODO.md`'s work and not a test's.
    """
    pytest.importorskip("diffusers")
    import torch
    from diffusers import StableDiffusionXLPipeline

    out_dir, _result = trained
    # ``variant`` is not optional and its absence is not a detail: the shipped
    # checkpoint holds ``model.fp16.safetensors`` and no unsuffixed sibling, so
    # a bare from_pretrained raises "no file named model.safetensors" on the
    # text encoder and never reaches the adapter at all. Text2Image passes
    # ``spec.variant``; passing anything else here would be testing a load the
    # app does not do.
    spec = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_dir, torch_dtype=torch.float16, use_safetensors=True, variant=spec.variant
    )
    try:
        pipe.load_lora_weights(out_dir, weight_name=lora_train.WEIGHTS_NAME)
        # Loading without raising is not the assertion -- ``load_lora_weights``
        # logs and continues for a prefix it recognises no keys under, which is
        # exactly how a mis-named adapter presents. These are the three things
        # a user would otherwise discover as "the style did nothing".
        adapted = {
            name: param
            for name, param in pipe.unet.named_parameters()
            if "lora_A" in name or "lora_B" in name
        }
        assert adapted, "the adapter loaded no parameters onto the UNet"
        down = [p for name, p in adapted.items() if "lora_A" in name]
        assert {tuple(p.shape)[0] for p in down} == {lora_train.RANK}
        assert all(torch.isfinite(p).all() for p in adapted.values())
        # ``lora_B`` is zero-initialised, so a loop that trained on nothing --
        # a detached loss, a zeroed gradient, an optimiser over the wrong
        # parameter set -- writes a perfectly well-formed adapter that is
        # exactly the identity. This is the one assertion in the file that
        # distinguishes a trained adapter from an untrained one.
        up = [p for name, p in adapted.items() if "lora_B" in name]
        assert up and all(float(p.detach().abs().max()) > 0 for p in up), (
            "every lora_B matrix is still zero: the adapter is the identity and "
            "the training loop moved nothing"
        )
    finally:
        del pipe
        torch.cuda.empty_cache()
