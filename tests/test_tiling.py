"""Circular padding: on for one sample, off again afterwards.

The restore is the part worth testing. The pipe is resident and shared, so a
patch that leaked would make every later job tile -- silently, and only
visibly at the edges of an image nobody looks at the edges of.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from warlock import models  # noqa: E402
from warlock.pipelines import text2image  # noqa: E402
from warlock.pipelines.prompt import (  # noqa: E402
    PROMPT_TEMPLATE,
    TILE_TEMPLATE,
)


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.inner = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1))
        self.linear = torch.nn.Linear(4, 4)


def _modes(net):
    return [m.padding_mode for m in net.modules() if isinstance(m, torch.nn.Conv2d)]


def test_every_conv_is_switched_inside_the_block():
    net = _Net()
    with text2image.circular_padding(net):
        assert _modes(net) == ["circular", "circular"]


def test_every_conv_is_restored_afterwards():
    net = _Net()
    before = _modes(net)
    with text2image.circular_padding(net):
        pass
    assert _modes(net) == before


def test_the_restore_survives_an_exception():
    net = _Net()
    before = _modes(net)
    with pytest.raises(RuntimeError), text2image.circular_padding(net):
        raise RuntimeError("boom")
    assert _modes(net) == before


def test_a_conv_that_was_already_circular_stays_circular():
    # Restored to what each module *was*, not to a blanket "zeros": a model
    # that ships a circular conv must come back the way it arrived.
    net = _Net()
    net.a.padding_mode = "circular"
    with text2image.circular_padding(net):
        pass
    assert net.a.padding_mode == "circular"


def test_nothing_but_convs_is_touched():
    net = _Net()
    with text2image.circular_padding(net):
        assert not hasattr(net.linear, "padding_mode")


def test_none_modules_are_skipped_rather_than_crashing():
    # A pipeline component can legitimately be absent (a VAE-less pipe in a
    # test, a future spec without one), and the context manager is the wrong
    # place to discover that.
    with text2image.circular_padding(None, _Net()):
        pass


def test_a_conv_reached_through_two_modules_is_still_restored_once():
    # The leak the finally exists to prevent, arrived at from the other
    # direction: if the same child were recorded twice, the second record
    # would capture the *patched* "circular" and restoring it last would
    # leave the shared conv tiled for every later job.
    shared = torch.nn.Conv2d(3, 3, 3, padding=1)
    one = torch.nn.Sequential(shared)
    two = torch.nn.Sequential(shared)
    with text2image.circular_padding(one, two):
        assert shared.padding_mode == "circular"
    assert shared.padding_mode == "zeros"


def test_the_patch_actually_wraps_the_receptive_field():
    # The claim the whole feature rests on, checked against torch rather than
    # assumed: with circular padding, a conv sees the far edge as adjacent.
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    torch.nn.init.constant_(conv.weight, 0.0)
    with torch.no_grad():
        conv.weight[0, 0, 1, 0] = 1.0  # take the pixel to the left
    x = torch.zeros(1, 1, 1, 4)
    x[0, 0, 0, 3] = 1.0

    zeros = conv(x)
    with text2image.circular_padding(conv):
        circular = conv(x)

    assert zeros[0, 0, 0, 0].item() == 0.0
    assert circular[0, 0, 0, 0].item() == pytest.approx(1.0)


# --- generate(): the patch is per call, and the unpatched path never enters it ---


class _StubImage:
    def save(self, path) -> None:
        Path(path).write_bytes(b"stub-png")


class _StubPipe:
    """The narrowest thing generate() will run against -- no diffusers, no CUDA.

    It records the padding mode of every conv *at the moment it is called*,
    because that moment is the only one that decides whether the sample tiles.
    Asserting on the modes before or after the call would prove nothing about
    the pixels.
    """

    def __init__(self) -> None:
        self.unet = _Net()
        self.vae = _Net()
        self.tokenizer = None
        self.tokenizer_2 = None
        self.scheduler = object()
        self.modes_during_call: list[str] = []
        self.raises: Exception | None = None
        # Only set by the conditioning test. Absent by default so every other
        # test's expected mode list is unchanged.
        self.controlnet = None

    def disable_lora(self) -> None:
        pass

    def __call__(self, **_kwargs):
        self.modes_during_call = _modes(self.unet) + _modes(self.vae)
        if self.controlnet is not None:
            self.modes_during_call += _modes(self.controlnet)
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(images=[_StubImage()])


@pytest.fixture
def stub_t2i(monkeypatch, tmp_path):
    spec = models.BaseModel(
        key="stub",
        label="Stub",
        dir_name="stub",
        image_size=64,
        steps=1,
        guidance_scale=0.0,
    )
    t2i = text2image.Text2Image(spec, tmp_path)
    pipe = _StubPipe()
    t2i._pipe = pipe
    monkeypatch.setattr(t2i, "load", lambda on_state=None: None)
    monkeypatch.setattr(text2image, "chunk", lambda text, tokenizers: [text])
    monkeypatch.setattr(text2image, "_encode_long_prompt", lambda p, chunks: (None, None))
    # The seed generator is the one line of generate() that would otherwise
    # want a CUDA context; nothing here is testing torch's RNG.
    monkeypatch.setattr(
        torch, "Generator", lambda device=None: SimpleNamespace(manual_seed=lambda s: None)
    )
    return t2i, pipe


def test_a_job_that_asks_for_no_tiling_never_enters_the_patch(stub_t2i, monkeypatch, tmp_path):
    # The Global Constraint states both halves: an untiled job must not merely
    # revert correctly, it must not reach the patch function at all.
    t2i, pipe = stub_t2i

    def _forbidden(*_modules):
        raise AssertionError("circular_padding reached on the untiled path")

    monkeypatch.setattr(text2image, "circular_padding", _forbidden)
    t2i.generate("a wooden crate", tmp_path / "a.png")

    assert pipe.modes_during_call == ["zeros"] * 4
    assert t2i.last_prompt == PROMPT_TEMPLATE.format(prompt="a wooden crate")
    assert t2i.last_recipe["tile"] is False


def test_tile_patches_the_unet_and_the_vae_for_that_call_only(stub_t2i, tmp_path):
    t2i, pipe = stub_t2i
    t2i.generate("cracked stone", tmp_path / "a.png", tile=True)

    assert pipe.modes_during_call == ["circular"] * 4
    assert _modes(pipe.unet) + _modes(pipe.vae) == ["zeros"] * 4
    assert t2i.last_prompt == TILE_TEMPLATE.format(prompt="cracked stone")
    assert t2i.last_recipe["tile"] is True


def test_a_tiled_generation_that_raises_leaves_the_next_one_untiled(stub_t2i, tmp_path):
    # The regression that is invisible without looking for it: the pipe is
    # resident and shared, so a patch stranded by an exception would make every
    # later ordinary reference tile at its edges, silently.
    t2i, pipe = stub_t2i
    pipe.raises = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        t2i.generate("cracked stone", tmp_path / "a.png", tile=True)
    assert _modes(pipe.unet) + _modes(pipe.vae) == ["zeros"] * 4

    pipe.raises = None
    t2i.generate("a wooden crate", tmp_path / "b.png")
    assert pipe.modes_during_call == ["zeros"] * 4


def test_an_attached_controlnet_is_patched_too(stub_t2i, monkeypatch, tmp_path):
    # The hint branch is a separate module tree whose residuals are added into
    # the UNet at every block, so leaving it zero-padded contributes a seam to
    # a sample whose every other convolution wraps. The alternative -- refusing
    # tile+Structure at the door -- would remove a legitimate combination
    # (a tiling brick wall guided by a depth hint) to avoid a one-word fix.
    t2i, pipe = stub_t2i
    pipe.controlnet = _Net()
    monkeypatch.setattr(t2i, "_conditioned", lambda cond: (pipe, {}, lambda: None))
    cond = SimpleNamespace(as_dict=lambda: {})

    t2i.generate("brick wall", tmp_path / "a.png", conditioning=cond, tile=True)

    # unet, vae, then the ControlNet: six convs, all wrapping.
    assert pipe.modes_during_call == ["circular"] * 6
    # And restored with the rest -- the ControlNet is dropped by teardown, but
    # a shared conv reached through it must not be left tiled.
    assert _modes(pipe.controlnet) == ["zeros", "zeros"]
