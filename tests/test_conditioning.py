"""The conditioning value object, and which pipeline class each shape selects.

No GPU: the dataclass is deliberately inert, and the routing under test is a
choice between diffusers classes made *before* any of them is called -- which
is exactly where it has gone wrong before, because diffusers accepts both
spellings of the img2img kwargs without complaint and produces a traced copy of
the wrong picture from the wrong one.

The two invariants that need a real card live in test_conditioning_gpu.py.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from warlock.pipelines.conditioning import Conditioning


def test_an_empty_conditioning_is_falsy():
    """The bit-identity gate. A falsy Conditioning never reaches _conditioned,
    which is what makes the unconditioned path byte-identical to the app before
    any of this existed."""
    assert not Conditioning()
    assert not Conditioning(ip_scale=0.9, control_scale=1.2, strength=0.6)


def test_each_axis_makes_it_truthy_on_its_own(tmp_path):
    png = tmp_path / "x.png"
    assert bool(Conditioning(init_image=png)) is True
    assert bool(Conditioning(control="canny", control_image=png)) is True
    assert bool(Conditioning(ip_adapter="plus", ip_image=png)) is True


def test_uses_init_is_the_presence_of_a_picture_not_a_strength():
    # strength has a default, so deriving "is this img2img" from it would make
    # every ordinary job an img2img job.
    assert not Conditioning(strength=0.6).uses_init
    assert Conditioning(init_image=Path("a.png")).uses_init


def test_as_dict_records_only_the_halves_in_play(tmp_path):
    png = tmp_path / "init.png"
    assert Conditioning().as_dict() == {}
    recorded = Conditioning(init_image=png, strength=0.5).as_dict()
    assert recorded == {"init_image": "init.png", "strength": 0.5}
    # A name, never a path: the recipe is read back on another machine.
    assert "/" not in recorded["init_image"]


# --- which pipeline class runs ------------------------------------------------


class _FakePipe:
    """Enough of a diffusers pipeline for _conditioned to route against."""

    def __init__(self):
        self.unet = types.SimpleNamespace(dtype="bfloat16")
        self.scheduler = object()


@pytest.fixture
def routed(monkeypatch, tmp_path):
    """Capture which pipeline class _conditioned picks, without diffusers."""
    from warlock import models
    from warlock.pipelines import text2image

    picked: dict[str, str] = {}

    def fake_class(name):
        class Fake:
            @staticmethod
            def from_pipe(pipe, **kwargs):
                picked["class"] = name
                out = _FakePipe()
                out.controlnet = kwargs.get("controlnet")
                return out

        return Fake

    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )
    fake_diffusers = types.SimpleNamespace(
        ControlNetModel=types.SimpleNamespace(
            from_pretrained=lambda *a, **k: types.SimpleNamespace(to=lambda _d: object())
        ),
        StableDiffusionXLControlNetPipeline=fake_class("controlnet"),
        StableDiffusionXLControlNetImg2ImgPipeline=fake_class("controlnet_img2img"),
        StableDiffusionXLImg2ImgPipeline=fake_class("img2img"),
        # The PAG variants, importable even though a pag_scale=0 spec never
        # picks one -- the imports sit beside their plain siblings.
        StableDiffusionXLControlNetPAGPipeline=fake_class("controlnet_pag"),
        StableDiffusionXLControlNetPAGImg2ImgPipeline=fake_class("controlnet_pag_img2img"),
        StableDiffusionXLPAGImg2ImgPipeline=fake_class("pag_img2img"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    # A ControlNet's weights are checked before any torch work, by design.
    root = tmp_path / models.CONTROLNETS["canny"].dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")

    t2i = text2image.Text2Image(models.BASE_MODELS["sdxl_cfg"], tmp_path)
    t2i._pipe = _FakePipe()

    def run(cond):
        picked.clear()
        target, extra, teardown = t2i._conditioned(cond)
        teardown()
        return picked.get("class"), extra

    return run


def _png(tmp_path, name):
    path = tmp_path / name
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return path


def test_an_init_image_alone_selects_the_plain_img2img_pipeline(routed, tmp_path):
    picked, extra = routed(Conditioning(init_image=_png(tmp_path, "init.png"), strength=0.5))
    assert picked == "img2img"
    assert extra["strength"] == 0.5
    assert extra["image"].size == (8, 8)


def test_a_control_hint_alone_keeps_the_controlnet_pipeline(routed, tmp_path):
    picked, extra = routed(
        Conditioning(control="canny", control_image=_png(tmp_path, "hint.png"))
    )
    assert picked == "controlnet"
    # On a ControlNet-only pipeline the hint *is* ``image``.
    assert "image" in extra and "control_image" not in extra


def test_control_plus_init_swaps_the_two_image_kwargs(routed, tmp_path):
    """The routing that is easy to get exactly backwards, and which diffusers
    accepts either way round: with an init picture, ``image`` is what is being
    denoised and the hint is demoted to ``control_image``."""
    init = _png(tmp_path, "init.png")
    hint = _png(tmp_path, "hint.png")
    picked, extra = routed(
        Conditioning(init_image=init, control="canny", control_image=hint, strength=0.45)
    )
    assert picked == "controlnet_img2img"
    assert extra["strength"] == 0.45
    assert set(extra) >= {"image", "control_image", "controlnet_conditioning_scale"}
    assert extra["image"] is not extra["control_image"]


# --- the frame an init picture is denoised at ---------------------------------


@pytest.fixture
def framed(tmp_path):
    """``_init_frame`` on a real Text2Image, with no diffusers and no card.

    The method is a pure choice between two sizes, and the choice is the whole
    bug: it is made before any pipeline is called, and both classes that get it
    wrong accept the wrong answer silently.
    """
    from warlock import models
    from warlock.pipelines import text2image

    return text2image.Text2Image(models.BASE_MODELS["sdxl_cfg"], tmp_path)


def _sized(width, height):
    return {"image": Image.new("RGB", (width, height), (10, 20, 30))}


def test_no_conditioning_still_takes_the_specs_square_frame(framed):
    """The bit-identity gate, restated as a size: an unconditioned job's frame
    may not move because a conditioned one's did."""
    assert framed._init_frame(None, {}, None) == (1024, 1024)
    assert framed._init_frame(None, {}, (768, 512)) == (768, 512)


def test_an_init_picture_is_denoised_at_its_own_aspect(framed):
    """The defect. ``inpaint.send_size`` sends a wide selection as 1024x448;
    the inpaint and ControlNet-img2img classes honour width/height by resizing
    the init image and the mask to them, so a square frame stretched the crop,
    denoised the distortion and let ``fit_back`` squash it back into the box."""
    cond = Conditioning(init_image=Path("init.png"), strength=0.6)
    assert framed._init_frame(cond, _sized(1024, 448), None) == (1024, 448)


def test_a_mask_does_not_change_the_frame_rule(framed):
    """Inpainting is the path the bug was found on, and it is not a special
    case -- the mask is resized to the same frame as the picture it masks."""
    cond = Conditioning(
        init_image=Path("init.png"), mask_image=Path("mask.png"), strength=0.6
    )
    assert framed._init_frame(cond, _sized(448, 1024), None) == (448, 1024)


def test_a_conditioning_without_a_picture_leaves_the_frame_alone(framed):
    """A ControlNet hint is ``extra["image"]`` too, and it is *not* a picture
    being denoised -- reading the frame off it would resize the output to the
    hint. Only ``uses_init`` moves the frame."""
    cond = Conditioning(control="canny", control_image=Path("hint.png"))
    assert framed._init_frame(cond, _sized(512, 512), None) == (1024, 1024)


def test_an_unaligned_init_picture_is_snapped_to_the_vae_stride(framed):
    """Nothing in the tree sends one, and ``_frame``'s docstring says what a
    side that is not a multiple of 8 costs: the VAE rounds it and the caller
    slices the returned image on a grid it no longer has."""
    cond = Conditioning(init_image=Path("init.png"), strength=0.6)
    assert framed._init_frame(cond, _sized(1000, 445), None) == (1000, 448)
    assert framed._init_frame(cond, _sized(3, 3), None) == (8, 8)
