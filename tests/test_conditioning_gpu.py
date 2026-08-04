"""The two conditioning invariants no CPU test can prove.

Both need a real diffusers pipeline on a real card, so they carry the gpu
marker and are deselected by default (`-m "not gpu"`). They exist because the
whole conditioning design rests on one assumption: that attaching an adapter
to the resident pipe and then detaching it leaves that pipe exactly as it was.

Run with: uv run pytest tests/test_conditioning_gpu.py -m gpu
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from warlock import models
from warlock.config import get_config
from warlock.pipelines.conditioning import Conditioning
from warlock.pipelines.text2image import Text2Image

pytestmark = pytest.mark.gpu


def _reference(path):
    im = Image.new("RGB", (512, 512), (200, 200, 200))
    ImageDraw.Draw(im).rectangle([120, 120, 391, 391], fill=(60, 100, 170))
    im.save(path)
    return path


@pytest.fixture(scope="module")
def pipe():
    config = get_config()
    spec = models.BASE_MODELS["sdxl_cfg"]
    t2i = Text2Image(spec, config.t2i_model_root)
    if not (t2i.model_dir / "model_index.json").exists():
        pytest.skip(f"{spec.label} weights not downloaded")
    yield t2i
    t2i.unload()


def _require(*keys: str) -> None:
    """Skip rather than fail when a conditioning download is missing.

    Every weight here is an optional manual download, the same as the base
    models -- an absent one is a machine that has not fetched it, not a
    regression.
    """
    config = get_config()
    for key in keys:
        if key == "ip":
            spec = models.IP_ADAPTERS["plus"]
            path = config.t2i_model_root / spec.dir_name / spec.subfolder / spec.weight_name
        else:
            cn = models.CONTROLNETS["canny"]
            path = config.t2i_model_root / cn.dir_name / "config.json"
        if not path.exists():
            pytest.skip(f"conditioning weights not downloaded: {path}")


def test_an_unconditioned_run_after_a_conditioned_one_is_byte_identical(pipe, tmp_path):
    """The single most likely failure in the whole design.

    IP-Adapter mutates the UNet's attention processors and from_pipe() shares
    that UNet, so the bit-identity of every later unconditioned job rests
    entirely on unload_ip_adapter() restoring it exactly. If this fails, the
    escape hatch is a full reload in the teardown -- ~20 s per job for a hard
    guarantee.
    """
    _require("ip", "canny")
    ref = _reference(tmp_path / "ref.png")
    hint = _reference(tmp_path / "hint.png")

    before = pipe.generate("a wooden crate", tmp_path / "a.png", seed=7)
    pipe.generate(
        "a wooden crate",
        tmp_path / "cond.png",
        seed=7,
        conditioning=Conditioning(
            ip_adapter="plus", ip_image=ref, control="canny", control_image=hint
        ),
    )
    after = pipe.generate("a wooden crate", tmp_path / "b.png", seed=7)

    assert before.read_bytes() == after.read_bytes()


def test_a_conditioned_run_gives_back_all_of_its_vram(pipe, tmp_path):
    """A ControlNet (~2.5 GB) plus the CLIP-ViT-H encoder (~1.2 GB) is loaded
    per call precisely so it cannot accumulate: two conditioned jobs in a row
    must not leave two ControlNets resident beside trellis."""
    import torch

    _require("ip")
    ref = _reference(tmp_path / "ref.png")
    # Warm the pipe first, so the baseline is "loaded, idle" rather than "not
    # yet loaded" -- otherwise this measures the base model, not the adapter.
    pipe.generate("a wooden crate", tmp_path / "warm.png", seed=7)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    pipe.generate(
        "a wooden crate",
        tmp_path / "cond.png",
        seed=7,
        conditioning=Conditioning(ip_adapter="plus", ip_image=ref),
    )
    torch.cuda.empty_cache()

    # A few MB of allocator noise is expected; a leaked adapter is hundreds.
    assert torch.cuda.memory_allocated() - baseline < 64 * 1024**2
