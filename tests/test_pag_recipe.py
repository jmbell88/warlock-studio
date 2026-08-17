"""The sdxl_cfg_pag recipe: opt-in sampling upgrades over unchanged weights.

What the default lane can pin without a card: the registry facts, the
absence-not-default rule that keeps every other recipe's path byte-identical,
and the loader's refusal of PAG on a non-SDXL architecture. The actual PAG
sample runs in the gpu lane.
"""

from __future__ import annotations

import dataclasses

import pytest

from warlock import fetch, models
from warlock.config import Config


def test_the_pag_row_is_a_recipe_not_a_download():
    spec = models.BASE_MODELS["sdxl_cfg_pag"]
    base = models.BASE_MODELS["sdxl_cfg"]
    assert spec.dir_name == base.dir_name
    assert spec.fetch == base.fetch  # the same _SDXL_BASE_1_0 record: one 7 GB
    assert spec.steps == base.steps and spec.guidance_scale == base.guidance_scale
    assert spec.pag_scale > 0 and spec.guidance_rescale > 0


def test_the_shared_checkpoint_still_plans_as_one_download(svc):
    entries = [
        e for e in fetch.entries() if e.row_key in ("base:sdxl_cfg", "base:sdxl_cfg_pag")
    ]
    assert len(entries) == 2
    assert len(fetch.plan(svc.config, entries)) == 1


def test_the_pag_row_joins_every_capability_list():
    assert "sdxl_cfg_pag" in models.cfg_bases()
    assert "sdxl_cfg_pag" in models.controlnet_bases()
    assert "sdxl_cfg_pag" in models.tile_bases()
    assert "sdxl_cfg_pag" in models.lora_bases()


def test_every_other_recipe_keeps_the_upgrades_off():
    """The bit-identity guard: pag_scale/guidance_rescale are only ever passed
    when set, so 0.0 everywhere else is what keeps every pinned output pinned.
    A new recipe that wants them gets its own row, as this one did."""
    for key, spec in models.BASE_MODELS.items():
        if key == "sdxl_cfg_pag":
            continue
        assert spec.pag_scale == 0.0, key
        assert spec.guidance_rescale == 0.0, key


def test_the_loader_refuses_pag_on_a_non_sdxl_family(tmp_path):
    from warlock.pipelines.text2image import Text2Image

    wrong = dataclasses.replace(
        models.BASE_MODELS["flux_klein"], pag_scale=3.0, dir_name="fake-flux"
    )
    (tmp_path / "fake-flux").mkdir()
    (tmp_path / "fake-flux" / "model_index.json").write_text("{}")
    t2i = Text2Image(wrong, tmp_path)
    with pytest.raises(RuntimeError, match="PAG"):
        t2i.load()


@pytest.mark.gpu
def test_pag_recipe_loads_and_samples():
    """Real weights, real card: the PAG pipeline class actually resolves, the
    pag_scale/guidance_rescale kwargs are accepted, and an image lands.

    A four-step 512 replica of the recipe, because the point is the class
    mechanics rather than the picture -- the full 30-step arm is the bench's
    business."""
    import tempfile
    from pathlib import Path

    config = Config()
    spec = dataclasses.replace(
        models.BASE_MODELS["sdxl_cfg_pag"], steps=4, image_size=512
    )
    if not (config.t2i_model_root / spec.dir_name / "model_index.json").exists():
        pytest.skip("sdxl-base-1.0 weights not downloaded")
    from warlock.pipelines.text2image import Text2Image

    t2i = Text2Image(spec, config.t2i_model_root)
    try:
        t2i.load()
        assert "PAG" in type(t2i._pipe).__name__
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.png"
            t2i.generate("a wooden crate", out, seed=7)
            assert out.exists() and out.stat().st_size > 0
        assert t2i.last_recipe["pag_scale"] == spec.pag_scale
        assert t2i.last_recipe["guidance_rescale"] == spec.guidance_rescale
    finally:
        t2i.unload()
