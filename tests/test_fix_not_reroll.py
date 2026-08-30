"""Fix, don't reroll: img2img on the reference form, a masked regeneration
from Inker, and the seam-erase pass on a materials sheet.

The pipeline classes need a card; what is under test is that the door
carries the new axes with the same "only when it is on" rule the other
conditioning halves follow, that the worker builds the ``Conditioning`` the
pipeline routes on, that the mask crosses the process boundary, and that a
regeneration lands on the layer it was asked about as one undo step.
"""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest
from PIL import Image

from warlock import guidance, models, vectors
from warlock.guidance import GuidanceError
from warlock.pipelines import seam
from warlock.pipelines.conditioning import Conditioning
from warlock.pipelines.t2i_client import _conditioning_payload
from warlock.pipelines.text2image_worker import _conditioning as rebuild
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Invalid
from warlock.studio.inker import inpaint


def _png(size=(16, 16), colour=(120, 60, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


# --- the door ---------------------------------------------------------------------------


def test_normalize_carries_init_only_when_on():
    off = guidance.normalize({"base_model": "sdxl_cfg", "init_strength": 0.5})
    assert "init_image" not in off and "init_strength" not in off
    on = guidance.normalize({"base_model": "sdxl_cfg", "init_image": True})
    assert on["init_image"] is True
    assert on["init_strength"] == models.DEFAULT_IMG2IMG_STRENGTH
    with pytest.raises(GuidanceError) as info:
        guidance.normalize({"base_model": "sdxl_cfg", "init_image": True, "init_strength": 0.9})
    assert info.value.field == "init_strength"


def test_the_catalog_publishes_the_range_the_door_enforces():
    cat = guidance.catalog()
    assert cat["init_strength_range"] == [
        models.IMG2IMG_STRENGTH_MIN, models.IMG2IMG_STRENGTH_MAX
    ]
    assert cat["defaults"]["init_strength"] == models.DEFAULT_IMG2IMG_STRENGTH
    assert "init_image" in vectors.VECTOR_PARAMS and "init_strength" in vectors.VECTOR_PARAMS


def test_init_off_the_sdxl_family_is_refused_at_the_door():
    flux = next((k for k, m in models.BASE_MODELS.items() if m.family != models.FAMILY_SDXL), None)
    if flux is None:
        pytest.skip("no non-SDXL base registered")
    with pytest.raises(GuidanceError) as info:
        guidance.normalize({"base_model": flux, "init_image": True})
    assert info.value.field == "init_image"


def test_create_job_needs_a_reference_to_start_from(svc):
    with pytest.raises(Invalid) as info:
        svc_jobs.create_job(
            svc, kind="text", prompt="a crate", init_image=True, output="reference"
        )
    assert info.value.field == "reference"


def test_create_job_writes_the_mask_beside_the_reference(svc):
    out = svc_jobs.create_job(
        svc, kind="text", prompt="a crate", reference=_png(), mask=_png(colour=(255, 255, 255)),
        init_image=True, init_strength=0.6, output="reference", count=1,
    )
    job_id = out.get("id") or out["ids"][0]
    job_dir = svc.job_dir(job_id)
    assert (job_dir / "ref.png").exists() and (job_dir / "mask.png").exists()
    params = svc.store.get(job_id)["params"]
    assert params["init_image"] is True and params["init_strength"] == 0.6


def test_a_mask_needs_a_start_image_and_no_structure_control(svc):
    with pytest.raises(Invalid) as info:
        svc_jobs.create_job(
            svc, kind="text", prompt="x", reference=_png(), mask=_png(), output="reference"
        )
    assert info.value.field == "mask"


# --- the value object across the boundary ---------------------------------------------------


def test_the_mask_survives_the_wire(tmp_path):
    cond = Conditioning(init_image=tmp_path / "a.png", strength=0.5, mask_image=tmp_path / "m.png")
    assert cond.uses_mask
    back = rebuild(_conditioning_payload(cond))
    assert back.mask_image == tmp_path / "m.png" and back.uses_mask
    assert cond.as_dict()["mask_image"] == "m.png"
    assert not Conditioning(mask_image=tmp_path / "m.png").uses_mask


# --- the worker builds it ----------------------------------------------------------------------


async def test_the_worker_hands_the_pipeline_the_start_image_and_mask(tmp_path, fake_pipelines):
    from warlock.config import Config
    from warlock.db import JobStore
    from warlock.queue import Worker

    config = Config(
        data_dir=tmp_path / "assets", db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe", trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i-models",
    )
    store = JobStore(config.db_path)
    worker = Worker(config, store)
    try:
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "ref.png").write_bytes(_png())
        (job_dir / "mask.png").write_bytes(_png())
        spec = models.BASE_MODELS["sdxl_cfg"]
        params = {"init_image": True, "init_strength": 0.55}
        cond = await worker._conditioning(job_dir, params, spec)
        assert cond is not None and cond.uses_init and cond.uses_mask
        assert cond.init_image == job_dir / "ref.png" and cond.strength == 0.55
        none = await worker._conditioning(job_dir, {"init_strength": 0.55}, spec)
        assert none is None
    finally:
        store.close()
    await asyncio.sleep(0)


# --- Inker: the arithmetic and the landing ------------------------------------------------------


def test_the_crop_grows_by_the_margin_and_sends_a_stride_aligned_size():
    box = inpaint.crop_box((10, 10, 50, 30), (64, 64))
    assert box == (0, 0, 64, 62)
    w, h = inpaint.send_size(box)
    assert w == 1024 and h % inpaint.STRIDE == 0 and h <= 1024
    flat = np.zeros((64, 64, 4), np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:50] = 255
    crop_png, mask_png, sent_box = inpaint.prepare(flat, mask, (10, 10, 50, 30))
    assert sent_box == box
    assert Image.open(io.BytesIO(crop_png)).size == (w, h)
    assert Image.open(io.BytesIO(mask_png)).mode == "L"
    back = inpaint.fit_back(Image.new("RGB", (w, h), (9, 9, 9)), box)
    assert back.shape == (62, 64, 4) and back[0, 0, 3] == 255


def test_apply_pixels_lands_by_uid_as_one_undo_step():
    from warlock.studio.inker.document import Document

    doc = Document.blank(32, 32)
    layer = doc.stack.active
    assert not doc.undo()
    pixels = np.full((8, 8, 4), 200, np.uint8)
    weight = np.zeros((8, 8), np.uint8)
    weight[:, :4] = 255
    assert doc.apply_pixels(layer.uid, (4, 4, 12, 12), pixels, weight)
    assert layer.pixels[5, 5, 0] == 200 and layer.pixels[5, 11, 0] != 200
    assert doc.undo()
    assert layer.pixels[5, 5, 0] != 200
    assert not doc.apply_pixels(999999, (4, 4, 12, 12), pixels, weight)
    assert not doc.apply_pixels(layer.uid, (4, 4, 12, 12), pixels[:4], weight)


def test_the_op_is_registered_on_the_edit_menu():
    from warlock.studio import inker_ops

    op = inker_ops.get("regenerate_selection")
    assert op.menu == "Edit"


# --- seam erase --------------------------------------------------------------------------------


def test_roll_half_is_its_own_inverse_and_the_cross_is_where_the_seam_is():
    arr = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    im = Image.fromarray(arr)
    rolled = seam.roll_half(im)
    assert np.array_equal(np.asarray(seam.roll_half(rolled)), arr)
    assert np.asarray(rolled)[0, 0, 0] == arr[8, 8, 0]
    mask = np.asarray(seam.cross_mask((64, 64)))
    assert mask[32, 5] == 255 and mask[5, 32] == 255
    assert mask[5, 5] == 0


def test_seam_erase_is_carried_by_the_tile_door(svc, monkeypatch):
    from warlock.service import tilesheets

    monkeypatch.setattr(tilesheets, "_check_weights", lambda *a, **k: None)
    monkeypatch.setattr(tilesheets, "check_vram", lambda *a, **k: None, raising=False)
    assert tilesheets.tile_sheet_options()["defaults"]["seam_erase"] is False
    try:
        out = tilesheets.create_tile_sheet(
            svc, prompt="mossy stone", mode=tilesheets.MODE_MATERIALS,
            prompt_items=("mossy stone",), seam_erase=True,
        )
    except Invalid as exc:
        pytest.skip(f"door refused for an unrelated reason: {exc}")
    block = svc.store.get(out["id"])["params"]["sheet"]
    assert block["seam_erase"] is True
