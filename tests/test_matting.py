"""The matte: a model when the weights are there, a flood fill when not.

The fallback is the interesting half -- it is what makes every 2D export work
on a fresh checkout, and it must be indistinguishable in *shape* from the model
path so nothing downstream has to know which one ran.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from warlock import models
from warlock.pipelines import matting


def _config(tmp_path):
    return SimpleNamespace(t2i_model_root=tmp_path)


def _subject():
    im = Image.new("RGB", (96, 96), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((24, 24, 72, 72), fill=(30, 30, 30))
    return im


def test_the_registry_entry_carries_a_download_command():
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    assert spec.dir_name and "hf download" in spec.download


def test_no_weights_means_not_available(tmp_path):
    assert matting.available(_config(tmp_path)) is False


def test_weights_on_disk_mean_available(tmp_path):
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    assert matting.available(_config(tmp_path)) is True


def test_without_weights_the_flood_fill_produces_the_mask(tmp_path):
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.dtype == bool
    assert mask.shape == (96, 96)
    assert mask[48, 48] and not mask[2, 2]


def test_an_image_with_alpha_uses_it_whatever_the_weights_say(tmp_path):
    # subject_mask already prefers a real alpha channel, and a matting model
    # asked to re-cut an existing cutout can only make it worse.
    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((8, 8, 24, 24), fill=(255, 0, 0, 255))
    mask, source = matting.mask(im, _config(tmp_path))
    assert source == "alpha"
    assert mask[16, 16] and not mask[0, 0]


def test_a_failing_model_falls_back_rather_than_raising(tmp_path, monkeypatch):
    # A corrupt or half-downloaded checkpoint must cost the user edge quality,
    # not the export: the flood fill is always there.
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        matting, "_model_mask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.any()


def test_the_model_path_is_used_when_it_works(tmp_path, monkeypatch):
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    fake = np.zeros((96, 96), dtype=bool)
    fake[10:20, 10:20] = True
    monkeypatch.setattr(matting, "_model_mask", lambda image, path, device: fake)

    mask, source = matting.mask(_subject(), _config(tmp_path))

    assert source == "birefnet"
    assert np.array_equal(mask, fake)


def test_a_model_mask_that_finds_nothing_falls_back(tmp_path, monkeypatch):
    # An all-false matte would make every export raise NoSubject. The flood
    # fill's answer is worse-looking and right, which beats correct and empty.
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        matting, "_model_mask", lambda image, path, device: np.zeros((96, 96), dtype=bool)
    )
    _mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
