"""Does this image actually tile?

A ratio, not an absolute difference: a busy texture legitimately differs a lot
between any two adjacent columns, so the only meaningful question is whether
the wrap seam differs *more* than the interior does. That normalisation is
what makes one threshold work for cobblestone and for flat plaster alike.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import seam


def _gradient(width=64, height=64, wrap=False):
    """A left-to-right ramp. Wrapped, it is a triangle wave and tiles."""
    x = np.arange(width)
    row = np.abs((x * 2.0 / width) - 1.0) if wrap else x / width
    arr = np.tile((row * 255).astype(np.uint8), (height, 1))
    return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")


def test_a_seamless_image_scores_near_the_interior(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=True).save(path)
    out = seam.report(path)
    assert out["horizontal"] < seam.SEAM_MAX
    assert out["seamless"] is True


def test_a_hard_seam_is_caught(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=False).save(path)
    out = seam.report(path)
    assert out["horizontal"] > seam.SEAM_MAX
    assert out["seamless"] is False


def test_both_axes_are_measured(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=False).transpose(Image.ROTATE_90).save(path)
    out = seam.report(path)
    assert out["vertical"] > seam.SEAM_MAX
    assert out["horizontal"] < seam.SEAM_MAX


def test_the_worst_axis_decides_the_verdict(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=False).save(path)
    out = seam.report(path)
    assert out["worst"] == max(out["horizontal"], out["vertical"])


def test_a_flat_image_is_seamless_and_says_so_without_dividing_by_zero(tmp_path):
    path = tmp_path / "tile.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)
    out = seam.report(path)
    assert out["seamless"] is True
    assert out["horizontal"] == 0.0


def test_an_almost_flat_image_with_one_hard_join_is_not_called_seamless(tmp_path):
    # The case the flat-image guard must not swallow. Two blocks of colour have
    # almost no grain, so the interior mean is small -- and a normalisation
    # that clamped a small denominator to "seamless" would call a picture with
    # a 160-level step across its wrap edge a tile.
    arr = np.full((32, 32, 3), 40, dtype=np.uint8)
    arr[:, 16:] = 200
    Image.fromarray(arr, "RGB").save(path := tmp_path / "tile.png")
    out = seam.report(path)
    assert out["seamless"] is False
    assert out["horizontal"] > seam.SEAM_MAX


def test_the_wrap_preview_rolls_by_half(tmp_path):
    src, dest = tmp_path / "tile.png", tmp_path / "preview.png"
    image = _gradient(wrap=False)
    image.save(src)

    seam.wrap_preview(src, dest)

    with Image.open(dest) as out:
        rolled = np.asarray(out.convert("RGB"))
    expected = np.roll(np.asarray(image), (32, 32), axis=(0, 1))
    assert np.array_equal(rolled, expected)


def test_the_wrap_preview_keeps_alpha(tmp_path):
    # A tile is a texture and a texture can carry an alpha channel; dropping it
    # in the preview would make a transparent tile look like it had a black
    # background, which is a failure the user would try to fix in the generator.
    src, dest = tmp_path / "tile.png", tmp_path / "preview.png"
    Image.new("RGBA", (16, 16), (10, 20, 30, 40)).save(src)

    seam.wrap_preview(src, dest)

    with Image.open(dest) as out:
        assert out.mode == "RGBA"


def test_a_tiny_image_is_refused_rather_than_measured(tmp_path):
    path = tmp_path / "tile.png"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(ValueError):
        seam.report(path)
