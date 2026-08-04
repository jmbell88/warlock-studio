"""Structure hints. Every fixture is Pillow-drawn: no GPU, no weights."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from warlock.pipelines import control


def _square(size=128, fill=(255, 255, 255), bg=(0, 0, 0)):
    im = Image.new("RGB", (size, size), bg)
    ImageDraw.Draw(im).rectangle([32, 32, 95, 95], fill=fill)
    return im


def test_canny_marks_the_boundary_and_nothing_else():
    edges = control.canny(_square())
    px = edges.load()
    assert px[64, 32] != (0, 0, 0) or px[64, 31] != (0, 0, 0)  # top edge
    assert px[64, 64] == (0, 0, 0)  # interior is flat
    assert px[8, 8] == (0, 0, 0)  # background is flat


def test_a_flat_image_produces_no_edges():
    flat = Image.new("RGB", (64, 64), (128, 128, 128))
    assert control.edge_fraction(control.canny(flat)) == 0.0


def test_hint_is_rgb_at_exactly_the_requested_size():
    out = control.hint("canny", _square(size=200), size=1024)
    assert out.mode == "RGB"
    assert out.size == (1024, 1024)


def test_hint_is_deterministic():
    a = control.hint("canny", _square(), size=256)
    b = control.hint("canny", _square(), size=256)
    assert a.tobytes() == b.tobytes()


def test_letterbox_pads_rather_than_crops():
    wide = Image.new("RGB", (200, 50), (0, 0, 0))
    ImageDraw.Draw(wide).rectangle([20, 10, 180, 40], fill=(255, 255, 255))
    out = control.hint("canny", wide, size=128)
    assert out.size == (128, 128)
    # The pad is black, so the top row carries no structure.
    assert out.crop((0, 0, 128, 4)).getbbox() is None


def test_edge_fraction_is_a_sane_share():
    frac = control.edge_fraction(control.canny(_square()))
    assert 0.0 < frac < 0.2


def test_depth_is_not_this_modules_job():
    # It needs a torch model; keeping it out is what keeps control.py
    # importable without the text2image extra.
    with pytest.raises(ValueError, match="depth"):
        control.hint("depth", _square(), size=512)


def test_write_hint_lands_atomically_with_provenance(tmp_path):
    src = tmp_path / "ref.png"
    _square().save(src)
    dest = tmp_path / "control.png"
    meta = control.write_hint(src, dest, kind="canny", size=512)

    assert dest.exists()
    assert not list(tmp_path.glob(".*tmp*")), "the staging name must not survive"
    assert meta["kind"] == "canny"
    assert meta["size"] == 512
    assert meta["low"] == control.DEFAULT_LOW
    assert 0.0 < meta["edge_fraction"] < 1.0
    with Image.open(dest) as im:
        assert im.size == (512, 512)
