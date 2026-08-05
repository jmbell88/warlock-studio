"""The 2D exports, decided without a model and asserted without a GPU.

Same contract pipelines/sheet.py has: everything about what an icon *is* --
where the subject is trimmed to, how much margin it keeps, where the pivot
sits -- is decided here, so the manifest, the file and the preview can never
disagree, and the whole thing is testable with a rectangle on a grey field.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from warlock.pipelines import asset2d


def _subject(size=(128, 128), box=(32, 20, 96, 100), colour=(200, 30, 30)):
    im = Image.new("RGB", size, (200, 200, 200))
    ImageDraw.Draw(im).rectangle(box, fill=colour)
    mask = np.zeros(size[::-1], dtype=bool)
    mask[box[1] : box[3] + 1, box[0] : box[2] + 1] = True
    return im, mask


def test_the_trim_box_is_the_subjects_own_bounds():
    _im, mask = _subject(box=(32, 20, 96, 100))
    assert asset2d.trim_box(mask) == (32, 20, 97, 101)


def test_an_empty_mask_has_no_trim_box():
    assert asset2d.trim_box(np.zeros((16, 16), dtype=bool)) is None


def test_a_cutout_carries_the_mask_as_alpha():
    im, mask = _subject()
    out = asset2d.cutout(im, mask)
    alpha = np.asarray(out)[:, :, 3]
    assert out.mode == "RGBA"
    assert alpha[60, 60] == 255
    assert alpha[2, 2] == 0


def test_an_icon_is_square_at_the_asked_for_size():
    im, mask = _subject()
    out, meta = asset2d.icon(im, mask, size=256)
    assert out.size == (256, 256)
    assert out.mode == "RGBA"
    assert meta["canvas"] == [256, 256]


def test_an_icon_keeps_the_subjects_aspect_ratio():
    # A tall sword must not come out as a square sword. The subject is fitted
    # inside the canvas, never stretched to it.
    im, mask = _subject(box=(50, 10, 70, 110))
    out, _meta = asset2d.icon(im, mask, size=200, pad=0.0)
    alpha = np.asarray(out)[:, :, 3] > 0
    ys, xs = np.nonzero(alpha)
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    assert height > width * 2


def test_icon_padding_leaves_a_margin_on_the_long_axis():
    im, mask = _subject(box=(10, 10, 110, 110))
    tight, _ = asset2d.icon(im, mask, size=100, pad=0.0)
    padded, _ = asset2d.icon(im, mask, size=100, pad=0.2)

    def extent(image):
        alpha = np.asarray(image)[:, :, 3] > 0
        ys, _xs = np.nonzero(alpha)
        return ys.max() - ys.min() + 1

    assert extent(padded) < extent(tight)


def test_an_icon_records_where_it_trimmed_from():
    im, mask = _subject(box=(32, 20, 96, 100))
    _out, meta = asset2d.icon(im, mask)
    assert meta["trim"] == [32, 20, 97, 101]
    assert meta["source"] == [128, 128]


def test_a_sprite_is_trimmed_to_the_subject_and_nothing_more():
    im, mask = _subject(box=(32, 20, 96, 100))
    out, meta = asset2d.sprite(im, mask)
    assert out.size == (65, 81)
    assert meta["trim"] == [32, 20, 97, 101]


def test_a_sprites_pivot_is_bottom_centre_by_default():
    im, mask = _subject(box=(32, 20, 96, 100))
    _out, meta = asset2d.sprite(im, mask)
    assert meta["pivot"] == [32.5, 81.0]


def test_a_sprite_is_a_cutout_not_a_crop_of_the_background():
    # A round shield's trim box has grey corners in it. A crop would keep them;
    # a cutout must not, which only a subject that is not itself a rectangle
    # can show.
    im = Image.new("RGB", (128, 128), (200, 200, 200))
    ImageDraw.Draw(im).ellipse((32, 20, 96, 100), fill=(200, 30, 30))
    mask = np.asarray(im.convert("RGB")).sum(axis=2) < 500

    out, _meta = asset2d.sprite(im, mask)

    alpha = np.asarray(out)[:, :, 3]
    assert alpha[0, 0] == 0
    assert alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255


def test_pixel_art_comes_out_at_the_asked_for_size():
    im, mask = _subject()
    out, meta = asset2d.pixel(im, mask, size=32)
    assert max(out.size) == 32
    assert meta["size"] == 32


def test_pixel_art_uses_nearest_neighbour_so_edges_stay_hard():
    # A resample that blends would put a ramp of in-between colours along the
    # subject's edge, which is the one thing pixel art must not have.
    im, mask = _subject(box=(20, 20, 108, 108), colour=(255, 0, 0))
    out, _meta = asset2d.pixel(im, mask, size=32)
    rgb = np.asarray(out.convert("RGB")).reshape(-1, 3)
    opaque = np.asarray(out)[:, :, 3].reshape(-1) > 0
    reds = rgb[opaque][:, 0]
    assert set(np.unique(reds)) <= {255}


def test_a_palette_cap_bounds_the_colour_count():
    im = Image.new("RGB", (64, 64))
    pixels = im.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
    mask = np.ones((64, 64), dtype=bool)

    out, meta = asset2d.pixel(im, mask, size=32, colors=8)

    rgb = np.asarray(out.convert("RGB")).reshape(-1, 3)
    assert len({tuple(c) for c in rgb}) <= 8
    assert meta["palette"] == 8


def test_no_palette_cap_is_recorded_as_none():
    im, mask = _subject()
    _out, meta = asset2d.pixel(im, mask, size=32)
    assert meta["palette"] is None


def test_an_empty_mask_is_refused_rather_than_producing_a_blank_file():
    im = Image.new("RGB", (32, 32), (200, 200, 200))
    empty = np.zeros((32, 32), dtype=bool)
    for call in (
        lambda: asset2d.icon(im, empty),
        lambda: asset2d.sprite(im, empty),
        lambda: asset2d.pixel(im, empty, size=32),
    ):
        with pytest.raises(asset2d.NoSubject):
            call()


def test_the_alpha_report_counts_islands():
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.rectangle((4, 4, 20, 20), fill=(255, 0, 0, 255))
    draw.rectangle((40, 40, 58, 58), fill=(0, 255, 0, 255))
    assert asset2d.alpha_report(im)["islands"] == 2


def test_the_alpha_report_ignores_specks_below_the_floor():
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((4, 4, 40, 40), fill=(255, 0, 0, 255))
    im.putpixel((60, 60), (255, 0, 0, 255))
    assert asset2d.alpha_report(im)["islands"] == 1


def test_the_alpha_report_measures_the_soft_rim():
    hard = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
    soft = hard.copy()
    for x in range(32):
        soft.putpixel((x, 0), (255, 0, 0, 128))
    assert asset2d.alpha_report(soft)["partial_fraction"] > 0
    assert asset2d.alpha_report(hard)["partial_fraction"] == 0.0


def test_a_recipe_hash_is_stable_and_order_independent():
    a = asset2d.recipe_hash({"seed": 1, "base_model": "turbo"})
    b = asset2d.recipe_hash({"base_model": "turbo", "seed": 1})
    assert a == b and len(a) == 12


def test_no_recipe_hashes_to_nothing():
    assert asset2d.recipe_hash(None) is None
    assert asset2d.recipe_hash({}) is None
