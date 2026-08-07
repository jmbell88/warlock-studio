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


def _disc(size=(128, 128), box=(14, 14, 113, 113), colour=(255, 0, 0)):
    """A round subject, so the downscale leaves a real partial-alpha rim."""
    im = Image.new("RGB", size, (200, 200, 200))
    ImageDraw.Draw(im).ellipse(box, fill=colour)
    mask = np.asarray(im.convert("RGB")).sum(axis=2) < 500
    return im, mask


def test_an_icons_soft_rim_keeps_the_subjects_own_colour():
    # Compositing the resample against the empty canvas *through its own alpha*
    # premultiplies it: every rim pixel gets its colour dragged toward the
    # transparent black it is drawn onto, which is a dark halo around every
    # icon in the set. The subject is a single flat red, so any visible pixel
    # that is not that red has been faded toward the background.
    im, mask = _disc(colour=(255, 0, 0))

    out, _meta = asset2d.icon(im, mask, size=48, pad=0.0)

    arr = np.asarray(out)
    visible = arr[:, :, 3] > 0
    rim = visible & (arr[:, :, 3] < 255)
    assert rim.sum() > 100, "no partial-alpha rim to test -- the fixture is wrong"
    assert arr[:, :, 0][visible].min() == 255


def test_an_icon_keeps_the_coverage_it_downsampled():
    # The other half of the same mistake: using the resample as its own paste
    # mask also squares the alpha, so the faintest rim pixels round away to
    # nothing and the subject quietly loses area. A downscale by a normalised
    # filter conserves coverage -- total alpha is the subject's area times the
    # scale squared -- so the coverage the icon carries is checkable without
    # reference to how it was resampled.
    im, mask = _disc()
    size = 48

    out, meta = asset2d.icon(im, mask, size=size, pad=0.0)

    left, top, right, bottom = meta["trim"]
    scale = size / max(right - left, bottom - top)
    expected = float(mask.sum()) * scale * scale
    carried = np.asarray(out)[:, :, 3].astype(float).sum() / 255.0
    assert carried == pytest.approx(expected, rel=0.005)


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


def test_a_padded_sprites_pivot_stays_on_the_subjects_feet():
    # Padding grows the canvas, and a pivot measured from the padded canvas's
    # own bottom edge sits in the empty margin rather than on the subject --
    # a whole set placed a few pixels into the floor, while pivot_rule still
    # claims bottom-centre. The pivot must land on the subject at any pad.
    im, mask = _subject(box=(32, 20, 96, 100))

    _flush, plain = asset2d.sprite(im, mask)
    out, meta = asset2d.sprite(im, mask, pad=0.1)

    assert plain["margin"] == 0 and plain["pad"] == 0.0
    assert meta["pivot_rule"] == "bottom-centre"

    alpha = np.asarray(out)[:, :, 3]
    x, y = meta["pivot"]
    # The pivot is the subject's bottom edge: the last subject row is opaque,
    # the row the pivot sits on is already margin.
    assert alpha[int(y) - 1, int(x)] == 255
    assert alpha[int(y), int(x)] == 0


def test_a_padded_sprites_placement_is_derivable_from_its_metadata():
    # An importer only ever sees the manifest. Without the margin recorded it
    # cannot tell a padded sprite from a subject that simply has transparent
    # edges, so it cannot put the subject back where it was.
    im, mask = _subject(box=(32, 20, 96, 100))

    _out, meta = asset2d.sprite(im, mask, pad=0.1)

    left, top, right, bottom = meta["trim"]
    margin = meta["margin"]
    assert margin > 0
    assert meta["canvas"] == [right - left + 2 * margin, bottom - top + 2 * margin]
    assert meta["pivot"] == [margin + (right - left) / 2.0, float(margin + bottom - top)]


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


def _pixel_art_subject(scale=8, phase=(5, 3), cells=16):
    """Authored pixel art, blown up the way a pixel-art model draws it, with a
    subject that does *not* start on a cell boundary in the blown-up frame."""
    rng = np.random.default_rng(11)
    palette = np.array(
        [[20, 20, 30], [200, 60, 60], [60, 180, 90], [240, 220, 130]], np.uint8
    )
    art = palette[rng.integers(0, len(palette), size=(cells, cells))]
    big = Image.fromarray(art, "RGB").resize(
        (cells * scale, cells * scale), Image.NEAREST
    )
    py, px = phase
    canvas = Image.new(
        "RGB", (cells * scale + 2 * scale, cells * scale + 2 * scale), (128, 128, 128)
    )
    canvas.paste(big, (px, py))
    mask = np.zeros(canvas.size[::-1], dtype=bool)
    mask[py : py + big.height, px : px + big.width] = True
    return canvas, mask, art


def test_pixel_with_default_opts_is_byte_identical_to_the_legacy_path():
    """Every 2D asset already on disk was cut by the old crop-then-scale path,
    and a manifest that says it was is only true while this holds."""
    im, mask = _subject()
    legacy, _box, _cap = asset2d._legacy_pixel(im, mask, 32, 0)
    out, _meta = asset2d.pixel(im, mask, size=32)
    assert np.array_equal(np.asarray(out), np.asarray(legacy))


def test_pixel_grid_branch_recovers_the_authored_cells():
    """The load-bearing ordering: reduce on the lattice, *then* crop. Cropping
    first moves the origin off the phase by whatever the subject's bounding box
    happens to be, which shears the reduction instead of reducing it."""
    im, mask, art = _pixel_art_subject()
    out, meta = asset2d.pixel(im, mask, size=64)
    assert meta["grid"] is not None
    assert meta["grid"]["scale"] == 8
    assert np.array_equal(np.asarray(out.convert("RGB")), art)
    # Reported in source coordinates, as it always was.
    assert meta["trim"] == [3, 5, 3 + art.shape[1] * 8, 5 + art.shape[0] * 8]


def test_the_grid_branch_snaps_alpha_and_records_its_qa():
    im, mask, _art = _pixel_art_subject()
    out, meta = asset2d.pixel(im, mask, size=64)
    assert set(np.unique(np.asarray(out)[:, :, 3])) <= {0, 255}
    assert meta["qa"]["colors"] == 4
    assert meta["cleanup"] == 0


def test_a_palette_file_replaces_the_median_cut_cap():
    from warlock.pipelines import pixel as pixelmod

    im, mask = _subject()
    palette = ((0, 0, 0), (255, 255, 255))
    opts = pixelmod.PixelOpts(
        colors=8, palette_name="two", palette=palette, palette_hash="abc"
    )
    out, meta = asset2d.pixel(im, mask, size=32, opts=opts)
    rgb = np.asarray(out.convert("RGBA"))
    assert {tuple(int(c) for c in p) for p in rgb[rgb[:, :, 3] > 0][:, :3]} <= set(
        palette
    )
    # The int cap is what median cut used; a palette file is a different thing
    # and the manifest says so rather than conflating the two.
    assert meta["palette"] is None
    assert meta["palette_file"] == "two"
    assert meta["palette_hash"] == "abc"


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
