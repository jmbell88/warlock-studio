"""Does this image actually tile?

A ratio, not an absolute difference: a busy texture legitimately differs a lot
between any two adjacent columns, so the only meaningful question is whether
the wrap seam differs *more* than the interior does. That normalisation is
what makes one threshold work for cobblestone and for flat plaster alike.

*Which* interior number it is normalised against changed on 2026-08-30
(``docs/measurements/2026-08-30-seam-dominance.md``): the verdict is now the
seam against the largest interior step rather than against the mean one. Both
are reported and the tests below say which is which, because the difference is
one word in a docstring and two very different answers on flat-cell pixel art.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import seam


def test_the_threshold_stays_inside_the_band_that_was_measured():
    """The calibration, as an assertion rather than as a comment.

    ``docs/measurements/2026-08-08-seam-threshold.md`` puts the highest
    legitimately seamless tile at 2.50 and the lowest visible seam at 5.52 over
    72 units. Anything inside that band scores identically on the corpus and
    anything outside it is known to misclassify -- 2.0, the value this replaced,
    raised two false alarms on tiles that wrap perfectly. This does not re-derive
    the number; it fails a re-guess that leaves the evidence behind.
    """
    assert 2.50 < seam.SEAM_MAX < 5.52


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
    # Both denominators are zero on a flat field, so both guards have to fire.
    assert out["dominance"] == 0.0


def test_an_almost_flat_image_with_one_hard_join_is_not_called_seamless(tmp_path):
    # The case the flat-image guard must not swallow: a picture with almost no
    # grain and one 160-level step across its wrap edge is not a tile, and a
    # normalisation that clamped a small denominator to "seamless" would say it
    # was.
    #
    # The join is spread over sixteen columns rather than one, and that shape is
    # required by what is being asserted rather than incidental. A single hard
    # interior step of the same height as the seam makes the picture a *stripe*,
    # and a stripe genuinely tiles -- laid side by side it repeats with no join
    # the interior does not already have. Under the dominance statistic that is
    # a tie at exactly 1.0 and the honest verdict is "seamless"; ramping the
    # interior transition instead leaves the wrap as the single worst step in
    # the frame, which is the picture this test means.
    ramp = np.clip((np.arange(32) - 8.0) / 16.0, 0.0, 1.0)
    arr = np.repeat((40 + 160 * ramp).astype(np.uint8)[None, :], 32, axis=0)
    Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB").save(path := tmp_path / "tile.png")
    out = seam.report(path)
    assert out["seamless"] is False
    assert out["dominance"] > seam.SEAM_DOMINANCE_MAX
    assert out["horizontal"] > seam.SEAM_MAX


def test_a_periodic_stripe_tiles_and_the_two_statistics_disagree_about_it(tmp_path):
    """The semantic change, pinned as the pair of numbers it actually is.

    Two blocks of colour: the wrap step and the one interior step are both 160
    levels, so the picture repeats without introducing any join it does not
    already contain -- it *is* a tile, of stripes. The edge-against-mean-grain
    ratio calls it seamed because a single hard edge over 32 columns leaves a
    tiny mean; dominance ties it at exactly 1.0 and passes it.

    This was the assertion that had to be rewritten to land the change, so it
    is kept as its own case rather than deleted: if a later edit makes the two
    statistics agree here, one of them has stopped being what it says it is.
    """
    arr = np.full((32, 32, 3), 40, dtype=np.uint8)
    arr[:, 16:] = 200
    Image.fromarray(arr, "RGB").save(path := tmp_path / "tile.png")
    out = seam.report(path)
    assert out["dominance"] == pytest.approx(1.0)
    assert out["seamless"] is True
    assert out["worst"] > seam.SEAM_MAX


def test_the_verdict_is_the_dominance_number_and_the_row_says_so(tmp_path):
    """``metric`` is what lets a stored row be read in its own vocabulary.

    Its absence marks a row written before 2026-08-30, and
    ``inspector.seam_verdict`` words those with the ratio. A report that
    decided on dominance but did not say so would be read as an edge/grain
    number of a completely different scale -- 0.94 and 3.4 are both plausible
    values of both statistics and mean opposite things.
    """
    path = tmp_path / "tile.png"
    _gradient(wrap=False).save(path)
    out = seam.report(path)
    assert out["metric"] == "dominance"
    assert out["threshold"] == seam.SEAM_DOMINANCE_MAX
    assert out["seamless"] is (out["dominance"] <= seam.SEAM_DOMINANCE_MAX)
    assert out["dominance"] == max(out["dominance_horizontal"], out["dominance_vertical"])
    # And the ratio is still there, unchanged, for the three corpora keyed on it.
    assert out["worst"] == max(out["horizontal"], out["vertical"])


def test_the_ideal_seamless_tile_ties_at_one_rather_than_failing(tmp_path):
    """Why the comparison is ``<=``.

    A triangle wave's wrap step is exactly its interior step -- that is what
    makes it the canonical seamless tile -- so it lands on 1.0 by construction
    and a strict comparison would false-alarm on the one picture nobody
    disputes.
    """
    path = tmp_path / "tile.png"
    _gradient(wrap=True).save(path)
    out = seam.report(path)
    assert out["dominance"] <= seam.SEAM_DOMINANCE_MAX
    assert out["seamless"] is True


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
