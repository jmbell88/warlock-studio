"""Comparability and the two metrics. Pillow-drawn fixtures, no weights."""

from __future__ import annotations

from PIL import Image, ImageDraw

from warlock.bench import calibrate as calibrate_mod
from warlock.bench import imageprep, metrics

BG = (200, 200, 200)


def _reference(tmp_path, name="ref.png", box=(64, 64, 191, 191), size=256):
    path = tmp_path / name
    im = Image.new("RGB", (size, size), BG)
    ImageDraw.Draw(im).rectangle(list(box), fill=(40, 90, 160))
    im.save(path)
    return path


def _render(tmp_path, name="view.png", box=(20, 20, 107, 107), size=128):
    """A render: transparent background, subject at a different scale -- which
    is the whole reason prepare_pair exists."""
    path = tmp_path / name
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle(list(box), fill=(90, 120, 200, 255))
    im.save(path)
    return path


# --- imageprep ---------------------------------------------------------------


def test_a_pair_comes_back_square_and_the_same_size(tmp_path):
    ref, render = imageprep.prepare_pair(_reference(tmp_path), _render(tmp_path))
    assert ref.size == render.size == (imageprep.PAIR_SIZE, imageprep.PAIR_SIZE)
    assert ref.mode == render.mode == "RGB"


def test_the_render_is_composited_onto_the_references_background(tmp_path):
    """Background hue has to cancel between the two, or the metric sees a
    difference that has nothing to do with the mesh."""
    _ref, render = imageprep.prepare_pair(_reference(tmp_path), _render(tmp_path))
    assert render.getpixel((2, 2)) == BG


def test_framing_and_scale_are_cancelled(tmp_path):
    """The two subjects are at very different scales in their own frames; after
    preparation they occupy the same share of the output."""
    ref, render = imageprep.prepare_pair(
        _reference(tmp_path, box=(100, 100, 155, 155)),
        _render(tmp_path, box=(4, 4, 123, 123)),
    )
    ref_box = ref.convert("L").point(lambda v: 255 if v < 150 else 0).getbbox()
    render_box = render.convert("L").point(lambda v: 255 if v < 150 else 0).getbbox()
    for a, b in zip(ref_box, render_box, strict=True):
        assert abs(a - b) <= 4


def test_an_empty_render_is_a_pair_of_nones_rather_than_a_crash(tmp_path):
    """An all-transparent render is a failed reconstruction, which is a score
    of zero -- not an exception that kills a 160-item run."""
    empty = tmp_path / "empty.png"
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(empty)
    assert imageprep.prepare_pair(_reference(tmp_path), empty) == (None, None)


def test_the_border_colour_is_the_background_not_an_average(tmp_path):
    with Image.open(_reference(tmp_path)) as im:
        assert imageprep.border_colour(im) == BG


def test_two_opaque_references_pair_up(tmp_path):
    # prepare_pair takes the render side's subject from its alpha, which two
    # SDXL images do not have -- so the reference pipeline needs its own.
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for path, box in ((a, (20, 20, 60, 90)), (b, (40, 30, 70, 100))):
        im = Image.new("RGB", (128, 128), BG)
        ImageDraw.Draw(im).rectangle(box, fill=(20, 20, 20))
        im.save(path)

    left, right = imageprep.prepare_references(a, b)

    assert left is not None and right is not None
    assert left.size == right.size == (imageprep.PAIR_SIZE, imageprep.PAIR_SIZE)
    assert left.mode == right.mode == "RGB"


def test_a_reference_with_no_subject_pairs_to_nothing(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (64, 64), BG).save(a)
    im = Image.new("RGB", (64, 64), BG)
    ImageDraw.Draw(im).rectangle((10, 10, 40, 50), fill=(0, 0, 0))
    im.save(b)

    assert imageprep.prepare_references(a, b) == (None, None)


# --- silhouette IoU ----------------------------------------------------------


def test_identical_silhouettes_score_one(tmp_path):
    ref = _reference(tmp_path, box=(64, 64, 191, 191))
    render = _render(tmp_path, box=(32, 32, 95, 95))  # same square, half scale
    assert metrics.silhouette_iou(ref, render) > 0.99


def test_disjoint_shapes_score_low(tmp_path):
    """A tall bar against a wide bar: cropped to their own boxes and squared,
    they barely overlap."""
    ref = _reference(tmp_path, box=(120, 20, 135, 235))
    render = _render(tmp_path, box=(10, 60, 117, 67))
    assert metrics.silhouette_iou(ref, render) < 0.2


def test_a_half_overlap_scores_between(tmp_path):
    ref = _reference(tmp_path, box=(64, 64, 191, 191))
    # An L: half the square is missing, so the union is the square.
    path = tmp_path / "L.png"
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, 119, 63], fill=(0, 0, 0, 255))
    d.rectangle([8, 64, 63, 119], fill=(0, 0, 0, 255))
    im.save(path)
    score = metrics.silhouette_iou(ref, path)
    assert 0.6 < score < 0.85


def test_an_empty_render_has_no_score(tmp_path):
    empty = tmp_path / "empty.png"
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(empty)
    assert metrics.silhouette_iou(_reference(tmp_path), empty) is None


def test_silhouette_iou_needs_no_weights_at_all(tmp_path):
    """The metric that measures what conditioning actually changes must not be
    gated behind an optional download."""
    assert "silhouette_iou" in metrics.available()
    assert metrics.silhouette_iou(_reference(tmp_path), _render(tmp_path)) is not None


def test_score_view_skips_a_metric_whose_weights_are_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "dino_available", lambda config=None: False)
    assert set(metrics.score_view(_reference(tmp_path), _render(tmp_path))) == {
        "silhouette_iou"
    }


def test_dino_says_how_to_get_its_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "_dino_dir", lambda config=None: tmp_path / "nope")
    import pytest

    with pytest.raises(RuntimeError, match="hf download"):
        metrics._dino_model()


# --- calibrate ---------------------------------------------------------------


def test_a_sweep_enumerates_every_yaw_without_the_atlas_guard():
    """sheet.plan would refuse 36 views at 256px -- its limit is about a packed
    atlas, and nothing here packs one."""
    cells = calibrate_mod.sweep_cells(36)
    assert len(cells) == 36
    assert cells[0]["yaw"] == 0.0
    assert all(c["bones"] == {} for c in cells)
    assert [c["index"] for c in cells] == list(range(36))


def test_the_spread_of_argmaxes_is_measured_on_the_circle():
    """350 and 10 are 20 degrees apart; a plain stdev would call that a
    240-degree disagreement and report a scatter that is not there."""
    assert calibrate_mod._circular_spread([350.0, 10.0, 0.0]) == 20.0
    assert calibrate_mod._circular_spread([0.0]) == 0.0
    assert calibrate_mod._circular_spread([0.0, 90.0, 180.0, 270.0]) == 270.0


def _result(job, yaw, score=0.8):
    return {
        "job": job,
        "cells": [{"yaw": yaw, "elevation": 20.0, "silhouette_iou": score}],
        "silhouette_iou": {
            "best_yaw": yaw, "best_elevation": 20.0,
            "best_score": score, "worst_score": 0.1, "mean_score": 0.4,
        },
    }


def test_one_job_is_reported_as_too_few_rather_than_as_a_result():
    """One argmax is an anecdote. Saying so is the point."""
    v = calibrate_mod.verdict([_result("a", 90.0)], "silhouette_iou")
    assert v["n"] == 1
    assert "too few" in v["verdict"]


def test_agreeing_jobs_are_reported_as_stable():
    results = [_result("a", 90.0), _result("b", 100.0), _result("c", 95.0)]
    v = calibrate_mod.verdict(results, "silhouette_iou")
    assert "stable" in v["verdict"]
    assert v["yaw_spread"] == 10.0


def test_disagreeing_jobs_are_reported_as_scattered():
    results = [_result("a", 0.0), _result("b", 120.0), _result("c", 250.0)]
    v = calibrate_mod.verdict(results, "silhouette_iou")
    assert "scattered" in v["verdict"]
    assert "max-over-8-views" in v["verdict"]


def test_a_table_leads_with_the_best_cells():
    table = calibrate_mod.format_table(
        {"job": "x", "cells": [
            {"yaw": 0.0, "elevation": 0.0, "silhouette_iou": 0.1},
            {"yaw": 90.0, "elevation": 0.0, "silhouette_iou": 0.9},
        ]},
        "silhouette_iou",
    )
    assert table.splitlines()[2].strip().startswith("90.0")


# --- pixel-art metrics -------------------------------------------------------


def _pixel_art(path, scale=8, phase=(3, 5), cells=32):
    """Authored pixel art blown up to a canvas the way a pixel-art model draws
    it: square blocks, on a lattice whose first cell does not start at 0."""
    import numpy as np

    rng = np.random.default_rng(5)
    palette = np.array(
        [[20, 20, 30], [200, 60, 60], [60, 180, 90], [240, 220, 130]], np.uint8
    )
    art = palette[rng.integers(0, len(palette), size=(cells, cells))]
    big = Image.fromarray(art, "RGB").resize((cells * scale, cells * scale), Image.NEAREST)
    canvas = Image.new("RGB", (big.width + scale, big.height + scale), (128, 128, 128))
    canvas.paste(big, (phase[1], phase[0]))
    canvas.save(path)
    return path


def test_score_reference_measures_the_grid_a_generator_drew(tmp_path):
    scores = metrics.score_reference(_pixel_art(tmp_path / "input.png"))
    assert set(scores) == set(metrics.REFERENCE_METRICS)
    assert scores["pixel_grid_scale"] == 8
    assert scores["pixel_grid_residual"] < 0.05
    assert scores["pixel_colors_128"] > 0


def test_a_smooth_reference_reports_no_grid_rather_than_failing(tmp_path):
    """The control arm's images are ordinary SDXL renders, and "no lattice" is
    the measurement, not an error."""
    import numpy as np

    ramp = np.dstack([np.tile(np.linspace(0, 255, 256), (256, 1))] * 3).astype(np.uint8)
    path = tmp_path / "smooth.png"
    Image.fromarray(ramp, "RGB").save(path)
    scores = metrics.score_reference(path)
    assert scores["pixel_grid_scale"] is None
    assert scores["pixel_grid_residual"] > 0.05


def test_the_reference_metrics_need_no_torch_at_all(tmp_path, monkeypatch):
    """A reference-stage run is scored on pixels, and paying seconds for a
    torch import to measure a PNG would be a cost with nothing behind it."""
    import builtins
    import sys

    path = _pixel_art(tmp_path / "input.png")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def refusing_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("score_reference must not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing_import)
    assert metrics.score_reference(path)["pixel_grid_scale"] == 8
