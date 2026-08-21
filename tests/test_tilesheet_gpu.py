"""The tile sheet's claims that no CPU test can make: the grid, and the tiles.

Everything else about this feature is arithmetic and is asserted without a card.
What needs one is the pair of things the whole design rests on, and neither is
provable against a fixture:

* **the guide is obeyed** -- the model puts its seams where the guide drew them,
  so slicing on the predetermined rectangles gives sixty-four whole tiles rather
  than sixty-four fragments straddling their neighbours;
* **the reduction keeps the art** -- a 128px cell brought down to 32px still has
  contrast rather than having regressed to its mean colour, which is the defect
  ``docs/measurements/2026-08-17-ground-reduction.md`` measured for the ground
  path this reduction was carried over from.

Both are asserted as *ratios* against the cell's own art rather than as absolute
numbers, because a busy tile and a plain one are legitimately different pictures
and an absolute bound would only ever be a claim about whichever prompt is
written below.

Run with: uv run pytest tests/test_tilesheet_gpu.py -m gpu -n 0
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock import guidance, models
from warlock.config import get_config
from warlock.pipelines import tilesheet
from warlock.pipelines.conditioning import Conditioning
from warlock.pipelines.text2image import Text2Image

pytestmark = pytest.mark.gpu

TILE = 32
PROMPT = "a damp stone dungeon: flagstones, moss, still water, rubble"
#: Spelled with the canonical constant rather than the ``orthogonal`` alias the
#: module still reads: nothing writes that spelling any more, and using it here
#: is how this file went a whole view-vocabulary change without noticing that
#: ``SheetGeometry.projection`` had become ``.view`` -- the lane does not run by
#: default, so the rename cost a card run rather than a test run.
VIEW = tilesheet.TOP_DOWN


@pytest.fixture(scope="module")
def pipe():
    config = get_config()
    spec = models.BASE_MODELS["sdxl_cfg"]
    t2i = Text2Image(spec, config.t2i_model_root)
    if not (t2i.model_dir / "model_index.json").exists():
        pytest.skip(f"{spec.label} weights not downloaded")
    yield t2i
    t2i.unload()


@pytest.fixture(scope="module")
def sheet(pipe, tmp_path_factory):
    """One real 8x8 generation, drawn against a real guide.

    Module-scoped because it is ~20 s of card and every test below asks a
    different question of the same picture -- which is also the honest shape:
    the sheet is one artifact, and a test that generated its own would be
    measuring a different roll of the same dice.
    """
    from warlock import fetch

    canny = models.CONTROLNETS["canny"]
    if not fetch.present(get_config(), "control", canny):
        pytest.skip("the canny ControlNet is not downloaded")

    scratch = tmp_path_factory.mktemp("tilesheet-gpu")
    geom = tilesheet.geometry(TILE, VIEW)
    guide_path = scratch / "guide.png"
    tilesheet.render_guide(geom).save(guide_path, "PNG")

    out = scratch / "sheet.png"
    pipe.generate(
        guidance.compose_prompt(tilesheet.sheet_subject(PROMPT, geom.view), {}),
        out,
        seed=42,
        lora=models.PIXEL_SHEET_LORA,
        negative_prompt=tilesheet.SHEET_NEGATIVE_PROMPT,
        conditioning=Conditioning(control="canny", control_image=guide_path),
        tilesheet=True,
        size=geom.source_size,
    )
    with Image.open(out) as opened:
        opened.load()
        return geom, np.asarray(opened.convert("RGBA"))


def _cell(array: np.ndarray, geom, row: int, col: int) -> np.ndarray:
    cell = geom.cells[row * geom.columns + col]
    return array[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]


def _edge_energy(array: np.ndarray, axis: int, index: int) -> float:
    """The mean absolute step across one line of the image."""
    grey = array[..., :3].astype(np.int64).mean(axis=2)
    if axis == 0:
        return float(np.abs(grey[index, :] - grey[index - 1, :]).mean())
    return float(np.abs(grey[:, index] - grey[:, index - 1]).mean())


def test_the_generation_comes_back_on_the_frame_that_was_asked_for(sheet):
    """The size override is what makes an isometric sheet possible at all, and
    a frame that came back square would slice into sixty-four wrong cells with
    nothing saying so."""
    geom, array = sheet
    assert (array.shape[1], array.shape[0]) == geom.source_size


def test_the_model_puts_its_seams_where_the_guide_drew_them(sheet):
    """The whole mechanism. The slicer cuts on rectangles decided before the
    call, so what has to be true of the *picture* is that its discontinuities
    line up with them -- otherwise every tile carries a sliver of its
    neighbour.

    Measured as a ratio of interior steps, because a busy sheet has a large
    interior step and a plain one a tiny one.
    """
    geom, array = sheet
    boundaries = [
        _edge_energy(array, 1, col * geom.cell_w) for col in range(1, geom.columns)
    ] + [_edge_energy(array, 0, row * geom.cell_h) for row in range(1, geom.rows)]
    interior = [
        _edge_energy(array, 1, col * geom.cell_w + geom.cell_w // 2)
        for col in range(geom.columns)
    ] + [
        _edge_energy(array, 0, row * geom.cell_h + geom.cell_h // 2)
        for row in range(geom.rows)
    ]
    ratio = float(np.median(boundaries)) / max(float(np.median(interior)), 1e-6)
    assert ratio > 1.5, (
        f"cell boundaries are no sharper than the interior (ratio {ratio:.2f}); "
        "the guide was not obeyed and the slice will straddle tiles"
    )


def test_the_cells_are_different_pictures(sheet):
    """A sheet of sixty-four identical tiles is what the negative prompt's
    "one large scene" clause exists to prevent, and it is indistinguishable
    from success in every arithmetic test."""
    geom, array = sheet
    means = np.array(
        [
            _cell(array, geom, row, col)[..., :3].mean(axis=(0, 1))
            for row in range(geom.rows)
            for col in range(geom.columns)
        ]
    )
    spread = float(np.abs(means - means.mean(axis=0)).mean())
    assert spread > 4.0, f"every cell is the same colour (spread {spread:.2f})"


def test_the_reduction_keeps_the_cells_contrast(sheet):
    """The measured defect the two-stage sampler replaced: a plain box mean
    averages uncorrelated art pixels and regresses every tile to its mean
    colour. The reference is the cell's own art resolution, not an absolute
    number."""
    geom, array = sheet
    reduced = tilesheet.reduce_sheet(array, geom)
    ratios = []
    for row in range(geom.rows):
        for col in range(geom.columns):
            cell = _cell(array, geom, row, col)[..., :3].astype(np.int64)
            # The cell at the art resolution: one mid pixel per LoRA art pixel.
            art = tilesheet._box_reduce(
                np.dstack([cell, np.full(cell.shape[:2], 255, np.int64)]).astype(
                    np.uint8
                ),
                128,
                128,
            )[..., :3].astype(np.int64)
            tile = reduced[
                row * geom.tile_h : (row + 1) * geom.tile_h,
                col * geom.tile_w : (col + 1) * geom.tile_w,
                :3,
            ].astype(np.int64)
            reference = float(art.std())
            if reference < 4.0:
                continue  # a legitimately flat cell has no contrast to keep
            ratios.append(float(tile.std()) / reference)
    assert ratios, "every cell was flat; this prompt says nothing about the reduction"
    assert min(ratios) >= 0.5, (
        f"the reduction flattened a tile to {min(ratios):.2f} of its own art "
        "resolution; the sampler is averaging rather than partitioning"
    )


def test_the_published_sheet_is_the_promised_size(sheet):
    geom, array = sheet
    reduced = tilesheet.reduce_sheet(array, geom)
    assert (reduced.shape[1], reduced.shape[0]) == geom.sheet_size
    assert reduced.dtype == np.uint8
