"""The seamless tileset's one claim that no fixture can make: **that it tiles**.

Everything else about ``tileatlas``/``tilemask`` is arithmetic and is asserted
without a card -- the layouts, the seed derivation, the refusals, the blob-47
collapse, the map round-trip over synthetic gradients. What needs a card is the
half of the design that lives inside the sampler and cannot be faked:

* **the circular padding fires on this path** -- ``text2image.generate(tile=True)``
  swaps every Conv2d in the UNet and the VAE to circular padding for the
  duration of one call, and *nothing else in the tree observes the result*. A
  fake pipeline records the flag; only a real sample can say whether the picture
  that came back actually wraps. ``seam.report`` on four real materials is that
  observation, and it is the claim the entire seamless track rests on: both
  modes, the grid one and the terrain one, are the same generation with a
  different tail, so a material that does not wrap makes every tile below it
  wrong at once.
* **the exact-partition rule was worth having** -- ``reduce_material`` refuses
  any factor that does not divide, on the argument that a torus' first and last
  block are neighbours and a block one pixel wider than its opposite number puts
  a step at the wrap seam. That argument has never been checked against a real
  wrap: this file measures the seam of the *reduced* 32px tile as well as the
  1024px source, which is the only thing that can tell a good rule from a
  superstition.
* **the round-trip identity holds on real texture** -- ``tests/test_tilemask.py``
  already pins "one 47-tile atlas renders any membership field identically to a
  whole-map render", but it pins it over two smooth sinusoidal gradients. A
  generated material carries its own high-frequency grain, which is exactly the
  signal a hairline disagreement at a cell boundary would hide in.

Two of the tests below assert nothing worth failing on and exist to *leave
evidence*: the wrap previews (a boolean cannot settle "does this tile" -- a
human looking at an image rolled by half can) and the palette occupancy (which
``service.tilesheets.sheet_colors`` is openly waiting for and which this lane is
the only place that can produce).

Ratios throughout rather than absolute numbers, ``test_tilesheet_gpu.py``'s
rule: a photograph of gravel and a sheet of plaster are legitimately different
pictures, and an absolute bound would only ever be a claim about whichever four
prompts are written below.

**``seam.SEAM_MAX`` is outside its own corpus here, and the first run showed
it.** ``docs/measurements/2026-08-08-seam-threshold.md`` measured 3.5 on
``sdxl-turbo`` at four steps and closes by naming the next thing to re-run the
scripts against: "a CFG base at 30 steps draws harder edges, and the failure
mode found here is *about* hard edges". This lane is that base at those steps.
The first run of this file scored its four tiled materials at 2.95, 3.16, 3.41
and 3.74 against a turbo tiled population whose *ceiling over 48 units* was
2.50 -- the whole distribution shifted up, exactly as predicted, and the one
unit over the line was the shape that document already names as the ratio's
known false alarm (large flat cells separated by thin hard lines; its two 2.0-era
false alarms were ceramic grout and metal ribs). Its wrap preview showed no join
at all. So a failure of the first test below is **not** first evidence that the
padding did not fire; the wrap preview is what says that, and the threshold owes
a re-measurement on this checkpoint rather than a widening here.

**No ControlNet is required and none is loaded.** That is not an omission: the
seamless modes never open one -- ``service.tilesheets.rows_needed`` gates the
canny row on the grid mode alone -- so a skip on canny weights here would refuse
this lane over a download the job would not have used.

Run with: uv run pytest tests/test_tileset_gpu.py -m gpu -n 0
Add ``-s`` to see the wrap previews' paths and the palette occupancy table; both
are printed rather than asserted, and pytest swallows them on a passing run.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from PIL import Image

from warlock import fetch, guidance, models
from warlock.config import get_config
from warlock.pipelines import seam, tileatlas, tilemask, tilesheet
from warlock.pipelines.pixelsheet import quantize_shared
from warlock.pipelines.text2image import Text2Image
from warlock.service import tilesheets as tilesheets_service
from warlock.studio.tilegrid import blob

#: ``pyproject.toml``'s ``--timeout 120`` is a hang net sized for the default
#: lane, whose slowest surviving test is ~5 s. It is not survivable here: the
#: module fixture below is one checkpoint load, one LoRA load and four full
#: 1024px SDXL samples, and *all* of that is charged to whichever test happens to
#: request the fixture first. Raised on the module rather than on that one test,
#: so the budget does not silently become a fact about the order the tests are
#: written in. Still a hang net and not a budget -- half an hour is far past any
#: honest run of four samples, and what it catches is a wedged pipe.
pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1800)]

#: The house tile size and ``service.tilesheets.DEFAULT_TILE_SIZE``. 1024/32 is
#: 32, an exact partition, which is what ``reduce_material`` requires and what
#: test 5 below is about.
TILE = 32

#: The request seed. One seed for the set, exactly as a real job records it --
#: ``material_seeds`` derives the four that actually run, so this file generates
#: from the same four integers the worker would.
SEED = 42

#: Four subjects that are *genuinely different materials*, chosen so that "the
#: materials are distinguishable" is a question rather than a formality. Two of
#: them (earth, sand) are deliberately the closest legitimate pair a real
#: request would contain: if the difference test only passed on grass against
#: lava it would be measuring the prompts, not the path.
MATERIALS: tuple[str, ...] = (
    "mossy cobblestone paving",
    "dry cracked earth",
    "shallow clear water over pebbles",
    "coarse sand",
)

#: The map the round-trip is checked over. ``tests/test_tilemask.py``'s 12x12 at
#: 32px, restated so the two tests are asking the same question of the same
#: field and only the *materials* differ -- that being the whole of what this
#: lane adds to it.
CELLS = 12


class Material(NamedTuple):
    """One generated material, at both the sizes anything below asks about."""

    index: int
    prompt: str
    seed: int
    source: Path
    #: The 1024px generation as it came back. uint8 ``(1024, 1024, 4)``.
    pixels: np.ndarray
    #: ``reduce_material`` at the tile size. uint8 ``(TILE, TILE, 4)``.
    tile: np.ndarray
    #: The reduced tile on disk, because ``seam.report`` reads a path.
    tile_path: Path


@pytest.fixture(scope="module")
def pipe():
    config = get_config()
    spec = models.BASE_MODELS[tilesheets_service.TILE_SHEET_BASE_MODEL]
    t2i = Text2Image(spec, config.t2i_model_root)
    if not (t2i.model_dir / "model_index.json").exists():
        pytest.skip(f"{spec.label} weights not downloaded")
    yield t2i
    t2i.unload()


@pytest.fixture(scope="module")
def materials(pipe, tmp_path_factory):
    """Four real seamless materials, drawn the way ``_q_tileset`` draws them.

    Module-scoped because it is ~80 s of card and every test below asks a
    different question of the same four pictures -- which is also the honest
    shape: the set is one artifact, and a test that generated its own would be
    measuring a different roll of the same dice.

    Composed to match ``_q_tileset._tile_set`` rather than to resemble it:
    ``material_subject`` through ``guidance.compose_prompt``, the door's default
    negative prompt, the pixel-art LoRA at its own default weight, ``tile=True``
    for the circular padding and an explicit ``(1024, 1024)`` frame. Anything
    composed differently would be a measurement of this file.

    Written into ``tmp_path_factory`` and nowhere else. **This lane sees the
    real ``~/.warlock``** -- it is exempt from the ``WARLOCK_HOME`` pinning in
    ``tests/conftest.py``, because it has to resolve real weights -- so every
    output here is a temporary path and none of it goes near the user's data.
    """
    config = get_config()
    style = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    if not fetch.present(config, "lora", style):
        pytest.skip(f"the {style.label} LoRA is not downloaded")

    scratch = tmp_path_factory.mktemp("tileset-gpu")
    seeds = tileatlas.material_seeds(SEED, len(MATERIALS))
    drawn: list[Material] = []
    for index, text in enumerate(MATERIALS):
        subject = tileatlas.material_subject(text, index=index, total=len(MATERIALS))
        source = scratch / f"material-{index:02d}.png"
        pipe.generate(
            guidance.compose_prompt(subject, {}),
            source,
            seed=seeds[index],
            lora=models.PIXEL_SHEET_LORA,
            lora_weight=style.default_weight,
            negative_prompt=tilesheet.SHEET_NEGATIVE_PROMPT,
            # The whole seamless mechanism, and the one flag this file exists to
            # observe. Not ``tilesheet=``: that is the grid template and carries
            # an explicit no-wrap rule.
            tile=True,
            size=(tileatlas.MATERIAL_PX, tileatlas.MATERIAL_PX),
        )
        with Image.open(source) as opened:
            opened.load()
            pixels = np.asarray(opened.convert("RGBA"))
        reduced = tileatlas.reduce_material(pixels, TILE, TILE)
        tile_path = scratch / f"tile-{index:02d}.png"
        Image.fromarray(reduced, "RGBA").save(tile_path, "PNG")
        drawn.append(
            Material(
                index=index,
                prompt=text,
                seed=seeds[index],
                source=source,
                pixels=pixels,
                tile=reduced,
                tile_path=tile_path,
            )
        )
    return scratch, tuple(drawn)


def _grain(pixels: np.ndarray) -> float:
    """The picture's own mean absolute step between adjacent pixels.

    ``seam.py``'s denominator, in both axes: the amount a texture already
    differs from itself pixel to pixel, which is the only honest reference for
    "how different are two of these".
    """
    rgb = pixels[..., :3].astype(np.float64)
    horizontal = float(np.abs(np.diff(rgb, axis=1)).mean())
    vertical = float(np.abs(np.diff(rgb, axis=0)).mean())
    return (horizontal + vertical) / 2.0


def _difference(first: np.ndarray, second: np.ndarray) -> float:
    """Mean absolute difference between two pictures of the same size."""
    return float(
        np.abs(first[..., :3].astype(np.float64) - second[..., :3].astype(np.float64)).mean()
    )


def test_every_material_is_seamless(materials):
    """The claim the whole track rests on.

    ``tile=True`` patches the convolutions of a resident pipeline for the
    duration of one call and reverts them before returning, and no other test in
    the tree can see whether the patch fired -- a fake pipe records the keyword
    and returns a picture that has nothing to do with it. If this fails, every
    materials sheet has a visible edge at every tile boundary and every terrain
    set's colour continuity argument (``tilemask``'s module docstring: "because
    both materials are seamless, colour is continuous across every tile boundary
    for free") is simply false, with nothing in the data saying so.

    All four ratios are reported whether or not they pass, because the useful
    failure says *which* material and by how much -- one bad draw out of four is
    a different diagnosis from four bad draws. Circular padding either applies to
    a call or it does not, so "the mechanism failed" is a claim about all four;
    a single outlier is a claim about the threshold, and the module docstring
    says which one this checkpoint has already produced.
    """
    _scratch, drawn = materials
    reports = [seam.report(material.source) for material in drawn]
    lines = "; ".join(
        f"{material.prompt} {report['worst']:.2f}"
        for material, report in zip(drawn, reports, strict=True)
    )
    worst = [report["worst"] for report in reports]
    assert max(worst) <= seam.SEAM_MAX, (
        f"a generated material scored past the seam threshold {seam.SEAM_MAX}: "
        f"{lines}. Look at the wrap preview for that material before concluding "
        f"the circular padding failed -- {seam.SEAM_MAX} was measured on "
        f"sdxl-turbo at 4 steps and this lane is a CFG base at 30, which "
        f"docs/measurements/2026-08-08-seam-threshold.md names as the first "
        f"thing that should re-run its scripts. A texture of flat cells parted "
        f"by thin hard lines is that document's own false-alarm shape"
    )


def test_the_wrap_previews_are_left_behind_to_look_at(materials):
    """Not an assertion. The evidence a boolean cannot carry.

    ``seam.report`` is a ratio with a measured threshold and it is still a
    number: ``seam.py`` says outright that ``SEAM_MAX`` was measured on turbo at
    four steps and should be re-measured per checkpoint, and this lane runs a
    CFG base at thirty. So the honest artifact of an overnight run is the
    picture rolled by half in both axes -- what was the wrap seam is now a cross
    through the middle of the frame, and a human deciding "yes, that tiles" is
    the only thing that actually settles it.

    Fails only if ``wrap_preview`` cannot write, which would mean the run left
    no evidence at all.
    """
    scratch, drawn = materials
    previews = scratch / "wrap-previews"
    written = [
        seam.wrap_preview(material.source, previews / f"{material.index:02d}-wrap.png")
        for material in drawn
    ]
    print("\nwrap previews (the seam is the cross through the middle):")
    for material, path in zip(drawn, written, strict=True):
        print(f"  {material.prompt}: {path}")
    assert all(path.exists() for path in written)


def test_the_four_materials_are_structurally_different(materials):
    """The defect ``docs/measurements/2026-08-18-tile-sheet-grid.md`` measured,
    asked of the path that replaced it.

    That run's arm A -- the shipped constants -- imposed an 8x8 grid on one
    generation and got **one continuous brick wall**, with the guide reading as
    mortar. Its ``test_the_cells_are_different_pictures`` *passed* on that sheet,
    because a colour-mean spread over a wall lit unevenly is not zero, and the
    document's own lesson is about the instrument: "the seams are on the grid"
    and "the cells are different tiles" are two claims and only the first is
    cheap to assert. Its diagnosis was that every cell of the guide is identical
    so there is no per-cell signal for variety, and its first-ranked candidate
    was N materials one grid -- variety as a property of the request rather than
    of the model's composition. This path is that candidate: four prompts, four
    generations, four seeds.

    **So this test is expected to pass comfortably, and that is the point.** A
    test that passes by a wide margin is evidence about the *design* -- it says
    the property is structural rather than lucky -- where the grid path's
    equivalent passed by a hair on a sheet that had failed. A ratio near 1 here
    would mean two materials differ from each other no more than each differs
    from itself pixel to pixel, which is what "the same picture twice" looks
    like arithmetically.

    Measured at the resolution the pictures were drawn at, against the larger of
    each pair's own grain, so neither the reduction nor the quantizer is in the
    denominator.
    """
    _scratch, drawn = materials
    grains = {material.index: _grain(material.pixels) for material in drawn}
    ratios: dict[tuple[str, str], float] = {}
    for first, second in itertools.combinations(drawn, 2):
        reference = max(grains[first.index], grains[second.index], 1e-6)
        ratios[(first.prompt, second.prompt)] = (
            _difference(first.pixels, second.pixels) / reference
        )
    lines = "; ".join(f"{a} vs {b} {value:.2f}" for (a, b), value in ratios.items())
    assert min(ratios.values()) > 2.0, (
        f"two materials are no more different from each other than each is from "
        f"itself: {lines}"
    )


def test_the_reduction_keeps_each_materials_contrast(materials):
    """``test_tilesheet_gpu.py``'s contrast-ratio test, at the material's size.

    The measured defect the two-stage sampler replaced: a plain box mean
    averages uncorrelated art pixels and regresses every tile to its mean
    colour (``docs/measurements/2026-08-17-ground-reduction.md``). The reference
    is the material's *own* art resolution -- the prefilter stage's output, one
    mid pixel per pixel-art-LoRA art pixel -- and not an absolute number,
    because a cobblestone and a sheet of still water legitimately have different
    contrast and only one of them says anything about the reducer.

    The same statistic and the same 0.5 floor as the grid lane, deliberately: if
    the two paths' reductions ever diverge, they should diverge against one
    ruler.
    """
    _scratch, drawn = materials
    ratios: dict[str, float] = {}
    for material in drawn:
        # 1024 -> 32 puts ``reduce_cell``'s cap at m = 4, so the art resolution
        # this is measured against is 32*4 = 128, exactly as the grid lane's
        # 128px cell is measured against itself.
        art = tilesheet._box_reduce(material.pixels, TILE * 4, TILE * 4)
        reference = float(art[..., :3].astype(np.int64).std())
        if reference < 4.0:
            continue  # a legitimately flat material has no contrast to keep
        ratios[material.prompt] = (
            float(material.tile[..., :3].astype(np.int64).std()) / reference
        )
    assert ratios, "every material came back flat; these prompts say nothing about it"
    lines = "; ".join(f"{name} {value:.2f}" for name, value in ratios.items())
    assert min(ratios.values()) >= 0.5, (
        f"the reduction flattened a material to below half of its own art "
        f"resolution's contrast; the sampler is averaging rather than "
        f"partitioning: {lines}"
    )


def test_the_reduced_material_still_tiles(materials):
    """The exact-partition refusal, checked against a real wrap.

    ``reduce_material`` refuses any factor that does not divide, and its
    argument is entirely about the wrap: ``_box_reduce`` lets blocks differ in
    size by one, which is invisible in the middle of a grid cell and is a
    one-pixel step at the seam of a torus -- "the one place nobody looks". That
    is a claim about a picture, and until this test it had only ever been
    checked as a ``ValueError``.

    So this is the test that separates a good rule from a superstition: it takes
    a material that wraps at 1024, reduces it on the exact partition the rule
    permits, and asks whether it still wraps at 32. A failure means the
    periodicity did not survive the reduction, and the whole tile-size table --
    not the refusal -- is what would need re-deriving.

    Measured at 32px, where the ratio is naturally harsher than at 1024: the
    interior grain of a pixel-art tile is large, so the denominator is large,
    but so is any real step, and ``SEAM_MAX`` is a ratio precisely so the two
    sizes can be held to one threshold.
    """
    _scratch, drawn = materials
    reports = [seam.report(material.tile_path) for material in drawn]
    lines = "; ".join(
        f"{material.prompt} {report['worst']:.2f}"
        for material, report in zip(drawn, reports, strict=True)
    )
    assert max(report["worst"] for report in reports) <= seam.SEAM_MAX, (
        f"a material wrapped at {tileatlas.MATERIAL_PX}px and stopped wrapping at "
        f"{TILE}px (threshold {seam.SEAM_MAX}): {lines}"
    )


def test_one_atlas_renders_the_whole_map_on_real_texture(materials):
    """``tests/test_tilemask.py``'s round-trip identity, on generated pixels.

    The CPU test already proves this: a 12x12 membership field rendered once as
    one 384px signed-distance field is byte-for-byte what you get by blitting
    ``atlas[blob.indices_from(field)]`` for member cells and the outer material
    for the rest. What it proves it over is two smooth sinusoids -- chosen there
    precisely because a flat colour would pass with the mask scrambled.

    What this adds is the grain. A real material is high-frequency and
    high-contrast by construction (``MATERIAL_STYLE_CLAUSE`` asks for exactly
    that), which is the signal a hairline disagreement at a cell boundary would
    disappear into: two paths that differ by one rounding step produce a visible
    hairline on a gradient and an invisible one on gravel. If the two ever stop
    agreeing, every map painted with a generated terrain set carries a seam at
    every cell boundary and nothing in the data says why.

    **Asserted on the composited uint8, not the float coverage.** The uint8 is
    the artifact and the comparison is exact; the float fields agree to a
    tolerance for reasons that are a property of these particular coordinates
    rather than of the construction.
    """
    _scratch, drawn = materials
    inner, outer = drawn[0].tile, drawn[1].tile
    inset = tilemask.BLOB_INSET_RATIO * TILE
    amplitude = tilemask.NOISE_AMPLITUDE_RATIO * TILE
    feather = tilemask.FEATHER_RATIO * TILE

    field = np.random.default_rng(0).random((CELLS, CELLS)) > 0.45
    assert 0 < int(field.sum()) < field.size  # a field with something in it

    span = CELLS * TILE
    member_rects: list[tuple[float, float, float, float]] = []
    outer_rects: list[tuple[float, float, float, float]] = []
    for cy in range(CELLS):
        for cx in range(CELLS):
            rect = (
                float(cx * TILE),
                float(cy * TILE),
                float((cx + 1) * TILE),
                float((cy + 1) * TILE),
            )
            (member_rects if field[cy, cx] else outer_rects).append(rect)

    whole_alpha = tilemask.coverage(
        member_rects,
        outer_rects,
        span,
        noise=np.tile(tilemask.wrap_noise(TILE, seed=0), (CELLS, CELLS)),
        inset=inset,
        amplitude=amplitude,
        feather=feather,
    )
    whole = tilemask.composite(
        np.tile(inner, (CELLS, CELLS, 1)), np.tile(outer, (CELLS, CELLS, 1)), whole_alpha
    )

    atlas = tilemask.blob_atlas(inner, outer, TILE, seed=0)
    assert atlas.shape == (TILE, blob.TILE_COUNT * TILE, 4)
    indices = blob.indices_from(field)

    assembled = np.empty_like(whole)
    for cy in range(CELLS):
        for cx in range(CELLS):
            box = (slice(cy * TILE, (cy + 1) * TILE), slice(cx * TILE, (cx + 1) * TILE))
            if field[cy, cx]:
                case = int(indices[cy, cx])
                assembled[box] = atlas[:, case * TILE : (case + 1) * TILE]
            else:
                # What the painter blits without consulting ``tilemask`` at all,
                # which is the half of the identity that has to hold exactly.
                assembled[box] = outer
    disagreeing = int(np.count_nonzero(np.any(whole != assembled, axis=2)))
    assert np.array_equal(whole, assembled), (
        f"the 47-tile atlas and the whole-map render disagree on {disagreeing} of "
        f"{span * span} pixels of real texture; a generated terrain set will paint "
        f"a seam at cell boundaries"
    )


def test_the_palette_occupancy_is_recorded_rather_than_asserted(materials):
    """A measurement, not a gate. The number ``sheet_colors`` is waiting for.

    ``service.tilesheets.sheet_colors`` ships **provisional and says so**:
    ``SHEET_COLORS = 64`` was measured over one generation of one subject, and
    the ``32 * cells`` linearity -- the assumption that a material's share of a
    shared table stays constant as materials are added -- has never been tested
    at all. Its docstring names exactly what would move it: occupancy measured
    per material against the shared table at 2, 4, 8 and 16 materials.

    This test produces the four-material row of that table and asserts nothing
    about it. **A threshold asserted here would be a number invented rather than
    measured**, which is the failure mode the whole ``docs/measurements/``
    convention exists to prevent -- and it would also be self-fulfilling, since
    the only run that could refute it is this one.

    The trivially safe assertion is the one that says the measurement happened:
    the quantizer returned a palette and every material got some of it. A
    material at zero would mean a cell the median cut spent nothing on, which is
    a real defect rather than a taste question.
    """
    _scratch, drawn = materials
    geom = tileatlas.material_geometry(TILE, tilesheet.TOP_DOWN, len(drawn))
    atlas = tileatlas.assemble([material.tile for material in drawn], geom)
    colors = tilesheets_service.sheet_colors(len(drawn))
    quantized, palette = quantize_shared(Image.fromarray(atlas, "RGBA"), colors)
    mapped = np.asarray(quantized.convert("RGBA"))

    print(
        f"\npalette occupancy at {len(drawn)} materials, "
        f"budget {colors} entries (sheet_colors({len(drawn)})):"
    )
    occupancy: list[int] = []
    for material, cell in zip(drawn, geom.cells, strict=True):
        top, left = cell.row * geom.tile_h, cell.col * geom.tile_w
        block = mapped[top : top + geom.tile_h, left : left + geom.tile_w, :3]
        distinct = len({tuple(int(c) for c in pixel) for pixel in block.reshape(-1, 3)})
        occupancy.append(distinct)
        print(
            f"  {material.prompt}: {distinct} distinct colours "
            f"({distinct / colors:.0%} of the shared table)"
        )
    print(
        f"  whole sheet: {len(palette)} of {colors} ({len(palette) / colors:.0%}); "
        f"sum of the four cells {sum(occupancy)} (materials share entries where "
        f"they overlap, so this exceeds the sheet's own count)"
    )
    assert len(palette) > 0
    assert min(occupancy) > 0, (
        f"the shared median cut spent nothing on a material: {occupancy}"
    )
