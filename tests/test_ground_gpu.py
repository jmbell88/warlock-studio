"""The ground set's one claim that no CPU test can make: the seam is real.

Everything else about this feature is arithmetic and is asserted without a card.
What needs one is the thing the whole design rests on -- that a texture
generated with circular padding, reduced to the tile size by a *partition* of
its own torus, is still seamless at the smaller period. A bilinear resize or a
crop would pass every test in ``tests/test_ground_pipeline.py`` and put a faint
grid on every painted field.

Run with: uv run pytest tests/test_ground_gpu.py -m gpu -n 0
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock import guidance, models
from warlock.config import get_config
from warlock.pipelines import ground
from warlock.pipelines import prompt as prompt_lib
from warlock.pipelines.text2image import Text2Image
from warlock.studio.plotter import blob, groundtex
from warlock.studio.plotter.tileset import TerrainSpec, Tileset

pytestmark = pytest.mark.gpu

TILE = 32
TERRAINS = (
    {"name": "stone", "material": "cracked stone flagstones", "color": (120, 120, 125, 255)},
    {"name": "water", "material": "still dark water", "color": (40, 90, 170, 255)},
)


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
def painted(pipe, tmp_path_factory):
    """Four real textures -- a fill and a rim for each of two terrains."""
    scratch = tmp_path_factory.mktemp("ground-gpu")
    out: dict[tuple[int, str], np.ndarray] = {}
    for index, row in enumerate(TERRAINS):
        subjects = ground.texture_prompts(
            "a damp dungeon", row["name"], row["material"], "", row["color"]
        )
        for role, subject in zip(ground.ROLES, subjects, strict=True):
            path = scratch / f"t{index}-{role}.png"
            pipe.generate(
                guidance.compose_prompt(subject, {}, fields=prompt_lib.TILE_FIELDS),
                path,
                seed=ground.texture_seed(42, index, role),
                tile=True,
            )
            with Image.open(path) as opened:
                opened.load()
                out[(index, role)] = np.asarray(opened.convert("RGBA"))
    return out


def _seam_ratio(texture: np.ndarray) -> float:
    """How much worse the wrap seam is than an average interior step.

    One number rather than an absolute threshold, because a busy texture has a
    large interior step and a smooth one has a tiny one -- an absolute bound
    would be a test about how noisy the generation happened to be.
    """
    values = texture[:, :, :3].astype(np.int64)
    interior = float(np.abs(np.diff(values, axis=1)).mean())
    seam = float(np.abs(values[:, 0] - values[:, -1]).mean())
    return seam / max(interior, 1e-6)


def test_a_generated_texture_survives_the_reduction_seamlessly(painted):
    """The claim, at both ends: seamless at 1024 and still seamless at 32."""
    for key, full in painted.items():
        small = ground.reduce_texture(full, TILE, TILE)
        assert small.shape == (TILE, TILE, 4)
        # Generous, deliberately: what is being caught is a *broken* reduction
        # -- a resize or a crop puts a hard discontinuity at the wrap, which is
        # many times the interior step, not a fraction more.
        assert _seam_ratio(small) < 3.0, f"{key} seams after reduction"


def test_the_result_is_a_tileset_the_terrain_brush_can_use(painted):
    """End to end: four real textures in, a 47-column terrain set out."""
    fills = [ground.reduce_texture(painted[(i, ground.FILL)], TILE, TILE) for i in range(2)]
    borders = [ground.reduce_texture(painted[(i, ground.BORDER)], TILE, TILE) for i in range(2)]
    atlas = groundtex.compose_atlas(
        TILE, TILE, "orthogonal", groundtex.default_border(TILE, TILE), fills, borders
    )
    tileset = Tileset(
        name="dungeon (AI)",
        pixels=atlas,
        tile_w=TILE,
        tile_h=TILE,
        terrains=tuple(
            TerrainSpec(
                name=row["name"],
                fill=row["color"],
                outline=(*(part * 3 // 5 for part in row["color"][:3]), 255),
            )
            for row in TERRAINS
        ),
    )
    assert tileset.columns == blob.TILE_COUNT
    assert tileset.terrain_of(tileset.local_for(1, blob.FULL)) == 1


def test_the_reduction_keeps_the_texture_contrast(painted):
    """The defect the two-stage sampler fixed: the old box mean flattened a
    texture to ~0.55-0.78 of its art-resolution contrast, and the tiles read as
    near-solid colours. Thresholds are the pre-registered rule from
    ``docs/measurements/2026-08-17-ground-reduction.md``: per-channel std at
    least half the same texture's 128px box-mean reference, floor 6.0
    (observed minimum ratio 0.92, minimum std 15.8)."""

    def contrast(texture: np.ndarray) -> float:
        return float(
            np.mean([texture[:, :, c].astype(np.float64).std() for c in range(3)])
        )

    for key, full in painted.items():
        reference = ground._box_reduce(full, 128, 128)
        small = ground.reduce_texture(full, TILE, TILE)
        assert contrast(small) >= 0.5 * contrast(reference), key
        assert contrast(small) >= 6.0, key


def test_two_terrains_come_back_distinct(painted):
    """Stone and water must not reduce to the same near-mean mush. Floors are
    half the measured minimums (fill-vs-fill 32, fill-vs-border 18)."""

    def diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(
            np.abs(a[:, :, :3].astype(np.int64) - b[:, :, :3].astype(np.int64)).mean()
        )

    small = {
        key: ground.reduce_texture(full, TILE, TILE) for key, full in painted.items()
    }
    assert diff(small[(0, ground.FILL)], small[(1, ground.FILL)]) >= 15.0
    for index in range(2):
        assert diff(small[(index, ground.FILL)], small[(index, ground.BORDER)]) >= 8.0


def test_an_interior_tile_still_wraps_after_compositing(painted):
    """The map-level promise: two FULL tiles side by side are two consecutive
    periods of one surface, so the interior tile must carry the seam through."""
    fills = [ground.reduce_texture(painted[(0, ground.FILL)], TILE, TILE)]
    borders = [ground.reduce_texture(painted[(0, ground.BORDER)], TILE, TILE)]
    atlas = groundtex.compose_atlas(TILE, TILE, "orthogonal", 4, fills, borders)
    x0 = blob.FULL * TILE
    assert _seam_ratio(atlas[0:TILE, x0 : x0 + TILE]) < 3.0
