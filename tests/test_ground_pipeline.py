"""The AI ground set's pure half: prompts, seeds, and the reduction.

The reduction carries the feature. A generated texture is seamless on its own
1024px torus, and the only reason a painted field continues across cells is that
bringing it down to tile size *partitions* that torus rather than resampling it
-- so the test that matters here is the commuting one: reducing a tiled source
gives the tiled reduction, exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.pipelines import ground
from warlock.service.validation import MAX_SEED


def _noise(width: int, height: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)


# -- the colour hint ---------------------------------------------------------


@pytest.mark.parametrize(
    "rgb,expected",
    [
        ((0, 0, 0), "black"),
        ((255, 255, 255), "white"),
        ((74, 124, 63), "mossy green"),
        ((25, 45, 110), "deep blue"),
    ],
)
def test_a_colour_lands_on_its_own_name(rgb, expected):
    assert ground.colour_hint(rgb) == expected


def test_a_colour_hint_ignores_alpha():
    assert ground.colour_hint((60, 170, 80, 255)) == ground.colour_hint((60, 170, 80, 0))


def test_every_colour_gets_a_name():
    """No hole in the table: a hint that came back empty would produce a prompt
    with a dangling comma in it."""
    for value in range(0, 256, 51):
        assert ground.colour_hint((value, 255 - value, value // 2))


def test_the_hint_is_the_same_answer_every_time():
    assert ground.colour_hint((130, 130, 130)) == ground.colour_hint((130, 130, 130))


# -- the prompts -------------------------------------------------------------


def test_both_subjects_carry_the_colour_hint():
    """The two materials have to read as one terrain -- a grey rim around a
    green field is a different terrain, not a border."""
    fill, border = ground.texture_prompts("dungeon", "water", "still water", "", (50, 90, 200))
    assert "blue" in fill and "blue" in border


def test_the_border_falls_back_to_a_rim_of_the_fill_material():
    fill, border = ground.texture_prompts("dungeon", "stone", "cracked flagstone", "", (128,) * 3)
    assert border.startswith(ground.default_edge("cracked flagstone"))
    assert fill.startswith("cracked flagstone")


def test_an_explicit_edge_material_is_used_verbatim():
    _fill, border = ground.texture_prompts(
        "cave", "lava", "molten lava", "black basalt", (220, 40, 40)
    )
    assert border.startswith("black basalt")


def test_a_terrain_with_no_material_falls_back_to_its_name():
    fill, _border = ground.texture_prompts("cave", "moss", "", "", (60, 170, 80))
    assert fill.startswith("moss")


def test_an_empty_theme_leaves_no_dangling_separator():
    fill, border = ground.texture_prompts("", "stone", "flagstone", "", (128,) * 3)
    for text in (fill, border):
        assert not text.endswith(",")
        assert ", ," not in text


def test_both_subjects_end_with_the_detail_clause():
    """Last comma part, deliberately: the material leads (CLIP weighs early
    tokens hardest) and the clause is the same on every subject."""
    fill, border = ground.texture_prompts("dungeon", "stone", "flagstone", "", (128,) * 3)
    assert fill.endswith(ground.DETAIL_CLAUSE)
    assert border.endswith(ground.DETAIL_CLAUSE)


def test_the_prompts_are_a_function_of_their_arguments():
    args = ("dungeon", "stone", "flagstone", "", (128, 128, 128))
    assert ground.texture_prompts(*args) == ground.texture_prompts(*args)


# -- the seeds ---------------------------------------------------------------


def test_the_seed_copy_matches_the_service_that_validates_it():
    """Restated rather than imported -- a pipeline may not import the service --
    so it is pinned here instead."""
    assert ground.MAX_SEED == MAX_SEED


def test_every_derived_seed_is_a_legal_seed():
    for base in (0, 1, 42, MAX_SEED):
        for index in range(8):
            for role in ground.ROLES:
                value = ground.texture_seed(base, index, role)
                assert 0 <= value <= MAX_SEED


def test_no_two_textures_in_one_set_share_a_seed():
    """A fill and a border that came back looking like the same picture twice is
    the failure this fan-out exists to prevent."""
    seeds = [
        ground.texture_seed(42, index, role)
        for index in range(8)
        for role in ground.ROLES
    ]
    assert len(set(seeds)) == len(seeds)


def test_a_derived_seed_is_a_function_of_the_stored_one():
    assert ground.texture_seed(42, 0, ground.FILL) == ground.texture_seed(42, 0, ground.FILL)
    assert ground.texture_seed(42, 0, ground.FILL) != ground.texture_seed(43, 0, ground.FILL)


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError):
        ground.texture_seed(42, 0, "outline")


# -- the reduction -----------------------------------------------------------


def test_a_reduced_texture_is_exactly_the_asked_for_size():
    out = ground.reduce_texture(_noise(64, 64), 12, 7)
    assert out.shape == (7, 12, 4)
    assert out.dtype == np.uint8


def test_reducing_a_tiled_source_gives_the_tiled_reduction():
    """The torus promise. If this fails, every painted field grows a faint grid
    and no amount of prompt work fixes it."""
    source = _noise(64, 64)
    small = ground.reduce_texture(source, 10, 10)
    tiled = np.tile(source, (2, 2, 1))
    assert np.array_equal(ground.reduce_texture(tiled, 20, 20), np.tile(small, (2, 2, 1)))


def test_an_odd_target_still_partitions_the_source():
    """Every input pixel counted once and only once, which is what makes the
    reduction commute with tiling in the first place."""
    source = np.ones((100, 100, 4), dtype=np.uint8) * 200
    out = ground.reduce_texture(source, 13, 7)
    assert (out == 200).all()


def test_an_exact_small_divisor_is_a_pure_centre_sample():
    """8x8 -> 4x4 is m=2: the prefilter stage is the identity, so the result is
    the centre sample of each 2x2 block."""
    source = _noise(8, 8)
    out = ground.reduce_texture(source, 4, 4)
    assert np.array_equal(out, source[1::2, 1::2])


def test_the_prefilter_stage_is_capped_at_four():
    """1024 -> 32 must prefilter to 128 -- one mid pixel per LoRA art pixel --
    then sample; an uncapped m would sample the raw source and keep noise."""
    source = _noise(64, 64)
    mid = ground._box_reduce(source, 16, 16)
    assert np.array_equal(ground.reduce_texture(source, 4, 4), mid[2::4, 2::4])


def test_block_structured_art_survives_the_reduction():
    """The defect this change exists to fix: an extreme 8x8 art grid blown up
    4x reduces back to *members of its own value set*, where the old box mean
    averaged neighbouring art pixels into colours the art never contained."""
    rng = np.random.default_rng(3)
    art = rng.choice(np.array([0, 255], dtype=np.uint8), size=(8, 8, 4))
    source = np.kron(art, np.ones((4, 4, 1), dtype=np.uint8))
    out = ground.reduce_texture(source, 8, 8)
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert np.array_equal(out, art)


def test_reducing_to_the_same_size_changes_nothing():
    source = _noise(16, 16)
    assert np.array_equal(ground.reduce_texture(source, 16, 16), source)


def test_the_reduction_is_deterministic():
    source = _noise(64, 64)
    assert (
        ground.reduce_texture(source, 11, 11).tobytes()
        == ground.reduce_texture(source, 11, 11).tobytes()
    )


def test_enlarging_is_refused_naming_both_sizes():
    with pytest.raises(ValueError) as excinfo:
        ground.reduce_texture(_noise(8, 8), 32, 32)
    assert "8x8" in str(excinfo.value) and "32x32" in str(excinfo.value)


def test_a_flat_array_is_refused():
    with pytest.raises(ValueError):
        ground.reduce_texture(np.zeros((8, 8), dtype=np.uint8), 4, 4)


# -- the phase factor ---------------------------------------------------------


@pytest.mark.parametrize(
    "tile_w,tile_h,projection,expected",
    [
        (16, 16, "orthogonal", 4),
        (32, 32, "orthogonal", 4),
        (64, 64, "orthogonal", 4),
        (65, 65, "orthogonal", 2),
        (128, 128, "orthogonal", 2),
        (129, 32, "orthogonal", 1),
        (512, 512, "orthogonal", 1),
        (32, 32, "isometric", 1),
        (32, 32, "oblique", 4),
    ],
)
def test_the_variant_factor_table(tile_w, tile_h, projection, expected):
    """Pinned, because the stored ``ground`` block carries the answer and a
    moved table would repaint old sets differently on rerun."""
    assert ground.variant_factor(tile_w, tile_h, projection) == expected


def test_the_variant_period_always_fits_the_generation():
    for edge in range(4, 513):
        k = ground.variant_factor(edge, edge, "orthogonal")
        assert k * edge <= 1024


# -- the sidecar -------------------------------------------------------------


def test_the_sidecar_records_the_version_and_the_recipe():
    doc = ground.ground_sidecar(
        theme="dungeon",
        tile_w=32,
        tile_h=32,
        projection="orthogonal",
        border=4,
        colors=32,
        phases=4,
        terrains=[{"name": "stone", "material": "flagstone"}],
        palette=["#000000"],
        recipe={"base_model": "sdxl_cfg", "seed": 42},
        created=1.5,
    )
    assert doc["version"] == ground.GROUND_VERSION
    assert doc["phases"] == 4
    assert doc["image"] == "input.png"
    assert doc["recipe"]["base_model"] == "sdxl_cfg"
    assert doc["terrains"][0]["name"] == "stone"


def test_the_sidecar_copies_its_inputs():
    """A sidecar that aliased the caller's dicts is one a later ``merge_params``
    could edit after it was written."""
    terrains = [{"name": "stone"}]
    doc = ground.ground_sidecar(
        theme="t",
        tile_w=8,
        tile_h=8,
        projection="orthogonal",
        border=2,
        colors=16,
        phases=1,
        terrains=terrains,
        palette=[],
        recipe={},
        created=0.0,
    )
    terrains[0]["name"] = "sand"
    assert doc["terrains"][0]["name"] == "stone"
