"""PBR maps derived from an albedo, and the one rule that is not taste.

Everything `material.py` produces is an estimate, so most of what could be
asserted about it would only be pinning an opinion. What is *not* an opinion is
that these maps come from a seamless tile and must stay seamless: every
operation in there is a neighbourhood operation, and a clamped or reflected
edge puts a seam into the derived map that the albedo does not have -- visible
only once the material is on a surface, as a hard line in the lighting with
nothing in the colour to explain it.

The shift test is what proves it. Derivation commutes with a cyclic shift if
and only if every neighbourhood wraps, so `derive(roll(x)) == roll(derive(x))`
catches a clamped edge anywhere in the chain without having to know where the
chain's edges are.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import material


def _tile(size=32, seed=3):
    """A wrapping texture: sums of whole-period sinusoids, so it is exactly
    periodic in both axes rather than merely smooth."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] * (2 * np.pi / size)
    field = np.zeros((size, size), dtype=np.float64)
    for _ in range(4):
        fx, fy = rng.integers(1, 4, size=2)
        field += np.sin(fx * x + rng.random() * 6.28) * np.cos(fy * y + rng.random() * 6.28)
    field = (field - field.min()) / (field.max() - field.min())
    return np.stack([field, field * 0.8, field * 0.6], axis=2).astype(np.float32)


# -- the wrap ----------------------------------------------------------------


@pytest.mark.parametrize("shift", [(5, 0), (0, 7), (11, 13)])
def test_every_derivation_commutes_with_a_cyclic_shift(shift):
    rgb = _tile()
    rolled = np.roll(rgb, shift, axis=(0, 1))

    for fn in (material.height, material.roughness):
        assert np.allclose(fn(rolled), np.roll(fn(rgb), shift, axis=(0, 1)), atol=1e-6), fn

    field = material.height(rgb)
    got = material.normal(np.roll(field, shift, axis=(0, 1)))
    assert np.allclose(got, np.roll(material.normal(field), shift, axis=(0, 1)), atol=1e-6)


def test_a_gradient_that_wraps_produces_a_normal_map_with_no_edge_spike():
    """The failure a clamped derivative actually produces.

    A wrapping ramp -- a triangle wave -- has the same slope magnitude
    everywhere, so its normal map is two flat halves. A clamped difference at
    the seam column would put a single column of a different value there, which
    is precisely the artefact the wrap exists to prevent.
    """
    size = 32
    x = np.arange(size)
    ramp = np.abs((x * 2.0 / size) - 1.0)
    field = np.tile(ramp, (size, 1)).astype(np.float32)

    out = material.normal(field)
    # Column 0's normal must be one of the two slopes present elsewhere, not a
    # third value of its own.
    column = out[:, 0, 0]
    interior = np.unique(np.round(out[:, 1:-1, 0], 5))
    assert np.all(np.isin(np.round(column, 5), interior))


# -- what the maps mean ------------------------------------------------------


def test_a_flat_albedo_has_no_relief_rather_than_amplified_noise():
    flat = np.full((32, 32, 3), 0.42, dtype=np.float32)
    assert np.allclose(material.height(flat), 0.5)
    # And its normal is flat: (0.5, 0.5, 1.0) encoded.
    out = material.normal(material.height(flat))
    assert np.allclose(out[..., 0], 0.5)
    assert np.allclose(out[..., 1], 0.5)
    assert np.allclose(out[..., 2], 1.0)


def test_a_flat_albedo_gets_one_roughness_in_the_middle_of_the_band():
    flat = np.full((32, 32, 3), 0.42, dtype=np.float32)
    out = material.roughness(flat)
    lo, hi = material.ROUGHNESS_RANGE
    assert np.allclose(out, (lo + hi) * 0.5)


def test_roughness_never_claims_a_mirror_or_a_perfect_diffuser():
    """Both ends of 0..1 are claims this heuristic has not earned."""
    out = material.roughness(_tile())
    lo, hi = material.ROUGHNESS_RANGE
    assert out.min() >= lo - 1e-6
    assert out.max() <= hi + 1e-6


def test_detail_reads_rougher_than_a_smooth_region():
    rng = np.random.default_rng(0)
    arr = np.full((64, 64, 3), 0.5, dtype=np.float32)
    arr[:, 32:] += rng.normal(0, 0.15, size=(64, 32, 3)).astype(np.float32)
    out = material.roughness(np.clip(arr, 0, 1))
    # Sampled away from the two boundaries, which the wrap makes noisy on both
    # sides: this is about the interior of each half.
    assert out[:, 8:24].mean() < out[:, 40:56].mean()


def test_darker_is_deeper():
    """The stated assumption, pinned so a change to it is deliberate."""
    arr = np.zeros((16, 16, 3), dtype=np.float32)
    arr[:, 8:] = 1.0
    out = material.height(arr)
    assert out[:, :8].mean() < out[:, 8:].mean()


def test_a_bump_radiates_outward_in_gltfs_convention():
    """glTF's tangent-space convention, and the one that is silently wrong.

    Stated on a *bump* rather than on a ramp, because a ramp's answer is easy
    to get backwards in the test as well as in the code -- a surface rising
    towards the top of the image leans its normal towards the bottom, away from
    the high side. A bump has no such trap: its normals point outward from the
    summit in every direction at once, so above it green must read *up*
    (> 0.5) and below it *down*, and red must read left on the left. Under
    DirectX's convention the two green readings swap and nothing else moves,
    which is exactly how a wrong sign survives a still image.
    """
    size = 32
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2
    field = np.exp(-(((x - c) ** 2 + (y - c) ** 2) / (2 * 5.0**2))).astype(np.float32)

    out = material.normal(field)
    mid = size // 2
    assert out[mid - 6, mid, 1] > 0.5, "above the summit the normal must lean up"
    assert out[mid + 6, mid, 1] < 0.5, "below it, down"
    assert out[mid, mid - 6, 0] < 0.5, "left of it, left"
    assert out[mid, mid + 6, 0] > 0.5, "right of it, right"


# -- files -------------------------------------------------------------------


def _write(tmp_path, rgb):
    path = tmp_path / "input.png"
    Image.fromarray((rgb * 255).astype(np.uint8), "RGB").save(path)
    return path


@pytest.mark.parametrize("name", material.MAP_NAMES)
def test_each_map_writes_a_png_the_size_of_its_albedo(tmp_path, name):
    src = _write(tmp_path, _tile(64))
    out = material.write_map(src, tmp_path / name, name)
    assert out is not None
    with Image.open(out) as im:
        assert im.size == (64, 64)
        assert im.mode == ("RGB" if name == "material_normal.png" else "L")


def test_an_unreadable_albedo_is_a_none_rather_than_a_raise(tmp_path):
    bad = tmp_path / "input.png"
    bad.write_bytes(b"not a png")
    assert material.write_map(bad, tmp_path / material.MAP_NAMES[0], material.MAP_NAMES[0]) is None


def test_a_name_this_module_does_not_produce_is_a_none(tmp_path):
    src = _write(tmp_path, _tile())
    assert material.write_map(src, tmp_path / "x.png", "material_metal.png") is None


def test_a_tile_too_small_for_a_neighbourhood_is_refused(tmp_path):
    src = _write(tmp_path, _tile(4))
    assert material.write_map(src, tmp_path / material.MAP_NAMES[0], material.MAP_NAMES[0]) is None


def test_the_material_declares_metalness_rather_than_estimating_it():
    doc = material.material_json()
    assert doc["pbrMetallicRoughness"]["metallicFactor"] == material.METALLIC == 0.0


def test_the_material_does_not_present_height_as_occlusion():
    """Naming the height map as occlusionTexture is the exact false authority
    this module exists to avoid: an engine would then light with it."""
    doc = material.material_json()
    assert "occlusionTexture" not in doc
    assert doc["extras"]["warlock"]["height"] == "material_height.png"


def test_the_allowlist_and_the_module_agree_in_both_directions():
    """``files.MATERIAL_2D`` is written out, so it is asserted.

    `config.SETTINGS`'s rule: an allowlist assembled from somewhere else is one
    whose contents can change without the allowlist file being edited, so it is
    spelled out -- and spelling it out is only safe while something fails when
    the two disagree. Both directions, because a map added to the module and
    missed here would never be servable, and a name added here and missed there
    would reach `_derive_material` with no branch to answer it.
    """
    from warlock.service.files import MATERIAL_2D, MEDIA, TILE_2D, derived_2d_for

    assert set(MATERIAL_2D) == set(material.MAP_NAMES) | {"material.zip"}
    for name in MATERIAL_2D:
        assert name in MEDIA, f"{name} is not servable"
        assert name in TILE_2D
        assert name in derived_2d_for("tile")
        # And never off a reference: a cutout's input.png has a subject in it,
        # and a normal map of a barrel on a background is a normal map of the
        # background too.
        assert name not in derived_2d_for("reference")


def test_the_module_imports_nothing_from_the_app():
    """Pure in the vram.py sense; asserted rather than intended."""
    import pathlib

    source = pathlib.Path(material.__file__).read_text(encoding="utf-8")
    for banned in ("from ..service", "from ..queue", "from ..studio", "import torch"):
        assert banned not in source
