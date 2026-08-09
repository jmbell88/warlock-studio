"""Albedo -> height, normal and roughness. Estimates, and they say so.

None of this is a measurement. A single diffuse image does not contain a
surface's geometry or its microfacet distribution, and no amount of filtering
recovers them -- what these functions produce is the convincing *illusion* the
whole games industry ships, derived from one stated assumption per map. The
assumptions are written next to each function rather than buried in a
constant, because the failure mode here is not a wrong number, it is a user
believing the normal map describes the rock.

That framing is the same one ``meshaudit`` and the quality badge are written
under: say what was actually established, and never let a derived figure wear
the authority of a measured one.

**The wrap is the load-bearing part.** These maps are derived from a tile whose
whole value is that it has no seam, and every operation here is a
neighbourhood operation -- a gradient, a box blur. Run one with clamped or
reflected edges and the derived map grows a seam the albedo does not have,
which is worse than not deriving it at all: the albedo tiles, so the failure
appears only once the material is on a surface, as a hard line in the lighting
with nothing in the colour to explain it. Every neighbourhood operation below
is ``np.roll``-based and therefore exactly periodic, and
``tests/test_material.py`` asserts it by deriving from a shifted albedo and
comparing against the shifted derivation.

Pure in the ``vram.py`` sense: stdlib and numpy, Pillow imported inside the
functions that touch files (as ``seam.py`` does), nothing from ``service``,
``queue`` or ``studio``, and no exception where a ``None`` will do. Written
against numpy alone with no native kernel behind it -- if one ever follows, the
rule is the repo's: this stays as the reference and the parity bar is
bit-identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Rec. 709 luminance. The same weights the sRGB primaries imply, so a grey ramp
# comes back as itself; a flat mean over RGB would make a saturated red brick
# read as darker relief than a grey stone of the same lightness.
LUMA = (0.2126, 0.7152, 0.0722)

# How much the normal map's slope is exaggerated. A gradient in *albedo* is a
# fraction of a unit per pixel, and taken literally the surface would be almost
# flat; every height-to-normal tool in existence has this knob and every one of
# them ships a default that is a matter of taste. 2.0 is this one's, and it is
# a default rather than a finding -- there is no corpus behind it and no
# measurement document is owed, because nothing is keyed on it.
DEFAULT_STRENGTH = 2.0

# Roughness is squeezed into this band rather than spanning 0..1. Neither end
# is ever the right answer for a generated material: 0.0 is a perfect mirror
# and 1.0 is a fully diffuse surface with no specular response at all, and a
# heuristic this coarse should not be claiming either.
ROUGHNESS_RANGE = (0.35, 0.95)

# The neighbourhood the roughness estimate measures detail over, in pixels. Big
# enough that a single hard edge does not dominate its own surroundings, small
# enough that grout lines and mortar stay distinct from the cells they separate.
ROUGHNESS_RADIUS = 4

# glTF's tangent-space convention, and it is declared rather than assumed
# because the other one exists and is silently wrong: glTF (and OpenGL) put +Y
# *up* in the texture, DirectX puts it down, and a map written with the wrong
# sign lights correctly from above and inverts every dent into a bump. The
# viewer and every engine these exports are aimed at read glTF.
GREEN_UP = True

# Metalness is not estimated. There is no signal for it in a diffuse image --
# a metal and a dark dielectric of the same colour are the same pixels -- so
# the exported material declares 0.0 and the user changes it if the surface is
# metal. A guess here would be indistinguishable from a measurement in the file
# it lands in, which is exactly the thing this module refuses to do.
METALLIC = 0.0


def _load_rgb(path: Path) -> Any | None:
    """The albedo as float32 in 0..1, or None if it cannot be read.

    None rather than an exception, the rule this module shares with
    ``vram.py``: a derived artifact that cannot be produced is a missing file,
    and the caller already has a sentence for that.
    """
    import numpy as np

    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        return None
    if arr.ndim != 3 or min(arr.shape[:2]) < 2 * ROUGHNESS_RADIUS + 1:
        # Too small to have a neighbourhood at all; the wrap would fold the
        # image onto itself several times and every estimate would be noise.
        return None
    return arr


def _wrap_box(a: Any, radius: int) -> Any:
    """A box blur that wraps, via a summed-area table on a tiled array.

    Separable and therefore two passes rather than ``(2r+1)**2``; wrapped by
    rolling rather than by padding, so the result is exactly periodic and a
    shifted input gives the shifted output to the bit.
    """
    import numpy as np

    out = a.astype(np.float64, copy=True)
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for offset in range(-radius, radius + 1):
            acc += np.roll(out, offset, axis=axis)
        out = acc / (2 * radius + 1)
    return out


def luminance(rgb: Any) -> Any:
    """Perceptual lightness of an RGB float image, 0..1."""
    import numpy as np

    return (rgb * np.asarray(LUMA, dtype=rgb.dtype)).sum(axis=2)


def height(rgb: Any) -> Any:
    """Height from albedo, normalised to 0..1.

    **The assumption is that darker is deeper**, which is true of the thing
    that makes most textures legible -- ambient occlusion in the crevices,
    baked into the diffuse image by whatever drew it -- and false wherever
    colour is doing something else. A white grout line between dark tiles comes
    out as a ridge rather than a groove. That is the assumption failing
    honestly, not a bug to be patched with special cases.

    Normalised over the tile's own range rather than against an absolute scale:
    the output is a relief, and its unit is "this texture's own contrast".
    """
    import numpy as np

    lum = luminance(rgb)
    low, high = float(lum.min()), float(lum.max())
    if high - low < 1e-6:
        # A flat albedo has no relief, and a half-grey plane is the honest
        # answer -- stretching sensor noise across the full range would invent
        # a surface out of nothing.
        return np.full(lum.shape, 0.5, dtype=np.float32)
    return ((lum - low) / (high - low)).astype(np.float32)


def normal(field: Any, strength: float = DEFAULT_STRENGTH) -> Any:
    """A tangent-space normal map from a height field. -> float32 (h, w, 3).

    Central differences, wrapped. The vector is ``(-dx*s, -dy*s, 1)``
    normalised: a *positive* slope in x means the surface rises to the right,
    so its normal leans to the left, which is where the leading minus comes
    from -- and getting that sign wrong produces a map that looks entirely
    plausible until it is lit from the side.

    Encoded 0..1 the way every normal map is, so flat is (0.5, 0.5, 1.0).
    """
    import numpy as np

    f = field.astype(np.float32)
    # Central difference over one pixel, so the divisor is 2. Rolling by -1 and
    # +1 is what makes the derivative periodic, which is the whole point.
    dx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) * 0.5
    dy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) * 0.5
    if GREEN_UP:
        # Array rows run downward and the texture's +Y runs up, so the y
        # derivative's sign flips exactly once, here.
        dy = -dy
    vec = np.stack([-dx * strength, -dy * strength, np.ones_like(f)], axis=2)
    length = np.sqrt((vec * vec).sum(axis=2, keepdims=True))
    return (vec / length * 0.5 + 0.5).astype(np.float32)


def roughness(rgb: Any) -> Any:
    """A roughness estimate from local detail. -> float32 (h, w) in 0..1.

    **The assumption is that fine detail means a rough surface** -- scratches,
    grain, weave, grit -- and that a locally smooth region is a locally smooth
    *surface*. It is the weakest assumption in this module and it is wrong in a
    recognisable way: a flat photograph of polished granite has strong local
    contrast from its speckle and comes out rough, and a blurry render of
    concrete comes out smooth. There is no ground truth here and nothing scores
    it; what the map buys is variation, which is what stops a generated
    material reading as plastic.

    Local standard deviation of luminance, wrapped, normalised over the tile's
    own range and squeezed into ``ROUGHNESS_RANGE``.
    """
    import numpy as np

    lum = luminance(rgb).astype(np.float64)
    mean = _wrap_box(lum, ROUGHNESS_RADIUS)
    # E[x^2] - E[x]^2, clipped because the two box passes are floating point
    # and a locally constant region lands a hair below zero.
    variance = np.maximum(_wrap_box(lum * lum, ROUGHNESS_RADIUS) - mean * mean, 0.0)
    detail = np.sqrt(variance)
    low, high = float(detail.min()), float(detail.max())
    lo, hi = ROUGHNESS_RANGE
    if high - low < 1e-9:
        # No variation to rank: one value for the whole tile, in the middle of
        # the band rather than at an end of it.
        return np.full(lum.shape, (lo + hi) * 0.5, dtype=np.float32)
    scaled = (detail - low) / (high - low)
    return (lo + scaled * (hi - lo)).astype(np.float32)


# -- file level --------------------------------------------------------------

# What ``maps`` produces, in the order a caller should offer them. Names are
# literals rather than a pattern for the reason ``files.MEDIA`` gives: they end
# up in an allowlist, and a pattern would put part of a filename back in a
# caller's hands.
MAP_NAMES = ("material_height.png", "material_normal.png", "material_roughness.png")


def _save_gray(array: Any, path: Path) -> None:
    import numpy as np
    from PIL import Image

    data = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data, "L").save(path, "PNG")


def _save_rgb(array: Any, path: Path) -> None:
    import numpy as np
    from PIL import Image

    data = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data, "RGB").save(path, "PNG")


def write_map(
    src: Path, dest: Path, name: str, *, strength: float = DEFAULT_STRENGTH
) -> Path | None:
    """Derive one named map from ``src`` onto ``dest``. -> dest, or None.

    One map per call rather than all three, because the caller derives under a
    per-artifact lock and a function that wrote three files would hold one
    artifact's lock while writing two others'.
    """
    rgb = _load_rgb(src)
    if rgb is None or name not in MAP_NAMES:
        return None
    if name == "material_height.png":
        _save_gray(height(rgb), dest)
    elif name == "material_normal.png":
        _save_rgb(normal(height(rgb), strength), dest)
    else:
        _save_gray(roughness(rgb), dest)
    return dest


def material_json(albedo: str = "input.png") -> dict[str, Any]:
    """A glTF 2.0 material referencing the four images by filename.

    A fragment rather than a whole glTF: it is what goes *in* a material
    library, and every engine these exports are aimed at reads either this
    shape or something a line of script away from it. ``metallicFactor`` is the
    declared 0.0 above, not an estimate.

    ``occlusionTexture`` is deliberately absent. The height map is not an
    ambient-occlusion map and naming it as one would be exactly the false
    authority this module exists to avoid -- height *is* offered, under
    ``KHR_materials_displacement``-shaped naming in ``extras`` where nothing
    will mistake it for a core PBR input.
    """
    return {
        "name": "warlock-material",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 2},
            "metallicFactor": METALLIC,
            "roughnessFactor": 1.0,
        },
        "normalTexture": {"index": 1},
        "extras": {
            "warlock": {
                "generated": True,
                "albedo": albedo,
                "height": "material_height.png",
                "normal": "material_normal.png",
                "roughness": "material_roughness.png",
                "note": (
                    "height, normal and roughness are estimated from the albedo; "
                    "they describe its contrast, not a measured surface"
                ),
            }
        },
    }
