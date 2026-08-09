"""Combining per-view projections into one texture atlas.

The half of a re-texture that is arithmetic rather than Blender. `op_views`
renders the mesh from a fixed set of directions, the host restyles those
renders through SDXL img2img, and `op_project` bakes each restyled view back
into the mesh's own UV atlas *separately*, alongside a weight image saying how
much that view is entitled to say about each texel. What is left is a weighted
sum, and it lives here for the reason `pipelines/sheet.py`'s grid does: it is
decided on the host, it is a pure function, and a second copy of it inside
Blender would be a second set of conventions to keep in agreement with the
first.

Pure in the `vram.py` sense: stdlib and numpy, Pillow inside the functions that
touch files, nothing from `service`, `queue` or `studio`, `None` rather than an
exception. No bpy anywhere -- this is the host side, and `rigging.py`'s rule
applies to it unchanged.

**Three rules here are the difference between a texture and a mess**, and each
one is a thing that goes wrong silently.

*A texel no view could see keeps its original colour.* Zero total weight means
"nothing looked at this", which is not the same as "this is black" -- it
happens inside every fold and under every overhang. Filling those with the sum
(which is zero) turns the inside of a barrel into a void; filling them with the
source texture leaves the mesh looking like what it was, which is the honest
answer for a surface the restyle never saw.

*The result is dilated past every island edge.* A UV island's texels stop at
its boundary, and a GPU sampling that boundary bilinearly mixes in whatever is
outside it -- so an atlas that is exactly the islands renders with a dark rim
around every one of them, at every mip level, worse the further you stand. The
fix is the standard one and it is not optional.

*There is no occlusion test, and that is a stated limitation rather than an
oversight.* The weight is a facing ratio, so a surface pointing at a camera
gets that camera's pixels whether or not anything is in the way. On a convex
prop this is right; on anything with an overhang the front view's colours are
smeared onto geometry hidden behind it. It is written down here, reported in
the job's params, and it is the finding that decides whether Tier 3 is needed
-- so it must not be quietly papered over with a blur.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# The directions the mesh is rendered and projected from, as (yaw, pitch) in
# degrees with yaw 0 looking along +Y in Blender -- `sheet.py`'s convention,
# imported in spirit rather than in code because that module's yaws are a
# sprite sheet's row and these are a projection basis.
#
# Six, and they are the axis directions rather than an even sphere packing: a
# UV-atlased reconstruction is overwhelmingly axis-aligned in practice, the six
# are what a person means by "front, back, both sides, top, bottom", and every
# additional view costs a full SDXL pass. The set is data so a caller can ask
# for fewer without this module learning about it.
VIEWS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),  # front
    (180.0, 0.0),  # back
    (90.0, 0.0),  # right
    (270.0, 0.0),  # left
    (0.0, 89.0),  # top
    (0.0, -89.0),  # bottom
)

# Below this a view is treated as saying nothing about a texel. A facing ratio
# near zero is a surface seen edge-on, where one rendered pixel covers a long
# strip of surface and the colour it reports is a smear of everything along it.
# Including those at their tiny weight is worse than excluding them, because
# they are exactly the texels no other view covers either -- so a smear would
# win by default wherever it was the only contributor.
MIN_FACING = 0.15

# How far the finished atlas is grown past the islands, in texels. Four covers
# bilinear sampling plus two mip levels, which is where a rim first becomes
# visible; growing further costs a pass each and never helps, because by the
# mip level that would need it the whole island is a few texels wide anyway.
DILATE = 4


def view_matrix(yaw: float, pitch: float) -> tuple[float, float, float]:
    """The unit direction a view looks *from*, in Blender axes.

    Yaw 0 on +Y and pitch positive upward, so the templates' "forward is -Y"
    convention puts yaw 0 in front of the subject -- the same sentence
    `sheet.py` carries about column 0, and it has to stay true here or a
    re-texture's front view would be the mesh's back.
    """
    y, p = math.radians(yaw), math.radians(pitch)
    return (
        math.sin(y) * math.cos(p),
        -math.cos(y) * math.cos(p),
        math.sin(p),
    )


def _read(path: Path, mode: str) -> Any | None:
    import numpy as np

    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            return np.asarray(im.convert(mode), dtype=np.float32) / 255.0
    except Exception:
        return None


def combine(colours: Any, weights: Any, base: Any) -> Any:
    """Weighted mean of N projections, falling back to ``base``.

    ``colours`` is (n, h, w, 3), ``weights`` is (n, h, w), ``base`` is
    (h, w, 3) -- the atlas the mesh already has. Everything in 0..1.

    Weights below ``MIN_FACING`` are dropped rather than scaled down, and a
    texel left with no contributor at all keeps its base colour. Both are the
    same decision: an answer nothing actually saw must not be invented, and the
    mesh's own texture is the one honest thing to say instead.
    """
    import numpy as np

    w = np.where(weights >= MIN_FACING, weights, 0.0).astype(np.float32)
    total = w.sum(axis=0)
    # The sum is computed everywhere and used only where it means something;
    # dividing under a where() rather than after it keeps the zero-weight
    # texels out of the divide entirely instead of relying on errstate.
    safe = np.where(total > 0.0, total, 1.0)
    mixed = (colours * w[..., None]).sum(axis=0) / safe[..., None]
    return np.where((total > 0.0)[..., None], mixed, base).astype(np.float32)


def dilate(image: Any, mask: Any, passes: int = DILATE) -> Any:
    """Grow ``image`` outward into the texels ``mask`` says are unwritten.

    One texel per pass, from the four neighbours that are already written --
    the cheapest thing that removes an island rim, and the only property that
    matters is that a sampler reaching just past an island edge finds the
    island's own colour rather than the background.

    Deliberately *not* wrapped, unlike everything in `material.py`: a UV atlas
    is not periodic. Rolling here would carry the left edge's islands onto the
    right, which is a bleed between unrelated parts of the mesh.
    """
    import numpy as np

    out = image.astype(np.float32, copy=True)
    filled = mask.astype(bool, copy=True)
    for _ in range(max(0, passes)):
        if filled.all():
            break
        acc = np.zeros_like(out)
        count = np.zeros(filled.shape, dtype=np.float32)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            # Shifted with an explicit zero edge rather than np.roll, so a
            # neighbour "off the top" contributes nothing instead of the
            # bottom row.
            src = np.zeros_like(out)
            have = np.zeros(filled.shape, dtype=bool)
            if axis == 0:
                if shift == 1:
                    src[1:], have[1:] = out[:-1], filled[:-1]
                else:
                    src[:-1], have[:-1] = out[1:], filled[1:]
            elif shift == 1:
                src[:, 1:], have[:, 1:] = out[:, :-1], filled[:, :-1]
            else:
                src[:, :-1], have[:, :-1] = out[:, 1:], filled[:, 1:]
            acc += src * have[..., None]
            count += have
        grow = (~filled) & (count > 0)
        if not grow.any():
            break
        out[grow] = acc[grow] / count[grow][..., None]
        filled |= grow
    return out


def assemble(
    view_dir: Path,
    base_path: Path | None,
    dest: Path,
    *,
    count: int,
) -> dict[str, Any] | None:
    """Combine ``count`` baked (colour, weight) pairs into one atlas PNG.

    Returns a small report -- how much of the atlas any view actually covered,
    and how much is base texture showing through -- or ``None`` if the inputs
    cannot be read. The coverage figure is the honest half of the "no occlusion
    test" limitation: a low number means most of the mesh kept its old skin,
    which is exactly what a caller should be told rather than left to discover.
    """
    import numpy as np
    from PIL import Image

    colours, weights = [], []
    for index in range(count):
        colour = _read(view_dir / f"bake_{index:02d}.png", "RGB")
        weight = _read(view_dir / f"weight_{index:02d}.png", "L")
        if colour is None or weight is None:
            continue
        colours.append(colour)
        weights.append(weight)
    if not colours:
        return None
    shape = colours[0].shape
    if any(c.shape != shape for c in colours):
        # A mismatched bake is a Blender-side bug, and averaging whatever
        # happens to broadcast would hide it in a texture nobody can read back.
        return None

    base = _read(base_path, "RGB") if base_path is not None else None
    if base is None or base.shape != shape:
        # No previous texture to fall back on, or one of another size. Mid grey
        # rather than black: an unreached texel should read as "nothing was
        # said about this", and black reads as a hole in every renderer.
        base = np.full(shape, 0.5, dtype=np.float32)

    stack_c = np.stack(colours)
    stack_w = np.stack(weights)
    mixed = combine(stack_c, stack_w, base)
    covered = (np.where(stack_w >= MIN_FACING, stack_w, 0.0).sum(axis=0) > 0.0)
    out = dilate(mixed, covered)

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB").save(
        dest, "PNG"
    )
    return {
        "views": len(colours),
        "coverage": float(covered.mean()),
        "size": [int(shape[1]), int(shape[0])],
        # Said in the record as well as in this module's docstring, because the
        # record is what a user reading the job's params sees.
        "occlusion_tested": False,
    }
