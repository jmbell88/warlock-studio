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

*The occlusion test is optional, and the report never lies about it.* Without
``occlusion`` the weight is a facing ratio alone, so a surface pointing at a
camera gets that camera's pixels whether or not anything is in the way -- the
behaviour the 2026-08-08 measurement measured, kept reachable as the probe's
A/B baseline. With it, each view's weight is multiplied by a per-texel depth
compare (``visibility``): the worker's third bake records what depth the
camera actually saw at each texel's pixel next to the texel's own, and a texel
behind a wall keeps its old skin instead of wearing the wall's colours.
``occlusion_tested`` in the report is True only when depth pairs were actually
consumed, because that flag is the finding that decides whether Tier 3 is
needed -- it must never be papered over.
"""

from __future__ import annotations

import contextlib
import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# The directions the mesh is rendered and projected from, as (yaw, pitch) in
# degrees with yaw 0 looking along +Y in Blender -- `sheet.py`'s convention,
# imported in spirit rather than in code because that module's yaws are a
# sprite sheet's row and these are a projection basis.
#
# Ten: the six axis directions plus the four upper 3/4 diagonals. The axis six
# are what a person means by "front, back, both sides, top, bottom" and a
# UV-atlased reconstruction is overwhelmingly axis-aligned in practice; the
# diagonals are the directions that look *through* a perforated shell onto the
# interior walls the axis set either misses or -- before the visibility test --
# smeared. Every additional view costs a full SDXL pass, which is why the
# under-side diagonals are not here: the straight-down bottom view measurably
# added ~0 coverage (docs/measurements/2026-08-08-retexture-bake.md), so
# their upside-down cousins have to earn a place in the next measurement
# rather than being presumed. The set is data so a caller can ask for fewer
# without this module learning about it.
VIEWS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),  # front
    (180.0, 0.0),  # back
    (90.0, 0.0),  # right
    (270.0, 0.0),  # left
    (0.0, 89.0),  # top
    (0.0, -89.0),  # bottom
    (45.0, 35.0),  # upper 3/4 diagonals
    (135.0, 35.0),
    (225.0, 35.0),
    (315.0, 35.0),
)

# Below this a view is treated as saying nothing about a texel. A facing ratio
# near zero is a surface seen edge-on, where one rendered pixel covers a long
# strip of surface and the colour it reports is a smear of everything along it.
# Including those at their tiny weight is worse than excluding them, because
# they are exactly the texels no other view covers either -- so a smear would
# win by default wherever it was the only contributor.
MIN_FACING = 0.15

# The self-occlusion bias band, in units of the normalised depth encoding. A
# texel is visible when the depth the camera recorded at its pixel is its own
# depth; "its own" has slack because the depth pair crosses an 8-bit PNG twice
# (one step is 1/255 ~ 0.8% of the framed range) and is bilinearly sampled once.
# Below LO the difference is quantisation and the texel is fully visible; above
# HI something genuinely nearer was in the way; between them a smoothstep fades
# so the frontier is a blend rather than a dotted line of acne. Provisional
# until docs/measurements pins them (the retexture-visibility doc), like every
# constant the corpus is keyed on.
DEPTH_EPS_LO = 2.0 / 255.0
DEPTH_EPS_HI = 6.0 / 255.0

# The feathered frontier between repainted and kept texels. Total effective
# weight below FEATHER_TOTAL fades the repaint in rather than switching it on,
# and the blend map is blurred a couple of texels so an occlusion boundary
# inside a UV island never shows as a hard line. Two passes reach two texels,
# safely inside DILATE's four -- whatever bleeds lands on texels dilation
# overwrites anyway.
FEATHER_TOTAL = 0.5
FEATHER_BLUR_PASSES = 2

# The atlas a re-texture writes, in texels. Larger than trellis's own 512
# (`Config.trellis_tex_res`, pinned there to dodge an upstream noise bug) and
# deliberately so: the UV layout is unchanged, but the views being projected
# into it are 1024px, so a 512 atlas would throw away detail the restyle
# actually produced. The sizes a caller may ask for are stated rather than free,
# because this number is baked into every derived export afterwards.
TEXTURE_PX = 1024
TEXTURE_SIZES: tuple[int, ...] = (512, 1024, 2048)

# Which of ``service.files.DERIVED`` a re-texture invalidates: the exports that
# carry the mesh's *skin*. STL is geometry only and a convex collision hull has
# no material at all, so deleting those would cost the user a re-export to
# produce a byte-identical file -- and a re-texture changes no geometry, which
# is the same sentence that keeps the rig valid.
#
# It lives here rather than in ``service.files`` because ``queue.py`` is what
# deletes them and may not import ``service``. ``files.DERIVED`` stays the
# authority on the whole set; this is a stated subset of it, and
# ``tests/test_retexture.py`` asserts the containment in both directions so a
# new export cannot join one list and quietly miss the other.
SURFACE_DERIVED: tuple[str, ...] = ("model_obj.zip", "textures.zip", "model.fbx")

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
        # None is this module's contract, but a silent one leaves a re-texture
        # refusal with nothing behind it: which of twelve files was unreadable,
        # and why, is only knowable here.
        log.exception("could not read %s as %s", path, mode)
        return None


def depth_encode(d: float, extent: float, distance: float) -> float:
    """Camera depth ``d`` in the inverted near=1 / far=0 encoding.

    The worker's shader nodes encode with ``(offset - d) * scale`` from
    ``blender_worker._depth_terms``; this is the host's copy of the same
    arithmetic, pinned against it by ``tests/test_retexture.py`` exactly as
    ``view_matrix`` is pinned against ``_view_direction``. Inverted on
    purpose: a pixel where nothing rendered decodes to 0, the far plane, so
    "no occluder here" needs no special case anywhere downstream.
    """
    span = max(2.0 * extent, 1e-9)
    return min(max(((distance + extent) - d) / span, 0.0), 1.0)


def visibility(zread: Any, zsurf: Any) -> Any:
    """How much a view is allowed to say about texels it may not have seen.

    ``zread`` is the depth the camera recorded at this texel's pixel and
    ``zsurf`` is the texel's own camera depth, both in the ``depth_encode``
    encoding. Equal means the texel *is* the nearest surface; ``zread``
    greater means something nearer was in the way, and past the bias band the
    view's opinion of this texel is a wall's. The band is a smoothstep rather
    than a threshold so quantisation at the frontier fades instead of
    stippling.
    """
    import numpy as np

    diff = zread.astype(np.float32) - zsurf.astype(np.float32)
    t = np.clip(
        (diff - DEPTH_EPS_LO) / (DEPTH_EPS_HI - DEPTH_EPS_LO), 0.0, 1.0
    )
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)


def feather_blend(total: Any) -> Any:
    """How much of the repaint each texel receives, from its total weight.

    A smoothstep to ``FEATHER_TOTAL`` and a short non-wrapping blur: the point
    is that the frontier between "views repainted this" and "this keeps its
    old skin" -- which the visibility test draws *inside* UV islands, not only
    at their edges -- arrives as a fade rather than a line.
    """
    import numpy as np

    t = np.clip(total.astype(np.float32) / FEATHER_TOTAL, 0.0, 1.0)
    blend = t * t * (3.0 - 2.0 * t)
    for _ in range(FEATHER_BLUR_PASSES):
        acc = blend.copy()
        count = np.ones(blend.shape, dtype=np.float32)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            # Shifted with an explicit zero edge, `dilate`'s pattern and for
            # its reason: a UV atlas is not periodic.
            src = np.zeros_like(blend)
            have = np.zeros(blend.shape, dtype=bool)
            if axis == 0:
                if shift == 1:
                    src[1:], have[1:] = blend[:-1], True
                else:
                    src[:-1], have[:-1] = blend[1:], True
            elif shift == 1:
                src[:, 1:], have[:, 1:] = blend[:, :-1], True
            else:
                src[:, :-1], have[:, :-1] = blend[:, 1:], True
            acc += src
            count += have
        blend = acc / count
    return blend


def depth_hint(depth_png: Path, view_png: Path, dest: Path) -> bool:
    """Turn a rendered depth pass into a ControlNet hint. -> whether it worked.

    What controlnet-depth-sdxl was trained on is inverted *relative* depth:
    nearest point white, furthest black, background black. The render is
    already inverted (``depth_encode``), so the work here is stretching the
    subject's own range to full scale -- a shallow prop must not whisper --
    and masking the background to zero with the colour render's alpha, which
    is the same alpha ``op_project`` treats as the subject mask.

    False rather than an exception for every way of not working, this
    module's rule; the caller decides whether a missing hint fails the job.
    """
    import numpy as np
    from PIL import Image

    depth = _read(depth_png, "L")
    view = _read(view_png, "RGBA")
    if depth is None or view is None or depth.shape != view.shape[:2]:
        return False
    mask = view[..., 3] > 0.5
    if mask.any():
        lo = float(depth[mask].min())
        hi = float(depth[mask].max())
        span = hi - lo
        norm = (depth - lo) / span if span > 1e-6 else np.ones_like(depth)
    else:
        norm = np.zeros_like(depth)
    hint = np.clip(norm, 0.0, 1.0) * mask
    arr = (hint * 255.0 + 0.5).astype(np.uint8)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB").save(dest, "PNG")
    except OSError:
        log.exception("could not write the depth hint %s", dest)
        return False
    return True


def combine(colours: Any, weights: Any, base: Any, vis: Any | None = None) -> Any:
    """Weighted mean of N projections, falling back to ``base``.

    ``colours`` is (n, h, w, 3), ``weights`` is (n, h, w), ``base`` is
    (h, w, 3) -- the atlas the mesh already has. Everything in 0..1.
    ``vis``, when given, is (n, h, w) from ``visibility`` and multiplies each
    view's say *after* the facing floor: an occluded texel of a square-on view
    is a wall's opinion at any facing, and a sub-floor facing stays dropped no
    matter how visible.

    Weights below ``MIN_FACING`` are dropped rather than scaled down, and a
    texel left with no contributor at all keeps its base colour. Both are the
    same decision: an answer nothing actually saw must not be invented, and the
    mesh's own texture is the one honest thing to say instead.
    """
    import numpy as np

    w = np.where(weights >= MIN_FACING, weights, 0.0).astype(np.float32)
    if vis is not None:
        w = w * vis.astype(np.float32)
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
    occlusion: bool = False,
) -> dict[str, Any] | None:
    """Combine ``count`` baked (colour, weight) pairs into one atlas PNG.

    Returns a small report -- how much of the atlas any view actually covered,
    and how much is base texture showing through -- or ``None`` if the inputs
    cannot be read. A low coverage number means most of the mesh kept its old
    skin, which is exactly what a caller should be told rather than left to
    discover.

    With ``occlusion`` a ``depthpair_NN.png`` is read beside every bake (R =
    the depth the camera recorded at this texel's pixel, G = the texel's own
    depth, both from the worker's third bake) and each view's weight is
    multiplied by ``visibility`` -- so a square-on surface behind a wall keeps
    its old skin instead of wearing the wall's colours, and the repaint fades
    in at the frontier through ``feather_blend``. Without it, the behaviour is
    byte-identical to what the 2026-08-08 measurement measured: that path is
    the probe's A/B baseline and must stay reachable.
    """
    import numpy as np
    from PIL import Image

    if count <= 0:
        return None
    colours, weights = [], []
    for index in range(count):
        colour = _read(view_dir / f"bake_{index:02d}.png", "RGB")
        weight = _read(view_dir / f"weight_{index:02d}.png", "L")
        if colour is None or weight is None:
            continue
        colours.append(colour)
        weights.append(weight)
    if len(colours) != count:
        # Every requested pair, or none. A partial set averages fine and looks
        # like a finished re-texture, but the views that failed to read back
        # are exactly the ones whose surface keeps its old skin -- so what
        # ships is a mesh restyled on three sides, with nothing anywhere
        # saying so. Refusing fails the job, which is a thing a user can act
        # on. Still ``None`` rather than an exception: this module's contract
        # is stated in its docstring and the caller turns it into the error.
        return None
    shape = colours[0].shape
    if any(c.shape != shape for c in colours):
        # A mismatched bake is a Blender-side bug, and averaging whatever
        # happens to broadcast would hide it in a texture nobody can read back.
        return None

    vis = None
    if occlusion:
        pairs = []
        for index in range(count):
            pair = _read(view_dir / f"depthpair_{index:02d}.png", "RGB")
            if pair is None or pair.shape != shape:
                # The same all-or-nothing rule as the bakes: a view whose
                # depth pair vanished is a view whose smear went back to
                # being invisible.
                return None
            pairs.append(visibility(pair[..., 0], pair[..., 1]))
        vis = np.stack(pairs)

    base = _read(base_path, "RGB") if base_path is not None else None
    if base is None or base.shape != shape:
        # No previous texture to fall back on, or one of another size. Mid grey
        # rather than black: an unreached texel should read as "nothing was
        # said about this", and black reads as a hole in every renderer.
        base = np.full(shape, 0.5, dtype=np.float32)

    stack_c = np.stack(colours)
    stack_w = np.stack(weights)
    mixed = combine(stack_c, stack_w, base, vis=vis)
    floored = np.where(stack_w >= MIN_FACING, stack_w, 0.0)
    if vis is not None:
        floored = floored * vis
    total = floored.sum(axis=0)
    covered = total > 0.0
    if occlusion:
        blend = feather_blend(total)
        mixed = base + (mixed - base) * blend[..., None]
    out = dilate(mixed, covered)

    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB").save(
        dest, "PNG"
    )
    report = {
        "views": len(colours),
        "coverage": float(covered.mean()),
        "size": [int(shape[1]), int(shape[0])],
        # Said in the record as well as in this module's docstring, because the
        # record is what a user reading the job's params sees -- and truthful:
        # True only when depth pairs were actually consumed.
        "occlusion_tested": bool(occlusion),
    }
    if occlusion:
        report["coverage_effective"] = float((blend > 0.5).mean())
    return report


# --- putting the atlas back into the GLB -------------------------------------
#
# Through ``glbio`` rather than through trimesh, and for the reason
# ``normalize_glb`` gives about the same choice: re-exporting a scene would
# re-encode every texture and rewrite the node graph -- and the node graph is
# where the grounding transform lives. Only the JSON chunk and the tail of the
# BIN chunk are touched; every existing byte is copied through verbatim.


def _first_base_colour(gltf: dict) -> int | None:
    """The index of the *texture* the first textured material samples as albedo."""
    for material in gltf.get("materials") or []:
        pbr = material.get("pbrMetallicRoughness") or {}
        slot = pbr.get("baseColorTexture") or {}
        index = slot.get("index")
        if isinstance(index, int):
            return index
    return None


def _has_uvs(gltf: dict) -> bool:
    """Whether anything in the file has the ``TEXCOORD_0`` a texture is sampled by.

    The one hard requirement for giving an *untextured* mesh a skin: without UVs
    there is nowhere for the atlas to land, and attaching one anyway produces a
    file that loads, renders a single texel's colour over everything, and looks
    like the restyle failed rather than like the mesh was never unwrapped.
    """
    return any(
        "TEXCOORD_0" in (primitive.get("attributes") or {})
        for mesh in gltf.get("meshes") or []
        for primitive in mesh.get("primitives") or []
    )


def _image_bytes(gltf: dict, buffer: bytes, image_index: int) -> bytes | None:
    image = (gltf.get("images") or [])[image_index]
    view_index = image.get("bufferView")
    if not isinstance(view_index, int):
        # An external or data: URI. trellis writes neither, and chasing one
        # would mean resolving paths relative to a GLB that may have moved.
        return None
    view = (gltf.get("bufferViews") or [])[view_index]
    start = int(view.get("byteOffset", 0))
    return buffer[start : start + int(view["byteLength"])]


def atlas_size(glb_path: Path) -> int | None:
    """The square edge of the mesh's current albedo, or None.

    What a re-texture should bake at unless told otherwise, and the measurement
    is the reason it is asked rather than assumed
    (``docs/measurements/2026-08-08-retexture-bake.md``). A trellis mesh's
    atlas is 2048; baking at a fixed 1024 made no visible difference to the
    parts a view covered -- and quietly halved the resolution of the ~63% that
    no view covered and therefore *kept its old colour*, which is the half a
    re-texture is supposed to leave alone.

    Non-square or unreadable gives None: the whole pipeline is square, and a
    caller falling back to ``TEXTURE_PX`` is a better answer than one atlas
    axis silently deciding both.
    """
    import io

    from PIL import Image

    from .. import glbio

    try:
        gltf, buffer = glbio.read_glb(glb_path)
        texture_index = _first_base_colour(gltf)
        if texture_index is None:
            return None
        source = (gltf.get("textures") or [])[texture_index].get("source")
        if not isinstance(source, int):
            return None
        raw = _image_bytes(gltf, buffer, source)
        if raw is None:
            return None
        with Image.open(io.BytesIO(raw)) as im:
            width, height = im.size
    except Exception:
        # Falling back to TEXTURE_PX is a decision worth a line in the log: it
        # silently halves the resolution of the part a re-texture leaves alone.
        log.exception("could not measure the albedo of %s", glb_path)
        return None
    return int(width) if width == height else None


def extract_base_colour(glb_path: Path, dest: Path, *, size: int) -> bool:
    """Write the mesh's current albedo out as a square PNG. -> whether it worked.

    ``size`` is the bake resolution rather than the atlas's own, because this
    image exists to be ``assemble``'s ``base``: a texel with no contributor
    keeps its old colour, and "its old colour" has to be addressable at the
    resolution the projections were baked at. Resampling here rather than in
    ``assemble`` keeps that function a pure array operation.

    False rather than an exception for every way of not having one -- an
    untextured mesh is a legitimate input, and ``assemble`` already has an
    honest stand-in for a missing base.
    """
    from PIL import Image

    from .. import glbio

    try:
        gltf, buffer = glbio.read_glb(glb_path)
        texture_index = _first_base_colour(gltf)
        if texture_index is None:
            return False
        source = (gltf.get("textures") or [])[texture_index].get("source")
        if not isinstance(source, int):
            return False
        raw = _image_bytes(gltf, buffer, source)
        if raw is None:
            return False
        import io

        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            atlas = im.convert("RGB")
        if atlas.size != (size, size):
            atlas = atlas.resize((size, size), Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(dest, "PNG")
    except Exception:
        # False is a legitimate answer here (an untextured mesh has no base),
        # which is exactly why a *failure* to read one that exists must not
        # look identical to it in the log.
        log.exception("could not extract the base colour of %s", glb_path)
        return False
    return True


def swap_base_colour(glb_path: Path, atlas_png: Path, dest: Path) -> bool:
    """Copy ``glb_path`` to ``dest`` with ``atlas_png`` as its albedo.

    The new PNG is **appended** and pointed at through a new image, a new
    texture and a rewritten ``baseColorTexture.index`` on every material that
    had one -- rather than overwriting the old image in place. Overwriting is
    the shorter code and it is wrong whenever anything else references the same
    image or the same texture: a mesh whose metallic-roughness map happens to
    be the same PNG would have its roughness replaced by an albedo, silently,
    and only some meshes would show it. The orphaned original stays in the
    buffer, which costs a megabyte and keeps every other offset exactly where
    it was.

    False rather than an exception, this module's rule: a mesh with no albedo
    slot to replace is a fact about the mesh.
    """
    import struct

    from .. import glbio

    try:
        data = glb_path.read_bytes()
        header, gltf, rest = glbio.split_glb(data)
        texture_index = _first_base_colour(gltf)
        if texture_index is None and not _has_uvs(gltf):
            # Nothing to attach a skin to. A Clay-built asset arrives with
            # materials and no textures, which is fine; one with no UVs at all
            # is not, and attaching an atlas anyway would render one texel's
            # colour over the whole mesh and read as a failed restyle.
            return False

        # The BIN chunk, and everything after it kept verbatim.
        offset, bin_at = 0, None
        while offset + 8 <= len(rest):
            chunk_len, chunk_type = struct.unpack_from("<II", rest, offset)
            if chunk_type == glbio.CHUNK_BIN:
                bin_at = (offset, chunk_len)
                break
            offset += 8 + chunk_len
        if bin_at is None:
            return False
        start, length = bin_at
        body = rest[start + 8 : start + 8 + length]
        tail = rest[start + 8 + length :]

        png = atlas_png.read_bytes()
        # 4-byte alignment is required of every bufferView offset, and the BIN
        # chunk itself pads with zeros (the JSON chunk is the one that pads with
        # spaces).
        padded = body + b"\0" * (-len(body) % 4)
        new_body = padded + png
        new_body += b"\0" * (-len(new_body) % 4)

        views = gltf.setdefault("bufferViews", [])
        views.append({"buffer": 0, "byteOffset": len(padded), "byteLength": len(png)})
        images = gltf.setdefault("images", [])
        images.append({"bufferView": len(views) - 1, "mimeType": "image/png"})
        textures = gltf.setdefault("textures", [])
        new_texture = {"source": len(images) - 1}
        if texture_index is not None:
            sampler = textures[texture_index].get("sampler")
            if sampler is not None:
                new_texture["sampler"] = sampler
        textures.append(new_texture)
        if texture_index is None:
            # An untextured mesh gains the slot rather than being refused. This
            # is the Clay case and it is the interesting one: a hand-modelled
            # box carries a real box unwrap, where a reconstruction carries an
            # atlas of roughly one texel per triangle.
            materials = gltf.setdefault("materials", [])
            if not materials:
                materials.append({})
                for mesh in gltf.get("meshes") or []:
                    for primitive in mesh.get("primitives") or []:
                        primitive.setdefault("material", 0)
            for material in materials:
                pbr = material.setdefault("pbrMetallicRoughness", {})
                pbr["baseColorTexture"] = {"index": len(textures) - 1}
                # A tinted base colour multiplies the atlas, which would show as
                # the restyle coming back the wrong hue. The texture *is* the
                # colour now.
                pbr.pop("baseColorFactor", None)
        else:
            # Every material sampling the old texture as albedo, not only the
            # first: _first_base_colour found *a* slot, and a multi-material
            # mesh sharing one atlas would otherwise come back half restyled.
            for material in gltf.get("materials") or []:
                slot = (material.get("pbrMetallicRoughness") or {}).get(
                    "baseColorTexture"
                )
                if slot is not None and slot.get("index") == texture_index:
                    slot["index"] = len(textures) - 1
        buffers = gltf.setdefault("buffers", [{}])
        buffers[0]["byteLength"] = len(new_body)

        new_rest = (
            rest[:start]
            + struct.pack("<II", len(new_body), glbio.CHUNK_BIN)
            + new_body
            + tail
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(glbio.rebuild_glb(header, gltf, new_rest))
    except OSError:
        # An I/O failure is not "this mesh has no albedo slot", and returning
        # False for it made the caller say so -- sending the user to look at a
        # mesh that was fine. The two semantic ``return False`` paths above
        # keep the module's contract; this one is a fact about the disk.
        log.exception("could not write the re-skinned %s", dest)
        with contextlib.suppress(OSError):
            dest.unlink(missing_ok=True)
        raise
    except Exception:
        log.exception("could not re-skin %s", glb_path)
        with contextlib.suppress(OSError):
            dest.unlink(missing_ok=True)
        return False
    return True
