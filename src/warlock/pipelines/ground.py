"""The half of an AI ground set that is arithmetic and words.

Its sibling ``warlock.studio.plotter.groundtex`` composites the atlas and knows
nothing about diffusion; this module knows what to ask SDXL for and how to get a
1024px generation down to a tile, and knows nothing about the forty-seven cases.
The split is the one ``pixelsheet.py`` already draws, and it is enforced from
both sides: the plotter package may not import a pipeline, and this module is
pure -- stdlib at module scope, numpy and Pillow inside the functions that need
them, no torch, no service, no decision about where a file goes.

**The period is the tile.** A generated texture is seamless on a torus of its
own 1024px period; :func:`reduce_texture` brings it to exactly ``tile_w`` by
``tile_h`` by averaging a *partition* of that torus, which leaves it seamless on
the smaller one. That is the entire trick behind the map continuing a pattern
across cells: a tile samples its material at tile-local coordinates, so two
interior tiles side by side are two consecutive periods of one surface. Any
resampling that did not partition -- a bilinear resize, a crop -- would put a
faint grid on every painted field, and no amount of prompt work would fix it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Bumped when anything below changes what a given form produces: the prompts,
#: the seed derivation, or the reduction. Recorded in the sidecar so a set can
#: say which compiler drew it, the way ``PIXEL_SHEET_VERSION`` does.
GROUND_VERSION = 1

#: A fill material and a border material. The whole reason a generated set reads
#: as art rather than as a flat fill with a line around it.
TEXTURES_PER_TERRAIN = 2

#: The two roles, in the order they are generated for a terrain. Named because
#: the seed derivation, the file names and the sidecar all key on them.
FILL = "fill"
BORDER = "border"
ROLES: tuple[str, str] = (FILL, BORDER)

#: ``service.validation.MAX_SEED``, restated rather than imported: a pipeline
#: may not import the service layer. ``tests/test_ground_pipeline.py`` pins the
#: two together, so the copy cannot drift.
MAX_SEED = 2**31 - 1

#: The palette hint, as words. A terrain row already carries an RGBA fill that
#: the user picked, and the nearest name in this table is how that colour reaches
#: a text encoder -- "#4a7c3f" means nothing to CLIP and "mossy green" means
#: exactly the right thing. A fixed table rather than a colour-science library:
#: the answer has to be the same on every machine, and the resolution needed is
#: "which of twenty words", not a delta-E.
COLOUR_NAMES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (0, 0, 0)),
    ("dark grey", (64, 64, 64)),
    ("grey", (128, 128, 128)),
    ("pale grey", (192, 192, 192)),
    ("white", (255, 255, 255)),
    ("deep red", (128, 0, 0)),
    ("red", (220, 40, 40)),
    ("orange", (230, 130, 30)),
    ("brown", (120, 80, 45)),
    ("tan", (200, 170, 120)),
    ("gold", (220, 190, 60)),
    ("yellow", (240, 230, 90)),
    ("olive", (110, 120, 50)),
    ("mossy green", (74, 124, 63)),
    ("green", (60, 170, 80)),
    ("teal", (40, 140, 140)),
    ("blue", (50, 90, 200)),
    ("deep blue", (25, 45, 110)),
    ("purple", (110, 60, 160)),
    ("pink", (230, 140, 180)),
)


def colour_hint(rgb: Sequence[int]) -> str:
    """The nearest name in :data:`COLOUR_NAMES`, deterministically.

    Squared distance in plain RGB, ties broken by table order -- not because RGB
    distance is perceptually right, but because the answer must be a stable
    function of three integers on every machine, and the question being asked is
    only "which of twenty words".
    """
    values = tuple(int(part) for part in rgb)[:3]
    if len(values) != 3:
        raise ValueError("a colour hint takes three channels")
    red, green, blue = values
    best_name, best_distance = COLOUR_NAMES[0][0], None
    for name, (r, g, b) in COLOUR_NAMES:
        distance = (red - r) ** 2 + (green - g) ** 2 + (blue - b) ** 2
        if best_distance is None or distance < best_distance:
            best_name, best_distance = name, distance
    return best_name


def default_edge(material: str) -> str:
    """What a terrain's border material is called when nobody said.

    Shared with the pane so the placeholder a user sees is the string that would
    be used if they left it alone -- a form whose greyed-out suggestion is not
    what the absent value means is a form that lies.
    """
    text = str(material).strip() or "stone"
    return f"weathered {text} rim"


def texture_prompts(
    theme: str,
    name: str,
    material: str,
    edge: str,
    rgb: Sequence[int],
) -> tuple[str, str]:
    """``(fill subject, border subject)`` for one terrain.

    Subjects, not finished prompts: the caller runs each through
    ``guidance.compose_prompt`` with ``prompt.TILE_FIELDS``, which is what adds
    the seamless-tileable, flat-orthographic, no-focal-object clauses. Putting
    those here would mean a tile prompt assembled two ways.

    The colour hint goes in both, because the two materials have to read as the
    same terrain -- a grey rim around a green field is a different terrain, not
    a border.
    """
    hint = colour_hint(rgb)
    fill_material = str(material).strip() or str(name).strip() or "ground"
    border_material = str(edge).strip() or default_edge(fill_material)
    theme_text = str(theme).strip()
    fill = ", ".join(part for part in (fill_material, f"{hint} palette", theme_text) if part)
    border = ", ".join(part for part in (border_material, f"{hint} palette", theme_text) if part)
    return (fill, border)


def texture_seed(base_seed: int, index: int, role: str) -> int:
    """One terrain-and-role's seed, derived from the job's one stored seed.

    Derived rather than stored, for the reason ``_sprite_synthesis`` stores two:
    a set is up to sixteen generations and sixteen seed fields in ``params``
    would be sixteen things a rerun has to keep consistent. One seed reproduces
    the whole sheet, and two adjacent terrains still get unrelated textures --
    which matters, since a fill and a border one apart in seed space come back
    looking like the same picture twice.
    """
    if role not in ROLES:
        raise ValueError(f"unknown texture role {role!r}")
    # Two large odd multipliers, so neither the terrain step nor the role step
    # is a small offset in seed space. Masked, not modulo'd by a prime: the
    # caller's contract is only "a legal seed", and a mask is exactly that.
    mixed = int(base_seed) + (int(index) + 1) * 2_654_435_761 + ROLES.index(role) * 40_503_197
    return mixed & MAX_SEED


def reduce_texture(pixels: Any, out_w: int, out_h: int) -> Any:
    """A generated texture at exactly the tile size, still seamless.

    Every output pixel is the integer mean of one *block* of input pixels, and
    the blocks partition the source exactly -- so the reduction commutes with
    tiling the source, which is what preserves the torus. Blocks are allowed to
    differ in size by one when the target does not divide the source; that is
    the price of not restricting tile sizes to divisors of 1024, and it is
    invisible because the neighbouring blocks are still adjacent.

    Integer accumulation in ``int64`` with round-half-up, so the answer does not
    depend on a float rounding mode -- ``groundtex`` promises byte determinism
    and this is the last place a float could get into the atlas.
    """
    import numpy as np

    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError("a texture is (h, w, channels)")
    height, width = int(array.shape[0]), int(array.shape[1])
    out_w, out_h = int(out_w), int(out_h)
    if out_w < 1 or out_h < 1:
        raise ValueError("a reduced texture is at least one pixel across")
    if out_w > width or out_h > height:
        raise ValueError(
            f"this texture is {width}x{height} and cannot be reduced to "
            f"{out_w}x{out_h}; generate it larger"
        )

    def starts(total: int, groups: int) -> Any:
        # Input index ``i`` belongs to output group ``i * groups // total``, so a
        # group begins at ``ceil(g * total / groups)``. Ceil by negation, which
        # keeps it integer.
        return np.array([-(-g * total // groups) for g in range(groups)], dtype=np.int64)

    rows = starts(height, out_h)
    columns = starts(width, out_w)
    row_counts = np.diff(np.append(rows, height))
    column_counts = np.diff(np.append(columns, width))

    summed = np.add.reduceat(array.astype(np.int64), rows, axis=0)
    summed = np.add.reduceat(summed, columns, axis=1)
    counts = (row_counts[:, None] * column_counts[None, :])[:, :, None]
    return ((summed + counts // 2) // counts).astype(np.uint8)


def ground_sidecar(
    *,
    theme: str,
    tile_w: int,
    tile_h: int,
    projection: str,
    border: int,
    colors: int,
    terrains: Sequence[Mapping[str, Any]],
    palette: Sequence[str],
    recipe: Mapping[str, Any],
    created: float,
) -> dict[str, Any]:
    """The set's own record: what was asked for, and what actually ran.

    Written last, after every PNG, because it is this job's completion marker --
    the rule ``_pixel_sheet`` follows about its own sidecar. The recipe records
    what ran rather than what was requested, so a style the base could not take
    is simply absent instead of being claimed.
    """
    return {
        "version": GROUND_VERSION,
        "created": float(created),
        "theme": str(theme),
        "image": "input.png",
        "tile_w": int(tile_w),
        "tile_h": int(tile_h),
        "projection": str(projection),
        "border": int(border),
        "colors": int(colors),
        "terrains": [dict(entry) for entry in terrains],
        "palette": list(palette),
        "recipe": dict(recipe),
    }
