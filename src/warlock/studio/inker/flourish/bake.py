"""A recipe, rendered out: every frame of every phase, in every direction.

``bake`` is the one place the engine's float planes become the uint8 cels an
Inker document holds, and the one place the pixel-art pass runs. Two rules,
both borrowed from the character-sheet pipeline and recorded there first:

* **The palette is resolved once, for the whole effect.** A palette per frame
  is a flicker; a palette per phase is a colour shift at the impact. So the
  pixel pass looks at *every* composited frame before it picks a colour set
  (``pipelines.pixelsheet.resolve_palette`` over a contact sheet of them), or
  takes the recipe's own, and maps every frame through that one set.
* **Quantise last.** The pixel pass is not idempotent (``sheetmerge`` records
  the trap), so it runs on the finished composite -- after the direction
  rotation, after every layer is blended -- and never on a layer. In pixel
  mode the bake therefore carries **one cel per frame** and no per-layer
  planes: a stack of individually quantised layers would not composite to the
  quantised composite, and the document would then export something the
  user had never seen. Painterly mode keeps the layers, because there the
  composite is exactly what the stack composites to.

Directions are the simulation turned, not the pixels: ``render`` rotates
every vector a primitive emits, so a spark stream that fires right fires down
at 90 degrees with the same spread, the same gravity, the same noise.

This is the module's one outward import: ``pipelines.pixelize`` and
``pipelines.pixelsheet`` are the authority on what pixel art means here
(``sheetout``'s argument about ``pipelines.sheet``, applied to the pixel
pass), and re-deriving an Oklab palette map would be a second one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import render as R
from .recipe import Recipe

RGB = tuple[int, int, int]

#: Compass names for the common direction counts, clockwise from "right" in
#: screen space (+y down). Anything else is named by its angle.
DIRECTION_NAMES = {
    1: ("E",),
    4: ("E", "S", "W", "N"),
    8: ("E", "SE", "S", "SW", "W", "NW", "N", "NE"),
}


def direction_names(count: int) -> tuple[str, ...]:
    names = DIRECTION_NAMES.get(count)
    if names is not None:
        return names
    return tuple(f"{round(i * 360.0 / count)}deg" for i in range(count))


def direction_angles(count: int) -> tuple[float, ...]:
    return tuple(i * 360.0 / count for i in range(count))


@dataclass
class Facing:
    """One direction's frames."""

    name: str
    degrees: float
    #: phase name -> list of straight-alpha uint8 ``(h, w, 4)`` composites.
    composites: dict[str, list[np.ndarray]] = field(default_factory=dict)
    #: phase name -> layer uid -> list of uint8 planes. Empty in pixel mode.
    layers: dict[str, dict[int, list[np.ndarray]]] = field(default_factory=dict)


@dataclass
class Bake:
    recipe: Recipe
    facings: list[Facing]
    #: ``(r, g, b)`` entries in pixel mode, ``None`` in painterly.
    palette: tuple[RGB, ...] | None
    palette_source: str  # "designed", "derived" or "none"

    @property
    def origin(self) -> tuple[int, int]:
        return (self.recipe.width // 2, self.recipe.height // 2)

    @property
    def fps(self) -> int:
        return self.recipe.fps

    @property
    def pixel(self) -> bool:
        return self.palette is not None

    def tags(self) -> list[tuple[str, int, int, bool]]:
        """``(name, first, last, loop)`` over the flat frame order, which is
        facing-major then phase-major: every phase of E, then every phase of
        SE... One tag per phase per facing, named ``phase`` alone when there is
        one facing and ``phase/E`` otherwise."""
        out = []
        cursor = 0
        many = len(self.facings) > 1
        for facing in self.facings:
            for phase in self.recipe.phases:
                n = phase.frames
                name = f"{phase.name}/{facing.name}" if many else phase.name
                out.append((name, cursor, cursor + n - 1, phase.loop))
                cursor += n
        return out

    def flat(self) -> list[np.ndarray]:
        """Every composite in tag order."""
        return [
            frame
            for facing in self.facings
            for phase in self.recipe.phases
            for frame in facing.composites[phase.name]
        ]

    @property
    def frame_count(self) -> int:
        return self.recipe.frame_count * len(self.facings)


def _straight(plane: np.ndarray) -> np.ndarray:
    """Premultiplied float -> straight uint8 at the *raster* size."""
    alpha = plane[..., 3]
    safe = np.maximum(alpha, 1e-6)[..., None]
    rgb = np.where(alpha[..., None] > 1e-6, plane[..., :3] / safe, 0.0)
    out = np.empty(plane.shape[:2] + (4,), dtype=np.uint8)
    out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    out[..., :3] *= (out[..., 3:4] > 0).astype(np.uint8)
    return out


def bake(
    recipe: Recipe,
    *,
    directions: int | None = None,
    palette: tuple[RGB, ...] | None = None,
    dither: bool = False,
    progress: Any = None,
    assets: dict[str, np.ndarray] | None = None,
) -> Bake:
    """Render every frame. ``directions`` overrides the recipe's count;
    ``palette`` overrides its colours (pixel mode only). ``progress`` is an
    optional ``(done, total) -> None`` callback; ``assets`` the textures the
    recipe's sprite layers name."""
    count = int(directions or recipe.directions)
    names = direction_names(count)
    angles = direction_angles(count)
    pixel = recipe.mode == "pixel"
    s = int(recipe.supersample)
    total = recipe.frame_count * count
    done = 0

    facings: list[Facing] = []
    raw_composites: list[tuple[Facing, str, np.ndarray]] = []
    for name, degrees in zip(names, angles, strict=True):
        facing = Facing(name=name, degrees=degrees)
        for phase in recipe.phases:
            facing.composites[phase.name] = []
            if not pixel:
                facing.layers[phase.name] = {}
            start = recipe.phase_start(recipe.phases.index(phase))
            for i in range(phase.frames):
                frame = start + i
                planes = R.render(recipe, frame, degrees, assets)
                comp = R.composite(recipe, planes, phase.name)
                if pixel:
                    raw_composites.append((facing, phase.name, comp))
                else:
                    facing.composites[phase.name].append(R.to_uint8(comp, s))
                    for layer in recipe.layers:
                        if not layer.active_in(phase.name):
                            continue
                        # A layer that painted nothing this frame still owns a
                        # cel, so every track has one per frame of its phases.
                        plane = planes.get(layer.uid)
                        cel = (
                            R.to_uint8(plane, s)
                            if plane is not None
                            else np.zeros((recipe.height, recipe.width, 4), dtype=np.uint8)
                        )
                        facing.layers[phase.name].setdefault(layer.uid, []).append(cel)
                done += 1
                if progress is not None:
                    progress(done, total)
        facings.append(facing)

    if not pixel:
        return Bake(recipe=recipe, facings=facings, palette=None, palette_source="none")

    entries, source = _resolve(recipe, raw_composites, palette)
    for facing, phase_name, comp in raw_composites:
        facing.composites[phase_name].append(_pixelize(comp, recipe, entries, dither))
    return Bake(recipe=recipe, facings=facings, palette=entries, palette_source=source)


def _resolve(
    recipe: Recipe,
    composites: list[tuple[Facing, str, np.ndarray]],
    override: tuple[RGB, ...] | None,
) -> tuple[tuple[RGB, ...], str]:
    from PIL import Image

    from warlock.pipelines import pixel, pixelsheet

    if override:
        return tuple(override), "designed"
    if recipe.palette:
        return pixel.parse_hex("\n".join(recipe.palette)), "designed"
    # A contact sheet of every frame at logical size: the palette has to see
    # the cast's glow and the dissipate's ash, not one or the other.
    s = int(recipe.supersample)
    tiles = [R.to_uint8(comp, s) for _, _, comp in composites]
    if not tiles:
        return ((0, 0, 0),), "derived"
    cols = max(1, int(np.ceil(np.sqrt(len(tiles)))))
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * recipe.height, cols * recipe.width, 4), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        rows_ = slice(r * recipe.height, (r + 1) * recipe.height)
        cols_ = slice(c * recipe.width, (c + 1) * recipe.width)
        sheet[rows_, cols_] = tile
    entries, source = pixelsheet.resolve_palette(
        Image.fromarray(sheet, "RGBA"), colors=recipe.colors
    )
    return tuple(entries), source


def _pixelize(
    comp: np.ndarray, recipe: Recipe, entries: tuple[RGB, ...], dither: bool
) -> np.ndarray:
    from PIL import Image

    from warlock.pipelines import pixelize

    big = Image.fromarray(_straight(comp), "RGBA")
    small, _report = pixelize.pixelize(
        big,
        size=(recipe.width, recipe.height),
        palette=entries,
        dither=dither,
        reduce_mode="box",
        outline_mode="none",
        clean=True,
    )
    return np.asarray(small.convert("RGBA"), dtype=np.uint8).copy()
