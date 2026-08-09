"""One texture, stretched to any size, drawn as nine quads (UX.md Phase 5).

Two of the four GPU-tier items are the same mechanism over different pixels: a
blurred rounded rect under a surface (:mod:`.shadows`) and a superellipse fill
behind one (:mod:`.surfaces`). Both are square, symmetric, and constant along
their own middle row and column, which is exactly the shape a nine-slice is
*correct* for rather than merely convenient -- the strip that gets stretched is
the one the true shape does not vary along.

So the slicing lives here once. What each caller supplies is the pixels and
what it means by "too small to slice"; what it gets back is a texture, its
corner size, and a draw that costs eight or nine quads.

Everything in here degrades rather than raises. A missing renderer is not a
failure (a headless test and the smoke suite both reach it and both want the
pre-Phase-5 drawing); a texture that will not build is a failure, and it is
recorded per module so one broken sprite does not disable the other item.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# The largest sprite that will ever be built, a side in physical pixels. A
# guard rather than a limit anything reaches: the biggest real request is a
# 10 px radius at overlay elevation on a 2x display, which is 92.
MAX_SIZE = 512


@dataclass(frozen=True)
class Sprite:
    """A built nine-patch: the texture, its imgui handle, and its corner size."""

    texture: Any
    # ``ImTextureRef``, built once with the texture. Nine quads per draw and
    # several draws a frame is enough that minting one per quad would be a
    # per-frame allocation, which the frame loop's own rule forbids -- and it
    # cannot go stale while the texture is alive, since both die together.
    ref: Any
    corner: int

    @property
    def size(self) -> int:
        return 2 * self.corner + 1

    def splits(self) -> tuple[float, float]:
        """``(u at the end of the first column, u at the start of the last)``.

        One pair for both axes: every sprite here is square and symmetric by
        construction, which is why they are generated centred rather than as
        one corner to be flipped four ways.
        """
        size = float(self.size)
        return self.corner / size, (self.corner + 1.0) / size


def build(alpha: Any, corner: int) -> Sprite | None:
    """Upload a square ``uint8`` alpha image as a nine-patch. ``None`` if not now.

    Raises nothing: a caller that cares whether this was a *failure* or merely
    "no renderer yet" asks :func:`have_renderer`, because the first is worth
    giving up on permanently and the second is worth trying again next frame.
    """
    renderer = _renderer()
    if renderer is None or 2 * corner + 1 > MAX_SIZE:
        return None
    import moderngl
    from imgui_bundle import imgui

    size = int(alpha.shape[0])
    # One channel: the tint carries the colour, so three quarters of an RGBA
    # sprite would be a constant white nothing samples.
    texture = renderer.ctx.texture((size, size), 1, alpha.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    # Clamped, or the outermost row wraps to the opposite edge and a surface
    # gets a bright hairline exactly where the sprite should have died.
    texture.repeat_x = texture.repeat_y = False
    # A one-channel texture samples as (r, 0, 0, 1) otherwise, and what is
    # stored here is *alpha*, so every channel has to carry it.
    texture.swizzle = "RRRR"
    renderer.register_texture(texture)
    return Sprite(texture=texture, ref=imgui.ImTextureRef(texture.glo), corner=corner)


def have_renderer() -> bool:
    return _renderer() is not None


def paint(
    draw: Any,
    sprite: Sprite,
    low: tuple[float, float],
    high: tuple[float, float],
    colour: int,
    *,
    centre: bool = True,
) -> bool:
    """Draw ``sprite`` stretched over the rectangle. -> whether it drew.

    ``centre=False`` leaves the middle quad out, which is what lets a shadow be
    drawn *under* a card and *over* an already-painted window background with
    one recipe: leave the interior alone and the surface above cannot be
    darkened by its own shadow.

    Refused, rather than fudged, when the rectangle is smaller than its own
    corners. The alternatives are to shrink the corner blocks, which distorts
    the shape the sprite exists to be exact about, or to draw them overlapping,
    which doubles the alpha in precisely the four places a reader is looking.
    """
    corner = float(sprite.corner)
    if (high[0] - low[0]) < 2.0 * corner + 1.0 or (high[1] - low[1]) < 2.0 * corner + 1.0:
        return False
    lo_uv, hi_uv = sprite.splits()
    xs = (low[0], low[0] + corner, high[0] - corner, high[0])
    ys = (low[1], low[1] + corner, high[1] - corner, high[1])
    us = (0.0, lo_uv, hi_uv, 1.0)
    for row in range(3):
        for col in range(3):
            if row == 1 and col == 1 and not centre:
                continue
            draw.add_image(
                sprite.ref,
                (xs[col], ys[row]),
                (xs[col + 1], ys[row + 1]),
                (us[col], us[row]),
                (us[col + 1], us[row + 1]),
                colour,
            )
    return True


def release(sprites: Any) -> None:
    """Drop every sprite in an iterable. Frame thread only, at teardown."""
    renderer = _renderer()
    for sprite in sprites:
        try:
            if renderer is not None:
                renderer.forget_texture(sprite.texture)
            sprite.texture.release()
        except Exception:  # noqa: BLE001 - teardown continues past one bad handle
            log.debug("could not release a nine-patch sprite", exc_info=True)


def _renderer() -> Any:
    from . import imgui_backend

    return imgui_backend.current()
