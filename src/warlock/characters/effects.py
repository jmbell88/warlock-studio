"""A theme's declared effect, rendered onto a character sheet's cells.

:class:`~.family.Theme` has carried ``effects`` and ``effect_params`` since the
registry landed and nothing read them; this module is the reader. A fire ogre's
theme says ``effects=("embers",)``, the archetype says where a socket called
``crown`` sits, ``blender_worker.op_sheet`` says where that socket landed in
every one of the 256 rendered cells, and Flourish already knows how to draw a
flame. What was missing was the four lines of arithmetic between them.

**The composite happens between the reduction and the pack**, which is
``_q_troupe``'s ordering to defend and not this module's; what this module owes
that ordering is that it works at the *logical* cell size, so the flame it
returns is a drop-in replacement for the reduced frame it was handed.

**The flame is drawn on its own cell-sized canvas with its base at the canvas
centre**, and the compositor then shifts that canvas so the centre lands on the
projected socket. The alternative -- placing the flame with the layer's own
``x``/``y`` -- would work equally well and be harder to test: an offset is
integer arithmetic a unit test can state, where a layer parameter is only
checkable by rendering. The shift is rounded to whole pixels: at 32px a
sub-pixel placement is invisible, and integer offsets are what make two runs of
the same sheet byte-identical.

**A four-frame idle flame does not loop seamlessly, and that is known and not
fixed.** The flame primitive erodes its silhouette with fbm scrolled along the
rise, and fbm is not periodic -- so frame 3 of a four-frame idle does not hand
back to frame 0. At the sizes a character sheet is quantised to (16-64px, 32
colours) the discontinuity is a flicker in the tongues rather than a pop, which
is why it ships. Making it loop means either a periodic noise field (a Flourish
change, and one that would alter every existing recipe's bytes) or rendering a
longer cycle and picking frames from it. ``docs/manual/33-troupe.md`` says the
same thing to the reader.

**Flourish is imported at function scope, always.** ``characters`` is a
registry the door imports to answer "what can we make", and Flourish drags
numpy and ten primitive modules in behind it. The pin in
``tests/characters/test_characters_imports.py`` is what keeps it that way.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .family import Socket, Theme

log = logging.getLogger(__name__)

__all__ = [
    "EFFECT_KINDS",
    "composite_effects",
    "effect_colors",
    "effect_height_px",
    "effect_kind",
    "effect_seed",
    "flame_recipe",
    "pick_socket",
]

#: Every effect kind this module can draw. The registry declares exactly one
#: today (``embers``, on the three ``fire`` themes), so this is a table of one
#: -- written as a table anyway because the second kind should be a function
#: and an entry here rather than an unpicking of ``composite_effects``. A theme
#: naming a kind that is not here draws nothing and says so once in the log,
#: which is the same "an unknown thing costs the effect, never the sheet" rule
#: ``blender_worker._socket_world_point`` follows for an unknown bone.
EFFECT_KINDS: tuple[str, ...] = ("embers",)

#: kind -> the socket names it prefers, best first. An effect has to hang off
#: *something* and the theme does not say which: ``Theme.effects`` names what a
#: species emits, and ``Archetype.sockets`` names where things can hang. This
#: table is the join, and it is ordered rather than exhaustive because the four
#: archetypes do not share a socket name -- an amorphous elemental has ``core``
#: and no ``crown`` is worth burning, a dragon has a ``crown`` and no ``core``.
#: The first name the archetype actually declares wins, and nothing draws twice:
#: a character on fire at five sockets is a bonfire, not a character.
_SOCKETS_FOR: dict[str, tuple[str, ...]] = {
    "embers": ("core", "crown", "back", "saddle", "weapon_main"),
}

#: How wide a flame is at its base, as a fraction of its height. A constant
#: rather than a parameter because it is a drawing decision and not a species
#: one: a flame twice as wide as tall reads as a puddle at 32px.
_FLAME_ASPECT = 0.6

#: ``effect_params["rate"]`` is embers per second, which the flame primitive has
#: no notion of -- it scrolls a noise field. Sixteen is the divisor that puts
#: the registry's shipped 24/s onto 1.5, a shade under the primitive's own 1.6
#: default, so a fire theme looks like the preset it was tuned against.
_RATE_TO_SCROLL = 16.0


def effect_kind(theme: Theme) -> str | None:
    """The one effect kind ``theme`` declares that this build can draw.

    First rather than all: two effects on one socket would composite over each
    other and the registry declares no theme with two.
    """
    for kind in theme.effects or ():
        if kind in EFFECT_KINDS:
            return kind
        log.info("theme %r declares effect %r, which this build cannot draw", theme.key, kind)
    return None


def pick_socket(kind: str, sockets: Sequence[Socket]) -> tuple[int, Socket] | None:
    """``(index within ``sockets``, the socket)`` an effect of ``kind`` hangs on.

    The index is returned because :func:`effect_seed` salts with it -- so two
    sockets of one character get two different flames, and the same socket of
    the same character gets the same flame on every run.
    """
    by_name = {socket.name: (i, socket) for i, socket in enumerate(sockets)}
    for name in _SOCKETS_FOR.get(kind, ()):
        found = by_name.get(name)
        if found is not None:
            return found
    return None


def effect_seed(recipe_seed: int, socket_index: int) -> int:
    """The seed one socket's effect renders from.

    ``7919`` is ``render.FrameCtx.lseed``'s multiplier, reused deliberately:
    the two are the same trick -- decorrelate one integer from its neighbours by
    a prime stride -- and a second constant would be a second thing to get
    wrong. Masked to 31 bits because ``recipe.clamp`` masks the seed it stores,
    so a value that did not fit would silently become a different flame.
    """
    return (int(recipe_seed) * 7919 + int(socket_index)) & 0x7FFFFFFF


def effect_colors(theme: Theme) -> tuple[str, str]:
    """``(base, tip)`` hex colours for ``theme``'s flame.

    **Not from ``effect_params``**, and that is a correction to the plan rather
    than an omission: ``Theme.effect_params`` is typed ``Mapping[str, float]``
    and every shipped row holds only floats, so a colour cannot live there
    without changing a registry file this increment does not own. The theme's
    *materials* are the honest source -- a fire theme's ``accent`` is the
    orange it was authored with, so the flame comes out the same orange as the
    character's own trim rather than a second opinion about what fire looks
    like. A string in ``effect_params`` still wins if a later row puts one
    there, so widening the registry's type is all that change would take.
    """
    params = theme.effect_params or {}
    materials = theme.materials or {}

    def _pick(param: str, regions: tuple[str, ...], fallback: str) -> str:
        raw = params.get(param)
        if isinstance(raw, str) and raw:
            return raw
        for region in regions:
            found = materials.get(region)
            if isinstance(found, str) and found:
                return found
        return fallback

    # The hot heart first (a core, an eye), then the trim. The fallbacks are
    # ``prims.flame.PARAMS``' own defaults, so a theme with neither region still
    # renders the primitive's stock flame instead of nothing.
    base = _pick("color_base", ("core", "eye", "accent"), "#FFE08A")
    tip = _pick("color_tip", ("accent", "core"), "#E0341C")
    return base, tip


def effect_height_px(socket: Socket, theme: Theme, *, logical: int) -> int:
    """How tall the flame at ``socket`` is, in cell pixels.

    ``Socket.reach`` is the radius in bone-length units that a prop may occupy
    before it clips the body, which is exactly the budget an effect should
    spend: a flame the size of the thing it is allowed to hang there is a flame
    that does not swallow the character. ``effect_params["rise"]`` scales it --
    0.35 on every shipped fire theme -- so "how big" stays a species decision
    and "how much room is there" stays the archetype's.
    """
    rise = float((theme.effect_params or {}).get("rise", 0.35) or 0.35)
    return max(2, int(round(float(socket.reach) * float(logical) * rise)))


def flame_recipe(
    theme: Theme,
    socket: Socket,
    *,
    seed: int,
    cell_px: int,
    height_px: int,
    frames: int = 12,
    fps: int = 12,
) -> Any:
    """One Flourish flame layer on a cell-sized canvas, as a clamped ``Recipe``.

    ``frames`` is the *movement's* frame count and ``fps`` its playback rate --
    both keywords rather than derived, because the flame has to animate over the
    same cycle the body does: a four-frame idle and an eight-frame walk want
    four and eight distinct flames, and the scroll distance between two of them
    is ``speed * frame / fps``.

    ``rise`` is pinned at -90 -- straight up the canvas -- **regardless of
    facing**. It is tempting to turn it with the cell's yaw, and it would be
    wrong: fire rises in world space, and a sheet whose flames lean east when
    the character faces east is a sheet where gravity turns with the camera.

    ``mode="painterly"`` and ``supersample=4`` because the sheet's own
    quantisation runs afterwards and is the one that decides the palette; a
    flame that had already cut itself to 16 colours would be quantised twice.
    """
    from ..studio.inker.flourish import recipe as flourish_recipe

    base, tip = effect_colors(theme)
    params = theme.effect_params or {}
    cell = max(8, int(cell_px))
    # Clamped to half the canvas because the base sits at the canvas centre and
    # the flame rises from it: a taller one would be cut off by the plane it is
    # drawn on rather than by the cell it is composited into, which is the same
    # pixels lost for a much less obvious reason.
    height = max(2, min(int(height_px), cell // 2))
    width = max(2, int(round(height * _FLAME_ASPECT)))
    scroll = min(10.0, max(0.0, float(params.get("rate", 24.0) or 24.0) / _RATE_TO_SCROLL))
    layer = flourish_recipe.Layer(
        # A literal uid, not ``new_uid()``: a recipe built here is rendered and
        # thrown away, never merged into an Inker stack, and a counter read
        # would make the recipe depend on how many recipes had been built
        # before it -- which is exactly the "same bytes every run" property
        # this whole path rests on.
        uid=1,
        kind="flame",
        name=f"{theme.key} {socket.name}",
        params={
            # The base at the canvas centre; the compositor does the moving.
            "x": 0.0,
            "y": 0.0,
            "width": float(width),
            "height": float(height),
            "rise": -90.0,
            "color_base": base,
            "color_tip": tip,
            "speed": scroll,
            # Cells sized to the flame rather than to the canvas: the
            # primitive's 12px default is a third of a 32px sheet's whole
            # frame, which would give one tongue and no shape.
            "scale": max(2.0, float(height) / 3.0),
        },
    )
    return flourish_recipe.clamp(
        flourish_recipe.Recipe(
            name=f"{theme.key}:{socket.name}",
            seed=int(seed),
            width=cell,
            height=cell,
            supersample=4,
            fps=max(1, int(fps)),
            mode="painterly",
            phases=(flourish_recipe.Phase("burn", max(1, int(frames)), True),),
            layers=(layer,),
        )
    )


#: kind -> the function that builds its recipe. One entry, on purpose: adding a
#: second effect is writing a ``smoke_recipe`` beside ``flame_recipe`` and
#: naming it here, not touching :func:`composite_effects`.
_RECIPE_FOR = {"embers": flame_recipe}


def composite_effects(
    reduced: Mapping[int, Any],
    cells_by_index: Mapping[int, Mapping[str, Any]],
    sockets_px: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *,
    theme: Theme,
    sockets: Sequence[Socket],
    recipe_seed: int,
    logical: int,
    out_dir: Any,
) -> dict[int, Path]:
    """Draw ``theme``'s effect onto the cells that project its socket.

    ``reduced`` is ``pixelize.reduce_frames``' ``{cell index: path}``;
    ``cells_by_index`` is ``{index: {"frame", "frames", "fps"}}`` -- the cell's
    place in its movement, that movement's length, and its playback rate;
    ``sockets_px`` is the worker's projection **already converted to cell
    pixels** by ``charsheet.point_in_cell``. Returns ``{index: path}`` for the
    cells that changed, same size, written into ``out_dir`` -- so the caller
    merges it over ``reduced`` and packs.

    ``sockets`` is the archetype's socket tuple, and it is here because
    ``Socket.reach`` is the effect's size budget and the *projection* does not
    carry it: the worker emits ``x``/``y``/``depth``/``behind`` and nothing
    else.

    **A cell whose socket the worker did not project is left exactly alone.**
    An unknown bone costs the effect and never the sheet -- and on a subset
    re-render the projection only covers the cells that were rendered, which is
    what makes "composite exactly its own cells" fall out rather than need a
    filter.
    """
    import numpy as np
    from PIL import Image

    from ..studio.inker.flourish import render as flourish_render

    kind = effect_kind(theme)
    if kind is None:
        return {}
    found = pick_socket(kind, sockets)
    if found is None:
        log.info(
            "theme %r emits %r but this archetype declares no socket to hang it on",
            theme.key, kind,
        )
        return {}
    socket_index, socket = found
    build = _RECIPE_FOR[kind]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cell_px = max(1, int(logical))
    seed = effect_seed(recipe_seed, socket_index)
    height_px = effect_height_px(socket, theme, logical=cell_px)

    # One recipe per ``(frames, fps)`` pair rather than per cell: every cell of
    # a movement renders from the same recipe at a different frame, which is
    # what makes the flame animate *with* the clip instead of restarting in
    # every cell.
    recipes: dict[tuple[int, int], Any] = {}
    written: dict[int, Path] = {}
    for index, path in sorted(reduced.items()):
        here = (sockets_px.get(index) or {}).get(socket.name)
        if not here:
            continue
        plan = cells_by_index.get(index) or {}
        frames = max(1, int(plan.get("frames", 1) or 1))
        fps = max(1, int(plan.get("fps", 12) or 12))
        frame = max(0, min(int(plan.get("frame", 0) or 0), frames - 1))
        key = (frames, fps)
        recipe = recipes.get(key)
        if recipe is None:
            recipe = build(
                theme, socket, seed=seed, cell_px=cell_px,
                height_px=height_px, frames=frames, fps=fps,
            )
            recipes[key] = recipe
        plane = flourish_render.render_frame(recipe, frame)
        flame = flourish_render.to_uint8(plane, recipe.supersample)

        with Image.open(path) as opened:
            opened.load()
            body = np.asarray(opened.convert("RGBA")).astype(np.float32) / 255.0
        h, w = body.shape[:2]
        # The flame canvas' centre is the flame's base; move it onto the socket.
        shifted = _shift_into(flame.astype(np.float32) / 255.0, (h, w),
                              dx=int(round(float(here["x"]) - cell_px / 2.0)),
                              dy=int(round(float(here["y"]) - cell_px / 2.0)))
        # Under when the socket is on the far side of the body's centre, over
        # otherwise. That single bit is the whole reason the worker measures
        # view depth: a back-mounted flame drawn over the body is a character
        # standing in front of their own fire.
        merged = (
            _over(shifted, body) if bool(here.get("behind")) else _over(body, shifted)
        )
        target = out / Path(path).name
        Image.fromarray(
            np.clip(np.rint(merged * 255.0), 0, 255).astype(np.uint8), "RGBA"
        ).save(target, format="PNG")
        written[index] = target
    return written


def _shift_into(rgba: Any, shape: tuple[int, int], *, dx: int, dy: int) -> Any:
    """``rgba`` translated by ``(dx, dy)`` onto a transparent canvas of ``shape``.

    Whole pixels, and clipped rather than wrapped: a flame at the edge of a cell
    is half a flame, never half a flame plus a stripe down the other side.
    """
    import numpy as np

    h, w = shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    sh, sw = rgba.shape[:2]
    dst_y0, dst_x0 = max(0, dy), max(0, dx)
    src_y0, src_x0 = max(0, -dy), max(0, -dx)
    copy_h = min(sh - src_y0, h - dst_y0)
    copy_w = min(sw - src_x0, w - dst_x0)
    if copy_h > 0 and copy_w > 0:
        out[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = rgba[
            src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
        ]
    return out


def _over(under: Any, top: Any) -> Any:
    """``top`` over ``under``, both straight-alpha float in 0..1.

    Straight rather than premultiplied because that is what a PNG on disk is,
    and the two files this composites are exactly that -- ``to_uint8`` has
    already taken Flourish's premultiplied plane back to straight alpha.
    """
    import numpy as np

    ta = top[..., 3:4]
    ua = under[..., 3:4]
    alpha = ta + ua * (1.0 - ta)
    safe = np.maximum(alpha, 1e-6)
    rgb = (top[..., :3] * ta + under[..., :3] * ua * (1.0 - ta)) / safe
    out = np.concatenate([np.where(alpha > 1e-6, rgb, 0.0), alpha], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
