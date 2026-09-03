"""One frame of a recipe: every layer's plane, and the composite.

``render(recipe, frame)`` returns the premultiplied float32 plane of every
layer active in that frame, keyed by uid, at the supersampled size;
``composite`` folds them bottom-first with each layer's blend and opacity;
``to_uint8`` takes the result down to the recipe's logical size as straight
alpha, which is what an Inker cel is. The three are separate so a caller that
only changed one layer can re-render that layer and re-composite, and so a
test can assert on a single primitive's output.

Compositing is premultiplied throughout. ``normal`` is the over operator;
``add`` sums colour and unions coverage, which is what a glow or a spark over
a dark background wants. A layer whose primitive ``REPLACES_BELOW`` takes the
running composite as its input and hands back the new running composite --
the distortion -- and so is never blended at all.

Deterministic by construction: the only inputs are the recipe, the frame number
and the direction, and every primitive is stateless. ``tests/inker/flourish/
test_render.py`` pins the bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import prims
from .recipe import Phase, Recipe


@dataclass
class FrameCtx:
    """Everything a primitive may ask about the frame it is rendering."""

    seed: int
    width: int  # supersampled raster size
    height: int
    scale: float  # raster px per logical px
    frame: int  # global frame number
    phase: Phase
    phase_index: int
    phase_frame: int
    fps: int
    direction: float = 0.0  # degrees; rotates every vector a primitive emits
    #: Set by ``render`` before each primitive runs: the layer's position in
    #: the stack. **Seeds come from this, never from ``Layer.uid``** -- uids
    #: are reissued on every load, and a recipe file must render the same
    #: bytes each time it is opened.
    layer_index: int = 0
    #: Asset id -> straight uint8 RGBA texture. Resolved by the document; the
    #: engine only reads. Empty on a recipe that stamps nothing.
    assets: dict[str, np.ndarray] | None = None

    _coords: tuple[np.ndarray, np.ndarray] | None = None
    _coarse: tuple[np.ndarray, np.ndarray] | None = None
    _index: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def t(self) -> float:
        """0..1 through the phase (1 is the last frame, so a one-frame phase is 0)."""
        n = max(self.phase.frames - 1, 1)
        return min(self.phase_frame / n, 1.0) if self.phase.frames > 1 else 0.0

    @property
    def time(self) -> float:
        """Seconds since the recipe's first frame."""
        return self.frame / self.fps

    @property
    def phase_time(self) -> float:
        """Seconds since the phase's first frame."""
        return self.phase_frame / self.fps

    @property
    def phase_seconds(self) -> float:
        return self.phase.frames / self.fps

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Logical-pixel coordinate planes, origin at the canvas centre, +y down."""
        if self._coords is None:
            s = np.float32(self.scale)
            xs = (np.arange(self.width, dtype=np.float32) + 0.5 - self.width / 2.0) / s
            ys = (np.arange(self.height, dtype=np.float32) + 0.5 - self.height / 2.0) / s
            self._coords = np.meshgrid(xs, ys)
        return self._coords

    def coarse(self) -> tuple[np.ndarray, np.ndarray]:
        """The same planes at logical resolution (one sample per logical px),
        centred on the pixel each ``scale`` x ``scale`` block covers."""
        if self._coarse is None:
            s = int(self.scale)
            w, h = self.width // s, self.height // s
            xs = np.arange(w, dtype=np.float32) + 0.5 - w / 2.0
            ys = np.arange(h, dtype=np.float32) + 0.5 - h / 2.0
            self._coarse = np.meshgrid(xs, ys)
        return self._coarse

    def index(self) -> tuple[np.ndarray, np.ndarray]:
        """``(rows, cols)`` integer index planes of the raster, for gathers."""
        if self._index is None:
            ys, xs = np.mgrid[0 : self.height, 0 : self.width]
            self._index = (ys.astype(np.float32), xs.astype(np.float32))
        return self._index

    def asset(self, asset_id: str) -> np.ndarray | None:
        """The texture behind an id, or None -- nothing is ever substituted."""
        if not asset_id or not self.assets:
            return None
        found = self.assets.get(asset_id)
        if found is None or found.ndim != 3 or found.shape[2] != 4 or found.size == 0:
            return None
        return found

    def lseed(self, salt: int = 0) -> int:
        """A seed private to the layer being rendered, plus ``salt``."""
        return int(self.seed) + 7919 * (self.layer_index + 1) + int(salt)

    def turn(self, x: float, y: float) -> tuple[float, float]:
        """A logical offset from the centre, rotated by the frame's direction."""
        return prims.rotate(x, y, self.direction)


def frame_ctx(
    recipe: Recipe,
    frame: int,
    direction: float = 0.0,
    assets: dict[str, np.ndarray] | None = None,
) -> FrameCtx:
    phase, index, within = recipe.phase_at(frame)
    s = int(recipe.supersample)
    return FrameCtx(
        assets=assets,
        seed=int(recipe.seed),
        width=recipe.width * s,
        height=recipe.height * s,
        scale=float(s),
        frame=frame,
        phase=phase,
        phase_index=index,
        phase_frame=within,
        fps=int(recipe.fps),
        direction=float(direction),
    )


def _blend_of(layer: Any) -> str:
    forced = getattr(prims.module(layer.kind), "FORCE_BLEND", None)
    return forced or layer.blend


def render(
    recipe: Recipe,
    frame: int,
    direction: float = 0.0,
    assets: dict[str, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Every active layer's premultiplied plane for ``frame``, keyed by uid,
    in stack order. A layer that replaces the composite below gets the
    composite of the layers under it as its input, so the planes returned are
    each layer's *own* output and ``composite`` reassembles them."""
    ctx = frame_ctx(recipe, frame, direction, assets)
    out: dict[int, np.ndarray] = {}
    running: np.ndarray | None = None
    for index, layer in enumerate(recipe.layers):
        if not layer.active_in(ctx.phase.name):
            continue
        ctx.layer_index = index
        mod = prims.module(layer.kind)
        plane = mod.render(layer, ctx, running)
        if plane is None:
            continue
        if mod.REPLACES_BELOW:
            running = plane
            out[layer.uid] = plane
            continue
        plane = plane * np.float32(layer.opacity)
        out[layer.uid] = plane
        running = _blend(running, plane, _blend_of(layer))
    return out


def composite(recipe: Recipe, planes: dict[int, np.ndarray], phase_name: str) -> np.ndarray:
    """Fold ``planes`` (from ``render``) bottom-first into one premultiplied plane."""
    running: np.ndarray | None = None
    for layer in recipe.layers:
        plane = planes.get(layer.uid)
        if plane is None or not layer.active_in(phase_name):
            continue
        if prims.module(layer.kind).REPLACES_BELOW:
            running = plane
            continue
        running = _blend(running, plane, _blend_of(layer))
    if running is None:
        s = int(recipe.supersample)
        return np.zeros((recipe.height * s, recipe.width * s, 4), dtype=np.float32)
    return running


def render_frame(
    recipe: Recipe,
    frame: int,
    direction: float = 0.0,
    assets: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """The composited frame, premultiplied float32 at the supersampled size."""
    phase, _, _ = recipe.phase_at(frame)
    return composite(recipe, render(recipe, frame, direction, assets), phase.name)


def _blend(dst: np.ndarray | None, src: np.ndarray, mode: str) -> np.ndarray:
    if dst is None:
        return np.clip(src, 0.0, 1.0).astype(np.float32)
    if mode == "add":
        out = dst.copy()
        out[..., :3] += src[..., :3]
        out[..., 3] = dst[..., 3] + src[..., 3] - dst[..., 3] * src[..., 3]
        return np.clip(out, 0.0, 1.0)
    # over
    inv = (1.0 - src[..., 3])[..., None]
    return np.clip(src + dst * inv, 0.0, 1.0).astype(np.float32)


def to_uint8(plane: np.ndarray, supersample: int) -> np.ndarray:
    """A premultiplied supersampled plane -> straight-alpha ``(h, w, 4)`` uint8
    at logical size, by alpha-weighted box reduction (the ``pixelize.reduce``
    rule: a plain mean drags the transparent background's black into every
    edge)."""
    s = int(supersample)
    h, w = plane.shape[0] // s, plane.shape[1] // s
    boxed = plane[: h * s, : w * s].reshape(h, s, w, s, 4).mean(axis=(1, 3)) if s > 1 else plane
    alpha = boxed[..., 3]
    safe = np.maximum(alpha, 1e-6)[..., None]
    rgb = np.where(alpha[..., None] > 1e-6, boxed[..., :3] / safe, 0.0)
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    # Straight alpha means a clear pixel carries no colour at all.
    out[..., :3] *= (out[..., 3:4] > 0).astype(np.uint8)
    return out
