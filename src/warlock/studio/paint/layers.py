"""The layer model: canvas-sized RGBA planes and the stack that orders them.

Two decisions that everything else leans on. A layer is *canvas-sized* -- no
per-layer offsets in memory, so every op is a plain slice and no code has to
reason about two coordinate spaces (ORA's offsets are applied at load and
written back out as zero). At 2048x2048 that is 16 MiB a layer; ten layers is
160 MiB, which is the price of never having a bug about where a layer starts.

And a layer has a stable ``uid`` that has nothing to do with its position.
Undo records patches against the uid, so reordering a stack between an edit and
its undo cannot make the patch land on a different layer.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import composite as cp

_uids = itertools.count(1)


@dataclass
class Layer:
    pixels: np.ndarray
    name: str = "Layer"
    opacity: float = 1.0
    visible: bool = True
    blend: str = "normal"
    uid: int = field(default_factory=lambda: next(_uids))

    def __post_init__(self) -> None:
        if self.pixels.dtype != np.uint8 or self.pixels.ndim != 3 or self.pixels.shape[2] != 4:
            raise ValueError("a layer holds (H, W, 4) uint8")
        if self.blend not in cp.BLEND_MODES:
            raise ValueError(f"unknown blend mode {self.blend!r}")

    @property
    def size(self) -> tuple[int, int]:
        return (self.pixels.shape[1], self.pixels.shape[0])

    @classmethod
    def empty(cls, width: int, height: int, name: str = "Layer") -> Layer:
        return cls(pixels=cp.empty(width, height), name=name)

    def copy(self, *, name: str | None = None) -> Layer:
        """A duplicate with its *own* uid -- a copy is a different layer, and
        sharing the uid would let one layer's undo patch rewrite the other."""
        return Layer(
            pixels=self.pixels.copy(),
            name=self.name if name is None else name,
            opacity=self.opacity,
            visible=self.visible,
            blend=self.blend,
        )


class LayerStack:
    """Bottom-first list plus an active index. Never empty."""

    def __init__(self, layers: list[Layer], active: int = 0) -> None:
        if not layers:
            raise ValueError("a document has at least one layer")
        self.layers = layers
        self.active_index = max(0, min(int(active), len(layers) - 1))

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    @property
    def size(self) -> tuple[int, int]:
        return self.layers[0].size

    @property
    def active(self) -> Layer:
        return self.layers[self.active_index]

    def index_of(self, uid: int) -> int:
        for i, layer in enumerate(self.layers):
            if layer.uid == uid:
                return i
        raise KeyError(uid)

    def by_uid(self, uid: int) -> Layer:
        return self.layers[self.index_of(uid)]

    # -- structure ---------------------------------------------------------

    def insert(self, index: int, layer: Layer) -> int:
        if layer.size != self.size:
            raise ValueError("a layer is canvas-sized")
        index = max(0, min(int(index), len(self.layers)))
        self.layers.insert(index, layer)
        self.active_index = index
        return index

    def add(self, layer: Layer | None = None, *, above: int | None = None) -> Layer:
        width, height = self.size
        layer = layer if layer is not None else Layer.empty(width, height)
        at = self.active_index + 1 if above is None else above
        self.insert(at, layer)
        return layer

    def remove(self, index: int) -> Layer:
        if len(self.layers) == 1:
            raise ValueError("the last layer cannot be removed")
        gone = self.layers.pop(index)
        self.active_index = min(self.active_index, len(self.layers) - 1)
        return gone

    def move(self, index: int, to: int) -> int:
        to = max(0, min(int(to), len(self.layers) - 1))
        layer = self.layers.pop(index)
        self.layers.insert(to, layer)
        self.active_index = to
        return to

    def duplicate(self, index: int) -> Layer:
        source = self.layers[index]
        copy = source.copy(name=f"{source.name} copy")
        self.insert(index + 1, copy)
        return copy

    # -- compositing -------------------------------------------------------

    def _entries(self, lo: int, hi: int) -> list[tuple[np.ndarray, float, str]]:
        return [
            (layer.pixels, layer.opacity, layer.blend)
            for layer in self.layers[lo:hi]
            if layer.visible and layer.opacity > 0.0
        ]

    def composite_region(
        self, rect: tuple[int, int, int, int], *, below: np.ndarray | None = None
    ) -> np.ndarray:
        """Composite ``rect`` of the whole stack, float32 straight.

        ``below`` is the cached composite of everything under the active layer,
        full-canvas; when it is given only the active layer and what is above
        it are recomputed. There is no matching cache for the layers *above*,
        and there cannot be: blend modes are not associative, so the layers
        over the active one have to be re-applied in order every time.
        """
        x0, y0, x1, y1 = rect
        if below is None:
            return cp.stack_region(self._entries(0, len(self.layers)), rect)
        base = below[y0:y1, x0:x1]
        return cp.stack_region(self._entries(self.active_index, len(self.layers)), rect, base)

    def composite_below(self) -> np.ndarray:
        """Full-canvas composite of the layers under the active one."""
        width, height = self.size
        return cp.stack_region(self._entries(0, self.active_index), (0, 0, width, height))

    def flatten(self) -> np.ndarray:
        width, height = self.size
        return cp.to_uint8(self.composite_region((0, 0, width, height)))
