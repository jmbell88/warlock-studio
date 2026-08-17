"""What a layer actually looks like once its ancestors have had their say.

**One resolver, both renderers.** ``panes/plotter_canvas`` draws one quad per
visible cell and ``render.py`` composites the whole map for an export; they
already agreed about orientation and placement because both take those from
:mod:`.gid` and :mod:`.project`, and this is the third answer neither is
allowed to work out for itself. A group is a *state its descendants inherit*
rather than a surface anything is composited onto, so "what opacity is this
layer drawn at" is a walk from the root, and two walks written twice is how the
canvas and an export come to disagree about a nested layer.

The five combination rules, and each of them is Tiled's:

* **offset** sums -- a pixel nudge on a group moves everything inside it;
* **parallax** multiplies -- factors compose, and the identity is 1;
* **opacity** multiplies -- and note this is *per leaf*, not per flattened
  subtree: two half-opaque siblings inside a half-opaque group show through
  each other, because the group is never composited on its own;
* **visible** is AND -- one hidden ancestor hides everything under it, and the
  leaf's own flag never moves, so unhiding the group restores exactly what was
  there;
* **tint** multiplies per channel, and **locked** is OR.

Locked is reported rather than enforced. The engine's rule has always been that
a lock stops *the user*, not the document -- ``write_region`` must keep working
on a locked layer or an undo could not put back what was there before the lock
was applied -- so this says what the studio should refuse and refuses nothing.

**Nothing here assumes ``(h, w)`` dense storage.** A :class:`Resolved` carries a
layer and six numbers; it never reads ``data``, never asks a layer for its
shape, and never indexes a cell. That is deliberate and load-bearing for
Milestone 5: an infinite (chunked) map has no dense rectangle at all, and the
resolver is the one piece of the drawing path that must not have to change when
one arrives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._map_model import DECORATION_FIELDS, OPAQUE_WHITE, GroupLayer, Layer
from .tileset import RGBA

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc

__all__ = ["DECORATION_FIELDS", "Resolved", "resolve", "resolved_for"]


@dataclass(frozen=True)
class Resolved:
    """One layer, and what its whole ancestry adds up to.

    Frozen for :class:`~._map_model.Rect`'s reason: a renderer holds a list of
    these for the length of a frame and a caller that could write into one would
    be changing what the *next* consumer draws, with nothing in the document to
    say why.
    """

    #: The layer itself, by reference. A leaf out of :func:`resolve`; possibly a
    #: group out of :func:`resolved_for`, which answers about any layer.
    layer: Layer
    offset: tuple[float, float]
    parallax: tuple[float, float]
    opacity: float
    visible: bool
    tint: RGBA
    locked: bool
    blend_mode: str


#: What a layer at the root inherits: nothing, spelled as the identity of every
#: one of the five rules.
IDENTITY: tuple[tuple[float, float], tuple[float, float], float, bool, RGBA, bool, str] = (
    (0.0, 0.0),
    (1.0, 1.0),
    1.0,
    True,
    OPAQUE_WHITE,
    False,
    "normal",
)


def _tint_product(over: RGBA, under: Any) -> RGBA:
    """Two tints multiplied channel by channel, in 0..255.

    Rounded rather than truncated, so a chain of identity tints stays exactly
    identity: ``255 * 255 / 255`` is 255 either way, but ``128 * 255 / 255``
    truncates to 127 and a document with three untinted groups over it would
    drift a channel per level.
    """
    return tuple(  # type: ignore[return-value]
        int(round(a * b / 255.0)) for a, b in zip(over, under, strict=True)
    )


def _combine(
    layer: Any,
    offset: tuple[float, float],
    parallax: tuple[float, float],
    opacity: float,
    visible: bool,
    tint: RGBA,
    locked: bool,
    blend_mode: str,
) -> tuple[tuple[float, float], tuple[float, float], float, bool, RGBA, bool, str]:
    """One layer's own decorations folded into what it inherited."""
    return (
        (offset[0] + float(layer.offset_x), offset[1] + float(layer.offset_y)),
        (parallax[0] * float(layer.parallax_x), parallax[1] * float(layer.parallax_y)),
        opacity * float(layer.opacity),
        visible and bool(layer.visible),
        _tint_product(tint, layer.tint),
        locked or bool(layer.locked),
        # A group's non-normal mode applies to descendants whose own mode is
        # normal. A leaf with an explicit mode wins, matching the way the other
        # inherited decorations let a child refine its parent.
        layer.blend_mode if layer.blend_mode != "normal" else blend_mode,
    )


def resolve(doc: MapDoc, *, include_hidden: bool = False) -> list[Resolved]:
    """Every *drawable* layer, in paint order, with its inherited state.

    Groups are not in the result: they draw nothing, and a consumer that had to
    skip them would be a consumer that could forget to. Bottom-first and
    depth-first, which is the order both renderers already loop in.

    ``include_hidden`` is the export's flag, and it reaches ancestors too -- a
    leaf inside a hidden group is hidden, so the same one switch that decides
    what you see decides what comes out. Without it a hidden group is not even
    descended into, which is the cheap half as well as the correct one.
    """
    out: list[Resolved] = []

    def visit(layers: list[Layer], state: Any) -> None:
        for layer in layers:
            combined = _combine(layer, *state)
            if not combined[3] and not include_hidden:
                continue
            if isinstance(layer, GroupLayer):
                visit(layer.children, combined)
            else:
                out.append(Resolved(layer, *combined))

    visit(doc.layers, IDENTITY)
    return out


def resolved_for(doc: MapDoc, uid: int | None) -> Resolved | None:
    """One named layer's inherited state, group or leaf, or ``None``.

    What hit-testing asks: the canvas has to subtract the *active* layer's
    resolved offset before turning a mouse position into a cell, and the active
    layer may be hidden, may be a group, and may be one the paint loop skipped.
    So this is a lookup rather than a filter over :func:`resolve`, and it
    answers about hidden layers unconditionally -- a hidden layer you have
    selected is still the layer your clicks are about.
    """
    if uid is None:
        return None

    def visit(layers: list[Layer], state: Any) -> Resolved | None:
        for layer in layers:
            combined = _combine(layer, *state)
            if layer.uid == uid:
                return Resolved(layer, *combined)
            if isinstance(layer, GroupLayer):
                found = visit(layer.children, combined)
                if found is not None:
                    return found
        return None

    return visit(doc.layers, IDENTITY)
