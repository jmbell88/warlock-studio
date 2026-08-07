"""The document: a layer stack, a history, a selection, and a cached composite.

This is the only module in the package that is allowed to know about all the
others, and the only one that pushes anything onto the undo stack. Every entry
point here follows the same three steps -- copy the region that is about to
change, change it, push a ``PatchEdit`` -- which is what makes "one gesture is
one Ctrl+Z" a property of the model rather than a rule the UI has to remember.

The composite is cached and repaired by rectangle. ``take_dirty()`` hands the
accumulated rectangle to whoever is uploading pixels to the GPU and clears it;
``rev`` still ticks on every change, so a caller that only wants to know
*whether* something moved does not have to care about rectangles at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import composite as cp
from . import gradient as grad
from . import transform as tf
from .brush import DEFAULT_SPACING, StrokeState, clamp_brush
from .layers import Layer, LayerStack, new_uid
from .selection import Clipboard, FloatingBuffer, SelectionMask, magic_wand
from .undo import (
    UNDO_BYTES,
    CompoundEdit,
    LayerAddEdit,
    LayerMoveEdit,
    LayerPropsEdit,
    LayerRemoveEdit,
    PatchEdit,
    ReplayEdit,
    SelectionEdit,
    UndoStack,
)

RGBA = tuple[int, int, int, int]

TRANSPARENT: RGBA = (0, 0, 0, 0)
OPAQUE_WHITE: RGBA = (255, 255, 255, 255)

SHAPES = ("line", "rect", "ellipse")

__all__ = ["Document", "RGBA", "TRANSPARENT", "OPAQUE_WHITE", "SHAPES", "normalise_rect"]


def normalise_rect(p0: tuple[int, int], p1: tuple[int, int]) -> tuple[int, int, int, int]:
    """Two corners in any order -> (x0, y0, x1, y1) with x0 <= x1."""
    x0, y0 = p0
    x1, y1 = p1
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _masked_alpha(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A copy of ``pixels`` with a selection mask folded into its alpha.

    The one invariant shared by a lifted buffer and the clipboard: pixels that
    carry their own coverage, so whatever composites them later needs no mask
    at all -- and a feathered edge stays feathered through both.
    """
    out = pixels.copy()
    out[..., 3] = (out[..., 3].astype(np.float32) * mask / 255.0).astype(np.uint8)
    return out


def matte_for(pixels: np.ndarray) -> RGBA | None:
    """What a flattened export puts behind transparency, decided once at load.

    The old editor asked this per eraser stroke and called it ``erase_color``.
    Now the eraser always cuts alpha -- which is what an eraser means in a
    layered document -- and the question moved to where it belongs: a photo
    opened here is still a photo when it is saved, so it flattens onto white; a
    sprite has transparency the user is thinking about, so it keeps it.
    """
    if pixels.shape[2] < 4 or int(pixels[..., 3].min()) < 255:
        return None
    return OPAQUE_WHITE


@dataclass
class Document:
    stack: LayerStack
    matte: RGBA | None = None
    rev: int = 0
    path: Path | None = None
    file_format: str = "png"
    mask: SelectionMask | None = None
    floating: FloatingBuffer | None = None
    clipboard: Clipboard = field(default_factory=Clipboard)
    history: UndoStack = field(default_factory=UndoStack)

    _composite: np.ndarray = field(init=False, repr=False)
    _below: np.ndarray | None = field(init=False, default=None, repr=False)
    _dirty: tuple[int, int, int, int] | None = field(init=False, default=None, repr=False)
    _stroke: StrokeState | None = field(init=False, default=None, repr=False)
    _full: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        width, height = self.stack.size
        self._composite = np.zeros((height, width, 4), dtype=np.uint8)
        self.invalidate_all()

    # -- construction ------------------------------------------------------

    @classmethod
    def blank(
        cls,
        width: int,
        height: int,
        *,
        matte: RGBA | None = None,
        budget: int = UNDO_BYTES,
    ) -> Document:
        layer = Layer.empty(width, height, "Background")
        return cls(stack=LayerStack([layer]), matte=matte, history=UndoStack(budget))

    @classmethod
    def from_pixels(
        cls, pixels: np.ndarray, *, name: str = "Background", budget: int = UNDO_BYTES
    ) -> Document:
        return cls(
            stack=LayerStack([Layer(pixels=pixels, name=name)]),
            matte=matte_for(pixels),
            history=UndoStack(budget),
        )

    @classmethod
    def load(cls, path: Path, *, budget: int = UNDO_BYTES) -> Document:
        """Open any image Pillow can read, or an ORA written by anyone."""
        path = Path(path)
        if path.suffix.lower() == ".ora":
            from .ora import read_ora

            doc = read_ora(path, budget=budget)
        else:
            from PIL import Image

            with Image.open(path) as im:
                im.load()
                pixels = np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
            doc = cls.from_pixels(pixels, budget=budget)
            doc.file_format = "png"
        doc.path = path
        return doc

    @property
    def size(self) -> tuple[int, int]:
        return self.stack.size

    @property
    def composite(self) -> np.ndarray:
        return self._composite

    @property
    def image(self) -> Any:
        """The composite as a Pillow image, sharing the array's memory.

        ``frombuffer`` rather than ``fromarray``: this is asked once a frame by
        the canvas, and copying a megapixel to answer "how big is it" would be
        the most expensive thing the editor does.
        """
        from PIL import Image

        width, height = self.size
        return Image.frombuffer("RGBA", (width, height), self._composite, "raw", "RGBA", 0, 1)

    def flatten(self, *, matte: bool = True) -> np.ndarray:
        return cp.flatten_onto(self._composite.copy(), self.matte if matte else None)

    def png_bytes(self) -> bytes:
        """The flattened document as a PNG. Blocking; callers go off-thread."""
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(self.flatten(), "RGBA").save(buf, "PNG")
        return buf.getvalue()

    # -- the composite cache -----------------------------------------------

    def invalidate(
        self, rect: tuple[int, int, int, int] | None = None, *, layer_uid: int | None = None
    ) -> None:
        """Recomposite a rectangle (or everything) and record it as dirty.

        ``layer_uid`` names the layer that was written. It matters only when
        that layer sits *below* the active one: ``_below`` is the cached
        composite of exactly those layers, so reusing it would recomposite the
        rectangle on top of pixels that no longer exist. Undo is how this
        happens in practice -- a patch is addressed by uid and can land
        anywhere in the stack, however the active layer has moved since.
        """
        if rect is None:
            self.invalidate_all()
            return
        box = self.clip(rect)
        if box is None:
            self.rev += 1
            return
        if self._below is None:
            self._below = self.stack.composite_below()
        elif layer_uid is not None and self.stack.index_of(layer_uid) < self.stack.active_index:
            bx0, by0, bx1, by1 = box
            self._below[by0:by1, bx0:bx1] = self.stack.composite_below_region(box)
        x0, y0, x1, y1 = box
        region = self.stack.composite_region(box, below=self._below)
        self._composite[y0:y1, x0:x1] = cp.to_uint8(region)
        self._mark(box)
        self.rev += 1

    def invalidate_all(self) -> None:
        """A structural change: the below-cache is stale, so nothing is reused.

        Full recompositing is O(layers x canvas) -- at 2048 square by ten
        layers, a fraction of a second. That is affordable *because* it only
        happens on a click (reorder, hide, switch layer), never on a stroke.
        """
        self._below = self.stack.composite_below()
        width, height = self.size
        if self._composite.shape[:2] != (height, width):
            self._composite = np.zeros((height, width, 4), dtype=np.uint8)
        self._composite[:] = cp.to_uint8(
            self.stack.composite_region((0, 0, width, height), below=self._below)
        )
        self._full = True
        self._dirty = None
        self.rev += 1

    def _mark(self, rect: tuple[int, int, int, int]) -> None:
        if self._full:
            return
        if self._dirty is None:
            self._dirty = rect
            return
        a, b, c, d = self._dirty
        x0, y0, x1, y1 = rect
        self._dirty = (min(a, x0), min(b, y0), max(c, x1), max(d, y1))

    def take_dirty(self) -> tuple[int, int, int, int] | None:
        """The rectangle that changed since the last call, or None for "all".

        None is the honest answer after a structural change and after a load;
        a caller that cannot do partial uploads can ignore the return value and
        watch ``rev`` instead.
        """
        if self._full:
            self._full = False
            self._dirty = None
            return None
        rect, self._dirty = self._dirty, None
        return rect

    # -- geometry helpers ---------------------------------------------------

    def clip(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        width, height = self.size
        x0, y0, x1, y1 = rect
        x0 = max(0, min(int(x0), width))
        y0 = max(0, min(int(y0), height))
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None
        return (x0, y0, x1, y1)

    def in_bounds(self, xy: tuple[int, int]) -> bool:
        width, height = self.size
        x, y = int(xy[0]), int(xy[1])
        return 0 <= x < width and 0 <= y < height

    def eyedrop(self, xy: tuple[int, int]) -> RGBA | None:
        """Reads the *composite*, not the active layer: the colour a user is
        pointing at is the one they can see."""
        if not self.in_bounds(xy):
            return None
        r, g, b, a = self._composite[int(xy[1]), int(xy[0])]
        return (int(r), int(g), int(b), int(a))

    # -- writing to a layer -------------------------------------------------

    def _commit_patch(
        self, layer: Layer, rect: tuple[int, int, int, int], before: np.ndarray
    ) -> None:
        """Push one undo step for a region already written, and recomposite."""
        after = layer.pixels[rect[1] : rect[3], rect[0] : rect[2]].copy()
        if np.array_equal(before, after):
            return
        self.history.push(PatchEdit(layer.uid, rect, before, after))
        self.invalidate(rect, layer_uid=layer.uid)

    def _weights(self, rect: tuple[int, int, int, int], weight: np.ndarray) -> np.ndarray:
        """Clip a write to the selection. The same multiply as a brush stamp --
        one rule, so a feathered selection softens every tool identically."""
        if self.mask is None:
            return weight
        x0, y0, x1, y1 = rect
        return weight * (self.mask.mask[y0:y1, x0:x1].astype(np.float32) / 255.0)

    def write_colour(
        self, rect: tuple[int, int, int, int], colour: RGBA, weight: np.ndarray
    ) -> bool:
        """Composite a flat colour into a region of the active layer."""
        box = self.clip(rect)
        if box is None:
            return False
        layer = self.stack.active
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        out = cp.paint_colour(
            before.astype(np.float32), colour, self._weights(box, weight)
        )
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(out)
        self._commit_patch(layer, box, before)
        return True

    # -- strokes ------------------------------------------------------------

    def begin_stroke(
        self,
        point: tuple[float, float],
        colour: RGBA,
        *,
        size: int = 8,
        hardness: float = 0.8,
        opacity: float = 1.0,
        spacing: float | None = None,
        mode: str = "paint",
        strength: float = 0.5,
        symmetry: str = "none",
    ) -> None:
        self.end_stroke()
        layer = self.stack.active
        self._stroke = StrokeState(
            layer_uid=layer.uid,
            size=self.size,
            before=layer.pixels.copy(),
            colour=tuple(colour),
            diameter=clamp_brush(size),
            hardness=hardness,
            opacity=opacity,
            spacing=DEFAULT_SPACING if spacing is None else spacing,
            mode=mode,
            strength=strength,
            symmetry=symmetry,
            clip=self.mask,
        )
        self._stroke.begin(point, layer.pixels)
        self._touch_stroke()

    def stroke_to(self, point: tuple[float, float]) -> None:
        if self._stroke is None:
            return
        self._stroke.to(point, self.stack.by_uid(self._stroke.layer_uid).pixels)
        self._touch_stroke()

    def _touch_stroke(self) -> None:
        """Recomposite what the last dab touched, without pushing history: the
        undo step is the whole stroke, and it is pushed once at release."""
        assert self._stroke is not None
        if self._stroke.dirty is not None:
            self.invalidate(self._stroke.dirty)

    def end_stroke(self) -> bool:
        """Close the stroke and make it exactly one undo step."""
        stroke, self._stroke = self._stroke, None
        if stroke is None or stroke.dirty is None:
            return False
        box = self.clip(stroke.dirty)
        if box is None:
            return False
        layer = self.stack.by_uid(stroke.layer_uid)
        x0, y0, x1, y1 = box
        self._commit_patch(layer, box, stroke.before[y0:y1, x0:x1])
        return True

    # -- fill, shapes, gradients -------------------------------------------

    def fill(
        self,
        xy: tuple[int, int],
        colour: RGBA,
        *,
        thresh: int = 32,
        contiguous: bool = True,
    ) -> bool:
        """Flood fill from a point, on the composite's colours.

        The threshold is what makes this usable on generated art: an SDXL image
        has no flat regions, only nearly-flat ones behind an antialiased edge,
        and an exact-match fill stops after a few hundred pixels every time.
        The region comes from the same predicate the magic wand uses, so the
        two can never disagree about what "similar" means.
        """
        if not self.in_bounds(xy):
            return False
        region = magic_wand(
            self._composite, xy, tolerance=thresh, contiguous=contiguous
        )
        bounds = region.bounds
        if bounds is None:
            return False
        x0, y0, x1, y1 = bounds
        weight = region.mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        return self.write_colour(bounds, colour, weight)

    def shape(
        self,
        kind: str,
        p0: tuple[int, int],
        p1: tuple[int, int],
        colour: RGBA,
        size: int,
        *,
        filled: bool = False,
    ) -> bool:
        """Line, rectangle or ellipse, antialiased through a coverage mask."""
        if kind not in SHAPES:
            raise ValueError(f"unknown shape {kind!r}")
        from PIL import Image, ImageDraw

        width, height = self.size
        size = clamp_brush(size)
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        if kind == "line":
            draw.line([tuple(p0), tuple(p1)], fill=255, width=size, joint="curve")
        else:
            x0, y0, x1, y1 = normalise_rect(p0, p1)
            method = draw.rectangle if kind == "rect" else draw.ellipse
            if filled:
                method((x0, y0, x1, y1), fill=255, outline=255, width=size)
            else:
                method((x0, y0, x1, y1), fill=None, outline=255, width=size)

        coverage = np.asarray(canvas, dtype=np.uint8)
        rows = np.flatnonzero(coverage.any(axis=1))
        cols = np.flatnonzero(coverage.any(axis=0))
        if rows.size == 0:
            return False
        rect = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
        weight = coverage[rect[1] : rect[3], rect[0] : rect[2]].astype(np.float32) / 255.0
        return self.write_colour(rect, colour, weight)

    def gradient(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        start: RGBA,
        end: RGBA,
        *,
        kind: str = "linear",
    ) -> bool:
        """Fill the selection (or the whole layer) with a ramp."""
        width, height = self.size
        rect = self.mask.bounds if self.mask is not None else None
        box = self.clip(rect or (0, 0, width, height))
        if box is None:
            return False
        rgba, weight = grad.render((width, height), p0, p1, start, end, kind)
        x0, y0, x1, y1 = box
        layer = self.stack.active
        before = layer.pixels[y0:y1, x0:x1].copy()
        crop = rgba[y0:y1, x0:x1]
        clipped = self._weights(box, weight[y0:y1, x0:x1])

        # Per-pixel colour, so ``paint_colour``'s flat-colour form does not fit;
        # this is the same arithmetic with the colour varying.
        src_a = clipped
        dst_a = before[..., 3].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        share = np.divide(src_a, out_a, out=np.zeros_like(src_a), where=out_a > 0.0)
        out = np.empty(before.shape, dtype=np.float32)
        rgb = before[..., :3].astype(np.float32)
        out[..., :3] = rgb + (crop[..., :3] * 255.0 - rgb) * share[..., None]
        out[..., 3] = out_a * 255.0
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(out)
        self._commit_patch(layer, box, before)
        return True

    # -- history ------------------------------------------------------------

    def undo(self) -> bool:
        """One Ctrl+Z is one step -- and cancelling a float *is* that step.

        Cancelling a lift reverses the lift's own history entry, so falling
        through to ``history.undo`` afterwards would spend a second step on a
        single keypress.
        """
        if self.cancel_floating():
            return True
        return self.history.undo(self)

    def redo(self) -> bool:
        self.cancel_floating()
        return self.history.redo(self)

    def restore_snapshot(
        self, layers: list[Layer], size: tuple[int, int], active: int
    ) -> None:
        """Undo hook for a whole-canvas operation.

        The layers are copied (the snapshot is held for as long as the edit is,
        and a later stroke must not write into it) but they keep their uids: an
        undo restores the document's *state*, not a set of new layers, and
        every patch recorded before this one addresses those uids.
        """
        self.stack = LayerStack([layer.copy(uid=layer.uid) for layer in layers], active)
        width, height = size
        self._composite = np.zeros((height, width, 4), dtype=np.uint8)
        self.invalidate_all()

    def set_selection_mask(self, mask: np.ndarray | None) -> None:
        """Undo hook for a selection change."""
        self.mask = None if mask is None else SelectionMask(mask)
        self.rev += 1

    # -- selection ----------------------------------------------------------

    def select(self, mask: SelectionMask | None, op: str = "replace") -> None:
        before = None if self.mask is None else self.mask.mask
        if mask is None:
            self.mask = None
        elif op == "replace" or (self.mask is None and op == "add"):
            self.mask = mask.copy()
        elif self.mask is None:
            # Subtracting from nothing, or intersecting with it, is nothing.
            # Adopting the new mask instead would make a subtract-drag on an
            # empty canvas *create* the selection it was meant to cut away.
            self.mask = None
        else:
            self.mask = self.mask.combined(mask, op)
        if self.mask is not None and self.mask.is_empty:
            self.mask = None
        after = None if self.mask is None else self.mask.mask
        # Nothing changed, so nothing to undo. Pushing anyway -- which
        # subtracting from an empty selection did -- moved the head, and dirty
        # is a comparison against the head: an Alt-drag on a canvas with no
        # selection made a saved document ask to be saved again, and spent a
        # Ctrl+Z doing nothing. The both-None case was the one that reproduced;
        # re-selecting the same region (a marquee redrawn over itself, Select
        # All twice, feathering by zero) is the same no-op and was still
        # pushing.
        if before is None and after is None:
            return
        if before is not None and after is not None and np.array_equal(before, after):
            return
        self.history.push(SelectionEdit(before, after))
        self.rev += 1

    def select_all(self) -> None:
        width, height = self.size
        self.select(SelectionMask.full(width, height))

    def deselect(self) -> None:
        self.commit_floating()
        if self.mask is not None:
            self.select(None)

    def invert_selection(self) -> None:
        width, height = self.size
        current = self.mask or SelectionMask(np.zeros((height, width), dtype=np.uint8))
        self.select(current.inverted())

    def feather_selection(self, radius: float) -> None:
        if self.mask is not None:
            self.select(self.mask.feathered(radius))

    def select_wand(
        self, xy: tuple[int, int], *, tolerance: int = 32, op: str = "replace",
        contiguous: bool = True,
    ) -> None:
        self.select(
            magic_wand(self._composite, xy, tolerance=tolerance, contiguous=contiguous), op
        )

    # -- floating pixels ----------------------------------------------------

    def lift(self, mask: SelectionMask | None = None) -> bool:
        """Cut the selection out of the active layer and float it. One step."""
        self.commit_floating()
        mask = mask or self.mask
        if mask is None:
            return False
        bounds = mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return False
        layer = self.stack.active
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        crop = mask.mask[y0:y1, x0:x1]

        # The floating pixels keep their own alpha *multiplied* by the mask, so
        # a feathered lift floats a feathered chunk rather than a hard one.
        pixels = _masked_alpha(before, crop)
        kept = 1.0 - crop.astype(np.float32) / 255.0
        cut = before.copy()
        cut[..., 3] = (before[..., 3].astype(np.float32) * kept).astype(np.uint8)
        layer.pixels[y0:y1, x0:x1] = cut

        after = layer.pixels[y0:y1, x0:x1].copy()
        edit = PatchEdit(layer.uid, box, before, after)
        self.floating = FloatingBuffer(
            pixels=pixels, mask=crop.copy(), offset=(x0, y0), layer_uid=layer.uid,
            lift_edit=edit,
        )
        self.history.push(edit)
        self.invalidate(box, layer_uid=layer.uid)
        return True

    def move_floating(self, dx: int, dy: int) -> None:
        if self.floating is not None:
            self.floating.moved(dx, dy)
            self.rev += 1

    def commit_floating(self) -> bool:
        """Write the floating pixels where they now sit and stop floating."""
        floating, self.floating = self.floating, None
        if floating is None:
            return False
        ox, oy = floating.offset
        fw, fh = floating.size
        box = self.clip((ox, oy, ox + fw, oy + fh))
        if box is None:
            self.rev += 1
            return True
        layer = self.stack.by_uid(floating.layer_uid)
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        crop = floating.pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
        merged = cp.over(cp.to_float(before), cp.to_float(crop))
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8(merged)
        self._commit_patch(layer, box, before)
        return True

    def cancel_floating(self) -> bool:
        """Put the pixels back where they were lifted from, exactly.

        Exact because it is the lift's own undo step rather than a re-paste --
        a feathered lift cannot be re-pasted without rounding. That step is
        reversed *and forgotten*: the user asked for the lift not to have
        happened, so leaving it redoable would let Ctrl+Y replay the alpha-cut
        with no buffer left to put back.

        The lift's *own* step, named on the buffer, and not simply the newest
        one. A buffer floats for as long as the user leaves it floating, and
        every selection op pushes a step meanwhile -- so "the newest step" is
        routinely something else, and reversing that instead dropped the
        lifted pixels, kept the alpha-cut and destroyed an unrelated edit, all
        three unrecoverably.

        A pasted buffer owns no step, and reversing one anyway would undo
        whatever the user did before pasting.
        """
        floating, self.floating = self.floating, None
        if floating is None:
            return False
        if not floating.lifted or not self.history.revoke(self, floating.lift_edit):
            # Nothing to reverse: a paste, or a lift whose step has since been
            # undone or evicted by the byte budget.
            self.rev += 1
        return True

    def delete_floating(self) -> bool:
        """Throw the floating pixels away; the hole is already cut."""
        if self.floating is None:
            return False
        self.floating = None
        self.rev += 1
        return True

    def delete_selection(self) -> bool:
        """Cut the selection out of the active layer without floating it."""
        if self.floating is not None:
            return self.delete_floating()
        if self.mask is None:
            return False
        bounds = self.mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return False
        layer = self.stack.active
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        keep = 1.0 - self.mask.mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        layer.pixels[y0:y1, x0:x1, 3] = (before[..., 3].astype(np.float32) * keep).astype(
            np.uint8
        )
        self._commit_patch(layer, box, before)
        return True

    # -- clipboard ----------------------------------------------------------

    def copy(self) -> bool:
        """Copy the selection (or the floating buffer) to the app clipboard.

        The pixels carry the mask in their alpha, which is exactly the
        invariant :meth:`lift` gives a :class:`FloatingBuffer` -- and it has to
        be, because ``paste`` floats what it takes verbatim and
        ``commit_floating`` composites a buffer without consulting its mask.
        Putting the *rectangular* crop on the clipboard instead meant an
        ordinary Ctrl+C over a lasso, ellipse, wand or feathered selection
        pasted its full bounding box, while hit-testing (which does read the
        mask) still only let the user grab the shape they selected.

        The mask travels beside the pixels regardless: a paste has to know
        which of them are its own for that hit-test.
        """
        if self.floating is not None:
            self.clipboard.put(self.floating.pixels, self.floating.mask)
            return True
        if self.mask is None:
            return False
        bounds = self.mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return False
        x0, y0, x1, y1 = box
        crop = self.mask.mask[y0:y1, x0:x1]
        self.clipboard.put(_masked_alpha(self.stack.active.pixels[y0:y1, x0:x1], crop), crop)
        return True

    def cut(self) -> bool:
        return self.copy() and self.delete_selection()

    # -- free transform -----------------------------------------------------

    def begin_transform(self) -> bool:
        """Lift whatever is being transformed, so it can be moved freely.

        A transform acts on floating pixels and nothing else: it needs to
        rotate a chunk out of alignment with the pixel grid, which cannot be
        expressed as a write back into the layer until it is committed. With no
        selection, the whole layer is lifted -- "rotate this layer" is the
        common case and requiring a select-all first would be busywork.
        """
        if self.floating is not None:
            return True
        if self.mask is None:
            width, height = self.size
            self.select(SelectionMask.full(width, height))
        return self.lift()

    def transform_floating(
        self, *, angle: float | None = None, scale: tuple[float, float] | None = None
    ) -> bool:
        if self.floating is None:
            return False
        self.floating.transform(angle=angle, scale=scale)
        self.rev += 1
        return True

    def flip_floating(self, axis: str) -> bool:
        if self.floating is None:
            return False
        self.floating.flip(axis)
        self.rev += 1
        return True

    def rotate_floating(self, degrees: float) -> bool:
        if self.floating is None:
            return False
        return self.transform_floating(angle=self.floating.angle + degrees)

    def paste(self, at: tuple[int, int] | None = None) -> bool:
        """Paste as a floating buffer, so it can be positioned before it lands."""
        taken = self.clipboard.take()
        if taken is None:
            return False
        self.commit_floating()
        pixels, mask = taken
        if at is None:
            bounds = self.mask.bounds if self.mask is not None else None
            at = (bounds[0], bounds[1]) if bounds else (0, 0)
        self.floating = FloatingBuffer(
            pixels=pixels, mask=mask, offset=(int(at[0]), int(at[1])),
            layer_uid=self.stack.active.uid,
        )
        # A paste is a new branch of history even though it pushes no step of
        # its own; without this, Ctrl+V then Ctrl+Y redoes an unrelated edit.
        self.history.forget_redo()
        self.rev += 1
        return True

    def paste_as_layer(self, pixels: np.ndarray | None = None) -> bool:
        """Paste onto a layer of its own, placed at the top-left.

        A separate entry point rather than a flag on ``paste``: this one lands
        immediately and is undone by removing the layer, where an ordinary
        paste floats until it is put down.
        """
        if pixels is None:
            taken = self.clipboard.take()
            if taken is None:
                return False
            # No second multiply: what the clipboard holds already carries its
            # mask in alpha, and applying it twice darkens every feathered edge.
            pixels, _mask = taken
        self.commit_floating()
        width, height = self.size
        placed = tf.resize_canvas(pixels, (width, height))
        layer = Layer(pixels=placed, name=f"Pasted {len(self.stack) + 1}")
        index = self.stack.insert(self.stack.active_index + 1, layer)
        self.history.push(LayerAddEdit(index, layer))
        self.invalidate_all()
        return True

    def put_clipboard(self, pixels: np.ndarray) -> None:
        """Load the app clipboard from outside -- an OS clipboard image.

        Everything pasted carries a mask, because a paste has to know which of
        its pixels are part of the selection; an image from elsewhere is fully
        selected by definition.
        """
        mask = np.full(pixels.shape[:2], 255, dtype=np.uint8)
        self.clipboard.put(pixels, mask)

    # -- layers -------------------------------------------------------------

    def add_layer(self, name: str | None = None) -> Layer:
        self.commit_floating()
        width, height = self.size
        layer = Layer.empty(width, height, name or f"Layer {len(self.stack) + 1}")
        index = self.stack.insert(self.stack.active_index + 1, layer)
        self.history.push(LayerAddEdit(index, layer))
        self.invalidate_all()
        return layer

    def duplicate_layer(self, index: int | None = None) -> Layer:
        self.commit_floating()
        index = self.stack.active_index if index is None else index
        copy = self.stack.duplicate(index)
        self.history.push(LayerAddEdit(index + 1, copy))
        self.invalidate_all()
        return copy

    def remove_layer(self, index: int | None = None) -> bool:
        if len(self.stack) == 1:
            return False
        self.commit_floating()
        index = self.stack.active_index if index is None else index
        gone = self.stack.remove(index)
        self.history.push(LayerRemoveEdit(index, gone))
        self.invalidate_all()
        return True

    def move_layer(self, index: int, to: int) -> bool:
        to = max(0, min(int(to), len(self.stack) - 1))
        if to == index:
            return False
        self.commit_floating()
        uid = self.stack[index].uid
        self.stack.move(index, to)
        self.history.push(LayerMoveEdit(uid, index, to))
        self.invalidate_all()
        return True

    def set_active_layer(self, index: int) -> None:
        """Not undoable: which layer is selected is a view state, not an edit."""
        if index == self.stack.active_index:
            return
        self.commit_floating()
        self.stack.active_index = max(0, min(int(index), len(self.stack) - 1))
        self.invalidate_all()

    def set_layer_props(
        self, index: int | None = None, *, was: dict | None = None, **props: Any
    ) -> bool:
        """Record a property change as one undo step.

        ``was`` is for a control that mutates the layer live so the canvas
        follows the drag, and only asks for the step once the drag is released.
        By then the layer already holds the new value, so reading "before" off
        it compares a value against itself: nothing is ever pushed, the change
        is not undoable, and the history head does not move -- which is what
        the tab compares against to decide whether the document is dirty.
        """
        index = self.stack.active_index if index is None else index
        layer = self.stack[index]
        source = {} if was is None else was
        before = {key: source.get(key, getattr(layer, key)) for key in props}
        if before == props:
            return False
        for key, value in props.items():
            setattr(layer, key, value)
        self.history.push(LayerPropsEdit(layer.uid, before, dict(props)))
        self.invalidate_all()
        return True

    def merge_down(self, index: int | None = None) -> bool:
        """Flatten a layer into the one beneath it, honouring its blend mode."""
        index = self.stack.active_index if index is None else index
        if index == 0:
            return False
        self.commit_floating()
        width, height = self.size
        upper = self.stack[index]
        lower = self.stack[index - 1]
        merged = cp.to_uint8(
            cp.stack_region(
                [
                    (lower.pixels, lower.opacity, lower.blend),
                    (upper.pixels, upper.opacity, upper.blend),
                ]
                if upper.visible
                else [(lower.pixels, lower.opacity, lower.blend)],
                (0, 0, width, height),
            )
        )
        before = lower.pixels.copy()
        lower.pixels[:] = merged
        # The merged result already has the lower layer's opacity baked into
        # it, so leaving that applied a second time would double it. Its blend
        # mode is *dropped* rather than absorbed -- there is nothing here for
        # it to blend against, since what sits below the pair is not part of
        # this composite -- and it has to be reset either way: a lower layer
        # left on "multiply" would then multiply the merged pixels against
        # everything beneath it.
        props_before = {"opacity": lower.opacity, "blend": lower.blend}
        lower.opacity, lower.blend = 1.0, "normal"
        removed = self.stack.remove(index)
        self.stack.active_index = index - 1
        self.history.push(
            CompoundEdit(
                [
                    PatchEdit(lower.uid, (0, 0, width, height), before, merged.copy()),
                    LayerPropsEdit(
                        lower.uid, props_before, {"opacity": 1.0, "blend": "normal"}
                    ),
                    LayerRemoveEdit(index, removed),
                ]
            )
        )
        self.invalidate_all()
        return True

    def flatten_layers(self) -> None:
        """Collapse the stack to one layer. Undoable as a canvas-level op."""
        if len(self.stack) == 1:
            return
        self.commit_floating()
        # Replay must be a pure function of the document, and minting a uid is
        # the one part of this op that is not: a redo would produce a layer
        # with a new identity, stranding every patch recorded above it. The uid
        # is drawn once and closed over, so every replay lands on the same one.
        uid = new_uid()
        self._replay(lambda: self._do_flatten(uid))

    def _do_flatten(self, uid: int) -> None:
        flat = self.stack.flatten()
        self.stack = LayerStack([Layer(pixels=flat, name="Flattened", uid=uid)], 0)

    # -- whole-canvas geometry ---------------------------------------------

    def _replay(self, run: Any) -> None:
        """Run a canvas-level op, recording a snapshot to undo it by.

        ``run`` is the *raw* work, never the public method: redo re-runs it
        directly, and a redo that went back through the public entry point
        would push a second history step for an operation that is already on
        the stack.

        These are the only operations that still cost a full copy. Redo
        replays instead of storing a second one, which is safe here and nowhere
        else -- flips, rotations and rescales are pure functions of the
        document, with nothing accumulated and nothing random.
        """
        snapshot = [layer.copy(uid=layer.uid) for layer in self.stack]
        size = self.size
        active = self.stack.active_index
        mask = None if self.mask is None else self.mask.mask.copy()
        run()

        def replay(doc: Any) -> None:
            run()
            doc.invalidate_all()

        self.history.push(ReplayEdit(snapshot, size, active, replay, mask))
        self.invalidate_all()

    def _map_planes(self, fn: Any) -> None:
        for layer in self.stack:
            layer.pixels = fn(layer.pixels)
        if self.mask is not None:
            self.mask = SelectionMask(fn(self.mask.mask))

    def flip(self, axis: str) -> None:
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.flip(plane, axis)))

    def rotate90(self, quarters: int = 1) -> None:
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.rotate90(plane, quarters)))

    def scale(self, size: tuple[int, int]) -> None:
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.scale(plane, size)))

    def crop(self, rect: tuple[int, int, int, int]) -> bool:
        box = self.clip(rect)
        if box is None:
            return False
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.crop(plane, box)))
        return True

    def crop_to_selection(self) -> bool:
        bounds = self.mask.bounds if self.mask is not None else None
        return self.crop(bounds) if bounds else False

    def resize_canvas(self, size: tuple[int, int], offset: tuple[int, int] = (0, 0)) -> None:
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.resize_canvas(plane, size, offset)))
