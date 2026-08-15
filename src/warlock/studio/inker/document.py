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

What stays *here* is the state and the plumbing: the fields, the constructors,
the composite and per-frame flatten caches, cel autovivify, and ``_commit_patch``
and ``_replay`` -- the two write paths every concern ends at. The concern blocks
themselves are method-only mixins in ``_doc_*.py`` siblings, listed on the class.
They hold no state of their own, so this is where to look for what a document
*is*, and there for what it can be asked to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import composite as cp

# Nothing in *this* module calls it any more -- the filter session moved to
# ``_doc_paint`` -- but the name has to stay bound here. The filter-memo tests
# reach for ``document.filters`` and monkeypatch ``apply_named`` on the module
# object they find; an import in ``_doc_paint`` alone binds the name there and
# on the package, never on this module, and the patch would land nowhere.
from . import filters  # noqa: F401
from . import indexed as ix
from ._doc_anim import AnimOps
from ._doc_geometry import GeometryOps
from ._doc_history import HistoryOps
from ._doc_indexed import IndexedOps
from ._doc_layers import LayerOps
from ._doc_paint import SHAPES, PaintOps, normalise_rect
from ._doc_ranges import RangeOps
from ._doc_selection import SelectionOps
from .anim_edits import CelSetEdit
from .animation import Animation
from .brush import StrokeState
from .layers import Layer, LayerStack
from .selection import Clipboard, FloatingBuffer, SelectionMask
from .undo import (
    UNDO_BYTES,
    CompoundEdit,
    PatchEdit,
    ReplayEdit,
    UndoStack,
)

RGBA = tuple[int, int, int, int]

TRANSPARENT: RGBA = (0, 0, 0, 0)
OPAQUE_WHITE: RGBA = (255, 255, 255, 255)

#: Ceiling on the per-frame flatten cache. Sized against the undo budget it
#: sits beside (192 MiB) rather than against a frame count: at 2048 square a
#: frame is 16 MiB, so this is eight of them, and at sprite sizes it is
#: thousands. A count would have been generous at one size and ruinous at the
#: other.
FRAME_CACHE_BYTES = 128 * 1024 * 1024

__all__ = [
    "FRAME_CACHE_BYTES",
    "Document",
    "RGBA",
    "TRANSPARENT",
    "OPAQUE_WHITE",
    "SHAPES",
    "normalise_rect",
]


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
class Document(
    AnimOps, PaintOps, HistoryOps, SelectionOps, LayerOps, GeometryOps, IndexedOps,
    RangeOps,
):
    stack: LayerStack
    matte: RGBA | None = None
    rev: int = 0
    path: Path | None = None
    file_format: str = "png"
    mask: SelectionMask | None = None
    floating: FloatingBuffer | None = None
    clipboard: Clipboard = field(default_factory=Clipboard)
    history: UndoStack = field(default_factory=UndoStack)
    anim: Animation | None = None
    #: The document's colour table, or None for an ordinary RGBA document.
    #: Present means **indexed**: every write is snapped onto these colours as
    #: it commits. It is a list rather than a set because the order is what the
    #: user arranged and what an exported ``.gpl`` and an exported GIF both
    #: carry. See :mod:`.indexed` for why this is a constraint on writes rather
    #: than an index plane.
    palette: list[RGBA] | None = None

    _composite: np.ndarray = field(init=False, repr=False)
    #: Per-frame change counters, keyed by frame uid, for the flatten cache.
    #: A plain dict rather than a field on ``Frame`` so a frame carries only
    #: what a *save* carries -- a stamp is cache bookkeeping and has no
    #: business round-tripping through a file.
    _frame_stamps: dict[int, int] = field(default_factory=dict, repr=False)
    _below: np.ndarray | None = field(init=False, default=None, repr=False)
    _dirty: tuple[int, int, int, int] | None = field(init=False, default=None, repr=False)
    _stroke: StrokeState | None = field(init=False, default=None, repr=False)
    #: An open filter session: the rect being filtered, the pixels as they were
    #: before it opened, and the **uid** of the layer they came off. A live
    #: preview recomputes from those pixels rather than from the last preview,
    #: which is the same rule a stroke's coverage follows -- filtering the
    #: filtered result compounds every slider move.
    #:
    #: The uid is the third element for the reason ``undo.py`` states as a rule:
    #: address the subject, never a position. A filter popup lives across frames
    #: while the user drags a slider, and this session used to name no layer at
    #: all -- commit and cancel acted on whatever ``stack.active`` happened to be
    #: by then, so moving the active layer or the playhead mid-popup wrote a
    #: filter into a layer that was never filtered, and on an animated document
    #: could land on the shared read-only placeholder plane and raise out of the
    #: middle of a draw. See :meth:`_filter_layer`.
    _filter: tuple[tuple[int, int, int, int], np.ndarray, int] | None = field(
        init=False, default=None, repr=False
    )
    #: The last ``(name, params) -> filtered pixels`` a preview computed, for
    #: the life of one filter session. See :meth:`preview_filter`.
    _filter_memo: tuple[tuple[Any, ...], np.ndarray] | None = field(
        init=False, default=None, repr=False
    )
    _full: bool = field(init=False, default=True, repr=False)
    #: Cels autovivified by writes that have not yet been committed. They ride
    #: along into the same ``CompoundEdit`` as the patch, so drawing on an empty
    #: frame is one Ctrl+Z rather than two. A *list* rather than one slot: the
    #: convention is that every write commits before the next one autovivifies,
    #: and while that holds there is never more than one -- but the cost of the
    #: convention being broken was a second write landing on the shared
    #: read-only placeholder plane and raising out of the middle of a stroke,
    #: which is not a failure mode worth keeping in exchange for a scalar.
    _pending_cels: list[Any] = field(init=False, default_factory=list, repr=False)
    #: frame uid -> (stamp it was built at, pixels), least-recently-used first
    #: in ``_frame_order``. See :meth:`frame_flat`.
    _frame_cache: dict[int, tuple[int, np.ndarray]] = field(
        default_factory=dict, repr=False
    )
    _frame_order: list[int] = field(default_factory=list, repr=False)
    #: Frames that have gone, for whoever is holding a *texture* keyed on one.
    #: Plain ints and a drain, so the document goes on knowing nothing about GL:
    #: see ``panes/inker_textures.release_dropped``.
    _dropped_frames: list[int] = field(default_factory=list, repr=False)
    #: Per-*layer* change counters, keyed by layer uid, beside the per-frame
    #: ones above and for a different consumer: a cel thumbnail is a picture of
    #: one cel, and a frame stamp moves whenever any track on that frame does.
    #: Keying a thumbnail on the frame stamp would re-shrink every cel in a
    #: column each time one of them was drawn on -- ten uploads for one dab.
    #:
    #: Bumped where a *write* is announced (``_stamp_layer``) and by the
    #: whole-grid paths, and deliberately **not** by ``invalidate_all``: that
    #: is the composite's cache and most of what reaches it changes no pixels
    #: at all, which is the same "stamps no frame" lesson stated there.
    _layer_stamps: dict[int, int] = field(default_factory=dict, repr=False)

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

        On an animated document the named layer may not be in the current
        stack at all: a patch addresses a *cel*, and undo can be pressed after
        the playhead has moved off the frame the edit was made on. That is not
        an error and it is not a no-op either -- the pixels really did change,
        just not any that are on screen -- so the frames carrying that cel have
        their flatten caches stamped and nothing is recomposited. Reaching
        ``index_of`` with such a uid is what used to raise ``KeyError`` out of
        the middle of an undo.
        """
        if rect is None:
            self.invalidate_all()
            return
        if layer_uid is not None:
            self._stamp_layer(layer_uid)
            if self.anim is not None and not self._in_stack(layer_uid):
                self.rev += 1
                return
        else:
            self._stamp_current()
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

        It stamps **no** frame. This is about the *composite* -- the cache of
        what is on screen -- and most of what reaches it changes no pixels at
        all: switching layer, switching frame, rebuilding the view after an
        edit that has already stamped what it touched. Stamping here instead
        looked conservative and was the opposite of a cache: every step of the
        playhead threw away every frame's flatten, so onion skinning
        recomposited its neighbours on every keypress. The callers that really
        do change every frame -- a grid edit, a matte, a whole-canvas geometry
        op, a snapshot restore -- say so themselves with ``_stamp_all``.
        """
        # The below-cache is dropped, not rebuilt (B28): passing no ``below``
        # lets ``composite_region`` do one pass over the whole stack, where
        # eagerly rebuilding ``_below`` first paid a second full-canvas pass
        # over the lower layers on every structural change -- for a cache the
        # next *stroke* rebuilds lazily anyway (see ``invalidate``).
        self._below = None
        width, height = self.size
        if self._composite.shape[:2] != (height, width):
            self._composite = np.zeros((height, width, 4), dtype=np.uint8)
        self._composite[:] = cp.to_uint8(
            self.stack.composite_region((0, 0, width, height))
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

    # -- the animation grid -------------------------------------------------

    def _in_stack(self, layer_uid: int) -> bool:
        return any(layer.uid == layer_uid for layer in self.stack)

    def layer_by_uid(self, uid: int) -> Layer:
        """The layer a uid names, on *any* frame.

        The current stack first, because that is the overwhelmingly common case
        and it is also the only correct answer when a placeholder and a cel
        could both match. Only then the grid, which is what makes an undo issued
        after the playhead moved land on the cel the edit was made to rather
        than raising.
        """
        try:
            return self.stack.by_uid(uid)
        except KeyError:
            if self.anim is None:
                raise
        for layer in self.anim.unique_cel_layers():
            if layer.uid == uid:
                return layer
        raise KeyError(uid)

    def frame_stamp(self, frame_uid: int) -> int:
        return self._frame_stamps.get(frame_uid, 0)

    def frame_flat(self, frame_uid: int) -> np.ndarray | None:
        """One frame's flattened RGBA, cached against that frame's stamp.

        Onion skinning asks for two or four of these every frame the app draws,
        and playback asks for one per tick, so it has to be a lookup rather than
        a composite. The cache is keyed on the *stamp* and not on ``rev``,
        because ``rev`` moves for any change anywhere and would throw away every
        frame's flatten each time the user drew a single dab on one of them.

        Bounded by bytes and evicted least-recently-used, which is the right
        policy here and not the one ``UndoStack`` uses: history has to keep the
        *newest* steps and drops from the old end, while a cache should keep
        whatever is being looked at -- and during playback that is a window
        sweeping round the timeline, which oldest-first eviction would evict
        exactly one tick before it came round again.
        """
        anim = self.anim
        if anim is None:
            return None
        try:
            frame = anim.frames[anim.frame_index(frame_uid)]
        except KeyError:
            return None
        stamp = self.frame_stamp(frame_uid)
        hit = self._frame_cache.get(frame_uid)
        if hit is not None and hit[0] == stamp:
            self._frame_order.remove(frame_uid)
            self._frame_order.append(frame_uid)
            return hit[1]
        flat = LayerStack(anim.layers_for(frame, self.size), 0).flatten()
        self._frame_cache[frame_uid] = (stamp, flat)
        if frame_uid in self._frame_order:
            self._frame_order.remove(frame_uid)
        self._frame_order.append(frame_uid)
        self._evict_frames()
        return flat

    def _evict_frames(self) -> None:
        # Never the entry just stored, however big it is: a single frame over
        # the whole budget must still be returnable, or a large canvas caches
        # nothing and recomposites on every draw.
        total = sum(pixels.nbytes for _stamp, pixels in self._frame_cache.values())
        while total > FRAME_CACHE_BYTES and len(self._frame_order) > 1:
            oldest = self._frame_order.pop(0)
            entry = self._frame_cache.pop(oldest, None)
            if entry is not None:
                total -= entry[1].nbytes

    def frame_cache_bytes(self) -> int:
        return sum(pixels.nbytes for _stamp, pixels in self._frame_cache.values())

    def _forget_frame(self, frame_uid: int) -> None:
        """Drop everything keyed on a frame that no longer exists.

        The stamp and the cache entry go **together**, which is the whole point
        of doing this in one method. Dropping the stamp alone restarts that
        frame's counter at zero, and a frame uid that comes back -- an undone
        delete re-inserts the same ``Frame`` object -- would then match a cache
        entry built before it was removed and show pixels from another edit.
        """
        self._frame_cache.pop(frame_uid, None)
        if frame_uid in self._frame_order:
            self._frame_order.remove(frame_uid)
        self._frame_stamps.pop(frame_uid, None)
        self._dropped_frames.append(frame_uid)

    def _forget_all_frames(self) -> None:
        for uid in set(self._frame_stamps) | set(self._frame_cache):
            self._forget_frame(uid)

    def take_dropped_frames(self) -> list[int]:
        """The frames that have gone since the last call, and clear the list.

        A drain rather than a set difference recomputed every draw: a pane that
        holds one texture per frame needs to know which ones to release, and
        walking the live grid to find out would be a per-frame scan to answer a
        question that is almost always "none".
        """
        gone, self._dropped_frames = self._dropped_frames, []
        return gone

    def _stamp(self, frame_uids: Any) -> None:
        for uid in frame_uids:
            self._frame_stamps[uid] = self._frame_stamps.get(uid, 0) + 1

    def _stamp_current(self) -> None:
        if self.anim is not None and self.anim.frames:
            self._stamp((self.anim.frame.uid,))

    def _stamp_layer(self, layer_uid: int) -> None:
        """Stamp every frame a write to this layer changes.

        The current frame is included unconditionally: the layer may be a
        placeholder, which is in no ``cels`` entry and so answers the grid query
        with nothing, and a caller that has just written to the visible canvas
        must not be told its flatten is unchanged.
        """
        self._layer_stamps[layer_uid] = self._layer_stamps.get(layer_uid, 0) + 1
        if self.anim is None:
            return
        uids = self.anim.frame_uids_of_layer(layer_uid)
        if self.anim.frames:
            uids.add(self.anim.frame.uid)
        self._stamp(uids)

    def _stamp_all(self) -> None:
        if self.anim is None:
            return
        self._stamp([frame.uid for frame in self.anim.frames])
        # Every distinct cel, not every slot: a whole-grid change (a track
        # property, a geometry op, a snapshot restore) really does alter what
        # each cel's thumbnail should show, and walking the slots would bump a
        # linked cel once per frame it appears on for no gain.
        for layer in self.anim.unique_cel_layers():
            self._layer_stamps[layer.uid] = self._layer_stamps.get(layer.uid, 0) + 1

    def layer_stamp(self, layer_uid: int) -> int:
        """How many writes this layer has been told about. See ``_layer_stamps``."""
        return self._layer_stamps.get(layer_uid, 0)

    # -- autovivify ---------------------------------------------------------

    def _ensure_cel_for(self, layer_uid: int) -> None:
        """Turn the placeholder a write is about to land on into a real cel.

        Called by every path that writes pixels, immediately before it does, and
        keyed by uid rather than by "the active layer" because committing a
        floating buffer writes to whichever layer it was lifted from.

        The new cel keeps the placeholder's uid, which was already stable, so a
        caller holding the uid across this call is unaffected -- and the layer
        is swapped into the stack *in place* rather than by re-materialising,
        because the caller is generally one line away from reading
        ``stack.active`` and would otherwise get the object it just replaced.

        An already-pending cel does not stop a second one: the convention is
        that every write commits before the next autovivifies, so there is
        normally at most one, and refusing here in the case where there is not
        left the second write pointed at the shared read-only placeholder plane
        -- a ``ValueError`` out of the middle of a gesture rather than a cel.
        """
        anim = self.anim
        if anim is None:
            return
        try:
            index = self.stack.index_of(layer_uid)
        except KeyError:
            return
        placeholder = self.stack[index]
        if not anim.is_placeholder(placeholder) or index >= len(anim.tracks):
            return
        track, frame = anim.tracks[index], anim.frame
        width, height = self.size
        # Every one of the five track properties, not four: ``alpha_lock`` was
        # the one left out, and the write that autovivified the cel is normally
        # one line away from reading it back off the layer -- ``begin_stroke``
        # samples ``layer.alpha_lock`` immediately after ``_ensure_active_cel``
        # -- so a missing lock did not merely mislabel the cel, it painted
        # through "preserve transparency" on the first stroke of every fresh
        # frame. ``placeholder`` and ``layers_for`` both copy all five; this is
        # the third copy of that list and it has to agree with them.
        real = Layer(
            pixels=cp.empty(width, height),
            name=track.name,
            opacity=track.opacity,
            visible=track.visible,
            blend=track.blend,
            alpha_lock=track.alpha_lock,
            uid=placeholder.uid,
        )
        anim.cels[(track.uid, frame.uid)] = real
        self.stack.layers[index] = real
        self._pending_cels.append(
            CelSetEdit(track.uid, frame.uid, None, real, pinned=True)
        )

    def _ensure_active_cel(self) -> None:
        if self.anim is not None:
            self._ensure_cel_for(self.stack.active.uid)

    def _discard_pending_cel(self) -> None:
        """Undo an autovivify whose write never happened, pushing nothing.

        "A step that changes nothing is not pushed" applies to the cel as much
        as to the pixels: a brush-down with no drag would otherwise leave a
        blank cel behind and a document asking to be saved.
        """
        pending, self._pending_cels = self._pending_cels, []
        for edit in reversed(pending):
            self._set_cel(edit.track_uid, edit.frame_uid, None)

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

    # -- writing to a layer -------------------------------------------------

    def _commit_patch(
        self, layer: Layer, rect: tuple[int, int, int, int], before: np.ndarray
    ) -> None:
        """Push one undo step for a region already written, and recomposite.

        On an animated document the step may be two things that must undo
        together: the cel this write brought into existence, and the write
        itself. A no-op write takes the cel back out again and pushes nothing,
        which is the same rule the selection ops follow -- a step that changes
        nothing must not move the history head, or the document asks to be saved
        for a gesture that did not happen.

        **This is where an indexed document is snapped**, and it is the one
        place worth doing it: every write that is undoable arrives here --
        strokes, fills, shapes, gradients, filters, pastes, floating commits --
        so the constraint holds for all of them rather than for the list
        somebody remembered. Snapped *before* ``after`` is read, so the undo
        step records the pixels the document actually ends up with, and an undo
        followed by a redo lands on the same colours.
        """
        if self.palette:
            x0, y0, x1, y1 = rect
            region = layer.pixels[y0:y1, x0:x1]
            layer.pixels[y0:y1, x0:x1] = ix.snap(region, self.palette)
        after = layer.pixels[rect[1] : rect[3], rect[0] : rect[2]].copy()
        if np.array_equal(before, after):
            self._discard_pending_cel()
            return
        pending, self._pending_cels = self._pending_cels, []
        edit: Any = PatchEdit(layer.uid, rect, before, after)
        self.history.push(edit if not pending else CompoundEdit([*pending, edit]))
        self.invalidate(rect, layer_uid=layer.uid)

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
        grid = self._grid_snapshot()
        snapshot = [
            layer.copy(uid=layer.uid)
            for layer in (self.stack if grid is None else self.anim.unique_cel_layers())
        ]
        size = self.size
        active = self.stack.active_index
        mask = None if self.mask is None else self.mask.mask.copy()
        run()

        def replay(doc: Any) -> None:
            run()
            doc.invalidate_all()

        self.history.push(ReplayEdit(snapshot, size, active, replay, mask, grid))
        self.invalidate_all()

    def _grid_snapshot(self) -> dict[str, Any] | None:
        """Enough of the grid to put the link structure back, by position.

        The cel slots are recorded as *indices into the snapshot list* rather
        than as layers, which is the whole trick: two slots holding one index
        come back as two slots holding one object, so a linked cel is still
        linked after an undo. Recording layers would restore two equal copies
        and quietly break the link -- and the user would only find out on the
        next stroke, when one frame changed and the other did not.
        """
        anim = self.anim
        if anim is None:
            return None
        order = {id(layer): i for i, layer in enumerate(anim.unique_cel_layers())}
        return {
            "frames": list(anim.frames),
            "tracks": list(anim.tracks),
            "slots": {key: order[id(layer)] for key, layer in anim.cels.items()},
            "current": anim.current,
        }

    def _map_planes(self, fn: Any) -> None:
        if self.mask is not None:
            self.mask = SelectionMask(fn(self.mask.mask))
        anim = self.anim
        if anim is None:
            for layer in self.stack:
                layer.pixels = fn(layer.pixels)
            return
        # Each *distinct* cel exactly once. Walking the stack, or the slots,
        # would rotate a background linked across three frames three times.
        for layer in anim.unique_cel_layers():
            layer.pixels = fn(layer.pixels)
        self._stamp_all()
        anim._blank = None
        probe = next(anim.unique_cel_layers(), None)
        if probe is not None:
            size = probe.size
        else:
            # A grid with no cels at all: nothing was mapped, so the new canvas
            # size has to be asked for rather than observed.
            width, height = self.size
            plane = fn(cp.empty(width, height))
            size = (plane.shape[1], plane.shape[0])
        # Rebuilt against ``size`` rather than through ``_materialize_frame``,
        # whose ``self.size`` still reads the old canvas off a stale stack.
        self.stack = LayerStack(anim.layers_for(anim.frame, size), self.stack.active_index)
