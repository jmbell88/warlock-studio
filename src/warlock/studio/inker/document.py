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

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import composite as cp
from . import gradient as grad
from . import transform as tf
from .anim_edits import (
    AnimateEdit,
    CelSetEdit,
    FrameAddEdit,
    FrameDurationEdit,
    FrameMoveEdit,
    FrameRemoveEdit,
    TagsEdit,
    TrackAddEdit,
    TrackMoveEdit,
    TrackPropsEdit,
    TrackRemoveEdit,
)
from .animation import Animation, Frame, Tag, Track, clamp_duration
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
    anim: Animation | None = None

    _composite: np.ndarray = field(init=False, repr=False)
    #: Per-frame change counters, keyed by frame uid, for the flatten cache.
    #: A plain dict rather than a field on ``Frame`` so a frame carries only
    #: what a *save* carries -- a stamp is cache bookkeeping and has no
    #: business round-tripping through a file.
    _frame_stamps: dict[int, int] = field(default_factory=dict, repr=False)
    _below: np.ndarray | None = field(init=False, default=None, repr=False)
    _dirty: tuple[int, int, int, int] | None = field(init=False, default=None, repr=False)
    _stroke: StrokeState | None = field(init=False, default=None, repr=False)
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

    # -- the animation grid -------------------------------------------------

    @property
    def animated(self) -> bool:
        return self.anim is not None

    def _in_stack(self, layer_uid: int) -> bool:
        return any(layer.uid == layer_uid for layer in self.stack)

    def _released(self, layers: Any) -> frozenset[int]:
        """Which of these the grid no longer holds, by ``id()``.

        Called *after* a removal, and it is the answer the byte budget wants:
        the history pins pixels only where nothing else does, so a cel still
        linked into another frame costs the budget nothing however the step is
        undone. See ``anim_edits.charged``.
        """
        live = (
            set()
            if self.anim is None
            else {id(layer) for layer in self.anim.cels.values()}
        )
        return frozenset(
            id(layer) for layer in layers if layer is not None and id(layer) not in live
        )

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
        if self.anim is None:
            return
        uids = self.anim.frame_uids_of_layer(layer_uid)
        if self.anim.frames:
            uids.add(self.anim.frame.uid)
        self._stamp(uids)

    def _stamp_all(self) -> None:
        if self.anim is not None:
            self._stamp([frame.uid for frame in self.anim.frames])

    def ensure_animation(self) -> Animation:
        """Turn a still document into a one-frame animation, in place.

        Nothing is copied and nothing is renumbered: each existing ``Layer``
        *becomes* the first frame's cel and each ``Track`` takes that layer's
        uid, so every patch already on the undo stack goes on addressing the
        same pixels. That is what lets the first "Add frame" be undone back to a
        plain document without the history underneath it noticing.
        """
        if self.anim is not None:
            return self.anim
        frame = Frame()
        tracks = [Track.of(layer) for layer in self.stack]
        self.anim = Animation(
            tracks=tracks,
            frames=[frame],
            cels={
                (track.uid, frame.uid): layer
                for track, layer in zip(tracks, self.stack, strict=True)
            },
            current=0,
        )
        return self.anim

    def drop_animation(self) -> None:
        """Back to a still document showing whatever frame is current.

        The undo hook for ``AnimateEdit``. The stack is left exactly as it
        stands, which is right because ``ensure_animation`` did not build it --
        on the only path that matters (animate, then undo) it is still the
        original layer objects.
        """
        self.anim = None
        self._forget_all_frames()

    def _materialize_frame(self, *, active: int | None = None) -> None:
        """Rebuild ``stack`` as the view of the current frame.

        This is the join between the two models, and the reason the rest of the
        editor needed almost no changes: panes, tools, the floating buffer, the
        selection, the ``_below`` cache and the compositor all go on seeing an
        ordinary ``LayerStack``, because that is genuinely what they are handed.
        """
        if self.anim is None or not self.anim.frames:
            return
        layers = self.anim.layers_for(self.anim.frame, self.size)
        if not layers:
            # A grid with no tracks. ``LayerStack`` has no empty form, so the
            # previous frame's list stays -- which is only ever reached while an
            # edit is midway through rebuilding the rows, and the edit ends with
            # another ``_anim_changed``.
            return
        index = self.stack.active_index if active is None else active
        self.stack = LayerStack(layers, max(0, min(index, len(layers) - 1)))

    def set_current_frame(self, index: int) -> None:
        """Move the playhead. Not undoable -- it is view state, like the
        active layer, and a document that asked to be saved because the user
        looked at another frame would make ``dirty`` a lie."""
        if self.anim is None:
            return
        index = max(0, min(int(index), len(self.anim.frames) - 1))
        if index == self.anim.current:
            return
        self.commit_floating()
        self.anim.current = index
        self._materialize_frame()
        self.invalidate_all()

    def _anim_changed(self, *, active: int | None = None) -> None:
        """What every grid edit ends with: rebuild the view, recomposite.

        Unconditional rather than "when the current frame is affected". Working
        out whether it was means asking whether a track property, a reorder or a
        link touched the frame on screen, and the cost of being wrong is a
        viewport showing pixels the document no longer has -- against a full
        recomposite on a click, which ``invalidate_all`` already argues is
        affordable.

        Every frame is stamped, for the reason ``invalidate_all`` deliberately
        does not: a track is authoritative over every frame it appears in, so a
        hide or an opacity change really does alter every cached flatten in the
        document.
        """
        self._stamp_all()
        self._materialize_frame(active=active)
        self.invalidate_all()

    # -- raw grid mutation, for the edits to call --------------------------

    def _set_animation(self, anim: Animation | None) -> None:
        self.anim = anim
        if anim is None:
            self._forget_all_frames()
        self._anim_changed()

    def _put_frame(self, index: int, frame: Frame, cels: dict[int, Layer]) -> None:
        anim = self.anim
        assert anim is not None
        anim.frames.insert(max(0, min(index, len(anim.frames))), frame)
        for track_uid, layer in cels.items():
            anim.cels[(track_uid, frame.uid)] = layer
        self._anim_changed()

    def _drop_frame(self, frame: Frame) -> None:
        anim = self.anim
        assert anim is not None
        # The floor is the public wrappers' -- ``remove_frame`` refuses the last
        # frame, and the only other caller is ``FrameAddEdit.undo``, whose add
        # was made against a grid that already had one. Stated here because
        # everything below assumes ``anim.frame`` still answers.
        assert len(anim.frames) > 1, "a grid keeps at least one frame"
        anim.frames.pop(anim.frame_index(frame.uid))
        for key in [key for key in anim.cels if key[1] == frame.uid]:
            del anim.cels[key]
        anim.forget_placeholders(frame_uid=frame.uid)
        self._forget_frame(frame.uid)
        anim.current = max(0, min(anim.current, len(anim.frames) - 1))
        self._anim_changed()

    def _move_frame(self, frame_uid: int, to: int) -> None:
        anim = self.anim
        assert anim is not None
        frame = anim.frames.pop(anim.frame_index(frame_uid))
        anim.frames.insert(max(0, min(to, len(anim.frames))), frame)
        anim.current = anim.frame_index(frame_uid)
        self._anim_changed()

    def _set_duration(self, frame_uid: int, ms: int) -> None:
        anim = self.anim
        assert anim is not None
        anim.frames[anim.frame_index(frame_uid)].duration_ms = clamp_duration(ms)
        self.rev += 1

    def _put_track(self, index: int, track: Track, cels: dict[int, Layer]) -> None:
        anim = self.anim
        assert anim is not None
        index = max(0, min(index, len(anim.tracks)))
        anim.tracks.insert(index, track)
        for frame_uid, layer in cels.items():
            anim.cels[(track.uid, frame_uid)] = layer
        self._anim_changed(active=index)

    def _drop_track(self, track: Track) -> None:
        anim = self.anim
        assert anim is not None
        # As in ``_drop_frame``: ``remove_layer`` refuses the last row and
        # ``TrackAddEdit.undo`` can only reverse an add made above one.
        assert len(anim.tracks) > 1, "a grid keeps at least one track"
        index = anim.track_index(track.uid)
        anim.tracks.pop(index)
        for key in [key for key in anim.cels if key[0] == track.uid]:
            del anim.cels[key]
        anim.forget_placeholders(track_uid=track.uid)
        self._anim_changed(active=min(index, len(anim.tracks) - 1))

    def _move_track(self, track_uid: int, to: int) -> None:
        anim = self.anim
        assert anim is not None
        track = anim.tracks.pop(anim.track_index(track_uid))
        to = max(0, min(to, len(anim.tracks)))
        anim.tracks.insert(to, track)
        self._anim_changed(active=to)

    def _set_track_props(self, track_uid: int, props: dict) -> None:
        anim = self.anim
        assert anim is not None
        track = anim.tracks[anim.track_index(track_uid)]
        for key, value in props.items():
            setattr(track, key, value)
        self._anim_changed()

    def _set_cel(self, track_uid: int, frame_uid: int, layer: Layer | None) -> None:
        anim = self.anim
        assert anim is not None
        if layer is None:
            anim.cels.pop((track_uid, frame_uid), None)
        else:
            anim.cels[(track_uid, frame_uid)] = layer
        # No targeted stamp: ``_anim_changed`` stamps every frame, this one
        # included, and a second bump of the same counter buys nothing.
        self._anim_changed()

    # -- frames -------------------------------------------------------------

    def add_frame(
        self, index: int | None = None, *, link: bool = False, copy: bool = False
    ) -> Frame:
        """Append or insert a frame, optionally carrying the current one's cels.

        Three shapes in one entry point because they differ only in what goes
        into ``cels``: blank (nothing), duplicate-linked (the same objects, so
        editing either frame edits both) and duplicate-copied (fresh copies).

        The first call on a still document is one ``CompoundEdit`` of
        ``AnimateEdit`` and this ``FrameAddEdit``, so a single Ctrl+Z goes all
        the way back to a plain document rather than leaving a one-frame
        animation nobody asked for.
        """
        self.commit_floating()
        edits: list[Any] = []
        if self.anim is None:
            edits.append(AnimateEdit(self.ensure_animation()))
        anim = self.anim
        assert anim is not None
        source = anim.frame
        at = len(anim.frames) if index is None else max(0, min(int(index), len(anim.frames)))
        frame = Frame(duration_ms=source.duration_ms)
        cels: dict[int, Layer] = {}
        if link or copy:
            for track in anim.tracks:
                cel = anim.cels.get((track.uid, source.uid))
                if cel is not None:
                    cels[track.uid] = cel if link else cel.copy(name=cel.name)
        # The playhead moves *before* the insert, so ``_put_frame``'s own
        # ``_anim_changed`` materialises the new frame -- rather than after it,
        # which meant rebuilding the whole view twice for one click.
        anim.current = at
        self._put_frame(at, frame, cels)
        # ``pinned`` only when this op allocated pixels of its own. A linked
        # duplicate holds objects the grid is keeping alive anyway, and charging
        # for them again would evict real history to make room for a number that
        # describes no memory.
        edits.append(FrameAddEdit(at, frame, cels, pinned=copy))
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        return frame

    def remove_frame(self, index: int | None = None) -> bool:
        anim = self.anim
        if anim is None or len(anim.frames) <= 1:
            return False
        self.commit_floating()
        index = anim.current if index is None else max(0, min(int(index), len(anim.frames) - 1))
        frame = anim.frames[index]
        cels = {
            track.uid: anim.cels[(track.uid, frame.uid)]
            for track in anim.tracks
            if (track.uid, frame.uid) in anim.cels
        }
        self._drop_frame(frame)
        self.history.push(
            FrameRemoveEdit(index, frame, cels, pinned=self._released(cels.values()))
        )
        return True

    def move_frame(self, index: int, to: int) -> bool:
        anim = self.anim
        if anim is None:
            return False
        to = max(0, min(int(to), len(anim.frames) - 1))
        if to == index or not 0 <= index < len(anim.frames):
            return False
        self.commit_floating()
        uid = anim.frames[index].uid
        self._move_frame(uid, to)
        self.history.push(FrameMoveEdit(uid, index, to))
        return True

    def set_frame_duration(self, index: int, ms: int) -> bool:
        anim = self.anim
        if anim is None or not 0 <= index < len(anim.frames):
            return False
        frame = anim.frames[index]
        before, after = frame.duration_ms, clamp_duration(ms)
        if before == after:
            return False
        self._set_duration(frame.uid, after)
        self.history.push(FrameDurationEdit(frame.uid, before, after))
        return True

    # -- cels ---------------------------------------------------------------

    def _slot(self, track_index: int | None, frame_index: int | None) -> tuple[Track, Frame] | None:
        anim = self.anim
        if anim is None:
            return None
        ti = self.stack.active_index if track_index is None else track_index
        fi = anim.current if frame_index is None else frame_index
        if not (0 <= ti < len(anim.tracks) and 0 <= fi < len(anim.frames)):
            return None
        return anim.tracks[ti], anim.frames[fi]

    def clear_cel(self, track_index: int | None = None, frame_index: int | None = None) -> bool:
        slot = self._slot(track_index, frame_index)
        if slot is None:
            return False
        track, frame = slot
        assert self.anim is not None
        before = self.anim.cels.get((track.uid, frame.uid))
        if before is None:
            return False
        self.commit_floating()
        self._set_cel(track.uid, frame.uid, None)
        # Clearing one slot of a linked cel releases nothing: the object is
        # alive in the frames it is still linked into.
        self.history.push(
            CelSetEdit(
                track.uid, frame.uid, before, None, pinned=bool(self._released([before]))
            )
        )
        return True

    def link_cel(
        self, source_frame: int, track_index: int | None = None, frame_index: int | None = None
    ) -> bool:
        """Point a slot at another frame's cel, so the two share one object."""
        slot = self._slot(track_index, frame_index)
        anim = self.anim
        if slot is None or anim is None or not 0 <= source_frame < len(anim.frames):
            return False
        track, frame = slot
        source = anim.cels.get((track.uid, anim.frames[source_frame].uid))
        before = anim.cels.get((track.uid, frame.uid))
        if source is None or source is before:
            return False
        self.commit_floating()
        self._set_cel(track.uid, frame.uid, source)
        # ``pinned=False``: the shared object is alive in the frame it came
        # from whatever the history does with this step.
        self.history.push(CelSetEdit(track.uid, frame.uid, before, source, pinned=False))
        return True

    def unlink_cel(self, track_index: int | None = None, frame_index: int | None = None) -> bool:
        """Give a linked slot a private copy, so editing it stops editing the
        other frames.

        The copy -- and therefore its uid -- is made **once, here**, and the
        edit holds it. A redo that copied again would mint a new identity and
        strand every patch recorded against the first one, which is the same
        rule ``flatten_layers`` follows when it draws its uid up front.
        """
        slot = self._slot(track_index, frame_index)
        anim = self.anim
        if slot is None or anim is None:
            return False
        track, frame = slot
        if not anim.is_linked(track.uid, frame.uid):
            return False
        before = anim.cels[(track.uid, frame.uid)]
        self.commit_floating()
        copy = before.copy(name=before.name)
        self._set_cel(track.uid, frame.uid, copy)
        self.history.push(CelSetEdit(track.uid, frame.uid, before, copy, pinned=True))
        return True

    # -- tags ---------------------------------------------------------------

    def _set_tags(self, tags: list[Tag]) -> None:
        """Undo hook for every tag op. Installs *copies*.

        The edit holds the lists it was given and undo/redo may run any number
        of times, so handing the document the same ``Tag`` objects would let the
        next rename write through into the step that is meant to reverse it.
        """
        anim = self.anim
        if anim is None:
            return
        anim.tags = [replace(tag) for tag in tags]
        self.rev += 1

    def _clamped_tag(self, tag: Tag) -> Tag:
        """A tag with its span inside the timeline and its ends the right way up."""
        anim = self.anim
        assert anim is not None
        last = len(anim.frames) - 1
        start = max(0, min(int(tag.start), last))
        end = max(0, min(int(tag.end), last))
        return replace(
            tag,
            name=(tag.name.strip() or "tag"),
            start=min(start, end),
            end=max(start, end),
            loop=bool(tag.loop),
        )

    def _push_tags(self, tags: list[Tag]) -> bool:
        """Install a new tag list as one undo step, or nothing if it is the same.

        The no-op check is the rule every other op here follows: dirty is a
        comparison against ``history.head``, so a step that changes nothing
        makes a saved document ask to be saved. ``Tag`` is a plain dataclass, so
        ``==`` is the value comparison this wants.
        """
        anim = self.anim
        if anim is None:
            return False
        before = [replace(tag) for tag in anim.tags]
        after = [self._clamped_tag(tag) for tag in tags]
        if before == after:
            return False
        self._set_tags(after)
        self.history.push(TagsEdit(before, after))
        return True

    def add_tag(
        self, name: str, start: int, end: int | None = None, *, loop: bool = True
    ) -> bool:
        """A named span of frames. Overlaps are allowed -- ``active_tag`` picks
        the innermost, which is what makes a short "hit" inside a long "combat"
        the useful arrangement rather than an ambiguous one."""
        if self.anim is None:
            return False
        span = Tag(name=name, start=start, end=start if end is None else end, loop=loop)
        return self._push_tags([*self.anim.tags, span])

    def remove_tag(self, index: int) -> bool:
        anim = self.anim
        if anim is None or not 0 <= index < len(anim.tags):
            return False
        return self._push_tags([t for i, t in enumerate(anim.tags) if i != index])

    def set_tag(self, index: int, **props: Any) -> bool:
        """Rename, retime or re-loop one tag. Unnamed keys are left alone."""
        anim = self.anim
        if anim is None or not 0 <= index < len(anim.tags):
            return False
        edited = replace(anim.tags[index], **props)
        return self._push_tags(
            [edited if i == index else t for i, t in enumerate(anim.tags)]
        )

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
        real = Layer(
            pixels=cp.empty(width, height),
            name=track.name,
            opacity=track.opacity,
            visible=track.visible,
            blend=track.blend,
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
        """Push one undo step for a region already written, and recomposite.

        On an animated document the step may be two things that must undo
        together: the cel this write brought into existence, and the write
        itself. A no-op write takes the cel back out again and pushes nothing,
        which is the same rule the selection ops follow -- a step that changes
        nothing must not move the history head, or the document asks to be saved
        for a gesture that did not happen.
        """
        after = layer.pixels[rect[1] : rect[3], rect[0] : rect[2]].copy()
        if np.array_equal(before, after):
            self._discard_pending_cel()
            return
        pending, self._pending_cels = self._pending_cels, []
        edit: Any = PatchEdit(layer.uid, rect, before, after)
        self.history.push(edit if not pending else CompoundEdit([*pending, edit]))
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
        self._ensure_active_cel()
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
        self._ensure_active_cel()
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
            # A brush-down with no dab. ``begin_stroke`` may have autovivified a
            # cel for it, and nothing below will reach ``_commit_patch`` to
            # decide the cel's fate, so it is decided here.
            self._discard_pending_cel()
            return False
        box = self.clip(stroke.dirty)
        if box is None:
            self._discard_pending_cel()
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
        self._ensure_active_cel()
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
        self,
        layers: list[Layer],
        size: tuple[int, int],
        active: int,
        grid: dict[str, Any] | None = None,
    ) -> None:
        """Undo hook for a whole-canvas operation.

        The layers are copied (the snapshot is held for as long as the edit is,
        and a later stroke must not write into it) but they keep their uids: an
        undo restores the document's *state*, not a set of new layers, and
        every patch recorded before this one addresses those uids.

        ``grid`` is the animated form of the same argument one level up. The
        copies are made once and shared back into the slots by index, so two
        frames that held one object hold one object again.
        """
        copies = [layer.copy(uid=layer.uid) for layer in layers]
        width, height = size
        if grid is None or self.anim is None:
            self.stack = LayerStack(copies, active)
        else:
            anim = self.anim
            anim.frames = list(grid["frames"])
            anim.tracks = list(grid["tracks"])
            anim.cels = {key: copies[i] for key, i in grid["slots"].items()}
            anim.current = max(0, min(grid["current"], len(anim.frames) - 1))
            anim._blank = None
            self.stack = LayerStack(anim.layers_for(anim.frame, size), active)
        # After the grid is back, not before: the frames being stamped are the
        # restored ones, and a frame the snapshot brings back would otherwise
        # keep a cached flatten from before it was removed.
        self._stamp_all()
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
        self._ensure_active_cel()
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
        edit: Any = PatchEdit(layer.uid, box, before, after)
        # The buffer names the step that is actually *on the stack*, because
        # ``cancel_floating`` revokes it by identity. Wrapping the patch in a
        # compound and then naming the bare patch would leave the revoke
        # searching for something the stack does not hold -- the alpha-cut would
        # stay and the lifted pixels would be dropped.
        pending, self._pending_cels = self._pending_cels, []
        if pending:
            edit = CompoundEdit([*pending, edit])
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
        # Keyed by the buffer's own layer, not the active one: a paste chooses
        # its target when it is made, and the user may have selected another row
        # while it floated.
        self._ensure_cel_for(floating.layer_uid)
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
        self._ensure_active_cel()
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
        if self.anim is not None:
            # A track carrying one cel, on the current frame only: the pasted
            # pixels are a thing that happens at a moment, not a thing that is
            # true for the whole clip.
            index = self.stack.active_index + 1
            track = Track(name=layer.name)
            cels = {self.anim.frame.uid: layer}
            self._put_track(index, track, cels)
            self.history.push(TrackAddEdit(index, track, cels, pinned=True))
            return True
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
        """A new empty layer, or on an animated document a new empty *track*.

        The five layer ops below each grow one of these branches, and they all
        take the same shape: a track index and a stack index are the same
        number, because ``Animation.tracks`` mirrors ``LayerStack`` bottom-first
        on purpose. That is what lets the layers panel go on calling these
        methods with the indices it already has.

        A new track gets no cels at all rather than one empty cel per frame. The
        grid is sparse, "empty everywhere" is the absence of keys, and the first
        stroke on any frame autovivifies exactly the one cel it needs.
        """
        self.commit_floating()
        if self.anim is not None:
            index = self.stack.active_index + 1
            track = Track(name=name or f"Layer {len(self.anim.tracks) + 1}")
            self._put_track(index, track, {})
            self.history.push(TrackAddEdit(index, track, {}, pinned=False))
            return self.stack[self.stack.active_index]
        width, height = self.size
        layer = Layer.empty(width, height, name or f"Layer {len(self.stack) + 1}")
        index = self.stack.insert(self.stack.active_index + 1, layer)
        self.history.push(LayerAddEdit(index, layer))
        self.invalidate_all()
        return layer

    def duplicate_layer(self, index: int | None = None) -> Layer:
        self.commit_floating()
        index = self.stack.active_index if index is None else index
        if self.anim is not None:
            track = self.anim.tracks[index]
            copy_track = Track(
                name=f"{track.name} copy",
                opacity=track.opacity,
                visible=track.visible,
                blend=track.blend,
            )
            # Copies, not links -- *from* the original: duplicating a layer to
            # paint a variation on it and having every stroke land on the
            # original too would be the opposite of what the button says.
            #
            # But one copy per distinct cel, not one per slot. A cel linked
            # across three frames is one object, and copying it three times
            # gives the duplicate three independent frames where the original
            # has one -- the link silently gone, and three times the memory.
            # This is ``unique_cel_layers``'s rule applied to a single row.
            copies: dict[int, Layer] = {}
            cels: dict[int, Layer] = {}
            for frame in self.anim.frames:
                cel = self.anim.cels.get((track.uid, frame.uid))
                if cel is None:
                    continue
                copy = copies.get(id(cel))
                if copy is None:
                    copy = cel.copy(name=copy_track.name)
                    copies[id(cel)] = copy
                cels[frame.uid] = copy
            self._put_track(index + 1, copy_track, cels)
            self.history.push(TrackAddEdit(index + 1, copy_track, cels, pinned=True))
            return self.stack[self.stack.active_index]
        copy = self.stack.duplicate(index)
        self.history.push(LayerAddEdit(index + 1, copy))
        self.invalidate_all()
        return copy

    def remove_layer(self, index: int | None = None) -> bool:
        if len(self.stack) == 1:
            return False
        self.commit_floating()
        index = self.stack.active_index if index is None else index
        if self.anim is not None:
            track = self.anim.tracks[index]
            cels = {
                frame.uid: cel
                for frame in self.anim.frames
                if (cel := self.anim.cels.get((track.uid, frame.uid))) is not None
            }
            self._drop_track(track)
            self.history.push(
                TrackRemoveEdit(index, track, cels, pinned=self._released(cels.values()))
            )
            return True
        gone = self.stack.remove(index)
        self.history.push(LayerRemoveEdit(index, gone))
        self.invalidate_all()
        return True

    def move_layer(self, index: int, to: int) -> bool:
        to = max(0, min(int(to), len(self.stack) - 1))
        if to == index:
            return False
        self.commit_floating()
        if self.anim is not None:
            uid = self.anim.tracks[index].uid
            self._move_track(uid, to)
            self.history.push(TrackMoveEdit(uid, index, to))
            return True
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
        # The track when there is one: it is authoritative, so writing the
        # property onto the materialised layer instead would last exactly until
        # the next time that frame was rebuilt.
        target = self.anim.tracks[index] if self.anim is not None else self.stack[index]
        source = {} if was is None else was
        before = {key: source.get(key, getattr(target, key)) for key in props}
        if before == props:
            return False
        for key, value in props.items():
            setattr(target, key, value)
        if self.anim is not None:
            self.history.push(TrackPropsEdit(target.uid, before, dict(props)))
            self._anim_changed()
            return True
        self.history.push(LayerPropsEdit(target.uid, before, dict(props)))
        self.invalidate_all()
        return True

    @property
    def can_restructure(self) -> bool:
        """Whether merge-down and flatten are available.

        They are not, on an animated document. Both are defined over *one*
        stack, and an animated document has one per frame -- so the honest
        versions are "merge these two tracks across every frame", which has to
        decide what a merge of a linked cel with an unlinked one even means, and
        "flatten this frame", which throws away every other frame's cels. Both
        are real features and neither is v1. Refusing is the answer that cannot
        silently destroy an animation; the layers panel reads this to disable
        the buttons rather than letting the user find out by pressing them.
        """
        return self.anim is None

    def merge_down(self, index: int | None = None) -> bool:
        """Flatten a layer into the one beneath it, honouring its blend mode."""
        if not self.can_restructure:
            return False
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
        if len(self.stack) == 1 or not self.can_restructure:
            return
        self.commit_floating()
        # Replay must be a pure function of the document, and minting a uid is
        # the one part of this op that is not: a redo would produce a layer
        # with a new identity, stranding every patch recorded above it. The uid
        # is drawn once and closed over, so every replay lands on the same one.
        uid = new_uid()
        self._replay(lambda: self._do_flatten(uid))

    def apply_matte(self, alpha: np.ndarray) -> bool:
        """Fold a cutout into the document's alpha, as one undoable step.

        ``alpha`` is a canvas-sized uint8 plane: 255 keeps a pixel, 0 cuts it.
        It is multiplied into *every* layer -- every cel of every frame, on an
        animated document, since a cutout describes the drawing rather than a
        moment of it -- rather than into the composite,
        because the composite is a cache and there is nowhere else for it to
        live -- and with the binary matte this is given, multiplying each
        layer's alpha and multiplying the normal-blended result are the same
        arithmetic. So a layered drawing keeps its layers, and the user goes on
        editing the cutout with the eraser and the brush, which is the whole
        point of handing them a matte here rather than a finished PNG.

        One ``CompoundEdit``, so a single Ctrl+Z reverses the whole cutout
        rather than one layer of it. ``matte`` is set to None outside the edit
        on purpose: it is a property of the document (what a flatten puts
        behind transparency), not of a region of pixels, and leaving it as the
        opaque white ``matte_for`` gives an opaque photo would flatten every
        cut pixel straight back to white on save -- which is exactly the file
        this edit exists to avoid writing.
        """
        width, height = self.size
        if alpha.shape[:2] != (height, width):
            return False
        self.commit_floating()
        rect = (0, 0, width, height)
        edits: list[Any] = []
        weight = alpha.astype(np.float32) / 255.0
        # Every distinct cel in the *whole grid*, not the current frame's stack.
        # A cutout is a statement about the drawing, and mattes applied to one
        # frame of a clip while ``_stamp_all`` announced that all of them had
        # changed is the worst of both -- the other frames keep their un-matted
        # alpha and every cached flatten is thrown away saying otherwise.
        # ``unique_cel_layers`` because a linked cel must be multiplied once:
        # twice would square the matte along its soft edge.
        targets = (
            list(self.stack) if self.anim is None else list(self.anim.unique_cel_layers())
        )
        for layer in targets:
            before = layer.pixels.copy()
            layer.pixels[:, :, 3] = np.clip(
                layer.pixels[:, :, 3].astype(np.float32) * weight + 0.5, 0, 255
            ).astype(np.uint8)
            if np.array_equal(before, layer.pixels):
                continue
            edits.append(PatchEdit(layer.uid, rect, before, layer.pixels.copy()))
        if not edits:
            return False
        self.history.push(CompoundEdit(edits))
        self.matte = None
        # Every cel in the grid was written -- so this is one of the few places
        # that has to say so, ``invalidate_all`` having stopped guessing.
        self._stamp_all()
        self.invalidate_all()
        return True

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
