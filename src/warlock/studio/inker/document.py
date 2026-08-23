"""The document: a layer stack, a history, a selection, and a cached composite.

This is the only module in the package that is allowed to know about all the
others. ``Document`` is the only *class* that pushes anything onto the undo
stack -- the ten ``_doc_*`` mixins push at sixty sites and are part of it, which
is what the sentence here used to claim about the module and had not been true
since they were split out. Every entry point follows the same three steps --
copy the region that is about to change, change it, push a ``PatchEdit`` --
which is what makes "one gesture is one Ctrl+Z" a property of the model rather
than a rule the UI has to remember.

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
from . import index_plane as ixp
from . import indexed as ix
from ._doc_anim import AnimOps
from ._doc_geometry import GeometryOps
from ._doc_history import HistoryOps
from ._doc_indexed import IndexedOps
from ._doc_layers import LayerOps
from ._doc_paint import (
    PATH_SHAPES,
    SHAPES,
    PaintOps,
    catmull_rom,
    curve_points,
    curve_spans,
    normalise_rect,
)
from ._doc_ranges import RangeOps
from ._doc_selection import SelectionOps
from ._doc_slices import SliceOps
from ._doc_tiles import TileOps
from .anim_edits import CelSetEdit
from .animation import CEL_PROPS, Animation
from .brush import StrokeState
from .layers import Layer, LayerStack
from .selection import Clipboard, FloatingBuffer, SelectionMask
from .slices import Slice
from .tiles import TilemapCel, grid_shape
from .undo import (
    UNDO_BYTES,
    CompoundEdit,
    IndexPatchEdit,
    MatteEdit,
    PatchEdit,
    ReplayEdit,
    UndoStack,
    one_step,
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

#: ``_map_planes``' default for ``mask_fn``: "whatever the pixels are getting".
#: A sentinel rather than ``None`` because ``None`` is a meaningful answer there
#: -- it is how a colour map says *leave the selection alone* -- and the two
#: have to be tellable apart.
_SAME_AS_PIXELS = object()

__all__ = [
    "FRAME_CACHE_BYTES",
    "Document",
    "RGBA",
    "TRANSPARENT",
    "OPAQUE_WHITE",
    "SHAPES",
    "normalise_rect",
    "PATH_SHAPES",
    "catmull_rom",
    "curve_points",
    "curve_spans",
]


def matte_for(pixels: np.ndarray) -> RGBA | None:
    """What a flattened export puts behind transparency, decided once at load.

    The old editor asked this per eraser stroke and called it ``erase_color``.
    Now the eraser always cuts alpha -- which is what an eraser means in a
    layered document -- and the question moved to where it belongs: a photo
    opened here is still a photo when it is saved, so it flattens onto white; a
    sprite has transparency the user is thinking about, so it keeps it.
    """
    # ``ndim`` before ``shape[2]``: the guard was the crash it was guarding
    # against, indexing the third axis of an array that may not have one. Every
    # caller passes a composite today, which is why this was latent rather than
    # a bug -- but "decided once at load" means the callers are reader code,
    # and a reader handed a 2-D plane should get "no matte", not an IndexError.
    if pixels.ndim < 3 or pixels.shape[2] < 4:
        return None
    if int(pixels[..., 3].min()) < 255:
        return None
    return OPAQUE_WHITE


@dataclass
class Document(
    AnimOps, PaintOps, HistoryOps, SelectionOps, LayerOps, GeometryOps, IndexedOps,
    SliceOps,
    TileOps,
    RangeOps,
):
    stack: LayerStack
    matte: RGBA | None = None
    rev: int = 0
    path: Path | None = None
    file_format: str = "png"
    #: Canvas resolution in pixels per inch, ``(x, y)``, or None when the file
    #: did not say. Carried rather than used: nothing in this editor renders at
    #: a physical size, but ORA stores ``xres``/``yres`` on the image element
    #: and dropping them meant a 300-DPI Krita document came back at Krita's
    #: default on every round trip -- pixels intact, physical size quietly
    #: gone. Construction-time state like ``matte`` and ``file_format``: no
    #: undo op edits it, so it needs no snapshot.
    dpi: tuple[int, int] | None = None
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
    #: ``"rgb"`` | ``"indexed"`` | ``"grayscale"``. The document's colour mode,
    #: and the branch every write takes at :meth:`_commit_patch` -- one funnel,
    #: three constraints (snap / index-resolve / luma).
    #:
    #: ``"rgb"`` is today's document bit for bit, **including** the
    #: constrain-on-write behaviour above when ``palette`` is set. That
    #: combination has a name now -- *palette-constrained RGB* -- and it is kept
    #: rather than replaced: documents saved before true indexing existed
    #: legally contain soft alpha, which one index per pixel cannot represent,
    #: so they must open exactly as they always did and entering indexed mode
    #: must be something the user asks for and can undo.
    #:
    #: ``"indexed"`` requires a palette of 1..256 entries and makes every
    #: layer's ``indices`` plane the record. ``"grayscale"`` is a constraint
    #: rather than a storage change -- see :meth:`_commit_patch`.
    color_mode: str = "rgb"
    #: The palette slot that materialises as ``(0, 0, 0, 0)``. Meaningful only
    #: in indexed mode, where it is the *only* way a pixel becomes a hole: it is
    #: never a nearest-match candidate, so a black the user painted cannot
    #: collapse into the black that means transparent.
    transparent_index: int = 0
    #: Named rectangles on the canvas -- pivots, hitboxes, nine-slice panels.
    #: On the *document* rather than on the grid because a still drawing has
    #: them too (a nine-slice button is one PNG), and the per-frame overrides
    #: live inside each slice. See :mod:`.slices`.
    slices: list[Slice] = field(default_factory=list)
    #: The layer-group tree, kept *beside* the stack rather than replacing it.
    #: ``groups`` is uid -> node; ``group_of`` is member uid -> the group it is
    #: in, with an absent key meaning the root. Members are stack rows (by track
    #: uid on an animated document -- see :meth:`member_uids`) and other groups.
    #: Empty on every document until somebody makes a group, which is what keeps
    #: the whole feature out of the ordinary composite path. See
    #: :mod:`.groups`.
    groups: dict[int, Any] = field(default_factory=dict)
    group_of: dict[int, int] = field(default_factory=dict)
    #: Every tileset this document owns, in insertion order. Document-level
    #: rather than per-track because tilesets are shared -- two tracks may
    #: bind the same slot -- and ``Track.tileset_uid`` is what names one of
    #: these. Typed via ``tiles.TilesetSlot`` at runtime; kept as ``list[Any]``
    #: here, and looked up by hand in ``_ensure_cel_for``, so this module does
    #: not have to import ``tiles`` (which reaches for the shared ``tilegrid``
    #: leaf) just to spell the field's type. See :mod:`.tiles`.
    tilesets: list[Any] = field(default_factory=list)
    #: Manual / Auto / Stack -- how a pixel edit on a tilemap cel is routed
    #: back onto its tileset (Wave 3 chunk 3.3). **View state**, INVARIANTS'
    #: Wave 3 entry 4: a toolbar toggle passed into a document call per
    #: gesture must not be able to dirty a document, so this is never
    #: serialized and never undoable -- ``repr=False`` for ``paint_slot``'s
    #: reason, it is not part of what the document *is*.
    tile_behavior: str = field(default="manual", repr=False)
    #: tileset uid -> (the strip image the index was built from, content_key ->
    #: local id). A derived cache, never serialized, keyed on the *identity* of
    #: the strip -- a tileset is edited by frozen-replace, so a changed atlas is
    #: always a changed array. See
    #: :meth:`~._doc_tiles.TileOps._tile_hash_index`.
    _tile_hashes: dict[int, tuple[np.ndarray, dict[bytes, int]]] = field(
        init=False, default_factory=dict, repr=False
    )

    _composite: np.ndarray = field(init=False, repr=False)
    #: Per-frame change counters, keyed by frame uid, for the flatten cache.
    #: A plain dict rather than a field on ``Frame`` so a frame carries only
    #: what a *save* carries -- a stamp is cache bookkeeping and has no
    #: business round-tripping through a file.
    _frame_stamps: dict[int, int] = field(default_factory=dict, repr=False, init=False)
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
    #: The mask ``deselect`` took away, for :meth:`reselect`. A plain field and
    #: deliberately so: it is neither persisted nor replayed nor undoable, in
    #: the way the playhead and the active layer are not. Reselect is a
    #: *convenience* -- it re-runs an ordinary ``select`` and pushes an ordinary
    #: step -- so making the memory itself part of the document's state would
    #: mean an undo could change what Ctrl+Shift+D is about to give you.
    _last_mask: np.ndarray | None = field(init=False, default=None, repr=False)
    #: The palette slot the foreground colour was picked from, or None when it
    #: came from the wheel, the eyedropper or anywhere else. Read only by
    #: :meth:`_commit_indexed_patch`, where it becomes ``index_plane.resolve``'s
    #: ``prefer``: pixels the user painted in *that colour* take *that slot*
    #: rather than the lowest-numbered duplicate of it.
    #:
    #: A field the studio sets when the swatch changes, rather than a parameter
    #: on ``begin_stroke``, because every tool needs it and not only the brush
    #: -- a fill, a shape, a gradient stop and a text stamp all paint in the
    #: foreground colour and all deserve to land on the slot the user clicked.
    #: One setter is also one thing to get wrong; twelve signatures are twelve.
    #:
    #: Not history, not persisted, not undoable -- ``_last_mask``'s rule. It
    #: describes what the user is holding, not what the document contains, and
    #: an undo that changed which swatch is selected would be answering a
    #: question nobody asked.
    paint_slot: int | None = field(init=False, default=None, repr=False)
    #: Cels autovivified by writes that have not yet been committed. They ride
    #: along into the same ``CompoundEdit`` as the patch, so drawing on an empty
    #: frame is one Ctrl+Z rather than two. A *list* rather than one slot: the
    #: convention is that every write commits before the next one autovivifies,
    #: and while that holds there is never more than one -- but the cost of the
    #: convention being broken was a second write landing on the shared
    #: read-only placeholder plane and raising out of the middle of a stroke,
    #: which is not a failure mode worth keeping in exchange for a scalar.
    _pending_cels: list[Any] = field(init=False, default_factory=list, repr=False)
    #: ``(frame uid, track uid or None)`` -> (stamp it was built at, pixels),
    #: least-recently-used first in ``_frame_order``. See :meth:`frame_flat`.
    #: The track is *in the key* because a current-layer-only onion skin asks
    #: for a filtered flatten of a frame the unfiltered draw also asks for, and
    #: one key for both would hand each of them the other's pixels.
    _frame_cache: dict[tuple[int, int | None], tuple[int, np.ndarray]] = field(
        default_factory=dict, repr=False, init=False
    )
    _frame_order: list[tuple[int, int | None]] = field(
        default_factory=list, repr=False, init=False
    )
    #: An open layer-move session: the active layer's pixels as they were when
    #: it opened, and the **uid** of the layer they came off. ``_filter``'s
    #: shape and for the same reasons -- a preview re-renders from the snapshot
    #: rather than from the last preview (a drag would compound otherwise), and
    #: the uid rather than "the active layer" because a session lives across
    #: frames. See ``_doc_paint.begin_layer_move``.
    _move: tuple[np.ndarray, int] | None = field(init=False, default=None, repr=False)
    #: Frames that have gone, for whoever is holding a *texture* keyed on one.
    #: Plain ints and a drain, so the document goes on knowing nothing about GL:
    #: see ``panes/inker_textures.release_dropped``.
    _dropped_frames: list[int] = field(default_factory=list, repr=False, init=False)
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
    _layer_stamps: dict[int, int] = field(default_factory=dict, repr=False, init=False)
    #: An open palette-conversion session: ``(layer uid, pixels as they were)``
    #: for every real cel on the frame the popup was opened over. The filter
    #: session's shape one dimension wider -- a conversion is whole-*document*,
    #: so there is no rectangle and there is more than one layer -- and it
    #: addresses each of them by uid for the same reason ``_filter`` does. See
    #: :meth:`~._doc_paint.PaintOps.begin_convert`.
    _convert: list[tuple[int, np.ndarray]] | None = field(
        init=False, default=None, repr=False
    )
    #: The last ``(table, method) -> converted planes`` a preview computed, for
    #: the life of one conversion session. Floyd-Steinberg is a Python loop over
    #: every pixel, so this is not an optimisation but the difference between a
    #: live preview and none.
    _convert_memo: tuple[tuple[Any, ...], list[np.ndarray]] | None = field(
        init=False, default=None, repr=False
    )

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
        """The whole document as RGBA.

        ``matte`` is the *stand-in* background (divergence #6) and it is only
        consulted when there is no real background layer (6.5): a document with
        one already has an opaque bottom, and compositing a second colour under
        it would be a colour nothing on screen or in any export could show.
        """
        under = self.matte if (matte and not self.has_background) else None
        return cp.flatten_onto(self._composite.copy(), under)

    @property
    def has_background(self) -> bool:
        """Whether the bottom layer is a real background layer."""

        return bool(len(self.stack) and getattr(self.stack[0], "background", False))

    def png_bytes(self, *, scale: int = 1) -> bytes:
        """The flattened document as a PNG. Blocking; callers go off-thread.

        ``scale`` is a whole-number nearest-neighbour magnification, applied
        after the flatten -- which is the only order that is exact: magnifying
        each layer first and compositing the results would blend at the
        magnified resolution and put half-covered pixels along every block
        edge. The default is 1 and takes the untouched path, so every existing
        caller writes the bytes it always did.
        """
        import io

        from PIL import Image

        from .transform import upscale

        buf = io.BytesIO()
        Image.fromarray(upscale(self.flatten(), scale), "RGBA").save(buf, "PNG")
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
        # The group fold is refreshed *here* rather than at each op, because
        # this is what every structural change already ends with -- and every
        # path that replaces ``self.stack`` (``_materialize_frame``,
        # ``restore_snapshot``, ``_map_planes``) is one of them. A document with
        # no groups pays one truthiness test.
        self.stack.group_fold = self.group_fold()
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

    # -- the group tree -----------------------------------------------------

    def member_uids(self) -> list[int]:
        """What the group tree knows each stack row by, bottom-first.

        **The placeholder trap, and it is the reason this is a method rather
        than a comprehension at three call sites.** On an animated document a
        materialised *empty* slot is a placeholder ``Layer`` carrying a
        per-slot uid of its own, not the track's -- so keying group membership
        on ``layer.uid`` would silently un-group every empty cel: hiding a group
        would leave its empty rows ungrouped (invisible either way, so nothing
        looks wrong) right up until somebody drew on one, at which point a
        drawing would appear inside a hidden group.

        Membership is a property of the *track*, exactly as name, opacity,
        visibility, blend and the two locks are, so the track's uid is what the
        tree is keyed on and what this returns.
        """
        if self.anim is None:
            return [layer.uid for layer in self.stack]
        return [track.uid for track in self.anim.tracks]

    def group_fold(self) -> list[tuple[bool, float]] | None:
        """``(visible, opacity)`` per stack row from the groups above it.

        None -- not a list of neutral pairs -- on a document with no groups, so
        ``LayerStack._entries`` can take its original path by identity rather
        than multiplying by 1.0 once per layer per composite.
        """
        if not self.groups:
            return None
        from . import groups as gp

        return [
            gp.resolve(self.groups, self.group_of, uid)[:2] for uid in self.member_uids()
        ]

    def member_uid_of(self, layer: Any) -> int:
        """The tree's name for a layer that is in the current stack.

        Falls back to the layer's own uid for anything the stack does not hold
        -- an off-frame cel reached through ``layer_by_uid``, say. Such a layer
        is not being composited or written to through a door, so the answer is
        only ever "no group", which is the honest one.
        """
        if self.anim is None:
            return layer.uid
        try:
            index = self.stack.index_of(layer.uid)
        except KeyError:
            return layer.uid
        tracks = self.anim.tracks
        return tracks[index].uid if index < len(tracks) else layer.uid

    def restore_groups(self, state: Any) -> None:
        """Undo hook for a whole-canvas op that rewrote the tree.

        Only ``flatten_layers`` does -- it replaces every layer in the document
        with one, so there is nothing left for a group to hold -- but the
        snapshot is taken unconditionally by ``_replay`` for the reason every
        snapshot in this file is unconditional: a rotate that quietly kept a
        tree describing layers it had just replaced is a much harder bug than a
        dictionary copy is a cost.
        """
        if state is None:
            return
        nodes, membership = state
        self.groups = dict(nodes)
        self.group_of = dict(membership)

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

    def frame_flat(
        self, frame_uid: int, *, track_uid: int | None = None
    ) -> np.ndarray | None:
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

        ``track_uid`` restricts the flatten to one track, for the current-layer
        -only onion skin. It is part of the cache key rather than a bypass,
        because this is asked for per drawn frame and must stay a lookup; the
        *stamp* stays keyed on the frame alone, so an edit on another track
        recomposites the filtered entry needlessly. That is conservative and
        correct, and a per-track stamp would be a second answer to drift from.
        """
        anim = self.anim
        if anim is None:
            return None
        try:
            frame = anim.frames[anim.frame_index(frame_uid)]
        except KeyError:
            return None
        if track_uid is not None and all(
            track.uid != track_uid for track in anim.tracks
        ):
            # A stale active index must not raise out of the middle of a draw.
            return None
        key = (frame_uid, track_uid)
        stamp = self.frame_stamp(frame_uid)
        hit = self._frame_cache.get(key)
        if hit is not None and hit[0] == stamp:
            self._frame_order.remove(key)
            self._frame_order.append(key)
            return hit[1]
        stack = (
            self.frame_stack(frame)
            if track_uid is None
            else self.frame_stack(frame, track_uids={track_uid})
        )
        flat = stack.flatten()
        self._frame_cache[key] = (stamp, flat)
        if key in self._frame_order:
            self._frame_order.remove(key)
        self._frame_order.append(key)
        self._evict_frames()
        return flat

    def frame_stack(
        self, frame: Any, *, track_uids: set[int] | None = None
    ) -> LayerStack:
        """One frame's layers as an ordinary stack, with the group fold on it.

        Every place that flattens a frame that is *not* the current one goes
        through here -- the onion-skin and playback cache, the animated
        flatten, the ORA writer's ``mergedimage`` -- because each of them built
        its own bare ``LayerStack`` and each would otherwise show a hidden group
        as visible in exactly one of onion skinning, playback and the saved
        file.

        ``track_uids`` keeps only the named tracks, for a split-by-layer export.
        ``None`` is the whole frame and is the call this always was, down to the
        objects handed to ``LayerStack``.

        **The fold is filtered with the rows**, not carried over whole: it is a
        list *parallel to the stack* (``group_fold`` walks ``member_uids()``,
        which is one entry per track), so an unfiltered fold on a filtered stack
        would hang each surviving row's inherited visibility on whichever row
        happened to land at its index. Filtering it is enough to be *correct*
        rather than merely convenient, because group compositing here is
        pass-through: a group contributes nothing but a ``(visible, opacity)``
        pair per leaf, so the leaves that survive inherit exactly what they
        inherited in the full stack. If groups ever composite in isolation --
        the v1 gap ``inker/groups.py`` names -- a subset that cut a group in
        half would stop being expressible this way, which is why
        :func:`sheetout.layer_splits` never cuts one.
        """
        layers = self.anim.layers_for(frame, self.size)
        fold = self.group_fold()
        if track_uids is not None:
            keep = [
                index
                for index, track in enumerate(self.anim.tracks)
                if track.uid in track_uids
            ]
            layers = [layers[index] for index in keep]
            fold = None if fold is None else [fold[index] for index in keep]
        stack = LayerStack(layers, 0)
        stack.group_fold = fold
        return stack

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
        # *Every* key on this frame, not one: the filtered flattens are
        # separate entries under the same frame uid.
        for key in [k for k in self._frame_cache if k[0] == frame_uid]:
            self._frame_cache.pop(key, None)
        self._frame_order = [k for k in self._frame_order if k[0] != frame_uid]
        self._frame_stamps.pop(frame_uid, None)
        self._dropped_frames.append(frame_uid)

    def _forget_all_frames(self) -> None:
        for uid in set(self._frame_stamps) | {k[0] for k in self._frame_cache}:
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
        if track.tileset_uid is not None:
            # A tilemap track autovivifies a ``TilemapCel``, not a ``Layer``:
            # ``refs`` all-zero (every cell empty/blank) rather than anything
            # copied forward -- ``continuous`` describes what a *raster*
            # track's fresh cel starts from and a tilemap track has no
            # matching notion yet (Wave 3 chunk 3.3 is where tile edits are
            # routed). The tile size comes off the bound slot -- and a track
            # whose ``tileset_uid`` names no slot in ``self.tilesets`` (a file
            # mid-load, or a bug upstream) is refused exactly like the
            # placeholder/track-index guard above: no cel is invented for a
            # binding that names nothing, the slot stays a placeholder, and
            # the caller's write lands nowhere until the dangling binding is
            # fixed.
            slot = next((s for s in self.tilesets if s.uid == track.tileset_uid), None)
            if slot is None:
                return
            tile_w, tile_h = slot.tileset.tile_w, slot.tileset.tile_h
            grid_h, grid_w = grid_shape((width, height), tile_w, tile_h)
            real: Layer = TilemapCel(
                pixels=cp.empty(width, height),
                refs=np.zeros((grid_h, grid_w), dtype=np.uint32),
                tileset_uid=track.tileset_uid,
                uid=placeholder.uid,
                **{key: getattr(track, key) for key in CEL_PROPS},
            )
        else:
            # A *continuous* track starts its new cels from the nearest earlier
            # drawing rather than from nothing. Walked backwards from this frame
            # rather than taken from frame zero: a gap in the middle of a track
            # must not send the copy all the way to the beginning.
            #
            # A copy, not a link -- Aseprite links here and we do not. Linking
            # would make the first stroke on the new frame edit the old one too,
            # which is the opposite of "carry it forward and change it", and
            # linking afterwards is a verb the timeline already has.
            source = None
            if track.continuous:
                for earlier in range(anim.frame_index(frame.uid) - 1, -1, -1):
                    found = anim.cels.get((track.uid, anim.frames[earlier].uid))
                    if found is not None:
                        source = found
                        break
            # Every one of the six track properties, not four: ``alpha_lock`` was
            # the one left out, and the write that autovivified the cel is normally
            # one line away from reading it back off the layer -- ``begin_stroke``
            # samples ``layer.alpha_lock`` immediately after ``_ensure_active_cel``
            # -- so a missing lock did not merely mislabel the cel, it painted
            # through "preserve transparency" on the first stroke of every fresh
            # frame. ``locked`` (the content lock) joined the list for the same
            # reason and would fail the same way, one door further out.
            # ``placeholder`` and ``layers_for`` both copy all six; this is the
            # third copy of that list and it has to agree with them.
            real = Layer(
                pixels=cp.empty(width, height) if source is None else source.pixels.copy(),
                # A fresh cel in an indexed document is a plane of the transparent
                # index, not of zero: they are only the same slot by coincidence,
                # and a document whose transparent index is 7 would autovivify a
                # canvas full of slot 0 -- an opaque rectangle of whatever colour
                # slot 0 happens to hold, appearing the instant a stroke starts.
                #
                # A continuous copy takes the source's plane instead, indices and
                # all: copying the pixels alone would leave the two describing
                # different pictures from the moment the cel existed.
                indices=(
                    (
                        None
                        if self.color_mode != "indexed"
                        else np.full((height, width), self.transparent_index, dtype=np.uint8)
                    )
                    if source is None
                    else (None if source.indices is None else source.indices.copy())
                ),
                name=track.name,
                opacity=track.opacity,
                visible=track.visible,
                blend=track.blend,
                alpha_lock=track.alpha_lock,
                locked=track.locked,
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

    # -- index planes -------------------------------------------------------
    #
    # The materialisation has **one owner**, and these three methods are it.
    # Anything that writes an index plane calls one of them on the way out, so
    # "pixels are what the indices say they are" is a property of the model
    # rather than a rule each caller has to remember. The suite asserts it after
    # every session close (see ``_check_materialized``).

    def _index_lut(self) -> np.ndarray:
        """The document's palette as a ``(P, 4)`` lookup table.

        Built on demand rather than cached beside the palette. A table of at
        most 256 rows is a few microseconds to assemble, and a cache would need
        invalidating from ``set_palette``, ``recolour_slot``, ``add_slot``,
        ``remove_slot``, ``move_slot``, ``sort_palette``, ``insert_ramp``,
        ``PaletteEdit.undo`` and ``ColorStateEdit.undo`` -- nine places, each
        one a silent wrong-colour bug if it is missed, to save a rounding error.
        """
        return ixp.lut(self.palette or [TRANSPARENT], self.transparent_index)

    def _rematerialize(
        self, layer: Layer, table: np.ndarray | None = None, *, notify: bool = True
    ) -> None:
        """Rewrite one layer's pixels from its index plane, in place.

        In place rather than by rebinding ``layer.pixels``: the frame-flatten
        cache, the texture uploader and an open floating buffer may all be
        holding the array, and a rebind would leave them looking at the plane
        the document has stopped using.

        It invalidates by default, because writing pixels and not saying so is
        the one mistake this method can make silently: ``flatten`` answers from
        the cached composite, so a rematerialisation nobody announced shows the
        *old* colours in the exported PNG, in the thumbnail and in the ORA's
        ``mergedimage`` while the canvas is correct. ``notify=False`` is for the
        bulk paths below, which end in one ``invalidate_all`` rather than paying
        for a recomposite per layer.
        """
        if layer.indices is None:
            return
        table = self._index_lut() if table is None else table
        layer.pixels[...] = ixp.materialize(layer.indices, table)
        if notify:
            width, height = layer.size
            self.invalidate((0, 0, width, height), layer_uid=layer.uid)

    def check_materialized(self) -> None:
        """Raise unless every layer's pixels are what its indices say.

        The one invariant a truly indexed document can lose *silently*: any code
        that writes ``pixels`` without going through the funnel leaves the two
        planes describing different pictures, and the document goes on looking
        right until it is saved, undone, or reordered -- at which point the
        indices win and the drawing changes.

        Cheap enough to be called from anywhere (a full-canvas comparison per
        distinct cel), and called from the indexed test suite after every
        operation rather than from the ops themselves: an assertion inside the
        model would be either always on (a whole extra materialisation per
        stroke) or off in the build that matters.
        """
        if self.color_mode != "indexed":
            return
        table = self._index_lut()
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        for layer in layers:
            if layer.indices is None:
                continue
            want = ixp.materialize(layer.indices, table)
            if not np.array_equal(layer.pixels, want):
                raise AssertionError(
                    f"layer {layer.uid} ({layer.name!r}) has drifted from its index plane"
                )

    def apply_indices(
        self, layer_uid: int, rect: tuple[int, int, int, int], plane: np.ndarray
    ) -> None:
        """Write one rectangle of one layer's index plane and re-derive it.

        The hook :class:`~.undo.IndexPatchEdit` calls in both directions, and
        public for that reason alone -- an edit type is not a member of this
        package's private surface, it is a stored object that outlives the call
        that made it.

        ``layer_by_uid`` rather than ``stack.by_uid``, for ``PatchEdit._put``'s
        reason: on an animated document the cel this patch was recorded against
        may be on a frame the playhead has since moved off, and it must still be
        written.
        """
        layer = self.layer_by_uid(layer_uid)
        if layer.indices is None:
            # The document left indexed mode under this edit. Refused by name
            # rather than silently skipped: a half-applied history walk is worse
            # than a stopped one, and the guard exists to say which happened.
            raise ValueError("an index patch cannot apply to a layer with no index plane")
        x0, y0, x1, y1 = rect
        layer.indices[y0:y1, x0:x1] = plane
        table = self._index_lut()
        layer.pixels[y0:y1, x0:x1] = ixp.materialize(layer.indices[y0:y1, x0:x1], table)
        self.invalidate(rect, layer_uid=layer_uid)

    def apply_remap(self, forward: np.ndarray) -> None:
        """Push a slot permutation through every distinct index plane.

        :class:`~.undo.IndexRemapEdit`'s hook, public for the same reason. The
        table is applied and the pixels re-derived; the *palette* is not touched
        here, because the compound this edit rides in carries a
        :class:`~.undo.ColorStateEdit` that owns it. Two edits, one act, and
        neither one duplicating the other's half.
        """
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        for layer in layers:
            if layer.indices is not None:
                layer.indices = ixp.apply_remap(layer.indices, forward)
        self._rematerialize_all()

    def _rematerialize_all(self) -> None:
        """Every *distinct* plane in the document, once each.

        ``unique_cel_layers`` for ``_map_planes``' reason: a background linked
        across three frames is one object, and materialising it three times is
        three times the work to reach the same bytes.
        """
        if self.color_mode != "indexed":
            return
        table = self._index_lut()
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        for layer in layers:
            self._rematerialize(layer, table, notify=False)
        self._stamp_all()
        self.invalidate_all()

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

        **This is where the colour mode is applied**, and it is the one place
        worth doing it: every write that is undoable arrives here -- strokes,
        fills, shapes, gradients, filters, pastes, floating commits -- so the
        constraint holds for all of them rather than for the list somebody
        remembered. Applied *before* ``after`` is read, so the undo step records
        the pixels the document actually ends up with, and an undo followed by a
        redo lands on the same colours.

        One funnel, three constraints: palette-constrained RGB snaps onto the
        table, indexed resolves to slots (and takes the whole rest of the method
        with it -- the step it pushes is an index patch), grayscale flattens to
        luma. Tools go on painting RGBA in all three, which is what keeps this
        the *only* place any of them has to be known.
        """
        if isinstance(layer, TilemapCel):
            # Before the colour modes, and taking the whole method with it: a
            # tilemap cel's pixels are a *materialization*, so the write has to
            # become a tileset edit or be reverted, and there is no version of
            # it that also lands in an index plane. See
            # :meth:`~._doc_tiles.TileOps._commit_tilemap_patch`.
            self._commit_tilemap_patch(layer, rect, before)
            return
        if self.color_mode == "indexed" and layer.indices is not None:
            self._commit_indexed_patch(layer, rect)
            return
        x0, y0, x1, y1 = rect
        layer.pixels[y0:y1, x0:x1] = self._constrained(layer.pixels[y0:y1, x0:x1])
        after = layer.pixels[y0:y1, x0:x1].copy()
        if np.array_equal(before, after):
            self._discard_pending_cel()
            return
        pending, self._pending_cels = self._pending_cels, []
        release = self._matte_release((before[..., 3] == 255) & (after[..., 3] == 0))
        self._push_patch([*pending, PatchEdit(layer.uid, rect, before, after)], release)
        self.invalidate(rect, layer_uid=layer.uid)

    def _matte_release(self, opened: Any) -> Any:
        """Clearing the flatten matte, when a write has just punched a hole.

        ``matte_for`` stamps ``OPAQUE_WHITE`` on any fully opaque image at load,
        so a photo or a flat PNG flattens onto white -- which is right for a
        photo and is what a user removing a background is trying to undo. Until
        this existed only the AI cutout (``apply_matte``) and ``to_background``
        cleared it, so erasing by hand cut alpha the export then filled straight
        back in: the pixels really were transparent and the file really was
        white, and the only way out was a menu item nobody knew to look for.

        **A hole, not a soft edge.** The test is opaque-to-*fully*-transparent,
        so an antialiased brush edge -- partial alpha, everywhere, on every
        ordinary stroke -- does not disturb the matte of a document nobody is
        cutting out. An eraser, ``delete_selection`` and a fill with a
        transparent colour all reach 0 and all mean it.

        Skipped on a document with a background layer, ``set_matte``'s reason:
        ``flatten`` does not consult the matte there and ``_shown_pixels``
        forces that layer opaque, so there is no hole to answer for and the step
        would only be a history entry that changes nothing.
        """
        if self.matte is None or self.has_background:
            return None
        if not bool(np.any(opened)):
            return None
        return MatteEdit(tuple(self.matte), None)

    def _push_patch(self, edits: list[Any], release: Any) -> None:
        """One write's edits as a single step, with the matte release last.

        **Last is load-bearing**: ``CompoundEdit.undo`` walks ``reversed()``, so
        appending the ``MatteEdit`` at the end restores the colour *before* the
        pixels come back -- the order ``apply_matte`` and ``to_background``
        already use, and the one that stopped an undone cutout from restoring
        the pixels and losing the colour for good.
        """
        if release is not None:
            self.matte = None
            edits.append(release)
        self.history.push(one_step(edits))

    def _constrained(self, region: np.ndarray) -> np.ndarray:
        """*region* with the document's colour constraints applied.

        **Two ``if``s and not an ``if``/``elif``**: a grayscale document with a
        palette gets both, and a snap onto a table of greys is not the same
        array as a snap onto the table it actually has. That one-line rule was
        written out at three sites -- the funnel, its list-returning sibling and
        the tilemap patch -- each with its own comment explaining it, which is
        three places to get it wrong and three comments to keep in step.

        Indexed mode is not here: it takes the whole method with it at each
        caller, because the step it makes is an index patch rather than a pixel
        one.
        """
        if self.color_mode == "grayscale":
            region = ix.grayscale(region)
        if self.palette:
            region = ix.snap(region, self.palette)
        return region

    def _commit_indexed_patch(self, layer: Layer, rect: tuple[int, int, int, int]) -> None:
        """The indexed half of the funnel: RGBA in, slots resolved and stored.

        The RGBA ``before`` its caller holds is deliberately **not** used. The
        indices are the record, so the before-state is the index crop -- and it
        is still sitting untouched in ``layer.indices`` at this moment, because
        the tool wrote pixels and nothing writes indices except this method.
        Reading it here rather than threading it through every call site is what
        lets the entire tool layer stay index-unaware.

        The materialisation happens **before** the no-op test rather than after
        it, which is the whole reason a soft nib is legal in indexed mode: the
        tool's antialiased edge is written, resolved, and then overwritten by
        what the slots actually say. A gesture whose every pixel thresholds back
        to where it started therefore changes nothing, pushes nothing, and takes
        its autovivified cel back out again -- the same rule the RGBA path
        follows, reached by different arithmetic.
        """
        x0, y0, x1, y1 = rect
        before = layer.indices[y0:y1, x0:x1].copy()
        table = self._index_lut()
        prefer = None
        slot = self.paint_slot
        if slot is not None and self.palette and 0 <= slot < len(self.palette):
            prefer = (self.palette[slot], slot)
        after = ixp.resolve(
            layer.pixels[y0:y1, x0:x1], table, self.transparent_index, prefer=prefer
        )
        layer.indices[y0:y1, x0:x1] = after
        layer.pixels[y0:y1, x0:x1] = ixp.materialize(after, table)
        if np.array_equal(before, after):
            self._discard_pending_cel()
            # Still recomposited: the tool's raw write is on screen and has just
            # been rolled back to what the slots say, so the canvas is showing
            # pixels the document no longer holds until this runs.
            self.invalidate(rect, layer_uid=layer.uid)
            return
        pending, self._pending_cels = self._pending_cels, []
        # The indexed spelling of "a hole was punched": the transparent slot is
        # what alpha 0 means when the plane is the record.
        clear = self.transparent_index
        release = self._matte_release((before != clear) & (after == clear))
        self._push_patch([*pending, IndexPatchEdit(layer.uid, rect, before, after)], release)
        self.invalidate(rect, layer_uid=layer.uid)

    def _commit_permuted_indices(
        self, layer: Layer, rect: tuple[int, int, int, int], before: np.ndarray
    ) -> None:
        """Record a write that permuted the index plane itself. Indexed only.

        :meth:`_commit_indexed_patch`'s contract is that the caller wrote
        *pixels* and the plane is still the pre-write one -- which is what lets
        it read ``before`` off the layer and re-resolve ``after`` from the
        colours. A permutation breaks both halves at once. It moves the plane
        deliberately, so re-resolving would collapse exactly the duplicate-slot
        identity a permutation exists to preserve; and it has *already* moved
        it, so there is no untouched crop left on the layer to read.

        So the caller takes its own ``before`` crop first and hands it here.
        The whole-canvas ops in ``_doc_geometry`` do not need this -- they are
        recorded by snapshot through ``_replay``/``_map_planes`` -- which
        leaves ``offset_layer``, a single-layer permutation, as the one caller.

        It had none, and went through the funnel instead: that read the rolled
        plane as *both* sides of the step, found them equal, and pushed
        nothing. The roll was permanent, the history head never moved, so
        ``InkerDoc.dirty`` reported the file saved and the next Ctrl+Z undid
        the *previous* edit while the roll stayed.
        """
        x0, y0, x1, y1 = rect
        after = layer.indices[y0:y1, x0:x1]
        if np.array_equal(before, after):
            # A uniform plane rolls onto itself. Nothing changed, so nothing is
            # pushed and the autovivified cel goes back out -- the same rule
            # the two funnel paths follow, reached by different arithmetic.
            self._discard_pending_cel()
            return
        pending, self._pending_cels = self._pending_cels, []
        edit: Any = IndexPatchEdit(layer.uid, rect, before, after)
        self.history.push(edit if not pending else CompoundEdit([*pending, edit]))
        self.invalidate(rect, layer_uid=layer.uid)

    def _patch_edit_for(
        self, layer: Layer, rect: tuple[int, int, int, int], before: np.ndarray
    ) -> Any:
        """The list-returning sibling of :meth:`_commit_patch`.

        A range op writes many cels under **one** Ctrl+Z, so it cannot go
        through the funnel: ``_commit_patch`` pushes a step per call, and a
        rect of five cels would arrive on the stack as five. What it can share
        is everything the funnel is actually *for* -- the colour mode. So the
        constraint is applied here, to a region the caller has already written,
        and the edit is handed back for the caller's ``_push_range`` instead of
        being pushed.

        The order matters and is the funnel's own: indexed takes the whole
        method with it (the step it makes is an index patch), and grayscale and
        the palette snap are two ``if``s rather than an ``if``/``elif``,
        because a grayscale document with a palette gets both.

        ``None`` is "the mode resolved this write to nothing" -- the caller
        skips the cel, and the materialisation has already been put back, so
        the document is left exactly as it was found.

        Refused by name on a :class:`~.tiles.TilemapCel`. This funnel writes
        RGBA pixels; a tilemap cel's pixels are a *materialization* of
        ``refs``, and a range op landing here means a caller is writing pixels
        into one directly, which would leave the two disagreeing the moment
        the write commits. There is no version of this write to make yet --
        that is :meth:`~._doc_tiles.TileOps.place_tiles`'s door, which takes
        refs, not pixels.
        """
        if isinstance(layer, TilemapCel):
            raise ValueError("a pixel patch of a tilemap layer is not yet modeled")
        x0, y0, x1, y1 = rect
        if self.color_mode == "indexed" and layer.indices is not None:
            # The RGBA ``before`` is deliberately unused on this path, for
            # ``_commit_indexed_patch``'s reason: the indices are the record,
            # and the crop of them sitting in ``layer.indices`` right now is
            # still the before-state, because nothing but these three methods
            # ever writes one.
            idx_before = layer.indices[y0:y1, x0:x1].copy()
            table = self._index_lut()
            prefer = None
            slot = self.paint_slot
            if slot is not None and self.palette and 0 <= slot < len(self.palette):
                prefer = (self.palette[slot], slot)
            idx_after = ixp.resolve(
                layer.pixels[y0:y1, x0:x1], table, self.transparent_index, prefer=prefer
            )
            layer.indices[y0:y1, x0:x1] = idx_after
            # Materialised *before* the no-op test, as the funnel does it: a
            # write whose every pixel resolves back to the slot it came from
            # has to leave the pixels saying what the slots say, or the caller
            # skips the cel and the drift it just made stays.
            layer.pixels[y0:y1, x0:x1] = ixp.materialize(idx_after, table)
            if np.array_equal(idx_before, idx_after):
                return None
            return IndexPatchEdit(layer.uid, rect, idx_before, idx_after)
        layer.pixels[y0:y1, x0:x1] = self._constrained(layer.pixels[y0:y1, x0:x1])
        after = layer.pixels[y0:y1, x0:x1].copy()
        if np.array_equal(before, after):
            return None
        return PatchEdit(layer.uid, rect, before, after)

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
        # Copied *before* the op, and with their uids kept, for the reason the
        # layer snapshot above gives: the edit holds these for as long as it is
        # on the stack, so a later drag must not write into them, and every
        # slice edit already pushed names its slice by the uid it has now.
        slices = [entry.copy() for entry in self.slices]

        from . import groups as gp

        tree = gp.copy_tree(self.groups, self.group_of)
        run()

        def replay(doc: Any) -> None:
            # ``doc`` is asserted rather than ignored. ``run`` closes over
            # ``self``, so replaying this step against a *different* document
            # would silently edit the original -- there is no such caller today
            # (a step is only ever redone on the stack it was pushed to) and
            # this is what keeps it that way.
            if doc is not self:
                raise ValueError("a replay step belongs to the document that made it")
            run()
            doc.invalidate_all()

        self.history.push(ReplayEdit(snapshot, size, active, replay, mask, grid, slices, tree))
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

    def _map_planes(
        self,
        fn: Any,
        *,
        mask_fn: Any = _SAME_AS_PIXELS,
        index_fn: Any = None,
        refs_fn: Any = None,
    ) -> None:
        """Apply *fn* to every distinct pixel plane, and *mask_fn* to the mask.

        ``mask_fn`` defaults to ``fn`` because the first callers were geometry:
        a rotate has to rotate the marquee with the pixels, or the selection
        comes back describing a different part of the picture. A **colour** map
        is the other kind of caller and passes ``mask_fn=None`` -- a palette is
        a statement about colour, and a mask is 8-bit coverage with no colour
        for it to say anything about.

        Passing ``fn`` through unconditionally was a live bug rather than a
        stylistic one: ``indexed.snap`` and ``indexed.remap`` both raise on a
        2-D array, so ``set_palette``, ``recolour_slot`` and ``remove_slot``
        every one of them refused outright -- out of the middle of the op, with
        the table already assigned -- whenever a selection happened to be up.

        ``index_fn`` is the third kind of caller, and its **default is the
        honest one rather than the exact one**. Given a callable, the index
        plane is transformed by it -- which for a flip, a quarter turn, a crop
        or a canvas resize is the same permutation the pixels get, so duplicate
        slots survive geometry exactly and nothing is ever re-inferred. Given
        nothing, the plane is re-*resolved* from the mapped pixels, which is the
        only thing that can be done when the op resampled (a smooth scale
        invents colours that were never in the table, so there is no
        permutation to apply) and is the one stated place in the whole design
        where indices come back from colours. An op that could have been exact
        and forgot to say so loses duplicate identity and stays correct; the
        reverse default would hand a 2-D array to ``indexed.snap`` and raise out
        of the middle of a rotate.

        ``refs_fn`` is the fourth kind, and the canvas resize is the only op
        that passes one (``_doc_tiles._tile_regrid``). It is called with each
        *distinct layer*, after that layer's pixels have been mapped, and
        ignores anything that is not a tilemap cel -- so an op that never
        learned about refs simply never passes one, and a document holding no
        tilemap pays nothing at all. Handed the layer rather than the plane,
        unlike the other three, because a re-grid must re-derive ``pixels``
        from the refs it just wrote: a tilemap cel's picture is a
        materialization, never a second copy kept in step.
        """
        if self.mask is not None:
            apply = fn if mask_fn is _SAME_AS_PIXELS else mask_fn
            if apply is not None:
                self.mask = SelectionMask(apply(self.mask.mask))
        anim = self.anim
        if anim is None:
            for layer in self.stack:
                layer.pixels = fn(layer.pixels)
                self._map_index_plane(layer, index_fn)
                if refs_fn is not None:
                    refs_fn(layer)
            return
        # Each *distinct* cel exactly once. Walking the stack, or the slots,
        # would rotate a background linked across three frames three times.
        for layer in anim.unique_cel_layers():
            layer.pixels = fn(layer.pixels)
            self._map_index_plane(layer, index_fn)
            if refs_fn is not None:
                refs_fn(layer)
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

    def _map_index_plane(self, layer: Layer, index_fn: Any) -> None:
        """One layer's index plane through a geometry op. See ``_map_planes``.

        The pixels are then re-derived rather than trusted: ``fn`` and
        ``index_fn`` are two spellings of the same permutation, and re-deriving
        is what makes them impossible to disagree. On the re-resolve path it is
        the definition of the result.
        """
        if self.color_mode != "indexed" or layer.indices is None:
            return
        if index_fn is None:
            layer.indices = ixp.resolve(
                layer.pixels, self._index_lut(), self.transparent_index
            )
        else:
            layer.indices = index_fn(layer.indices)
        self._rematerialize(layer, notify=False)
