"""The layer model: canvas-sized RGBA planes and the stack that orders them.

Two decisions that everything else leans on. A layer is *canvas-sized* -- no
per-layer offsets in memory, so every op is a plain slice and no code has to
reason about two coordinate spaces (ORA's offsets are applied at load and
written back out as zero). At 2048x2048 that is 16 MiB a layer; ten layers is
160 MiB, which is the price of never having a bug about where a layer starts.

And a layer has a stable ``uid`` that has nothing to do with its position.
Undo records patches against the uid, so reordering a stack between an edit and
its undo cannot make the patch land on a different layer.

A layer on a **truly indexed** document carries a third thing: an ``(H, W)``
index plane which is the record, with ``pixels`` kept beside it as the
materialisation. The RGBA invariant above is unchanged -- ``pixels`` is always
``(H, W, 4)`` and always valid -- so nothing that reads a layer needs a new
case. See :mod:`.index_plane` for why identity, not storage, is what that buys.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from . import composite as cp

# ``groups`` imports ``new_uid`` from this module, so the span helpers it owns
# are reached through a function-body import rather than at module scope. The
# same shape ``Document.group_fold`` uses, for the same cycle.

_uids = itertools.count(1)


def new_uid() -> int:
    """Mint a layer uid without building a layer.

    A replayed operation that *creates* a layer needs the same uid every time
    it runs, or a redo strands every patch recorded above it. The op takes its
    uid from here once and closes over it.
    """
    return next(_uids)


#: The fields :meth:`Layer.copy` passes explicitly -- the two planes it deep-copies
#: and the two identity fields it may override. Everything else is carried by
#: name, so a new ``Layer`` field is copied without anyone remembering to add it.
_COPY_EXPLICIT = frozenset({"pixels", "indices", "name", "uid"})


@dataclass
class Layer:
    pixels: np.ndarray
    name: str = "Layer"
    opacity: float = 1.0
    visible: bool = True
    blend: str = "normal"
    # "Preserve transparency": writes may change this layer's colours but not
    # its alpha, so painting inside a shape cannot spill past its edge. An
    # editing aid rather than picture data -- nothing about the composite reads
    # it, and a locked layer flattens exactly as an unlocked one does.
    alpha_lock: bool = False
    # "Content lock": this layer refuses tool-level writes altogether -- see
    # ``LayerOps.write_locked`` for the one list of doors that enforce it and
    # the argument for where the line is drawn. Like ``alpha_lock`` it is an
    # editing aid rather than picture data: nothing about the composite reads
    # it, and a locked layer flattens exactly as an unlocked one does.
    locked: bool = False
    #: **A real background layer** (6.5), and the re-argument of divergence #6.
    #:
    #: ``Document.matte`` was the stand-in: a colour composited under the stack
    #: at flatten time, invisible to the layer model and therefore droppable on
    #: every export -- an ``.aseprite`` written from this app said nothing
    #: about it because there was nothing in the model to say. A flag on the
    #: bottom layer is what Aseprite has, is what the file format carries
    #: (layer flag ``0x08``), and is a thing the user can paint on.
    #:
    #: What the flag *means* here is one sentence: **a background layer is
    #: opaque**. The composite forces its alpha to 255, so erasing on it
    #: reveals the colour under the eraser rather than a hole -- which is
    #: Aseprite's behaviour and the reason the type exists at all. It does not
    #: change what any tool writes, which is what keeps every write door,
    #: every undo patch and every blend mode exactly as they are.
    #:
    #: Only the bottom layer may carry it; ``LayerStack`` is the one place that
    #: rule is enforced, so a reorder cannot leave a background in the middle
    #: of the stack.
    background: bool = False
    #: A **reference layer**: drawn, never edited. Aseprite's own type, and
    #: ``asein`` has been opening them (hidden) since the reader landed. It is
    #: the content lock plus a statement of intent -- ``write_locked`` reads it
    #: as well as ``locked``, so no new door has to learn about it.
    reference: bool = False
    uid: int = field(default_factory=lambda: next(_uids))
    #: The **record**, on a document in ``color_mode == "indexed"`` -- and None
    #: on every other document, which is every document until somebody converts
    #: one. When it is present, ``pixels`` is its materialisation through the
    #: document's palette (``pixels == index_plane.materialize(indices, lut)``)
    #: and this plane is what geometry permutes, what undo records and what the
    #: ORA writer stores.
    #:
    #: ``pixels`` stays ``(H, W, 4)`` and stays *valid* either way, which is the
    #: decision the whole indexed design turns on: the compositor's nineteen
    #: blend modes, the frame-flatten cache, texture upload, GIF and sheet
    #: flatten, thumbnails and every native kernel read ``pixels`` and change by
    #: zero lines. The invariant above is demoted from "the record" to "the
    #: projection" on an indexed document; it is not weakened.
    #:
    #: A **linked cel shares its index plane for free**, because a link is the
    #: same ``Layer`` object in two ``Animation.cels`` slots and there is only
    #: one plane to share. See :mod:`.index_plane`.
    indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.pixels.dtype != np.uint8 or self.pixels.ndim != 3 or self.pixels.shape[2] != 4:
            raise ValueError("a layer holds (H, W, 4) uint8")
        if self.blend not in cp.BLEND_MODES:
            raise ValueError(f"unknown blend mode {self.blend!r}")
        if self.indices is not None:
            # Shape-matched against the pixels rather than against the canvas:
            # a layer is canvas-sized by ``LayerStack.insert``'s rule, and the
            # one thing that must never come apart is these two describing
            # different rectangles -- every materialisation writes the whole of
            # ``pixels`` from the whole of ``indices``.
            if self.indices.dtype != np.uint8 or self.indices.ndim != 2:
                raise ValueError("an index plane is (H, W) uint8")
            if self.indices.shape != self.pixels.shape[:2]:
                raise ValueError("an index plane is the size of the layer it indexes")

    @property
    def plane_bytes(self) -> int:
        """What this layer costs the undo budget and the frame caches.

        One helper rather than ``pixels.nbytes`` spelled at each cost site,
        because an index plane is a fifth field those sites did not know about
        and an edit that under-reports its size is one the byte budget cannot
        evict.
        """
        return int(self.pixels.nbytes) + (0 if self.indices is None else int(self.indices.nbytes))

    @property
    def size(self) -> tuple[int, int]:
        return (self.pixels.shape[1], self.pixels.shape[0])

    @classmethod
    def empty(cls, width: int, height: int, name: str = "Layer") -> Layer:
        return cls(pixels=cp.empty(width, height), name=name)

    def copy(self, *, name: str | None = None, uid: int | None = None) -> Layer:
        """A duplicate with its *own* uid -- a copy is a different layer, and
        sharing the uid would let one layer's undo patch rewrite the other.

        ``uid`` overrides that for the one caller who needs the opposite: a
        history snapshot is not a new layer, it is the *same* layer's pixels
        held aside, and restoring it has to give back the identity every patch
        recorded before it is addressed to.

        **Every field is carried by name rather than re-listed.** The hand-written
        version this replaces named seven of the eleven and silently dropped
        ``background`` and ``reference``, which made every whole-canvas op lose
        them: ``_replay`` and ``restore_snapshot`` both snapshot with
        ``copy(uid=layer.uid)``, so rotating the canvas and pressing Ctrl+Z gave
        back a background layer that was no longer a background layer.
        """
        return Layer(
            # Both planes are copied rather than shared, and for the same
            # reason: a history snapshot holds them for as long as it is on the
            # stack, and a shared array would let the next stroke rewrite what
            # the undo is supposed to restore.
            pixels=self.pixels.copy(),
            indices=None if self.indices is None else self.indices.copy(),
            name=self.name if name is None else name,
            **{
                spec.name: getattr(self, spec.name)
                for spec in fields(Layer)
                if spec.name not in _COPY_EXPLICIT
            },
            **({} if uid is None else {"uid": uid}),
        )


def _shown_pixels(layer: Layer, pixels: np.ndarray | None = None) -> np.ndarray:
    """What the compositor blends for this layer.

    ``pixels`` overrides the layer's own plane and is how a per-frame palette
    reaches the composite (``LayerStack.frame_pixels``). The background rule
    below is applied to whichever plane is being shown, because it is a
    property of the *layer* and not of the array.

    Its own array, except for a **background layer** (6.5), whose alpha is
    forced opaque. That is the whole of what the flag means: erasing on a
    background reveals the colour under the eraser rather than a hole, which is
    Aseprite's behaviour and the reason the type exists.

    Forced *here*, at the composite, rather than at every write door: a rule
    applied at the doors would have to be taught to fifteen of them and would
    change what undo restores, while this changes only what is shown -- so the
    pixels the user painted are still the pixels in the file, and unmarking the
    layer gives back exactly the drawing that was there.

    A copy only when the flag is set and the layer is not already opaque, so
    every ordinary document composites the array it always did.
    """
    pixels = layer.pixels if pixels is None else pixels
    if not getattr(layer, "background", False):
        return pixels
    if pixels.size and int(pixels[..., 3].min()) == 255:
        return pixels
    out = pixels.copy()
    out[..., 3] = 255
    return out


class LayerStack:
    """Bottom-first list plus an active index. Never empty."""

    def __init__(self, layers: list[Layer], active: int = 0) -> None:
        if not layers:
            raise ValueError("a document has at least one layer")
        self.layers = layers
        self.active_index = max(0, min(int(active), len(layers) - 1))
        #: Per-row ``(visible, opacity)`` inherited from the layer groups above
        #: it, or None on a document with no groups -- which is every document
        #: until somebody makes one, so the ordinary path allocates nothing and
        #: compares one reference.
        #:
        #: It lives on the *stack* rather than on the layers because it is not a
        #: property of a layer: the same cel can be in a group on one frame's
        #: view and be a placeholder on another's, and writing the folded value
        #: onto the layer would make a hidden group's cel look hidden to
        #: everything that reads ``layer.visible`` -- including the panel's own
        #: eye checkbox, which must go on showing what the *layer* is set to.
        #: The document refreshes it in ``invalidate_all``, which is what every
        #: structural change ends with. See ``inker/groups.py``.
        self.group_fold: list[tuple[bool, float]] | None = None
        #: Per-row **z offset** from ``Animation.cel_z``, parallel to
        #: ``self.layers``, or None on every stack that does not use the
        #: feature -- which is every still document and every animated one
        #: until somebody sets a z, so the ordinary path compares one
        #: reference and sorts nothing.
        #:
        #: It lives on the *stack* rather than on the layers for
        #: ``group_fold``'s reason: it is a property of the **slot**, not of
        #: the cel, and a linked cel is one ``Layer`` object in two slots that
        #: may sit at two different heights. ``Document.invalidate_all`` and
        #: ``Document.frame_stack`` are the two places that fill it in, exactly
        #: as they are for the fold.
        #:
        #: **None rather than a list of zeros, and that distinction is load
        #: bearing**: it is what ``Document.invalidate`` reads to decide
        #: whether the ``_below`` cache is still sound. See :meth:`_entries`.
        self.cel_z: list[int] | None = None
        #: The isolated groups over this stack as ``groups.Span`` slices,
        #: outermost-first, or None when nothing is isolated -- which is every
        #: document until somebody gives a group a blend mode, so the ordinary
        #: path compares one reference and recurses not at all.
        #:
        #: It lives beside the fold rather than replacing it because the two
        #: answer different questions: the fold is what a *row* inherits, and a
        #: span is a compositing stage the rows belong to. A pass-through
        #: group appears only in the fold; an isolated one appears in both,
        #: with its opacity in the span alone (see ``groups.resolve``).
        #:
        #: **None rather than an empty list**, for ``cel_z``'s reason exactly:
        #: it is what :meth:`cuts_isolated` reads to tell ``Document.invalidate``
        #: whether the ``_below`` cache is sound this frame.
        self.group_spans: list[Any] | None = None
        #: **Per-frame palette materialisation**: layer uid -> the pixels this
        #: frame shows for that layer, or None when the document has no
        #: per-frame palettes -- which is every document until somebody gives a
        #: frame its own colour table.
        #:
        #: An *override beside* ``Layer.pixels`` rather than a rewrite of it,
        #: and that is the whole design. ``Layer.pixels`` goes on being the
        #: document table's materialisation, so ``check_materialized``'s
        #: invariant is untouched, a stroke still writes where it always wrote,
        #: and both file writers go on reading what they always read. Only what
        #: is *shown* changes, which is all a palette swap ever was.
        #:
        #: Rewriting the shared ``Layer`` in place instead would have been five
        #: lines and a trap: a linked cel is one object in several frames, so
        #: materialising it for the onion-skinned neighbour would silently
        #: repaint the frame being edited.
        self.frame_pixels: dict[int, np.ndarray] | None = None

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
        """Put *layer* at *index* and **select it**.

        Selecting it is deliberate and is what makes undo read correctly: a
        ``LayerRemoveEdit`` puts the layer back through here, so undoing a
        delete leaves you on the layer you just got back rather than wherever
        the cursor happened to be. Every editor behaves this way and it is the
        reason the move is not itself recorded as an edit -- the selection is
        view state, so a redo does not owe the user the index they had.
        """
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
        # Shifted, not just clamped. Removing a layer *below* the active one
        # moves every index above it down by one, so a clamp alone left
        # ``active_index`` pointing at the layer that used to be above it --
        # the active layer silently became a different layer, and the next
        # stroke landed on it.
        if index < self.active_index:
            self.active_index -= 1
        self.active_index = max(0, min(self.active_index, len(self.layers) - 1))
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

    def _order(self, lo: int, hi: int) -> list[int] | None:
        """The row indices ``[lo, hi)`` blend in, or None for "as they lie".

        None on every stack with no ``cel_z`` -- one ``is None`` test, no
        allocation, no sort -- and on every *partial* range even when there is
        one. **That second half is the whole safety argument.** A z offset can
        move a row across the active layer, so a partial range no longer names
        a contiguous slice of the finished picture and there is no correct
        answer to give for one; the only sound thing to do is refuse to be
        asked, which is what returning None here does. ``Document.invalidate``
        is the caller that makes that possible: it drops the ``_below`` cache
        for exactly these documents so every composite is a full-stack one.

        The key is ``(index + z, index)``. The tiebreak on the raw index is
        what keeps the sort stable *and* meaningful: two rows given the same
        effective height stay in track order, which is what an offset of zero
        means on every row and therefore why an unused feature is the identity
        permutation rather than merely a cheap one.
        """
        zs = self.cel_z
        if zs is None or lo != 0 or hi != len(self.layers):
            return None
        return sorted(
            range(lo, hi),
            key=lambda index: (
                index + (zs[index] if index < len(zs) else 0),
                index,
            ),
        )

    def cuts_isolated(self, index: int) -> bool:
        """Whether splitting the stack at ``index`` would cut an isolated group.

        The ``_below`` cache's soundness test, and the reason isolation does
        **not** turn that cache off wholesale the way per-cel z does. The cache
        holds a composite of rows ``[0, index)`` on the premise that they are
        finished business, and an isolated group sitting entirely below the
        active layer *is* finished business -- its result is computed once and
        folds into the base like any other row.

        It is unsound in exactly one case: the split falls strictly inside a
        span, so one half of a group would have to be rendered in isolation
        without the other. There is no correct answer to give for that, so
        ``Document.invalidate`` takes the uncached path instead of asking.

        A boundary is not a cut. ``lo == index`` puts the whole group above the
        split and ``hi == index`` puts it entirely below; both are ranges that
        hold the group whole, which is all the recursion needs.
        """
        spans = self.group_spans
        if spans is None:
            return False
        return any(span.lo < index < span.hi for span in spans)

    def _entries(
        self, lo: int, hi: int, rect: tuple[int, int, int, int] | None = None
    ) -> list[tuple[Any, float, str]]:
        """The rows a composite of ``[lo, hi)`` actually blends.

        The group fold is applied *here* and only here, which is what keeps the
        feature out of the compositor entirely: by the time
        ``composite.stack_region`` sees a row it is an ordinary
        ``(pixels, opacity, blend)`` triple and knows nothing about a tree.
        Pass-through, so each layer still blends against everything beneath it
        -- see ``inker/groups.py`` for why isolated compositing is a v1 gap.

        **Per-cel z-index is applied here too, and for the same reason**: by
        the time ``composite.stack_region`` or the native stack kernel sees this
        list it is a list of triples in blend order and neither learns that a
        row was moved. :meth:`_order` decides the order; a stack with no z gets
        None back and takes the path it always took, down to the comprehension.

        **An isolated group is the one thing that cannot be flattened away
        here**, because its members compose against each other rather than
        against the document -- so when ``group_spans`` is set this hands off
        to :meth:`_isolated_entries`, which renders each span with a nested
        ``stack_region`` and returns the result as one ordinary entry. ``rect``
        is what that nested call needs and is why it is a parameter at all; it
        is unused on every other path, and required on that one.
        """
        if self.group_spans is not None and rect is not None:
            return self._isolated_entries(lo, hi, rect)
        fold = self.group_fold
        order = self._order(lo, hi)
        if fold is None and self.frame_pixels is None:
            # The comprehension is kept for the document that has neither a
            # group fold nor a per-frame palette, which is almost all of them:
            # ``_row_entry`` is the same arithmetic and one call per row, and
            # this path runs once per layer per composite.
            if order is None:
                return [
                    (_shown_pixels(layer), layer.opacity, layer.blend)
                    for layer in self.layers[lo:hi]
                    if layer.visible and layer.opacity > 0.0
                ]
            return [
                (_shown_pixels(self.layers[i]), self.layers[i].opacity, self.layers[i].blend)
                for i in order
                if self.layers[i].visible and self.layers[i].opacity > 0.0
            ]
        out: list[tuple[np.ndarray, float, str]] = []
        for index in range(lo, hi) if order is None else order:
            entry = self._row_entry(index)
            if entry is not None:
                out.append(entry)
        return out

    def _row_entry(self, index: int) -> tuple[np.ndarray, float, str] | None:
        """One stack row as an entry, or None if it contributes nothing.

        The body both ``_entries`` paths share once a row's position is
        settled: the fold's ``(visible, opacity)``, the layer's own, and the
        background-layer alpha forcing that ``_shown_pixels`` does.
        """
        layer = self.layers[index]
        fold = self.group_fold
        shown = self.frame_pixels
        # A fold shorter than the stack is a rebuild caught midway; treat the
        # missing rows as ungrouped rather than raising out of a draw.
        visible, opacity = (
            fold[index] if fold is not None and index < len(fold) else (True, 1.0)
        )
        opacity *= layer.opacity
        if not (layer.visible and visible and opacity > 0.0):
            return None
        pixels = layer.pixels if shown is None else shown.get(layer.uid, layer.pixels)
        return (_shown_pixels(layer, pixels), opacity, layer.blend)

    def _isolated_entries(
        self,
        lo: int,
        hi: int,
        rect: tuple[int, int, int, int],
        inside: int | None = None,
    ) -> list[tuple[Any, float, str]]:
        """``[lo, hi)`` as entries, with each isolated group folded to one.

        One level of the bracket sequence at a time. The range is partitioned
        into loose rows and whole spans; each span is composited onto
        transparency by a recursive call at the same ``rect`` and joins the
        list as a :class:`composite.PreparedRegion` carrying the group's own
        opacity and mode. Nesting falls out of the recursion, and the scratch
        planes are *rect*-sized rather than canvas-sized -- which is what keeps
        a dab-sized invalidation a dab-sized allocation.

        **Per-cel z sorts the items at each level, not the stack.** A span is
        one item and sorts on its own start with no offset of its own, so a
        lifted row can still cross a whole group -- above it or below it -- and
        cannot enter one. That is the boundary an isolated group needs to keep
        being a group, and it costs a z offset nothing it could express before,
        because a row inside a group could only ever be lifted past its
        siblings anyway.
        """
        from . import groups as gp

        spans = self.group_spans or ()
        zs = self.cel_z

        items: list[tuple[int, Any]] = []
        at = lo
        for span in gp.top_level(spans, lo, hi, inside):
            items.extend((index, None) for index in range(at, span.lo))
            items.append((span.lo, span))
            at = span.hi
        items.extend((index, None) for index in range(at, hi))

        if zs is not None:
            items.sort(
                key=lambda item: (
                    item[0]
                    + (
                        0
                        if item[1] is not None or item[0] >= len(zs)
                        else zs[item[0]]
                    ),
                    item[0],
                )
            )

        out: list[tuple[Any, float, str]] = []
        for index, span in items:
            if span is None:
                entry = self._row_entry(index)
                if entry is not None:
                    out.append(entry)
                continue
            if span.opacity <= 0.0:
                continue
            inner = self._isolated_entries(span.lo, span.hi, rect, span.guid)
            if not inner:
                # An empty group contributes a transparent plane, which ``over``
                # would spend a full-region pass discovering is a no-op.
                continue
            region = cp.stack_region(inner, rect)
            out.append((cp.PreparedRegion(region), span.opacity, span.blend))
        return out

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
            return cp.stack_region(self._entries(0, len(self.layers), rect), rect)
        base = below[y0:y1, x0:x1]
        return cp.stack_region(
            self._entries(self.active_index, len(self.layers), rect), rect, base
        )

    def composite_below(self) -> np.ndarray:
        """Full-canvas composite of the layers under the active one."""
        width, height = self.size
        full = (0, 0, width, height)
        return cp.stack_region(self._entries(0, self.active_index, full), full)

    def composite_below_region(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        """One rectangle of that composite, for repairing the cache in place.

        A write to a layer *under* the active one invalidates the cached base
        but not the whole document, so the base is patched over the same
        rectangle rather than rebuilt at full canvas size.
        """
        return cp.stack_region(self._entries(0, self.active_index, rect), rect)

    def flatten(self) -> np.ndarray:
        width, height = self.size
        return cp.to_uint8(self.composite_region((0, 0, width, height)))
