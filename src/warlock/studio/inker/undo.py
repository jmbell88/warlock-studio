"""History as typed edits, not as whole-image snapshots.

The old model copied the entire image before every operation. That was honest
-- a snapshot cannot be wrong about what it restores -- and it stopped scaling
the moment documents grew layers: ten layers at 2048x2048 is 160 MiB a step, so
the 192 MiB budget bought exactly one undo.

The replacement keeps the "cannot be wrong" property where it matters by
storing *exact pixels*, just only the ones that changed: a stroke's edit is the
before/after crops of its dirty rectangle, typically a megabyte or two, which
is a hundred-odd steps inside the same budget. Structural changes (add, remove,
reorder, rename) carry no pixels at all. Only whole-canvas geometry -- rotate,
scale, crop -- falls back to copying, and it redoes by replaying rather than by
storing a second copy.

Every edit addresses its layer by ``uid``, never by index, so an undo issued
after a reorder still lands on the layer the edit was made to.

The engine itself -- ``Edit``, ``CompoundEdit``, ``UndoStack``, the serial
counter and the byte budget -- has no opinion about pixels and now lives in
``studio/undo.py``, so Clay can have the same history with its own edit
types without depending on the raster editor. It is re-exported here unchanged:
every module in this package imports those names from this one, and renaming
those imports as part of a move would have made "did the move change
behaviour?" unanswerable from the diff. What stays below is everything that is
genuinely about pixels or layers.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..undo import (
    UNDO_BYTES,
    UNDO_HARD_BYTES,
    UNDO_MAX_DEPTH,
    UNDO_MIN_DEPTH,
    CompoundEdit,
    Edit,
    UndoStack,
    _serials,
)

# This list exists to declare the re-export above as intentional -- without it
# every imported name is an unused import (ruff F401) -- so it names the eight
# re-exported names plus the public types defined below. ``_serials`` is in it
# for that reason alone and not because it is public; ``_pack``/``_unpack`` are
# defined here rather than imported, so they need no such declaration and stay
# out, as private helpers ordinarily would.
__all__ = [
    "UNDO_BYTES",
    "UNDO_HARD_BYTES",
    "UNDO_MAX_DEPTH",
    "UNDO_MIN_DEPTH",
    "ColorStateEdit",
    "CompoundEdit",
    "Edit",
    "IndexPatchEdit",
    "IndexRemapEdit",
    "LayerAddEdit",
    "LayerFlagEdit",
    "LayerMoveEdit",
    "LayerPropsEdit",
    "LayerRemoveEdit",
    "MatteEdit",
    "PaletteEdit",
    "PatchEdit",
    "ReplayEdit",
    "SelectionEdit",
    "UndoStack",
    "_serials",
    "one_step",
]


def one_step(edits: list[Any]) -> Any:
    """*edits* as a single undo step: the edit itself when there is only one.

    The ``edits[0] if len(edits) == 1 else CompoundEdit(edits)`` line, which was
    written out at nine call sites. Not merely repetition: a lone
    ``CompoundEdit`` wrapping one edit is a real difference -- the history panel
    labels a step by its class name, so an unwrapped step reads as what it did
    and a wrapped one reads as "compound" -- so every site has to make the same
    choice, and nine copies of a choice is how one of them comes to differ.

    Empty is a programming error rather than a no-op: a caller with nothing to
    push must not push, because a step that changes nothing moves the history
    head and asks the user to save a gesture that did not happen.
    """
    if not edits:
        raise ValueError("a step needs at least one edit")
    return edits[0] if len(edits) == 1 else CompoundEdit(edits)


def _plane_bytes(layer: Any) -> int:
    """What holding *layer* costs the byte budget.

    Its pixels plus its index plane, where it has one. Written as a ``getattr``
    against the layer's own ``plane_bytes`` rather than reaching for the two
    arrays here, so a layer subclass that carries a third plane -- a tilemap
    cel's refs, next wave -- answers for itself and this stays the one question.

    The fallback is for the duck-typed stand-ins the animation tests build,
    which have pixels and nothing else. A ``0`` here would be the failure this
    helper exists to prevent, so it is deliberately not the fallback.
    """
    cost = getattr(layer, "plane_bytes", None)
    return int(cost) if cost is not None else int(layer.pixels.nbytes)


@dataclass
class PatchEdit(Edit):
    """Exact pixels of one rectangle of one layer, before and after.

    This is the shape almost every edit takes: strokes, fills, gradients,
    shapes, lifting and committing a floating buffer, transforming a region.
    """

    layer_uid: int
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        # Own the pixels rather than viewing someone else's. A slice of a
        # full-canvas array reports only its own ``nbytes`` while keeping the
        # whole base alive, so a stroke that costs four kilobytes by the
        # budget's reckoning can pin sixteen megabytes -- invisibly, because
        # eviction is driven by exactly that number.
        #
        # Copied unconditionally, not only when ``base is not None``. That test
        # catches a *view* and misses the other way a caller keeps hold of the
        # pixels: handing over an array it owns and then writing to it, at
        # which point the recorded "before" quietly becomes the "after" and the
        # undo restores nothing. Every call site here happens to copy
        # defensively already, which is what made the weaker test look
        # sufficient -- and is also why paying for the copy twice costs
        # nothing that shows up in a profile against a stroke's own work. The
        # sibling stacks in ``plotter/edits.py`` and ``clay/edits.py`` both
        # take ownership outright; this is the same rule, spelled the same way.
        self.before = self.before.copy()
        self.after = self.after.copy()
        self.cost = int(self.before.nbytes + self.after.nbytes)

    def _put(self, doc: Any, pixels: np.ndarray) -> None:
        # ``layer_by_uid``, not ``stack.by_uid``: on an animated document the
        # cel this patch was recorded against may be on a frame the playhead has
        # since moved off, and it must still be written. ``invalidate`` is the
        # other half -- it stamps that frame's cached flatten and recomposites
        # nothing, because none of those pixels are on screen.
        x0, y0, x1, y1 = self.rect
        layer = doc.layer_by_uid(self.layer_uid)
        layer.pixels[y0:y1, x0:x1] = pixels
        doc.invalidate(self.rect, layer_uid=self.layer_uid)

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)


@dataclass
class IndexPatchEdit(Edit):
    """:class:`PatchEdit` for a truly indexed document: **index** crops.

    A quarter of the bytes of the RGBA patch it replaces, which is not the
    reason it exists -- the reason is that indices are the record, and an RGBA
    patch cannot restore them. Undoing a stroke by writing back the *colours*
    would land every pixel on the lowest-numbered slot holding that colour, so
    one undo would silently collapse the duplicate-slot identity the mode is
    for.

    Both directions write indices and then re-materialise through the
    document's **live** lookup table rather than through one recorded here. That
    is correct by history *ordering* rather than by luck: a palette change is
    itself an edit on the same stack, inside the same compound where the two
    happen together, so by the time this runs the table is always the one that
    belongs beside these indices. Recording a table here would be a second
    answer to the same question and the two would drift.
    """

    layer_uid: int
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        # Owned outright, for ``PatchEdit``'s reason spelled at length there.
        self.before = self.before.copy()
        self.after = self.after.copy()
        self.cost = int(self.before.nbytes + self.after.nbytes)

    def _put(self, doc: Any, plane: np.ndarray) -> None:
        doc.apply_indices(self.layer_uid, self.rect, plane)

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)


@dataclass
class IndexRemapEdit(Edit):
    """A slot permutation applied to every index plane in the document.

    About a kilobyte for a whole-document reorder that would otherwise be a full
    snapshot per layer per frame: the two tables are 256 bytes each, and the
    work is one fancy-index per distinct plane. That is what makes reordering a
    palette in indexed mode affordable *and* undoable, where in constrained-RGB
    mode it is free and unrecorded because no pixel moves.

    ``forward`` and ``inverse`` are both carried rather than one being derived:
    a permutation's inverse is cheap to compute but the *pair* is what
    ``index_plane.permutation_tables`` guarantees agrees with the list operation
    performed on the table, and re-deriving it here would be a second place for
    that agreement to be got wrong.
    """

    forward: np.ndarray
    inverse: np.ndarray

    def __post_init__(self) -> None:
        self.forward = self.forward.copy()
        self.inverse = self.inverse.copy()
        self.cost = int(self.forward.nbytes + self.inverse.nbytes)

    def undo(self, doc: Any) -> None:
        doc.apply_remap(self.inverse)

    def redo(self, doc: Any) -> None:
        doc.apply_remap(self.forward)


@dataclass
class ColorStateEdit(Edit):
    """Colour mode, palette and transparent index, before and after.

    :class:`PaletteEdit` generalised to everything a document's colour state
    holds, and both types stay: ``PaletteEdit`` remains what a
    palette-constrained RGB op pushes, so that path's steps are byte-for-byte
    what they were, and this is what the indexed ops push.

    Each side is ``(mode, palette, transparent)``. Restoring all three together
    is the point -- the three are one fact. A mode restored without its palette
    is a document declaring itself indexed with nothing to index onto, and a
    transparent index restored without its mode moves the hole in a document
    that has none.

    Applying it **re-materialises**, which is what makes an instant palette
    recolour instant in both directions: the indices never moved, so putting the
    old table back is the whole undo.
    """

    before: tuple[str, list[Any] | None, int]
    after: tuple[str, list[Any] | None, int]

    def __post_init__(self) -> None:
        self.before = _copy_state(self.before)
        self.after = _copy_state(self.after)
        # ``PaletteEdit``'s reckoning: bookkeeping rather than a figure eviction
        # will act on, counted anyway because an edit reporting zero is one the
        # budget cannot see at all.
        self.cost = 16 * sum(len(state[1] or ()) for state in (self.before, self.after))

    def _apply(self, doc: Any, state: tuple[str, list[Any] | None, int]) -> None:
        mode, table, transparent = state
        doc.color_mode = mode
        doc.palette = _copy_table(table)
        doc.transparent_index = int(transparent)
        doc.rev += 1
        # After the assignment, never before: the materialisation reads the
        # table off the document.
        doc._rematerialize_all()

    def undo(self, doc: Any) -> None:
        self._apply(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._apply(doc, self.after)


def _copy_state(state: tuple[str, list[Any] | None, int]) -> tuple[str, list[Any] | None, int]:
    mode, table, transparent = state
    return (str(mode), _copy_table(table), int(transparent))


@dataclass
class LayerAddEdit(Edit):
    """The layer object itself is held, not a copy of it: re-inserting the same
    object is what keeps its uid, and keeping its uid is what lets a patch made
    before the undo still apply after the redo."""

    index: int
    layer: Any

    def __post_init__(self) -> None:
        # An undone add holds the *only* reference to a full-canvas layer, so
        # a cost of zero made it invisible to the byte budget: a document could
        # pin an unbounded number of them past the 192 MiB ceiling. Same
        # measurement LayerRemoveEdit already made.
        self.cost = _plane_bytes(self.layer)

    def undo(self, doc: Any) -> None:
        doc.stack.remove(doc.stack.index_of(self.layer.uid))
        doc.invalidate_all()

    def redo(self, doc: Any) -> None:
        doc.stack.insert(self.index, self.layer)
        doc.invalidate_all()


@dataclass
class LayerRemoveEdit(Edit):
    index: int
    layer: Any

    def __post_init__(self) -> None:
        self.cost = _plane_bytes(self.layer)

    def undo(self, doc: Any) -> None:
        doc.stack.insert(self.index, self.layer)
        doc.invalidate_all()

    def redo(self, doc: Any) -> None:
        doc.stack.remove(doc.stack.index_of(self.layer.uid))
        doc.invalidate_all()


@dataclass
class LayerMoveEdit(Edit):
    layer_uid: int
    index: int
    to: int

    def undo(self, doc: Any) -> None:
        doc.stack.move(doc.stack.index_of(self.layer_uid), self.index)
        doc.invalidate_all()

    def redo(self, doc: Any) -> None:
        doc.stack.move(doc.stack.index_of(self.layer_uid), self.to)
        doc.invalidate_all()


@dataclass
class LayerPropsEdit(Edit):
    """Name, opacity, visibility, blend mode -- anything that is not pixels."""

    layer_uid: int
    before: dict
    after: dict

    def _apply(self, doc: Any, props: dict) -> None:
        layer = doc.stack.by_uid(self.layer_uid)
        for key, value in props.items():
            setattr(layer, key, value)
        doc.invalidate_all()

    def undo(self, doc: Any) -> None:
        self._apply(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._apply(doc, self.after)


@dataclass
class LayerFlagEdit(Edit):
    """``background`` and ``reference``: the two flags no props edit can carry.

    They are real persisted document state -- ``.aseprite`` stores them as the
    layer-chunk flags ``0x08`` and ``0x40``, ``.ora`` as ``warlock-background``
    and ``warlock-reference`` -- but they are deliberately outside
    ``TRACK_PROPS``, because that allowlist is what ``set_range_props`` writes
    through and "make every selected row a background" is not an operation:
    the bottom layer is the only one that may be one, which is the rule
    ``LayerStack`` enforces on every reorder.

    So they get their own type rather than a widened allowlist. It writes both
    the track (authoritative on an animated document) and the materialised
    layer under one shared uid, which is the same identity by construction --
    see ``Track.of``.
    """

    layer_uid: int
    before: dict
    after: dict

    def undo(self, doc: Any) -> None:
        doc._set_layer_flags(self.layer_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._set_layer_flags(self.layer_uid, self.after)


@dataclass
class MatteEdit(Edit):
    """The document's matte colour, before and after.

    ``Document.matte`` is what a flatten composites *under* the stack where
    there is no background layer, so converting the bottom layer to a
    background folds it into pixels and clears it. The pixels come back on an
    undo through the ``PatchEdit`` that rides beside this one; without this the
    colour did not, and it was unrecoverable -- nothing else in the document
    ever held it.

    A tuple of four ints. Cost is zero for ``TagsEdit``'s reason: there are no
    pixels here for the byte budget to have an opinion about.
    """

    before: tuple | None
    after: tuple | None

    def _apply(self, doc: Any, matte: tuple | None) -> None:
        doc.matte = None if matte is None else tuple(matte)
        doc.invalidate_all()

    def undo(self, doc: Any) -> None:
        self._apply(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._apply(doc, self.after)


@dataclass
class PaletteEdit(Edit):
    """The document's colour *table*, before and after. A few dozen tuples.

    The pixels a palette op rewrites are already covered by the ``ReplayEdit``
    it is bundled with, and the table was covered by nothing at all: an undo put
    the old colours back into a document still claiming the new palette, so the
    very next stroke snapped them straight back to the table the user had just
    undone. The two are one step because they are one act -- ``_doc_indexed``
    pushes ``CompoundEdit([PaletteEdit, ReplayEdit])``, so the table is restored
    after the pixels on the way back and assigned before them on the way
    forward.

    The list is copied on the way in and on the way out. It is the live
    ``Document.palette`` object at both ends otherwise, and a later ``add_slot``
    appending to it would edit the history's idea of what came before.
    """

    before: list[Any] | None
    after: list[Any] | None

    def __post_init__(self) -> None:
        self.before = _copy_table(self.before)
        self.after = _copy_table(self.after)
        # Bytes, nominally: a 256-entry table of 4-tuples is a few kilobytes and
        # the budget is in mebibytes, so this is bookkeeping rather than a
        # figure eviction will ever act on. Counted anyway, because an edit that
        # reports zero is one the budget cannot see at all.
        self.cost = 16 * sum(len(t) for t in (self.before, self.after) if t)

    def _apply(self, doc: Any, table: list[Any] | None) -> None:
        doc.palette = _copy_table(table)
        # The composite does not depend on the table, so there is nothing to
        # recomposite -- but ``rev`` is what the palette pane's usage count is
        # keyed on, and a count taken against the other table must not survive
        # the swap looking current.
        doc.rev += 1

    def undo(self, doc: Any) -> None:
        self._apply(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._apply(doc, self.after)


@dataclass
class FramePaletteEdit(Edit):
    """One frame's own colour table, before and after. ``None`` is "no override".

    :class:`PaletteEdit` one axis over, and the same two decisions: the tables
    are copied at both ends (the live list is the document's own object
    otherwise, and a later ``recolour_slot`` would edit the history's idea of
    what came before), and the cost is bookkeeping rather than a figure
    eviction will act on.

    Addressed by **frame uid**, like every other edit in this package, so a
    frame inserted or moved between the edit and its undo cannot make the table
    land on a different frame.
    """

    frame_uid: int
    before: list[Any] | None
    after: list[Any] | None

    def __post_init__(self) -> None:
        self.before = _copy_table(self.before)
        self.after = _copy_table(self.after)
        self.cost = 16 * sum(len(t) for t in (self.before, self.after) if t)

    def _apply(self, doc: Any, table: list[Any] | None) -> None:
        anim = getattr(doc, "anim", None)
        if anim is None:  # pragma: no cover - a still document has no frames
            return
        if table is None:
            anim.frame_palettes.pop(self.frame_uid, None)
        else:
            anim.frame_palettes[self.frame_uid] = _copy_table(table)
        # Unlike the document table this *does* reach the picture: on an
        # indexed document the frame's pixels are materialised through it, so
        # the whole document is restamped rather than merely revved.
        doc._frame_palettes_changed()

    def undo(self, doc: Any) -> None:
        self._apply(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._apply(doc, self.after)


def _copy_table(table: list[Any] | None) -> list[Any] | None:
    return None if table is None else [tuple(colour) for colour in table]


def _pack(mask: np.ndarray | None) -> tuple[bytes, tuple[int, int]] | None:
    if mask is None:
        return None
    return zlib.compress(mask.tobytes(), 1), mask.shape


def _unpack(blob: tuple[bytes, tuple[int, int]] | None) -> np.ndarray | None:
    if blob is None:
        return None
    data, shape = blob
    return np.frombuffer(zlib.decompress(data), dtype=np.uint8).reshape(shape).copy()


@dataclass
class SelectionEdit(Edit):
    """A selection change. Masks compress to almost nothing -- they are mostly
    runs of 0 and 255 -- which is the only reason selection changes can afford
    to be undoable at all."""

    before: Any
    after: Any

    def __post_init__(self) -> None:
        self.before = _pack(self.before)
        self.after = _pack(self.after)
        self.cost = sum(len(b[0]) for b in (self.before, self.after) if b)

    def undo(self, doc: Any) -> None:
        doc.set_selection_mask(_unpack(self.before))

    def redo(self, doc: Any) -> None:
        doc.set_selection_mask(_unpack(self.after))


@dataclass
class ReplayEdit(Edit):
    """A whole-canvas operation: flip, rotate, scale, crop, canvas resize.

    Undo restores copied layers (there is no smaller truthful answer when the
    canvas size itself changed); redo *replays* the operation rather than
    holding a second full copy, which halves the cost of the one edit type that
    is expensive. Replay is safe here and nowhere else because these ops are
    pure functions of the document -- no accumulation, no randomness.
    """

    snapshot: list[Any]
    size: tuple[int, int]
    active: int
    replay: Callable[[Any], None]
    selection: Any = None
    #: The animation grid as it stood, or None on a still document. Trailing
    #: and defaulted, so every existing construction site is unchanged. It
    #: carries the *slot* structure, without which an undo would restore two
    #: equal copies where the document had one shared object and silently
    #: break every link in it.
    grid: Any = None
    #: The document's slices as they stood, or None on a document that has
    #: none. Trailing and defaulted, so every existing construction site is
    #: unchanged. They are carried rather than re-derived because a whole-canvas
    #: op *moves* them -- a crop that clamped a slice to 1x1 cannot be reversed
    #: by any arithmetic on the result.
    slices: Any = None
    #: The layer-group tree as it stood, as ``(nodes, membership)``. Trailing
    #: and defaulted like ``grid`` above, and taken unconditionally rather than
    #: only by the one op that rewrites it (``flatten_layers`` replaces every
    #: layer in the document, so nothing is left for a group to hold): a rotate
    #: that quietly kept a tree describing layers it had just replaced is a much
    #: harder bug than a dictionary copy is a cost.
    groups: Any = None

    def __post_init__(self) -> None:
        self.selection = _pack(self.selection)
        self.cost = sum(_plane_bytes(layer) for layer in self.snapshot)

    def undo(self, doc: Any) -> None:
        # Before the snapshot, so the ``invalidate_all`` that ends
        # ``restore_snapshot`` folds the tree that is coming back rather than
        # the one being replaced. Skipped outright when no tree was recorded,
        # which keeps a step built without one -- there is no such caller in
        # the app, but there is in the tests -- from needing the hook at all.
        if self.groups is not None:
            doc.restore_groups(self.groups)
        doc.restore_snapshot(self.snapshot, self.size, self.active, self.grid, self.slices)
        doc.set_selection_mask(_unpack(self.selection))

    def redo(self, doc: Any) -> None:
        self.replay(doc)
