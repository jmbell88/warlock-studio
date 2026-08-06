"""The undo engine, with no opinion about what an edit edits.

``Edit``, ``CompoundEdit`` and ``UndoStack`` were written for the raster editor
and lived in ``studio/inker/undo.py``, but nothing in them is about pixels: an
edit is a pair of ``undo``/``redo`` callbacks and a ``cost``, and the stack is a
pair of lists bounded by bytes. The raster editor's own edit types -- patches,
layer add/remove/move/props, selections, whole-canvas replays -- stay where they
are, because *those* are about pixels and layers. What is here is the part Clay
mode needs too, with a ``MeshEdit`` in place of a ``PatchEdit``; a pure
``studio/build/`` package that reached into the raster editor for its history
would be depending on the raster editor for no reason other than where the file
happened to sit.

Two rules travel with the engine rather than with either user of it, and both
are as load-bearing for a mesh as for a bitmap.

**An edit addresses its subject by uid, never by index.** An undo issued after a
reorder must still land on the thing the edit was made to -- the layer, or the
primitive -- and an index stopped naming that thing the moment anything moved.

**An edit owns its data.** ``cost`` is what eviction is driven by, and a slice
of a larger buffer reports only its own ``nbytes`` while keeping the whole base
alive: a step that costs four kilobytes by the budget's reckoning can pin
sixteen megabytes, invisibly. A numpy view of a full-canvas layer and a view of
a shared vertex array are the same trap, so every edit type that stores array
data copies it when it is a view.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

# The budget is bytes, because the cost of a step now varies by three orders of
# magnitude between a dab and a crop. The depth bounds are the other half: a
# huge document must still get a few levels, and a document of dabs must not
# grow an unbounded stack of them.
UNDO_BYTES = 192 * 1024 * 1024
UNDO_MAX_DEPTH = 64
UNDO_MIN_DEPTH = 8


# Serial numbers handed out at push time. They exist so a caller can ask "is
# the document where it was when I saved it" -- ``rev`` cannot answer that,
# because it counts *changes* and an undo is a change, so undoing back to the
# saved pixels would still read as unsaved.
_serials = itertools.count(1)


class Edit:
    """One reversible step. ``cost`` is what the budget is spent on."""

    cost: int = 0
    serial: int = 0

    def undo(self, doc: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def redo(self, doc: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class CompoundEdit(Edit):
    """Several edits that must undo together -- committing a floating buffer
    writes pixels *and* drops the selection, and splitting that across two
    Ctrl+Z presses shows the user a state that never existed."""

    edits: list[Edit] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cost = sum(e.cost for e in self.edits)

    def undo(self, doc: Any) -> None:
        for edit in reversed(self.edits):
            edit.undo(doc)

    def redo(self, doc: Any) -> None:
        for edit in self.edits:
            edit.redo(doc)


class UndoStack:
    """Bounded by bytes first, then by a depth floor and a depth ceiling."""

    def __init__(self, budget: int = UNDO_BYTES) -> None:
        self.budget = budget
        self._done: list[Edit] = []
        self._undone: list[Edit] = []

    def __len__(self) -> int:
        return len(self._done)

    @property
    def can_undo(self) -> bool:
        return bool(self._done)

    @property
    def can_redo(self) -> bool:
        return bool(self._undone)

    @property
    def bytes(self) -> int:
        return sum(e.cost for e in self._done) + sum(e.cost for e in self._undone)

    @property
    def top(self) -> Edit | None:
        """The step an ``undo`` would reverse, without reversing it.

        A caller that has to reconcile *its own* non-undoable state against a
        step -- Clay drops an element selection when the geometry under it
        changes -- has to know what the step was, and asking afterwards is too
        late because the stack has already moved it.
        """
        return self._done[-1] if self._done else None

    @property
    def redo_top(self) -> Edit | None:
        """The step a ``redo`` would replay. The counterpart of :attr:`top`."""
        return self._undone[-1] if self._undone else None

    @property
    def head(self) -> int:
        """Identifies the current position in history.

        A save records this and compares against it later, which is what makes
        "dirty" a comparison rather than a latching flag: undo back to the
        saved step and the head is that step's serial again. Serials are
        per-edit, so a *new* branch of the same length is a different head --
        which a plain depth count would miss.
        """
        return self._done[-1].serial if self._done else 0

    def push(self, edit: Edit) -> None:
        edit.serial = next(_serials)
        self._done.append(edit)
        # No history tree: a redo onto a branch the user did not take is worse
        # than no redo at all.
        self._undone.clear()
        self._evict()

    def _evict(self) -> None:
        while len(self._done) > UNDO_MAX_DEPTH:
            self._done.pop(0)
        while len(self._done) > UNDO_MIN_DEPTH and self.bytes > self.budget:
            self._done.pop(0)

    def undo(self, doc: Any, *, redoable: bool = True) -> bool:
        """Reverse the newest step.

        ``redoable=False`` reverses it and forgets it. That is what cancelling
        a lift needs: the lift's own step is how the pixels get put back
        exactly, but the user asked for the lift to *not have happened*, and
        leaving it on the redo stack lets Ctrl+Y replay the alpha-cut with no
        floating buffer left to restore -- which erases the region outright.
        """
        if not self._done:
            return False
        edit = self._done.pop()
        edit.undo(doc)
        if redoable:
            self._undone.append(edit)
        return True

    def revoke(self, doc: Any, edit: Edit) -> bool:
        """Reverse and forget one *named* step, wherever it sits in the stack.

        ``undo(redoable=False)`` reverses whichever step is newest, which is
        only the same thing when nothing has been pushed since. Cancelling a
        lift cannot assume that: the buffer floats across an unbounded number
        of frames and every selection op pushes a step of its own, so the lift
        was routinely no longer on top -- and reversing the top instead dropped
        the lifted pixels and left the alpha-cut, unrecoverably.

        Reversing out of order is sound here because the only steps that can be
        pushed over a floating buffer are selection and layer-property edits:
        everything that writes pixels calls ``commit_floating`` first, so
        nothing above this edit describes the region it restores.

        Returns False when the edit has already been evicted or undone, in
        which case there is nothing to put back and nothing to corrupt.
        """
        # By identity, not by ``list.remove``: edits are dataclasses, so ``==``
        # compares fields, and a PatchEdit's fields are numpy arrays whose
        # comparison is an array rather than a bool.
        for i, candidate in enumerate(self._done):
            if candidate is edit:
                del self._done[i]
                edit.undo(doc)
                return True
        return False

    def forget_redo(self) -> None:
        """Drop the redo branch without recording a step.

        For an action that changes what the user sees but has no undo entry of
        its own -- a paste, which only floats pixels -- and so cannot rely on
        ``push`` to clear the branch it just diverged from.
        """
        self._undone.clear()

    def redo(self, doc: Any) -> bool:
        if not self._undone:
            return False
        edit = self._undone.pop()
        edit.redo(doc)
        self._done.append(edit)
        return True

    def drop(self) -> None:
        """Discard the newest step without reversing it -- for an operation
        that already undid itself by hand, where leaving the step on the stack
        would make the next Ctrl+Z appear to do nothing."""
        if self._done:
            self._done.pop()

    def clear(self) -> None:
        self._done.clear()
        self._undone.clear()
