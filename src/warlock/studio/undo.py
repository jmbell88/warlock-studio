"""The undo engine, with no opinion about what an edit edits.

``Edit``, ``CompoundEdit`` and ``UndoStack`` were written for the raster editor
and lived in ``studio/inker/undo.py``, but nothing in them is about pixels: an
edit is a pair of ``undo``/``redo`` callbacks and a ``cost``, and the stack is a
pair of lists bounded by bytes. The raster editor's own edit types -- patches,
layer add/remove/move/props, selections, whole-canvas replays -- stay where they
are, because *those* are about pixels and layers. What is here is the part Clay
mode needs too, with a ``MeshEdit`` in place of a ``PatchEdit``; a pure
``studio/clay/`` package that reached into the raster editor for its history
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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

# The budget is bytes, because the cost of a step now varies by three orders of
# magnitude between a dab and a crop. The depth bounds are the other half: a
# huge document must still get a few levels, and a document of dabs must not
# grow an unbounded stack of them.
UNDO_BYTES = 192 * 1024 * 1024
UNDO_MAX_DEPTH = 64
UNDO_MIN_DEPTH = 8

# The depth floor above is a floor on *steps*, and it was sized when the
# largest imaginable step was one layer's snapshot. It is not: a canvas resize
# or a rotate on a 4096-square document with thirty layers snapshots all of
# them -- ~1.9 GiB in a single step -- and eight of those are just over 15 GiB,
# reached without the byte budget ever getting a vote, because
# ``len(self._done) > UNDO_MIN_DEPTH`` stays false the whole way up. What that
# produces is an out-of-memory kill holding an unsaved document, which loses
# the painting *and* everything the two-minute journal has not written yet:
# the most expensive way this app can fail, from an ordinary gesture.
#
# So the floor is a floor only while the stack can afford it. Past this ceiling
# depth loses and eviction runs down to a single step -- a one-step history is
# still an undo, and a process that is killed is not.
#
# Four times the soft budget, which is what it takes to keep the floor's own
# worst *intended* case -- eight snapshots of one 4096-square RGBA layer, 512
# MiB -- while refusing the case it was never sized for. One step can still
# exceed this on its own and nothing here can help that: the guarantee is
# "one step, or under the ceiling", never both.
UNDO_HARD_BYTES = 4 * UNDO_BYTES


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


def _label(edit: Any) -> str:
    """An edit's class name as words: ``LayerAddEdit`` -> "layer add"."""
    import re

    name = type(edit).__name__.removesuffix("Edit")
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower() or "step"


class UndoStack:
    """Bounded by bytes first, then by a depth floor and a depth ceiling.

    The floor yields to :data:`UNDO_HARD_BYTES` -- see that constant for why a
    stack that is small in steps can still be too big to keep.
    """

    def __init__(self, budget: int = UNDO_BYTES, *, hard: int = UNDO_HARD_BYTES) -> None:
        self.budget = budget
        self.hard = hard
        # Steps dropped because the stack was too *big*, ever, on this stack.
        # Depth-cap evictions are deliberately not counted: a session that runs
        # past UNDO_MAX_DEPTH is working normally, and saying so every time is
        # noise.
        #
        # A counter rather than a callback, because this module imports nothing
        # and two headless packages depend on that. Whoever can reach a status
        # bar compares it against what it last saw -- with ``!=`` and not
        # ``>``, since ``clear`` puts it back to zero along with the history.
        self.trimmed = 0
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

    def edits(self) -> Iterator[Edit]:
        """Every step the stack is holding, done and undone, compounds opened.

        The stack has no opinion about what an edit edits, and this does not
        give it one -- it hands back opaque ``Edit``s and the caller decides
        which of them it recognises. What it is *for* is the one thing an owner
        of the document cannot otherwise see: a step holds objects that are in
        no document (an undone add, a done remove), and a document-wide
        renumbering that walked only the live objects would leave those holding
        numbers from before it. Clay's palette shift is that caller.

        A ``CompoundEdit`` yields its children rather than itself, recursively,
        because whether an edit was bundled with another is a fact about how it
        was pushed and never about what it holds.
        """

        def walk(edit: Edit) -> Iterator[Edit]:
            if isinstance(edit, CompoundEdit):
                for child in edit.edits:
                    yield from walk(child)
            else:
                yield edit

        for edit in (*self._done, *self._undone):
            yield from walk(edit)

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

    def collapse_since(self, depth: int) -> bool:
        """Fold every step pushed since ``depth`` into a single one.

        The alternative to threading a list of edits through half a dozen ops
        that each already know how to push their own. A caller records
        ``len(stack)``, does whatever it does, and asks for the run to become
        one gesture -- which is the rule the rest of this package follows and
        could not follow for a *composed* op like "delete these eight rows".

        Nothing to fold (an empty run, or a single step) is ``False`` and no
        change: a lone ``CompoundEdit`` around one edit would read as
        "compound" in the history panel where the edit reads as what it did,
        which is ``one_step``'s argument one level up.
        """
        if depth < 0 or len(self._done) - depth < 2:
            return False
        tail = self._done[depth:]
        del self._done[depth:]
        self.push(CompoundEdit(tail))
        return True

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
            self.trimmed += 1
        # And here the floor loses. Every loop above stops at UNDO_MIN_DEPTH;
        # this one stops at one step. See UNDO_HARD_BYTES for why.
        while len(self._done) > 1 and self.bytes > self.hard:
            self._done.pop(0)
            self.trimmed += 1

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

    def history(self) -> list[tuple[str, bool]]:
        """Every step, oldest first, as ``(label, is_done)``. For the panel.

        **Read-only and derived**: the panel it feeds shows what the stack
        holds and asks the stack to move, rather than holding a list of its
        own -- which is the way an undo panel goes wrong, by drifting from the
        stack it claims to picture after a step is evicted by the byte budget.

        The label is the edit's class name turned into words, because an
        ``Edit`` has no name of its own and giving every one of the twenty a
        string is twenty places for a rename to be missed. ``PatchEdit`` reads
        as "patch", which is what an undo of one is.
        """
        out = [(_label(edit), True) for edit in self._done]
        # The undone half is newest-first on its own stack; reversed here so
        # the whole list reads as one timeline in the order it happened.
        out.extend((_label(edit), False) for edit in reversed(self._undone))
        return out

    def step_to(self, doc: Any, index: int) -> bool:
        """Move the head to *index* (the count of done steps). -> whether it moved.

        Undo or redo, repeatedly, through the two methods that already exist --
        never by reaching into either list. Jumping is not a third operation on
        the stack, it is the one the panel makes easy, and an implementation
        that spliced the lists would be a second definition of what a step is.
        """
        wanted = max(0, min(int(index), len(self._done) + len(self._undone)))
        moved = False
        while len(self._done) > wanted and self.undo(doc):
            moved = True
        while len(self._done) < wanted and self.redo(doc):
            moved = True
        return moved

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

        **That precondition is the caller's, and this class cannot check it.**
        The stack holds opaque ``Edit``s and has no idea which of them touch
        pixels; the guarantee comes from ``Document`` calling
        ``commit_floating`` at the top of every mutating method, which
        ``tests/inker/test_regressions.py`` pins from the outside. A future
        caller that revokes a step with pixel writes above it gets a corrupt
        document and no complaint from here.

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
        # With the history rather than alongside it: a stack that reports steps
        # trimmed from a history it no longer holds is describing nothing.
        self.trimmed = 0
