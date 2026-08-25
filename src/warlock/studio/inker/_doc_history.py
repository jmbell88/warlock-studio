"""Undo, redo, and the two hooks the edits call back into.

The document is the only thing that pushes onto the undo stack, and this is the
half that *reverses* it: ``undo``/``redo`` are the user-facing pair, and
``restore_snapshot``/``set_selection_mask`` are the hooks ``ReplayEdit`` and
``SelectionEdit`` reach for when the stack unwinds.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from .layers import Layer, LayerStack
from .selection import SelectionMask

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document


class HistoryOps:
    """Undo/redo and its callbacks, mixed into :class:`~.document.Document`."""

    @contextmanager
    def one_gesture(self: Document) -> Any:
        """Everything pushed inside the block becomes a single undo step.

        The composed ops -- delete these eight rows, merge this run -- are each
        built out of ops that already push their own step, and eight Ctrl+Z to
        reverse one click is exactly what ``set_layers_props`` exists to stop
        one level down. This is that rule for a *sequence* rather than for a
        property write.

        The collapse happens on the way out even if the body raises, because a
        half-applied gesture is still one gesture: leaving its steps loose would
        make the user walk back through the middle of it.
        """
        depth = len(self.history)
        try:
            yield
        finally:
            self.history.collapse_since(depth)

    def undo(self: Document) -> bool:
        """One Ctrl+Z is one step -- and cancelling a float *is* that step.

        Cancelling a lift reverses the lift's own history entry, so falling
        through to ``history.undo`` afterwards would spend a second step on a
        single keypress.
        """
        if self.cancel_floating():
            return True
        return self.history.undo(self)

    def redo(self: Document) -> bool:
        self.cancel_floating()
        return self.history.redo(self)

    def restore_snapshot(
        self: Document,
        layers: list[Layer],
        size: tuple[int, int],
        active: int,
        grid: dict[str, Any] | None = None,
        slices: list[Any] | None = None,
    ) -> None:
        """Undo hook for a whole-canvas operation.

        The layers are copied (the snapshot is held for as long as the edit is,
        and a later stroke must not write into it) but they keep their uids: an
        undo restores the document's *state*, not a set of new layers, and
        every patch recorded before this one addresses those uids.

        ``grid`` is the animated form of the same argument one level up. The
        copies are made once and shared back into the slots by index, so two
        frames that held one object hold one object again.

        ``slices`` is the same idea for the document's named rectangles, and it
        is copied here for the same reason the layers are: the step stays on the
        stack and a later drag must not write into what it restores. None means
        "this step predates slices", which is every step a still ``ReplayEdit``
        constructed positionally still produces.

        A snapshot that carries a grid onto a document that is no longer
        animated is **refused**, not silently flattened. The two are unreachable
        past each other through the ordinary stack -- de-animating is itself an
        ``AnimateEdit`` and sits above any replay it followed, so LIFO undo
        re-animates first -- which is exactly why the case deserves a raise
        rather than a fallback: reaching it means an assumption elsewhere has
        already broken, and the old branch answered by dropping every frame but
        one and every link between them, with nothing said and nothing to undo.
        """
        if (grid is None) != (self.anim is None):
            raise ValueError(
                "this undo step describes a "
                f"{'still' if grid is None else 'animated'} document and this "
                f"one is {'still' if self.anim is None else 'animated'}"
            )
        copies = [layer.copy(uid=layer.uid) for layer in layers]
        if slices is not None:
            self.slices = [entry.copy() for entry in slices]
        width, height = size
        if grid is None:
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

    def set_selection_mask(self: Document, mask: np.ndarray | None) -> None:
        """Undo hook for a selection change."""
        self.mask = None if mask is None else SelectionMask(mask)
        self.rev += 1
