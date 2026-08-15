"""Indexed colour: the palette, and the ops that edit it.

A palette here is a *constraint on writes* rather than an index plane -- see
:mod:`.indexed` for why -- so everything below is either bookkeeping on the
table or a whole-document rewrite pushed through ``_replay``. The snap itself
lives in ``_commit_patch``, which is the single place every undoable write
passes through.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from . import dither
from . import indexed as ix
from .undo import CompoundEdit, PaletteEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import RGBA, Document


class IndexedOps:
    """Palette state and palette edits, mixed into :class:`~.document.Document`."""

    def _palette_step(self: Document, before: list[RGBA] | None, run: Any) -> None:
        """Rewrite the pixels *and* record the table, as one undo step.

        ``_replay`` records the pixels and pushes its own ``ReplayEdit``; the
        table is not pixels and nothing else was putting it back. So the step it
        pushed is taken straight back off (``drop`` -- it is reversed by nothing
        and reversed by nobody, it has only just been made) and re-pushed inside
        a compound with a :class:`~.undo.PaletteEdit` in front of it.

        Every caller assigns ``self.palette`` *before* calling, for the reason
        ``set_palette`` always did: ``run`` is the raw work and redo re-runs it,
        so the table it snaps against has to be the document's by then.
        """
        after = self.palette
        self._replay(run)
        replayed = self.history.top
        if replayed is None:  # pragma: no cover - the push was one line ago
            return
        self.history.drop()
        self.history.push(CompoundEdit([PaletteEdit(before, after), replayed]))

    @property
    def is_indexed(self: Document) -> bool:
        """Whether writes are constrained to a palette. See :mod:`.indexed`."""
        return bool(self.palette)

    def set_palette(self: Document, colours: Sequence[RGBA] | None, *, snap: bool = True) -> bool:
        """Adopt a colour table, snapping the document onto it. -> whether it
        changed anything.

        ``None`` leaves indexed mode: the pixels stay exactly as they are, which
        is the only honest answer -- the colours in the document *are* the ones
        that were painted, and there is nothing to restore them from.

        The snap is a whole-document rewrite, so it goes through ``_replay``
        like a rotate rather than through a patch per layer. That is not merely
        cheaper (redo replays instead of storing a second full copy): it is what
        makes the operation *one* undo step across every layer and every frame,
        which is what the user did.
        """
        wanted = None if not colours else [tuple(c) for c in colours]
        if wanted == self.palette:
            return False
        if wanted is None or not snap:
            self.palette = wanted
            self.rev += 1
            return True
        return self.convert_to_palette(wanted, "nearest")

    def convert_to_palette(
        self: Document, colours: Sequence[RGBA], method: str = "nearest"
    ) -> bool:
        """Adopt a table and rewrite every pixel onto it by *method*.

        The general form of :meth:`set_palette`, which is now its
        ``method="nearest"`` case -- one path, so a document converted with a
        dither and a document simply indexed differ in the arithmetic and in
        nothing else about how the step is recorded.

        **The conversion is whole-document, and a selection is ignored.** This
        is a change of *mode*: the table it installs constrains every write
        afterwards, everywhere, so a version of it that converted only the
        marquee would leave the pixels outside it off the palette they are now
        declared to be on -- a state the mode cannot describe and the next
        stroke anywhere would start silently repairing. Aseprite ignores the
        selection here for the same reason.

        One undo step across every layer and every frame, because that is what
        the user did. Links survive it: ``_replay`` snapshots the *grid*, so two
        frames holding one cel hold one cel again after an undo.
        """
        if not colours:
            raise ValueError("a conversion needs at least one colour")
        wanted = [tuple(c) for c in colours]
        if wanted == self.palette and method == "nearest":
            # Already exactly this table, and nearest is idempotent on a
            # document that is already snapped onto it -- so there is no step to
            # push and nothing for it to hold. The dithers are *not* let through
            # this door: re-running one is a real request (it is how a user
            # compares two matrices) and it does move pixels.
            return False
        self.commit_floating()
        before, self.palette = self.palette, wanted
        self._palette_step(
            before,
            lambda: self._map_planes(
                lambda plane: dither.convert(plane, wanted, method), mask_fn=None
            ),
        )
        return True

    def palette_from_document(
        self: Document, max_colours: int = 32, method: str = "nearest"
    ) -> bool:
        """Build a table out of the document's own pixels and convert onto it.

        The entry point for "make this drawing indexed" when the user has no
        palette in mind. It is two published operations and no third one:
        :func:`dither.build_palette` over every distinct cel, then
        :meth:`convert_to_palette` -- so the resulting document is
        indistinguishable from one indexed to a hand-authored table.
        """
        colours = dither.build_palette(self._palette_planes(), max_colours)
        return self.convert_to_palette(colours, method)

    def _palette_planes(self: Document) -> list[Any]:
        """Every distinct pixel plane in the document, once each.

        ``unique_cel_layers`` and not the stack: a background linked across
        three frames is one plane, and counting it three times would weight the
        median cut by how many frames a cel happens to appear on.
        """
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        return [layer.pixels for layer in layers]

    def add_slot(self: Document, colour: RGBA) -> bool:
        """Append a colour to the palette. Nothing is repainted: a new swatch
        is a colour the user may now paint *with*, not a claim about what is
        already on the canvas."""
        if not self.palette:
            return self.set_palette([colour])
        if tuple(colour) in self.palette:
            return False
        self.palette = [*self.palette, tuple(colour)]
        self.rev += 1
        return True

    def recolour_slot(self: Document, index: int, colour: RGBA) -> bool:
        """Change one swatch, rewriting every pixel painted in it.

        This is the payoff of carrying a palette at all: editing a slot is a
        recolour of the whole clip, addressed by *colour* rather than by
        selection, and it is one Ctrl+Z. Exact-match, never nearest -- see
        ``indexed.remap`` -- so the swatch beside it is not dragged along.
        """
        if not self.palette or not 0 <= index < len(self.palette):
            return False
        old = self.palette[index]
        new = tuple(colour)
        if old == new:
            return False
        table = [*self.palette]
        table[index] = new
        self.commit_floating()
        before, self.palette = self.palette, table
        self._palette_step(
            before,
            lambda: self._map_planes(
                lambda plane: ix.remap(plane, old, new), mask_fn=None
            ),
        )
        return True

    def remove_slot(self: Document, index: int) -> bool:
        """Drop a swatch, merging its pixels into the nearest survivor.

        Merging rather than erasing, and rather than refusing: the pixels are
        the picture, and a palette edit is a statement about the *table*. The
        last swatch cannot go -- an indexed document with no colours is one
        every visible pixel would have to snap to nothing.
        """
        if not self.palette or not 0 <= index < len(self.palette):
            return False
        if len(self.palette) == 1:
            return False
        gone = self.palette[index]
        table = [c for i, c in enumerate(self.palette) if i != index]
        self.commit_floating()
        before, self.palette = self.palette, table
        into = table[ix.nearest(gone, table)]
        self._palette_step(
            before,
            lambda: self._map_planes(
                lambda plane: ix.remap(plane, gone, into), mask_fn=None
            ),
        )
        return True

    def move_slot(self: Document, index: int, to: int) -> bool:
        """Reorder the table. No pixel changes -- order is presentation here,
        and the exported ``.gpl`` and GIF colour table are what it decides."""
        if not self.palette or not 0 <= index < len(self.palette):
            return False
        to = max(0, min(to, len(self.palette) - 1))
        if to == index:
            return False
        table = [*self.palette]
        table.insert(to, table.pop(index))
        self.palette = table
        self.rev += 1
        return True

    def palette_usage(self: Document) -> list[int]:
        """Visible pixels sitting exactly on each swatch, over the whole
        document. What tells the user which slots are safe to delete."""
        if not self.palette:
            return []
        planes = self.stack if self.anim is None else self.anim.unique_cel_layers()
        totals = [0] * len(self.palette)
        for layer in planes:
            for slot, count in enumerate(ix.histogram(layer.pixels, self.palette)):
                totals[slot] += count
        return totals
