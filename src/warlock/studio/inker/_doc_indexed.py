"""Colour modes: the palette, the index planes, and the ops that edit them.

**Two mechanisms live here, and both are deliberate.**

*Palette-constrained RGB* (``color_mode == "rgb"`` with a palette set) is what
this module used to be all of: the pixels stay RGBA and every write is snapped
onto the table as it commits -- see :mod:`.indexed` for the original argument.
It is kept unchanged rather than replaced, because documents written before
true indexing existed legally contain soft alpha, which one index per pixel
cannot represent. Every op below takes exactly the steps it always took in that
mode, and the ORA bytes of such a document are unchanged.

*True indexed* (``color_mode == "indexed"``) stores an index plane per layer and
materialises the pixels from it. What that buys is **identity**: two slots
holding the same colour stay two slots, so a recolour reaches the pixels of one
and not the other, a reorder is an exact permutation, and a ``.aseprite`` file
with a duplicate swatch round-trips. See :mod:`.index_plane`.

The pattern every op follows is the same in both: assign the colour state, then
push it in front of whatever restores the pixels. Which of the three shapes that
takes depends only on what the op does to the *planes* --

===================================  =========================================
op                                   step
===================================  =========================================
recolour / add / ramp / transparent  ``ColorStateEdit`` alone (no plane moves)
move / sort                          ``[ColorStateEdit, IndexRemapEdit]``
remove (merge), mode conversions     ``[ColorStateEdit, ReplayEdit]``
===================================  =========================================

-- and a slot move therefore **acquires an undo step in indexed mode** that it
does not have in constrained mode. It has to: in constrained mode a reorder
moves no pixel, and in indexed mode it rewrites every plane in the document.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from . import dither
from . import index_plane as ixp
from . import indexed as ix
from .undo import (
    ColorStateEdit,
    CompoundEdit,
    FramePaletteEdit,
    IndexRemapEdit,
    PaletteEdit,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import RGBA, Document

#: The three colour modes, in the order a menu should list them.
COLOR_MODES = ("rgb", "indexed", "grayscale")


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

    # -- colour state -------------------------------------------------------

    @property
    def is_indexed(self: Document) -> bool:
        """Whether the *index planes* are the record. See :mod:`.index_plane`.

        Deliberately **not** "has a palette". A palette-constrained RGB document
        has one and is not indexed, and conflating the two is how a pane comes
        to offer a transparent-index marker on a document that has no index to
        mark. :attr:`is_palette_locked` is the other question.
        """
        return self.color_mode == "indexed"

    @property
    def is_palette_locked(self: Document) -> bool:
        """Whether writes are snapped onto a table without being indexed.

        Warlock's own mode, which Aseprite has no equivalent of (divergence
        #19): the pixels are ordinary RGBA with soft alpha allowed, and the
        table is a constraint applied as each write commits. Every document that
        was indexed before true indexing existed opens in this mode.
        """
        return self.color_mode == "rgb" and bool(self.palette)

    @property
    def is_grayscale(self: Document) -> bool:
        return self.color_mode == "grayscale"

    def _color_state(self: Document) -> tuple[str, list[RGBA] | None, int]:
        """The three fields :class:`~.undo.ColorStateEdit` restores together."""
        return (self.color_mode, self.palette, self.transparent_index)

    def _color_step(self: Document, before: tuple[str, Any, int], run: Any) -> None:
        """``_palette_step`` for the modes: rewrite the planes and the state.

        Identical in shape and for identical reasons -- ``_replay`` records the
        pixels and knows nothing about the colour state, so its step is taken
        straight back off (it is one line old and reversed by nobody) and
        re-pushed inside a compound with the state edit in front of it. Callers
        assign the state *before* calling, because ``run`` is the raw work and
        redo re-runs it against whatever the document says by then.
        """
        after = self._color_state()
        self._replay(run)
        replayed = self.history.top
        if replayed is None:  # pragma: no cover - the push was one line ago
            return
        self.history.drop()
        self.history.push(CompoundEdit([ColorStateEdit(before, after), replayed]))

    def _push_color_state(self: Document, before: tuple[str, Any, int]) -> None:
        """One ``ColorStateEdit`` for an op that moved no plane, applied.

        The table has already been assigned by the caller; this records the
        change and re-materialises. That last part is the instant-recolour
        payoff: the indices never moved, so a whole-clip repaint is a lookup
        table swap in both directions.
        """
        self.history.push(ColorStateEdit(before, self._color_state()))
        self.rev += 1
        self._rematerialize_all()

    def _push_remap(self: Document, before: tuple[str, Any, int], order: Sequence[int]) -> None:
        """A reorder: state, then the permutation pushed through every plane.

        **The transparent index moves with its slot**, and forgetting that is a
        silent catastrophe rather than a cosmetic slip: the transparent index is
        a *position*, so a reorder that left it where it was would point it at
        whatever colour slid into that position. Sorting a palette by hue would
        turn every pixel of some innocent swatch into a hole and make the slot
        that used to be the hole paint solid -- across every frame at once, with
        no visible cause.

        Set before the ``after`` state is read, so the one edit carries all
        three fields consistently and the undo puts the index back where the
        pixels are.
        """
        forward, inverse = ixp.tables_from_order(order)
        if 0 <= self.transparent_index < forward.shape[0]:
            self.transparent_index = int(forward[self.transparent_index])
        edit = ColorStateEdit(before, self._color_state())
        remap = IndexRemapEdit(forward, inverse)
        self.history.push(CompoundEdit([edit, remap]))
        self.rev += 1
        self.apply_remap(forward)

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
        if self.is_indexed:
            # A truly indexed document cannot adopt a table by assignment: its
            # pixels are slot *numbers*, and numbers that meant one thing under
            # the old table mean something else under a new one of a different
            # length. So the general operation is the only one available, and
            # ``None`` -- "leave indexed mode" -- is spelled by the op that
            # means it rather than by a table nobody can index onto.
            if wanted is None:
                return self.convert_to_rgb() and self.set_palette(None)
            hole = self.transparent_index if self.transparent_index < len(wanted) else 0
            return self.convert_to_indexed(wanted, "nearest", transparent=hole)
        if wanted is None or not snap:
            self.palette = wanted
            self.rev += 1
            return True
        return self.convert_to_palette(wanted, "nearest")

    def set_frame_palette(
        self: Document, colours: Sequence[RGBA] | None, frame_index: int | None = None
    ) -> bool:
        """Give one frame a colour table of its own, or take its override away.

        Aseprite's palette animation: the drawing does not move and the colours
        do. On a truly indexed document that is *all* it is -- the index planes
        are untouched and only the materialisation changes -- which is why this
        is a table swap rather than the whole-document rewrite
        :meth:`set_palette` has to be. Nothing is snapped and no pixel is
        re-matched: the point of the feature is that slot 4 stays slot 4 and
        becomes a different colour.

        ``None`` removes the override, putting the frame back on the document's
        own table.

        Refused on a still document, which has no frames to hang a table on --
        ``set_range_props``' rule and its reason: a door that quietly did
        nothing would read as a bug in the palette rather than as the document
        being the wrong shape.
        """
        anim = self.anim
        if anim is None or not anim.frames:
            return False
        index = anim.current if frame_index is None else int(frame_index)
        if not 0 <= index < len(anim.frames):
            return False
        frame = anim.frames[index]
        wanted = None if not colours else [tuple(c) for c in colours]
        before = anim.frame_palette(frame.uid)
        if wanted == before:
            return False
        self.commit_floating()
        edit = FramePaletteEdit(frame.uid, before, wanted)
        edit.redo(self)
        self.history.push(edit)
        self.rev += 1
        return True

    def clear_frame_palette(self: Document, frame_index: int | None = None) -> bool:
        """Put one frame back on the document's own table."""
        return self.set_frame_palette(None, frame_index)

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
        self._refuse_tilemap_convert("snapped onto a palette")
        wanted = [tuple(c) for c in colours]
        if self.is_indexed:
            # The indexed document's version of this operation is
            # ``convert_to_indexed``: re-resolving the planes onto the new table
            # rather than snapping the materialised pixels and leaving the
            # indices behind saying something else.
            hole = self.transparent_index if self.transparent_index < len(wanted) else 0
            return self.convert_to_indexed(wanted, method, transparent=hole)
        if wanted == self.palette and method == "nearest":
            # Already exactly this table, and nearest is idempotent on a
            # document that is already snapped onto it -- so there is no step to
            # push and nothing for it to hold. The dithers are *not* let through
            # this door: re-running one is a real request (it is how a user
            # compares two matrices) and it does move pixels.
            #
            # Grouped is not let through either, and deliberately not for want
            # of idempotence -- it *is* idempotent on a document already snapped
            # onto this table, since every distinct colour is a palette member
            # and the distance-zero assignment maps each to itself. It stays out
            # because widening this door to a second method would mean deciding,
            # every time a method is added, whether it belongs here; "nearest is
            # the snap and the snap is a no-op" is a statement about one method
            # and stays one.
            return False
        self.commit_floating()
        before, self.palette = self.palette, wanted

        def run() -> None:
            # **Inside the replay closure, not captured beside it.** Two reasons,
            # both load-bearing. Redo must recompute the grouping against the
            # planes as they have just been *restored*, not against a table built
            # from the planes as they were when the step was first pushed; and a
            # table for a large document is potentially megabytes, which the undo
            # stack would then pin for the rest of the session.
            #
            # Whole-document scope, through ``_palette_planes`` -- the same
            # reduction the palette builder uses, so linked cels are counted once
            # and a plane mid-preview contributes its snapshot. Grouping per plane
            # instead would group each layer against itself, and two layers with
            # disjoint ranges would land the same grey on different swatches.
            table = (
                dither.grouped_table(self._palette_planes(), wanted)
                if method == "grouped"
                else None
            )
            self._map_planes(
                lambda plane: dither.convert(plane, wanted, method, table=table),
                mask_fn=None,
            )

        self._palette_step(before, run)
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
        return self.convert_to_palette(self.built_palette(max_colours), method)

    def built_palette(self: Document, max_colours: int = 32) -> list[RGBA]:
        """The table :meth:`palette_from_document` would use, without converting.

        Split out because the conversion *popup* needs it a control at a time:
        the max-colours slider has to show the table it would produce before the
        user commits to it.
        """
        return dither.build_palette(self._palette_planes(), max_colours)

    def _palette_planes(self: Document) -> list[Any]:
        """Every distinct pixel plane in the document, once each.

        ``unique_cel_layers`` and not the stack: a background linked across
        three frames is one plane, and counting it three times would weight the
        median cut by how many frames a cel happens to appear on.

        A plane currently showing a **conversion preview** contributes the
        snapshot that preview was computed from, not what is on screen. Building
        a palette out of an already-converted picture would collapse it onto
        itself -- drag the slider from 32 down to 8 and back and the 32 would
        never come back, because by then the drawing only has 8 colours in it.
        """
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        previewing = {uid: before for uid, before in (self._convert or ())}
        return [previewing.get(layer.uid, layer.pixels) for layer in layers]

    def add_slot(self: Document, colour: RGBA) -> bool:
        """Append a colour to the palette. Nothing is repainted: a new swatch
        is a colour the user may now paint *with*, not a claim about what is
        already on the canvas."""
        if not self.palette:
            return self.set_palette([colour])
        if tuple(colour) in self.palette:
            return False
        if self.is_indexed:
            self._check_room(1)
            state = self._color_state()
            self.palette = [*self.palette, tuple(colour)]
            # Undoable here and not in constrained mode, for ``move_slot``'s
            # reason turned around: in indexed mode the table is part of what
            # every pixel *means*, so a slot appearing and disappearing under an
            # undo would change the picture. In constrained mode it is a list of
            # colours the user may now paint with and changes no pixel, which is
            # why that path is left exactly as it was.
            self._push_color_state(state)
            return True
        self.palette = [*self.palette, tuple(colour)]
        self.rev += 1
        return True

    def _check_room(self: Document, wanted: int) -> None:
        """Refuse by name rather than silently truncate at the 256 cap.

        The cap is what makes a ``uint8`` plane, a PNG-P member and a
        ``.aseprite`` palette chunk all able to hold this document. Truncating
        would produce a file three formats agree is valid and that has quietly
        lost colours; the refusal names the number so the user knows what to
        delete.
        """
        have = len(self.palette or ())
        if have + int(wanted) > ixp.MAX_COLOURS:
            raise ValueError(
                f"an indexed palette holds at most {ixp.MAX_COLOURS} colours; "
                f"this one already has {have}"
            )

    def _transparent_after(self: Document, removed: int) -> int:
        """Where the transparent index lands once slot *removed* is gone.

        A position, so it shifts down by one when the slot that went was below
        it. Never equal to *removed*: ``remove_slot`` refuses that case before
        it gets here.
        """
        hole = self.transparent_index
        return hole - 1 if hole > removed else hole

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
        if self.is_indexed:
            # **The payoff, finally instant.** Not one pixel is rewritten and
            # not one distance is computed: the indices already say which pixels
            # are in this slot, so the whole clip repaints by swapping one row
            # of the lookup table. It is also exact where the constrained path
            # cannot be -- a table with two identical browns recolours the one
            # the user clicked and leaves the other alone, which is the single
            # clearest thing an index plane buys.
            state = self._color_state()
            self.palette = table
            self._push_color_state(state)
            return True
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
        if self.is_indexed:
            if index == self.transparent_index:
                # Refused by name. Every other slot merges into its nearest
                # colour; the transparent slot has no colour to be near, and a
                # document that lost it would have no way to be transparent at
                # all. Moving the transparency elsewhere first
                # (``set_transparent_index``) is what the user actually wants,
                # and it is one undoable act away.
                raise ValueError("the transparent slot cannot be removed")
            # Non-invertible: two slots become one and no table puts them back,
            # which is why this is the one reorder-shaped op that undoes by
            # snapshot rather than by an inverse remap. The *forward* direction
            # is still an exact table -- no colour comparison touches a pixel --
            # so the merge itself keeps every other duplicate distinct.
            #
            # The survivor is chosen by nearest colour among the remaining
            # slots, as the constrained path chooses it -- but **the transparent
            # slot is excluded from the search**, ``index_plane.resolve``'s rule
            # one level up. Merging into it would turn a solid area into a hole
            # for no reason the user could see, and it is the one slot whose
            # colour is not a claim about anything.
            keep = [i for i in range(len(table)) if i != self._transparent_after(index)]
            survivor = keep[ix.nearest(gone, [table[i] for i in keep])] if keep else 0
            into = survivor + 1 if survivor >= index else survivor
            forward = ixp.merge_table(len(self.palette), index, into)
            state = self._color_state()
            self.palette = table
            # Every slot above the removed one moved down, and the transparent
            # index is a *position*: see ``_push_remap`` for why leaving it
            # behind is a silent catastrophe rather than a cosmetic slip.
            self.transparent_index = self._transparent_after(index)
            self._color_step(state, lambda: self._remap_planes(forward))
            return True
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
        """Reorder the table.

        **In palette-constrained RGB no pixel changes and no step is pushed** --
        order is presentation there, and the exported ``.gpl`` and GIF colour
        table are all it decides.

        **In indexed mode it moves index data**, so it is an edit and it costs a
        Ctrl+Z. That is not a wart: in indexed mode the slot number *is* the
        pixel, and an operation that rewrites every plane in the document while
        claiming to change nothing would be the wart.
        """
        if not self.palette or not 0 <= index < len(self.palette):
            return False
        to = max(0, min(to, len(self.palette) - 1))
        if to == index:
            return False
        table = [*self.palette]
        table.insert(to, table.pop(index))
        if self.is_indexed:
            order = list(range(len(self.palette)))
            order.insert(to, order.pop(index))
            self.commit_floating()
            state = self._color_state()
            self.palette = table
            self._push_remap(state, order)
            return True
        self.palette = table
        self.rev += 1
        return True

    def sort_palette(
        self: Document,
        key: str,
        *,
        indices: Sequence[int] | None = None,
        counts: Sequence[int] | None = None,
        descending: bool = False,
    ) -> bool:
        """Reorder the table by *key*. In palette-constrained RGB no pixel
        changes and no undo step; in indexed mode it moves index data and
        costs a Ctrl+Z, exactly as ``move_slot`` does.

        ``move_slot``'s rule applied to the general case, and the reason it is
        the rule: order is presentation here -- the exported ``.gpl`` and the
        GIF colour table are what it decides, and not one pixel moves -- so a
        sort has nothing to restore and spending a Ctrl+Z on it would make
        undoing the *stroke* before it take two presses.

        ``indices`` sorts a **subset in place**: the selected positions keep
        their positions, and only which colour sits in each of them changes.
        That is what makes "sort these five" a thing you can do to the middle of
        a hand-arranged table without the rest of it moving.

        ``counts`` is the per-slot usage for ``key="usage"``, which the caller
        already has (the palette pane counts on demand and caches). It is asked
        for rather than recomputed because counting is a pass over every pixel
        of every cel, and doing it here would make an idle sort the most
        expensive control on the panel.
        """
        if not self.palette:
            return False
        table = list(self.palette)
        if key == "usage" and counts is None:
            counts = self.palette_usage()
        sorted_order = ix.sort_order(table, key, counts=counts, descending=descending)
        # ``moved[p]`` is the slot that ends up at position *p*. Built for both
        # branches rather than only for the indexed one, so the table below is
        # derived from the same list the index remap is -- the two cannot end up
        # describing different rearrangements.
        if indices is None:
            moved = list(sorted_order)
        else:
            wants = {i for i in indices if 0 <= i < len(table)}
            if len(wants) < 2:
                return False
            places = sorted(wants)
            chosen = [i for i in sorted_order if i in wants]
            moved = list(range(len(table)))
            for place, source in zip(places, chosen, strict=True):
                moved[place] = source
        wanted = [table[i] for i in moved]
        if wanted == table:
            return False
        if self.is_indexed:
            # ``move_slot``'s argument at whole-table scale: this rewrites every
            # plane in the document, so it is an edit. And it is *exact* -- two
            # identical swatches sort to two positions and the pixels of each
            # follow their own slot, which no colour-keyed sort could manage.
            self.commit_floating()
            state = self._color_state()
            self.palette = wanted
            self._push_remap(state, moved)
            return True
        self.palette = wanted
        self.rev += 1
        return True

    def insert_ramp(self: Document, a: int, b: int, steps: int) -> bool:
        """Fill the gap between two slots with an interpolated run of colours.

        The new colours go **between** the two in table order, running from the
        lower position's colour to the higher's, which is what "make a ramp
        between these two swatches" means in a palette that is read left to
        right. Passing the two the other way round therefore produces the same
        run -- the direction is the table's, not the click order's.

        Colours already in the table are skipped rather than inserted twice: a
        four-step ramp between two swatches three apart would otherwise plant
        duplicates of colours the user can already see, and a duplicate slot is
        one nothing distinguishes from its neighbour.

        Table-only, like :meth:`sort_palette` and ``move_slot``: adding a swatch
        repaints nothing, so there is no step to push.
        """
        if not self.palette or steps < 1:
            return False
        count = len(self.palette)
        if not (0 <= a < count and 0 <= b < count) or a == b:
            return False
        low, high = sorted((a, b))
        table = list(self.palette)
        # Against a running set, not against ``table`` alone: a ramp of more
        # steps than the two ends are apart repeats colours *within itself*, and
        # a duplicate is a duplicate whichever side of the insert it came from.
        seen = set(table)
        fresh: list[RGBA] = []
        for colour in ix.ramp_between(table[low], table[high], steps):
            if colour not in seen:
                seen.add(colour)
                fresh.append(colour)
        if not fresh:
            return False
        if self.is_indexed:
            self._check_room(len(fresh))
            # The float commits FIRST, as every sibling op commits it
            # (``move_slot``, ``sort_palette``, ``remove_slot``, the
            # conversions): committed after the table is reassigned, the
            # buffer's pixels resolve into *new*-table numbering and are then
            # pushed through a remap sized to the old table -- clamped, so the
            # floated pixels land as the wrong colour, silently
            # self-consistent.
            self.commit_floating()
            state = self._color_state()
            # Inserted *inside* the table, so every slot above the insertion
            # point shifts -- and every pixel in one of them has to shift with
            # it. ``add_slot`` appends and needs no remap; this one does, which
            # is the whole difference between the two.
            self.palette = [*table[: low + 1], *fresh, *table[low + 1 :]]
            forward = np.arange(len(table), dtype=np.uint8)
            forward[low + 1 :] += len(fresh)
            if 0 <= self.transparent_index < len(table):
                self.transparent_index = int(forward[self.transparent_index])
            self.history.push(
                CompoundEdit(
                    [
                        ColorStateEdit(state, self._color_state()),
                        IndexRemapEdit(forward, self._shrink_map(forward, len(self.palette))),
                    ]
                )
            )
            self.rev += 1
            self.apply_remap(forward)
            return True
        self.palette = [*table[: low + 1], *fresh, *table[low + 1 :]]
        self.rev += 1
        return True

    @staticmethod
    def _shrink_map(forward: np.ndarray, count: int) -> np.ndarray:
        """The inverse of a *widening* remap: ``inverse[forward[s]] == s``.

        :func:`index_plane.tables_from_order` cannot serve here because an
        insertion is not a permutation -- the new table is longer than the old
        one, and the fresh slots have no pixels and no predecessor. The inverse
        is still exact where it matters: every slot the forward map produced
        goes back to where it came from, and the fresh slots (which no plane
        references) map to zero.
        """
        inverse = np.zeros(count, dtype=np.uint8)
        inverse[forward] = np.arange(forward.shape[0], dtype=np.uint8)
        return inverse

    def set_transparent_index(self: Document, index: int) -> bool:
        """Move which slot means "hole". Indexed documents only.

        Nothing is repainted and no index moves: the *materialisation* changes,
        so pixels in the old slot become opaque again and pixels in the new one
        become holes. That is exactly what Aseprite's transparent-index control
        does, and expressing it as one lookup-table swap is why it is instant
        and why its undo is one edit carrying three small values.
        """
        if not self.is_indexed or not self.palette:
            return False
        index = int(index)
        if not 0 <= index < len(self.palette) or index == self.transparent_index:
            return False
        self.commit_floating()
        state = self._color_state()
        self.transparent_index = index
        self._push_color_state(state)
        return True

    def palette_usage(self: Document) -> list[int]:
        """Pixels in each slot, over the whole document. What tells the user
        which slots are safe to delete.

        **Indexed mode counts the planes**, which is both exact and finally
        per-*slot*: the colour-keyed count below cannot tell two identical
        swatches apart and reports each as holding every pixel of either, so a
        duplicate always looks used. Constrained mode has no planes to count and
        keeps the colour histogram, where that limitation is inherent.
        """
        if not self.palette:
            return []
        planes = self.stack if self.anim is None else self.anim.unique_cel_layers()
        totals = [0] * len(self.palette)
        for layer in planes:
            if self.is_indexed and layer.indices is not None:
                counts = ixp.histogram(layer.indices, len(self.palette))
            else:
                counts = ix.histogram(layer.pixels, self.palette)
            for slot, count in enumerate(counts):
                totals[slot] += count
        return totals

    def used_slots(self: Document) -> list[int]:
        """Which palette slots the drawing actually holds pixels in.

        Derived from :meth:`palette_usage` rather than counted a second way --
        Aseprite's *Select Used Colors* and *Select Unused* are two readings of
        one histogram, and two walks would be two answers on a document whose
        planes changed between them.
        """
        return [slot for slot, count in enumerate(self.palette_usage()) if count]

    def unused_slots(self: Document) -> list[int]:
        used = set(self.used_slots())
        return [slot for slot in range(len(self.palette or [])) if slot not in used]

    def select_slots(self: Document, slots: Sequence[int]) -> bool:
        """Select every pixel drawn in *slots*, as **one** undo step.

        Beside :meth:`used_slots` because it is the other half of the same
        gesture -- Aseprite's *Select Used Colors* and *Select Unused* are two
        readings of one histogram and this is what either reading is for.

        The union is built here rather than by calling
        :meth:`~._doc_selection.SelectionOps.select_colour_range` once per slot,
        which is what the toolbox used to do: that method pushes a
        ``SelectionEdit`` of its own, so "Select used colours" on a sixty-colour
        palette cost sixty-one Ctrl+Z to put back and walked the whole composite
        sixty times. One gesture is one step, the rule the timeline's own row
        ops already follow ("Eight rows crossed used to cost eight Ctrl+Z").

        The *composite* is read, not the index plane, for
        ``select_colour_range``'s reason: the user is pointing at what they can
        see, and a palette-constrained RGB document has no plane to point at.
        """
        from .selection import colour_range

        if not slots or not self.palette:
            return False
        found = None
        for slot in slots:
            if not 0 <= slot < len(self.palette):
                continue
            one = colour_range(self._composite, self.palette[slot], 0)
            found = one if found is None else found.combined(one, "add")
        if found is None:
            return False
        self.select(found)
        return True

    def remap_palette(self: Document, colours: Sequence[RGBA]) -> bool:
        """Swap the palette and keep every index. -> whether it changed.

        Aseprite's *Remap Palette*, and the distinction it turns on is the
        whole reason it is a second verb beside ``set_palette``: setting a
        palette **re-matches** every pixel to the nearest new colour, so a
        drawing keeps its *colours* and gets new indices; remapping keeps the
        indices, so the drawing keeps its *structure* and every pixel in slot 4
        becomes whatever slot 4 now is. That is what makes a palette swap a
        recolour rather than a re-quantisation.

        Refused on a document that is not indexed, by name: without planes
        there are no indices to keep, and the honest operation there is
        ``set_palette``.
        """
        if not self.is_indexed:
            raise ValueError(
                "remapping keeps each pixel's palette index, and only an "
                "indexed document has one -- convert it first"
            )
        wanted = [tuple(int(channel) for channel in colour) for colour in colours]
        if not wanted or wanted == [tuple(c) for c in (self.palette or [])]:
            return False
        return self.set_palette(wanted, snap=False)

    # -- mode conversions ---------------------------------------------------

    def _index_planes(self: Document) -> list[Any]:
        """Every distinct cel in the document, once each. ``_map_planes``' rule:
        a background linked across three frames is one object and one plane."""
        return list(self.stack if self.anim is None else self.anim.unique_cel_layers())

    def _remap_planes(self: Document, forward: np.ndarray) -> None:
        """The raw work of a remap, for ``_color_step``'s replay closure."""
        self.apply_remap(forward)

    def _resolve_planes(
        self: Document, colours: Sequence[RGBA], method: str, transparent: int
    ) -> None:
        """The raw work of entering indexed mode: every cel to slots and back.

        The raw work rather than the public method, ``_replay``'s rule: redo
        re-runs this directly, and going back through the entry point would push
        a second step for an operation already on the stack.
        """
        table = ixp.lut(colours, transparent)
        planes = self._index_planes()
        # **Strictly before the loop.** The loop below writes each layer's
        # pixels through ``materialize``, so a grouping built inside it would see
        # the first layers already converted and group the later ones against a
        # document that no longer exists. Built over the candidate slots, so the
        # transparent slot is never a target.
        grouping = (
            dither.grouped_index_table(
                [layer.pixels for layer in planes], colours, transparent=transparent
            )
            if method == "grouped"
            else None
        )
        for layer in planes:
            layer.indices = dither.convert_indices(
                layer.pixels, colours, method, transparent=transparent, table=grouping
            )
            layer.pixels[...] = ixp.materialize(layer.indices, table)
        self._stamp_all()
        self.invalidate_all()

    def _drop_planes(self: Document) -> None:
        """The raw work of leaving indexed mode: the planes go, the pixels stay.

        Lossless by construction and worth saying why: the pixels *are* the
        materialisation, so dropping the record loses nothing anybody can see.
        What it loses is identity -- the two brown slots become one brown -- and
        that is the whole content of the warning the UI gives before doing it.
        """
        for layer in self._index_planes():
            layer.indices = None
        self._stamp_all()
        self.invalidate_all()

    def convert_to_indexed(
        self: Document,
        colours: Sequence[RGBA] | None = None,
        method: str = "nearest",
        *,
        transparent: int = 0,
        max_colours: int = 32,
    ) -> bool:
        """Enter true indexed mode, resolving every cel onto *colours*.

        One undo step across every layer and every frame, because that is what
        the user did -- and links survive it, because ``_replay`` snapshots the
        *grid* rather than the layers, so two frames holding one cel hold one
        cel again after an undo.

        With no table, one is built from the document's own pixels, exactly as
        :meth:`palette_from_document` does for the constrained mode: two
        published operations and no third one.

        A **selection is ignored**, for :meth:`convert_to_palette`'s reason
        stated at length there: this is a change of mode, and a version of it
        that converted only the marquee would leave the pixels outside it in a
        state the mode cannot describe.
        """
        self._refuse_tilemap_convert("converted to indexed")
        wanted = [tuple(c) for c in (colours or self.built_palette(max_colours))]
        if not wanted:
            raise ValueError("a conversion needs at least one colour")
        if len(wanted) > ixp.MAX_COLOURS:
            raise ValueError(
                f"an indexed palette holds at most {ixp.MAX_COLOURS} colours, not {len(wanted)}"
            )
        if method not in dither.METHODS:
            raise ValueError(f"unknown dither method {method!r}")
        hole = int(transparent)
        if not 0 <= hole < len(wanted):
            raise ValueError(f"the transparent index must name a slot, not {transparent}")
        already = self.color_mode == "indexed" and self.transparent_index == hole
        if already and self.palette == wanted and method == "nearest":
            # Exactly this state already, and nearest is idempotent on planes
            # that are already resolved onto it. The dithers are deliberately
            # let through -- re-running one is a real request, and it moves
            # pixels -- which is ``convert_to_palette``'s rule, restated here
            # because this is the door that mode changes come through.
            return False
        self.commit_floating()
        state = self._color_state()
        self.color_mode = "indexed"
        self.palette = wanted
        self.transparent_index = hole
        self._color_step(state, lambda: self._resolve_planes(wanted, method, hole))
        return True

    def _flatten_planes(self: Document) -> None:
        """The raw work of entering grayscale mode: every cel through the luma.

        ``_replay``'s rule again -- this is the closure redo re-runs, never the
        public method.
        """
        for layer in self._index_planes():
            layer.pixels[...] = ix.grayscale(layer.pixels)
        self._stamp_all()
        self.invalidate_all()

    def convert_to_grayscale(self: Document) -> bool:
        """Enter grayscale mode: every visible pixel gets ``r == g == b``.

        A constraint over the storage this document already has, not a change of
        storage -- see :func:`indexed.grayscale` for the three-part argument, of
        which the load-bearing part is that all nineteen blend modes preserve
        grayness, so even the *composite* stays grey.

        An indexed document converting here **leaves indexed mode**: an index
        plane and a luma constraint are two different answers to "what is a
        pixel", and a document cannot hold both. Its palette is kept, which
        makes it a grayscale document with a table of colours it may not paint
        -- so the palette is dropped here, the one place a mode change discards
        something the user authored, because keeping it would be keeping a lie.
        """
        if self.color_mode == "grayscale":
            return False
        self._refuse_tilemap_convert("converted to grayscale")
        self.commit_floating()
        state = self._color_state()
        # Read before the mode is assigned, or it is always False and the planes
        # are left behind describing a picture nothing materialises from.
        had_planes = self.is_indexed
        self.color_mode = "grayscale"
        self.palette = None
        self.transparent_index = 0

        def run() -> None:
            if had_planes:
                self._drop_planes()
            self._flatten_planes()

        self._color_step(state, run)
        return True

    def convert_to_rgb(self: Document) -> bool:
        """Leave indexed (or grayscale) mode. The pixels stay exactly as they are.

        **The palette is kept**, and the document becomes palette-constrained
        RGB rather than free RGB. That is the honest mapping onto the modes this
        editor actually has: the table is one the user authored, throwing it away
        on a mode change would be the most destructive thing this menu could do,
        and a document that keeps it goes on snapping writes onto it -- which is
        what it did before it was indexed. Clearing the palette afterwards is one
        further, separate, undoable act (``set_palette(None)``).
        """
        if self.color_mode == "rgb":
            return False
        self.commit_floating()
        state = self._color_state()
        self.color_mode = "rgb"
        self._color_step(state, self._drop_planes)
        return True
