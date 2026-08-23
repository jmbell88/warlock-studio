"""Layers -- and, on an animated document, tracks.

The five structural ops each grow one of two branches, and they all take the
same shape: a track index and a stack index are the same number, because
``Animation.tracks`` mirrors ``LayerStack`` bottom-first on purpose. That is
what lets the layers panel go on calling these with the indices it already has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import composite as cp
from . import groups as gp
from . import index_plane as ixp
from .anim_edits import (
    CelSetEdit,
    TrackAddEdit,
    TrackMoveEdit,
    TrackPropsEdit,
    TrackRemoveEdit,
)
from .animation import TRACK_PROPS, Track
from .groups import (
    GroupAddEdit,
    GroupDissolveEdit,
    GroupPropsEdit,
    MembershipEdit,
)
from .layers import Layer, LayerStack, new_uid
from .undo import (
    CompoundEdit,
    LayerAddEdit,
    LayerFlagEdit,
    LayerMoveEdit,
    LayerPropsEdit,
    LayerRemoveEdit,
    MatteEdit,
    one_step,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document


class LayerOps:
    """Layer and track structure, mixed into :class:`~.document.Document`."""

    # -- the content lock ---------------------------------------------------
    #
    # This is the one comment that owns where the line is drawn, because the
    # lock is only worth anything if every door agrees on it.
    #
    # **What it guards is tool-level writes**, and the enforcement sits at the
    # engine ops themselves -- ``begin_stroke``, ``write_colour`` (which is
    # where fill and the shapes end up), ``gradient``, ``begin_filter``,
    # ``lift``, ``paste``, ``layer_from_selection``'s cut half,
    # ``begin_layer_move`` and ``merge_down`` -- each refusing *before* it
    # mutates anything. Below those doors, undo writes raw pixels through
    # ``PatchEdit``, and deliberately so:
    # a lock switched on after an edit must not wedge the history that already
    # holds it, and the alternative (a lock that silently swallows a Ctrl+Z) is
    # much worse than one that lets a user undo work they locked afterwards.
    #
    # **``commit_floating`` is deliberately not refused.** A float outlives lock
    # toggles -- it survives selecting another layer, another frame and the
    # panel checkbox -- and every save, every geometry op and every structural
    # op commits it first, so refusing here would not protect the pixels, it
    # would wedge the document: the buffer could never land and never be saved.
    #
    # **Document-scope ops apply regardless**: geometry (flip, rotate, scale,
    # crop, canvas resize), ``apply_matte``, the palette rewrites and
    # ``flatten_layers``. Those are statements about the whole document rather
    # than about a layer, and a single locked layer vetoing a canvas resize is a
    # document that cannot be worked on at all. Property edits, rename, hide,
    # reorder and delete stay legal for the same reason -- the lock is about
    # what is *painted*, not about whether the layer may be managed.
    #
    # The engine's answer is a plain refusal (False / None): it is the panes
    # that turn one press into one toast, because only they know what a press
    # is and the engine is asked this sixty times a second while a stroke runs.

    def write_locked(self: Document, layer: Layer | None = None) -> bool:
        """Whether a tool-level write to this layer must be refused.

        Defaults to the active layer, which is what every door but
        ``merge_down`` is asking about. On an animated document the property is
        the *track*'s and ``layers_for`` copies it down onto the materialised
        cel, so reading it off the layer here is reading the track.
        """
        layer = self.stack.active if layer is None else layer
        # A reference layer is drawn and never edited (6.5). Read here rather
        # than at each door for the reason the group lock is: the doors are a
        # list of fifteen and a flag they have to be taught one at a time is a
        # flag that is on in the panel and off at one of them.
        if bool(layer.locked) or bool(getattr(layer, "reference", False)):
            return True
        if not self.groups:
            return False
        # A group's lock folds down onto everything inside it (L3): locking a
        # folder is how a user protects six layers at once, and a lock that
        # stopped at the folder would be a checkbox that does nothing.
        return gp.resolve(self.groups, self.group_of, self.member_uid_of(layer))[2]

    def add_layer(self: Document, name: str | None = None) -> Layer:
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
        # The group the new row joins, read *before* the insert: a new layer
        # goes above the active one, and one that stayed outside the group the
        # active row is in would land in the middle of that group's span and
        # break its contiguity. Inheriting is also what a user means -- "add a
        # layer" while working inside a folder is a layer in that folder.
        parent = self._parent_of_active()
        if self.anim is not None:
            index = self.stack.active_index + 1
            track = Track(name=name or f"Layer {len(self.anim.tracks) + 1}")
            self._put_track(index, track, {})
            self._push_with_inheritance(
                TrackAddEdit(index, track, {}, pinned=False), parent
            )
            return self.stack[self.stack.active_index]
        width, height = self.size
        layer = Layer(
            pixels=cp.empty(width, height),
            # A fresh layer in an indexed document is a plane of the
            # transparent index -- ``_ensure_cel_for``'s rule for a fresh cel,
            # which the still branch has to agree with. Left ``None``, every
            # stroke on the new layer would fall through the funnel's RGBA
            # path, ``check_materialized`` would silently skip it, and the ORA
            # writer would store RGBA under a mode that promises slots.
            indices=(
                None
                if self.color_mode != "indexed"
                else np.full((height, width), self.transparent_index, dtype=np.uint8)
            ),
            name=name or f"Layer {len(self.stack) + 1}",
        )
        index = self.stack.insert(self.stack.active_index + 1, layer)
        self.invalidate_all()
        self._push_with_inheritance(LayerAddEdit(index, layer), parent)
        return layer

    def _parent_of_active(self: Document) -> int | None:
        if not self.groups:
            return None
        order = self.member_uids()
        index = self.stack.active_index
        return self.group_of.get(order[index]) if 0 <= index < len(order) else None

    def inherit_group_edits(self: Document, parent: int | None) -> list[Any]:
        """Put the row that was *just added* into ``parent``, returning the step.

        Every op that inserts a row above the active one has to do this, and
        "the row that was just added" is unambiguous because all of them leave
        it active: ``LayerStack.insert`` sets ``active_index`` and ``_put_track``
        materialises with ``active=index``. Taking the member uid from
        ``member_uids()[active_index]`` rather than from a parameter is what
        makes this callable from ``_add_layer_edit``'s callers too, which is
        where the inheritance was missing.

        A row inserted into the middle of a group's span that did **not** join
        it splits that span, and the contiguity invariant would be false with
        nothing to say so -- which is exactly what a paste inside a folder used
        to do.
        """
        if parent is None:
            return []
        member = self.member_uids()[self.stack.active_index]
        self._set_membership(member, parent)
        return [MembershipEdit(member, None, parent)]

    def _push_with_inheritance(self: Document, edit: Any, parent: int | None) -> None:
        """Push an add, with the membership the new row inherits folded in."""
        edits = [edit, *self.inherit_group_edits(parent)]
        self.history.push(one_step(edits))

    def duplicate_layer(self: Document, index: int | None = None) -> Layer:
        self.commit_floating()
        index = self.stack.active_index if index is None else index
        # The source's own group, not the active row's: a duplicate belongs
        # beside what it copies, and it is inserted directly above it.
        parent = (
            self.group_of.get(self.member_uids()[index]) if self.groups else None
        )
        if self.anim is not None:
            track = self.anim.tracks[index]
            copy_track = Track(
                name=f"{track.name} copy",
                opacity=track.opacity,
                visible=track.visible,
                blend=track.blend,
                # The two locks as well. The still branch below copies every
                # property (``Layer.copy`` does), and this list stopping at
                # four made duplicating a layer quietly *unlock* it on an
                # animated document and nowhere else -- the fifth place the
                # track-property list has to agree with itself.
                alpha_lock=track.alpha_lock,
                locked=track.locked,
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
            self._push_with_inheritance(
                TrackAddEdit(index + 1, copy_track, cels, pinned=True), parent
            )
            return self.stack[self.stack.active_index]
        copy = self.stack.duplicate(index)
        self.invalidate_all()
        self._push_with_inheritance(LayerAddEdit(index + 1, copy), parent)
        return copy

    def remove_layer(self: Document, index: int | None = None) -> bool:
        if len(self.stack) == 1:
            return False
        self.commit_floating()
        index = self.stack.active_index if index is None else index
        # Taken out of the tree *before* the row goes, while the uid is still
        # findable, and folded into the same step -- a delete that left a
        # dangling membership behind would keep an empty group alive and the
        # undo would put the row back outside it.
        member = self.member_uids()[index] if self.groups else None
        if self.anim is not None:
            track = self.anim.tracks[index]
            cels = {
                frame.uid: cel
                for frame in self.anim.frames
                if (cel := self.anim.cels.get((track.uid, frame.uid))) is not None
            }
            self._drop_track(track)
            edit: Any = TrackRemoveEdit(
                index, track, cels, pinned=self._released(cels.values())
            )
        else:
            gone = self.stack.remove(index)
            edit = LayerRemoveEdit(index, gone)
            self.invalidate_all()
        edits = [edit, *(self._forget_member(member) if member is not None else [])]
        self.history.push(one_step(edits))
        return True

    def _move_row_edit(self: Document, index: int, to: int) -> Any:
        """Move a stack row and *return* the step rather than pushing it.

        The body ``move_layer`` used to be, extracted so ``move_into_group``
        can fold the same move into a compound with the membership change that
        goes with it -- the ``_add_layer_edit`` refactor, one concern over.
        """
        if self.anim is not None:
            uid = self.anim.tracks[index].uid
            self._move_track(uid, to)
            return TrackMoveEdit(uid, index, to)
        uid = self.stack[index].uid
        self.stack.move(index, to)
        self.invalidate_all()
        return LayerMoveEdit(uid, index, to)

    def move_layer(self: Document, index: int, to: int) -> bool:
        to = max(0, min(int(to), len(self.stack) - 1))
        if to == index:
            return False
        self.commit_floating()
        # A reorder can carry a row across a group boundary, and the tree has
        # to follow or the span it lands in stops being contiguous. Both are
        # one gesture and therefore one step.
        edits: list[Any] = [self._move_row_edit(index, to)]
        if self.groups:
            order = self.member_uids()
            member = order[to]
            before = self.group_of.get(member)
            after = self._group_for_position(to)
            if before != after:
                self._set_membership(member, after)
                edits.append(MembershipEdit(member, before, after))
                edits.extend(self._prune_group(before))
        self.history.push(one_step(edits))
        return True

    def layer_at(self: Document, xy: tuple[int, int]) -> int | None:
        """The topmost *visible* layer with paint under ``xy``, or None.

        The gesture Alt already gives for colour, given for layers: point at a
        drawing and get the layer it is on. Top-down, because that is the one
        the user is looking at.

        **Alpha > 0, not a threshold.** An antialiased edge at alpha 1 is paint
        at that pixel, and any cut-off above it would be arbitrary.

        **Visibility is consulted, opacity is not** -- ``select_layer_alpha``'s
        split read forwards. You cannot point at a layer you cannot see, so a
        hidden layer (in its own right, or through the group fold) is not under
        the cursor; opacity is a continuum with no non-arbitrary cut, and a
        layer at 2% is still one the user chose to show.
        """
        if not self.in_bounds(xy):
            return None
        x, y = int(xy[0]), int(xy[1])
        fold = self.group_fold()
        for index in range(len(self.stack) - 1, -1, -1):
            layer = self.stack[index]
            if not layer.visible:
                continue
            if fold is not None and not fold[index][0]:
                continue
            if layer.pixels[y, x, 3] > 0:
                return index
        return None

    def set_active_layer(self: Document, index: int) -> None:
        """Not undoable: which layer is selected is a view state, not an edit."""
        if index == self.stack.active_index:
            return
        self.commit_floating()
        self.stack.active_index = max(0, min(int(index), len(self.stack) - 1))
        self.invalidate_all()

    def set_layer_props(
        self: Document, index: int | None = None, *, was: dict | None = None, **props: Any
    ) -> bool:
        """Record a property change as one undo step.

        ``was`` is for a control that mutates the layer live so the canvas
        follows the drag, and only asks for the step once the drag is released.
        By then the layer already holds the new value, so reading "before" off
        it compares a value against itself: nothing is ever pushed, the change
        is not undoable, and the history head does not move -- which is what
        the tab compares against to decide whether the document is dirty.

        The keys are checked against the allowlist rather than trusted, for
        ``set_range_props``'s reason: this writes with ``setattr``, and an
        unknown one would otherwise mint a new attribute on the track,
        silently, and be lost at the next save.
        """
        index = self.stack.active_index if index is None else index
        unknown = set(props) - TRACK_PROPS
        if unknown:
            raise ValueError(f"unknown track property: {sorted(unknown)[0]}")
        if "continuous" in props and self.anim is None:
            # The one track property a ``Layer`` has no counterpart for: it
            # says what autovivification writes, and a still document has no
            # timeline for a cel to be carried forward into. Refused by name
            # rather than let through, because the branch below would setattr
            # it onto a Layer where nothing would ever read it again.
            raise ValueError("a still image has no timeline to be continuous on")
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

    def set_layers_props(
        self: Document,
        indices: Any = None,
        *,
        was: dict[int, dict] | None = None,
        **props: Any,
    ) -> bool:
        """The same property change across several layers, as **one** undo step.

        ``indices`` is the rows to write, or ``None`` for the whole stack.

        :meth:`set_layer_props` pushes its own edit per call that changes
        something, so the header row's "hide every layer" cost a ten-layer
        document ten Ctrl+Z to reverse one click, and a drag down the eye column
        cost one per row it crossed -- against the one-gesture-one-step rule the
        filters, the palette conversion and ``apply_matte`` all follow.

        ``set_range_props`` is this operation over a *span*, and it is not
        reusable here: it refuses outright when ``anim`` is None, and a still
        document is exactly where the header row is most often clicked. So this
        covers both shapes, pushing the track edit or the layer edit per row and
        compounding them.

        ``was`` is :meth:`set_layer_props`' own, per row: a gesture that mutates
        the layers live so the canvas follows the drag has already written the
        new value by the time it asks for the step, so reading "before" off the
        row would compare a value against itself and record nothing.

        Only rows that actually change contribute an edit, ``set_range_props``'
        rule: hiding a stack most of which is already hidden must not make undo
        walk back through a row of no-ops.
        """
        rows = self.anim.tracks if self.anim is not None else self.stack
        wanted = range(len(rows)) if indices is None else indices
        return self._set_row_props({index: props for index in wanted}, was=was)

    def _set_row_props(
        self: Document,
        per_row: dict[int, dict],
        *,
        was: dict[int, dict] | None = None,
    ) -> bool:
        """Write a *different* set of props to each row, as one undo step.

        :meth:`set_layers_props` is this with one ``props`` shared by every row,
        and :meth:`solo` is the reason it is not enough on its own: solo writes
        ``visible=True`` to one row and ``False`` to the rest, which is two
        values and so was two calls, and two calls are two undo steps -- the
        exact ten-Ctrl+Z defect ``set_layers_props`` exists to have fixed.

        Everything else is that method's, unchanged: only rows that actually
        change contribute an edit, ``was`` supplies the "before" for a gesture
        that already wrote the new value live, and the edits compound into one.
        """
        unknown = set().union(*(set(props) for props in per_row.values()), set()) - TRACK_PROPS
        if unknown:
            raise ValueError(f"unknown track property: {sorted(unknown)[0]}")
        self.commit_floating()
        anim = self.anim
        rows = list(anim.tracks) if anim is not None else list(self.stack)
        sources = was or {}
        edits: list[Any] = []
        for index, props in per_row.items():
            if not 0 <= index < len(rows):
                continue
            target = rows[index]
            source = sources.get(index, {})
            before = {
                key: source.get(key, getattr(target, key)) for key in props
            }
            if before == props:
                continue
            for key, value in props.items():
                setattr(target, key, value)
            edits.append(
                TrackPropsEdit(target.uid, before, dict(props))
                if anim is not None
                else LayerPropsEdit(target.uid, before, dict(props))
            )
        if not edits:
            return False
        self.history.push(one_step(edits))
        if anim is not None:
            self._anim_changed()
        else:
            self.invalidate_all()
        return True

    def set_all_layer_props(self: Document, **props: Any) -> bool:
        """:meth:`set_layers_props` over the whole stack. The header row's."""

        return self.set_layers_props(None, **props)

    def _resolved_plane(self: Document, pixels: np.ndarray) -> np.ndarray | None:
        """The index plane a freshly *minted* layer needs, or ``None`` outside
        indexed mode.

        Every op that mints a ``Layer`` from computed pixels inside a
        still-indexed document -- a flatten, a track merge, a paste-as-layer --
        has to give it a plane, because a planeless layer in an indexed
        document is a silent gap: ``check_materialized`` skips it,
        ``palette_usage`` falls back to the colour histogram, and the ORA
        writer stores it as RGBA in an archive that claims indexed, so
        duplicate-slot identity is quietly lost on the next open.

        The resolution is the funnel's own (:func:`.index_plane.resolve`, no
        ``prefer`` -- output no gesture authored has no paint slot to express),
        and ``pixels`` is rewritten **in place** to the materialisation, so the
        minted layer is born already agreeing with its plane -- soft alpha a
        composite produced is thresholded here exactly as a committed stroke's
        would be.
        """
        if self.color_mode != "indexed" or not self.palette:
            return None
        table = self._index_lut()
        indices = ixp.resolve(pixels, table, self.transparent_index)
        pixels[...] = ixp.materialize(indices, table)
        return indices

    def merge_down(self: Document, index: int | None = None) -> bool:
        """Flatten a layer into the one beneath it, honouring its blend mode."""
        index = self.stack.active_index if index is None else index
        if index == 0:
            return False
        # Either participant: the merge writes into the lower layer and takes
        # the upper one away, so a lock on either is a refusal for the whole op
        # rather than a merge of half of it.
        if self.write_locked(self.stack[index]) or self.write_locked(self.stack[index - 1]):
            return False
        # Either participant again, and *before* the merge is computed: the
        # write lands in the lower layer's ``pixels`` a dozen lines below and
        # only then reaches ``_patch_edit_for``, whose own tilemap refusal
        # would fire with the merge already half-applied.
        self._refuse_tilemap_layer(self.stack[index].uid, "merging onto")
        self._refuse_tilemap_layer(self.stack[index - 1].uid, "merging onto")
        self.commit_floating()
        # The *upper* row stops existing, so its membership goes with it and
        # may empty a group; the lower keeps its own, which is what "a merge
        # across a boundary keeps the lower's membership" means -- the merged
        # drawing stays where the layer it merged into was.
        upper_member = self.member_uids()[index] if self.groups else None
        if self.anim is not None:
            return self._merge_tracks(index, upper_member)
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
        # Through the funnel's list-returning sibling rather than a raw
        # ``PatchEdit``: the merge is a write into the lower layer like any
        # other, so the colour mode applies -- and on an indexed document the
        # lower's index plane has to be re-resolved from the merged pixels, or
        # it goes on describing the pre-merge picture while the upper layer is
        # gone: the save writes pre-merge indices and the merged artwork is
        # silently lost. ``None`` (a merge that changed no slot) still removes
        # the upper row; there is simply no pixel step to record.
        patch = self._patch_edit_for(lower, (0, 0, width, height), before)
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
        self.invalidate_all()
        self.history.push(
            CompoundEdit(
                [
                    *([patch] if patch is not None else []),
                    LayerPropsEdit(
                        lower.uid, props_before, {"opacity": 1.0, "blend": "normal"}
                    ),
                    LayerRemoveEdit(index, removed),
                    *(
                        self._forget_member(upper_member)
                        if upper_member is not None
                        else []
                    ),
                ]
            )
        )
        return True

    def _merge_tracks(self: Document, index: int, upper_member: int | None = None) -> bool:
        """Merge-down across every frame at once, as one undo step.

        The question this has to answer -- and the reason it was refused
        outright for so long -- is what a merge of a linked cel with an unlinked
        one means. The answer is the one the rest of the grid already gives: a
        link is two slots holding one object, so the merge is **memoised on the
        pair of cels it consumes**. Two frames whose lower and upper cels are
        both the same objects are two frames whose merged result is the same
        drawing, so they get one ``Layer`` and stay linked; a frame where the
        upper differs gets its own. Nothing has to be told which slots were
        linked -- the identity of the inputs decides it.

        Three rules fall out of that and each of them was a bug waiting:

        The lower's cels are **never mutated in place**, as the still branch
        mutates its layer. A lower cel may be linked across frames where the
        upper is not, so writing merged pixels into it would push one frame's
        merge into another frame that never asked for it.

        The merged layers are minted **once, here**, and the ``CelSetEdit``
        holds them -- ``unlink_cel``'s rule. A redo that merged again would hand
        back layers with new identities and strand every patch recorded above.

        And a slot with **no upper cel at all** gets no edit: there is nothing
        to merge into the lower, and minting a copy of it would break its links
        and cost the grid a full plane per frame for no change.
        """
        anim = self.anim
        assert anim is not None
        width, height = self.size
        upper_track, lower_track = anim.tracks[index], anim.tracks[index - 1]

        merged_for: dict[tuple[int, int], Layer] = {}
        upper_cels: dict[int, Layer] = {}
        edits: list[Any] = []
        for frame in anim.frames:
            upper = anim.cels.get((upper_track.uid, frame.uid))
            if upper is not None:
                upper_cels[frame.uid] = upper
            lower = anim.cels.get((lower_track.uid, frame.uid))
            if upper is None:
                continue
            key = (id(lower), id(upper))
            layer = merged_for.get(key)
            fresh = layer is None
            if layer is None:
                entries = []
                if lower is not None:
                    entries.append((lower.pixels, lower_track.opacity, lower_track.blend))
                if upper_track.visible:
                    entries.append((upper.pixels, upper_track.opacity, upper_track.blend))
                merged = cp.to_uint8(cp.stack_region(entries, (0, 0, width, height)))
                # ``_resolved_plane``: a cel minted inside an indexed document
                # is born with its index plane, or the save silently records
                # RGBA under a mode that promises slots.
                layer = Layer(
                    pixels=merged,
                    indices=self._resolved_plane(merged),
                    name=lower_track.name,
                )
                merged_for[key] = layer
            # Written straight into the grid rather than through ``_set_cel``:
            # that helper ends in ``_anim_changed``, and a forty-frame clip
            # would rebuild the whole view and recomposite forty times for one
            # click. The single ``_anim_changed`` at the end is the same answer.
            anim.cels[(lower_track.uid, frame.uid)] = layer
            # Charged only where this edit is the first to introduce the pair.
            # A repeated pair means both the ``before`` and the ``after`` were
            # counted by an earlier step, and counting them again would spend
            # the budget on memory that exists once.
            pinned: Any = frozenset()
            if fresh:
                pinned = frozenset(
                    [id(layer)] + ([] if lower is None else [id(lower)])
                )
            edits.append(
                CelSetEdit(lower_track.uid, frame.uid, lower, layer, pinned=pinned)
            )

        if not edits:
            # Nothing anywhere in the upper row: the merge is the removal, and
            # the still branch's props bake would be a step that changes
            # nothing. ``remove_layer`` takes the membership with it.
            return self.remove_layer(index)

        props_before = {"opacity": lower_track.opacity, "blend": lower_track.blend}
        after = {"opacity": 1.0, "blend": "normal"}
        # The same argument the still branch makes: the merged pixels already
        # carry the lower's opacity, and its blend has nothing left to blend
        # against. Set on the track directly, for the reason the cels above are.
        lower_track.opacity, lower_track.blend = 1.0, "normal"
        edits.append(TrackPropsEdit(lower_track.uid, props_before, after))

        self._drop_track(upper_track)
        edits.append(
            TrackRemoveEdit(
                index,
                upper_track,
                upper_cels,
                pinned=self._released(upper_cels.values()),
            )
        )
        if upper_member is not None:
            edits.extend(self._forget_member(upper_member))
        self.history.push(CompoundEdit(edits))
        self._stamp_all()
        self._anim_changed(active=index - 1)
        return True

    def flatten_layers(self: Document) -> None:
        """Collapse the stack to one layer. Undoable as a canvas-level op."""
        if self.anim is not None:
            self._flatten_grid()
            return
        if len(self.stack) == 1:
            return
        self.commit_floating()
        # Replay must be a pure function of the document, and minting a uid is
        # the one part of this op that is not: a redo would produce a layer
        # with a new identity, stranding every patch recorded above it. The uid
        # is drawn once and closed over, so every replay lands on the same one.
        uid = new_uid()
        self._replay(lambda: self._do_flatten(uid))

    def _flatten_grid(self: Document) -> None:
        """Collapse an animated document to one track, frame by frame.

        Through ``_replay`` rather than a compound of grid edits, for the reason
        the still branch takes that path: every cel in the document is thrown
        away and one per frame appears, so a snapshot is not the expensive
        answer here, it is the only truthful one.

        **The link partition is computed structurally, at op time.** Two frames
        whose whole stacks are the same objects flatten to the same picture, so
        they share one ``Layer`` and stay linked -- and the grouping is worked
        out *once*, here, then closed over, because replay has to be a pure
        function of the document and the ``id()``s it is computed from will not
        survive the snapshot restore that precedes a redo. The uids go the same
        way and for ``flatten_layers``' own reason: one per group plus one for
        the track, drawn here and reused by every replay.

        Durations, tags and the layout are untouched. Flatten is a statement
        about the layers, and a timeline that came back at 10 fps because its
        rows were merged would be a different kind of edit.
        """
        anim = self.anim
        assert anim is not None
        if len(anim.tracks) <= 1:
            return
        self.commit_floating()
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, frame in enumerate(anim.frames):
            key = tuple(
                id(anim.cels.get((track.uid, frame.uid))) for track in anim.tracks
            )
            groups.setdefault(key, []).append(i)
        partition = list(groups.values())
        uids = [new_uid() for _ in partition]
        track_uid = new_uid()
        self._replay(lambda: self._do_flatten_grid(partition, uids, track_uid))

    def _do_flatten_grid(
        self: Document, partition: list[list[int]], uids: list[int], track_uid: int
    ) -> None:
        anim = self.anim
        assert anim is not None
        size = self.size
        track = Track(name="Flattened", uid=track_uid)
        cels: dict[tuple[int, int], Layer] = {}
        for indices, uid in zip(partition, uids, strict=True):
            frame = anim.frames[indices[0]]
            # ``frame_stack``, not a bare one: the flatten has to see the group
            # fold, or a hidden folder's layers reappear in the flattened cel.
            # Necessarily *before* the tree is cleared below.
            flat = self.frame_stack(frame).flatten()
            # ``_resolved_plane``, for ``_do_flatten``'s reason -- and per
            # link-group, so two frames sharing one flattened cel share one
            # plane, exactly as they share the pixels.
            layer = Layer(
                pixels=flat, indices=self._resolved_plane(flat), name=track.name, uid=uid
            )
            for i in indices:
                cels[(track_uid, anim.frames[i].uid)] = layer
        anim.tracks = [track]
        anim.cels = cels
        # One track is nothing for a group to hold, as in ``_do_flatten``.
        # Idempotent, which replay requires; ``ReplayEdit`` carries the tree it
        # replaced and puts it back.
        self.groups, self.group_of = {}, {}
        # Every frame's flatten really did change, which ``invalidate_all``
        # deliberately does not assume.
        self._stamp_all()
        self.stack = LayerStack(anim.layers_for(anim.frame, size), 0)

    def _set_layer_flags(self: Document, uid: int, props: dict) -> None:
        """Write ``background``/``reference`` onto both homes of one identity.

        The track is authoritative on an animated document -- writing only the
        materialised layer would last until the next time that frame was
        rebuilt -- and the still document has no track at all. A track and its
        layer share one uid by construction (``Track.of``), so this is one
        address with two places to put the answer, not two objects to keep in
        step.
        """
        if self.anim is not None:
            for track in self.anim.tracks:
                if track.uid == uid:
                    for key, value in props.items():
                        setattr(track, key, value)
                    break
        for layer in self.stack:
            if layer.uid == uid:
                for key, value in props.items():
                    setattr(layer, key, value)
                break
        self.invalidate_all()

    def _flag_edit(self: Document, uid: int, **props: Any) -> Any:
        """A :class:`LayerFlagEdit` capturing the current values as ``before``."""

        target = None
        if self.anim is not None:
            target = next((t for t in self.anim.tracks if t.uid == uid), None)
        if target is None:
            target = self.stack.by_uid(uid)
        before = {key: getattr(target, key) for key in props}
        return LayerFlagEdit(uid, before, dict(props))

    def to_background(self: Document) -> bool:
        """Make the bottom layer a real background layer. -> whether it changed.

        **The matte becomes pixels.** ``Document.matte`` was the stand-in
        (divergence #6): a colour composited under the stack at flatten time,
        with nothing in the layer model to carry it and therefore nothing for
        an ``.aseprite`` or an ``.ora`` to record. Converting fills the
        transparent parts of the bottom layer with it and clears it, so what
        was a flatten-time overlay becomes a thing the user can paint on and
        every writer already knows how to store.

        Only the bottom layer, which is the rule ``LayerStack`` enforces on
        every reorder: a background in the middle of a stack is a layer that
        hides everything under it while claiming to be the floor.

        **One ``CompoundEdit`` covering all three of the things this writes.**
        The pixels are a ``PatchEdit``, the flag a ``LayerFlagEdit`` and the
        matte a ``MatteEdit``: pushing only the first put the pixels back under
        a layer still calling itself a background -- so the composite still
        forced alpha to 255 and the canvas did not change -- and left the matte
        colour, which nothing else in the document holds, gone for good.
        """
        if not len(self.stack) or self.stack[0].background:
            return False
        layer = self.stack[0]
        before = layer.pixels.copy()
        matte_before = None if self.matte is None else tuple(self.matte)
        if self.matte is not None:
            fill = np.asarray(self.matte, dtype=np.float32)
            alpha = layer.pixels[..., 3:4].astype(np.float32) / 255.0
            blended = fill[None, None, :] * (1.0 - alpha) + layer.pixels.astype(
                np.float32
            ) * alpha
            layer.pixels[...] = cp.to_uint8_255(blended)
        layer.pixels[..., 3] = 255
        flag = self._flag_edit(layer.uid, background=True)
        edits: list[Any] = []
        patch = self._patch_edit_for(layer, (0, 0, *self.size), before)
        if patch is not None:
            edits.append(patch)
        edits.append(flag)
        edits.append(MatteEdit(matte_before, None))
        self._set_layer_flags(layer.uid, {"background": True})
        self.matte = None
        self.history.push(CompoundEdit(edits))
        self.invalidate_all()
        return True

    def from_background(self: Document) -> bool:
        """Turn the background back into an ordinary layer. -> whether it did.

        The pixels are left exactly as they are: what was painted stays
        painted, and the only thing that changes is that alpha starts meaning
        something again. That flag *is* the whole edit, which is why one
        ``LayerFlagEdit`` is the whole step -- before this pushed one, the
        conversion was not undoable at all while still marking the document
        dirty.
        """
        if not len(self.stack) or not self.stack[0].background:
            return False
        uid = self.stack[0].uid
        self.history.push(self._flag_edit(uid, background=False))
        self._set_layer_flags(uid, {"background": False})
        self.rev += 1
        return True

    def set_matte(self: Document, colour: Any) -> bool:
        """Set what a flattened export puts behind transparency. -> changed?

        ``matte_for`` decides this once, at load, and until this existed
        nothing revisited it: a photo opened here flattened onto white, the
        eraser cut alpha the export then filled back in, and the only way out
        was the AI cutout's ``apply_matte`` or ``to_background``. That default
        is still right -- a photo opened here is still a photo when it is
        saved -- but it is a choice, and a choice needs somewhere to be made.

        Refused on a background document, because ``flatten`` consults the
        matte only where there is no real background layer (divergence #6.5):
        a control that flipped it there would change nothing the user can see.
        """
        after = None if colour is None else tuple(colour)
        before = None if self.matte is None else tuple(self.matte)
        if after == before or self.has_background:
            return False
        self.matte = after
        self.history.push(MatteEdit(before, after))
        self.invalidate_all()
        return True

    def toggle_matte(self: Document) -> bool:
        """Flip the flatten matte between opaque white and off. -> changed?"""
        from .document import OPAQUE_WHITE

        return self.set_matte(None if self.matte is not None else OPAQUE_WHITE)

    def set_reference(self: Document, index: int, value: bool) -> bool:
        """Mark a layer as reference -- drawn, never edited."""

        if not 0 <= index < len(self.stack):
            return False
        uid = self.stack[index].uid
        value = bool(value)
        if self.stack[index].reference == value:
            return False
        self.history.push(self._flag_edit(uid, reference=value))
        self._set_layer_flags(uid, {"reference": value})
        self.rev += 1
        return True

    def solo(self: Document, index: int) -> bool:
        """Show only this layer, or show everything again. -> whether it changed.

        Aseprite's solo mode, and it is deliberately **not a mode**: it writes
        the ordinary ``visible`` flags, so there is no second visibility state
        for the compositor, the export path or the ORA writer to learn about --
        and pressing it again on the layer that is already alone restores the
        rest rather than leaving the user to click eight eyes back on.
        """
        if not 0 <= index < len(self.stack):
            return False
        alone = all(
            layer.visible == (at == index) for at, layer in enumerate(self.stack)
        )
        # One step, not one per row. This used to loop ``set_layer_props``,
        # which pushes its own edit per call that changes something -- so
        # soloing in a ten-layer document cost nine Ctrl+Z to undo one click,
        # and each partial state in between was independently reachable. That
        # is the defect ``set_layers_props`` was added for; solo could not use
        # it only because it writes two different values, which is what
        # ``_set_row_props`` takes.
        return self._set_row_props(
            {
                at: {"visible": True if alone else at == index}
                for at in range(len(self.stack))
            }
        )

    def apply_matte(self: Document, alpha: np.ndarray) -> bool:
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
        rather than one layer of it -- and the matte is *in* that edit, as a
        ``MatteEdit`` appended last. Clearing it is not optional: leaving it as
        the opaque white ``matte_for`` gives an opaque photo would flatten
        every cut pixel straight back to white on save, which is exactly the
        file this edit exists to avoid writing. It used to be cleared outside
        the edit, which meant undoing a cutout put the pixels back and lost the
        colour for good -- nothing else in the document holds it.
        ``CompoundEdit.undo`` walks ``reversed(...)``, so appending it last
        restores the matte before the pixels, the ordering ``to_background``
        uses.
        """
        width, height = self.size
        if alpha.shape[:2] != (height, width):
            return False
        # Every distinct cel is multiplied below, so one tilemap cel anywhere in
        # the document is enough -- and the refusal has to be here rather than
        # at ``_patch_edit_for``, which is reached only after that cel's alpha
        # channel has already been written.
        self._refuse_tilemaps("matte")
        matte_before = None if self.matte is None else tuple(self.matte)
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
            # ``to_uint8_255``: the same narrowing, in the one place that owns
            # it, and with the native kernel behind it.
            layer.pixels[:, :, 3] = cp.to_uint8_255(
                layer.pixels[:, :, 3].astype(np.float32) * weight
            )
            # The funnel's list-returning sibling, not a raw ``PatchEdit``: on
            # an indexed document a cut-out pixel has to become the transparent
            # index in the layer's plane, or the plane keeps the colour slot
            # and the save resurrects everything the matte removed. It also
            # owns the no-op test -- an indexed layer whose every pixel stayed
            # over the threshold resolves to the same slots, contributes no
            # step, and has its materialisation already put back.
            edit = self._patch_edit_for(layer, rect, before)
            if edit is None:
                continue
            edits.append(edit)
        if not edits:
            return False
        edits.append(MatteEdit(matte_before, None))
        self.matte = None
        self.history.push(CompoundEdit(edits))
        # Every cel in the grid was written -- so this is one of the few places
        # that has to say so, ``invalidate_all`` having stopped guessing.
        self._stamp_all()
        self.invalidate_all()
        return True

    def _do_flatten(self: Document, uid: int) -> None:
        flat = self.stack.flatten()
        # ``_resolved_plane``: the one layer an indexed document is left with
        # must carry a plane, or the whole document is "indexed" with nothing
        # indexed in it. Deterministic, which replay requires.
        indices = self._resolved_plane(flat)
        self.stack = LayerStack(
            [Layer(pixels=flat, indices=indices, name="Flattened", uid=uid)], 0
        )
        # One layer is nothing for a group to hold, so the tree goes with them.
        # Idempotent, which replay requires; ``ReplayEdit`` carries the tree it
        # replaced and puts it back.
        self.groups, self.group_of = {}, {}

    # -- layer groups -------------------------------------------------------
    #
    # A parallel tree over the flat stack; ``groups.py`` argues for the shape
    # and owns the invariant (a group's leaves are contiguous, spans nest,
    # groups are never empty). What is here is the ops that *maintain* it, and
    # every one of them is a dictionary edit plus, where it has to be, the
    # stack move that keeps the contiguity true.

    def _put_group(
        self: Document, node: Any, members: tuple[int, ...], parent: int | None
    ) -> None:
        """Raw hook for ``GroupAddEdit.redo`` / ``GroupDissolveEdit.undo``."""
        self.groups[node.uid] = node
        if parent is None:
            self.group_of.pop(node.uid, None)
        else:
            self.group_of[node.uid] = parent
        for uid in members:
            self.group_of[uid] = node.uid
        self._groups_changed()

    def _drop_group(self: Document, group_uid: int) -> None:
        """Raw hook for the inverse.

        Members are reparented to the group's own parent rather than orphaned,
        which is what makes dissolving a *nested* group put its layers into the
        group it was inside rather than at the root -- and is why this is not
        simply a ``del``.
        """
        parent = self.group_of.get(group_uid)
        for uid, at in list(self.group_of.items()):
            if at != group_uid:
                continue
            if parent is None:
                del self.group_of[uid]
            else:
                self.group_of[uid] = parent
        self.group_of.pop(group_uid, None)
        self.groups.pop(group_uid, None)
        self._groups_changed()

    def _set_group_props(self: Document, group_uid: int, props: dict) -> None:
        node = self.groups.get(group_uid)
        if node is None:
            return
        for key, value in props.items():
            setattr(node, key, value)
        self._groups_changed()

    def _set_membership(self: Document, member_uid: int, parent: int | None) -> None:
        if parent is None:
            self.group_of.pop(member_uid, None)
        else:
            self.group_of[member_uid] = parent
        self._groups_changed()

    def _groups_changed(self: Document) -> None:
        """What every group edit ends with.

        ``invalidate_all`` refreshes the fold and recomposites; ``_stamp_all``
        is the animated half, for ``_anim_changed``'s reason -- a group is
        authoritative over every frame its tracks appear in, so hiding one
        really does alter every cached flatten in the document.
        """
        self._stamp_all()
        self.invalidate_all()

    def _prune_group(self: Document, group_uid: int | None) -> list[Any]:
        """Dissolve ``group_uid`` if nothing is left in it, and its parents too.

        Empty groups are disallowed rather than tolerated, because an empty one
        has no span: it is neither contiguous nor not, it composites nothing,
        and it would sit in the panel as a folder that cannot be refilled
        without a drop target it does not have. So the last member leaving
        takes the group with it, upwards.
        """
        edits: list[Any] = []
        while group_uid is not None and group_uid in self.groups:
            if gp.leaves_of(self.group_of, self.member_uids(), group_uid):
                break
            node = self.groups[group_uid]
            parent = self.group_of.get(group_uid)
            self._drop_group(group_uid)
            edits.append(GroupDissolveEdit(node, (), parent))
            group_uid = parent
        return edits

    def _forget_member(self: Document, member_uid: int) -> list[Any]:
        """Take a row out of the tree, returning the steps that did it.

        Called by the ops that make a row stop existing -- a delete, and the
        upper half of a merge. Split out because both have to answer the same
        two questions: what was this row's parent, and did taking it away leave
        an empty group behind.
        """
        parent = self.group_of.get(member_uid)
        if parent is None:
            return []
        self._set_membership(member_uid, None)
        return [MembershipEdit(member_uid, parent, None), *self._prune_group(parent)]

    def _group_for_position(self: Document, index: int) -> int | None:
        """Which group a row at ``index`` belongs in to keep spans contiguous.

        The deepest group that contains *both* neighbours, or the one neighbour
        there is at the ends of the stack. That is the only answer that cannot
        split a span: a row dropped between two members of one group must join
        it, and a row dropped between two different groups must be in neither.
        """
        order = self.member_uids()
        below = gp.ancestry(self.group_of, order[index - 1]) if index > 0 else None
        above = (
            gp.ancestry(self.group_of, order[index + 1])
            if index + 1 < len(order)
            else None
        )
        if below is None:
            return above[0] if above else None
        if above is None:
            return below[0] if below else None
        for guid in below:  # innermost first
            if guid in above:
                return guid
        return None

    def group_layers(
        self: Document, indices: list[int] | None = None, name: str = "Group"
    ) -> Any:
        """Wrap a contiguous run of layers in a new group. Returns it, or None.

        Created *around* rows that already exist rather than as an empty folder
        to drag things into, which is the same decision "empty groups are
        disallowed" makes from the other side -- and it means the contiguity
        invariant holds by construction at the moment of creation rather than
        being restored afterwards.

        A run that is not contiguous is refused rather than gathered up. Moving
        layers is a visible change to the painter's order, and a grouping
        gesture that silently reordered somebody's stack would be much worse
        than one that says no; the panel offers the run it can see.
        """
        if indices is None:
            indices = [self.stack.active_index]
        rows = sorted({max(0, min(int(i), len(self.stack) - 1)) for i in indices})
        if not rows or rows != list(range(rows[0], rows[-1] + 1)):
            return None
        order = self.member_uids()
        members = tuple(order[i] for i in rows)
        # The new group goes *inside* whatever already contains the run, which
        # is how nesting happens without a second entry point. Taken from the
        # first member, and the others must agree -- otherwise the run straddles
        # a boundary and grouping it would cut a span in half.
        parent = self.group_of.get(members[0])
        if any(self.group_of.get(uid) != parent for uid in members[1:]):
            return None
        self.commit_floating()
        node = gp.GroupNode(name=name)
        self._put_group(node, members, parent)
        self.history.push(GroupAddEdit(node, members, parent))
        return node

    def ungroup(self: Document, group_uid: int) -> bool:
        """Dissolve a group, leaving its members where they are in the stack."""
        node = self.groups.get(group_uid)
        if node is None:
            return False
        self.commit_floating()
        members = tuple(uid for uid, at in self.group_of.items() if at == group_uid)
        parent = self.group_of.get(group_uid)
        self._drop_group(group_uid)
        self.history.push(GroupDissolveEdit(node, members, parent))
        return True

    def set_group_props(self: Document, group_uid: int, **props: Any) -> bool:
        """Name, visibility, opacity, lock. One undo step, or none if nothing
        changed -- the rule ``set_layer_props`` follows one level down."""
        node = self.groups.get(group_uid)
        if node is None:
            return False
        before = {key: getattr(node, key) for key in props}
        if before == props:
            return False
        self._set_group_props(group_uid, dict(props))
        self.history.push(GroupPropsEdit(group_uid, before, dict(props)))
        return True

    def move_into_group(
        self: Document, index: int, group_uid: int | None, *, at_top: bool = True
    ) -> bool:
        """Move the layer at ``index`` into ``group_uid`` (or out to the root).

        One ``CompoundEdit`` of the membership change and the stack move that
        keeps the group contiguous, because the two are one gesture: a
        membership edit undone without its move would leave a group whose
        leaves are scattered through the stack, and the invariant everything
        else leans on would be false with nothing to say so.

        A group may not be moved into its own subtree, and the refusal is by
        name (``groups.descends_from``) rather than by letting ``ancestry``
        quietly stop at the repeat -- a group inside itself is not a state any
        op should be able to reach. (v1 moves stack rows; a *group* is moved by
        dissolving it and grouping again, which is what the panel offers.)
        """
        if not 0 <= index < len(self.stack):
            return False
        order = self.member_uids()
        member = order[index]
        if group_uid is not None:
            if group_uid not in self.groups:
                return False
            if gp.descends_from(self.group_of, group_uid, member):
                return False
        before = self.group_of.get(member)
        if before == group_uid:
            return False
        self.commit_floating()

        # Where the row has to end up for every span to stay contiguous, worked
        # out *before* the membership changes, against the stack as it stands.
        to = index
        if group_uid is not None:
            leaves = [
                order.index(uid)
                for uid in gp.leaves_of(self.group_of, order, group_uid)
            ]
            if leaves:
                # Landing at the top of the span keeps it contiguous whichever
                # side the layer came from: an index above the span's top is
                # adjacent to it, and one below it is inside it already.
                to = max(leaves) if at_top else min(leaves)
        elif before is not None:
            # Going to the root, and the harder direction: a row taken out of
            # the *middle* of a span leaves that span in two halves, so it has
            # to move out of the way as well. To the top of the outermost group
            # it is leaving, which -- because removing it closes the gap behind
            # it -- lands it directly above everything that stays. A row
            # already at either end of the span is out of the way where it is.
            chain = gp.ancestry(self.group_of, member)
            outer = chain[-1] if chain else before
            span = [
                order.index(uid)
                for uid in gp.leaves_of(self.group_of, order, outer)
                if uid != member
            ]
            if span and min(span) < index < max(span):
                to = max(span)
        edits: list[Any] = [MembershipEdit(member, before, group_uid)]
        self._set_membership(member, group_uid)
        if to != index:
            # Through ``_move_row_edit`` rather than ``move_layer``: the public
            # one pushes a step of its own and would also re-derive the
            # membership from the destination, undoing the line above.
            edits.append(self._move_row_edit(index, to))
        edits.extend(self._prune_group(before))
        self.history.push(CompoundEdit(edits) if len(edits) > 1 else edits[0])
        return True
