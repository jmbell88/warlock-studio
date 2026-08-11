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
from .anim_edits import TrackAddEdit, TrackMoveEdit, TrackPropsEdit, TrackRemoveEdit
from .animation import Track
from .layers import Layer, LayerStack, new_uid
from .undo import (
    CompoundEdit,
    LayerAddEdit,
    LayerMoveEdit,
    LayerPropsEdit,
    LayerRemoveEdit,
    PatchEdit,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document


class LayerOps:
    """Layer and track structure, mixed into :class:`~.document.Document`."""

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

    def duplicate_layer(self: Document, index: int | None = None) -> Layer:
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

    def remove_layer(self: Document, index: int | None = None) -> bool:
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

    def move_layer(self: Document, index: int, to: int) -> bool:
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
    def can_restructure(self: Document) -> bool:
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

    def merge_down(self: Document, index: int | None = None) -> bool:
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

    def flatten_layers(self: Document) -> None:
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
            # ``to_uint8_255``: the same narrowing, in the one place that owns
            # it, and with the native kernel behind it.
            layer.pixels[:, :, 3] = cp.to_uint8_255(
                layer.pixels[:, :, 3].astype(np.float32) * weight
            )
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

    def _do_flatten(self: Document, uid: int) -> None:
        flat = self.stack.flatten()
        self.stack = LayerStack([Layer(pixels=flat, name="Flattened", uid=uid)], 0)
