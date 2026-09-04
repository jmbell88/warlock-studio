"""Flourish on a document: a layer group whose cels are rendered from a recipe.

The door between :mod:`.flourish` -- which knows what a frame of an effect is
and nothing about documents -- and the grid. An effect *is* a group: one
track per recipe layer (one track in pixel mode, since the pixel pass runs on
the composite and a stack of quantised layers would not composite to it), one
tag per phase, and the recipe kept beside the group in ``Document.flourish``
so a later change to a slider can render the cels again.

**Nothing painted is overwritten silently.** Every render records a digest of
what it put in each cel (``sheetmerge.cell_digest``, the same function the
character-sheet merge uses). A regenerate replaces a cel only when the cel
still holds what the last render gave it; a cel the user painted on keeps the
paint and is *flagged*, and a person decides per cel. That is
``merge_render``'s rule, and the reason is the same: a cel wrongly kept is one
click to re-take, and a cel wrongly taken is an afternoon gone.

**One gesture, one step.** Inserting an effect is one ``one_step`` of every
edit it makes -- animating the document, the frames it adds, the tracks, the
tags, the group, and the recipe itself (``FlourishEdit``). A regenerate is one
step of its cel patches plus the recipe-and-digests it advances to, which is
``SheetBaseEdit``'s argument: a render undone without its digests would leave
the document's idea of "what the renderer last gave us" in the future.

Tracks are matched to recipe layers by the recipe layer's uid, recorded at
insert time; the composite in pixel mode uses ``COMPOSITE`` as its key. The
recipe's canvas need not be the document's: the effect lands centred, and the
offset is recorded so a regenerate lands in the same place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from . import groups as gp
from .anim_edits import AnimateEdit, CelSetEdit, FrameAddEdit, TagsEdit, TrackAddEdit
from .animation import Frame, Tag, Track
from .layers import Layer
from .undo import FlourishEdit, one_step

if TYPE_CHECKING:
    from .document import Document

#: The recipe-layer key a pixel-mode bake's single track is recorded under.
COMPOSITE = 0


@dataclass
class FlourishState:
    """Everything the document keeps about one effect group."""

    recipe: Any
    #: recipe layer uid (or ``COMPOSITE``) -> track uid.
    tracks: dict[int, int] = field(default_factory=dict)
    #: ``(track uid, frame uid)`` -> digest of the pixels the last render put there.
    digests: dict[tuple[int, int], str] = field(default_factory=dict)
    #: Cels a regenerate could not decide: the user painted them *and* the
    #: render changed. The paint stands until a person resolves it.
    conflicts: set[tuple[int, int]] = field(default_factory=set)
    #: Where the recipe's canvas sits on the document's, in pixels.
    offset: tuple[int, int] = (0, 0)
    #: Asset id -> straight uint8 RGBA texture a sprite or textured particle
    #: layer names. Immutable arrays by convention: an edit replaces the
    #: entry, never writes into it, so a snapshot can share them.
    assets: dict[str, np.ndarray] = field(default_factory=dict)

    def copy(self) -> FlourishState:
        return FlourishState(
            recipe=self.recipe,
            tracks=dict(self.tracks),
            digests=dict(self.digests),
            conflicts=set(self.conflicts),
            offset=tuple(self.offset),
            assets=dict(self.assets),
        )

    def next_asset_id(self, stem: str = "tex") -> str:
        n = 1
        while f"{stem}{n}" in self.assets:
            n += 1
        return f"{stem}{n}"


@dataclass(frozen=True)
class FlourishCounts:
    taken: int = 0
    kept: int = 0
    agreed: int = 0
    conflicts: int = 0
    added: int = 0

    def sentence(self) -> str:
        parts = []
        if self.taken:
            parts.append(f"rendered {self.taken} cel(s)")
        if self.added:
            parts.append(f"added {self.added}")
        if self.agreed:
            parts.append(f"{self.agreed} unchanged")
        if self.kept:
            parts.append(f"kept {self.kept} you painted")
        if self.conflicts:
            parts.append(f"flagged {self.conflicts} conflict(s)")
        return ", ".join(parts) + "." if parts else "Nothing to render."


def _digest(pixels: np.ndarray) -> str:
    from . import sheetmerge

    return sheetmerge.cell_digest(pixels)


def _place(cel: np.ndarray, size: tuple[int, int], offset: tuple[int, int]) -> np.ndarray:
    """``cel`` on a clear document-sized plane at ``offset``, cropped to fit."""
    width, height = size
    out = np.zeros((height, width, 4), dtype=np.uint8)
    ox, oy = offset
    h, w = cel.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(width, ox + w), min(height, oy + h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = cel[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    return out


def _centred(recipe: Any, size: tuple[int, int]) -> tuple[int, int]:
    return ((size[0] - recipe.width) // 2, (size[1] - recipe.height) // 2)


def _layer_keys(baked: Any) -> list[tuple[int, str]]:
    """``(key, track name)`` per track the bake wants, bottom first."""
    if baked.pixel:
        return [(COMPOSITE, baked.recipe.name)]
    return [(layer.uid, layer.name or layer.kind) for layer in baked.recipe.layers]


def _cel_for(baked: Any, key: int, flat_index: int) -> np.ndarray | None:
    """The bake's pixels for one track at one flat frame, or None when the
    layer is not active in that frame's phase."""
    cursor = 0
    for facing in baked.facings:
        for phase in baked.recipe.phases:
            n = phase.frames
            if flat_index < cursor + n:
                i = flat_index - cursor
                if key == COMPOSITE:
                    return facing.composites[phase.name][i]
                cels = facing.layers.get(phase.name, {}).get(key)
                return None if cels is None else cels[i]
            cursor += n
    return None


class FlourishOps:
    """The effect-group verbs. Mixed into :class:`Document`."""

    # -- queries -------------------------------------------------------------

    def flourish_state(self: Document, group_uid: int) -> FlourishState | None:
        return self.flourish.get(int(group_uid))

    def flourish_group_for(self: Document, member_uid: int) -> int | None:
        """The nearest enclosing group of ``member_uid`` that holds a recipe."""
        for guid in gp.ancestry(self.group_of, int(member_uid)):
            if guid in self.flourish:
                return guid
        return None

    def flourish_group_of_active(self: Document) -> int | None:
        if not len(self.stack):
            return None
        layer = self.stack[self.stack.active_index]
        return self.flourish_group_for(self.member_uid_of(layer))

    def flourish_conflicts(self: Document, group_uid: int) -> list[int]:
        """Flagged cels of the group, as frame indices."""
        state = self.flourish_state(group_uid)
        if state is None or self.anim is None:
            return []
        at = {frame.uid: i for i, frame in enumerate(self.anim.frames)}
        return sorted({at[f] for _t, f in state.conflicts if f in at})

    # -- insert ----------------------------------------------------------------

    def insert_flourish(self: Document, baked: Any) -> int:
        """Land a bake as a new group above the active row. One undo step.
        Returns the group uid."""
        recipe = baked.recipe
        self.commit_floating()
        edits: list[Any] = []
        parent = self._parent_of_active()
        if self.anim is None:
            edits.append(AnimateEdit(self.ensure_animation()))
        anim = self._require_anim()
        edits += self._flourish_ensure_frames(baked.frame_count, baked.fps)
        frame_uids = [frame.uid for frame in anim.frames[: baked.frame_count]]
        size = self.size
        offset = _centred(recipe, size)

        index = self.stack.active_index + 1
        tracks: dict[int, int] = {}
        digests: dict[tuple[int, int], str] = {}
        members: list[int] = []
        for k, (key, name) in enumerate(_layer_keys(baked)):
            track = Track(name=name)
            cels: dict[int, Layer] = {}
            for i, frame_uid in enumerate(frame_uids):
                cel = _cel_for(baked, key, i)
                if cel is None:
                    continue
                pixels = _place(cel, size, offset)
                cels[frame_uid] = Layer(pixels=pixels, name=name)
                digests[(track.uid, frame_uid)] = _digest(pixels)
            self._put_track(index + k, track, cels)
            edits.append(TrackAddEdit(index + k, track, cels, pinned=True))
            tracks[key] = track.uid
            members.append(track.uid)

        before_tags = list(anim.tags)
        after_tags = [*before_tags]
        for name, first, last, loop in baked.tags():
            after_tags.append(Tag(name=name, start=first, end=last, loop=loop))
        self._set_tags(after_tags)
        edits.append(TagsEdit(before_tags, after_tags))

        node = gp.GroupNode(name=recipe.name)
        self._put_group(node, tuple(members), parent)
        edits.append(gp.GroupAddEdit(node, tuple(members), parent))

        state = FlourishState(recipe=recipe, tracks=tracks, digests=digests, offset=offset)
        self.flourish[node.uid] = state
        edits.append(FlourishEdit(node.uid, None, state))
        self.history.push(one_step(edits))
        self.invalidate_all()
        return node.uid

    def _flourish_ensure_frames(self: Document, count: int, fps: int) -> list[Any]:
        """Append frames until the grid has ``count``. Returns the edits."""
        anim = self._require_anim()
        edits: list[Any] = []
        duration = max(1, int(round(1000.0 / max(int(fps), 1))))
        while len(anim.frames) < count:
            at = len(anim.frames)
            frame = Frame(duration_ms=duration)
            self._put_frame(at, frame, {})
            edits.append(FrameAddEdit(at, frame, {}, pinned=False))
        return edits

    # -- regenerate --------------------------------------------------------------

    def apply_flourish(
        self: Document, group_uid: int, baked: Any, *, force: bool = False
    ) -> FlourishCounts:
        """Land a fresh render on an existing effect group. One undo step.

        A cel that still holds the last render takes the new one; a cel the
        user painted keeps the paint and is flagged unless ``force``; a cel
        whose paint already equals the new render is left alone. Recipe layers
        with no track yet (added in the inspector) get one, inside the group.
        """
        group_uid = int(group_uid)
        state = self.flourish_state(group_uid)
        if state is None:
            raise ValueError("that group is not a Flourish effect")
        if group_uid not in self.groups:
            raise ValueError("that effect's group is no longer in the document")
        self.commit_floating()
        anim = self._require_anim()
        edits: list[Any] = []
        edits += self._flourish_ensure_frames(baked.frame_count, baked.fps)
        frame_uids = [frame.uid for frame in anim.frames[: baked.frame_count]]
        size = self.size
        offset = state.offset
        box = (0, 0, size[0], size[1])
        tracks = dict(state.tracks)
        digests: dict[tuple[int, int], str] = {}
        conflicts: set[tuple[int, int]] = set()
        counts = {"taken": 0, "kept": 0, "agreed": 0, "conflicts": 0, "added": 0}
        present = {track.uid for track in anim.tracks}

        # Every target that would raise must be found before the loop below
        # writes anything: a raise partway through leaves earlier cels
        # already mutated and no undo step pushed to cover them (finding #1).
        for key, _name in _layer_keys(baked):
            track_uid = tracks.get(key)
            if track_uid is None or track_uid not in present:
                continue
            for frame_uid in frame_uids:
                if anim.cels.get((track_uid, frame_uid)) is not None and anim.is_linked(
                    track_uid, frame_uid
                ):
                    raise ValueError(
                        "unlink this effect's cels before regenerating: a linked cel "
                        "cannot take two different renders"
                    )

        for key, name in _layer_keys(baked):
            track_uid = tracks.get(key)
            if track_uid is None or track_uid not in present:
                track_uid = self._flourish_add_track(group_uid, name, edits)
                tracks[key] = track_uid
                counts["added"] += 1
            self._track_by_uid(track_uid)  # raises if the track has gone
            for i, frame_uid in enumerate(frame_uids):
                cel = _cel_for(baked, key, i)
                pixels = None if cel is None else _place(cel, size, offset)
                existing = anim.cels.get((track_uid, frame_uid))
                fresh = None if pixels is None else _digest(pixels)
                if existing is None:
                    if pixels is not None and pixels[..., 3].any():
                        layer = Layer(pixels=pixels, name=name)
                        self._set_cel(track_uid, frame_uid, layer)
                        edits.append(CelSetEdit(track_uid, frame_uid, None, layer, pinned=True))
                        counts["taken"] += 1
                    if fresh is not None:
                        digests[(track_uid, frame_uid)] = fresh
                    continue
                if pixels is None:
                    pixels = np.zeros_like(existing.pixels)
                    fresh = _digest(pixels)
                if anim.is_linked(track_uid, frame_uid):
                    raise ValueError(
                        "unlink this effect's cels before regenerating: a linked cel "
                        "cannot take two different renders"
                    )
                recorded = state.digests.get((track_uid, frame_uid))
                current = _digest(existing.pixels)
                if current == fresh:
                    counts["agreed"] += 1
                elif force or recorded is None or current == recorded:
                    before = existing.pixels.copy()
                    existing.pixels[...] = pixels
                    edit = self._patch_edit_for(existing, box, before)
                    if edit is not None:
                        self._stamp_layer(existing.uid)
                        edits.append(edit)
                    counts["taken"] += 1
                else:
                    counts["kept"] += 1
                    counts["conflicts"] += 1
                    conflicts.add((track_uid, frame_uid))
                digests[(track_uid, frame_uid)] = fresh

        after = FlourishState(
            recipe=baked.recipe, tracks=tracks, digests=digests, conflicts=conflicts, offset=offset
        )
        self.flourish[group_uid] = after
        edits.append(FlourishEdit(group_uid, state.copy(), after))
        if self.groups[group_uid].name == state.recipe.name != baked.recipe.name:
            self.groups[group_uid].name = baked.recipe.name
        self.history.push(one_step(edits))
        self.invalidate_all()
        return FlourishCounts(**counts)

    def _flourish_add_track(self: Document, group_uid: int, name: str, edits: list[Any]) -> int:
        """A new empty track at the top of the group's span, as a member."""
        self._require_anim()
        order = self.member_uids()
        leaves = gp.leaves_of(self.group_of, order, group_uid)
        top = max((order.index(uid) for uid in leaves), default=self.stack.active_index)
        index = top + 1
        track = Track(name=name)
        self._put_track(index, track, {})
        edits.append(TrackAddEdit(index, track, {}, pinned=False))
        self._set_membership(track.uid, group_uid)
        edits.append(gp.MembershipEdit(track.uid, None, group_uid))
        return track.uid

    # -- resolve and detach -----------------------------------------------------

    def resolve_flourish(self: Document, group_uid: int, frames: list[int]) -> bool:
        """Accept the hand edit on these frames: the flags come off. One step."""
        state = self.flourish_state(group_uid)
        if state is None or self.anim is None or not frames:
            return False
        uids = {self.anim.frames[i].uid for i in frames if 0 <= i < len(self.anim.frames)}
        after = state.copy()
        after.conflicts = {(t, f) for t, f in state.conflicts if f not in uids}
        if after.conflicts == state.conflicts:
            return False
        self.flourish[int(group_uid)] = after
        self.history.push(FlourishEdit(int(group_uid), state.copy(), after))
        return True

    def add_flourish_asset(
        self: Document, group_uid: int, pixels: np.ndarray, *, stem: str = "tex"
    ) -> str:
        """Hold a texture beside the recipe. One step; returns its id."""
        state = self.flourish_state(group_uid)
        if state is None:
            raise ValueError("that group is not a Flourish effect")
        if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.size == 0:
            raise ValueError("a texture is a non-empty RGBA plane")
        # Not charged against ``FlourishEdit.cost`` (which counts ``digests``
        # only), deliberately: assets are shared by convention -- ``copy()``
        # is a shallow dict copy, so every snapshot from an insert to the
        # latest edit holds the *same* array objects, not a duplicate per
        # step. Byte-costing them the way ``digests`` is costed would charge
        # the undo budget once per snapshot for memory that is in fact held
        # once.
        after = state.copy()
        asset_id = after.next_asset_id(stem)
        after.assets[asset_id] = np.ascontiguousarray(pixels, dtype=np.uint8).copy()
        self.flourish[int(group_uid)] = after
        self.history.push(FlourishEdit(int(group_uid), state.copy(), after))
        return asset_id

    def remove_flourish_asset(self: Document, group_uid: int, asset_id: str) -> bool:
        state = self.flourish_state(group_uid)
        if state is None or asset_id not in state.assets:
            return False
        after = state.copy()
        after.assets.pop(asset_id, None)
        self.flourish[int(group_uid)] = after
        self.history.push(FlourishEdit(int(group_uid), state.copy(), after))
        return True

    def selection_cutout(self: Document) -> np.ndarray | None:
        """The active layer's pixels inside the selection, the mask folded
        into their alpha, cropped to the selection's bounds -- ``copy``'s
        picture without the clipboard. None with no selection."""
        from ._doc_selection import _masked_alpha

        if self.mask is None:
            return None
        bounds = self.mask.bounds
        box = self.clip(bounds) if bounds else None
        if box is None:
            return None
        x0, y0, x1, y1 = box
        crop = self.mask.mask[y0:y1, x0:x1]
        return _masked_alpha(self.stack.active.pixels[y0:y1, x0:x1], crop)

    def insert_flourish_track(
        self: Document, group_uid: int, name: str, cels: dict[int, np.ndarray]
    ) -> int:
        """A *snapshot* track inside the effect group -- restyled keyframes,
        say -- holding ``cels`` keyed by frame index. One step. Recorded under
        a negative key so a regenerate, which walks the recipe's layers, leaves
        it alone: what the image model painted is not the recipe's to redo."""
        state = self.flourish_state(group_uid)
        if state is None:
            raise ValueError("that group is not a Flourish effect")
        anim = self._require_anim()
        self.commit_floating()
        edits: list[Any] = []
        needed = max(cels) + 1 if cels else 0
        edits += self._flourish_ensure_frames(needed, state.recipe.fps)
        track_uid = self._flourish_add_track(int(group_uid), name, edits)
        after = state.copy()
        key = min([k for k in after.tracks if k < 0] + [0]) - 1
        after.tracks[key] = track_uid
        size = self.size
        for index, pixels in cels.items():
            frame = anim.frames[index]
            placed = _place(pixels, size, (0, 0)) if pixels.shape[:2] != size[::-1] else pixels
            layer = Layer(pixels=np.ascontiguousarray(placed, dtype=np.uint8).copy(), name=name)
            self._set_cel(track_uid, frame.uid, layer)
            edits.append(CelSetEdit(track_uid, frame.uid, None, layer, pinned=True))
            after.digests[(track_uid, frame.uid)] = _digest(layer.pixels)
        self.flourish[int(group_uid)] = after
        edits.append(FlourishEdit(int(group_uid), state.copy(), after))
        self.history.push(one_step(edits))
        self.invalidate_all()
        return track_uid

    def set_flourish_recipe(self: Document, group_uid: int, recipe: Any) -> bool:
        """Change the recipe without rendering -- a rename, say. One step."""
        state = self.flourish_state(group_uid)
        if state is None or state.recipe == recipe:
            return False
        after = replace(state.copy(), recipe=recipe)
        self.flourish[int(group_uid)] = after
        self.history.push(FlourishEdit(int(group_uid), state.copy(), after))
        return True

    def detach_flourish(self: Document, group_uid: int) -> bool:
        """Forget the recipe and leave the layers as plain layers. One step."""
        state = self.flourish_state(group_uid)
        if state is None:
            return False
        self.flourish.pop(int(group_uid), None)
        self.history.push(FlourishEdit(int(group_uid), state.copy(), None))
        return True

    def _set_flourish(self: Document, group_uid: int, state: FlourishState | None) -> None:
        """Raw hook for ``FlourishEdit``."""
        if state is None:
            self.flourish.pop(int(group_uid), None)
        else:
            self.flourish[int(group_uid)] = state
