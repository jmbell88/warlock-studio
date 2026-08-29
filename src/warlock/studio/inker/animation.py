"""The animation grid: frames across, tracks down, cels in the cells.

A document without an ``Animation`` is exactly the document this editor has
always had, and that is the point -- ``Document.anim is None`` takes every path
below out of the picture, so the still-image editor is not paying for a feature
it is not using.

Three decisions shape everything here.

**A track is a layer's identity, not a layer.** Name, opacity, visibility and
blend mode live on the ``Track``, once, and are copied onto whichever ``Layer``
materialises for the current frame. The alternative -- storing them per cel --
means hiding a layer is an edit to every frame it appears in, and a frame with
no cel has nowhere to put the answer at all.

**A cel is a ``Layer`` object, and a link is two slots holding the same
object.** There is no link table and no "linked to" field: ``cels[(t, f1)] is
cels[(t, f2)]`` *is* the link, so an edit to a linked cel shows up on every
frame it occupies for free, with no propagation step that could be forgotten.
The price is that anything walking the whole grid must walk
``unique_cel_layers()`` instead, or a rotate applied per slot rotates a
three-frame linked background three times. Every such caller is one line away
from that bug, so the helper exists and is named for it.

**An absent cel is an absent key.** The grid is sparse: ``cels`` holds only the
slots that were drawn in. Materialising a frame fills the gaps with placeholder
layers that share one read-only transparent plane -- read-only because nothing
may ever write to a placeholder (a write autovivifies a real cel first), and a
shared writable plane would let one such bug silently paint into every empty
cel in the document at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from . import composite as cp
from .layers import Layer, new_uid

#: What a new frame lasts. 100 ms is Aseprite's default and reads as 10 fps.
DEFAULT_DURATION_MS = 100

#: Bounds on a frame's duration. The floor is one millisecond because zero is a
#: frame that can never be reached by ``advance`` and would hang a loop that
#: divides by it; the ceiling is a minute, past which the timeline is being
#: used as a storage format rather than an animation.
MIN_DURATION_MS = 1
MAX_DURATION_MS = 60_000

__all__ = [
    "ACTION_FRAMES",
    "DEFAULT_DURATION_MS",
    "DIRECTIONS",
    "DIRECTION_COUNTS",
    "DIRECTION_ORDER",
    "DIRECTION_YAWS",
    "MAX_DURATION_MS",
    "MIN_DURATION_MS",
    "SHEET_KINDS",
    "SPRITE_DIRECTIONS",
    "SPRITE_YAWS",
    "Animation",
    "DirectionalLayout",
    "Frame",
    "Tag",
    "Track",
    "advance",
    "clamp_duration",
]


#: The four directions a four-direction sheet carries, in the order a frame
#: index means, and the yaw each one stands for.
#:
#: Repeated from ``pipelines.spritesynth`` rather than imported, because this
#: package imports nothing outward -- that is what makes the inker usable
#: headless and is pinned by a test. The two copies agreeing is therefore not
#: something either module can enforce, so a cross-check test
#: (``tests/test_sprite_geometry_agreement.py``) owns it instead, and is the
#: only place the agreement is asserted.
DIRECTION_ORDER = ("front", "left", "right", "back")
DIRECTION_YAWS = {"front": 0, "left": 90, "right": 270, "back": 180}

#: All eight direction names and their yaws, degrees clockwise from the front
#: view. The third copy of this table in the repo -- ``pipelines.charsheet``
#: owns it, ``pipelines.spritesynth`` imports it from there, and this package
#: may import neither. Same arrangement as :data:`DIRECTION_ORDER` above and
#: same single owner of the agreement.
SPRITE_YAWS: dict[str, int] = {
    "front": 0,
    "front_left": 45,
    "left": 90,
    "back_left": 135,
    "back": 180,
    "back_right": 225,
    "right": 270,
    "front_right": 315,
}

DIRECTION_COUNTS: tuple[int, ...] = (4, 8)

#: The order a *row* index means, per direction count. Four is
#: :data:`DIRECTION_ORDER` -- the order every sprite draft on disk was written
#: in, which is why it is not the clockwise sweep the eight-direction tuple is.
SPRITE_DIRECTIONS: dict[int, tuple[str, ...]] = {
    4: DIRECTION_ORDER,
    8: (
        "front",
        "front_left",
        "left",
        "back_left",
        "back",
        "back_right",
        "right",
        "front_right",
    ),
}

#: How many frames each action carries. ``pipelines.spritesynth.ACTIONS``' table
#: again, for the reason the direction tables are here twice; the shared five
#: also have to agree with ``pipelines.charsheet.ANIMATIONS``, which is a
#: separate claim owned by ``tests/test_spritesynth.py``.
ACTION_FRAMES: dict[str, int] = {
    "idle": 4,
    "walk": 8,
    "run": 8,
    "attack": 6,
    "cast": 6,
    "hurt": 4,
    "jump": 6,
}

#: ``kind -> (columns, rows, frames per direction, directions)``. The grid is
#: *derived* from the kind and never stored, so a document cannot carry a layout
#: that disagrees with itself.
#:
#: The two legacy kinds are literals because they are literals in
#: ``spritesynth`` too: a ``turnaround`` folds four directions into a 2x2 grid,
#: which is the one layout here where the row count and the direction count are
#: different numbers. Every planned kind is one direction per row and one frame
#: per column, which is what makes :meth:`DirectionalLayout.cell`'s
#: ``row = index // columns`` true for all of them at once.
#:
#: **``walk`` and ``walk4`` are two different sheets**, and the near-collision
#: is the trap here. Legacy ``walk`` is a four-frame cycle over four directions;
#: ``walk4`` is :data:`ACTION_FRAMES`' eight-frame cycle over the same four.
#: The frame count is the action's, not the direction count's, so neither can be
#: an alias of the other without silently halving or doubling a stored cycle.
SHEET_KINDS: dict[str, tuple[int, int, int, int]] = {
    "turnaround": (2, 2, 1, 4),
    "walk": (4, 4, 4, 4),
    **{
        f"{action}{count}": (frames, count, frames, count)
        for action, frames in ACTION_FRAMES.items()
        for count in DIRECTION_COUNTS
    },
}


@dataclass(frozen=True)
class DirectionalLayout:
    """What a document's frames *mean* when they came from a sprite sheet.

    Only the kind is state. Everything else -- how many columns the export
    grid has, which direction frame 9 is, what yaw goes in the sidecar -- is
    computed from it, so there is no second copy to drift and nothing for an
    edit to half-update.

    It is construction-time state, like ``Document.matte``: no V1 undo op
    edits it, and a whole-canvas snapshot restores by mutating the existing
    ``Animation`` in place, so it survives undo without any snapshot change.
    Adding or removing frames *can* leave the timeline no longer matching the
    grid -- that is caught at export, by a refusal naming both counts, rather
    than by forbidding the edit: a sheet is an ordinary animation and being
    able to draw a fifth frame in it is not a bug.
    """

    kind: str

    @classmethod
    def of(cls, kind: object) -> DirectionalLayout | None:
        """The layout for ``kind``, or None for anything this build cannot draw.

        None rather than raising, for ``Tag.__post_init__``'s reason: a layout
        arrives from a file as often as from this process, and a kind a later
        build introduced must cost the document its grid, not its openability.
        That tolerance is what lets a wider set of sheet kinds ship with no
        document migration behind it -- an older build opening a ``walk8`` draft
        loses the grid and keeps the frames.
        """
        return cls(str(kind)) if kind in SHEET_KINDS else None

    @property
    def columns(self) -> int:
        return SHEET_KINDS[self.kind][0]

    @property
    def rows(self) -> int:
        return SHEET_KINDS[self.kind][1]

    @property
    def frames_per_direction(self) -> int:
        return SHEET_KINDS[self.kind][2]

    @property
    def direction_count(self) -> int:
        return SHEET_KINDS[self.kind][3]

    @property
    def directions(self) -> tuple[str, ...]:
        """The direction names, in the order a row index means."""
        return SPRITE_DIRECTIONS[self.direction_count]

    @property
    def frame_count(self) -> int:
        return self.direction_count * self.frames_per_direction

    def cell(self, index: int) -> tuple[int, int, str, int, int]:
        """``(row, col, direction, yaw, frame)`` for timeline frame ``index``."""
        if not 0 <= index < self.frame_count:
            raise IndexError(f"frame {index} is outside a {self.kind} sheet")
        direction = self.directions[index // self.frames_per_direction]
        return (
            index // self.columns,
            index % self.columns,
            direction,
            SPRITE_YAWS[direction],
            index % self.frames_per_direction,
        )


#: Every property a track owns and an editor may set: the six that are copied
#: down onto a materialised ``Layer``, plus ``continuous``, which is not.
#:
#: An allowlist rather than a `hasattr` check, because the writers use
#: ``setattr``: an unknown key would otherwise mint a new attribute on the
#: track, silently, and lose it at the next save. It lives here beside the
#: dataclass so the two cannot drift.
TRACK_PROPS = frozenset(
    {"name", "opacity", "visible", "blend", "alpha_lock", "locked", "continuous"}
)

#: The track properties **copied down onto a materialised ``Layer``**, in one
#: place because there are four sites that copy them -- :meth:`Track.of`,
#: :meth:`Animation.placeholder`, :meth:`Animation.layers_for` and
#: ``Document._ensure_cel_for`` -- and a hand-maintained list at each is how
#: ``background`` and ``reference`` came to be copied by one of the four and
#: forgotten by the other three. The consequence was not cosmetic: a reference
#: track's cels are write-locked through ``Document.write_locked``, which reads
#: the flag off the *layer*, so the row was editable on every frame but the one
#: the flag was set on; and a background track composited opaque on that frame
#: and transparent on the rest.
#:
#: **Not the same set as ``TRACK_PROPS``** above, and deliberately so. That one
#: is the ``setattr`` allowlist for ``set_layer_props``; these two flags are
#: outside it because they carry their own ``LayerFlagEdit``, and ``continuous``
#: is in it but is never copied down (a cel has no use for what decides
#: autovivification). The two lists answer different questions and are kept
#: apart so neither is edited on the other's behalf.
CEL_PROPS: tuple[str, ...] = (
    "name",
    "opacity",
    "visible",
    "blend",
    "alpha_lock",
    "locked",
    "background",
    "reference",
)


@dataclass
class Track:
    """One row of the grid: a layer's identity and its properties.

    The properties are authoritative over the materialised ``Layer``'s own --
    ``Document._materialize_frame`` copies them down every time -- so a track is
    the only place they are edited and a cel never disagrees with its row.
    """

    name: str = "Layer"
    opacity: float = 1.0
    visible: bool = True
    blend: str = "normal"
    alpha_lock: bool = False
    #: The sixth track property -- content lock. A track's properties are
    #: copied down onto whichever ``Layer`` materialises for the current frame,
    #: so this list and the three places that copy it (``of``, ``placeholder``,
    #: ``layers_for``, plus ``Document._ensure_cel_for``) have to agree; a
    #: property left out of one of them is a lock that is on in the panel and
    #: off at the door.
    locked: bool = False
    #: The seventh and eighth track properties (6.5): a real background layer
    #: and a reference layer. On the list for the reason the sixth is -- a
    #: property left out of one of the four places that copy it is a flag that
    #: is on in the panel and off at the door.
    background: bool = False
    reference: bool = False
    #: Whether a fresh cel on this row starts as a *copy* of the drawing before
    #: it rather than blank -- Aseprite's continuous layer, and how a held pose
    #: or a background is carried forward.
    #:
    #: Deliberately **not** one of the six properties above and deliberately
    #: absent from :meth:`props`. Those are copied down onto whichever ``Layer``
    #: materialises for the current frame, because they describe how a cel is
    #: shown; this describes what autovivification *writes*, so a cel has no
    #: use for it and a copy would only be a second answer to drift from.
    continuous: bool = False
    #: Which tileset (a :class:`~.tiles.TilesetSlot` uid) this track's cels
    #: draw from, or ``None`` for an ordinary raster track. ``continuous``'s
    #: shape exactly, and for the same reason: this describes what kind of
    #: cel :meth:`~.document.Document._ensure_cel_for` autovivifies on this
    #: row, not how an existing cel is *shown* -- so it is deliberately
    #: **not** one of the six copied-down properties, **not** in
    #: :meth:`props`, and **not** in ``TRACK_PROPS``. A tilemap track binds
    #: exactly one tileset (Aseprite's rule); this field is that binding, is
    #: authoring state serialized in ``tiles.json``, and is never copied down
    #: onto a materialised ``Layer`` because a plain ``Layer`` has nowhere to
    #: put the answer.
    tileset_uid: int | None = None
    uid: int = field(default_factory=new_uid)

    @classmethod
    def of(cls, layer: Layer) -> Track:
        """The track that describes an existing layer, uid and all.

        The uid is *shared* with the layer deliberately. Converting a still
        document to an animated one must not renumber anything: a patch already
        on the undo stack addresses the layer by uid, and the layer object
        itself becomes the first frame's cel, so the two identities are the same
        identity and giving the track a fresh one would only invent a mapping
        for later code to get wrong.
        """
        return cls(
            uid=layer.uid,
            **{key: getattr(layer, key) for key in CEL_PROPS},
        )

    def props(self) -> dict[str, object]:
        return {
            "name": self.name,
            "opacity": self.opacity,
            "visible": self.visible,
            "blend": self.blend,
            "alpha_lock": self.alpha_lock,
            "locked": self.locked,
            "background": self.background,
            "reference": self.reference,
        }


@dataclass
class Frame:
    """One column of the grid: how long it is held, and nothing else.

    Frames carry no pixels. Which cels they contain is a property of ``cels``,
    keyed by this frame's uid, so reordering frames is a list operation that
    moves no image data.
    """

    duration_ms: int = DEFAULT_DURATION_MS
    uid: int = field(default_factory=new_uid)

    def __post_init__(self) -> None:
        self.duration_ms = clamp_duration(self.duration_ms)


#: Which way playback moves through a tag's span.
#:
#: ``pingpong`` is the one that earns the field. Reverse is expressible by
#: drawing the frames the other way round, and a plain forward loop is the
#: default -- but a there-and-back swing (a torch flickering, an idle breath, a
#: pendulum) drawn as frames costs the whole span again in cels, all of them
#: linked duplicates of frames already in the file, and every edit to the middle
#: of the swing then has to be made twice.
DIRECTIONS = ("forward", "reverse", "pingpong")


@dataclass
class Tag:
    """A named, optionally looping range of frames -- "walk", "idle", "hit".

    ``start`` and ``end`` are frame *indices*, inclusive at both ends, because
    that is how a user reads "frames 3 to 7" and how the timeline draws it. They
    are indices rather than uids for the one reason indices are ever the right
    choice here: a tag names a span of the timeline, so inserting a frame inside
    it should widen it, which uids would not do.

    ``direction`` and ``loop`` are separate questions and both are needed:
    direction is the path through the span, loop is whether reaching the end of
    that path starts it again. A non-looping ping-pong plays out and back once,
    which is what a "swing" is; a looping one never stops.
    """

    name: str = "tag"
    start: int = 0
    end: int = 0
    loop: bool = True
    direction: str = "forward"
    #: How many times the span plays before stopping. **0 is "the loop flag
    #: decides"** -- today's semantics exactly, which is what makes this a
    #: trailing field with no migration behind it: every document ever written
    #: reads back as 0 and plays as it always did. N > 0 plays the span N times
    #: and stops, and ``loop`` is deliberately *not* folded into it, because
    #: folding would have to invent an answer for the files already on disk.
    #:
    #: A ping-pong counts one out-and-back as one, which is what "play it three
    #: times" means about a swing.
    #:
    #: **A finite repeat does not fall through past its span** -- playback is
    #: confined to ``loop_range``, so the clip stops at the tag's end rather
    #: than carrying on into the frames after it. That is a deliberate
    #: divergence from Aseprite and the manual says so.
    repeat: int = 0

    def __post_init__(self) -> None:
        # Coerced rather than refused: a tag arrives from a file as often as
        # from the menu, and a spelling this build does not carry is a tag that
        # should still play, forwards, rather than a document that will not open.
        if self.direction not in DIRECTIONS:
            self.direction = "forward"
        # Coerced for ``direction``'s reason: a repeat count arrives from a
        # file as often as from the menu, and a negative or unparseable one is
        # a tag that should still play rather than a document that will not
        # open. Zero is the "loop flag decides" default, so it is also the
        # right answer for nonsense.
        try:
            self.repeat = max(0, int(self.repeat))
        except (TypeError, ValueError):
            self.repeat = 0


def clamp_duration(ms: object) -> int:
    try:
        value = int(ms)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return DEFAULT_DURATION_MS
    return max(MIN_DURATION_MS, min(value, MAX_DURATION_MS))


@dataclass
class Animation:
    """The grid itself, plus the playhead.

    ``tracks`` is bottom-first, mirroring ``LayerStack``, so a track index and a
    stack index are the same number and every layer-ordering rule the editor
    already has keeps applying unchanged.

    ``current`` is the playhead and is **view state**: moving it pushes no undo
    step, exactly as ``set_active_layer`` pushes none. A document that asked to
    be saved because the user looked at another frame would make ``dirty``
    dishonest, which is the one property the save path leans on.
    """

    tracks: list[Track] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    cels: dict[tuple[int, int], Layer] = field(default_factory=dict)
    tags: list[Tag] = field(default_factory=list)
    current: int = 0

    #: Set when these frames are a directional sprite sheet's cells, in order.
    #: None for every ordinary animation, which is the default and takes the
    #: fixed-grid export path out of the picture entirely.
    layout: DirectionalLayout | None = None

    #: Stable uids for the empty slots, so a placeholder materialised twice is
    #: the same identity twice. Not the layers themselves: those share one
    #: plane and are cheap to rebuild, while a uid that changed under a caller
    #: holding it would be a bug with no symptom until much later.
    _placeholder_uids: dict[tuple[int, int], int] = field(
        default_factory=dict, repr=False
    )
    _blank: np.ndarray | None = field(default=None, repr=False)

    # -- lookups -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def frame(self) -> Frame:
        """The frame the playhead is on.

        ``current`` is clamped rather than validated, so no caller has to keep
        it in range across an insert or a reorder. The one precondition is that
        the timeline is not empty -- an ``Animation`` with no frames has no
        answer to give, and ``Document`` keeps a floor of one so the question
        never arises.
        """
        if not self.frames:
            raise IndexError("an animation with no frames has no current frame")
        self.current = max(0, min(self.current, len(self.frames) - 1))
        return self.frames[self.current]

    def frame_index(self, uid: int) -> int:
        for i, frame in enumerate(self.frames):
            if frame.uid == uid:
                return i
        raise KeyError(uid)

    def track_index(self, uid: int) -> int:
        for i, track in enumerate(self.tracks):
            if track.uid == uid:
                return i
        raise KeyError(uid)

    def cel(self, track_uid: int, frame_uid: int) -> Layer | None:
        return self.cels.get((track_uid, frame_uid))

    def duration_ms(self) -> int:
        return sum(frame.duration_ms for frame in self.frames)

    # -- the identity rules -------------------------------------------------

    def unique_cel_layers(self) -> Iterator[Layer]:
        """Every distinct cel exactly once, in frame-then-track order.

        The order is deterministic so the ORA writer's ``data/cel*.png`` names
        are stable between saves of an unchanged document; the uniqueness is
        what stops a whole-grid operation touching a linked cel once per slot.
        Identity is ``id(layer)``, not ``layer.uid`` -- a snapshot restore can
        legitimately put two distinct objects with the same uid in play while it
        is swapping them, and the question here is only "have I already handled
        this array".
        """
        seen: set[int] = set()
        for frame in self.frames:
            for track in self.tracks:
                layer = self.cels.get((track.uid, frame.uid))
                if layer is None or id(layer) in seen:
                    continue
                seen.add(id(layer))
                yield layer

    def slots_of(self, layer: Layer) -> list[tuple[int, int]]:
        """Every ``(track_uid, frame_uid)`` holding *this* object."""
        return [key for key, value in self.cels.items() if value is layer]

    def frame_uids_of_layer(self, layer_uid: int) -> set[int]:
        """The frames whose flatten a write to ``layer_uid`` would change.

        By uid rather than by object, because the caller is an undo patch and a
        patch only ever carries a uid. A linked cel answers with every frame it
        occupies, which is exactly the set of cached flattens to throw away.
        """
        return {
            frame_uid
            for (_, frame_uid), layer in self.cels.items()
            if layer.uid == layer_uid
        }

    def is_linked(self, track_uid: int, frame_uid: int) -> bool:
        layer = self.cels.get((track_uid, frame_uid))
        if layer is None:
            return False
        count = 0
        for other in self.cels.values():
            if other is layer:
                count += 1
                if count > 1:
                    return True
        return False

    # -- materialisation ----------------------------------------------------

    def blank_plane(self, width: int, height: int) -> np.ndarray:
        """The one transparent plane every empty cel shows, read-only.

        Shared because allocating a full canvas per empty slot would cost more
        than the drawing does -- ten tracks over twenty empty frames at 2048
        square is 3.2 GiB of nothing. Read-only because sharing is only safe
        while nothing writes, and a stray write would otherwise appear in every
        empty cel in the document with no way to tell where it came from.
        """
        if (
            self._blank is None
            or self._blank.shape[0] != height
            or self._blank.shape[1] != width
        ):
            plane = cp.empty(width, height)
            plane.flags.writeable = False
            self._blank = plane
        return self._blank

    def placeholder(self, track: Track, frame: Frame, size: tuple[int, int]) -> Layer:
        """The stand-in ``Layer`` for an empty slot.

        It is an ordinary layer as far as the stack, the compositor and the
        panes are concerned -- which is the whole trick, since it means nothing
        downstream of ``Document.stack`` needs to know the grid exists.
        """
        key = (track.uid, frame.uid)
        uid = self._placeholder_uids.get(key)
        if uid is None:
            uid = new_uid()
            self._placeholder_uids[key] = uid
        width, height = size
        return Layer(
            pixels=self.blank_plane(width, height),
            uid=uid,
            **{key: getattr(track, key) for key in CEL_PROPS},
        )

    def is_placeholder(self, layer: Layer) -> bool:
        return self._blank is not None and layer.pixels is self._blank

    def forget_placeholders(self, *, track_uid: int | None = None,
                            frame_uid: int | None = None) -> None:
        """Drop remembered placeholder uids for a removed row or column.

        Uids are per-process and monotonic, so keeping them would leak a slow
        dictionary rather than break anything -- but a frame uid can never come
        back, and a stale entry would outlive every reader of it.
        """
        for key in [
            key
            for key in self._placeholder_uids
            if (track_uid is not None and key[0] == track_uid)
            or (frame_uid is not None and key[1] == frame_uid)
        ]:
            del self._placeholder_uids[key]

    def layers_for(self, frame: Frame, size: tuple[int, int]) -> list[Layer]:
        """The bottom-first layer list this frame presents as an ordinary stack.

        Track properties are copied down onto real cels as well as placeholders:
        the track is authoritative, and a cel that kept its own stale opacity
        would composite differently on the frame it was drawn on than on the
        frame it was linked into.
        """
        out: list[Layer] = []
        for track in self.tracks:
            layer = self.cels.get((track.uid, frame.uid))
            if layer is None:
                out.append(self.placeholder(track, frame, size))
                continue
            for key in CEL_PROPS:
                setattr(layer, key, getattr(track, key))
            out.append(layer)
        return out

    # -- playback -----------------------------------------------------------

    def tag_span(self, tag: Tag) -> tuple[int, int]:
        """``tag``'s range as it applies to the timeline *now*.

        Tags are clamped when they are written (``Document._clamped_tag``), but
        deleting frames does not rewrite them -- deliberately, so that undoing
        the deletion brings the tag back at its authored range rather than at
        whatever survived. That leaves a stored range that can point past the
        end, and the reason this is a method rather than a comment is what used
        to happen when it did: ``active_tag`` compared against the raw numbers,
        so a tag whose whole span was past the last frame contained no index at
        all and simply stopped existing -- it never played, never highlighted,
        and nothing anywhere said why. Clamping here brings it back to the tail
        of the timeline instead, and it springs back the moment the frames do.
        """
        last = max(len(self.frames) - 1, 0)
        start = max(0, min(int(tag.start), last))
        end = max(0, min(int(tag.end), last))
        return min(start, end), max(start, end)

    def active_tag(self, index: int) -> Tag | None:
        """The tag whose range contains a frame index, innermost first.

        Innermost because tags nest in practice -- a short "hit" inside a long
        "combat" -- and the narrower one is the one the user just clicked into.
        Measured on the clamped span (:meth:`tag_span`), so a tag left pointing
        past the end by a frame deletion is still reachable.
        """
        spans = [(tag, self.tag_span(tag)) for tag in self.tags]
        containing = [(tag, span) for tag, span in spans if span[0] <= index <= span[1]]
        if not containing:
            return None
        return min(containing, key=lambda item: item[1][1] - item[1][0])[0]

    def loop_range(self, index: int) -> tuple[int, int, bool]:
        """The span playback moves through, and whether it wraps."""
        last = len(self.frames) - 1
        tag = self.active_tag(index)
        if tag is None:
            return 0, last, True
        start, end = self.tag_span(tag)
        return start, end, tag.loop

    def play_direction(self, index: int) -> str:
        """Which way the tag containing a frame plays, or forward outside one.

        A second lookup rather than a fourth element of ``loop_range``: that
        tuple is passed straight into ``advance`` as its span, and widening it
        would put the answer to "which way" inside a value named for the range.
        Both go through ``active_tag``, so they cannot disagree about which tag
        is in force.
        """
        tag = self.active_tag(index)
        return "forward" if tag is None else tag.direction

    def play_repeat(self, index: int) -> int:
        """How many times the tag containing a frame plays, or 0 outside one.

        A third lookup beside ``play_direction`` rather than a fourth element
        of ``loop_range``, for exactly the reason that one gives: the tuple is
        passed straight into ``advance`` as its *span*, and this is not part of
        a span.
        """
        tag = self.active_tag(index)
        return 0 if tag is None else int(tag.repeat)


def _step(
    index: int, forward: bool, start: int, end: int, loop: bool, direction: str
) -> tuple[int, bool, bool, bool]:
    """One frame onward. Returns ``(index, forward, playing, wrapped)``.

    ``wrapped`` is True at the three points where the span has just been played
    through once -- the end of a forward pass, the start of a reverse one, and
    the *return* to the start of a ping-pong, because one cycle of a swing is
    out **and** back. It is reported whether or not the span loops, so a caller
    counting cycles does not have to know which case it is in.

    Split out of ``advance`` because it is the whole of what a direction means
    and the rest of that function is timekeeping: the two were readable together
    only while there was one direction.

    ``forward`` is state a ping-pong needs and neither other direction reads: an
    index alone cannot say which leg of the swing it is on, and the frame in the
    middle of the span is genuinely visited twice per cycle going opposite ways.
    """
    if direction == "reverse":
        if index <= start:
            return (end, forward, True, True) if loop else (start, forward, False, True)
        return index - 1, forward, True, False
    if direction == "pingpong":
        if start == end:
            # A one-frame swing has nowhere to turn around; it is a still --
            # and each held frame-time is one whole pass through the span.
            return start, forward, loop, True
        if forward:
            # Turning around costs no frame time: the end frame was just held
            # for its own duration, so resuming *at* it would hold it twice and
            # put a visible hitch at each extreme of the swing. Not a wrap: the
            # swing is halfway through, and counting it would make "play three
            # times" mean one and a half.
            return (
                (end - 1, False, True, False)
                if index >= end
                else (index + 1, True, True, False)
            )
        if index <= start:
            return (
                (start + 1, True, True, True) if loop else (start, False, False, True)
            )
        return index - 1, False, True, False
    if index >= end:
        return (start, forward, True, True) if loop else (end, forward, False, True)
    return index + 1, forward, True, False


def advance(
    durations: list[int],
    index: int,
    accum_ms: float,
    dt_ms: float,
    span: tuple[int, int, bool],
    *,
    direction: str = "forward",
    forward: bool = True,
    repeat: int = 0,
    cycles: int = 0,
) -> tuple[int, float, bool, bool, int]:
    """Where the playhead is ``dt_ms`` later. Pure, so it is testable at speed.

    Returns ``(index, accumulator, playing, forward, cycles)``. Time is
    *accumulated* rather than the index being derived from a wall clock,
    because a frame's duration is per-frame: a 40 ms frame followed by a 400 ms
    one is not a rate.

    The loop is a ``while`` and not an ``if`` for the case that actually bites
    -- a 10 ms frame with a dropped-frame ``dt`` of 200 ms behind it, where
    stepping once would run the clip at a twentieth speed whenever the machine
    was busy. ``dt_ms`` is clamped by the caller; a pathological one still
    terminates here because every duration is at least ``MIN_DURATION_MS``.

    ``playing`` comes back False for a non-looping span that has reached its
    end, and for a span with a finite ``repeat`` that has been round it that
    many times -- the two cases where the animation stops itself. ``forward``
    and ``cycles`` are carried in and back out rather than kept here, because
    this function has no state and both a ping-pong's leg and a repeat count
    have to survive between ticks.

    ``repeat`` of 0 means "the ``loop`` flag decides", which is the behaviour
    this function had before the parameter existed. A count above zero decides
    instead -- the span wraps until the count runs out whatever the flag says --
    and when it does run out the step is taken **again with ``loop`` forced
    off**, so the playhead lands where a non-looping span would have left it:
    at the end of a forward pass rather than back at its start, which is the
    frame the user expects to be looking at when a clip stops.
    """
    start, end, loop = span
    if not durations or start > end:
        return index, 0.0, False, forward, cycles
    # **A finite count overrides the flag.** ``loop_range`` hands back
    # ``tag.loop`` verbatim, so a tag already set to "once" and then given a
    # count of three would stop on its *first* wrap -- the flag deciding a
    # question the count is the more specific answer to. The exhaustion path
    # below re-steps with looping forced off, so the clip still stops on the
    # span's last frame; what this restores is the three passes before it.
    #
    # The alternative -- writing ``loop`` True when a count is set -- was
    # rejected: it destroys the flag the user chose, so clearing the count
    # would silently leave a "once" tag looping forever.
    if repeat > 0:
        loop = True
    index = max(start, min(index, end))
    if repeat > 0 and cycles >= repeat:
        # Already played out. Only reachable when the count was *lowered* under
        # a clip that had passed it; stepping on would play one whole extra
        # pass before the check below noticed.
        return index, 0.0, False, forward, cycles
    accum_ms += max(0.0, dt_ms)
    while accum_ms >= durations[index]:
        accum_ms -= durations[index]
        nxt, leg, playing, wrapped = _step(index, forward, start, end, loop, direction)
        if wrapped:
            cycles += 1
            if repeat > 0 and cycles >= repeat:
                nxt, leg, _stopped, _ = _step(
                    index, forward, start, end, False, direction
                )
                return nxt, 0.0, False, leg, cycles
        index, forward = nxt, leg
        if not playing:
            return index, 0.0, False, forward, cycles
    return index, accum_ms, True, forward, cycles
