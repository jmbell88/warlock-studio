"""What each workspace is made of, as data rather than as seven functions.

The insight this wave turns on: **Warlock's skeleton is already an Aseprite
dock tree, hard-coded.** The left column is an ordered stack, the centre is an
anchor plus optional strips, the right column is an ordered stack. Moving a
pane is therefore a *list permutation across three ordered lists*, not a tree
edit -- which is what makes saved arrangements a small feature instead of a
docking system.

General docking is refused, with the cost named. ``layout.fit``,
``layout.centre_width`` and ``rail.expanded_fits`` are all arithmetic over
exactly two sidebars and one centre; under a dock tree
``test_the_expanded_rail_gives_way_before_the_columns_do`` becomes *unstatable*
rather than merely failing. And floating panes need imgui's docking branch,
whose whole persistence story is the ini -- the thing ``main`` disables and
``tests/conftest.py`` polices.

**Share keys are declared, never derived.** A key like ``"inker-colors"`` is in
users' settings files forever; deriving one from ``f"{workspace}/{slot}"``
would silently reset every proportion anybody had dragged, with no failure
anywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: How a slot is sized within its column.
FIXED = "fixed"
"""A known height in design px: a preview, a rail. Never shares."""

SHARE = "share"
"""A proportion of what is left, dragged against the next shareable slot."""

FILL = "fill"
"""Whatever remains. A column has at most one, and it is the last drawn."""


@dataclass(frozen=True)
class Slot:
    """One pane's place in a column.

    ``when`` is what makes a workspace's shape depend on its document without
    a branch in the composition: the tile panel appears with the tilesets and
    the preview with the frames, and both are predicates over the ctx rather
    than ``if`` statements around ``layout.pane`` calls.

    ``movable``/``hideable`` are the two permissions a saved layout has over a
    slot. A slot that is neither is structural -- the canvas is not something
    a user can reorder out of the middle of the window.
    """

    id: str
    label: str
    draw: Callable[[Any], None]
    role: Any = None
    edge: Any = None
    sizing: str = FILL
    height: float = 0.0
    share_key: str = ""
    floor: float = 0.0
    movable: bool = True
    hideable: bool = True
    when: Callable[[Any], bool] | None = None

    def applies(self, ctx: Any) -> bool:
        """Whether this slot is part of the workspace right now."""

        return True if self.when is None else bool(self.when(ctx))


@dataclass
class Column:
    """One ordered stack of slots, and the edge its divider sits on."""

    id: str
    slots: tuple[Slot, ...] = ()
    width: float = 0.0

    def live(self, ctx: Any) -> list[Slot]:
        return [slot for slot in self.slots if slot.applies(ctx)]


def heights(
    slots: list[Slot], avail: float, shares: dict[str, float], scale: float = 1.0
) -> list[float]:
    """How tall each slot is drawn, in physical px. Pure, and the whole rule.

    This is the arithmetic that was written out three different ways at
    ``main.py:466``, ``:3546`` and ``:3649``, and it is why they could disagree
    about what a drag meant. Stated once and as numbers, so every case -- a
    column of one, a column with no fill, a window too short for the floors --
    is a plain assertion rather than a screenshot.

    Two properties it guarantees, and neither was true of all three originals:

    * **A fill slot is never negative.** A negative child height silently kills
      the canvas -- it draws nothing, uploads no textures, and looks like a
      hang. It floors at zero, and the caller drops a zero-height slot.
    * **The heights sum to the room available**, so a column cannot quietly
      overflow the window and clip its last pane.
    """

    fixed = sum(sp(slot.height, scale) for slot in slots if slot.sizing == FIXED)
    room = max(0.0, avail - fixed)
    fills = [slot for slot in slots if slot.sizing == FILL]
    # **A share gives way to what the fill under it needs.** Only the share
    # slot's own floor used to be read here, so a fill slot's floor was a
    # number nothing consulted -- and Inker's Colour pane took its 0.55 of an
    # 833 px column and left the Picker 363 px against a 400 px floor, which is
    # how the Picker came to draw its hex field past its own bottom edge for
    # imgui to clip away. The same rule ``layout.give_way`` states for one
    # stacked pair, applied to the column. The ``max`` below still puts the
    # share slot's own floor first when the two cannot both be met: the
    # alternative is a pane with a heading and nothing under it.
    fill_floors = sum(sp(slot.floor, scale) for slot in fills)
    # **A share's default is an even division of the column, not a flat half.**
    # It was 0.5 whatever the column held, which is exactly right for the one
    # shape it was written against -- one share over one fill -- and wrong for
    # every other: two shares at 0.5 each leave the fill *zero* pixels, and
    # three leave the third share zero as well. Both were real. Plotter's Map
    # file panel and Clay's Document panel vanished the day a second share
    # landed above them, and Sirens has been drawing its Sound effects and Song
    # file panes at zero height for longer than that -- panes that still ran,
    # still passed every test that called their ``draw``, and were simply
    # allocated no room by the column.
    #
    # A saved drag still wins: this is only what a column does before anybody
    # has dragged anything. And it is unchanged for the one-share case, where
    # ``1 / 2`` is the half it always was.
    portions = len([slot for slot in slots if slot.sizing in (SHARE, FILL)])
    default = 1.0 / portions if portions else 0.5
    out: list[float] = []
    taken = 0.0
    for slot in slots:
        if slot.sizing == FIXED:
            out.append(sp(slot.height, scale))
            continue
        if slot.sizing == SHARE:
            want = room * float(shares.get(slot.share_key, default))
            headroom = max(0.0, room - taken - fill_floors)
            want = max(sp(slot.floor, scale), min(want, headroom))
            taken += want
            out.append(want)
            continue
        out.append(0.0)
    if fills:
        left = max(0.0, room - taken)
        share_each = left / len(fills)
        for index, slot in enumerate(slots):
            if slot.sizing == FILL:
                out[index] = share_each
    return out


def sp(value: float, scale: float) -> float:
    """Design px to physical px. Taken as an argument so this file is pure."""

    return float(value) * float(scale)


def reconcile(builtin: list[str], stored: list[str]) -> list[str]:
    """A saved arrangement, brought up to date with the built-in one.

    Two rules, and the second is the one that matters:

    * **A retired id is dropped**, so a pane deleted by a later wave does not
      leave a hole in everybody's saved layout.
    * **A new id is inserted after its last already-placed predecessor**, not
      appended. Appending would put every pane added after a user saved their
      layout at the bottom of a column -- so the tileset editor a designer
      placed second would arrive last for everyone who had ever dragged
      anything.

    Never written back: a launch that changes nothing must not rewrite the
    file.
    """

    known = set(builtin)
    out = [slot for slot in stored if slot in known]
    placed = set(out)
    for index, slot in enumerate(builtin):
        if slot in placed:
            continue
        # The last of this slot's predecessors that is already in the list.
        at = 0
        for before in builtin[:index]:
            if before in placed:
                at = out.index(before) + 1
        out.insert(at, slot)
        placed.add(slot)
    return out


@dataclass
class Skeleton:
    """One workspace's three columns, in drawn order."""

    workspace: str
    left: Column
    centre: Column
    right: Column
    #: Slots drawn along the bottom of the centre column (the timeline strip).
    strips: tuple[Slot, ...] = ()
    _: Any = field(default=None, repr=False)

    def columns(self) -> tuple[Column, Column, Column]:
        return (self.left, self.centre, self.right)

    def slot_ids(self, ctx: Any) -> dict[str, list[str]]:
        """Every live slot, by column. What a saved layout is a permutation of."""

        return {
            column.id: [slot.id for slot in column.live(ctx)]
            for column in self.columns()
        }
