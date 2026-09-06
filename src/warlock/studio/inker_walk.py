"""The walk-cycle session: the shell side of ``inker/walk/``.

Flourish's three-file split, one file shorter. There is no off-thread half here
and there does not need to be: a bake of eight frames is a few RotSprite turns of
limb-sized planes, which is milliseconds, so the preview re-renders inline on a
revision comparison rather than through ``TaskRunner``. If a drawing ever arrives
big enough for that to show, the ceiling refuses it first
(``walk.WALK_MAX_PIXELS``) -- and a refusal is a better answer than a spinner.

**Nothing in this module writes to the source document.** Parts are lifted with
``selection_cutout`` (which pushes no edit) or copied off a layer's pixels, the
joints live on the session, and Bake builds a *new* document. So cancelling is
dropping an object, and the test that says the source's ``rev`` and history head
are untouched across an open-and-cancel is checking a property the design has
rather than one the code remembers to maintain.

The session is one field on ``InkerState`` whose non-``None``-ness *is* "the tool
is open", and it carries ``tab_uid`` for ``transform_uid``'s reason: this state
object is shared by every tab and the panes draw whichever tab is in front, so a
bare flag would let a tab switch point the overlay at somebody else's drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .inker import walk
from .inker.walk import gait
from .inker.walk import rig as R

#: How close, in *screen* pixels, a click has to be to grab a joint. Screen
#: rather than document pixels so the grab is the same size for the hand at
#: every zoom -- ``plotter_canvas._handle_at``'s rule.
GRAB_PX = 9.0

#: Radius the joint dots are drawn at, in design pixels.
JOINT_R = 4.0

NOTHING_OPEN = "Nothing is open."
BUSY = "The document is busy -- a save, an export or playback is still running."
ALREADY_OPEN = "A walk cycle is already being set up."
NO_SELECTION = "Select the pixels for this part first."
NOT_OPEN = "No walk cycle is being set up."

#: What a fresh session copies a far limb at. Dark enough to read as behind the
#: body, light enough that it is clearly the same limb.
DEFAULT_FAR_BRIGHTNESS = 0.72


@dataclass
class WalkSession:
    """Everything the setup tool holds, and none of it on any document."""

    tab_uid: str
    size: tuple[int, int]
    rig: R.Rig
    settings: gait.WalkSettings
    #: Which joint the parts list has highlighted, and what a click on empty
    #: canvas would place. Also what the overlay rings.
    joint: str = "near_hip"
    #: The joint (or ``"ground"``) a drag is currently moving, if any.
    grab: str = ""
    far_brightness: float = DEFAULT_FAR_BRIGHTNESS
    #: part -> the combo key it was assigned from: a layer uid as a string, or
    #: ``"selection"``. Kept here rather than read back off ``Part.source``,
    #: which is a *label* -- two layers may share a name (a duplicate is called
    #: "Layer 2" twice as often as not), and matching on it would tick the wrong
    #: row in the parts list while the right pixels sat in the rig.
    assigned_from: dict[str, str] = field(default_factory=dict)
    #: Preview playhead. Wall-clock rather than an accumulator on the tab: the
    #: session is temporary and owns no document, so there is nothing for a
    #: stalled frame to desynchronise from.
    playing: bool = True
    play_at: float = 0.0
    play_index: int = 0
    #: ``settings`` the user has touched, so the defaults stop being re-derived
    #: from under them once they have an opinion. A moved joint changes the leg
    #: length, and a stride that silently jumped back to 45% of it every time
    #: would be unusable.
    pinned: set[str] = field(default_factory=set)
    #: The last render, and the ``(rig.rev, settings)`` it was made from.
    _frames: list[np.ndarray] = field(default_factory=list)
    _stamp: tuple[Any, ...] = ()


# -- opening and closing ---------------------------------------------------------------


def session(state: Any, tab: Any) -> WalkSession | None:
    """The session, but only if it belongs to the tab being drawn."""
    open_session = getattr(state, "walk", None)
    if open_session is None or tab is None:
        return None
    return open_session if open_session.tab_uid == getattr(tab, "uid", "") else None


def is_open(state: Any, tab: Any) -> bool:
    return session(state, tab) is not None


def can_open(state: Any, tab: Any) -> bool:
    return bool(open_reason(state, tab) == "")


def open_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return NOTHING_OPEN
    if getattr(tab, "busy", False):
        return BUSY
    if getattr(state, "walk", None) is not None:
        return ALREADY_OPEN
    return walk.too_large(tab.doc.size)


def open_session(ctx: Any, tab: Any) -> bool:
    """Start setting a walk up on ``tab``. Writes nothing to the document."""
    state = ctx.state.inker
    if not can_open(state, tab):
        return False
    size = tab.doc.size
    rig = walk.blank(size)
    state.walk = WalkSession(
        tab_uid=tab.uid,
        size=size,
        rig=rig,
        settings=gait.WalkSettings(),
        play_at=clock(),
    )
    return True


def cancel(ctx: Any, tab: Any) -> bool:
    """Throw the session away. There is nothing to undo, by construction."""
    state = ctx.state.inker
    if session(state, tab) is None:
        return False
    _release_textures(ctx, tab)
    state.walk = None
    return True


def _release_textures(ctx: Any, tab: Any) -> None:
    from . import docmodes

    if ctx.viewer is None:
        return
    docmodes.release_prefix(ctx, f"inker_tex:{tab.uid}:walk")


def clock() -> float:
    """Wall-clock seconds. A seam, so a test can drive playback without sleeping."""
    import time

    return time.monotonic()


# -- assembling the rig ----------------------------------------------------------------


def assign_layer(ctx: Any, tab: Any, part: str, layer_uid: int) -> bool:
    """Point a part at one of the drawing's layers.

    The layer's pixels are *copied* into the rig, trimmed to what they cover.
    A reference would make a later stroke on the source silently restyle the
    walk halfway through setting it up, which is a surprise with no undo.
    """
    open_session = session(ctx.state.inker, tab)
    if open_session is None or part not in R.PART_NAMES:
        return False
    layer = tab.doc.stack.by_uid(int(layer_uid))
    if layer is None:
        return False
    _set_part(
        open_session,
        part,
        walk.part_from_plane(layer.pixels, source=layer.name),
    )
    open_session.assigned_from[part] = str(int(layer_uid))
    return True


def assign_selection(ctx: Any, tab: Any, part: str) -> bool:
    """Lift a part out of the current selection, leaving the drawing alone.

    ``selection_cutout`` is the door precisely because it pushes no edit:
    ``layer_from_selection`` would add a layer to the drawing the user asked us
    to keep intact.
    """
    open_session = session(ctx.state.inker, tab)
    if open_session is None or part not in R.PART_NAMES:
        return False
    cutout = tab.doc.selection_cutout()
    if cutout is None or not cutout[:, :, 3].any():
        return False
    bounds = tab.doc.mask.bounds if tab.doc.mask is not None else None
    origin = (int(bounds[0]), int(bounds[1])) if bounds else (0, 0)
    _set_part(
        open_session, part, walk.part_from_plane(cutout, origin=origin, source="selection")
    )
    open_session.assigned_from[part] = "selection"
    return True


def clear_part(ctx: Any, tab: Any, part: str) -> bool:
    open_session = session(ctx.state.inker, tab)
    if open_session is None or part not in R.PART_NAMES:
        return False
    _set_part(open_session, part, R.Part())
    open_session.assigned_from.pop(part, None)
    return True


def selection_reason(state: Any, tab: Any) -> str:
    if tab is None:
        return NOTHING_OPEN
    return "" if getattr(tab.doc, "mask", None) is not None else NO_SELECTION


def copy_near_to_far(ctx: Any, tab: Any, limb: str) -> bool:
    open_session = session(ctx.state.inker, tab)
    if open_session is None:
        return False
    open_session.rig = walk.copy_near_to_far(
        open_session.rig, limb, brightness=open_session.far_brightness
    )
    for spec in R.PARTS:
        if spec.limb != f"far_{limb}":
            continue
        near = "near_" + spec.name.removeprefix("far_")
        source = open_session.assigned_from.get(near)
        if source is None:
            open_session.assigned_from.pop(spec.name, None)
        else:
            open_session.assigned_from[spec.name] = source
    _settle(open_session)
    return True


def set_far_brightness(ctx: Any, tab: Any, value: float) -> bool:
    open_session = session(ctx.state.inker, tab)
    if open_session is None:
        return False
    open_session.far_brightness = max(0.1, min(1.0, float(value)))
    return True


def set_joint(ctx: Any, tab: Any, name: str, point: tuple[float, float]) -> bool:
    open_session = session(ctx.state.inker, tab)
    if open_session is None or name not in R.JOINTS:
        return False
    open_session.rig = walk.set_joint(open_session.rig, name, point)
    _settle(open_session)
    return True


def set_ground(ctx: Any, tab: Any, y: float) -> bool:
    open_session = session(ctx.state.inker, tab)
    if open_session is None:
        return False
    open_session.rig = walk.set_ground(open_session.rig, y)
    _settle(open_session)
    return True


def _set_part(open_session: WalkSession, part: str, value: R.Part) -> None:
    open_session.rig = walk.set_part(open_session.rig, part, value)
    _settle(open_session)


def _settle(open_session: WalkSession) -> None:
    """Re-derive whatever the user has not claimed, and re-clamp what they have.

    The stride's bound is geometry, so dragging a hip moves it -- and a stride
    left sitting above the new bound would be silently shortened by the clamp
    every frame instead of showing the user a number they can act on.
    """
    rig = open_session.rig
    if R.leg_length(rig) <= 0.0 or "near_hip" not in rig.joints:
        return
    defaults = gait.defaults_for(rig)
    settings = open_session.settings
    for name in ("stride", "lift", "bob", "arm_swing"):
        if name not in open_session.pinned:
            settings = settings.replaced(**{name: getattr(defaults, name)})
    ceiling = gait.reachable_stride(rig)
    if settings.stride > ceiling:
        settings = settings.replaced(stride=ceiling)
    open_session.settings = settings


def set_setting(ctx: Any, tab: Any, name: str, value: float) -> bool:
    """One slider moved. Pins it, so ``_settle`` stops overwriting it."""
    open_session = session(ctx.state.inker, tab)
    if open_session is None:
        return False
    if name == "duration_ms":
        open_session.settings = open_session.settings.replaced(
            duration_ms=max(1, int(value))
        )
        return True
    if name not in ("stride", "lift", "bob", "arm_swing"):
        return False
    open_session.pinned.add(name)
    open_session.settings = open_session.settings.replaced(**{name: float(value)})
    return True


# -- dragging a joint on the canvas ----------------------------------------------------


def handles(open_session: WalkSession) -> dict[str, tuple[float, float]]:
    """Every draggable point, in document pixels.

    Only the joints some assigned part actually needs, so a rig with no arms
    does not carpet the drawing with dots nothing will read.
    """
    needed = set(R.required_joints(open_session.rig))
    return {
        name: point for name, point in open_session.rig.joints.items() if name in needed
    }


def nearest(
    open_session: WalkSession, point: tuple[float, float], radius: float
) -> str:
    """The joint within ``radius`` document pixels of ``point``, or ``""``.

    The ground line is a handle too, matched on its ``y`` alone -- it spans the
    canvas, so an ``x`` test would mean hunting for the one place it can be
    grabbed.
    """
    best, best_distance = "", radius
    for name, (x, y) in handles(open_session).items():
        distance = float(np.hypot(point[0] - x, point[1] - y))
        if distance <= best_distance:
            best, best_distance = name, distance
    if best:
        return best
    if abs(point[1] - open_session.rig.ground_y) <= radius:
        return "ground"
    return ""


def press(ctx: Any, tab: Any, point: tuple[float, float], radius: float) -> bool:
    """Begin a drag, or place the selected joint where the user clicked.

    A click on empty canvas *places* rather than doing nothing, which is what
    makes the setup a walk down the list rather than a hunt: the panel says
    which joint is next and the canvas is where it goes.
    """
    open_session = session(ctx.state.inker, tab)
    if open_session is None:
        return False
    grabbed = nearest(open_session, point, radius)
    if grabbed:
        open_session.grab = grabbed
        return True
    if open_session.joint in R.JOINTS:
        set_joint(ctx, tab, open_session.joint, point)
        open_session.grab = open_session.joint
        return True
    return False


def drag(ctx: Any, tab: Any, point: tuple[float, float]) -> bool:
    open_session = session(ctx.state.inker, tab)
    if open_session is None or not open_session.grab:
        return False
    if open_session.grab == "ground":
        return set_ground(ctx, tab, point[1])
    return set_joint(ctx, tab, open_session.grab, point)


def release(ctx: Any, tab: Any) -> bool:
    """End the drag, and walk the panel on to whatever is still unplaced.

    The advance is what turns fifteen points into a sequence: the row above the
    canvas says which joint is next, a click puts it there, and the next one is
    named without the reader going back to the panel to choose it.

    It advances only once the selected joint has actually *been* placed, so
    dragging an existing hip to adjust it leaves the selection alone -- the
    common gesture after the first pass is a correction, and a selection that
    jumped away after every correction would be worse than none.
    """
    open_session = session(ctx.state.inker, tab)
    if open_session is None:
        return False
    open_session.grab = ""
    following = next_unplaced(open_session)
    if following and open_session.joint in open_session.rig.joints:
        open_session.joint = following
    return True


def select_joint(ctx: Any, tab: Any, name: str) -> bool:
    open_session = session(ctx.state.inker, tab)
    if open_session is None or (name and name not in R.JOINTS):
        return False
    open_session.joint = name
    return True


def next_unplaced(open_session: WalkSession) -> str:
    """The joint the panel should be pointing at next, or ``""`` when done."""
    missing = walk.missing_joints(open_session.rig)
    return missing[0] if missing else ""


# -- the preview -----------------------------------------------------------------------


def ready(open_session: WalkSession) -> bool:
    """Whether there is enough of a rig to render anything at all."""
    return walk.refusal(open_session.rig) == ""


def frames(open_session: WalkSession) -> list[np.ndarray]:
    """The eight composites, re-rendered only when the rig or the settings move.

    Keyed on ``rig.rev`` rather than on the rig's contents, which is why every
    mutator in ``walk.rig`` bumps it.
    """
    if not ready(open_session):
        return []
    stamp = (open_session.rig.rev, open_session.settings)
    if stamp != open_session._stamp or not open_session._frames:
        open_session._frames = walk.composite_frames(
            open_session.rig, open_session.settings, open_session.size
        )
        open_session._stamp = stamp
    return open_session._frames


def clipping(open_session: WalkSession) -> tuple[int, int, int, int]:
    if not ready(open_session):
        return (0, 0, 0, 0)
    rendered = walk.frames(open_session.rig, open_session.settings)
    return walk.clipping(rendered, open_session.size)


def tick(open_session: WalkSession, *, now: float | None = None) -> int:
    """Advance the preview playhead off the wall clock.

    Wall-clock rather than a per-frame accumulator, ``plotter_tileset_editor``'s
    idiom and its reason: a stalled frame catches up in one step instead of
    drifting, and there is no cadence to keep in step with a document that does
    not exist yet.

    **Whole frames since the last advance, not a division of the time since Play
    was pressed.** The second spelling is the obvious one and it is subtly wrong
    on resume: it has to synthesise a start time by subtracting ``index * step``
    from now, and ``100.0 - 3 * 0.1`` divided back by ``0.1`` is 2.999..., so
    resuming on frame three showed frame two. Advancing from the frame that is
    already up has nothing to round.
    """
    total = walk.WALK_FRAMES
    if not open_session.playing:
        return int(open_session.play_index) % total
    now = clock() if now is None else now
    step = max(1, int(open_session.settings.duration_ms)) / 1000.0
    advanced = int((now - open_session.play_at) / step)
    if advanced > 0:
        open_session.play_index = (open_session.play_index + advanced) % total
        open_session.play_at += advanced * step
    return open_session.play_index


def toggle_play(open_session: WalkSession, *, now: float | None = None) -> None:
    """Play or pause. Resuming starts the frame already on screen, now."""
    now = clock() if now is None else now
    open_session.playing = not open_session.playing
    if open_session.playing:
        open_session.play_at = now


def step_frame(open_session: WalkSession, delta: int, *, now: float | None = None) -> int:
    """One frame either way, and stop -- stepping is a thing done while paused."""
    del now
    open_session.playing = False
    open_session.play_index = (open_session.play_index + int(delta)) % walk.WALK_FRAMES
    return open_session.play_index


# -- baking ----------------------------------------------------------------------------


def can_bake(state: Any, tab: Any) -> bool:
    return bake_reason(state, tab) == ""


def cancel_reason(state: Any, tab: Any) -> str:
    return "" if is_open(state, tab) else NOT_OPEN


def bake_reason(state: Any, tab: Any) -> str:
    open_session = session(state, tab)
    if open_session is None:
        return NOT_OPEN
    if getattr(tab, "busy", False):
        return BUSY
    return walk.refusal(open_session.rig) or walk.too_large(open_session.size)


def bake(ctx: Any, tab: Any) -> Any:
    """Land the cycle as a new document and open it in a new tab.

    The source is left exactly as it was -- it was never written to -- so the
    user comes back to a still drawing beside a walking one.
    """
    from . import inker_mode

    state = ctx.state.inker
    open_session = session(state, tab)
    if open_session is None or not can_bake(state, tab):
        return None
    doc = walk.document(
        open_session.rig,
        open_session.settings,
        open_session.size,
        matte=getattr(tab.doc, "matte", None),
    )
    title = f"{getattr(tab, 'title', 'Untitled')} walk"
    cancel(ctx, tab)
    # ``inker_open``'s call, verbatim: this is the one door a generated
    # document comes through, and a public alias for it would be a second name.
    return inker_mode._adopt(ctx, state, doc, path=None, title=title, file_format="ora")
