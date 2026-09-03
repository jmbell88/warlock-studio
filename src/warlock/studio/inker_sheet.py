"""Sheet corrections between the tab and the document. No imgui here.

``inker_ops`` registers the verbs and ``panes/inker_sheet.py`` draws the
strip; both come here for the answers -- which frames a scope names, whether
there is a mark to propagate, what a mirror would change -- so the menu row,
the strip button and the probe census cannot disagree about why a verb is
greyed. ``matte_preview``'s arrangement: the logic testable without a window,
the pane a thin reader of it.

**The mark is view state on the tab.** It is a copy of the active track's cel
taken when the playhead lands on a frame of a sheet document, so that "what
did I just change on this cell" has an answer without a marquee: the weight is
the pixels that differ from the copy. It is journal-exempt and never persisted
for ``range_sel``'s reason, and refreshed after a propagation so the same
correction is not sent twice by a second press.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .inker import mirror, sheetscope

__all__ = [
    "NO_SHEET",
    "active_track_uid",
    "can_mirror",
    "can_propagate",
    "counterpart",
    "mirror_reason",
    "mirror_report",
    "mark_weight",
    "propagate",
    "propagate_reason",
    "remark",
    "replace_colour",
    "run_of",
    "shift",
    "sync_mark",
    "targets",
    "mirror_to",
    "mirror_run",
    "can_merge",
    "conflicts",
    "has_base",
    "merge",
    "merge_reason",
    "next_conflict",
    "resolve_keep",
]

NO_SHEET = (
    "Open a Troupe character sheet -- its tags are named animation_direction, "
    "and that is what a sheet correction reads."
)
NO_MARK = "Nothing has changed on this cell since it was marked."
ONE_DIRECTION = "This animation has only one direction on the sheet."
NO_MIRROR = "{direction} has no mirror direction."
NO_SELECTION = "Select the pixels to move first."
NO_REACH = "That scope reaches no other cell."
NO_BASE = (
    "This document was not opened from a rendered sheet, so there is no render "
    "to merge against."
)
NO_CONFLICTS = "No cell is in conflict."


def _doc(tab: Any) -> Any:
    return None if tab is None else getattr(tab, "doc", None)


def is_sheet(tab: Any) -> bool:
    doc = _doc(tab)
    return bool(doc is not None and doc.anim is not None and doc.has_sheet())


def is_sheet_tab(state: Any, tab: Any) -> bool:
    """:func:`is_sheet` in the ``(state, tab)`` shape an ``Op`` predicate takes."""
    return is_sheet(tab)


def no_sheet_reason(state: Any, tab: Any) -> str:
    return "" if is_sheet(tab) else NO_SHEET


def active_track_uid(tab: Any) -> int | None:
    doc = _doc(tab)
    if doc is None or doc.anim is None:
        return None
    index = doc.stack.active_index
    tracks = doc.anim.tracks
    if 0 <= index < len(tracks):
        return int(tracks[index].uid)
    return None


def current_frame(tab: Any) -> int:
    doc = _doc(tab)
    return 0 if doc is None or doc.anim is None else int(doc.anim.current)


def run_of(tab: Any) -> tuple[sheetscope.Run, int] | None:
    doc = _doc(tab)
    if doc is None or doc.anim is None:
        return None
    return sheetscope.locate(doc.sheet_runs(), doc.anim.current)


def counterpart(tab: Any) -> int | None:
    doc = _doc(tab)
    if doc is None or doc.anim is None:
        return None
    return sheetscope.counterpart(doc.sheet_runs(), doc.anim.current)


# --- the mark ---------------------------------------------------------------


def _cel_pixels(tab: Any) -> tuple[int, int, np.ndarray] | None:
    doc = _doc(tab)
    track_uid = active_track_uid(tab)
    if doc is None or doc.anim is None or track_uid is None or not doc.anim.frames:
        return None
    frame = doc.anim.frame
    cel = doc.anim.cels.get((track_uid, frame.uid))
    if cel is None or getattr(cel, "pixels", None) is None:
        return None
    return track_uid, int(frame.uid), cel.pixels


def remark(tab: Any) -> bool:
    """Take the mark from the cel under the playhead. -> whether one was."""
    found = _cel_pixels(tab)
    if found is None:
        tab.sheet_mark = None
        return False
    track_uid, frame_uid, pixels = found
    tab.sheet_mark = (track_uid, frame_uid, pixels.copy())
    return True


def sync_mark(tab: Any) -> None:
    """Keep the mark on the cell under the playhead. Frame thread; cheap.

    One cel copy when the playhead or the active track moves, and an integer
    compare otherwise. Only on a sheet document -- an ordinary animation pays
    nothing for a feature it cannot use.
    """
    if not is_sheet(tab):
        tab.sheet_mark = None
        return
    found = _cel_pixels(tab)
    if found is None:
        tab.sheet_mark = None
        return
    mark = getattr(tab, "sheet_mark", None)
    if mark is None or (mark[0], mark[1]) != (found[0], found[1]):
        remark(tab)


def mark_weight(tab: Any) -> np.ndarray | None:
    """The pixels changed since the mark, narrowed by the selection if there
    is one. None when there is no mark or nothing changed."""
    mark = getattr(tab, "sheet_mark", None)
    found = _cel_pixels(tab)
    if mark is None or found is None or (mark[0], mark[1]) != (found[0], found[1]):
        return None
    if mark[2].shape != found[2].shape:
        return None
    weight = mirror.changed_weight(mark[2], found[2])
    if weight is None:
        return None
    doc = _doc(tab)
    if doc.mask is not None and doc.mask.mask.shape == weight.shape:
        weight = np.minimum(weight, doc.mask.mask)
        if not weight.any():
            return None
    return weight


# --- scopes -----------------------------------------------------------------


def targets(state: Any, tab: Any, scope: str | None = None) -> list[int]:
    doc = _doc(tab)
    if doc is None or doc.anim is None:
        return []
    scope = scope or getattr(state, "sheet_scope", "directions")
    explicit: list[int] = []
    if scope == "explicit":
        rect = getattr(tab, "range_sel", None)
        if rect is not None:
            _t0, _t1, f0, f1 = rect
            explicit = list(range(min(f0, f1), max(f0, f1) + 1))
    return sheetscope.frames_for(
        doc.sheet_runs(),
        doc.anim.current,
        scope,
        explicit,
        frame_count=len(doc.anim.frames),
    )


def _ready(state: Any, tab: Any) -> bool:
    return tab is not None and not getattr(tab, "busy", False) and not getattr(
        state, "transforming", False
    )


def can_propagate(state: Any, tab: Any) -> bool:
    return (
        _ready(state, tab)
        and is_sheet(tab)
        and mark_weight(tab) is not None
        and bool(targets(state, tab))
    )


def propagate_reason(state: Any, tab: Any) -> str:
    if not is_sheet(tab):
        return NO_SHEET
    if mark_weight(tab) is None:
        return NO_MARK
    if not targets(state, tab):
        if state.sheet_scope == "directions":
            return ONE_DIRECTION
        return NO_REACH
    return ""


def can_scope(state: Any, tab: Any) -> bool:
    return _ready(state, tab) and is_sheet(tab) and bool(targets(state, tab))


def scope_reason(state: Any, tab: Any) -> str:
    if not is_sheet(tab):
        return NO_SHEET
    if not targets(state, tab):
        return NO_REACH
    return ""


def can_shift(state: Any, tab: Any) -> bool:
    doc = _doc(tab)
    return can_scope(state, tab) and doc.mask is not None


def shift_reason(state: Any, tab: Any) -> str:
    said = scope_reason(state, tab)
    if said:
        return said
    return NO_SELECTION if _doc(tab).mask is None else ""


def can_mirror(state: Any, tab: Any) -> bool:
    return _ready(state, tab) and is_sheet(tab) and counterpart(tab) is not None


def mirror_reason(state: Any, tab: Any) -> str:
    if not is_sheet(tab):
        return NO_SHEET
    here = run_of(tab)
    if here is None:
        return "The playhead is not on a cell of the sheet."
    if counterpart(tab) is None:
        return NO_MIRROR.format(direction=here[0].direction)
    return ""


# --- the mirror preview -----------------------------------------------------


def mirror_report(tab: Any) -> tuple[int, int, np.ndarray, tuple[int, int, int, int] | None] | None:
    """``(outside, inside, map, face_box)`` for the cell under the playhead
    against its counterpart, cached on the tab by document revision."""
    doc = _doc(tab)
    track_uid = active_track_uid(tab)
    target_frame = counterpart(tab)
    if doc is None or track_uid is None or target_frame is None:
        return None
    fraction = float(getattr(tab, "face_fraction", mirror.FACE_FRACTION))
    key = (doc.rev, track_uid, int(doc.anim.frame.uid), target_frame, fraction)
    cached = getattr(tab, "mirror_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    source = doc.anim.cels.get((track_uid, doc.anim.frame.uid))
    target = doc.anim.cels.get((track_uid, doc.anim.frames[target_frame].uid))
    if source is None or target is None:
        return None
    flipped = mirror.mirrored(source.pixels)
    box = mirror.face_box(flipped, fraction)
    weight = mirror.face_weight(flipped.shape[:2], box)
    outside, inside, changed = mirror.diff_report(source.pixels, target.pixels, weight)
    report = (outside, inside, changed, box)
    tab.mirror_cache = (key, report)
    return report


# --- the verbs --------------------------------------------------------------


def _framed(ctx: Any, verb: str, work: Any) -> bool:
    """Run one engine call and frame its refusal as a sentence, the
    ``_run_range_verb`` rule: the engine names what is wrong with the
    document and the user needs what is wrong with what they pressed."""
    try:
        return bool(work())
    except ValueError as exc:
        ctx.toast(f"{verb} was not applied: {exc}.", "warn")
        return False


# -- merging a re-render ------------------------------------------------------
#
# The other half of the cleanup loop. ``_doc_sheet.merge_render`` decides what a
# fresh render may do to each cell and :mod:`~warlock.studio.inker.sheetmerge`
# holds the rule; this is the tab-shaped wrapper the ops call.


def has_base(tab: Any) -> bool:
    """Whether this document remembers what the renderer last gave it."""
    return getattr(_doc(tab), "sheet_base", None) is not None


def can_merge(state: Any, tab: Any) -> bool:
    return is_sheet_tab(state, tab) and has_base(tab)


def merge_reason(state: Any, tab: Any) -> str:
    """Why the merge is greyed, or "" when it is not. Never silently disabled."""
    if not is_sheet_tab(state, tab):
        return no_sheet_reason(state, tab)
    return "" if has_base(tab) else NO_BASE


def merge(ctx: Any, tab: Any, incoming: Sequence[np.ndarray]) -> bool:
    """Land ``incoming`` on this document. -> whether anything was pushed.

    The toast names all three outcomes rather than a total, because "took 48"
    and "flagged 2 conflicts" are the two things the reader has to act on and a
    single number hides the second inside the first.
    """
    from .inker import sheetmerge

    doc = tab.doc
    track_uid = active_track_uid(tab)
    if track_uid is None:
        return False
    counts: list[Any] = []

    def run() -> bool:
        counts.append(doc.merge_render(track_uid, incoming))
        return True

    if not _framed(ctx, "The merge", run):
        return False
    result = counts[0]
    ctx.toast(
        sheetmerge.counts_sentence(result),
        "warn" if result.conflicts else "success",
    )
    return True


def conflicts(tab: Any) -> list[int]:
    """The flagged cells, as frame indices. Resolved from uids at the door."""
    doc = _doc(tab)
    base = getattr(doc, "sheet_base", None)
    if base is None or doc.anim is None:
        return []
    at = {frame.uid: i for i, frame in enumerate(doc.anim.frames)}
    return sorted(at[uid] for uid in base.conflicts if uid in at)


def next_conflict(tab: Any, after: int) -> int | None:
    """The first flagged cell past ``after``, wrapping. None if there are none.

    Wraps because the point is to walk every conflict once and stop, and a
    reader who starts halfway down the timeline should not have to scroll back
    up to reach the ones above them.
    """
    flagged = conflicts(tab)
    if not flagged:
        return None
    return next((i for i in flagged if i > after), flagged[0])


def resolve_keep(ctx: Any, tab: Any, frames: Sequence[int]) -> bool:
    """Accept the hand edit on these cells and clear their flags.

    Nothing is written: the edit is already what is on the canvas, and the base
    already holds the render, so resolving is only the flag coming off. Pushed
    as its own step so it can be undone like everything else.
    """
    from .inker.undo import SheetBaseEdit

    doc = _doc(tab)
    base = getattr(doc, "sheet_base", None)
    if base is None or doc.anim is None:
        return False
    at = [frame.uid for frame in doc.anim.frames]
    wanted = {at[i] for i in frames if 0 <= i < len(at)} & base.conflicts
    if not wanted:
        return False
    after = base.copy()
    after.conflicts -= wanted
    doc.history.push(SheetBaseEdit(before=base.copy(), after=after))
    doc.sheet_base = after
    ctx.toast(f"Kept the hand edit on {len(wanted)} cell(s).", "success")
    return True


def propagate(ctx: Any, tab: Any) -> bool:
    state = ctx.state.inker
    doc = tab.doc
    weight = mark_weight(tab)
    track_uid = active_track_uid(tab)
    if weight is None or track_uid is None:
        return False
    frames = targets(state, tab)
    done = _framed(
        ctx,
        "The patch",
        lambda: doc.propagate_patch(track_uid, doc.anim.current, frames, weight),
    )
    if done:
        remark(tab)
        ctx.toast(f"Sent the correction to {len(frames)} cell(s).", "success")
    return done


def replace_colour(ctx: Any, tab: Any) -> bool:
    state = ctx.state.inker
    doc = tab.doc
    track_uid = active_track_uid(tab)
    if track_uid is None:
        return False
    frames = [doc.anim.current, *targets(state, tab)]
    done = _framed(
        ctx,
        "The recolour",
        lambda: doc.replace_colour_frames(
            track_uid,
            frames,
            tuple(int(v) for v in state.sheet_old),
            tuple(int(v) for v in state.sheet_new),
            float(state.sheet_tolerance),
        ),
    )
    if done:
        remark(tab)
    return done


def shift(ctx: Any, tab: Any) -> bool:
    state = ctx.state.inker
    doc = tab.doc
    track_uid = active_track_uid(tab)
    if track_uid is None:
        return False
    frames = [doc.anim.current, *targets(state, tab)]
    done = _framed(
        ctx,
        "The shift",
        lambda: doc.shift_frames(
            track_uid, frames, int(state.sheet_dx), int(state.sheet_dy)
        ),
    )
    if done:
        remark(tab)
    return done


def mirror_to(ctx: Any, tab: Any) -> bool:
    doc = tab.doc
    track_uid = active_track_uid(tab)
    target = counterpart(tab)
    if track_uid is None or target is None:
        return False
    fraction = float(getattr(tab, "face_fraction", mirror.FACE_FRACTION))
    done = _framed(
        ctx,
        "The mirror",
        lambda: doc.mirror_to(track_uid, doc.anim.current, target, fraction),
    )
    if not done:
        ctx.toast("The mirror already matches outside the face.", "info")
    return done


def mirror_run(ctx: Any, tab: Any) -> bool:
    doc = tab.doc
    track_uid = active_track_uid(tab)
    here = run_of(tab)
    if track_uid is None or here is None:
        return False
    fraction = float(getattr(tab, "face_fraction", mirror.FACE_FRACTION))
    done = _framed(
        ctx, "The mirror", lambda: doc.mirror_run(track_uid, here[0], fraction)
    )
    if not done:
        ctx.toast("Every cell of that run already matches its mirror outside the face.", "info")
    return done
