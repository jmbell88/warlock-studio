"""The asset library: filters, cards, bulk actions, storage.

A card shows what the job *is* and offers exactly one primary action -- the
obvious next step, from :func:`~warlock.studio.state.primary_action` -- with
everything else behind the overflow menu. That ladder is the browser's, ported
verbatim, because "the button is always the thing I wanted" is most of what
made the old library usable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import followups
from ...service import export as svc_export
from ...service import jobs as svc_jobs
from ...service import rig as svc_rig
from .. import controls, dialogs, icons, jobs_cache, motion, theme, tokens, toolbar, widgets
from ..manual import render as manual_render
from ..state import ACTIONS, QUERY_FIELDS, SORTS, primary_action
from ..tokens import sp
from . import thumbs

log = logging.getLogger(__name__)

# Both heights are a *stack of measured rows* held as one constant, which is
# why they move when the type ramp does: 92/46 were fitted against a 13 px body
# and the 14 px one (the UI redesign, wave 1) adds roughly a pixel to every line and
# to the padding around every control in them. Raised by 4 rather than by the
# arithmetic exactly, because the slack is what stops a card clipping at the
# scales the smoke pass runs at -- a card that fits at 1.0 and loses its last
# row at 1.75 is the failure these constants exist to prevent.
CARD_HEIGHT = 96.0
THUMB_SIZE = 72.0
# The compact row (J89): the same card with the second line of badges and the
# action row dropped, so twice as many assets fit on a screen. Two constants
# rather than a scale factor -- a card is a fixed stack of a title, a pill and
# a button row, and multiplying its height would leave the contents overflowing
# or floating rather than fitting.
COMPACT_HEIGHT = 50.0
COMPACT_THUMB = 32.0

# The drag-and-drop type a library card offers (I83). imgui's Python binding
# carries an *integer* payload -- ``id(object)``, in its own documentation --
# and a job id is a string whose ``id()`` is the address of a temporary. So the
# payload's data_id is a constant marker and the job it names travels in
# ``AppState.dragging_job``: one drag is in flight at a time by construction,
# and the state field is what the drop target and the highlight both read.
DRAG_JOB = "warlock-job"
_DRAG_MARKER = 1


def draggable_source(ctx: Any, job: Any) -> None:
    """Offer the last-drawn item as a draggable asset.

    Called on the card, and only ever for a job something can be *dropped on*:
    a drag that no target accepts is a gesture the user has to discover is
    pointless, so a card that is not a finished reference simply does not lift.
    """
    if not can_drag(job):
        return
    if imgui.begin_drag_drop_source():
        ctx.state.dragging_job = job["id"]
        imgui.set_drag_drop_payload_py_id(DRAG_JOB, _DRAG_MARKER)
        imgui.text(job.get("name") or job.get("prompt") or job["id"])
        imgui.end_drag_drop_source()


def can_drag(job: Any) -> bool:
    """A finished 2D reference, which is what the 3D source slot takes.

    The same predicate the slot accepts by, stated once: a card that lifts and
    a slot that refuses it is worse than a card that does not lift.
    """
    return job.get("stage") == "reference" and job.get("status") == "done"


def dragged_job(ctx: Any) -> Any:
    """The row being dragged, or ``None``. Read by a drop target that wants to
    highlight itself before the mouse is released."""
    job_id = getattr(ctx.state, "dragging_job", None)
    return ctx.cache.get(job_id) if job_id else None


def draw(ctx: Any) -> None:
    # Resolved before the filter row rather than after it, because the row's
    # select-all acts on exactly this list and computing it twice a frame to
    # keep the old order would be paying for the same filter pass twice.
    jobs = ctx.cache.visible(ctx.state.filters)
    # Draws the (?) too, on the sort row that reserves the width for it --
    # ``render.help_button`` right-aligns with an unconditional ``same_line``,
    # so where it lands is a property of the row above it and belongs with the
    # code that lays that row out. Chapter 08's body has no other way in from
    # here.
    _filters(ctx, jobs)
    imgui.separator()
    _read_error(ctx)
    height = -_footer_reserve()
    # Hoisted out of the card loop (B20): each queued card used to rebuild the
    # whole reversed-list scan to learn its own position.
    queue_pos = {
        job_id: i + 1
        for i, job_id in enumerate(
            j["id"] for j in reversed(ctx.cache.jobs) if j.get("status") == "queued"
        )
    }
    if imgui.begin_child("library-list", (0, height)):
        if not jobs:
            _empty(ctx)
        group = ""
        clip = _clipper(ctx, len(jobs))
        for job in jobs:
            # Date grouping (J89), and only under the date sort: a "Today"
            # heading above a list ordered by size would be a lie about what
            # separates the rows below it.
            if ctx.state.filters.sort == "newest":
                heading = date_group(job.get("created_at"))
                if heading != group:
                    group = heading
                    widgets.section(heading)
            if clip is not None and clip(job["id"]):
                continue
            _card(ctx, job, queue_pos)
        _load_more(ctx)
    imgui.end_child()
    below = imgui.get_cursor_pos_y()
    _bulk(ctx, jobs)
    _storage(ctx, jobs)
    _measure_footer(below)


# What the footer below the list actually took, last frame (K98). The
# reservation used to be two constants -- 96 with a bulk bar, 34 without --
# guessed against a bar whose real height is a function of the theme's frame
# padding, the item spacing, the UI scale and how many buttons the current
# *view* draws. The trash's bulk bar has two buttons where the workshop's has
# three, so a constant could not have been right for both.
#
# One frame late by construction, and that is fine: the footer's height changes
# when a tick is added or the view is switched, both of which redraw for many
# frames afterwards. The seed is the old no-bulk-bar constant, so the very
# first frame is exactly what it always was -- carried up by 2 with the type
# ramp (the UI redesign, wave 1), since the row it seeds is one line of text plus
# ``frame_padding`` and both grew.
_footer_px = [36.0]


def _footer_reserve() -> float:
    return sp(_footer_px[0])


def _measure_footer(top: float) -> None:
    height = imgui.get_cursor_pos_y() - top
    if height > 0:
        # Back into design pixels, because ``sp`` is applied on the way out and
        # the UI scale can change between frames (K99).
        _footer_px[0] = height / max(tokens.SCALE, 0.01)


def _empty(ctx: Any) -> None:
    """Why the list is empty, in the three senses it can be."""
    if ctx.state.filters.trash:
        widgets.empty_state(
            icons.TRASH,
            "The trash is empty.",
            "Deleted assets wait here until you empty it.",
        )
        return
    fresh = not ctx.cache.jobs
    widgets.empty_state(
        icons.IMAGE,
        "Nothing here yet." if fresh else "Nothing matches.",
        "Generated references and meshes appear here."
        if fresh
        else "Loosen the filters above.",
    )


# The day boundaries a date grouping uses, as a pure function of two timestamps
# so the wording is testable without a clock (J89).
def date_group(created_at: Any, now: float | None = None) -> str:
    """``Today`` / ``Yesterday`` / ``This week`` / ``2026-07`` for a row.

    Calendar days, not 24-hour windows: something made at 23:50 last night is
    "yesterday" at 00:10, and an elapsed-hours rule would call it "today".
    Older than a week collapses to the month, because a workshop's older half
    is browsed by roughly-when rather than by exactly-when.

    Sentence case because these are ``widgets.section`` headings and nothing
    else (the UI redesign, wave 6) -- the month rung already read as a proper label
    while the four words above it were lowercase, so one column of headings was
    drawn in two registers.
    """
    import datetime as dt

    if not created_at:
        return "Undated"
    now = time.time() if now is None else now
    then_day = dt.date.fromtimestamp(float(created_at))
    today = dt.date.fromtimestamp(now)
    days = (today - then_day).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return "This week"
    return then_day.strftime("%Y-%m")


def _read_error(ctx: Any) -> None:
    """What went wrong reading the list, in words, with a way to try again (E46).

    The text mattered more than it looks: a locked database and a vanished job
    directory are the two realistic causes and they want opposite responses
    (wait, versus look at the disk), which one fixed sentence cannot tell apart.
    Retry is ``invalidate`` rather than a re-read on this thread -- the list is
    read on the refresh tick and a pane must not do sqlite on the frame thread.
    """
    if not ctx.cache.error:
        return
    widgets.text_colored(theme.ERR, "Could not read the job list.")
    widgets.muted(ctx.cache.error)
    if controls.small_button("Try again##library-retry"):
        ctx.cache.invalidate()


def filtering(filters: Any) -> bool:
    """Whether anything is narrowing the list right now (J88).

    Read off the fields rather than tracked, so a filter added later is
    included by construction. ``sort``, ``descending`` and ``trash`` are
    excluded deliberately: they reorder or re-address, they do not hide.
    """
    return bool(
        filters.text
        or filters.favorites_only
        or filters.status != "all"
        or filters.kind != "all"
    )


#: Below this many rows the list is drawn whole. Every card is the same height,
#: so clipping one is exact -- the threshold is not about correctness, it is
#: about not putting a scroll-dependent code path underneath the ordinary case:
#: a library of thirty assets fits on screen, gains nothing from being clipped,
#: and is what every screenshot and nearly every test draws.
CLIP_THRESHOLD = 40


def _clipper(ctx: Any, count: int):
    """A predicate saying which cards are off screen, or ``None`` to draw all.

    Written by hand rather than with ``ImGuiListClipper``, which was not in
    this repo at all before now and does not fit this list: a clipper owns the
    loop and wants a flat run of identical items, and this loop interleaves
    date-group headings that decide where the rows after them sit. What it
    needs from a clipper is the *arithmetic*, and that is exact here because
    ``_card`` draws one child of a fixed height -- ``CARD_HEIGHT`` or
    ``COMPACT_HEIGHT`` -- plus one item spacing.

    Skipping a card emits a ``dummy`` of the same footprint, so the scrollbar,
    the scroll position and every row's place in the list are what they would
    have been. Two cards are never skipped: the one keyboard navigation has
    asked to scroll to, because ``set_scroll_here_y`` has to be *called* on it,
    and the selected one, so that arriving by any other route still lands.
    """
    if count < CLIP_THRESHOLD:
        return None
    view = float(imgui.get_window_size().y)
    if view <= 0.0:
        return None
    scroll = float(imgui.get_scroll_y())
    # ``dummy`` takes the same item spacing after it that ``end_child`` does,
    # so the height alone reproduces a card's whole footprint and the spacing
    # does not appear in this arithmetic.
    row = sp(COMPACT_HEIGHT if filters_compact(ctx) else CARD_HEIGHT)
    keep = {ctx.state.library_scroll_to, ctx.state.selected}

    def skip(job_id: str) -> bool:
        if job_id in keep:
            return False
        top = float(imgui.get_cursor_pos_y())
        if top + row < scroll or top > scroll + view:
            imgui.dummy((0.0, row))
            return True
        return False

    return skip


def _load_more(ctx: Any) -> None:
    """The window is the newest N of M -- and the filters above apply only to
    that window, so a history longer than it needs to say so rather than let a
    search quietly miss what it never loaded."""
    loaded = len(ctx.cache.jobs)
    total = ctx.cache.total
    # J88, and it is the whole of that item: a filter applied to a window is
    # not a search, and the difference is invisible unless something says so at
    # the exact moment it matters -- which is when a filter is on *and* the
    # window is short of the history. "Load older" is right there, and pressing
    # it is what turns the filter into a search of everything loaded.
    if filtering(ctx.state.filters) and total > loaded:
        widgets.text_colored(
            theme.WARN,
            f"Filtering the newest {loaded} of {total}.",
        )
        widgets.muted("Older assets are not searched until they are loaded.")
    # O119. The window only ever grows, and after a few presses "newest N" is a
    # page nobody wants to scroll back through -- and the only way back was to
    # restart the app.
    if ctx.cache.limit > jobs_cache.LIST_LIMIT and controls.small_button(
        "Jump back to the newest##library-reset"
    ):
        ctx.cache.reset_window()
    if total <= loaded:
        # A failed COUNT(*) makes ``total`` a floor rather than a total (E43),
        # and on a full page that floor is exactly ``loaded`` -- so the honest
        # answer is "there may be more", with the button still there. Silently
        # returning here is what made a locked database look like a short
        # history.
        if ctx.cache.count_error and loaded >= ctx.cache.limit:
            widgets.muted(f"Showing the newest {loaded}; could not count the rest.")
            if ctx.cache.can_load_more() and controls.button(
                "Load older##library-more", (-1, 0)
            ):
                ctx.cache.load_more()
        return
    widgets.muted(f"Showing the newest {loaded} of {total}.")
    if not ctx.cache.can_load_more():
        # The window is at the service's ceiling. Offering the button here is
        # offering an action that cannot happen, which is the one thing a
        # button must never be -- and it read as broken rather than as full.
        widgets.muted("This is as far back as the library loads at once.")
        return
    if controls.button("Load older##library-more", (-1, 0)):
        ctx.cache.load_more()


# --- filters ----------------------------------------------------------------
#
# The query syntax used to be a four-line tooltip on the box (J87), which is the
# shape of help that is only ever read by somebody who already suspects it is
# there: a tooltip cannot be discovered by a user who has no reason to hover, it
# says nothing while you are typing, and none of its four lines can be acted on.
# UX.md Phase 4 replaces it with the prefixes themselves, under the box, while
# the box has focus -- in-flow discovery, and one click puts a prefix in the
# query rather than describing it.

# Whether the box should be given the keyboard on the *next* frame, because a
# chip was clicked on this one. Module state rather than ``AppState``: it lives
# for one frame and is meaningless to anything that is not this pane, which is
# the same argument ``_footer_px`` above makes.
_prefix_focus = [False]

# Whether the chip row is up. Latched, because clicking a chip takes the
# keyboard *off* the box: a row drawn only while the box is active would vanish
# on the mouse-down and the click would land on whatever the layout collapsed
# up into that space.
_prefix_open = [False]


def append_prefix(text: str, field: str) -> str:
    """The query with ``field:`` appended, ready to be typed into.

    Pure, so the rules are assertable without a frame. Two of them matter. A
    prefix already at the end is not doubled -- clicking ``tag:`` twice is a
    mis-click, not a request for ``tag:tag:`` -- and a prefix is separated from
    whatever came before it by a space, because ``rustytag:wood`` is one word
    that parses as free text and quietly matches nothing.
    """
    head = text.rstrip()
    if head.endswith(f"{field}:"):
        return text
    return f"{head} {field}:" if head else f"{field}:"


#: What ``Filters.status`` and ``Filters.kind`` may be, and what each is called.
#:
#: Module constants rather than literals inside ``_filters`` because the
#: full-window library offers the same two sets as lists of selectables
#: (the UI redesign, wave 4.4). Two copies would be two vocabularies: a rail entry
#: naming a kind the combo does not have is a filter that can be set and not
#: cleared.
STATUS_OPTIONS = [
    ("all", "any status"),
    ("done", "done"),
    ("running", "running"),
    ("error", "failed"),
]

KIND_OPTIONS = [
    ("all", "any kind"),
    ("reference", "references"),
    ("tile", "tiles"),
    ("model", "meshes"),
    ("rig", "rigs"),
    ("sheet", "sheets"),
    ("sprite", "sprite sheets"),
    ("tilesheet", "tile sheets"),
]


def prefix_chips(filters: Any, *, active: bool) -> None:
    """The clickable prefixes under the filter box, while it is in use."""
    if active:
        _prefix_open[0] = True
    if not _prefix_open[0]:
        return
    hovered = False
    for index, field in enumerate(QUERY_FIELDS):
        # Derived from the parser's own table, never re-typed: a chip offering a
        # prefix ``parse_query`` does not know is a button that turns the query
        # into free text, which is precisely the failure the prefixes exist to
        # avoid being silent about.
        label = f"{field}:"
        if index:
            widgets.same_line_or_wrap(imgui.calc_text_size(label).x + sp(tokens.SP_3))
        if controls.small_button(f"{label}##filter-prefix-{field}"):
            filters.text = append_prefix(filters.text, field)
            _prefix_focus[0] = True
        hovered = hovered or imgui.is_item_hovered()
    if not active and not hovered:
        _prefix_open[0] = False


def _filters(ctx: Any, jobs: list[Any]) -> None:
    """The three selects, the star and the tick, on two rows rather than one.

    Measured, not guessed: three 110 px combos plus two square buttons come to
    417 px, and the sidebar this pane lives in is one of three fixed widths
    (``layout.SIDEBAR_WIDTHS``, 260/300/360 design px -- M106 replaced the
    single 300 the rest of this comment was written against), the narrowest of
    which leaves 250 inside the padding. A child window *clips* rather than
    wraps, so the star spent its whole life drawn past the right edge -- neither
    visible nor clickable -- and the tick would have joined it. Widths come off
    the live content region for the same reason a constant was the bug: ``sp``
    scales the sidebar with the monitor, the user picks its width, and 110 did
    not scale with either.
    """
    filters = ctx.state.filters
    imgui.set_next_item_width(-1)
    if _prefix_focus[0]:
        # A *command*, issued on the frame after a chip was clicked and never
        # again -- ``focus.py``'s rule, and for the same reason: asking every
        # frame puts the caret back at the start of the box mid-word.
        _prefix_focus[0] = False
        imgui.set_keyboard_focus_here()
    # The hint carries the syntax (J87). Nowhere else can: the box is
    # full-width in a sidebar of 260 to 360 design px, so a (?) beside it would
    # push it off the edge at any of them, and a line of help under it would
    # cost a row on every frame to explain something most queries never use.
    filters.text = widgets.input_text(
        "##filter", filters.text, max_length=120, hint="Filter... (tag: status: kind:)"
    )
    prefix_chips(filters, active=imgui.is_item_active())
    spacing = imgui.get_style().item_spacing.x
    half = (imgui.get_content_region_avail().x - spacing) * 0.5
    filters.status = widgets.combo("##status", filters.status, STATUS_OPTIONS, width=half)
    imgui.same_line()
    filters.kind = widgets.combo("##kind", filters.kind, KIND_OPTIONS, width=half)
    # The three square buttons share the sort row, so what is left for the
    # combo is what they and their gaps do not take. Three, not two: the (?)
    # right-aligns itself onto whatever line is current, so a row that reserved
    # room for only the star and the tick put it exactly on top of the tick --
    # same pixels, and the later item takes the click, so the select-all opened
    # the manual instead.
    # Three square buttons on this row: the direction toggle, the tick and the
    # (?). The star left for the view row, where the *other* control that picks
    # a pile already was -- see ``_view_row``. The reservation is computed
    # rather than written out for the reason the comment above gives -- a child
    # window clips, so anything that overruns the sidebar's chosen width is
    # neither visible nor clickable, and that width is now the user's to change.
    buttons = 3 * (imgui.get_frame_height() + spacing)
    filters.sort = widgets.combo(
        "##sort",
        filters.sort,
        list(SORTS),
        width=imgui.get_content_region_avail().x - buttons,
    )
    imgui.same_line()
    # Direction (J85). A glyph rather than two combo entries per key: "size,
    # ascending" and "size, descending" as separate options doubles a list the
    # user reads every time, to express one bit.
    #
    # ASCII carets rather than lucide chevrons, and that is not laziness: the
    # icon atlas is a pinned 0.525.0 subset that carries chevron-down and *not*
    # chevron-up, and a codepoint the atlas does not hold renders as the
    # missing-glyph box. Two glyphs that disagree about which family they are
    # from would be worse than two that agree about being text.
    down = filters.descending
    if widgets.icon_button(
        "v" if down else "^",
        # "Reverse the order" rather than a list of what each key's two ends
        # are called: there is no descending order of a *name* anybody means by
        # default, so the only wording that stays true for every key is the one
        # that describes the toggle rather than the result.
        "Sorted the usual way - click to reverse" if down else "Reversed - click to restore",
    ):
        filters.descending = not down
    imgui.same_line()
    _select_all(ctx, jobs)
    # Before ``_failures``, which starts a line of its own: the (?) joins the
    # line that is current when it is called, and joining that one would put it
    # on a row that is not always there.
    manual_render.help_button(ctx, "library")
    _view_row(ctx)
    _failures(ctx)


def _view_row(ctx: Any) -> None:
    """Favourites, the trash and density: which *set* is on screen, and how
    tightly it is drawn (J89/J91).

    Its own row rather than three more squares on the sort row, which is
    already a computed reservation against a sidebar that clips -- and one
    whose width is the user's to pick (``layout.SIDEBAR_WIDTHS``). These are
    also a different kind of control from the ones above: everything on the
    rows above narrows what you are looking at, and these change which pile it
    comes from and how it is drawn.

    Favourites is here, beside the trash, rather than on the sort row where it
    used to sit. ``library_full._rail`` draws the same two fields together
    under one **Collections** heading, on exactly that argument -- they pick a
    pile -- and the two compositions write the same ``Filters`` object. One
    model deserves one vocabulary, so the pairing and its order agree in both.
    """
    filters = ctx.state.filters
    # A star that lights up, not a checkbox labelled "*".
    starred = filters.favorites_only
    if starred:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.WARN)))
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(*theme.rgba(theme.WARN, 0.2)))
    if widgets.icon_button(icons.STAR, "Favourites"):
        filters.favorites_only = not starred
    if starred:
        imgui.pop_style_color(2)
    imgui.same_line()
    # Trash next, immediately after the star, because the rail lists the two
    # of them together in that order.
    lit = filters.trash
    if lit:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.WARN)))
    if widgets.icon_button(icons.TRASH, "Show the trash" if not lit else "Back to the library"):
        filters.trash = not lit
        # The tick set is per-view: a bulk Delete armed in the library and
        # fired in the trash would mean something completely different.
        ctx.state.checked.clear()
    if lit:
        imgui.pop_style_color()
    imgui.same_line()
    # Density trails the pair: it is the one control here that changes how the
    # set is *drawn* rather than which set it is.
    compact = filters_compact(ctx)
    if widgets.icon_button(
        icons.LIST if compact else icons.LAYERS,
        "Comfortable rows" if compact else "Compact rows",
    ):
        ctx.settings.set("library_compact", not compact)
    if lit:
        imgui.same_line()
        widgets.text_colored(theme.WARN, "Trash")


def filters_compact(ctx: Any) -> bool:
    """Whether cards are drawn tightly (J89). Persisted, because it is a
    preference about a workshop rather than about a moment in one."""
    return bool(ctx.settings.get("library_compact"))


def _failures(ctx: Any) -> None:
    """A way to the failed jobs, when there are any and they are not shown.

    A failed job says why it failed in the inspector and nowhere else, so
    before this the only route to the reason was to already know which card to
    click -- and after a sweep or an overnight batch that is the one thing the
    user does not know. Its own full-width row rather than a fourth control on
    the filter rows: three combos and two square buttons already overrun the
    sidebar at every width it is offered in, and a child window clips rather
    than wraps.
    """
    filters = ctx.state.filters
    if filters.status == "error":
        return
    count = ctx.cache.failures(filters)
    if not count:
        return
    label = "1 job failed" if count == 1 else f"{count} jobs failed"
    imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
    clicked = controls.small_button(f"{label} - show##library-failures")
    imgui.pop_style_color()
    if clicked:
        filters.status = "error"


def _select_all(ctx: Any, jobs: list[Any]) -> None:
    """Tick every card the filters are showing -- the bulk bar's other half.

    Deliberately *shown*, not "every job": the list is a window onto the newest
    N of M (see ``_load_more``), so a control claiming to select everything
    would silently leave the older ones out of the delete that follows. It
    flips to Deselect once the shown set is fully ticked, so the same button
    undoes it rather than leaving Clear as the only way back.
    """
    shown = [job["id"] for job in jobs]
    checked = ctx.state.checked
    picked = bool(shown) and all(job_id in checked for job_id in shown)
    icon = icons.SQUARE_DASHED if picked else icons.CHECK
    tip = "Deselect the assets shown" if picked else "Select every asset shown, for bulk actions"
    if widgets.icon_button(icon, tip, enabled=bool(shown)):
        if picked:
            checked.difference_update(shown)
        else:
            checked.update(shown)


# --- cards ------------------------------------------------------------------


def _card(ctx: Any, job: Any, queue_pos: dict[str, int] | None = None) -> None:
    state = ctx.state
    job_id = job["id"]
    selected = state.selected == job_id
    imgui.push_id(job_id)
    # The selection moves between cards rather than cutting: the card being
    # left fades its fill and its accent edge out while the one being chosen
    # fades them in, which is one number per card and no knowledge of where the
    # selection was. Both halves ride the same value, or a card would keep its
    # raised fill for 200 ms after its edge had gone.
    chosen = motion.value(f"library/sel/{job_id}", 1.0 if selected else 0.0)
    if chosen > 0.0:
        imgui.push_style_color(
            imgui.Col_.child_bg.value,
            imgui.ImVec4(*theme.mix(theme.PANEL, theme.ELEV_2, chosen)),
        )
    # Keyboard navigation asks the list to bring the selection into view (I79).
    # Before ``begin_child`` and not inside it: ``set_scroll_here_y`` scrolls
    # the *current* window, and inside the card that is the card. Consumed here
    # so a scroll is a one-frame event rather than a position the list keeps
    # snapping back to while the user drags the scrollbar.
    if state.library_scroll_to == job_id:
        state.library_scroll_to = None
        imgui.set_scroll_here_y(0.5)
    origin = imgui.get_cursor_screen_pos()
    height = sp(COMPACT_HEIGHT if filters_compact(ctx) else CARD_HEIGHT)
    # A card, not a pane: this is ``widgets.card``'s idiom drawn by hand, so it
    # keeps the surface radius rather than taking ``tokens.RADIUS_PANE``.
    imgui.push_style_var(imgui.StyleVar_.child_rounding.value, sp(tokens.RADIUS_L))
    if imgui.begin_child("card", (0, height), imgui.ChildFlags_.borders.value):
        _card_body(ctx, job, queue_pos)
    imgui.end_child()
    imgui.pop_style_var()
    # The child window is the item a drag lifts, so this has to follow
    # ``end_child`` and precede the click test below -- imgui treats a started
    # drag as not-a-click, which is what stops a drag from also selecting.
    draggable_source(ctx, job)
    if chosen > 0.0:
        imgui.pop_style_color()
        # The accent edge is the selection mark; a raised fill alone reads as
        # hover, not choice. It grows out of the card's middle as it arrives,
        # so a selection reads as travelling down the list rather than as two
        # unrelated things fading.
        half = height * 0.5
        imgui.get_window_draw_list().add_rect_filled(
            (origin.x, origin.y + half * (1.0 - chosen)),
            (origin.x + sp(3), origin.y + half * (1.0 + chosen)),
            imgui.get_color_u32(theme.rgba(theme.ACCENT, chosen)),
            sp(2),
        )
    # The whole card selects, not just a title: the card *is* the affordance.
    if imgui.is_item_clicked():
        select(ctx, job_id)
    imgui.pop_id()


#: The placeholder glyph table moved to :mod:`.thumbs` in the UI redesign, wave 4.3,
#: when Home's Resume grid started drawing the same pictures. Re-exported under
#: its old name because it is what the library means by "what does this job look
#: like", and because a pane asking the library is asking the right module.
thumb_glyph = thumbs.thumb_glyph


def _card_body(ctx: Any, job: Any, queue_pos: dict[str, int] | None = None) -> None:
    job_id = job["id"]
    compact = filters_compact(ctx)
    thumbs.job_thumb(ctx, job, sp(COMPACT_THUMB if compact else THUMB_SIZE))
    imgui.same_line()

    # begin_group returns nothing -- the pair is unconditional, and wrapping it
    # in an `if` is how a frame ends up one end_group short.
    imgui.begin_group()
    name = job.get("name") or job.get("prompt") or job_id
    imgui.text_wrapped(name if len(name) <= 46 else name[:43] + "...")
    # J90. The truncation is what makes a workshop of long prompts navigable at
    # all, and it is also what makes two assets that differ in their last five
    # words indistinguishable -- so the whole string is one hover away. Only
    # when it *was* truncated: a tooltip repeating what is already on screen is
    # noise on every card.
    if len(name) > 46 and imgui.is_item_hovered():
        imgui.set_tooltip(name)
    if compact:
        # The status pill alone, on the same line: a compact row is for
        # scanning names, and the badges below are the detail you switch out of
        # compact to read.
        imgui.same_line()
        widgets.status_pill(job["status"])
        _followup_badge(job, compact=True)
        imgui.end_group()
        _card_context(ctx, job)
        return
    widgets.status_pill(job["status"])
    _followup_badge(job)
    # Between the two, and that order is the rule rather than a preference:
    # ``quality_badge`` may draw nothing and owns its own ``same_line`` for it,
    # so anything unconditional has to be ahead of it -- a badge placed after
    # would inherit the ``same_line`` that call did not spend.
    widgets.stage_badge(job, inline=True)
    widgets.quality_badge(job, inline=True)
    rank = (job.get("params") or {}).get("rank")
    if isinstance(rank, dict) and rank.get("score") is not None:
        imgui.same_line()
        widgets.muted(f"{float(rank['score']) * 100:.0f}%")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "How well framed this reference is, how close it looks to the "
                "profile's style anchor when there is one, and -- with "
                "PickScore installed -- how likely a person is to prefer it. "
                "Advisory."
            )
    if job["status"] == "queued":
        # Where in line it is: the queue used to be invisible past the pill.
        if queue_pos is None:
            queue_pos = {
                jid: i + 1
                for i, jid in enumerate(
                    j["id"] for j in reversed(ctx.cache.jobs) if j.get("status") == "queued"
                )
            }
        position = queue_pos.get(job_id)
        if position is not None:
            imgui.same_line()
            widgets.muted(f"#{position} in queue")
    if job.get("parent_id"):
        # ``same_line_or_wrap``, not ``same_line``: this is the *fourth* thing
        # on the badge row, and behind a status pill, a stage badge and a
        # quality badge there is nothing left of a 300 dp card -- it reached
        # the screen as "| from a refere", clipped at the card's edge with no
        # scrollbar to the rest of it (the UI redesign, wave 6).
        # And the leading pipe goes with the line: it is a separator from the
        # badges to its left, so on its own row it would be a stray glyph
        # against the card's edge.
        inline = "| from a reference"
        imgui.same_line()
        fits = imgui.get_content_region_avail().x >= imgui.calc_text_size(inline).x
        if not fits:
            imgui.new_line()
        widgets.muted(inline if fits else "From a reference")

    progress = ctx.runtime.progress(job_id)
    if progress is not None:
        widgets.progress_bar(float(progress.get("percent") or 0.0))
        widgets.muted(str(progress.get("label") or ""))
    else:
        _card_actions(ctx, job)
    imgui.end_group()
    _card_context(ctx, job)


def followup_failure_tooltip(job: Any) -> str:
    """One compact explanation for a card whose parent job still succeeded."""
    params = job.get("params") if isinstance(job, dict) else None
    stage = job.get("stage") if isinstance(job, dict) else None
    return "\n".join(
        f"{item['label']} was not queued: {item['message']}"
        for item in followups.records(params, stage)
    )


def _followup_badge(job: Any, *, compact: bool = False) -> None:
    message = followup_failure_tooltip(job)
    if not message:
        return
    imgui.same_line()
    widgets.text_colored(theme.WARN, "!" if compact else "follow-up failed")
    if imgui.is_item_hovered():
        imgui.set_tooltip(message)


def _card_context(ctx: Any, job: Any) -> None:
    """Right-click anywhere on the card opens the same menu the ellipsis does
    (I82) -- the same menu, drawn once, rather than a second list of actions
    that would drift from it the first time one is added.

    Drawn from here rather than from ``_card_actions``, which a *running* job
    replaces with its progress bar and which a compact row does not draw at
    all: the overflow was unreachable for the whole length of a job, which is
    exactly when Cancel and Delete are wanted. ``open_popup`` and
    ``begin_popup`` have to share an ID stack, so both the detection and the
    menu live inside the card's own child window.
    """
    if imgui.is_window_hovered() and imgui.is_mouse_clicked(1):
        # Right-clicking selects first: a menu acting on a card other than the
        # marked one is how the wrong asset gets deleted.
        #
        # Which is also why the comparison baseline is remembered *here*, on
        # the frame the menu opens and before the selection moves. "Compare
        # with selected" read ``state.selected`` when the item was clicked, by
        # which time this line had already made it the target -- so it compared
        # a mesh with itself (UX-04).
        ctx.state.compare_baseline = ctx.state.selected
        select(ctx, job["id"])
        imgui.open_popup("more")
    _overflow(ctx, job)


def _card_actions(ctx: Any, job: Any) -> None:
    action = primary_action(job, rigging_available=ctx.rigging_available)
    if action is not None and controls.small_button(ACTIONS[action]):
        run_action(ctx, job, action)
    imgui.same_line()
    if widgets.small_icon_button(icons.ELLIPSIS, "More actions"):
        imgui.open_popup("more")
    imgui.same_line()
    checked = job["id"] in ctx.state.checked
    changed, value = controls.checkbox("##pick", checked)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Select for bulk actions")
    if changed and value != checked:
        ctx.state.toggle_check(job["id"])
    imgui.same_line()
    favourite = bool(job.get("favorite"))
    if favourite:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.WARN)))
    if widgets.small_icon_button(icons.STAR, "Unfavourite" if favourite else "Favourite"):
        ctx.submit(
            f"fav:{job['id']}",
            svc_jobs.update_job,
            ctx.svc,
            job["id"],
            {"favorite": not favourite},
        )
    if favourite:
        imgui.pop_style_color()


def _overflow(ctx: Any, job: Any) -> None:
    if not imgui.begin_popup("more"):
        return
    widgets.popup_chrome(_imgui=imgui)
    job_id = job["id"]
    files = job.get("files") or []
    if job.get("deleted_at"):
        # A trashed asset offers exactly two things, and nothing else: every
        # action below would either fail or quietly resurrect the row into the
        # workshop without saying so.
        _trash_menu(ctx, job_id)
        imgui.end_popup()
        return
    # N115: the two "get me out of the app with this" actions, at the top
    # because they are the ones a user arrives at the menu already looking for.
    if controls.menu_item("Copy job id", "", False)[0]:
        imgui.set_clipboard_text(job_id)
        ctx.toast("Job id copied.", "success")
    if controls.menu_item("Open folder", "", False)[0]:
        _open_folder(ctx, job_id)
    imgui.separator()
    if controls.menu_item("Rename...", "", False)[0]:
        ctx.prompts.ask(
            dialogs.Prompt(
                title="Rename",
                label="Name",
                value=job.get("name") or "",
                on_accept=lambda value: ctx.submit(
                    f"rename:{job_id}", svc_jobs.update_job, ctx.svc, job_id, {"name": value}
                ),
            )
        )
    copyable = bool(job.get("params"))
    if copyable and controls.menu_item("Copy settings to form", "", False)[0]:
        copy_settings(ctx, job)
    if job["status"] in ("done", "error", "cancelled"):
        # A hand-made reference has no generator behind it, so there is nothing
        # a new seed could change; the service refuses it, and offering the
        # menu item anyway only buys the user an error toast.
        rerollable = not (
            job["kind"] == "image" and job.get("stage") == "reference"
        )
        if rerollable and controls.menu_item("Reroll", "", False)[0]:
            ctx.submit(f"rerun:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="reroll")
        if _remeshable(job) and controls.menu_item("Remesh", "", False)[0]:
            ctx.submit(f"remesh:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode="remesh")
    # The 2D half of the same pair as Open in Clay, and gated the same way: on
    # a predicate the mode owns, answered from the cached row alone so the
    # frame thread never stats anything. Above the mesh block because the two
    # are mutually exclusive -- a reference has no ``model.glb`` -- and the
    # loader reuses an already-open tab rather than forking a second one over
    # the same file.
    from .. import inker_mode

    if inker_mode.can_edit_job(ctx, job) and controls.menu_item("Open in Inker", "", False)[0]:
        inker_mode.open_job_reference(ctx, job)
    _map_and_atlas_items(ctx, job, files)
    _troupe_item(ctx, job)
    if "model.glb" in files:
        # Clay prefers the ``build.wblk`` sidecar when the asset was authored
        # here, and imports ``model.glb`` -- the optimized, grounded, served
        # mesh -- when it was not. Never ``source.glb``: that is the raw
        # reconstruction, and nothing downstream of the pipeline uses it.
        if controls.menu_item("Open in Clay", "", False)[0]:
            from .. import clay_mode

            clay_mode.edit_asset_in_clay(ctx, job)
        # The baseline is captured *before* the menu ran, not read after it.
        # Right-clicking a card selects it first (see ``_context_menu`` -- a
        # menu acting on a card other than the marked one is how the wrong
        # asset gets deleted), so by the time this item is clicked
        # ``state.selected`` is already this card: "Compare with selected"
        # compared a mesh with itself, and a later sync could make both sides
        # identical (UX-04). The label says which way round it goes.
        if controls.menu_item("Compare selected with this", "", False)[0]:
            compare(ctx, job_id)
        if ctx.rigging_available and controls.menu_item("Rig", "", False)[0]:
            run_action(ctx, job, "rig")
        # The Poser was reachable from the pipeline through exactly one button,
        # inside the inspector's Rig & Pose tab -- so a user with six rigged
        # props and a walk cycle to author had to already know the mode
        # existed. Same gate the pose pane uses, answered from the cached row.
        if "rig.glb" in files and controls.menu_item("Open in Poser", "", False)[0]:
            from . import pose_panel

            pose_panel.open_in_poser(ctx, job)
        _send_to_troupe_item(ctx, job)
    imgui.separator()
    if controls.menu_item("Delete", "Delete", False)[0]:
        # No confirm (J91): the trash *is* the confirmation, and it is a better
        # one -- an undo the user can take an hour later beats a question they
        # answer in half a second while looking at something else. The
        # irreversible delete has kept the question, in the trash.
        delete_asset(ctx, job_id)
    imgui.end_popup()


def _send_to_troupe_item(ctx: Any, job: dict[str, Any]) -> None:
    """Take this mesh into Troupe -- rigging it first if it is not rigged.

    Kept separate from :func:`_troupe_item`, which goes the other way: that
    one opens a finished *charsheet row* in the mode that plays it, and this
    one sends a *mesh* in to have one made. Offered for an unrigged mesh
    deliberately -- that is the case the door exists for -- so the label is
    not enough on its own and the hint says the rig happens, because for an
    unrigged mesh that is minutes of CPU behind a button not called "Rig".
    """
    from .. import troupe_mode

    if not troupe_mode.can_send_to_troupe(ctx, job):
        return
    rigged = "rig.glb" in (job.get("files") or [])
    # Both branches name the humanoid requirement now. The unrigged one always
    # did; the *rigged* one said only "render a character sheet", which is the
    # case where the mesh may already be a quadruped -- Poser offers seven
    # skeletons and Troupe's clip library carries clips for exactly one. The
    # refusal is real, immediate and well worded, but a user who has to press
    # the button to discover the rule has been told too late.
    hint = (
        "Render a character sheet from this mesh. Needs a humanoid rig -- "
        "Troupe's clip library has no clips for the other six skeletons."
        if rigged
        else "Rig this mesh as a humanoid, then render a character sheet from it."
    )
    if controls.menu_item("Send to Troupe", "", False, tooltip=hint)[0]:
        troupe_mode.send_to_troupe(ctx, job)


def _troupe_item(ctx: Any, job: dict[str, Any]) -> None:
    """Back into Troupe from a finished sheet.

    The route *out* of the library for a charsheet row, where
    :func:`_send_to_troupe_item` above is the route in for a mesh.

    Troupe's sheets *are* library assets -- a ``charsheet`` row, filed under
    the "sheet" card kind -- and there was no way to get from one back to the
    mode that plays it: the only doors were Home's tile and the rail. The row
    carries both halves of the selection (``source_job`` and ``sheet_id``), so
    this is answered from the cached row like every other item here.
    """
    if job.get("kind") != "charsheet" or job.get("status") != "done":
        return
    params = job.get("params") or {}
    source = str(params.get("source_job") or "")
    sheet_id = str(params.get("sheet_id") or "")
    if not source:
        return
    if controls.menu_item("Open in Troupe", "", False)[0]:
        from .. import troupe_mode

        # The same call a finished charsheet's "Show" toast makes, so the
        # menu item and the toast cannot disagree about which sheet.
        troupe_mode.open_sheet(ctx, source, sheet_id)


def _map_and_atlas_items(ctx: Any, job: dict[str, Any], files: Any) -> None:
    """The Plotter and Packwright half of the same pair as *Open in Clay*.

    Two kinds of entry, and the difference matters. **Reopening** is offered
    only when the asset carries an authored document beside it -- answered from
    ``params["authored"]``, which the exporter set, rather than from a stat,
    because this menu is built on the frame thread. Unlike *Open in Clay* there
    is no fallback: a reference with no ``map.wmap`` cannot be reopened as a
    map at all, so the item must not be offered rather than offered and refused.
    **Consuming** an image is offered for any reference: a tileset or an atlas
    source can be any picture, including one the pipeline generated, which is
    most of why these entries are worth having.
    """
    from .. import packwright_mode, plotter_mode

    authored = (job.get("params") or {}).get("authored")
    if authored == "plotter" and controls.menu_item("Open in Plotter", "", False)[0]:
        plotter_mode.edit_asset_in_plotter(ctx, job)
    if authored == "packwright" and controls.menu_item("Open in Packwright", "", False)[0]:
        packwright_mode.edit_asset_in_packwright(ctx, job)
    if "input.png" not in files:
        return
    if controls.menu_item("Add to Plotter as a tileset", "", False)[0]:
        plotter_mode.use_as_tileset(ctx, job)
    if controls.menu_item("Add to a Packwright atlas", "", False)[0]:
        packwright_mode.add_job_source(ctx, job)


def _trash_menu(ctx: Any, job_id: str) -> None:
    if controls.menu_item("Restore", "", False)[0]:
        restore_asset(ctx, job_id)
    if controls.menu_item("Delete permanently...", "", False)[0]:
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Delete permanently?",
                message="The job and everything derived from it are removed from disk. "
                "This cannot be undone.",
                confirm_label="Delete",
                cancel_label="Keep",
                on_confirm=lambda: purge_asset(ctx, job_id),
            )
        )


def _open_folder(ctx: Any, job_id: str) -> None:
    """Show a job's directory in the OS file manager (N115).

    ``ctx.open_log``'s rule, applied to a directory: hand the path to the
    shell rather than spawning a browser, so it is not a subprocess and
    therefore not a kill-on-close-job question. Silent about a missing
    directory in the sense that it says so as a toast -- a job whose files were
    removed by hand is a real state, not a fault.

    ``startfile`` itself goes through ``ctx.submit`` rather than running
    inline -- it is a blocking shell call (Explorer can take a moment to
    launch on a cold start), and ``open_log`` already makes the same call the
    same way. A raised ``OSError`` is caught by ``TaskRunner.poll`` and
    reaches the frame thread as the ordinary failed-task toast; the friendlier
    "Could not open the folder" wording is one this generic path cannot carry,
    which is the trade for not blocking the frame on a shell call.
    """
    import os

    path = ctx.job_dir(job_id)
    if not path.exists():
        ctx.toast("That job's folder is not on disk.", "warn")
        return
    ctx.submit(f"open-folder:{job_id}", os.startfile, str(path))  # noqa: S606 -- a directory, handed to the shell


# --- actions ----------------------------------------------------------------


def select(ctx: Any, job_id: str) -> None:
    ctx.state.select(job_id)
    job = ctx.cache.get(job_id)
    if job is None:
        return
    # Promotion's source follows the selection when the selection is something
    # that can be promoted; anything else leaves it alone, so switching to a
    # mesh to look at it does not silently change what Make 3D would submit.
    if job.get("stage") == "reference" and job.get("status") == "done":
        ctx.state.source_job = job_id


def select_relative(ctx: Any, delta: int) -> None:
    """Move the selection ``delta`` cards through the list on screen (I79).

    *The list on screen*: the same filtered, sorted sequence the cards are
    drawn from, so Down always goes to the card below and never to whatever the
    unfiltered order would have put there.

    Clamped rather than wrapped. The list is the newest N of M and the ends
    mean something -- one press at the top landing on the oldest asset the
    window happens to hold is a jump the user cannot undo by pressing the other
    arrow, because the wrap is not symmetric once the window grows.
    """
    jobs = ctx.cache.visible(ctx.state.filters)
    if not jobs:
        return
    ids = [job["id"] for job in jobs]
    current = ctx.state.selected
    if current in ids:
        index = min(max(ids.index(current) + delta, 0), len(ids) - 1)
    else:
        # Nothing selected, or a selection the filters no longer show: enter the
        # list from the end the key is travelling away from.
        index = 0 if delta > 0 else len(ids) - 1
    select(ctx, ids[index])
    ctx.state.library_scroll_to = ids[index]


#: Where the full-window grid publishes how many columns it drew, so the
#: keyboard can move by one *row*. In ``state.preview`` because it is a fact
#: about this frame's layout and nothing else: it changes when the window is
#: dragged, and a persisted copy would be a stale number the arrows obeyed.
COLUMNS_SLOT = "library_columns"


def columns(ctx: Any) -> int:
    """How many cards the grid fitted across, last frame. 1 when unknown.

    One, not zero: the sidebar list is a single column and so is a grid nobody
    has drawn yet, and a zero here would make Up and Down do nothing at all.
    """
    return max(int(ctx.state.preview.get(COLUMNS_SLOT) or 1), 1)


def select_grid(ctx: Any, dx: int, dy: int) -> None:
    """Arrow keys over the grid. Left/right by one, up/down by a row.

    The extension of :func:`select_relative` the full window needs (the UI redesign,
    wave 4.4), and deliberately a second entry point rather than a flag on it:
    the same function is bound to Up/Down in the *sidebar*, where the list is
    one card wide and a row is one card. Which of the two a key means is a
    property of the pane it was pressed in, so the pane's dispatcher chooses.
    """
    select_relative(ctx, dx + dy * columns(ctx))


def open_selected(ctx: Any) -> None:
    """Enter, from the library mode's keyboard path.

    Opens the selected asset in the mode that shows it -- the same routing
    Home's Resume list applies to an asset row, so the two lists cannot
    disagree about where an asset opens. Through ``asset_open`` (and so through
    ``create_stages.go``) rather than assignment, so Esc still knows it came
    from the library. Silently nothing with no selection: Enter with no cursor
    has nothing it could mean.
    """
    from .. import asset_open

    job = ctx.cache.get(ctx.state.selected)
    if job is None:
        return
    asset_open.open_asset(ctx, job)


def copy_settings(ctx: Any, job: Any) -> None:
    """Load a job's recipe back into the 2D form, so it can be varied.

    Reroll re-runs a job as it was; this is the other half -- start from what
    it used and change one thing. Prompt history only ever restored the prompt
    text, which left every guidance field, the model, the LoRA and the
    conditioning strengths behind.

    Public: the results tray's "Vary" is the same verb on the same row and was
    reaching for the private name to do it.
    """
    from .. import create_assets
    from ..state import form_from_params

    form = form_from_params(job.get("params") or {})
    # The output switch is restored from the *stage*, because that is where a
    # job's tile-ness lives -- params never carries it, so a form filled from
    # params alone would open in Object mode and quietly offer to make a mesh
    # of a texture. The whole job row is in hand here; form_from_params is not
    # given it, because it is the params allowlist and must stay one.
    form["output"] = "tile" if job.get("stage") == "tile" else "reference"
    params = job.get("params") or {}
    if "asset_type" not in params:
        form["asset_type"] = create_assets.legacy_asset_type(form)
    create_assets.sync_legacy_fields(form)
    from .. import create_stages

    ctx.state.form_2d = form
    create_stages.go(ctx, "reference")
    ctx.toast("Settings copied to the form.")


def _remeshable(job: Any) -> bool:
    """Whether "keep this image, rebuild the mesh" is offered for this job.

    One predicate rather than the same expression at each call site, because
    the two have already drifted apart once. The retry ladder learned that a
    tile has no subject to reconstruct and the context menu did not, so
    right-click -> Remesh on a finished tile reached ``rerun_job``, was refused
    and came back as an error toast. That is the rule ``rerollable`` two lines
    above the menu item states in prose: never offer an action the service will
    refuse. Stated once, it cannot be honoured in one place and not the other.

    The tile-sheet arm is the same lesson a second time. A tile sheet publishes
    an ``input.png`` and is not staged "tile", so it satisfied both terms above
    and the exclusion list quietly grew a hole the day the kind was added:
    ``rerun_job`` refuses it by *kind*, in both modes, so Remesh offered it an
    error toast -- and the retry ladder, which routes an errored card through
    ``mode="remesh" if _remeshable(job) else "reroll"``, had no reachable answer
    at all. Keyed on the kind rather than the stage for exactly that reason:
    the service's refusal is by kind, and a predicate that guesses at a
    different column is one that goes stale the next time a stage moves.
    """
    return (
        "input.png" in (job.get("files") or [])
        and job.get("stage") != "tile"
        and job.get("kind") != "tile_sheet"
    )


def run_action(ctx: Any, job: Any, action: str) -> None:
    job_id = job["id"]
    if action == "cancel":
        ctx.submit(f"cancel:{job_id}", svc_jobs.cancel_job, ctx.svc, job_id)
    elif action == "retry":
        mode = "remesh" if _remeshable(job) else "reroll"
        ctx.submit(f"retry:{job_id}", svc_jobs.rerun_job, ctx.svc, job_id, mode=mode)
    elif action == "promote":
        from .. import create_stages

        ctx.state.source_job = job_id
        # ``follow=False``: the source is named on the line above, and walking
        # the selection onto a mesh this reference has *already* produced would
        # put a finished asset in the inspector beside a form about to make a
        # different one.
        create_stages.go(ctx, "mesh", follow=False)
    elif action == "rig":
        # The skeleton and the key are the Rig stage's, not a second copy:
        # picking "quadruped" there and then rigging from a card here has to
        # use the same template, and two spellings of the submit key would let
        # both fire at once.
        from . import stage_rig

        ctx.submit(
            stage_rig.rig_key(job),
            svc_rig.create_rig,
            ctx.svc,
            job_id,
            template=stage_rig.skeleton(ctx),
        )
    elif action == "open":
        # A job that stops at an image opens at the stage that made it; a tile
        # has no mesh at all; and a follow-up row opens the asset whose
        # directory holds what it made. All three are ``asset_open``'s to know,
        # which is what keeps this and the landing page's identical "open this
        # asset" from being two answers to one question.
        from .. import asset_open

        asset_open.open_asset(ctx, job)
    elif action == "inker":
        from .. import inker_mode

        inker_mode.open_job_reference(ctx, job)
    elif action == "clay":
        from .. import clay_mode

        clay_mode.edit_asset_in_clay(ctx, job)
    elif action == "plotter":
        from .. import plotter_mode

        plotter_mode.use_as_tileset(ctx, job)


#: The one task key an import lands under, claimed by ``main`` so the result
#: is not "delivered nowhere". Unkeyed by path on purpose: two imports at once
#: is not a workflow, and the dedupe ``submit`` already does on a repeated key
#: is what stops a double-clicked button opening two pickers.
IMPORT_MESH_KEY = "import-mesh"


def pick_and_import_mesh(ctx: Any) -> None:
    """Ask for a mesh file, then bring it into the library.

    **The gap this closes.** Until 2026-08-30 geometry could enter the library
    exactly two ways -- reconstructed by the pipeline, or built in Clay -- and a
    mesh a user already had could not enter at all. The only file-open path for
    a ``.glb`` was a drop onto Clay, which converts it into an editable Clay
    *document* and refuses a rigged one outright, because Clay has no skinning.
    So the supplied-base-mesh path that Troupe's whole intake assumes had no
    door: ``jobs.import_mesh`` was written, tested and reachable only from Clay
    exporting its own work back out.

    Nothing new happens to the file. It becomes an ordinary finished model row,
    and every consumer inherits it at once -- Send to Troupe, the Poser, the
    triangle retarget, every export -- because all of them are pure functions of
    ``model.glb``. That is the same payoff ``import_mesh``'s own docstring
    claims for Clay, collected a second time for a door that costs nothing.
    """
    ctx.submit(IMPORT_MESH_KEY, _pick_and_import, ctx)


def import_mesh_path(ctx: Any, path: Any) -> None:
    """The same, for a file that arrived by drag and drop."""
    ctx.submit(IMPORT_MESH_KEY, _read_and_import, ctx, Path(path))


def _pick_and_import(ctx: Any) -> dict[str, Any] | None:
    """Blocking; task thread only. ``None`` is a cancel and says nothing."""
    picked = dialogs.open_file("Import a mesh", dialogs.GLB_FILTER)
    if picked is None:
        return None
    return _read_and_import(ctx, picked)


def _read_and_import(ctx: Any, path: Path) -> dict[str, Any]:
    """Blocking; task thread only.

    The size is checked against the file rather than after reading it: the
    ceiling exists because a 100 MB GLB is already past what anything
    downstream wants, and reading a two-gigabyte one into memory to discover
    that is how a refusal becomes a swap storm. ``import_mesh`` re-checks the
    bytes it is handed, so this is a cheaper first gate and not the rule.
    """
    from ...service.errors import Invalid
    from ...service.validation import MAX_MESH_BYTES

    if path.stat().st_size > MAX_MESH_BYTES:
        raise Invalid(
            f"{path.name} is over {MAX_MESH_BYTES // (1024 * 1024)} MB.", field="mesh"
        )
    # ``stem`` for both: the name is what the library shows, and the prompt is
    # what every "what was this?" reader falls back to -- a file called
    # ``knight_base`` should answer both rather than leaving one blank.
    return svc_jobs.import_mesh(
        ctx.svc, path.read_bytes(), name=path.stem, prompt=path.stem
    )


COMPARE_KEY = "compare-parse"


def compare(ctx: Any, job_id: str) -> None:
    """Put ``job_id``'s mesh beside the selected one. Off-thread and contained.

    Three things were wrong with this and they compound (UX-04).

    *It could compare a mesh with itself.* Right-clicking a card selects it
    before the menu runs, so "Compare with selected" read a selection this very
    gesture had just moved onto the target. The baseline is captured in
    ``_context_menu`` now, on the frame the menu opens, and an identical pair is
    refused with a sentence rather than drawn as two copies of one mesh.

    *It parsed on the frame thread.* ``Viewer.compare`` loads the GLB and
    uploads GPU resources inline -- on a large model that is a visible freeze,
    and ``_sync_viewer`` grew its ``pending``/task split for precisely this
    reason. The parse goes through ``TaskRunner`` here, exactly as the primary
    viewer's does.

    *It had no error boundary.* The primary path catches around adopt and
    toasts; this one let a malformed GLB or a failed upload propagate out of
    the frame loop, which exits the app. A compare is a *look* at something --
    the one thing it must never do is take the session with it.
    """
    if ctx.state.comparing == job_id:
        ctx.state.comparing = None
        if ctx.viewer is not None:
            ctx.viewer.exit_compare()
        return
    baseline = ctx.state.compare_baseline or ctx.state.selected
    if baseline == job_id:
        ctx.toast("Select a different asset to compare this one with.", "error")
        return
    if ctx.viewer is None:
        return
    path = ctx.job_dir(job_id) / "model.glb"
    # Held so a refusal can put them back. These two fields *are* the record of
    # the compare already in flight, and the two lines below overwrite it --
    # so clearing them to None on refusal cancelled the earlier compare as
    # well as declining this one: ``_adopt_compare`` then saw
    # ``compare_pending`` was None, failed its own ``!= wanted`` check and
    # dropped the result that was on its way. Neither mesh appeared and
    # nothing was said, so the menu item read as inert.
    was_comparing, was_pending = ctx.state.comparing, ctx.state.compare_pending
    ctx.state.comparing = job_id
    ctx.state.compare_pending = path
    if not ctx.submit(COMPARE_KEY, ctx.viewer.parse_model, path, tag=path):
        # Another compare parse is in flight. Its result is checked against
        # ``compare_pending`` before adoption, so this one is simply dropped
        # rather than queued -- the same rule ``_sync_viewer`` follows. The
        # one in flight is left exactly as it was.
        ctx.state.comparing = was_comparing
        ctx.state.compare_pending = was_pending


def delete_assets(ctx: Any, job_ids: list[str]) -> None:
    """Trash several rows as one act: one toast, one undo, N submits.

    The candidate picker's "Keep -> Delete the others" raised a separate
    "Moved to trash." with a separate Undo *per loser*, so choosing between
    eight attempts finished with seven stacked toasts and no way to put them
    all back at once. The undo carries the whole batch, because what the user
    did was one thing.
    """

    kept = [str(job_id) for job_id in job_ids if job_id]
    if not kept:
        return
    if len(kept) == 1:
        delete_asset(ctx, kept[0])
        return
    for job_id in kept:
        ctx.state.checked.discard(job_id)
        if ctx.state.selected == job_id:
            ctx.state.select(None)
        ctx.submit(f"delete:{job_id}", svc_jobs.trash_job, ctx.svc, job_id)
    ctx.toast(f"Moved {len(kept)} to trash.", "info", "undo", " ".join(kept))


def delete_asset(ctx: Any, job_id: str) -> None:
    """Move one asset to the trash: the tick, the selection, then the row (J91).

    Public because the candidate picker offers exactly this for the attempts
    the user did not keep -- through this function rather than a second
    ``svc_jobs`` call, so a loser is removed by the one path that also clears
    the selection and the tick set.

    It trashes rather than deletes, and every caller keeps its old name: what
    the user asked for is "get this out of my library", and the *reversibility*
    of that is a property of the app rather than of each call site. The
    permanent version is :func:`purge_asset`, and the only thing that reaches
    it is the trash.
    """
    ctx.state.checked.discard(job_id)
    if ctx.state.selected == job_id:
        ctx.state.select(None)
    if ctx.submit(f"delete:{job_id}", svc_jobs.trash_job, ctx.svc, job_id):
        # The offer, in the one place the user is already looking (UX.md Phase
        # 3). Raised on submission rather than on completion, which is the same
        # optimism the row's disappearance already shows: the toast dwells for
        # seconds and ``restore_job`` on a row that never got trashed is a
        # no-op refusal the runner already toasts.
        ctx.toast("Moved to trash.", "info", "undo", job_id)


def restore_asset(ctx: Any, job_id: str) -> None:
    """Undo a trashing. Takes one id, or several space-separated.

    The batch spelling is what ``delete_assets``' one Undo hands back, and it
    is a string because that is what a toast action carries.
    """

    for one in str(job_id).split():
        ctx.state.checked.discard(one)
        ctx.submit(f"restore:{one}", svc_jobs.restore_job, ctx.svc, one)


def purge_asset(ctx: Any, job_id: str) -> None:
    """Delete one trashed asset for real."""
    ctx.state.checked.discard(job_id)
    if ctx.state.selected == job_id:
        ctx.state.select(None)
    ctx.submit(f"purge:{job_id}", svc_jobs.delete_job, ctx.svc, job_id)


# --- bulk and storage -------------------------------------------------------


def _bulk(ctx: Any, jobs: list[Any]) -> None:
    """The actions for what is ticked, and how much of it is off screen.

    ``state.checked`` is not pruned when the filters change, and that is worth
    keeping -- ticking a few meshes, then switching to references to tick a few
    more, is a real way to use this. What is not defensible is doing it
    silently: tick everything shown, narrow the filter, press Delete, and the
    wider set goes while the confirm names only a count. So the count says how
    much of itself is no longer on screen, and the confirm repeats it, because
    the destructive path must never describe a smaller act than it performs.

    *Not shown* rather than *filtered out*: an id can also be here because its
    job fell off the newest-N window or was deleted from somewhere else, and
    the honest word covers all three.

    Laid out through :mod:`~warlock.studio.toolbar` (the UI redesign, wave 4.2), and
    the count is drawn *before* it so the row's arithmetic sees the width the
    sentence actually left. **Clear and Delete are pinned**, which is the two
    halves of the pinning rule in one row: a destructive action behind ``...``
    is one somebody finds by accident, and the escape from a selection has to
    be at least as reachable as the thing that acts on it.
    """
    picked = sorted(ctx.state.checked)
    if not picked:
        return
    shown = {job["id"] for job in jobs}
    hidden = sum(1 for job_id in picked if job_id not in shown)
    imgui.separator()
    imgui.text(f"{len(picked)} selected" + (f" ({hidden} not shown)" if hidden else ""))
    imgui.same_line()
    items = [
        toolbar.Item(
            "clear", "Clear", icons.X, tooltip="Untick everything.", pinned=True
        )
    ]
    if ctx.state.filters.trash:
        # The trash's own two, and *only* those two: exporting a zip of assets
        # the user has thrown away is not a thing anybody wants, and offering
        # it would put "Delete" beside "Export" in the one view where Delete
        # cannot be taken back.
        items += [
            toolbar.Item("restore", "Restore", icons.UNDO),
            toolbar.Item(
                "purge",
                "Delete permanently...",
                icons.TRASH,
                role=toolbar.ButtonRole.DESTRUCTIVE,
                pinned=True,
            ),
        ]
    else:
        # The one bulk action the failures affordance sets you up for and never
        # offered. "N jobs failed - show" filters the list, Select all ticks
        # them, and then the bar had Export/Save/Delete -- nothing to *retry*
        # with, so a user who had just been told about twelve failures retried
        # them one card at a time (UX-14). ``run_action(..., "retry")`` already
        # exists per job; this is a loop over it, and it picks remesh-or-reroll
        # per job exactly as the single-job path does.
        failed = [
            job
            for job in (ctx.cache.get(j) for j in picked)
            if job and job["status"] == "error"
        ]
        if failed:
            items.append(
                toolbar.Item("retry", f"Try again ({len(failed)})", icons.REFRESH)
            )
        items.append(toolbar.Item("zip", "Export zip...", icons.DOWNLOAD, priority=1))
        if ctx.export_dir:
            # Only when one is configured: the feature is off unless
            # WARLOCK_EXPORT_DIR is set, and a button that can only fail is
            # worse than no button.
            items.append(
                toolbar.Item("folder", "Save to project", icons.SAVE, priority=1)
            )
        items.append(
            toolbar.Item(
                "delete",
                "Delete",
                icons.TRASH,
                role=toolbar.ButtonRole.DESTRUCTIVE,
                pinned=True,
            )
        )
    toolbar.toolbar(
        "library-bulk", items, lambda key: _bulk_action(ctx, key, picked, hidden)
    )


def _bulk_action(ctx: Any, key: str, picked: list[str], hidden: int) -> None:
    """What each key in the bulk row does. Its own function because the row is
    now data, and because a confirm is worth reading beside the count it names."""
    if key == "clear":
        ctx.state.checked.clear()
    elif key == "restore":
        for job_id in picked:
            restore_asset(ctx, job_id)
    elif key == "purge":
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Delete permanently?",
                message=_delete_message(len(picked), hidden) + " This cannot be undone.",
                confirm_label="Delete",
                cancel_label="Keep",
                on_confirm=lambda: [purge_asset(ctx, j) for j in picked],
            )
        )
    elif key == "retry":
        for job in (ctx.cache.get(j) for j in picked):
            if job and job["status"] == "error":
                run_action(ctx, job, "retry")
    elif key == "zip":
        _export_zip(ctx, picked)
    elif key == "folder":
        ctx.submit(
            "export-folder", svc_export.export_to_folder, ctx.svc, picked, ["model.glb"]
        )
    elif key == "delete":
        # The confirm stays here where the per-card one went (J91): a bulk
        # action's count is the thing worth checking, and "not shown" in that
        # sentence is the case the whole message exists for.
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Move these to the trash?",
                message=_delete_message(len(picked), hidden),
                confirm_label="Move to trash",
                cancel_label="Keep",
                on_confirm=lambda: [delete_asset(ctx, j) for j in picked],
            )
        )


def _delete_message(total: int, hidden: int) -> str:
    """What the confirm says, as a function rather than inline so the wording
    the destructive path uses is something a test can read."""
    message = f"{total} jobs and everything derived from them."
    if hidden:
        message += f" {hidden} of them are not in the list you can see."
    return message


def _export_zip(ctx: Any, ids: list[str]) -> None:
    def run():
        dest = dialogs.save_file("Export selection", "warlock_export.zip", dialogs.ZIP_FILTER)
        if dest is None:
            return None
        return svc_export.bulk_export(ctx.svc, ids, ["model.glb"], dest)

    ctx.submit("export-zip", run)


def _storage(ctx: Any, jobs: list[Any]) -> None:
    if ctx.state.filters.trash:
        # The bar describes the set on screen, so in the trash it is the
        # trash's own figure rather than the workshop's -- two totals on one
        # row, one of which counts everything the other one does not, is worse
        # than either alone.
        _trash_bar(ctx, jobs)
        return
    storage = ctx.cache.storage
    if ctx.cache.storage_error:
        # Before the figure rather than after it (E45): a stale total beside a
        # warning reads as current, and this line is the only thing that says
        # the number below is the last one that worked.
        widgets.text_colored(theme.WARN, "Could not measure disk use.")
        if imgui.is_item_hovered():
            imgui.set_tooltip(ctx.cache.storage_error)
    if storage:
        from ..state import format_bytes

        widgets.muted(f"{storage['job_dirs']} jobs - {format_bytes(storage['bytes'])}")
    # **Prune...** and **Clean library...** used to be here, and moved to
    # Settings > Storage in the UI redesign, wave 4.1. The figure stays because it
    # describes the list above it; the two bulk deletes went because a footer
    # under a scrolling list is where "clean library" reads as an action on
    # what you can see. That also retires the layout hazard this function
    # carried three comments about: the storage line was the only item holding
    # the buttons' ``same_line`` inside the panel, so before the first
    # measurement landed they attached to the full-width list child above and
    # were drawn off the right edge -- unclickable exactly while a new install
    # had nothing else on screen.


# The task key the trash measurement runs under, and the ``AppState.preview``
# slot its answer lands in. Named once because three lines read them, and
# because a task key is a namespace: ``main._on_task_done`` claims several by
# prefix and this one must collide with none of them.
TRASH_SIZE_KEY = "trash-size"
TRASH_SIZE_SLOT = "trash_size"


def measure_trash(ctx: Any, jobs: list[Any]) -> dict[str, Any] | None:
    """The trash's count and bytes, measured off the frame thread. -> the last
    answer, or None before the first one lands.

    Named for the asking rather than for the answer, so it cannot be mistaken
    at a call site for ``service.jobs.trash_size``, which is what it runs.

    ``service.jobs.trash_size`` walks a directory per trashed job, so it is a
    task and the bar draws the answer it gave -- the storage walk's own
    arrangement (``main._request_storage``) and for exactly the same reason: a
    disk walk on the frame thread is the stall ``attach_files`` exists to avoid.
    That figure was written for this bar and then never wired to it, so the one
    question a trash view has to answer -- "is emptying this worth anything" --
    had no answer on screen at all.

    Re-measured when the *loaded* trash page changes identity, which is
    ``jobs_cache.sweep_summary``'s stated limitation applied to a number rather
    than to a toast: the figure itself is complete (``trash_size`` reads
    ``store.trashed()``, not a page), and only the decision to re-measure is
    window-scoped -- so a trash deeper than one page is reported correctly and
    merely re-measured a little less eagerly. The submit is re-attempted every
    frame the key is stale rather than armed once, because ``TaskRunner.submit``
    refuses a key already in flight and nothing else would re-arm it (the
    ``findings_dirty`` lesson).

    Its own function, and returning the answer rather than drawing it, so the
    arithmetic that decides *when* to walk the disk is assertable without a GL
    context.
    """
    key = tuple(job["id"] for job in jobs)
    cached = ctx.state.preview.get(TRASH_SIZE_SLOT)
    if cached is not None and cached[0] == key:
        return cached[1]

    def run(stamp: Any = key) -> Any:
        answer = svc_jobs.trash_size(ctx.svc)
        # Written on the task thread and read on the frame thread, as
        # ``JobsCache.storage_error`` beside it already is: one dict assignment,
        # and nothing branches on it twice. Stamped with the key this run was
        # asked for, so an answer arriving after the trash moved on is replaced
        # rather than believed.
        ctx.state.preview[TRASH_SIZE_SLOT] = (stamp, answer)
        return answer

    ctx.submit(TRASH_SIZE_KEY, run)
    # The last measurement meanwhile, stale or not, exactly as the workshop
    # figure above is drawn: a number that vanishes while it is being re-taken
    # is a row that changes height every time something is restored.
    return cached[1] if cached else None


def trash_summary(answer: Any) -> str | None:
    """"12 assets - 4.1 GB", or None when nothing has been measured yet.

    Pure, for the reason ``_delete_message`` is: the wording is the feature.
    Assets rather than jobs, because the trash is a view of the library and the
    library calls its rows assets everywhere else.
    """
    if not isinstance(answer, dict):
        return None
    from ..state import format_bytes

    count = int(answer.get("count") or 0)
    what = "1 asset" if count == 1 else f"{count} assets"
    return f"{what} - {format_bytes(int(answer.get('bytes') or 0))}"


def _trash_bar(ctx: Any, jobs: list[Any]) -> None:
    """What the trash is holding, and the one way to let go of it."""
    summary = trash_summary(measure_trash(ctx, jobs))
    if summary is not None:
        widgets.muted(summary)
        # Inside the branch for the reason the storage figure's ``same_line``
        # is: before the first measurement lands there is no item on this line,
        # and a ``same_line`` there attaches to the full-width list child above.
        imgui.same_line()
    if widgets.destructive_button("Empty trash...", (0, 0)):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Empty the trash?",
                message="Every trashed asset and everything derived from it is "
                "removed from disk. This cannot be undone.\n\n"
                "Assets you accepted, and any image you labelled, are kept: "
                "they are what the quality judge and the tier checks are "
                "measured against. Delete those individually if you want them "
                "gone.",
                confirm_label="Empty",
                cancel_label="Keep",
                on_confirm=lambda: ctx.submit("empty-trash", svc_jobs.empty_trash, ctx.svc),
            )
        )


# How many jobs a prune keeps, while its question is on screen (O116). Module
# state rather than AppState because it is the *dialog's* state: it exists for
# as long as the dialog does, and it is deliberately not persisted -- a
# destructive default the user set once and forgot is exactly the kind of
# setting that deletes something they wanted.
_prune_keep = [20]
PRUNE_KEEP_DEFAULT = 20


def ask_prune(ctx: Any) -> None:
    _prune_keep[0] = PRUNE_KEEP_DEFAULT

    def body() -> None:
        imgui.set_next_item_width(sp(120))
        changed, value = controls.input_int("Keep the newest", _prune_keep[0], 1, 10)
        if changed:
            # Floored at zero because the service refuses a negative, and a
            # question that can be answered unanswerably is worse than one that
            # clamps: the refusal would arrive as a toast after the dialog is
            # gone, describing a number the user can no longer see.
            _prune_keep[0] = max(0, value)

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Prune old assets?",
            # *Deleted*, not trashed, and that asymmetry with the card's Delete
            # is deliberate: prune exists to reclaim disk, and a prune that
            # moved two hundred jobs into the trash would free nothing while
            # reporting that it had.
            message="Everything but the newest N jobs is deleted from disk. "
            "Running jobs are kept, and so is anything you accepted or "
            "labelled. This cannot be undone.",
            confirm_label="Prune",
            cancel_label="Keep",
            body=body,
            on_confirm=lambda: ctx.submit(
                "prune", svc_jobs.prune_jobs, ctx.svc, _prune_keep[0]
            ),
        )
    )


def ask_clean(ctx: Any) -> None:
    """The one destructive action that keeps nothing, so it has to say so.

    Prune's wording ends "and so is anything you accepted or labelled", which
    is true of prune and is the exact promise this breaks. Saying only "this
    cannot be undone" would be technically honest and practically a trap: the
    user has been told twice already that their evidence is safe from a bulk
    delete, and this is the button for which that is false.
    """

    def body() -> None:
        storage = ctx.cache.storage
        if storage:
            from ..state import format_bytes

            widgets.muted(
                f"{storage['job_dirs']} jobs - {format_bytes(storage['bytes'])} "
                f"will be freed."
            )

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Delete every asset?",
            message="Every asset is removed from disk, trashed or not -- "
            "including ones you accepted and images you labelled, which is "
            "what the quality judge and the tier checks are measured against. "
            "The verdict rows survive; the pixels behind them do not. Your "
            "pose library, style anchors and settings are kept. This cannot "
            "be undone.",
            confirm_label="Delete everything",
            cancel_label="Keep",
            body=body,
            # The ``delete:`` prefix is load-bearing, not decoration:
            # ``main._on_task_done`` re-measures storage only for keys starting
            # delete:/prune/rename:/..., and a clean whose freed space did not
            # appear until the next idle tick would read as having done nothing.
            on_confirm=lambda: ctx.submit("delete:clean", svc_jobs.clean_jobs, ctx.svc),
        )
    )
