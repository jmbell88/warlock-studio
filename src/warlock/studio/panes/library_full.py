"""The library as a whole window: a filter rail, a grid of pictures, the inspector.

:mod:`.library` is still the library. It owns the filters, the cards, the
primary-action ladder, the bulk bar, the storage figure and every action a card
offers, and it goes on drawing the *compact* list in the right-hand sidebar of
the two generate modes -- unchanged, same API, same call sites. What this module
adds is a second **composition** of those parts for the one place there is room
for a better one (the UI redesign, wave 4.4).

The sidebar list is a good answer to "which of these am I picking while I look
at something else". It is a poor answer to "where is that chest I made on
Tuesday", which is what the full-window mode is for, and it was drawing exactly
the same 300 px column stretched across a 1600 px window: two thirds of the
screen given to the whitespace beside a filename.

So the three columns do the three jobs. The **rail** is where you narrow: the
query, the collections, the sort. The **grid** is where you recognise: the
asset's own picture at a size you can identify it by, with its state on top of
it. The **inspector** is where you act, and it is the library's existing one,
drawn only when a selection resolves -- an empty 340 px column beside a grid is
340 px of nothing.

Nothing here re-implements a filter or an action. Every control writes to the
same ``ctx.state.filters`` the sidebar's combos write to, and every card routes
through :func:`library.select`, so the two views cannot disagree about what is
shown or what a click does.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, layout, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import inspector, library, thumbs

#: The left rail, in design pixels. Narrower than a sidebar (260-360) because
#: it holds no form: a query box, a column of one-line selectables and a sort.
RAIL_W = 220.0

#: The inspector column. The same width the right sidebar gives it, so a job
#: looks the same here as it does beside the viewport.
INSPECTOR_W = 340.0

#: One grid cell. The picture is the cell; the name and the badges are what is
#: left under it. 160 is four to a 1600 px window with the rail and the
#: inspector both up, and eight with neither -- which is the range the mode is
#: actually used at.
CELL = 160.0


def draw(ctx: Any) -> None:
    jobs = ctx.cache.visible(ctx.state.filters)
    spacing = imgui.get_style().item_spacing.x
    # Resolved before the columns are sized, because whether it resolves is
    # what decides how wide the middle one is.
    selected = ctx.cache.get(ctx.state.selected)

    with layout.pane(
        "library-full/rail",
        (sp(RAIL_W), 0),
        layout.PaneRole.SIDEBAR,
        edge=layout.PaneEdge.RIGHT,
    ) as visible:
        if visible:
            _rail(ctx, jobs)
    imgui.same_line()

    width = -(sp(INSPECTOR_W) + spacing) if selected is not None else 0.0
    with layout.pane(
        "library-full/grid", (width, 0), layout.PaneRole.CONTENT
    ) as visible:
        if visible:
            _grid(ctx, jobs)

    if selected is not None:
        imgui.same_line()
        with layout.pane(
            "library-full/inspector",
            (0, 0),
            layout.PaneRole.INSPECTOR,
            edge=layout.PaneEdge.LEFT,
        ) as visible:
            if visible:
                inspector.draw(ctx)


# --- the rail ---------------------------------------------------------------


def _rail(ctx: Any, jobs: list[Any]) -> None:
    """Narrow the set: the query, the collections, the order.

    Selectables rather than the sidebar's combos, and that is the whole reason
    this column exists: a combo is a control you open to find out what it can
    say, and a rail with the room to list its options says them without being
    asked. The values written are identical -- the same ``Filters`` fields, off
    the same two option tables -- so switching between this and the sidebar
    never loses a filter.
    """
    filters = ctx.state.filters
    # The mode's one loud thing (the UI redesign, wave 6). ``app_settings`` argues
    # that a three-column workspace is named by the lit rail item and needs no
    # title of its own; this is not one of those. Clay, Inker and Review put a
    # *document* in the middle column, and the window is about the thing you
    # can see in it -- the library's middle column is a wall of other people's
    # pictures, and the only thing on screen that could say what the wall is
    # was the navigation chrome outside the pane. It goes at the top of the
    # rail rather than across the window: the columns are laid out before this
    # runs, so a full-width banner would mean a fourth region, and a heading
    # above the narrowing controls is where the eye starts anyway.
    # The former pane_title("Library") contract is now the shared pane header:
    # the title remains, with its contextual help anchored in the same row.
    widgets.pane_header("Library", help_text="Browse, filter, select, and export your assets.")
    widgets.section("Find")
    manual_render.help_button(ctx, "library")
    imgui.set_next_item_width(-1)
    filters.text = widgets.input_text(
        "##library-full-filter",
        filters.text,
        max_length=120,
        hint="Filter... (tag: status: kind:)",
    )
    library.prefix_chips(filters, active=imgui.is_item_active())

    widgets.section("Collections")
    # The two toggles first, because they change which *pile* you are looking
    # at rather than narrowing the one you are in -- the distinction
    # ``library._view_row`` already draws on.
    if _toggle("Favourites", icons.STAR, filters.favorites_only, theme.WARN):
        filters.favorites_only = not filters.favorites_only
    if _toggle("Trash", icons.TRASH, filters.trash, theme.WARN):
        filters.trash = not filters.trash
        # The tick set is per-view: a bulk Delete armed in the library and
        # fired in the trash would mean something completely different.
        ctx.state.checked.clear()

    widgets.section("Status")
    for key, label in library.STATUS_OPTIONS:
        if _entry(f"status/{key}", label, filters.status == key):
            filters.status = key

    widgets.section("Kind")
    for key, label in library.KIND_OPTIONS:
        if _entry(f"kind/{key}", label, filters.kind == key):
            filters.kind = key

    widgets.section("Order")
    spacing = imgui.get_style().item_spacing.x
    filters.sort = widgets.combo(
        "##library-full-sort",
        filters.sort,
        list(library.SORTS),
        width=imgui.get_content_region_avail().x
        - (imgui.get_frame_height() + spacing),
    )
    imgui.same_line()
    # ASCII carets rather than lucide chevrons: the pinned 0.525.0 subset
    # carries chevron-down and not chevron-up, and a codepoint the atlas does
    # not hold renders as the missing-glyph box.
    down = filters.descending
    if widgets.icon_button(
        "v" if down else "^",
        "Sorted the usual way - click to reverse"
        if down
        else "Reversed - click to restore",
    ):
        filters.descending = not down

    imgui.dummy((0, sp(tokens.SP_4)))
    library._storage(ctx, jobs)


def _entry(key: str, label: str, selected: bool) -> bool:
    """One collection row. -> whether it was clicked this frame."""
    return bool(controls.selectable(f"{label}##library-full-{key}", selected)[0])


def _toggle(label: str, icon: str, lit: bool, colour: int) -> bool:
    """A collection that is on or off rather than one of a set.

    Lit in its own colour rather than only by the selectable's highlight: these
    two are the rows that change what set you are in, and "am I looking at the
    trash" has to be answerable from across the room.
    """
    if lit:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(colour)))
    clicked = controls.selectable(f"{icon}  {label}##library-full-{label}", lit)[0]
    if lit:
        imgui.pop_style_color()
    return bool(clicked)


# --- the grid ---------------------------------------------------------------


def _grid(ctx: Any, jobs: list[Any]) -> None:
    style = imgui.get_style()
    pad = style.window_padding
    gap = style.item_spacing.x
    cell_w = sp(CELL)
    thumb = cell_w - pad.x * 2
    # One line under the picture, not two: the name is all a library cell says
    # in words -- the status is drawn *on* the thumbnail and everything else is
    # in the inspector -- so a second line's worth of height is a strip of
    # empty card under every asset.
    cell_h = thumb + pad.y * 2 + imgui.get_text_line_height_with_spacing()
    avail = imgui.get_content_region_avail().x
    count = max(int((avail + gap) // (cell_w + gap)), 1)
    # Published for the keyboard, which has to know how far a row is before it
    # can move by one. Per frame, because the window can be dragged.
    ctx.state.preview[library.COLUMNS_SLOT] = count

    _tools(ctx, jobs)
    imgui.separator()
    height = -_footer_height(ctx)
    if imgui.begin_child("library-full/cells", (0, height)):
        if not jobs:
            library._empty(ctx)
        group = ""
        column = 0
        for job in jobs:
            # Date grouping (J89), and only under the date sort: a "Today"
            # heading above a list ordered by size would be a lie about what
            # separates the rows below it. A heading also breaks the row, which
            # is why the column counter is reset rather than derived from the
            # index.
            if ctx.state.filters.sort == "newest":
                heading = library.date_group(job.get("created_at"))
                if heading != group:
                    group = heading
                    widgets.section(heading)
                    column = 0
            if column:
                imgui.same_line()
            column = (column + 1) % count
            _cell(ctx, job, (cell_w, cell_h), thumb, pad)
        library._load_more(ctx)
    imgui.end_child()
    library._bulk(ctx, jobs)


def _tools(ctx: Any, jobs: list[Any]) -> None:
    """Select-all and the failures affordance, over the grid.

    The density toggle is deliberately absent: a compact *card* is a shorter
    row, and this view has no rows. The trash lives in the rail with the other
    collections.
    """
    library._select_all(ctx, jobs)
    imgui.same_line()
    total = ctx.cache.total or 0
    shown = len(jobs)
    widgets.muted(
        f"{shown} shown" if shown == total else f"{shown} of {total} shown"
    )
    imgui.same_line()
    library._failures(ctx)


def _footer_height(ctx: Any) -> float:
    """What the bulk bar under the grid will take, or nothing when it is not
    drawn. One frame's arithmetic rather than ``library``'s measured footer:
    this pane's footer is the bulk bar alone -- the storage figure moved to the
    rail -- and the bulk bar is a single row that is either there or not."""
    if not ctx.state.checked:
        return 0.0
    return imgui.get_frame_height_with_spacing() + imgui.get_style().item_spacing.y


def _cell(
    ctx: Any, job: Any, size: tuple[float, float], thumb: float, pad: Any
) -> None:
    """One asset: its picture, its state on top of it, its name under it.

    The status pill is drawn *over* the thumbnail rather than beside the name,
    which is the one place in this app that overlaps two things on purpose: a
    grid is scanned rather than read, and a queued or failed asset has to be
    identifiable without the eye leaving the picture. The cursor is put back
    where the image left it afterwards, so the card's own height arithmetic is
    untouched.
    """
    job_id = job["id"]
    selected = ctx.state.selected == job_id
    name = str(job.get("name") or job.get("prompt") or job_id)
    imgui.push_id(job_id)
    # Keyboard navigation asks the grid to bring the selection into view (I79).
    # Before ``card``, not inside it: ``set_scroll_here_y`` scrolls the
    # *current* window, and inside the cell that is the cell.
    if ctx.state.library_scroll_to == job_id:
        ctx.state.library_scroll_to = None
        imgui.set_scroll_here_y(0.5)
    origin = imgui.get_cursor_screen_pos()
    with widgets.card(f"library-full/cell/{job_id}", size) as visible:
        if visible:
            imgui.dummy((0, pad.y - imgui.get_style().item_spacing.y))
            imgui.indent(pad.x)
            top = imgui.get_cursor_screen_pos()
            thumbs.job_thumb(ctx, job, thumb)
            after = imgui.get_cursor_screen_pos()
            imgui.set_cursor_screen_pos((top.x + sp(4), top.y + sp(4)))
            widgets.status_pill(job["status"])
            if job.get("favorite"):
                imgui.same_line()
                widgets.text_colored(theme.WARN, icons.STAR)
            imgui.set_cursor_screen_pos(after)
            imgui.text(widgets.fit_text(name, thumb))
            imgui.unindent(pad.x)
    library.draggable_source(ctx, job)
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
        # Always, not only when the label was trimmed: a grid cell shows about
        # twenty characters of a prompt, so the full one is worth having for
        # every card rather than for the long ones.
        imgui.set_tooltip(name)
    if imgui.is_item_clicked():
        library.select(ctx, job_id)
    library._card_context(ctx, job)
    if selected:
        widgets.ring(
            imgui.ImVec2(origin.x, origin.y),
            imgui.ImVec2(origin.x + size[0], origin.y + size[1]),
            theme.ACCENT,
            1.0,
            2.0,
        )
    imgui.pop_id()



