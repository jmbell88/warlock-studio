"""The Home screen: what changed, what the machine is doing, what you were on.

It used to be a grid of nine tiles, seven of which were the mode switch again
-- the same names under the same glyphs, one click either way -- while the
switch itself is drawn unconditionally at the top of every frame. So Home spent
its whole screen re-offering navigation that is permanently on screen, and the
two tiles that were *not* modes (Open Existing, Profiles) were real destinations
wearing a tile because there was nowhere else to put them. They are modes now,
and the grid is gone.

What is left is the three questions nothing in the app answered: what changed,
what is the machine doing, and what was I working on -- in one column, in that
order, with the answers weighted by how often they are the reason somebody is
here. "What changed" is a card you dismiss once per release; "what is the
machine doing" is one quiet line; "what was I working on" is the rest of the
screen, as pictures (REDESIGN.md wave 4.3).

Nothing here is persisted: ``AppState.mode`` defaults to ``"home"``, which is
what makes this appear on every launch rather than only the first ever.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import changelog
from .. import fonts, icons, modes, profiles, recents, theme, tokens, widgets
from ..manual import render as manual_render
from ..state import DEFAULT_FORM_3D, default_form_2d, format_bytes, set_mode
from ..tokens import sp
from . import thumbs

log = logging.getLogger(__name__)

# How many Resume rows are offered. A shortlist rather than a history: past a
# dozen the eye is scanning rather than recognising, and the Library mode is one
# click away for everything else.
MAX_RESUME = 12

# How stale the unreviewed-mesh count may get before Home re-asks. It is a table
# scan behind the one serialized connection, so it can never run on the frame
# thread -- ``pump`` submits it and the block draws the last answer, which is the
# same bargain the storage walk already makes.
UNREVIEWED_TTL = 30.0

# Enough to answer "is there a review pass waiting" without scanning the whole
# table. At the cap the figure is rendered with a "+", because a count that
# silently saturates is a wrong number rather than a rounded one.
UNREVIEWED_LIMIT = 200

# The mode each document kind opens in, and the glyph its Resume row wears.
# Off ``modes.MODES`` rather than a second table: a row that opened Clay under
# Plotter's icon is exactly the drift a hand-copied glyph produces.
_MODE_ICONS = {key: icon for key, _label, icon in modes.MODES}


@dataclass(frozen=True)
class Row:
    """One Resume row.

    ``kind`` is a document mode for a file and ``"asset"`` for a library row,
    which is the whole of what tells :func:`activate` how to open it. ``when``
    is epoch seconds, or ``None`` for a legacy recent-files entry that carries
    an order but no clock -- those sort last, the rule ``Filters.order`` applies
    to every unanswerable row.
    """

    kind: str
    key: str
    icon: str
    name: str
    when: float | None = None


# --- the resume list --------------------------------------------------------


def _asset_rows(ctx: Any) -> list[Row]:
    """Finished assets, newest first, derived from the job cache.

    Deliberately *not* written into :mod:`.recents`: an asset already has a
    ``created_at`` in the job store, and a second copy in the settings file
    would be a record of the library the library could contradict -- a row that
    is trashed or pruned simply stops being an answer here, with nothing to
    clean up.
    """
    cache = getattr(ctx, "cache", None)
    if cache is None:
        return []
    out: list[Row] = []
    for job in cache.jobs:
        # The cache page is the workshop and the trash together (that is what
        # lets ``Filters.trash`` be a view), so the trash is excluded here on
        # the row's own ``deleted_at`` -- a column already on the row, costing
        # no DB scan, and independent of whatever the library's filter bar is
        # set to. Home is not the library.
        if job.get("deleted_at"):
            continue
        if job.get("status") != "done":
            continue
        name = str(job.get("name") or job.get("prompt") or job.get("id") or "asset")
        # The same reference/tile/model split the library filter uses: a job
        # that stops at an image belongs to the pane that made it.
        mode = "2d" if job.get("stage") in ("reference", "tile") else "3d"
        when = job.get("created_at")
        out.append(
            Row(
                kind="asset",
                key=str(job.get("id") or ""),
                icon=_MODE_ICONS.get(mode, icons.IMAGE),
                name=name,
                when=float(when) if isinstance(when, int | float) else None,
            )
        )
    return out


def _document_rows(ctx: Any) -> list[Row]:
    settings = getattr(ctx, "settings", None)
    if settings is None:
        return []
    return [
        Row(
            kind=entry.kind,
            key=entry.path,
            icon=_MODE_ICONS.get(entry.kind, icons.FILE_IMAGE),
            name=Path(entry.path).name or entry.path,
            when=entry.when,
        )
        for entry in recents.entries(settings)
    ]


def rows(ctx: Any) -> list[Row]:
    """Every Resume row this frame, newest first.

    One merged list rather than a section per mode: the question is "what was I
    working on", and the answer to that does not sort by which editor it was in.
    Three things index this list -- the click, the arrow keys and Enter -- so
    there is exactly one function answering "what is the nth row".
    """
    found = _document_rows(ctx) + _asset_rows(ctx)
    found.sort(key=lambda row: (row.when is None, -(row.when or 0.0)))
    return found[:MAX_RESUME]


def activate(ctx: Any, index: int) -> None:
    """Open the ``index``-th row. Shared by the click and by Enter."""
    drawn = rows(ctx)
    if not drawn:
        return
    open_row(ctx, drawn[index % len(drawn)])


def open_row(ctx: Any, row: Row) -> None:
    if row.kind == "asset":
        # ``state.select`` plus a mode, never ``viewer.load_model``:
        # ``_sync_viewer`` owns the pose guard, ``viewer.pending`` and the
        # off-thread parse, and a shortcut past it reproduces two of those three
        # and gets the last one wrong.
        job = ctx.cache.get(row.key) if getattr(ctx, "cache", None) else None
        ctx.state.select(row.key)
        stage = (job or {}).get("stage")
        set_mode(ctx.state, "2d" if stage in ("reference", "tile") else "3d")
        return
    path = Path(row.key)
    if not path.exists():
        # A recent list that keeps offering a moved file is worse than a short
        # one, and a click that silently does nothing is worse than either.
        recents.forget(ctx.settings, row.kind, row.key)
        ctx.toast(f"{path.name} is no longer there.", "warn")
        return
    opener = {
        "inker": lambda: _open_with("inker_mode", ctx, path),
        "clay": lambda: _open_with("clay_mode", ctx, path),
        "plotter": lambda: _open_with("plotter_mode", ctx, path),
        "packwright": lambda: _open_with("packwright_mode", ctx, path),
    }.get(row.kind)
    if opener is None:
        return
    set_mode(ctx.state, row.kind)
    opener()


def _open_with(module: str, ctx: Any, path: Path) -> None:
    from importlib import import_module

    import_module(f"..{module}", __package__).open_path(ctx, path)


def move(ctx: Any, delta: int) -> None:
    """Up/Down between the Resume rows. Wraps: it is a short ring, not a list."""
    count = len(rows(ctx))
    if count:
        ctx.state.home_index = (ctx.state.home_index + delta) % count


def ago(when: float | None, now: float | None = None) -> str:
    """``just now`` / ``4m ago`` / ``yesterday``. Empty for an unstamped row.

    Coarse on purpose, as ``main._ago`` already is: the question is "was that
    the one from this afternoon", and a figure with more precision than that
    invites being read as a measurement.
    """
    if when is None:
        return ""
    delta = max(0.0, (time.time() if now is None else now) - when)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 172800:
        return "yesterday"
    return f"{int(delta // 86400)}d ago"


# --- the status block -------------------------------------------------------
#
# Four rows, each a sentence about this machine rather than about the work, and
# each clickable where it has somewhere to go. Pure, so the wording and the
# destinations are assertable without a window -- which is the half that went
# wrong the last time a Home row counted something.


@dataclass(frozen=True)
class Status:
    key: str
    icon: str
    text: str
    colour: int
    target: str = ""  # a mode to switch to, or "" for a row that only reports


def _health_status(ctx: Any) -> Status:
    checks = list(getattr(ctx.runtime, "checks", []) or [])
    failing = [c for c in checks if not c.ok]
    fatal = [c for c in failing if c.fatal]
    if not checks:
        return Status("health", icons.LOADER, "Diagnostics - still checking", theme.MUTED)
    if not failing:
        return Status(
            "health", icons.CIRCLE_CHECK, "Everything checks out", theme.MUTED, "settings"
        )
    what = "1 thing needs" if len(failing) == 1 else f"{len(failing)} things need"
    return Status(
        "health",
        icons.TRIANGLE_ALERT if not fatal else icons.CIRCLE_ALERT,
        f"{what} attention - set up models",
        theme.ERR if fatal else theme.WARN,
        "settings",
    )


def _queue_status(ctx: Any) -> Status:
    job = getattr(getattr(ctx, "cache", None), "active", None)
    if job is None:
        return Status("queue", icons.CIRCLE, "Queue idle", theme.MUTED)
    name = str(job.get("name") or job.get("prompt") or job.get("id") or "a job")
    if len(name) > 30:
        name = name[:29] + "-"
    percent = ""
    reporter = getattr(ctx, "progress", None)
    found = reporter(str(job.get("id") or "")) if callable(reporter) else None
    if isinstance(found, dict) and isinstance(found.get("percent"), int | float):
        percent = f" {int(found['percent'])}%"
    word = "running" if job.get("status") == "running" else "queued"
    return Status("queue", icons.PLAY, f"{word}: {name}{percent}", theme.ACCENT)


def _library_status(ctx: Any) -> Status:
    cache = getattr(ctx, "cache", None)
    if cache is None:
        return Status("library", icons.FOLDER_OPEN, "Library", theme.MUTED, "library")
    total = getattr(cache, "total", 0) or 0
    parts = [f"{total} assets"]
    try:
        failed = cache.failures(ctx.state.filters)
    except Exception:
        # Logged as well as swallowed, the rule every comparable site here
        # follows (``pose_panel._open_rig``). Silent, a broken filter or a
        # cache that cannot answer showed up as "0 failed" -- which is not a
        # missing number but a *wrong* one, and the one it is most reassuring
        # to be wrong about. The count is cosmetic, so it still must not take
        # the status line down with it.
        log.exception("could not count failed jobs for the landing status")
        failed = 0
    if failed:
        parts.append(f"{failed} failed")
    storage = getattr(cache, "storage", None)
    if isinstance(storage, dict) and storage.get("bytes"):
        parts.append(format_bytes(storage["bytes"]))
    return Status(
        "library", icons.FOLDER_OPEN, " - ".join(parts), theme.MUTED, "library"
    )


def _review_status(ctx: Any) -> Status | None:
    count = getattr(ctx.state, "home_unreviewed", None)
    if not count:
        # ``None`` is "not asked yet" and 0 is "nothing waiting"; neither is a
        # sentence worth a row, and they are deliberately not distinguished on
        # screen -- the difference matters to the cache, not to the reader.
        return None
    figure = f"{UNREVIEWED_LIMIT}+" if count >= UNREVIEWED_LIMIT else str(count)
    noun = "mesh" if count == 1 else "meshes"
    return Status(
        "review", icons.CIRCLE_CHECK, f"{figure} {noun} unreviewed", theme.MUTED, "review"
    )


def status_rows(ctx: Any) -> list[Status]:
    found = [_health_status(ctx), _queue_status(ctx), _library_status(ctx)]
    review = _review_status(ctx)
    if review is not None:
        found.append(review)
    return found


def _count_unreviewed(svc: Any) -> int:
    from ..review_mode import SOURCE

    return len(svc.store.unverdicted_models(source=SOURCE, limit=UNREVIEWED_LIMIT))


def pump(ctx: Any) -> None:
    """Re-ask for the unreviewed count when the last answer has gone stale.

    A submit refused by the runner (a key already in flight) leaves the stamp
    alone, so the next frame simply asks again -- the stamp is moved only when
    the request was *accepted*, which is the ``findings_dirty`` rule and the
    reason a burst of requests cannot strand the row on a stale figure.
    """
    now = time.monotonic()
    if now - ctx.state.home_unreviewed_at < UNREVIEWED_TTL:
        return
    if ctx.submit("home-unreviewed", _count_unreviewed, ctx.svc):
        ctx.state.home_unreviewed_at = now


# --- drawing ----------------------------------------------------------------
#
# One column (REDESIGN.md wave 4.3). It used to be two, news and status on the
# left and Resume on the right, which gave half of the app's first screen to a
# changelog -- the thing a user reads once per release -- and left "what was I
# working on" as a column of one-line selectables in the other half. The card
# now appears only when there is a release you have not dismissed, the six
# start-something buttons collapse into one primary and a menu, and Resume gets
# the room: a grid of pictures, because a thumbnail is how anybody recognises
# their own work and a filename is not.


#: Which status rows Home draws. ``status_rows`` still computes all of them --
#: it is the shared source the rail's health badge reads, and forking it would
#: be two answers to "is this install healthy" -- but the *library* row is not
#: one of Home's questions any more: it counted assets on a screen whose whole
#: lower half is now assets, and its destination is one click away in the rail.
HOME_STATUS = ("health", "queue", "review")

#: How many bullets the What's New card shows before "all release notes".
#: Three is a summary; the eighth bullet of a release is a changelog, and a
#: changelog is what the popup is for.
NEWS_BULLETS = 3

#: Where the dismissal is remembered. Keyed on the *version* rather than on a
#: boolean, so the next release brings the card back on its own -- a "seen"
#: flag would have to be cleared by whatever wrote the release, and nothing
#: does.
NEWS_SEEN_KEY = "news_seen_version"

#: The Resume grid's thumbnail, in design pixels. The picture is the cell:
#: everything else is one line of name and one of "4m ago" under it.
RESUME_THUMB = 136.0


def news_should_show(release: Any, seen: str) -> bool:
    """Whether the What's New card is drawn this frame.

    Pure, and taking the two values rather than a ``Ctx``, because the whole
    question is an interaction between a file and a settings key: a card that
    reappears after being dismissed, and one that never returns for the next
    release, are both bugs with no pixels in them.

    A release with no bullets is not shown. ``changelog.current`` falls back to
    the newest entry when the running version has no section of its own, which
    is right for "what changed recently" and would be wrong for a card that
    names a version -- but the fallback still carries bullets, so the empty
    case here is the genuinely empty one.
    """
    if release is None or not release.bullets:
        return False
    return seen != release.version


def draw(ctx: Any) -> None:
    pump(ctx)
    _header(ctx)
    # Once per draw: ``_status`` is the only consumer now, but the rule stands
    # for the reason it was written -- recomputing is two walks over the
    # runtime checks and the queue for one line of sentences.
    status = status_rows(ctx)
    # A scroller, because the Resume grid is as tall as it needs to be and the
    # window is not. Borderless -- this is the mode's whole surface, and a frame
    # drawn around the only thing on screen is a frame around nothing -- but
    # ``always_use_window_padding`` with it: a *borderless* child gets zero
    # window padding by default, which put the heading's first letter and the
    # first card's left edge hard against the window's.
    if imgui.begin_child(
        "landing/body", (0, 0), imgui.ChildFlags_.always_use_window_padding.value
    ):
        _news(ctx)
        _start(ctx)
        _status(ctx, status)
        _resume(ctx)
        _news_footer(ctx)
    imgui.end_child()


def _header(ctx: Any) -> None:
    with fonts.display(imgui):
        imgui.text("Warlock Studio")
    imgui.same_line()
    with fonts.small(imgui):
        # The app's version, on screen for the first time: ``main._version``
        # existed only for crash provenance, and "which build is this" was a
        # question the UI could not answer at all.
        widgets.muted(_version())
    manual_render.help_button(ctx, "home")
    if ctx.state.errors:
        for message in ctx.state.errors:
            widgets.text_colored(theme.ERR, message)
    imgui.dummy((0, sp(tokens.SP_3)))


# The installed version, once: ``main._version`` is an importlib.metadata
# distribution walk, the header and the news block both ask, and an installed
# version cannot change under a running process -- the changelog cache's own
# argument, applied to the other half of "what build is this".
_VERSION: str | None = None


def _version() -> str:
    global _VERSION
    if _VERSION is None:
        from ..main import _version as installed

        _VERSION = installed()
    return _VERSION


def _news(ctx: Any) -> None:
    """The dismissible What's New card, or nothing at all."""
    release = changelog.current(_version())
    seen = str(ctx.settings.get(NEWS_SEEN_KEY, "") or "")
    if not news_should_show(release, seen):
        return
    bullets = tuple(release.bullets[:NEWS_BULLETS])
    style = imgui.get_style()
    pad = style.window_padding
    avail = imgui.get_content_region_avail().x
    # The width ``text_wrapped`` will actually use inside the card, because
    # ``widgets.card`` has no scrollbar by design: a bullet measured against a
    # different number than it is drawn at is a bullet clipped off the bottom.
    # The dismiss button shortens only the *headline*'s line, which is measured
    # unwrapped anyway.
    wrap = max(avail - pad.x * 2, sp(120))
    with fonts.label(imgui):
        height = imgui.get_text_line_height() + style.item_spacing.y
    with fonts.small(imgui):
        for bullet in bullets:
            # The second positional is ``text_end``, not the hide-after-##
            # flag: the wrap width is the fourth argument, and passing it third
            # measures an unwrapped line.
            height += (
                imgui.calc_text_size(f"- {bullet}", None, False, wrap).y
                + style.item_spacing.y
            )
    height += pad.y * 2
    with widgets.card("landing/news", (0, height)) as visible:
        if visible:
            _card_margin(pad)
            with fonts.label(imgui):
                imgui.text(f"What's new in {release.version}")
            offset = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
            imgui.same_line(max(offset - sp(24), 0.0))
            if widgets.small_icon_button(
                f"{icons.X}##landing-news-dismiss", "Dismiss", borderless=True
            ):
                ctx.settings.set(NEWS_SEEN_KEY, release.version)
            for bullet in bullets:
                with fonts.small(imgui):
                    widgets.muted_wrapped(f"- {bullet}")
            imgui.unindent(pad.x)
    imgui.dummy((0, sp(tokens.SP_3)))


def _news_footer(ctx: Any) -> None:
    """The way back to every release, whether or not the card is showing.

    A ghost at the foot of the screen rather than a section of its own: the
    full changelog is a thing you go and look up, and it spent the whole of
    this pane's first design taking half the width to be looked up in.
    """
    imgui.dummy((0, sp(tokens.SP_4)))
    if widgets.ghost_button("All release notes..."):
        imgui.open_popup("landing-news-all")
    _news_popup()


def _news_popup() -> None:
    if not imgui.begin_popup("landing-news-all"):
        return
    releases = changelog.entries()
    if not releases:
        widgets.muted("No changelog shipped with this build.")
        imgui.end_popup()
        return
    current = changelog.current(_version())
    if imgui.begin_child("landing/news-scroll", (sp(520), sp(420))):
        # The current release open and the rest collapsed: "what changed" is
        # about this build, and the older ones are there so the answer has a
        # context rather than so it has a scrollback.
        for index, release in enumerate(releases):
            headline = release.version + (f" - {release.date}" if release.date else "")
            opened = current is not None and release.version == current.version
            flags = imgui.TreeNodeFlags_.default_open.value if opened else 0
            if imgui.collapsing_header(f"{headline}##news{index}", flags):
                for bullet in release.bullets:
                    with fonts.small(imgui):
                        widgets.muted_wrapped(f"- {bullet}")
    imgui.end_child()
    imgui.end_popup()


def _start(ctx: Any) -> None:
    """One primary and a menu, where six equal buttons used to be.

    The six were a 3-across grid of identically-weighted buttons -- a menu
    claiming all of them matter equally, drawn at the loudest weight the pane
    had, above the thing the user actually came back for. The menu says the
    same six words; the button says there is one place to start.
    """
    if widgets.primary_button(f"{icons.PLUS}  New...", (sp(160), 0)):
        imgui.open_popup("landing-new")
    if imgui.begin_popup("landing-new"):
        for key, label, icon, action in NEW_ITEMS:
            if imgui.menu_item(f"{icon}  {label}##landing-new-{key}", "", False)[0]:
                action(ctx)
        imgui.end_popup()
    imgui.dummy((0, sp(tokens.SP_2)))


def _status(ctx: Any, status: list[Status]) -> None:
    """One quiet line: what is wrong, what is running, what is waiting.

    Four stacked rows became one line because three of the four say "nothing to
    report" on a healthy idle install -- a block of chrome saying so at the top
    of every launch. What is left is only the rows that are *actionable*, and
    each still carries its own destination.
    """
    gap = imgui.get_style().item_spacing.x
    first = True
    for row in status:
        if row.key not in HOME_STATUS:
            continue
        width = (
            imgui.calc_text_size(row.icon).x
            + imgui.calc_text_size(row.text).x
            + gap * 3
        )
        if not first:
            widgets.same_line_or_wrap(width)
        first = False
        widgets.text_colored(row.colour, row.icon)
        imgui.same_line()
        if row.target:
            # A button with no fill until it is pointed at: this is a *line*,
            # and three filled pills across the top of Home is three calls to
            # action for "everything checks out". The ghost register, at
            # small_button height so it sits on the line rather than raising it.
            imgui.push_style_color(
                imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(row.colour))
            )
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
            )
            imgui.push_style_color(
                imgui.Col_.button_hovered.value,
                imgui.ImVec4(*theme.rgba(theme.ELEV_2)),
            )
            clicked = imgui.small_button(f"{row.text}##landing-status-{row.key}")
            imgui.pop_style_color(3)
            if imgui.is_item_hovered():
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
            if clicked:
                set_mode(ctx.state, row.target)
        else:
            widgets.text_colored(row.colour, row.text)


def _resume(ctx: Any) -> None:
    """What you were working on, as pictures.

    A grid rather than a list, and the dominant thing on the screen rather than
    half of one column. Recognising your own work is a visual act: the row that
    said ``chest_of_drawers_v3.wmap - 2h ago`` was a filename and a clock, and
    the picture that would have identified it in one glance was sitting in the
    job directory the whole time.
    """
    widgets.section("resume")
    drawn = rows(ctx)
    if not drawn:
        widgets.muted("Nothing yet. Start something above.")
        return
    style = imgui.get_style()
    pad = style.window_padding
    gap = style.item_spacing.x
    thumb = sp(RESUME_THUMB)
    cell_w = thumb + pad.x * 2
    cell_h = thumb + pad.y * 2 + imgui.get_text_line_height_with_spacing() * 2
    avail = imgui.get_content_region_avail().x
    columns = max(int((avail + gap) // (cell_w + gap)), 1)
    focus = ctx.state.home_index % len(drawn)
    for index, row in enumerate(drawn):
        if index % columns:
            imgui.same_line()
        _resume_cell(
            ctx, row, index, (cell_w, cell_h), thumb, pad, focused=index == focus
        )


def _card_margin(pad: Any) -> None:
    """Open a gutter inside a ``widgets.card``.

    A card is a *borderless* child as far as imgui is concerned -- it draws its
    own fill and its own border on the parent's list -- so it gets zero window
    padding, and everything in one sat flush against its left edge. Indented
    rather than pushed as a style var, because the push would have to reach
    inside ``widgets.card``'s context manager to land before ``begin_child``.

    The caller unindents; the pair is deliberately not a context manager,
    because the cards here already sit inside one.
    """
    imgui.dummy((0, pad.y - imgui.get_style().item_spacing.y))
    imgui.indent(pad.x)


def _fit(text: str, width: float) -> str:
    """``text``, trimmed with a trailing dash until it fits ``width`` px.

    Measured rather than counted. A character budget is what the Resume list
    used, and it is wrong at every scale but the one it was fitted at: 22
    characters fit a 136 dp cell at 1.0 and run a third of the way past it at
    1.5, where the card has no scrollbar and simply cuts them off.
    """
    if imgui.calc_text_size(text).x <= width:
        return text
    trimmed = text
    while trimmed and imgui.calc_text_size(trimmed + "-").x > width:
        trimmed = trimmed[:-1]
    return trimmed + "-"


def _resume_cell(
    ctx: Any,
    row: Row,
    index: int,
    size: tuple[float, float],
    thumb: float,
    pad: Any,
    *,
    focused: bool,
) -> None:
    """One cell: the picture, the name, and how long ago.

    The whole card is the affordance, as a library card's is -- the click test
    follows ``end_child`` because the child window *is* the item, which is what
    makes the hover lift work there too.

    ``focused`` is drawn as a ring rather than as a fill, because the fill is
    already spoken for by hover: a keyboard cursor that looked exactly like the
    pointer being somewhere else is a cursor you cannot find.
    """
    imgui.push_id(f"landing-cell-{row.kind}-{row.key}")
    origin = imgui.get_cursor_screen_pos()
    with widgets.card(f"landing/cell/{index}", size) as visible:
        if visible:
            _card_margin(pad)
            job = ctx.cache.get(row.key) if row.kind == "asset" else None
            if job is not None:
                thumbs.job_thumb(ctx, job, thumb)
            else:
                # A document has no rendered picture yet -- the mode glyph in a
                # frame is what a library card draws for a job with no
                # thumbnail, and the same placeholder here says "this is a
                # drawing" rather than "this is broken".
                widgets.thumb_placeholder(thumb, row.icon)
            name = _fit(row.name, thumb)
            imgui.text(name)
            if name != row.name and imgui.is_item_hovered():
                imgui.set_tooltip(row.name)
            stamp = ago(row.when)
            if stamp:
                with fonts.small(imgui):
                    widgets.muted(stamp)
            imgui.unindent(pad.x)
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
        # Hovering moves the keyboard cursor too, so the two never disagree
        # about which cell Enter would take.
        ctx.state.home_index = index
        if row.kind != "asset":
            imgui.set_tooltip(row.key)
    if imgui.is_item_clicked():
        open_row(ctx, row)
    if focused:
        # ImVec2 and not a tuple: ``ring`` reads ``.x``/``.y``, which is the
        # shape every other caller hands it (a rect min/max straight off imgui).
        widgets.ring(
            imgui.ImVec2(origin.x, origin.y),
            imgui.ImVec2(origin.x + size[0], origin.y + size[1]),
            theme.ACCENT,
            1.0,
            2.0,
        )
    imgui.pop_id()


# --- the actions ------------------------------------------------------------


def start_2d(ctx: Any) -> None:
    """A clean 2D form, wearing the active profile.

    ``default_form_2d`` rolls its own seed, so this is genuinely a fresh start
    rather than last session's form with the prompt cleared.
    """
    ctx.state.form_2d = profiles.apply(default_form_2d(), profiles.active_fields(ctx.settings))
    ctx.state.select(None)
    set_mode(ctx.state, "2d")


def start_3d(ctx: Any) -> None:
    ctx.state.form_3d = dict(DEFAULT_FORM_3D)
    ctx.state.select(None)
    set_mode(ctx.state, "3d")


def start_inker(ctx: Any) -> None:
    """Inker keeps whatever was open: unlike the two generate panes, there is
    no "fresh form" here -- the documents *are* the work, and its own empty
    state offers the canvas sizes."""
    set_mode(ctx.state, "inker")


def start_clay(ctx: Any) -> None:
    """Clay keeps whatever was open, as Inker does.

    The one addition is an empty Clay: the button says "new model", so arriving
    at a mode with nothing in it and no obvious way to begin is a dead end. A
    document is minted only when there are none -- opening one over existing
    work would break the keeps-whatever-was-open contract.
    """
    from .. import clay_mode

    set_mode(ctx.state, "clay")
    if not clay_mode.ensure(ctx).docs:
        clay_mode.new_document(ctx)


def start_plotter(ctx: Any) -> None:
    from .. import plotter_mode

    set_mode(ctx.state, "plotter")
    if not plotter_mode.ensure(ctx).docs:
        plotter_mode.new_document(ctx)


def start_packwright(ctx: Any) -> None:
    from .. import packwright_mode

    set_mode(ctx.state, "packwright")
    if not packwright_mode.ensure(ctx).docs:
        packwright_mode.new_document(ctx)


#: The six things this app can start from nothing, in the order the menu offers
#: them. "New 3D model" and "New Model" used to sit side by side and the second
#: one meant *Clay* -- two buttons whose labels differ by a word neither of them
#: defines, one of which generates a mesh from a prompt and the other opens a
#: modelling workspace (UX-23). Named for what they do instead: the mode is the
#: noun a user can act on.
#:
#: Below the functions rather than above them, because it names them: wave 5
#: retargets the first two entries at ``create_stages.go`` and nothing else on
#: this pane has to know.
NEW_ITEMS: tuple[tuple[str, str, str, object], ...] = (
    ("2d", "New 2D image", icons.IMAGE, start_2d),
    ("3d", "New 3D model", icons.BOX, start_3d),
    ("inker", "New drawing", icons.PEN_TOOL, start_inker),
    ("clay", "New Clay model", icons.RULER, start_clay),
    ("plotter", "New tile map", icons.GRID, start_plotter),
    ("packwright", "New sprite atlas", icons.LAYERS, start_packwright),
)
