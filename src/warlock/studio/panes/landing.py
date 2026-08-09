"""The Home screen: what changed, what the machine is doing, what you were on.

It used to be a grid of nine tiles, seven of which were the mode switch again
-- the same names under the same glyphs, one click either way -- while the
switch itself is drawn unconditionally at the top of every frame. So Home spent
its whole screen re-offering navigation that is permanently on screen, and the
two tiles that were *not* modes (Open Existing, Profiles) were real destinations
wearing a tile because there was nowhere else to put them. They are modes now,
and the grid is gone.

What is left is the three questions nothing in the app answered: what changed,
what is the machine doing, and what was I working on. Two columns -- news and
status on one side, resume on the other.

Nothing here is persisted: ``AppState.mode`` defaults to ``"home"``, which is
what makes this appear on every launch rather than only the first ever.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import changelog
from .. import fonts, icons, layout, modes, profiles, recents, theme, tokens, widgets
from ..manual import render as manual_render
from ..state import DEFAULT_FORM_3D, default_form_2d, format_bytes
from ..tokens import sp

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
        ctx.state.mode = "2d" if stage in ("reference", "tile") else "3d"
        return
    path = Path(row.key)
    if not path.exists():
        # A recent list that keeps offering a moved file is worse than a short
        # one, and a click that silently does nothing is worse than either.
        recents.forget(ctx.settings, row.kind, row.key)
        ctx.toast(f"{path.name} is no longer there.", "warning")
        return
    opener = {
        "inker": lambda: _open_with("inker_mode", ctx, path),
        "clay": lambda: _open_with("clay_mode", ctx, path),
        "plotter": lambda: _open_with("plotter_mode", ctx, path),
        "packwright": lambda: _open_with("packwright_mode", ctx, path),
    }.get(row.kind)
    if opener is None:
        return
    ctx.state.mode = row.kind
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


def draw(ctx: Any) -> None:
    pump(ctx)
    _header(ctx)

    avail = imgui.get_content_region_avail()
    spacing = imgui.get_style().item_spacing.x
    column = max((avail.x - spacing) * 0.5, sp(240))
    height = max(avail.y - sp(tokens.SP_2), sp(200))

    if layout.pane_child("landing/news", (column, height)):
        _news(ctx)
        _status(ctx)
    imgui.end_child()
    imgui.same_line()
    if layout.pane_child("landing/resume", (0, height)):
        _resume(ctx)
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


def _version() -> str:
    from ..main import _version as installed

    return installed()


def _news(ctx: Any) -> None:
    widgets.section("what's new")
    releases = changelog.entries()
    if not releases:
        widgets.muted("No changelog shipped with this build.")
        imgui.dummy((0, sp(tokens.SP_4)))
        return
    # The current release open and the rest collapsed: "what changed" is about
    # this build, and the older ones are there so the answer has a context
    # rather than so it has a scrollback.
    current = changelog.current(_version())
    # The scroller takes everything the status block below it does not, measured
    # off the rows that block will actually draw rather than a reserved literal:
    # a fixed reservation is a gap on an install with nothing wrong and a
    # clipped row on one with four things wrong, which is the wrong way round.
    reserved = _status_height(ctx)
    inner = max(imgui.get_content_region_avail().y - reserved, sp(120))
    if imgui.begin_child("landing/news-scroll", (0, inner)):
        for index, release in enumerate(releases):
            headline = release.version + (f" - {release.date}" if release.date else "")
            opened = current is not None and release.version == current.version
            flags = imgui.TreeNodeFlags_.default_open.value if opened else 0
            if imgui.collapsing_header(f"{headline}##news{index}", flags):
                for bullet in release.bullets:
                    with fonts.small(imgui):
                        widgets.muted_wrapped(f"- {bullet}")
    imgui.end_child()
    imgui.dummy((0, sp(tokens.SP_3)))


def _status_height(ctx: Any) -> float:
    """What :func:`_status` will occupy, so the scroller above it can take the
    rest. A clickable row is a frame tall and a plain one is a line."""
    style = imgui.get_style()
    rows = status_rows(ctx)
    line = imgui.get_frame_height_with_spacing()
    return line * (len(rows) + 1) + style.item_spacing.y + sp(tokens.SP_3)


def _status(ctx: Any) -> None:
    widgets.section("status")
    for row in status_rows(ctx):
        widgets.text_colored(row.colour, row.icon)
        imgui.same_line()
        if row.target:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(row.colour)))
            clicked = imgui.small_button(f"{row.text}##landing-status-{row.key}")
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
            if clicked:
                ctx.state.mode = row.target
        else:
            widgets.text_colored(row.colour, row.text)


def _resume(ctx: Any) -> None:
    widgets.section("start something")
    _actions(ctx)
    imgui.dummy((0, sp(tokens.SP_3)))
    widgets.section("resume")
    drawn = rows(ctx)
    if not drawn:
        widgets.muted("Nothing yet. Start something above.")
        return
    focus = ctx.state.home_index % len(drawn)
    for index, row in enumerate(drawn):
        _resume_row(ctx, row, index, focused=index == focus)


def _resume_row(ctx: Any, row: Row, index: int, *, focused: bool) -> None:
    """One row: mode glyph, name, and how long ago.

    Everything the row *is* goes in the selectable's own label, and the
    timestamp is drawn beside it against a width the selectable was told to
    leave. Painting the text over the selectable with ``set_cursor_screen_pos``
    is the tempting version and it is wrong: imgui grows a window's content
    bounds from where the cursor has been, so putting it back afterwards makes
    the pane's scroll extent disagree with what is in it, and imgui says so.

    The key is in the imgui id, not just the label: two documents can share a
    basename and one imgui id between them is one row. ``focused`` rides the
    selectable's own selected state rather than a ring drawn around it, so the
    keyboard cursor is the same highlight the mouse gives.
    """
    stamp = ago(row.when)
    avail = imgui.get_content_region_avail().x
    with fonts.small(imgui):
        stamp_w = imgui.calc_text_size(stamp).x if stamp else 0.0
    gap = imgui.get_style().item_spacing.x
    width = max(avail - (stamp_w + gap if stamp else 0.0), sp(120))
    name = row.name if len(row.name) <= 34 else row.name[:33] + "-"
    label = f"{row.icon}  {name}##landing-row-{row.kind}-{row.key}"
    if imgui.selectable(label, focused, 0, (width, 0))[0]:
        open_row(ctx, row)
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
        # Hovering moves the keyboard cursor too, so the two never disagree
        # about which row Enter would take.
        ctx.state.home_index = index
        if row.kind != "asset":
            imgui.set_tooltip(row.key)
    if stamp:
        imgui.same_line()
        with fonts.small(imgui):
            widgets.muted(stamp)


def _actions(ctx: Any) -> None:
    """The "New ..." row. One per thing this app can start from nothing."""
    buttons: tuple[tuple[str, str, Any], ...] = (
        ("2D image", icons.IMAGE, start_2d),
        ("3D model", icons.BOX, start_3d),
        ("Drawing", icons.PEN_TOOL, start_inker),
        ("Model", icons.RULER, start_clay),
        ("Map", icons.GRID, start_plotter),
        ("Atlas", icons.LAYERS, start_packwright),
    )
    width = widgets.grid_width(3)
    for index, (label, icon, action) in enumerate(buttons):
        if index % 3:
            imgui.same_line()
        if imgui.button(f"{icon}  New {label}##landing-new-{index}", (width, 0)):
            action(ctx)


# --- the actions ------------------------------------------------------------


def start_2d(ctx: Any) -> None:
    """A clean 2D form, wearing the active profile.

    ``default_form_2d`` rolls its own seed, so this is genuinely a fresh start
    rather than last session's form with the prompt cleared.
    """
    ctx.state.form_2d = profiles.apply(default_form_2d(), profiles.active_fields(ctx.settings))
    ctx.state.select(None)
    ctx.state.mode = "2d"


def start_3d(ctx: Any) -> None:
    ctx.state.form_3d = dict(DEFAULT_FORM_3D)
    ctx.state.select(None)
    ctx.state.mode = "3d"


def start_inker(ctx: Any) -> None:
    """Inker keeps whatever was open: unlike the two generate panes, there is
    no "fresh form" here -- the documents *are* the work, and its own empty
    state offers the canvas sizes."""
    ctx.state.mode = "inker"


def start_clay(ctx: Any) -> None:
    """Clay keeps whatever was open, as Inker does.

    The one addition is an empty Clay: the button says "new model", so arriving
    at a mode with nothing in it and no obvious way to begin is a dead end. A
    document is minted only when there are none -- opening one over existing
    work would break the keeps-whatever-was-open contract.
    """
    from .. import clay_mode

    ctx.state.mode = "clay"
    if not clay_mode.ensure(ctx).docs:
        clay_mode.new_document(ctx)


def start_plotter(ctx: Any) -> None:
    from .. import plotter_mode

    ctx.state.mode = "plotter"
    if not plotter_mode.ensure(ctx).docs:
        plotter_mode.new_document(ctx)


def start_packwright(ctx: Any) -> None:
    from .. import packwright_mode

    ctx.state.mode = "packwright"
    if not packwright_mode.ensure(ctx).docs:
        packwright_mode.new_document(ctx)
