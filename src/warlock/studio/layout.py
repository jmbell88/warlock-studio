"""Measurements for Warlock's bounded, three-column editor shell.

Side panels tile around a fixed centre canvas: they may be resized, reordered
inside their column, or hidden, but never float, tab together, or cross through
the canvas. Version-2 named layouts retain each workspace's desired left and
right widths plus its internal vertical shares. Narrow windows compress those
widths for the current frame without changing the saved preference.

Sizes are design pixels and are multiplied by :data:`tokens.SCALE` at use, so
a saved width has the same meaning on every display scale.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from imgui_bundle import imgui

from . import guard, motion, theme, tokens
from .settings import as_dict
from .tokens import sp

# Legacy global width presets remain as fallback seeds for untouched v1
# workspaces. Explicit splitter edits are stored per workspace in layout v2.
# Defined in ``tokens`` and named here, where the eight readers already look:
# ``layouts`` needs the same numbers and cannot import this module.
SIDEBAR_WIDTHS = tokens.SIDEBAR_WIDTHS

# The width in force. Module state, set by ``Layout`` at construction and when
# the option changes, exactly as ``tokens.SCALE`` is -- eight call sites read
# this directly and threading a Layout through all of them would put the
# measurement in eight places instead of one.
SIDEBAR_W = SIDEBAR_WIDTHS["default"]
# What ``SIDEBAR_W`` is on its way to. The two are the same number except
# during the ~200 ms after the user picks a different size; see :func:`tick`.
SIDEBAR_TARGET = SIDEBAR_W
# The motion key the width eases on, named here because ``set_sidebar`` has to
# forget it when it snaps.
_SIDEBAR_KEY = "layout/sidebar"


def set_sidebar(key: str, *, animate: bool = False) -> str:
    """Apply a named sidebar width. -> the key actually applied.

    An unknown key falls back to the default rather than raising: this is read
    from a settings file, and a value written by a build that offered a fourth
    size must not stop the window opening.

    ``animate`` is the split between the two callers and it is not cosmetic.
    The construction path (and every headless test) has to leave ``SIDEBAR_W``
    correct *immediately* -- there may be no frame loop at all -- so it snaps;
    the user changing the option in Settings goes through :func:`tick`, where
    the two columns slide to their new widths instead of jumping.
    """
    global SIDEBAR_W, SIDEBAR_TARGET
    key = key if key in SIDEBAR_WIDTHS else "default"
    SIDEBAR_TARGET = SIDEBAR_WIDTHS[key]
    if animate:
        # Stated rather than assumed: a key that is not live yet snaps on its
        # first sighting, so an animated change made before the first tick --
        # which is every change made in the same frame the option is read back
        # -- would jump. Seeding says where the slide starts.
        motion.seed(_SIDEBAR_KEY, SIDEBAR_W)
    else:
        SIDEBAR_W = SIDEBAR_TARGET
        # Forgotten rather than assigned: motion snaps a key's first sighting,
        # so dropping it is how the next tick starts settled at the new width
        # rather than easing from the old one.
        motion.forget(_SIDEBAR_KEY)
    return key


# What a centre pane wants, and what it will accept before the sidebars start
# giving way instead (UX-01), both in design px. 300 was already the floor in
# ``centre_width`` -- what it was missing is that a floor which cannot be met
# does not stop being claimed, it just pushes the column after it off the
# window. The two numbers separate "this is the comfortable size" from "this
# is the size below which shrinking the centre is the wrong answer".
CENTRE_MIN = 300.0
CENTRE_FLOOR = 220.0

# How narrow a sidebar may be squeezed. Below this a form's labels wrap to one
# word a line, which is a pane nobody can use rather than a small one -- so
# this is where squeezing stops and the centre gives instead.
SIDEBAR_MIN = 200.0

# Saved workspace widths are continuous rather than one of the legacy named
# sidebar sizes.  The values are preferences in design pixels; the fitted
# values below may temporarily be smaller when UI scale or the window makes
# the preference impossible.  That temporary fit is never written back.
PANEL_MIN = tokens.PANEL_MIN
PANEL_MAX = tokens.PANEL_MAX

# The width the two sidebars actually get this frame, in *physical* px. Module
# state, computed once by :func:`measure` for the same reason ``SIDEBAR_W`` is
# eased in ``tick``: the left column and the right must agree within a frame,
# and the right one is sized ``(0, h)`` from whatever the centre left behind.
#
# ``None`` until a frame has measured one. Its own sentinel rather than a
# plausible default, so that the headless callers -- tests, and anything asking
# what a sidebar is before there is a window -- get the unconstrained width
# rather than a fit computed against a viewport of zero.
SIDEBAR_FIT: float | None = None
SIDE_FIT: dict[str, float | None] = {"left": None, "right": None}

# What the navigation rail has taken out of the window this frame, in physical
# px (the UI redesign, wave 3). Module state set by ``rail.tick`` immediately before
# :func:`tick`, for the reason ``SIDEBAR_W`` is: the sidebars are measured from
# what is left after the rail, and a number threaded through would be the same
# figure in two places.
#
# Zero by default, which is what makes every headless caller -- ``fit`` is pure
# and the tests drive it directly -- see the window it always saw. A rail that
# has never been drawn has taken nothing.
RAIL_RESERVED: float = 0.0


def fit(available: float, spacing: float) -> float:
    """How wide each sidebar can be, given the room. Physical px, pure.

    The whole of UX-01 is that this used to be unconditional. Three columns
    reserved two 300-design-px sidebars and floored the centre at 300 more, so
    at 1.5x the workspace demanded ~1350 physical px and at 2x ~1800 -- while
    the resize floor follows the *monitor's* scale alone and stays near 1100.
    (Deliberately: see ``main._min_window_size``. A floor that multiplied the
    user's zoom in demanded a window bigger than a 1080p display and refused to
    shrink, which is worse.) The arithmetic simply overflowed, and since the
    right-hand column is the one sized from the leftovers, the pane that fell
    off the edge was always the inspector.

    So the columns give way in a stated order: the sidebars narrow first, down
    to :data:`SIDEBAR_MIN`, and only then does the centre drop below
    :data:`CENTRE_MIN`. Somebody who enlarges the UI to read it ends up with
    three narrow columns rather than two comfortable ones and a third they
    cannot reach.
    """
    want = sp(SIDEBAR_W)
    if available >= want * 2 + sp(CENTRE_MIN) + spacing * 2:
        return want
    # What is left for the two sidebars once the centre keeps its comfortable
    # width -- which may be negative, hence the floor rather than a clamp.
    slack = (available - sp(CENTRE_MIN) - spacing * 2) / 2.0
    return max(min(slack, want), sp(SIDEBAR_MIN))


def fit_widths(
    available: float,
    left: float,
    right: float,
    spacing: float,
    *,
    scale: float | None = None,
    fixed_left: float | None = None,
) -> tuple[float, float, float]:
    """Fit independent desired side widths without changing the preferences.

    Inputs ``left``/``right`` and ``fixed_left`` are design pixels; the result
    is physical ``(left, right, centre)`` pixels.  Side columns give way down
    to their normal 220 dp bound, then may compress further when scale makes
    even that impossible.  The centre keeps 300 dp when it can and 220 dp
    before anything is allowed to run off-screen.
    """

    use_scale = tokens.SCALE if scale is None else max(float(scale), 0.01)
    content = max(float(available) - max(float(spacing), 0.0) * 2.0, 0.0)
    centre_comfort = CENTRE_MIN * use_scale
    centre_floor = min(CENTRE_FLOOR * use_scale, content)
    left_want = (
        float(fixed_left) if fixed_left is not None else min(max(float(left), PANEL_MIN), PANEL_MAX)
    ) * use_scale
    right_want = min(max(float(right), PANEL_MIN), PANEL_MAX) * use_scale

    wanted = left_want + right_want
    if content >= wanted + centre_comfort:
        return left_want, right_want, content - wanted

    if fixed_left is None:
        normal_min = PANEL_MIN * use_scale * 2.0
    else:
        normal_min = left_want + PANEL_MIN * use_scale
    # Keep a useful centre first.  If the normal panel floors cannot coexist
    # with it, panel widths are a frame-local compression rather than a saved
    # edit.  On a physically impossible window the centre gets the remainder.
    panel_budget = max(content - centre_comfort, 0.0)
    if panel_budget < normal_min:
        panel_budget = max(content - centre_floor, 0.0)
    panel_budget = min(panel_budget, wanted)

    if fixed_left is not None:
        left_fit = min(left_want, panel_budget)
        right_fit = max(panel_budget - left_fit, 0.0)
    elif wanted > 0.0:
        left_fit = panel_budget * left_want / wanted
        right_fit = panel_budget - left_fit
    else:  # pragma: no cover - clamping above makes this defensive only
        left_fit = right_fit = 0.0
    return left_fit, right_fit, max(content - left_fit - right_fit, 0.0)


def tick() -> None:
    """Advance the sidebar width one frame. Called once, before anything reads
    ``SIDEBAR_W`` -- a half-eased width read by the left column and the settled
    one by the right would be two columns disagreeing about the same frame."""
    global SIDEBAR_W
    SIDEBAR_W = motion.value(_SIDEBAR_KEY, SIDEBAR_TARGET, duration=tokens.DUR_BASE)


def measure(library: Any = None, workspace: str = "", *, fixed_left: float | None = None) -> float:
    """Settle this frame's sidebar fit. Called once, straight after :func:`tick`.

    Separate from ``tick`` rather than folded into it, and the reason is not
    tidiness: ``tick`` is pure arithmetic over the motion table and is called
    by headless tests that have no imgui context at all, where reaching for a
    viewport is an access violation rather than an error.

    Measured from the whole window rather than from the content region a column
    happens to be standing in: ``centre_width`` runs *after* the left sidebar is
    drawn, so recomputing there would fit the right sidebar against a width the
    left has already taken its share of, and the two columns would come out
    different widths.
    """
    global SIDEBAR_FIT
    style = imgui.get_style()
    room = imgui.get_main_viewport().work_size[0] - style.window_padding.x * 2
    if RAIL_RESERVED:
        # The rail and the gap after it, both taken off before the columns are
        # fitted. Measured rather than assumed because the rail eases between
        # two widths: fitting against the settled figure would leave the three
        # columns wrong for the ~200 ms of every expand.
        room -= RAIL_RESERVED + style.item_spacing.x
    if library is None or not workspace:
        SIDEBAR_FIT = fit(room, style.item_spacing.x)
        SIDE_FIT["left"] = SIDE_FIT["right"] = SIDEBAR_FIT
        return SIDEBAR_FIT
    left = library.width(workspace, "left")
    right = library.width(workspace, "right")
    left_fit, right_fit, _centre = fit_widths(
        room, left, right, style.item_spacing.x, fixed_left=fixed_left
    )
    SIDE_FIT["left"], SIDE_FIT["right"] = left_fit, right_fit
    # The left column, not the mean of the two. ``SIDE_FIT`` is what every
    # caller actually reads (``sidebar_width`` consults it first and it is
    # always populated by the time anything draws), so this is only the
    # pre-measure fallback and the value ``measure`` hands back -- and an
    # average is a width neither column has, which is a worse thing for a
    # fallback to be than simply the side ``sidebar_width`` defaults to.
    SIDEBAR_FIT = left_fit
    return SIDEBAR_FIT


# Between a pane's border and its content. 5 was the most utilitarian number in
# the codebase: it made a pane's contents start one pixel-and-a-bit from the
# hairline around them, which is what a mixer strip wants and not what a form
# does. SP_3 rather than the window's own SP_4 (UX.md Phase 2) because a pane is
# already inside the host window's gutter -- matching it would double the inset
# on the two sidebars, which are the width-constrained case.
PANE_PADDING = tokens.SP_3
SHARE_MIN, SHARE_MAX = 0.25, 0.75

#: Splits whose sensible starting proportion is not the shared default.
#:
#: One entry, and it is the timeline strip: ``settings_share`` is a *sidebar*
#: proportion, and handing it to a horizontal split along the bottom of the
#: canvas would open every Inker document with half its centre column full of
#: layer rows. Per key rather than per workspace, for the reason
#: ``_SHARE_SPLITS`` exists at all -- and a stored value still wins, so this
#: moves the starting point and nobody's drag.
SHARE_DEFAULTS: dict[str, float] = {"inker-timeline": 0.28}


def give_way(avail: float, share: float, wanted: float, floor: float) -> float:
    """Height for the upper of two stacked panes. -> design px.

    **The share is a preference, not a promise.** A pane whose content has a
    known minimum -- a fixed grid of buttons, plus enough of the form under it
    to be worth scrolling -- must be allowed to overrule a share that would cut
    that minimum in half, and must still yield before it starves the pane below
    it. That is the sidebar/centre give-way rule of INVARIANTS.md read down a
    column instead of across a row, and it is written here rather than in the
    workspace so it can be asserted without a window.

    ``wanted`` and ``floor`` are the upper pane's minimum and the lower pane's,
    both in the same units as ``avail``. When the two cannot both be met the
    lower pane's floor wins, because the alternative is a pane with a heading
    and nothing under it.
    """
    if avail <= 0:
        return 0.0
    ceiling = max(avail - floor, avail * SHARE_MIN)
    return min(max(avail * share, min(wanted, ceiling)), ceiling)


def give_way_drag(avail: float, share: float, wanted: float, floor: float, delta: float) -> float:
    """The share after dragging the handle under a :func:`give_way` pane.

    A drag measured against the *stored* share goes dead the moment give_way
    pins the pane: ``min(wanted, ceiling)`` holds the height still while the
    share keeps sliding, so the handle under the cursor moved nothing -- and
    every other pane keyed to the same share moved instead, which read as a
    broken handle dragging a pane across the window. So the delta is applied
    to the height the pane is actually drawn at, clamped to the heights this
    pane can really take, and the share re-derived from the result; a drag
    the pane cannot follow leaves the share exactly where it was.
    """
    if avail <= 0:
        return share
    at = give_way(avail, share, wanted, floor)
    lo = give_way(avail, SHARE_MIN, wanted, floor)
    hi = give_way(avail, SHARE_MAX, wanted, floor)
    target = min(max(at + delta, lo), hi)
    if target == at:
        return share
    return min(max(target / avail, SHARE_MIN), SHARE_MAX)


GRIP = 7.0  # hit-zone width in design px; the drawn line is 1px


class Layout:
    """The one pane proportion the user still controls."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._workspace_library: Any = None
        self._workspace = ""
        stored = as_dict(settings.get("layout"))
        try:
            share = float(stored.get("settings_share", 0.55))
        except (TypeError, ValueError):
            share = 0.55
        self.settings_share = min(max(share, SHARE_MIN), SHARE_MAX)
        # **One share per split, not one for the whole app.** ``settings_share``
        # above is a single number that eight workspaces read and one -- Inker
        # -- can write, from two handles. So dragging Inker's toolbox handle
        # silently re-split Create, Clay, Plotter, Packwright, Troupe and
        # Review, and Inker's own right column besides: a handle that resizes
        # six screens the user cannot see is not a handle, it is a setting.
        #
        # Keyed now, and the scalar above is what an unkeyed split *starts*
        # at, so an existing profile opens on exactly the layout it had and
        # each split then goes its own way from there. Written under a separate
        # settings key for the same reason: an old build reading a new file
        # still finds ``settings_share`` where it left it.
        self.shares: dict[str, float] = {}
        for key, value in (stored.get("settings_shares") or {}).items():
            try:
                self.shares[str(key)] = min(max(float(value), SHARE_MIN), SHARE_MAX)
            except (TypeError, ValueError):
                continue
        self._migrate_shares()
        self.sidebar = set_sidebar(str(stored.get("sidebar", "default")))
        # Whether the navigation rail shows its labels. A *name* rather than a
        # width, exactly as ``sidebar`` is, so a stored value can never be a
        # size this build does not offer -- and anything unrecognised reads as
        # "icons", which is the state the rail is designed around rather than a
        # degraded version of the other one.
        # A profile that has never expressed a preference gets *icons*, and this
        # is the one place that decision reversed. The old argument was that ten
        # workspaces are a glyph with no name until you hover one; what answers
        # it now is that every rail row has a tooltip and the editor shell put
        # the mode names in the Window menu as well, so a first-time user has
        # two ways to read the column without spending 188 px of a 44 px rail on
        # it permanently. A *stored* "labels" still wins -- this moves the
        # default, not anyone's choice.
        stored_rail = stored.get("rail")
        self.rail = "labels" if stored_rail == "labels" else "icons"

    # A workspace that stacks two panes on the left *and* two on the right is
    # two splits, and until now both read one key -- so dragging Clay's tools
    # handle also re-split its outliner column, one pane the user was not
    # looking at. The keying above already supported a key per split; the
    # callers simply passed the same string twice.
    #
    # Seeded rather than dropped: an existing profile opens on exactly the
    # proportion it had, on both of the splits the old key served, and they go
    # their own way from there. The old key is then *deleted*, because
    # :meth:`save` writes ``self.shares`` wholesale -- left in place it would
    # be re-seeded from on every launch for ever, and a later build that
    # reused the name would inherit a value from a split that no longer
    # exists.
    _SHARE_SPLITS: dict[str, tuple[str, ...]] = {
        "clay": ("clay-tools", "clay-outliner"),
        "plotter": ("plotter-tools", "plotter-layers"),
        "troupe": ("troupe-cast", "troupe-sheets"),
        "packwright": ("packwright-sources", "packwright-items"),
        "sirens": ("sirens-transport", "sirens-instruments"),
        "review": ("review-runs",),
        "create": ("create-inspector",),
    }

    def _migrate_shares(self) -> None:
        """Split each retired one-per-workspace key across the splits it served."""
        for old, new_keys in self._SHARE_SPLITS.items():
            if old not in self.shares:
                continue
            value = self.shares.pop(old)
            for key in new_keys:
                self.shares.setdefault(key, value)

    def share(self, key: str) -> float:
        """The split ``key`` is drawn at -- its own, or the shared default.

        ``key`` names a *split*, not a mode: Inker has two independent handles
        and they are two entries, because one value behind both is the same
        defect one rung down.
        """
        if self._workspace_library is not None and self._workspace:
            return min(
                max(
                    self._workspace_library.share(
                        self._workspace, key, self.shares.get(key, self._default_share(key))
                    ),
                    SHARE_MIN,
                ),
                SHARE_MAX,
            )
        return self.shares.get(key, self._default_share(key))

    def _default_share(self, key: str) -> float:
        return SHARE_DEFAULTS.get(key, self.settings_share)

    def set_share(self, key: str, value: float) -> None:
        """Move one split. Clamped here so no caller has to remember to."""
        value = min(max(value, SHARE_MIN), SHARE_MAX)
        if self._workspace_library is not None and self._workspace:
            self._workspace_library.set_share(self._workspace, key, value)
            return
        self.shares[key] = value

    def bind_workspace(self, library: Any, workspace: str) -> None:
        """Route vertical split reads/writes through the active arrangement."""

        self._workspace_library = library
        self._workspace = str(workspace)

    def set_sidebar_width(self, key: str) -> None:
        """Adopt a named side-column width, everywhere.

        **Write-through, or the control is decorative.** ``SIDEBAR_W`` is what
        this used to move and nothing in the running app reads it: every
        workspace is measured by :func:`measure` through
        ``layouts.Library.width``, which consults a stored per-workspace width
        first and a seed taken at *construction* second. A width picked here
        reached neither, so the combo in Settings did nothing at all.

        The named widths are a global preference and a splitter drag is a local
        override, so picking one here replaces the overrides -- which is what
        "set the sidebar width" means, and the drag is available again the
        moment the user wants a different answer for one workspace.
        """
        self.sidebar = set_sidebar(key, animate=True)
        if self._workspace_library is not None:
            self._workspace_library.set_width_seed(SIDEBAR_WIDTHS[self.sidebar])
        self.save()

    def set_rail(self, key: str) -> None:
        self.rail = "labels" if key == "labels" else "icons"
        self.save()

    def reset_sizes(self) -> None:
        """Every split and side width back to the built-in proportions.

        Sizes only. ``columns`` and ``hidden`` are the pane *arrangement*,
        which is a different button, so this is deliberately not
        ``layouts.Library.reset``.

        Clearing ``self.shares`` alone -- which is what "Reset pane sizes" did
        -- reset nothing a user could see: :meth:`share` consults the bound
        library first, and every drag persists there, so the only splits the
        legacy dict still governed were the ones nobody had ever moved.
        """
        self.settings_share = 0.55
        self.shares.clear()
        if self._workspace_library is not None:
            self._workspace_library.reset_sizes()
        self.save()

    def save(self) -> None:
        # **No early return here, whatever else has already persisted.** This
        # is the only writer of ``rail`` and ``sidebar``, and it used to skip
        # the whole write whenever ``layouts.Library`` had just taken a share
        # edit -- so a rail toggle or a sidebar width chosen after any splitter
        # drag was silently discarded, exactly as the paragraph below warns.
        # Rewriting the legacy share blob unchanged in that case costs nothing:
        # it is a migration seed, not a source of truth.
        #
        # Only the surviving keys: Settings.set replaces the whole dict, so the
        # stale sidebar_w/inspector_w a settings file may still carry are gone
        # the first time anything saves. ``sidebar`` is the *name* of a width,
        # never a number, so a stored value can never be a size this build does
        # not offer -- and ``rail`` follows it for the same reason. Both have to
        # be written here every time, because the whole dict is replaced: a key
        # this method forgets is a preference that silently resets the next time
        # the *other* one changes.
        self._settings.set(
            "layout",
            {
                "settings_share": round(self.settings_share, 3),
                "settings_shares": {
                    key: round(value, 3) for key, value in sorted(self.shares.items())
                },
                "sidebar": self.sidebar,
                "rail": self.rail,
            },
        )


class PaneRole(StrEnum):
    """Semantic surface role for a major region of a workspace."""

    SIDEBAR = "sidebar"
    CONTENT = "content"
    PRIMARY = "content"  # the plan's prose name; CONTENT reads better at calls
    INSPECTOR = "inspector"
    SHEET = "sheet"
    OVERLAY = "overlay"


class PaneEdge(StrEnum):
    """The single major boundary a pane owns, if any."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


def _pane_fill(role: PaneRole) -> int:
    if role in (PaneRole.SIDEBAR, PaneRole.INSPECTOR, PaneRole.SHEET):
        return theme.PANEL
    if role is PaneRole.OVERLAY:
        return theme.ELEV_2
    return theme.BG


def _divider(edge: PaneEdge) -> None:
    if edge is PaneEdge.NONE:
        return
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    # Panes round (``tokens.RADIUS_PANE``), so the fill retreats from each
    # corner and a hairline drawn across the child's *full* rect overshoots
    # into the window colour at both ends. Both endpoints are pulled in by the
    # rounding actually in force -- read off the live style rather than the
    # token, because this runs from ``pane``'s finally, where the style is
    # whatever the caller left standing (``widgets.card`` pushes its own).
    inset = float(getattr(imgui.get_style(), "child_rounding", 0.0))
    if edge is PaneEdge.LEFT:
        start, end = (low.x, low.y + inset), (low.x, high.y - inset)
    elif edge is PaneEdge.RIGHT:
        start, end = (high.x, low.y + inset), (high.x, high.y - inset)
    elif edge is PaneEdge.TOP:
        start, end = (low.x + inset, low.y), (high.x - inset, low.y)
    else:
        start, end = (low.x + inset, high.y), (high.x - inset, high.y)
    imgui.get_window_draw_list().add_line(
        start,
        end,
        imgui.get_color_u32(theme.rgba(theme.DIVIDER)),
        sp(tokens.DIVIDER_WIDTH),
    )


@contextmanager
def pane(
    pane_id: str,
    size: tuple[float, float],
    role: PaneRole | str = PaneRole.CONTENT,
    *,
    edge: PaneEdge | str = PaneEdge.NONE,
    window_flags: int = 0,
    title: str = "",
):
    """Begin a role-aware major pane and close it reliably.

    ``title`` is what the user is told stopped drawing when this pane raises;
    ``column`` passes ``slot.label``, so the twenty panes drawn through the
    layout get real names from the one table rather than a second label map.

    Child borders stay disabled.  A caller names at most the one edge that
    separates this pane from another major region, which avoids turning every
    stacked child into a bordered card.

    ``always_use_window_padding`` goes with that borderlessness rather than
    beside it: Dear ImGui zeroes a *borderless* child's window padding, so the
    ``PANE_PADDING`` pushed two lines below had no effect at all and every
    pane's content sat flush against the column edge. Two panes had already
    worked around it locally (``panes/app_settings``, ``panes/landing``) --
    those two draw into ``##content`` rather than through here, so they keep
    their own flag.
    """

    try:
        resolved_role = role if isinstance(role, PaneRole) else PaneRole(role)
    except ValueError:
        resolved_role = PaneRole.CONTENT
    try:
        resolved_edge = edge if isinstance(edge, PaneEdge) else PaneEdge(edge)
    except ValueError:
        resolved_edge = PaneEdge.NONE
    pad = sp(PANE_PADDING)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    imgui.push_style_color(
        imgui.Col_.child_bg.value, imgui.ImVec4(*theme.rgba(_pane_fill(resolved_role)))
    )
    visible = imgui.begin_child(
        pane_id, size, imgui.ChildFlags_.always_use_window_padding.value, window_flags
    )
    imgui.pop_style_color()
    imgui.pop_style_var()
    # The recovery mark is taken *after* ``begin_child``, and that is the one
    # thing here it would be fatal to get backwards: the stored window-stack
    # size then includes this child, so an unwind closes everything deeper and
    # stops exactly on us, leaving the ``end_child`` below to fire once as it
    # always has. Marking before ``begin_child`` would have the unwind close
    # this child too, and that ``end_child`` would then be the unbalanced call
    # the whole mechanism exists to prevent.
    mark = guard.enter(pane_id)
    live = visible and not guard.tripped(pane_id)
    failed = False
    try:
        # Editor panes stay flat: headers and separators establish hierarchy.
        # Card/block surfaces remain opt-in for Home, dialogs and overlays.
        yield live
    except Exception as exc:  # noqa: BLE001 -- see ``studio/guard.py``
        # ``recover`` re-raises rather than returning for every case where
        # carrying on would be a lie, and those land in ``App.run``'s handler
        # exactly as they do today -- journal first, then the crash report.
        guard.recover(mark, pane_id, title or pane_id, exc)
        failed = True
    else:
        if live:
            guard.ok(pane_id)
    finally:
        # Skipped on the escalation path by construction: ``failed`` is only set
        # once ``recover`` has returned, and a re-raise never records a failure,
        # so a pane on its way to the crash dialog draws nothing extra.
        if visible and (failed or guard.tripped(pane_id)):
            guard.placeholder(pane_id, title or pane_id)
        imgui.end_child()
        _divider(resolved_edge)


FILL_SIZING = "fill"
"""``layout_skeleton.FILL``, spelled here so this module imports it lazily."""


def pane_child(pane_id: str, size: tuple[float, float], window_flags: int = 0) -> bool:
    """Compatibility entry point for third-party panes and older tests.

    Studio code uses :func:`pane`; this retains the old begin/end shape for
    callers outside the migration boundary. It is intentionally borderless --
    and carries ``always_use_window_padding`` for that reason, exactly as
    :func:`pane` does.
    """

    pad = sp(PANE_PADDING)
    imgui.push_style_var(imgui.StyleVar_.window_padding.value, (pad, pad))
    visible = imgui.begin_child(
        pane_id, size, imgui.ChildFlags_.always_use_window_padding.value, window_flags
    )
    imgui.pop_style_var()
    return visible


def sidebar_width(side: str = "left") -> float:
    """A sidebar's width this frame, in physical px. Narrowed to fit (UX-01).

    What the eight workspaces call instead of ``sp(SIDEBAR_W)``. Same value
    whenever there is room, which is the ordinary case; the difference only
    shows at high UI scale in a small window, which is exactly the case a user
    who enlarged the UI to read it is in.
    """
    fitted = SIDE_FIT.get(side)
    if fitted is not None:
        return fitted
    return sp(SIDEBAR_W) if SIDEBAR_FIT is None else SIDEBAR_FIT


def centre_width() -> float:
    """What is left for the centre pane once the right sidebar is reserved.

    Called with the cursor between the left sidebar and the centre, so the
    available width still contains the centre, one item spacing and the right
    sidebar. The right sidebar itself is then sized ``(0, h)`` and resolves to
    :data:`SIDEBAR_FIT` because this reserved exactly that.

    The floor is :data:`CENTRE_FLOOR` and not ``CENTRE_MIN``, which is the
    other half of UX-01: a floor is a claim, and a claim that cannot be met
    does not fail -- it silently pushes the *next* column past the window edge.
    Reserving the sidebar's real width and flooring lower is what keeps the
    right-hand pane on screen when the arithmetic is tight.
    """
    spacing = imgui.get_style().item_spacing.x
    return max(
        imgui.get_content_region_avail().x - (sidebar_width("right") + sp(GRIP) + spacing * 2.0),
        sp(CENTRE_FLOOR),
    )


def resize_side(library: Any, workspace: str, side: str, delta: float) -> bool:
    """Apply a column-boundary drag in design pixels.  Returns whether saved."""

    if side not in ("left", "right") or not delta:
        return False
    before = library.width(workspace, side)
    direction = 1.0 if side == "left" else -1.0
    after = min(max(before + direction * float(delta), PANEL_MIN), PANEL_MAX)
    if after == before:
        return False
    library.set_width(workspace, side, after)
    return True


def column_splitter(library: Any, workspace: str, side: str, *, length: float = 0.0) -> None:
    """Draw and persist one of the two bounded side-column splitters."""

    delta = splitter(f"{workspace}-{side}-width", vertical=True, length=length)
    if delta:
        resize_side(library, workspace, side, delta)


def splitter(split_id: str, *, vertical: bool = True, length: float = 0.0) -> float:
    """A drag handle between two panes. -> this frame's drag delta in design px.

    ``vertical`` is the *bar's* orientation: a vertical bar sits between
    columns and drags horizontally. The bar is an ``invisible_button`` with a
    1px line drawn down its middle, accent-coloured while hovered or dragged.
    """
    if _EDITING:
        # **Suppressed while the layout editor is open** (P5.4): a resize
        # handle and a drag target on the same two pixels is a gesture nobody
        # can aim. Sizes are dragged in normal mode, where the handles are the
        # only thing there.
        return 0.0
    grip = sp(GRIP)
    if length <= 0:
        avail = imgui.get_content_region_avail()
        length = avail.y if vertical else avail.x
        length = max(length, 1.0)
    size = (grip, length) if vertical else (length, grip)
    pos = imgui.get_cursor_screen_pos()
    imgui.invisible_button(f"##split/{split_id}", size)
    active = imgui.is_item_active()
    hovered = imgui.is_item_hovered()
    if hovered or active:
        imgui.set_mouse_cursor(
            (imgui.MouseCursor_.resize_ew if vertical else imgui.MouseCursor_.resize_ns).value
        )
    # Faded rather than switched: a hairline that snaps from edge-grey to full
    # accent as the pointer crosses it is the loudest hard cut left in the
    # workspace, and it fires on every pass over the column boundary whether or
    # not anybody meant to drag.
    lit = motion.value(
        f"splitter/{split_id}", 1.0 if (hovered or active) else 0.0, duration=tokens.DUR_FAST
    )
    colour = theme.mix(theme.EDGE, theme.ACCENT, lit, 0.6 + 0.4 * lit)
    draw = imgui.get_window_draw_list()
    if vertical:
        x = pos.x + grip * 0.5
        draw.add_line((x, pos.y), (x, pos.y + length), imgui.get_color_u32(colour), sp(1))
    else:
        y = pos.y + grip * 0.5
        draw.add_line((pos.x, y), (pos.x + length, y), imgui.get_color_u32(colour), sp(1))
    if not active:
        return 0.0
    delta = imgui.get_io().mouse_delta
    return (delta.x if vertical else delta.y) / tokens.SCALE


#: Where every pane drawn this frame ended up, in screen px: ``id -> (x, y, w,
#: h)``. Recorded by :func:`column` and cleared at the top of each frame by
#: :func:`begin_frame`.
#:
#: The layout *editor* draws its handles and drop bars from this, in a single
#: full-viewport overlay -- so it draws nothing inside any pane, and
#: ``widgets.section_blocks``' double-split corruption is out of the picture by
#: construction rather than by care.
FRAME_PANES: dict[str, tuple[float, float, float, float]] = {}


#: Whether the layout editor is open. Read by :func:`splitter`, written once a
#: frame by :func:`begin_frame` -- a module-level flag rather than an argument
#: threaded through eleven call sites, because it is a property of the *frame*
#: and every splitter in it answers the same way.
_EDITING = False


def begin_frame(editing: bool = False) -> None:
    """Forget last frame's pane rects. Called once, before anything draws."""

    global _EDITING

    FRAME_PANES.clear()
    _EDITING = bool(editing)


def column(
    ctx: Any,
    lay: Any,
    slots: Any,
    *,
    width: float = 0.0,
    edge: Any = None,
    handle_length: float = 0.0,
    on_hidden: Any = None,
) -> None:
    """Draw one ordered stack of :class:`~.layout_skeleton.Slot`, with handles.

    The one answer to a question three different pieces of arithmetic used to
    give (``main.py:466``, ``:3546``, ``:3649``). Heights come from
    :func:`layout_skeleton.heights`, which is pure and tested as numbers; this
    adds the imgui: the child per slot, a splitter between each adjacent
    shareable pair, and the ``FRAME_PANES`` record.

    **A slot with no room is dropped, not drawn at zero.** A negative child
    height silently kills a canvas, and a zero-height one is a pane the user
    cannot see and cannot reach; ``on_hidden`` is called with its id so a
    workspace can clean up after it -- which is where Inker's
    ``pending_zoom_rung = 0`` lives.
    """

    from . import layout_skeleton as skeleton
    from . import tokens

    live = [slot for slot in slots if slot.applies(ctx)]
    if not live:
        return
    imgui.begin_group()
    avail_y = imgui.get_content_region_avail().y
    # **The handles come off the room before it is split.** Every splitter
    # drawn below costs its grip plus the item spacing either side, and
    # ``heights`` -- which promises the heights sum to the room available --
    # never saw that: the column handed out more than it could draw, and the
    # shortfall landed on the last pane. Inker's Picker was 23 px short of the
    # floor it had just been granted, which is exactly one handle, and drew its
    # hex field over its own bottom edge.
    handles = sum(
        1
        for index, slot in enumerate(live)
        if slot.share_key and index + 1 < len(live)
    )
    if handles:
        avail_y = max(0.0, avail_y - handles * (sp(GRIP) + imgui.get_style().item_spacing.y * 2.0))
    shares = {slot.share_key: lay.share(slot.share_key) for slot in live if slot.share_key}
    tall = skeleton.heights(live, avail_y, shares, tokens.SCALE)
    for index, (slot, height) in enumerate(zip(live, tall, strict=True)):
        if height <= 0.0 and slot.sizing != FILL_SIZING:
            if on_hidden is not None:
                on_hidden(slot.id)
            continue
        role = slot.role or PaneRole.CONTENT
        with pane(
            slot.id,
            (width, 0.0 if slot.sizing == FILL_SIZING else height),
            role,
            edge=slot.edge or edge or PaneEdge.NONE,
            title=slot.label or slot.id,
        ) as visible:
            if visible:
                slot.draw(ctx)
            elif on_hidden is not None:
                on_hidden(slot.id)
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        FRAME_PANES[slot.id] = (low.x, low.y, high.x - low.x, high.y - low.y)
        following = live[index + 1] if index + 1 < len(live) else None
        if slot.share_key and following is not None:
            drag = splitter(
                f"{slot.share_key}-share", vertical=False, length=handle_length or width
            )
            if drag and avail_y > 0:
                before = lay.share(slot.share_key)
                lay.set_share(slot.share_key, before + drag * tokens.SCALE / avail_y)
                if lay.share(slot.share_key) != before:
                    lay.save()
    imgui.end_group()
