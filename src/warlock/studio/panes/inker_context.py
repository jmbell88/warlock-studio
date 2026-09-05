"""The row under the tabs: what the tool in your hand is set to.

Aseprite's context bar, and the reason the toolbox can be a 90 px rail: the
options that were nine ``labeled_slider``s and five ``labeled_combo``s down the
left column -- two rows each, a label line and a full-width control, ~812 px of
sidebar -- are one 38 px row above the canvas, beside the drawing they are
about.

**Which widgets appear is a table**, ``inker_state.CONTEXT_WIDGETS``, not a
chain of ifs here: that is what lets a test assert both directions, and in
particular that *every* key in ``TOOL_OPTION_DEFAULTS`` is reachable from some
tool's bar. An option that quietly became unreachable when it left the sidebar
would be the one real risk of this move.

**Four state bars take precedence over the tool bar**, checked in order, and
the order is the modality: a transform, then a floating buffer, then a
half-finished gesture, then a selection. Each is about something the user is in
the middle of, and while they are in the middle of it that is what the row is
for. The selection bar is the only one that also draws under a tool bar,
because a mask outlives the gesture that made it.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, inker, inker_mode, inker_state, toolbar, widgets
from ..tokens import sp

#: How wide a context field is, in design px, per widget kind. A number rather
#: than a measurement: a field's natural size is a decision, and ``toolbar``
#: needs it *before* it can choose tiers.
WIDE = 130.0
NARROW = 92.0
COMPACT = 54.0

DYNAMICS_POPUP = "inker-dynamics"
OPTIONS_POPUP = "inker-tool-panels"
#: Behind the ``Sym`` word: the mirrors again in words, radial, the axis and
#: Reset. It replaces ``inker_tools``' canvas popover, which was opened by a
#: flip-horizontal glyph in another pane and named after neither thing it held.
SYMMETRY_POPUP = "inker-symmetry"
#: Behind the eye: the grid, snapping, the rulers and the rest of the aids that
#: were split between three unlabelled toolbox glyphs and the View menu.
VIEW_POPUP = "inker-view"

#: The options that are a box rather than a number or a menu.
_CHECKS = frozenset(
    {
        "pixel_perfect",
        "shape_filled",
        "wand_contiguous",
        "wand_eight",
        "fill_stop_grid",
        "sample_layer",
        "aa",
        "use_stamp",
    }
)

NIB_LABELS = [
    ("soft", "Soft"),
    ("pixel", "Pixel"),
    ("square", "Square"),
    ("line", "Line"),
]
INK_LABELS = inker_state.INK_LABELS
#: One entry per ``brush.STAMP_ALIGN`` member, and pinned against it in
#: ``tests/inker/test_pattern_fill.py``. It used to read ``free``/``origin``/
#: ``tile`` -- three names, two of which no longer existed anywhere in the
#: engine, so picking either wrote a value ``StrokeState.__post_init__`` then
#: snapped back to ``free``: a combo with two settings that did nothing.
ALIGN_LABELS = [("free", "Free"), ("aligned", "Aligned")]
#: What the paint bucket reads its region off. Aseprite's "Refer to", and the
#: reason lineart on its own layer is fillable at all.
REFER_LABELS = [("canvas", "Canvas"), ("layer", "Layer")]
COMBINE_LABELS = [
    ("replace", "Replace"),
    ("add", "Add"),
    ("subtract", "Subtract"),
    ("intersect", "Intersect"),
]

#: The four mirrors, as independent toggles -- Aseprite's context bar, and the
#: reason ``brush.SYMMETRY_AXES`` exists: they *compose*, so this is four
#: buttons rather than a seven-way combo, and "horizontal and top-left
#: diagonal" is a state the engine can now be in.
#:
#: **Labelled with ASCII rather than glyphs**, which is ``inker_timeline._STEPS``'
#: rule and its reason: ``icons.py`` is a transcription of lucide-static
#: 0.525.0's codepoint assignments and its docstring forbids guessing one, and
#: the vendored subset carries ``flip-horizontal`` but no ``flip-vertical`` and
#: nothing for either diagonal. Three invented codepoints to avoid one honest
#: character is the wrong trade, and a mirror line is a character.
#:
#: ``diag`` draws as ``\`` and ``anti`` as ``/`` because the *screen* y axis
#: grows downward: the reflection ``inker`` calls top-left-to-bottom-right is
#: the backslash on the page, and getting that pair the wrong way round is a
#: button that mirrors the other way from the one it is drawn as.
SYMMETRY_TOGGLES: tuple[tuple[str, str, str], ...] = (
    ("x", "H", "Horizontal symmetry: every dab is mirrored left/right about the axis"),
    ("y", "V", "Vertical symmetry: every dab is mirrored top/bottom about the axis"),
    (
        "diag",
        "\\",
        "Diagonal symmetry, top-left to bottom-right. Combines with the others.",
    ),
    (
        "anti",
        "/",
        "Diagonal symmetry, bottom-left to top-right. Combines with the others.",
    ),
)

SYMMETRY_RESET = "symreset"


def draw(ctx: Any, state: Any, tab: Any) -> None:
    """The bar. Called by the canvas, between the tab bar and the image."""

    if tab is None:
        return
    if state.transforming:
        _transform_bar(ctx, state, tab)
        return
    if tab.doc.floating is not None:
        _float_bar(ctx, state, tab)
        return
    if state.gesture_pts:
        _gesture_bar(ctx, state, tab)
        return
    _tool_bar(ctx, state, tab)
    if tab.doc.mask is not None:
        _selection_bar(ctx, state, tab)


# --- the state bars ---------------------------------------------------------


def _transform_bar(ctx: Any, state: Any, tab: Any) -> None:
    """Deliberately empty here: the transform row is the canvas's own.

    ``inker_canvas._transform_row`` draws it, because the numeric handles read
    and write the floating buffer the canvas is already holding. Named as a bar
    all the same, so the precedence order above is the whole truth about what
    this row shows.
    """


def _float_bar(ctx: Any, state: Any, tab: Any) -> None:
    """A pasted or lifted buffer, and the two ways out of it."""

    widgets.text_colored(_accent(), "Floating")
    imgui.same_line()
    items = [
        toolbar.Item("drop", "Drop", icons.CHECK, role=controls.ButtonRole.PRIMARY, pinned=True),
        toolbar.Item("cancel", "Cancel", icons.X, pinned=True),
    ]
    hit = toolbar.toolbar("inker-float", items)
    if hit == "drop":
        tab.doc.commit_floating()
    elif hit == "cancel":
        tab.doc.cancel_floating()
    widgets.divider()


def _gesture_bar(ctx: Any, state: Any, tab: Any) -> None:
    """A half-drawn polygon, its vertex count, and how to end it.

    It closes a real hole: Enter has always finished a multi-click shape and
    only the manual said so, which is a gesture with no way out on screen.
    """

    count = len(state.gesture_pts)
    widgets.text_colored(_accent(), f"{inker_state.tool_label(state.tool)}: {count} points")
    imgui.same_line()
    items = [
        toolbar.Item(
            "finish",
            "Finish",
            icons.CHECK,
            tooltip="Close the shape (Enter)",
            role=controls.ButtonRole.PRIMARY,
            enabled=count >= 2,
            reason="A shape needs at least two points.",
            pinned=True,
        ),
        toolbar.Item("cancel", "Cancel", icons.X, tooltip="Drop it (Esc)", pinned=True),
    ]
    hit = toolbar.toolbar("inker-gesture", items)
    if hit == "finish":
        inker_mode.commit_gesture(state, tab)
    elif hit == "cancel":
        state.clear_gesture()
    widgets.divider()


def _selection_bar(ctx: Any, state: Any, tab: Any) -> None:
    """The combine mode, shown whenever a mask exists.

    Not only under a select tool, which is the change: ``state.combine`` has
    always been sampled at press and has never been visible anywhere, so
    "Shift adds, Alt+Shift subtracts" was prose in a panel rather than a
    control with a state. A mask outlives the tool that made it, so the bar
    that says what the next one will do to it does too.
    """

    imgui.same_line()
    # ``sticky_combine``, not ``combine``: the second is what the *current*
    # gesture is doing (a held Shift wins for its own drag), and drawing the
    # bar from it would make the control flicker to Add while Shift is down
    # and back afterwards -- and, before 2026-09-03, made every press
    # overwrite what this control had set.
    changed, mode = controls.segmented_choice(
        "inker-combine", COMBINE_LABELS, state.sticky_combine, compact=True
    )
    if changed:
        state.sticky_combine = mode


# --- the tool bar -----------------------------------------------------------


def _tool_bar(ctx: Any, state: Any, tab: Any) -> None:
    tool = state.tool
    keys = inker_state.widgets_for(tool, tab.doc, state)
    inline = [key for key in keys if not inker_state.context_group(key)]
    behind = [key for key in keys if inker_state.context_group(key) == "dynamics"]
    fields = [_field(ctx, state, tab, key) for key in inline]
    fields = [field for field in fields if field is not None]
    items = []
    from . import inker_tools

    if inker_tools._has_panels(state, tab):
        # The panels a 38 px row is the wrong shape for -- a list of gradient
        # stops, the document's slices, the image brush's verbs, the named
        # presets -- behind one button rather than in a column that would have
        # to exist all session for the tools that have one.
        items.append(
            toolbar.Item(
                "panels",
                inker_state.tool_label(tool),
                icons.SETTINGS,
                tooltip=f"{inker_state.tool_label(tool)} panels",
                pinned=True,
            )
        )
    if behind:
        items.append(
            toolbar.Item(
                "dynamics",
                "Dynamics",
                icons.SLIDERS,
                tooltip="Spacing, smoothing, taper -- how the stroke is laid down",
                pinned=True,
            )
        )
    hit = toolbar.toolbar(
        "inker-context", items, fields=fields, trailing=symmetry_trailing(ctx, state)
    )
    if hit == "dynamics":
        imgui.open_popup(DYNAMICS_POPUP)
    elif hit == "panels":
        imgui.open_popup(OPTIONS_POPUP)
    _dynamics_popup(ctx, state, tab, behind)
    _panels_popup(ctx, state, tab)
    widgets.divider()


def symmetry_trailing(ctx: Any, state: Any) -> Any:
    """The sitting's settings, at the **end** of every tool bar: ``View`` and a
    named ``Sym`` group holding the four mirrors.

    **On the bar rather than in the toolbox popover**, which is where the old
    seven-way combo lived and is what changed on 2026-08-23. The argument for
    the popover was that symmetry is a setting of the *sitting* rather than of
    the tool -- which is true, and is also true of Aseprite's, where it is on
    the context bar anyway. What settles it is the gesture: a mirror is
    switched on to draw one thing and off again a moment later, which is two
    round trips through a popover per use, and the guide it draws is on the
    canvas the bar is sitting over.

    A ``trailing`` block rather than ``items``, and that is a layout decision
    rather than a tidiness one: ``toolbar`` draws its items *before* its
    fields, so as buttons these landed between the Dynamics glyph and the size
    slider -- in the middle of the tool's own settings, which is the one place
    a session-wide setting must not be. Trailing puts them past the fields, at
    the right-hand end, which is where Aseprite's are.

    **What changed on 2026-08-31**, and why it is not the deleted row coming
    back:

    * The four mirrors were five loose buttons labelled ``H``, ``V``, ``\\``
      and ``/`` with nothing on screen saying what they were. They are one
      named group now -- the word ``Sym``, then the mirrors as a pill group --
      which is what makes four glyphs read as one control rather than four
      unrelated ones. ``Reset`` moved into the popover, and its width is what
      pays for the word.
    * Radial symmetry and the axis were behind a *flip-horizontal* icon in the
      toolbox, a pane away from the canvas and named after something else
      entirely. They are behind the ``Sym`` word now, which is the thing they
      belong to.
    * ``View`` is one glyph, and behind it are the seven aids that were split
      between three unlabelled toolbox glyphs and the View menu.

    And the block **can collapse** now (:class:`toolbar.Trailing`), to ``View``
    plus ``Sym`` alone. That is what answers the tombstone at ``inker_canvas``:
    the old five buttons took their 201 px unconditionally, so a bar with no
    room simply clipped or wrapped. This one has a floor of 93 px and a rule
    about when it reaches it.

    **The rule is that a label is cheaper to lose than a control**, and it is
    measured rather than argued. At the app's default 1600x950 the bar has
    about 835 px -- the mode rail takes 70 that a two-sidebars sum misses --
    the brush's row wants 689 with every label on, and this block wants 217.
    Both cannot have everything. What gives way first is the row's *words*: a
    slider reading ``129`` instead of ``129 px`` is still that slider and still
    says what it is on hover, whereas a folded mirror is not a narrower mirror,
    it is a button that is no longer on screen. So the mirrors stay and the
    labels go, which is also what this bar did before any of this and what
    Aseprite does.
    """
    from ..inker import brush

    axes = brush.axes_of(state.symmetry)
    style = imgui.get_style()
    gap = style.item_spacing.x
    side = imgui.get_frame_height()
    word = imgui.calc_text_size("Sym").x + style.frame_padding.x * 2.0
    view_w = imgui.calc_text_size("View").x + style.frame_padding.x * 2.0
    # No inter-segment gap: ``segmented_flags`` draws the mirrors contiguously,
    # which is both what makes four glyphs read as one control and what keeps
    # the group on the bar beside a five-field tool at the default window size.
    mirrors = side * len(SYMMETRY_TOGGLES)
    full = view_w + gap + word + gap + mirrors
    compact = view_w + gap + word

    def draw_it(tight: bool) -> None:
        # **A word, not a glyph.** The whole complaint this answers is that
        # these controls were unreadable, and an eye icon beside four ASCII
        # mirrors would have been one more thing to hover to identify. It also
        # keeps the door inside ``controls``, which is what the census can see
        # -- ``widgets.icon_button`` draws through imgui directly and is
        # invisible to ``probe``, so a View button that stopped working could
        # not be caught by a test that presses it.
        if controls.button(
            "View##ctxview",
            selected=_view_dirty(state),
            tooltip="The grid, snapping, the rulers and the rest of the view aids",
        ):
            imgui.open_popup(VIEW_POPUP)
        _view_popup(ctx, state)
        imgui.same_line()
        # The word is the group's *name* and its door in one. ``selected`` is
        # load-bearing rather than decorative: collapsed, this is the only
        # thing on screen that can say a mirror is armed.
        if controls.button(
            "Sym##ctxsym/open",
            selected=bool(axes),
            tooltip="Symmetry -- the mirrors, radial, and the axis they turn about",
        ):
            imgui.open_popup(SYMMETRY_POPUP)
        _symmetry_popup(ctx, state)
        if tight:
            return
        imgui.same_line()
        hit = controls.segmented_flags(
            "ctxsym",
            [(axis, label) for axis, label, _tip in SYMMETRY_TOGGLES],
            axes,
            tooltips={axis: tip for axis, _label, tip in SYMMETRY_TOGGLES},
            width=side,
        )
        if hit:
            _symmetry_hit(ctx, state, f"sym/{hit}")

    return toolbar.Trailing(full, compact, draw_it)


#: The one symmetry that is **not** a mirror, and so not one of the four
#: toggles: a radial symmetry is a set of turns about the axis, it takes a
#: *count*, and it is reached when a mandala is being drawn rather than between
#: strokes. It lives in the popover, beside the number it needs and the axis
#: both it and the mirrors turn about.
#:
#: It was ``inker_tools.RADIAL_AXIS`` until 2026-08-31, in a popover off a
#: flip-horizontal glyph in that pane. It is here now because this is where the
#: rest of the symmetry is, and a setting reached from a control named after
#: something else is one nobody finds.
RADIAL_AXIS = "radial"


def _view_dirty(state: Any) -> bool:
    """Whether any view aid is on, so the ``View`` word can say so.

    The same job ``selected`` does on ``Sym``: the aids are behind a door, and
    a door that looks identical whether or not the grid is on is one the user
    has to open to find out.
    """
    return bool(
        state.grid
        or state.grid_snap
        or state.pixel_grid
        or state.layer_edges
        or state.tile_numbers
    )


def _symmetry_popup(ctx: Any, state: Any) -> None:
    """Behind the ``Sym`` word: the mirrors in words, radial, the axis, Reset."""

    from ..inker import brush

    with controls.menu_popup(SYMMETRY_POPUP) as opened:
        if not opened:
            return
        axes = brush.axes_of(state.symmetry)
        # **The mirrors again, in words.** Deliberate duplication, and it is
        # ``toolbar``'s own doctrine: what collapses out of a row comes back
        # with its full label rather than in a second, smaller place to lose
        # it. Collapsed this is the only door to them; at full width it is
        # where the words live, because ``\`` and ``/`` are honest characters
        # and still not English.
        for axis, _label, tip in SYMMETRY_TOGGLES:
            changed, _value = controls.checkbox(
                f"{_MIRROR_WORDS[axis]}##popsym/{axis}", axis in axes
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip(tip)
            if changed:
                _symmetry_hit(ctx, state, f"sym/{axis}")
                axes = brush.axes_of(state.symmetry)
        widgets.divider()
        changed, _radial = controls.checkbox(
            "Radial##popsym/radial", RADIAL_AXIS in axes
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Turns about the axis rather than mirrors across it, and "
                "composes with the four mirrors."
            )
        if changed:
            _symmetry_hit(ctx, state, f"sym/{RADIAL_AXIS}")
            axes = brush.axes_of(state.symmetry)
        if RADIAL_AXIS in axes:
            imgui.set_next_item_width(sp(90))
            changed, count = controls.slider_int(
                "Ways", int(state.radial_count), brush.MIN_RADIAL, brush.MAX_RADIAL
            )
            if changed:
                state.radial_count = int(count)
                inker_mode.persist(ctx)
        _symmetry_axis(ctx, state)
        widgets.divider()
        # Disabled rather than hidden: a reset that appears only once something
        # is set is a control the user cannot learn is there.
        dirty = bool(axes) or state.symmetry_axis is not None or (
            int(state.radial_count) != brush.DEFAULT_RADIAL
        )
        if widgets.disabled_button(
            f"Reset##popsym/{SYMMETRY_RESET}",
            dirty,
            reason="No symmetry is set.",
            tooltip=(
                "Switch every symmetry off, put the axis back in the middle of "
                "the canvas and the radial count back to its default."
            ),
        ):
            _symmetry_hit(ctx, state, f"sym/{SYMMETRY_RESET}")


#: The mirrors as words, for the popover. The bar keeps the characters -- a
#: mirror line *is* a character, and ``icons.py`` forbids inventing a codepoint
#: for the two diagonals -- and this is where the words come back.
_MIRROR_WORDS = {
    "x": "Horizontal",
    "y": "Vertical",
    "diag": "Diagonal \\",
    "anti": "Diagonal /",
}


def _symmetry_axis(ctx: Any, state: Any) -> None:
    """Where the mirrors sit. Empty means the canvas centre.

    Offered as two numbers rather than a draggable handle because the useful
    values are exact ones -- the centre, a character's spine, a tile edge --
    and a handle can only be dragged near them.

    **Both writers persist.** Neither did until 2026-08-31: this function was
    the one place that changed ``symmetry_axis`` without calling ``persist``
    at all, so an axis moved off centre was gone at the next launch even after
    the rest of the symmetry began to be written down.
    """
    axis = state.symmetry_axis
    imgui.set_next_item_width(sp(120))
    changed, values = controls.input_float2(
        "Axis##symaxis", list(axis or (0.0, 0.0)), "%.0f"
    )
    if changed:
        state.symmetry_axis = (float(values[0]), float(values[1]))
    # On the deactivation rather than on every changed frame, which is the
    # idiom the grid's step already uses: a settings write per pixel of drag is
    # what that rule exists to avoid, and this is what survives a crash.
    if imgui.is_item_deactivated_after_edit():
        inker_mode.persist(ctx)
    imgui.same_line()
    if widgets.disabled_button(
        "Centre##symcentre",
        axis is not None,
        reason="The axis is already centred.",
        tooltip="Put the symmetry axis back in the middle of the canvas.",
    ):
        # Back to None rather than to the middle of the current document: None
        # *is* the centre, and stays the centre across a resize.
        state.symmetry_axis = None
        inker_mode.persist(ctx)
    if axis is None:
        widgets.muted("centred")


def _view_popup(ctx: Any, state: Any) -> None:
    """Behind the eye: how the canvas is *drawn*, all seven aids in one place.

    They were split three ways before -- grid, snap and rulers as unlabelled
    glyphs in the toolbox, a pane away from the drawing; the pixel grid, the
    layer edges and the tile numbers in the View menu only; the grid's *step*
    in a popover named after symmetry. Filed by kind they are one thing:
    app-level state about how the canvas is drawn, persisted beside the
    swatches, and none of them is a verb or a readout.

    **Every row goes through the op registry** rather than setting the flag
    here. That is what keeps the menu row, this row and the persisted value
    from ever disagreeing, and it is where the refusal sentences come from.

    **The tiling four are not app-level** -- ``tab.tiled`` is a property of the
    document -- so they are separated and labelled as such. They stay View-menu
    ops as well, and deliberately did not become bar controls: a four-way
    setting nobody changes mid-stroke is what a checked menu row is *for*, and
    it would cost the bar a combo's width to say what a tick already says.
    """
    from .. import inker_ops

    with controls.menu_popup(VIEW_POPUP) as opened:
        if not opened:
            return
        for key in ("toggle_grid", "toggle_snap"):
            _op_row(ctx, inker_ops, key)
        imgui.set_next_item_width(sp(80))
        changed, size = controls.input_int("Grid size##viewgrid", state.grid_size, 0)
        if changed:
            state.grid_size = max(2, min(512, size))
        if imgui.is_item_deactivated_after_edit():
            inker_mode.persist(ctx)
        _op_row(ctx, inker_ops, "grid_from_selection")
        widgets.divider()
        for key in ("toggle_rulers", "toggle_pixel_grid", "toggle_layer_edges",
                    "toggle_tile_numbers"):
            _op_row(ctx, inker_ops, key)
        widgets.divider()
        widgets.muted("This document")
        for mode, label in inker_ops.TILED_MODES:
            _op_row(ctx, inker_ops, f"tiled_{mode}", label=label)
        _op_row(ctx, inker_ops, "wrap_half")


def _op_row(ctx: Any, inker_ops: Any, key: str, *, label: str = "") -> None:
    """One registry op as a checkable row, with its own enablement and reason.

    Through the registry rather than through a ``setattr`` here, and that is
    the point of the whole popover: ``inker_ops._toggle`` stays the one writer,
    so this row, the View-menu row and the value ``inker_mode.persist`` writes
    down cannot drift apart -- and the refusal sentence comes for free instead
    of being written a second time.
    """
    op = inker_ops.get(key)
    state = ctx.state.inker
    tab = state.active if state is not None else None
    enabled = bool(op.enabled(state, tab))
    checked = bool(op.checked(state, tab)) if op.checked else False
    hit, _value = controls.checkbox(
        f"{label or op.label}##view/{key}",
        checked,
        enabled=enabled,
        reason=inker_ops.reason_for(op, state, tab),
    )
    if hit and enabled:
        inker_ops.run(ctx, op)


def _symmetry_hit(ctx: Any, state: Any, hit: str) -> None:
    """One of the five symmetry buttons, or nothing this function owns."""

    from ..inker import brush

    if not hit.startswith("sym/"):
        return
    axis = hit[len("sym/") :]
    if axis == SYMMETRY_RESET:
        state.symmetry = "none"
        state.symmetry_axis = None
        state.radial_count = brush.DEFAULT_RADIAL
    else:
        state.symmetry = brush.toggled(state.symmetry, axis)
    # Persisted on the press: symmetry is part of the session snapshot the
    # colours and the grid are in, and a mirror that switches itself off at
    # the next launch is a setting the user has to rediscover.
    #
    # **This call was here long before that was true.** ``persist`` wrote seven
    # canvas keys and not one of them was a symmetry, so every press paid for a
    # settings write that dropped the thing it was called to save, and the
    # comment above described an intention rather than the code. Three keys
    # joined the block on 2026-08-31; ``test_the_mirrors_survive_a_restart`` is
    # what stops it becoming a claim again.
    inker_mode.persist(ctx)


def _dynamics_popup(ctx: Any, state: Any, tab: Any, keys: list[str]) -> None:
    """The settings of a *sitting* rather than of a stroke, behind one glyph.

    Aseprite's own arrangement. These four are reached when a way of working
    changes and then left for an hour, so a row that showed them permanently
    would spend its width on the controls least often touched.
    """

    if not keys:
        return
    with controls.menu_popup(DYNAMICS_POPUP) as opened:
        if not opened:
            return
        for key in keys:
            widgets.field_label(inker_state.context_label(key))
            imgui.set_next_item_width(sp(WIDE))
            field = _field(ctx, state, tab, key)
            if field is not None:
                field.draw(False)


def _panels_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The tool's panels, in a popover off the bar."""

    with controls.menu_popup(OPTIONS_POPUP) as opened:
        if not opened:
            return
        from . import inker_tools

        imgui.push_item_width(sp(200))
        inker_tools.panels(ctx, state, tab)
        imgui.pop_item_width()


def _field(ctx: Any, state: Any, tab: Any, key: str) -> Any:
    """One option as a :class:`toolbar.Field`, or None if it has no widget.

    A ``Field`` rather than a drawn control, so the row can measure the whole
    bar before it draws any of it -- ``same_line`` past the content region
    clips rather than wraps, and a clipped option is one the user cannot reach
    and cannot see to ask about.
    """

    options = state.options_for(state.tool)
    label = inker_state.context_label(key)

    def slider_int(low: int, high: int, fmt: str = "%d", width: float = NARROW) -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.slider_int(
                f"##ctx/{key}", int(options[key]), low, high, fmt
            )
            if changed:
                options[key] = int(value)

        return toolbar.Field(key, label, draw, width=width, compact=COMPACT)

    def slider_float(low: float, high: float, fmt: str = "%.2f") -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.slider_float(
                f"##ctx/{key}", float(options[key]), low, high, fmt
            )
            if changed:
                options[key] = float(value)

        return toolbar.Field(key, label, draw, width=NARROW, compact=COMPACT)

    def percent_slider(low: float, high: float) -> Any:
        """A 0..1 option drawn as ``Opacity 85%``, stored as the fraction.

        The same rule ``widgets.labeled_slider_float`` applies everywhere else
        in the app: the model keeps its natural units and the reader gets the
        ones they think in. The word on the face is the option's *full* label
        from ``inker_state.context_label`` -- this bar said ``opa 1.00`` and
        ``hard 0.85`` while the timeline, the menu and Plotter's layer pane
        all said "Opacity", which the 2026-09-05 consistency review caught.
        """

        def draw(compact: bool) -> None:
            changed, shown = controls.slider_float(
                f"##ctx/{key}",
                float(options[key]) * 100.0,
                low * 100.0,
                high * 100.0,
                f"{label} %.0f%%",
            )
            if changed:
                options[key] = float(shown) / 100.0

        return toolbar.Field(key, label, draw, width=NARROW, compact=COMPACT)

    def combo(choices: list[tuple[str, str]], width: float = NARROW) -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.combo(f"##ctx/{key}", str(options[key]), choices)
            if changed:
                options[key] = value

        # **A combo does not compact.** Its content is a word, and a combo
        # narrower than its own longest option shows "So" where it means
        # "Soft" -- which a screenshot at 1.0 caught. A slider can shrink
        # because its content is a number and the number stays whole.
        return toolbar.Field(key, label, draw, width=width, compact=width)

    def check() -> Any:
        def draw(compact: bool) -> None:
            changed, value = controls.checkbox(f"{label}##ctx/{key}", bool(options[key]))
            if changed:
                options[key] = value

        return toolbar.Field(key, label, draw, width=WIDE, compact=WIDE, priority=1)

    # **Every slider carries its own name in its format string.** A row of
    # bare numbers -- 12, 0.85, 1.00 -- is three settings the user has to
    # count along the bar to identify, which is what the screenshot pass
    # found. The name is the option's full label: the 92 px field once
    # justified "opa" and "hard", and the consistency review of 2026-09-05
    # found the same setting called "Opacity" everywhere else, including
    # Inker's own timeline. A percentage is shorter than a two-place float,
    # which is what buys the whole word back.
    if key == "brush_size":
        return slider_int(inker.MIN_BRUSH, inker.MAX_BRUSH, "%d px")
    if key == "nib":
        # Not COMPACT: "Soft" drawn in a 54 px combo reads "So", which the
        # screenshot pass at 1.0 caught. A combo is as wide as its longest
        # word or it is lying about what is selected.
        return combo(NIB_LABELS)
    if key == "hardness":
        return percent_slider(0.0, 1.0)
    if key == "opacity":
        return percent_slider(0.05, 1.0)
    if key == "paint_ink":
        def draw(compact: bool) -> None:
            changed, value = controls.combo(
                "##ctx/paint_ink", state.ink, list(INK_LABELS)
            )
            if changed:
                # Through ``set_ink``, which is where "same in all tools" is
                # honoured -- writing ``options`` directly would set the ink on
                # the tool in hand and leave the others behind.
                state.set_ink(value)
            if imgui.is_item_hovered():
                imgui.set_tooltip(inker_state.ink_hint(state.ink))

        return toolbar.Field(key, label, draw, width=WIDE, compact=NARROW)
    if key == "spray_rate":
        return slider_int(5, 400, "%d/s")
    if key in ("spacing", "stabilise", "speed_taper", "strength"):
        return percent_slider(0.0, 1.0)
    if key == "wand_tolerance":
        return slider_int(0, 255, "tol %d")
    if key == "brush_angle":
        return slider_int(0, 180, "%d deg")
    if key == "corner_radius":
        # **The slider is the only door that says the setting exists.** The
        # gesture -- hold ``C`` and roll while dragging a rectangle -- is
        # Aseprite's and stays, but it is undiscoverable on its own, and this
        # key sat in ``CONTEXT_WIDGETS`` with no branch here for exactly that
        # long: ``_tool_bar`` filters a ``None`` out in silence, so the option
        # was listed everywhere and drawn nowhere.
        #
        # The 64 is this control's ceiling, not the engine's: ``_doc_paint``
        # clamps a radius to half the rectangle's shorter side anyway, so the
        # wheel stays uncapped and touching the slider afterwards brings an
        # oversized radius back into range.
        return slider_int(0, 64, "corner %d")
    if key == "text_size":
        return slider_int(4, 256, "%d px")
    if key == "shade_dir":
        return combo([("1", "Lighter"), ("-1", "Darker")], COMPACT)
    if key == "stamp_align":
        return combo(ALIGN_LABELS)
    if key == "fill_refer":
        return combo(REFER_LABELS)
    if key == "gradient_dither":
        from ..inker import dither

        return combo([("none", "None")] + [(m, m.title()) for m in dither.ORDERED])
    if key == "font":
        return combo(inker_mode.font_choices(), WIDE)
    if key in _CHECKS:
        return check()
    return None


def _accent() -> Any:
    from .. import theme

    return theme.ACCENT
