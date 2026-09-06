"""Developer-only visual catalogue for the studio control contract."""

from __future__ import annotations

import os

from imgui_bundle import imgui

from . import controls, fonts, forms, icons, theme, tokens, toolbar, widgets
from .tokens import sp

ENV_KEY = "WARLOCK_DEV_COMPONENTS"
POPUP = "Component gallery##developer"
_requested = False
_scroll_to: float | None = None


def enabled(environ: dict[str, str] | None = None) -> bool:
    """Whether the developer surface is exposed in this process."""

    source = os.environ if environ is None else environ
    return str(source.get(ENV_KEY, "")).strip().lower() in {"1", "true", "yes", "on"}


def request() -> None:
    global _requested
    _requested = True


def scroll_to(fraction: float | None) -> None:
    """Ask the next frame to park the gallery's scroller at ``fraction``.

    For ``scripts/screenshot_modes.py``, which photographs one frame: the
    gallery is a popup with a scrolling child, so a single capture only ever
    showed the blocks above the fold -- four of ten once the catalogue was
    completed (2026-09-05), which is most of the executable catalogue absent
    from the pictures the catalogue is supposed to make reviewable. ``None``
    leaves the scroller where the reader put it, which is what the app does.
    """
    global _scroll_to
    _scroll_to = fraction


def _state_samples() -> None:
    widgets.section("Interaction states")
    states = tuple(controls.ControlState)
    if imgui.begin_table("gallery/states", 2):
        for state in states:
            imgui.table_next_column()
            widgets.muted(state.value.capitalize())
            imgui.table_next_column()
            controls.button(
                f"Button##gallery-state-{state.value}",
                role=controls.ButtonRole.SECONDARY,
                preview=state,
                reason="This sample is disabled.",
            )
        imgui.end_table()


def _buttons() -> None:
    widgets.section("Button roles")
    for index, role in enumerate(controls.ButtonRole):
        if index:
            imgui.same_line()
        controls.button(
            f"{role.value.capitalize()}##gallery-role-{role.value}",
            role=role,
            tooltip=f"{role.value.capitalize()} button",
        )
    widgets.muted("Regular 30 dp")
    controls.button("Regular##gallery-regular")
    imgui.same_line()
    controls.button(
        "Compact##gallery-compact", control_size=controls.ControlSize.COMPACT
    )


def _fields() -> None:
    widgets.section("Inputs and validation")
    imgui.set_next_item_width(sp(220))
    controls.input_text("##gallery-text", "Default")
    imgui.same_line()
    imgui.set_next_item_width(sp(220))
    controls.input_text(
        "##gallery-error", "Needs attention", error="Example validation message"
    )
    imgui.set_next_item_width(sp(220))
    controls.input_float("##gallery-number", 1.25, 0.0, 0.0, "%.2f")
    imgui.same_line()
    imgui.set_next_item_width(sp(220))
    controls.slider_float("##gallery-slider", 0.4, 0.0, 1.0, "%.0f%%")
    changed, _value = controls.combo(
        "##gallery-menu", "one", (("one", "One"), ("two", "Two"))
    )
    del changed


def _choices() -> None:
    widgets.section("Choices and selection")
    controls.switch("Immediate setting", True, control_id="gallery-switch")
    controls.checkbox("Included in export", True)
    controls.segmented_choice(
        "gallery-segments",
        (("one", "One"), ("two", "Two"), ("three", "Three")),
        "two",
    )
    controls.selectable_row("gallery-row-a", "Unselected row")
    controls.selectable_row("gallery-row-b", "Selected row", selected=True)


def _noop() -> None:
    """Every gallery control is live so it draws its pressed and hovered
    states honestly; none of them may *do* anything, because the gallery is a
    popup over whatever mode happened to be open."""


# The row the toolbar block degrades. Two priority groups plus one pinned
# destructive item, because that is the arrangement that shows all three tiers:
# priority 1 gives up its labels first, then moves into the menu, while Delete
# collapses to its glyph and stays on the row (``toolbar``'s pinning rule).
_BAR_ITEMS = (
    toolbar.Item("new", "New", icon=icons.PLUS, role=controls.ButtonRole.PRIMARY),
    toolbar.Item("open", "Open", icon=icons.FOLDER_OPEN),
    toolbar.Item("save", "Save", icon=icons.SAVE, priority=1),
    toolbar.Item("export", "Export", icon=icons.UPLOAD, priority=1),
    toolbar.Item(
        "delete",
        "Delete",
        icon=icons.TRASH,
        role=controls.ButtonRole.DESTRUCTIVE,
        priority=2,
        pinned=True,
    ),
)

# Design pixels. Chosen against the widths ``_BAR_ITEMS`` measures at rather
# than round numbers: one that fits, one that has to drop a group to glyphs,
# one that has to spend the overflow menu as well.
_BAR_WIDTHS = ((480.0, "Full"), (210.0, "Icons"), (140.0, "Menu"))


def _toolbars() -> None:
    widgets.section("Toolbar tiers")
    # Which tier each width *reaches* is a measurement, not a promise: the row
    # degrades by priority group against the physical width, so at 175 % the
    # middle bar has already spent its glyphs on the two lowest groups while
    # the top one still has its words. So the rows are labelled by the width
    # they were given, and the tiers are named in the sentence above them.
    widgets.muted(
        "The same five actions at three widths: labels, then glyphs, then an overflow menu."
    )
    for width, tier in _BAR_WIDTHS:
        widgets.field_label(f"{width:.0f} dp")
        # A child, because ``toolbar`` reads the *content region* to choose its
        # tiers. Asking for three widths in one pane means giving it three
        # content regions; there is no width argument and there should not be.
        if imgui.begin_child(
            f"gallery/bar/{tier}",
            (sp(width), imgui.get_frame_height() + sp(tokens.SP_2)),
        ):
            toolbar.toolbar(f"gallery-bar-{tier}", _BAR_ITEMS)
        imgui.end_child()


def _forms() -> None:
    widgets.section("Form layout")
    widgets.muted(
        f"Stacked below {forms.FORM_BREAKPOINT:.0f} dp, a label column at and above it."
    )
    for width, height, note in (
        (forms.FORM_BREAKPOINT + 120.0, 150.0, "Above the breakpoint"),
        (forms.FORM_BREAKPOINT - 160.0, 200.0, "Below the breakpoint"),
    ):
        widgets.field_label(note)
        # An explicit height: a zero one means "the rest of the pane" here, not
        # "as tall as the form", and the first child then ate the whole gallery.
        if imgui.begin_child(f"gallery/form/{note}", (sp(width), sp(height)), True):
            with forms.Form(f"gallery-form-{note}", errors={"seed": "Must be a number."}) as form:
                form.text("name", "Name", "Goblin", helper="Shown in the library.")
                form.text("seed", "Mesh seed", "abc")
        imgui.end_child()


def _empty_states() -> None:
    widgets.section("Empty states")
    height = sp(215)
    # Both forms, because the ``action=`` half arrived later than the sentence
    # half and a catalogue that showed one of them would say the other is gone.
    for key, args, kwargs in (
        ("plain", (icons.SEARCH, "No matches", "Try a different search."), {}),
        (
            "action",
            (icons.IMAGE, "No references yet", ""),
            {"action": ("Make a reference", _noop)},
        ),
    ):
        if imgui.begin_child(f"gallery/empty/{key}", (0, height), True):
            widgets.empty_state(*args, **kwargs)
        imgui.end_child()
    widgets.field_label("Nothing open")
    if imgui.begin_child("gallery/nothing-open", (0, sp(395)), True):
        widgets.nothing_open(
            "Open a document or start a new one.",
            (("New##gallery-nothing-new", _noop), ("Open a file...##gallery-nothing-open", _noop)),
        )
    imgui.end_child()


class _FakeTab:
    """What :func:`widgets.document_header` reads off a tab, and nothing else.

    Deliberately not a real ``docmodes`` tab: the gallery is drawn over a live
    mode and must not be able to reach into one's state, which is the whole of
    why this module imports no mode.
    """

    path = "C:/assets/goblin.wlk"
    dirty = True
    saving = False


def _document_header() -> None:
    widgets.section("Document header")
    widgets.document_header(
        _FakeTab(), new=_noop, open_=_noop, save=_noop, save_as=_noop
    )


def _badges() -> None:
    widgets.section("Status and stage")
    for index, status in enumerate(theme.STATUS_GLYPHS):
        if index:
            imgui.same_line()
        widgets.status_pill(status)
    for index, stage in enumerate(widgets.STAGE_BADGES):
        if index:
            imgui.same_line()
        widgets.stage_badge({"stage": stage})


# The five roles ``fonts`` exposes, largest first, so the ramp reads as a ramp.
_TYPE_RAMP = (
    ("display", fonts.display),
    ("heading", fonts.heading),
    ("title", fonts.title),
    ("label", fonts.label),
    ("small", fonts.small),
)


def _type_ramp() -> None:
    widgets.section("Type ramp")
    for name, role in _TYPE_RAMP:
        with role(imgui):
            imgui.text(f"{name} - The quick brown fox")


def draw() -> None:
    """Draw the gallery popup when requested by the developer build."""

    global _requested
    if not enabled():
        _requested = False
        return
    if _requested:
        imgui.open_popup(POPUP)
        _requested = False
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(
        viewport.get_center(), imgui.Cond_.appearing.value, (0.5, 0.5)
    )
    # Floored. The inset is subtracted unconditionally, so on a viewport
    # narrower than the margin itself the requested size goes *negative* and
    # imgui is asked for a window of impossible extent. 240 design px is small
    # enough to fit anything that can host a window at all and large enough to
    # still be a gallery rather than a sliver.
    inset = sp(64)
    floor = sp(240)
    imgui.set_next_window_size(
        (
            max(floor, min(sp(760), viewport.work_size.x - inset)),
            max(floor, min(sp(640), viewport.work_size.y - inset)),
        ),
        imgui.Cond_.appearing.value,
    )
    if not imgui.begin_popup(POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    widgets.pane_header(
        "Component gallery",
        help_text="Developer preview of shared control states.",
        actions=(("close", f"{icons.X} Close", imgui.close_current_popup),),
    )
    widgets.muted("Shared controls in the current theme and UI scale.")
    global _scroll_to
    if imgui.begin_child("gallery/scroll", (0, 0)):
        if _scroll_to is not None:
            # After ``begin_child`` because the scroll maximum is a property of
            # the child, and only once: a request that reapplied every frame
            # would pin the scroller and the reader could not move it.
            imgui.set_scroll_y(_scroll_to * imgui.get_scroll_max_y())
            _scroll_to = None
        # The gallery exists to show shared controls as a pane draws them, so it
        # asks for the blocks a pane gets -- a gallery that showed the headings
        # flat would be showing a look no pane has.
        with widgets.section_blocks():
            _state_samples()
            _buttons()
            _fields()
            _choices()
            _toolbars()
            _forms()
            _empty_states()
            _document_header()
            _badges()
            _type_ramp()
    imgui.end_child()
    imgui.end_popup()
