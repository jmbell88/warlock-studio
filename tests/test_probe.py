"""The per-frame control census, and the blind spot it does not cover.

Headless throughout: ``probe.record`` is guarded by an imgui-context check, so
every one of these runs without a window. The one thing that cannot be tested
here is a real click -- that needs a window, and it is what
``scripts/exercise_mode.py`` is for.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from warlock.studio import controls, probe

#: Every raw imgui widget call outside ``controls.py``. Pinned rather than
#: chased: each one is a control the probe cannot see, and the number belongs
#: in a report rather than in a silent gap. Lower it when one is migrated;
#: raising it means a new control bypassed the presentational layer.
RAW_IMGUI_CONTROLS = 11

_RAW_WIDGETS = {
    "button",
    "small_button",
    "checkbox",
    "radio_button",
    "selectable",
    "collapsing_header",
    "menu_item",
    "menu_item_simple",
}


def _raw_calls() -> list[str]:
    root = Path(inspect.getfile(controls)).resolve().parent
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "controls.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "imgui"
                and (
                    node.func.attr in _RAW_WIDGETS
                    or node.func.attr.startswith(("input_", "drag_", "slider_"))
                )
            ):
                found.append(f"{path.relative_to(root)}:{node.lineno}:{node.func.attr}")
    return found


def test_record_is_a_no_op_when_the_probe_is_not_enabled():
    assert probe.ENABLED is False, "the suite must never run with WARLOCK_UI_PROBE=1"
    probe.begin_frame()
    probe.record(label="Generate", kind="button")
    assert probe.FRAME_CONTROLS == []


def test_record_refuses_an_untyped_row_even_when_enabled(monkeypatch):
    # A census of rows with no ``kind`` is a census nobody can act on: the
    # driver picks a click strategy from it.
    monkeypatch.setattr(probe, "ENABLED", True)
    probe.begin_frame()
    probe.record(label="Generate", kind="")
    assert probe.FRAME_CONTROLS == []


def test_record_reads_the_rect_rather_than_computing_one(monkeypatch):
    """The rect must come from imgui, for ``anchors``' reason one level down."""

    class _Vec:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Imgui:
        def get_current_context(self):
            return object()

        def get_item_rect_min(self):
            return _Vec(10.0, 20.0)

        def get_item_rect_max(self):
            return _Vec(60.0, 44.0)

        def is_item_visible(self):
            return True

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _Imgui())
    probe.begin_frame()
    probe.record(label="Bucket##inker/tool", kind="button", selected=True)
    (one,) = probe.FRAME_CONTROLS
    assert one.rect == (10.0, 20.0, 50.0, 24.0)
    assert one.centre == (35.0, 32.0)
    assert one.text == "Bucket"
    assert one.selected is True
    assert one.kind == "button"


def test_the_window_is_read_at_submission_time(monkeypatch):
    """``FRAME_PANES`` misses a workspace that opens a pane directly."""

    class _Vec:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Window:
        name = "##host/##content_F65/inker-timeline_9AAF9373"

    class _Internal:
        def get_current_window(self):
            return _Window()

    class _Imgui:
        internal = _Internal()

        def get_current_context(self):
            return object()

        def get_item_rect_min(self):
            return _Vec(0.0, 0.0)

        def get_item_rect_max(self):
            return _Vec(24.0, 24.0)

        def is_item_visible(self):
            return True

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _Imgui())
    probe.begin_frame()
    probe.record(label="1", kind="button")
    one = probe.FRAME_CONTROLS[0]
    assert one.window == "inker-timeline"
    # And ``where`` prefers a resolved pane but never comes back empty.
    assert one.where == "inker-timeline"


def test_an_icon_control_is_named_by_its_tooltip(monkeypatch):
    """A tool button's label is one private-use codepoint from lucide."""

    class _Vec:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Imgui:
        def get_current_context(self):
            return object()

        def get_item_rect_min(self):
            return _Vec(0.0, 0.0)

        def get_item_rect_max(self):
            return _Vec(24.0, 24.0)

        def is_item_visible(self):
            return True

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _Imgui())
    probe.begin_frame()
    probe.record(label="##inker/tool/bucket", kind="button", tooltip="Bucket")
    assert probe.FRAME_CONTROLS[0].name == "Bucket"
    probe.begin_frame()
    probe.record(label="Add layer", kind="button", tooltip="Add a layer above")
    assert probe.FRAME_CONTROLS[0].name == "Add layer"


def test_the_pane_is_resolved_by_census_not_by_record(monkeypatch):
    """``layout`` records a pane when it *closes*, so ``record`` is too early.

    Attributing at submission time put every control in a pane that had not
    finished drawing yet -- which is to say, in no pane at all.
    """
    from warlock.studio import layout

    class _Vec:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Imgui:
        def get_current_context(self):
            return object()

        def get_item_rect_min(self):
            return _Vec(110.0, 210.0)

        def get_item_rect_max(self):
            return _Vec(150.0, 234.0)

        def is_item_visible(self):
            return True

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _Imgui())
    monkeypatch.setitem(layout.FRAME_PANES, "inker_tools", (100.0, 200.0, 200.0, 400.0))
    probe.begin_frame()
    probe.record(label="Bucket", kind="button")
    assert probe.FRAME_CONTROLS[0].pane == ""
    assert probe.census()[0].pane == "inker_tools"


def test_record_survives_having_no_imgui_context(monkeypatch):
    class _Imgui:
        def get_current_context(self):
            return None

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _Imgui())
    probe.begin_frame()
    probe.record(label="Generate", kind="button")
    assert probe.FRAME_CONTROLS == []


def test_begin_frame_clears_the_previous_frames_census(monkeypatch):
    monkeypatch.setattr(
        probe,
        "FRAME_CONTROLS",
        [probe.Control(label="stale", kind="button", rect=(0, 0, 1, 1))],
    )
    probe.begin_frame()
    assert probe.FRAME_CONTROLS == []


def test_the_frame_clears_the_census_where_it_clears_the_others():
    """Beside ``anchors``' clear, so the two are visibly one decision."""

    source = Path(inspect.getfile(importlib.import_module("warlock.studio.main")))
    text = source.read_text(encoding="utf-8")
    assert "probe.begin_frame()" in text
    assert text.index("anchors.begin_frame()") < text.index("probe.begin_frame()")


def test_probe_imports_nothing_the_headless_packages_forbid():
    """It lives in ``studio/`` and imports imgui, so it must stay out of them."""

    root = Path(inspect.getfile(controls)).resolve().parent
    for package in ("inker", "clay", "plotter", "packwright"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = {alias.name for alias in node.names}
                    assert "probe" not in names, path
                    assert (node.module or "").split(".")[-1] != "probe", path
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.endswith("probe"), path


def test_every_finish_item_call_site_names_a_kind():
    """Otherwise the census fills with untyped rows the driver cannot press."""

    tree = ast.parse(Path(inspect.getfile(controls)).read_text(encoding="utf-8"))
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_finish_item"
    ]
    assert sites, "the chokepoint the census is derived from has moved"
    for node in sites:
        kinds = [kw for kw in node.keywords if kw.arg == "kind"]
        assert kinds, f"_finish_item at line {node.lineno} passes no kind"
        assert kinds[0].value is not None


def test_raw_imgui_control_count_is_pinned():
    found = _raw_calls()
    assert len(found) == RAW_IMGUI_CONTROLS, found


class _Vec2:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _LabelledImgui:
    """An imgui whose last item spans a widget *and* its trailing label.

    ``ColorEdit4``, the sliders, ``Checkbox`` and ``BeginCombo`` all wrap the
    widget and its label in one group, so ``get_item_rect_max`` is the right
    edge of the *text*. The measured case is Inker's foreground swatch: a 30 px
    chip at x=76 with "Foreground" beside it, one item 100 px wide.
    """

    class _Style:
        item_inner_spacing = _Vec2(4.0, 4.0)

    def get_current_context(self):
        return object()

    def get_item_rect_min(self):
        return _Vec2(76.0, 92.0)

    def get_item_rect_max(self):
        return _Vec2(176.0, 122.0)

    def is_item_visible(self):
        return True

    def get_style(self):
        return self._Style()

    def calc_text_size(self, text):
        # "Foreground" -- the 66 px that must come off the hit rect.
        return _Vec2(6.6 * len(text), 17.0)


def test_a_trailing_label_is_not_part_of_the_clickable_rect(monkeypatch):
    """The click point must land on the widget, never on the text beside it.

    Measured on 2026-08-23: Inker's ``color_edit4("Foreground", ...)`` records
    a 100 px item whose colour button is only the first 30 px, so the rect
    centre at x=126 landed on the label -- and the exercise pass called a live
    control ``inert`` because a click there does nothing. The label's own width
    is what separates the two.
    """
    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _LabelledImgui())
    probe.begin_frame()
    probe.record(label="Foreground", kind="color_edit4", trailing_label=True)
    (one,) = probe.FRAME_CONTROLS
    # The full item is still reported: it is what says whether imgui clipped it.
    assert one.rect == (76.0, 92.0, 100.0, 30.0)
    # 100 - (66 + 4) = 30, the chip itself.
    assert one.hit == (76.0, 92.0, 30.0, 30.0)
    assert one.centre == (91.0, 107.0)


def test_a_label_drawn_inside_the_widget_is_left_alone(monkeypatch):
    """A button's text is *in* it, so trimming would shrink a live target."""

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _LabelledImgui())
    probe.begin_frame()
    probe.record(label="Swap (X)", kind="button")
    (one,) = probe.FRAME_CONTROLS
    assert one.hit == one.rect
    assert one.centre == (126.0, 107.0)


def test_a_hidden_label_trims_nothing(monkeypatch):
    """``##inkfg`` draws no text, so the item already is the widget."""

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _LabelledImgui())
    probe.begin_frame()
    probe.record(label="##inkfg", kind="color_edit4", trailing_label=True)
    (one,) = probe.FRAME_CONTROLS
    assert one.hit == one.rect


def test_a_label_wider_than_its_item_trims_nothing(monkeypatch):
    """Never trim to nothing: a zero-width hit rect reads as clipped."""

    monkeypatch.setattr(probe, "ENABLED", True)
    monkeypatch.setattr(probe, "imgui", _LabelledImgui())
    probe.begin_frame()
    probe.record(
        label="a label longer than the hundred pixels the item has",
        kind="slider_int",
        trailing_label=True,
    )
    (one,) = probe.FRAME_CONTROLS
    assert one.hit == one.rect


def test_every_field_call_reports_its_trailing_label():
    """The fields imgui draws a label beside must say so, or the trim is dead."""

    tree = ast.parse(Path(inspect.getfile(controls)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in (
            "_field_call",
            "combo",
            "radio_button",
        ):
            continue
        sites = [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_finish_item"
        ]
        assert sites, f"{node.name} no longer reaches the census"
        for call in sites:
            flags = [kw for kw in call.keywords if kw.arg == "trailing_label"]
            assert flags, f"{node.name} passes no trailing_label"
