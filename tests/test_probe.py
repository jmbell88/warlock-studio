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
