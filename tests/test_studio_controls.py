"""Studio design-system contracts that do not need a GL context."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from warlock.studio import component_gallery, controls, forms, layout, theme, tokens, toolbar
from warlock.studio.panes import landing, overlay


def test_control_sizes_follow_the_display_scale():
    before = tokens.SCALE
    try:
        tokens.set_scale(1.75)
        assert controls.control_height(controls.ControlSize.REGULAR) == pytest.approx(52.5)
        assert controls.control_height(controls.ControlSize.COMPACT) == pytest.approx(45.5)
    finally:
        tokens.set_scale(before)


@pytest.mark.parametrize("scale", [1.0, 1.5, 1.75])
def test_form_layout_uses_a_design_pixel_breakpoint(scale):
    narrow = forms.adaptive_layout((forms.FORM_BREAKPOINT - 1) * scale, scale=scale)
    wide = forms.adaptive_layout(forms.FORM_BREAKPOINT * scale, scale=scale)
    assert narrow.stacked and narrow.label_width == 0
    assert not wide.stacked
    assert wide.label_width == pytest.approx(forms.LABEL_WIDTH * scale)


def test_sentence_case_normalises_legacy_labels_without_losing_acronyms():
    assert forms.sentence_case("OUTPUT SIZE") == "Output size"
    assert forms.sentence_case("Style LoRA") == "Style LoRA"
    assert forms.sentence_case("mesh_seed") == "Mesh seed"


def test_every_palette_carries_a_subtle_dedicated_divider():
    for palette in tokens.PALETTES.values():
        assert "DIVIDER" in palette
        ratio = tokens.contrast(palette["DIVIDER"], palette["BG"])
        assert 1.02 < ratio < tokens.CONTRAST_UI


def test_toolbar_items_use_roles_and_explicit_selection():
    item = toolbar.Item(
        "apply", "Apply", role=controls.ButtonRole.PRIMARY, selected=True
    )
    assert item.role is controls.ButtonRole.PRIMARY
    assert item.selected is True
    # Constructor-only compatibility does not change the model the renderer reads.
    assert toolbar.Item("delete", "Delete", danger=True).role is controls.ButtonRole.DESTRUCTIVE


def test_document_actions_have_one_shared_order():
    assert controls.DOCUMENT_ACTION_ORDER == (
        "New",
        "Open",
        "Save",
        "Save As",
        "Export",
    )


def test_buttons_expose_the_shared_disabled_reason_contract():
    parameters = inspect.signature(controls.button).parameters
    assert {"role", "control_size", "selected", "enabled", "reason", "tooltip"} <= set(
        parameters
    )


def test_shell_issue_summary_is_compact_and_singular_or_plural():
    assert overlay.doctor_summary(["  Install   the model. "]) == (
        "1 setup issue",
        "Install the model.",
    )
    assert overlay.doctor_summary(["A", "B"])[0] == "2 setup issues"


def test_home_suppresses_its_health_row_while_the_shell_summary_is_visible():
    health = landing.Status("health", "!", "2 things need attention", 0)
    queue = landing.Status("queue", "o", "Queue idle", 0)
    assert landing.visible_home_rows(
        [health, queue], shell_issue_visible=True
    ) == [queue]
    assert landing.visible_home_rows(
        [health, queue], shell_issue_visible=False
    ) == [health, queue]


def test_component_gallery_is_developer_only():
    assert not component_gallery.enabled({})
    assert component_gallery.enabled({component_gallery.ENV_KEY: "true"})


# Every palette that ships, not a literal pair: a third one was added without
# this list noticing, and a gallery that renders under two of three palettes
# is exactly the coverage this test exists to deny.
@pytest.mark.parametrize("palette", sorted(tokens.PALETTES))
@pytest.mark.parametrize("scale", [1.0, 1.5, 1.75])
def test_component_gallery_builds_every_state(
    gl, monkeypatch, palette, scale
):
    from imgui_bundle import imgui

    from warlock.studio import imgui_backend

    # Saved and put back below. ``destroy_context()`` leaves *no* current
    # context, so a test that builds its own over the session-scoped
    # ``imgui_ctx`` orphans it -- every later test that draws then fails on
    # "No current context", in files that have nothing to do with this one.
    # It used to be safe only by collection order, which stopped being true the
    # moment a second file needed the shared fixture. ``test_poser_panes_smoke``
    # has always done this; this is the same three lines.
    prev_ctx = imgui.get_current_context()
    prev_screen = type(gl).__dict__.get("screen")
    fbo = gl.simple_framebuffer((1600, 1000))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)
    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600, 1000)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    renderer = imgui_backend.ImguiRenderer(gl)
    old_theme, old_scale = tokens.THEME, tokens.SCALE
    monkeypatch.setenv(component_gallery.ENV_KEY, "1")
    try:
        tokens.set_theme(palette)
        tokens.set_scale(scale)
        theme.apply(imgui)
        imgui.new_frame()
        imgui.set_next_window_size((1600, 1000))
        imgui.begin(f"##gallery-host-{palette}-{scale}")
        component_gallery.request()
        component_gallery.draw()
        imgui.end()
        imgui.render()
        renderer.render(imgui.get_draw_data())
    finally:
        tokens.set_theme(old_theme)
        tokens.set_scale(old_scale)
        renderer.shutdown()
        imgui.destroy_context()
        if prev_screen is not None:
            type(gl).screen = prev_screen
        if prev_ctx is not None:
            imgui.set_current_context(prev_ctx)


def test_major_panes_have_roles_and_no_production_pane_child_calls():
    assert layout.PaneRole.SIDEBAR.value == "sidebar"
    assert layout.PaneRole.INSPECTOR.value == "inspector"
    root = Path(inspect.getfile(layout)).resolve().parent
    sources = [root / "main.py", *(root / "panes").glob("*.py")]
    for path in sources:
        assert ".pane_child(" not in path.read_text(encoding="utf-8"), path.name


FORBIDDEN = {
    "button",
    "small_button",
    "input_text",
    "input_text_multiline",
    "input_int",
    "input_float",
    "drag_int",
    "drag_float",
    "slider_int",
    "slider_float",
    "checkbox",
    "radio_button",
    "selectable",
    "collapsing_header",
    "menu_item",
    "menu_item_simple",
}


def test_panes_do_not_bypass_the_presentational_control_layer():
    pane_dir = Path(inspect.getfile(overlay)).resolve().parent
    found: list[str] = []
    for path in pane_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "imgui"
                and (
                    node.func.attr in FORBIDDEN
                    or node.func.attr.startswith(("input_", "drag_", "slider_"))
                )
            ):
                found.append(f"{path.name}:{node.lineno}:{node.func.attr}")
    assert not found, found


def test_divider_role_resolves_live_through_theme():
    before = tokens.THEME
    try:
        tokens.set_theme("light")
        assert tokens.PALETTES["light"]["DIVIDER"] == theme.DIVIDER
    finally:
        tokens.set_theme(before)
