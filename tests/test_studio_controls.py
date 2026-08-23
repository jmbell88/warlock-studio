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


def test_every_palette_carries_the_tours_scrim_and_ring():
    """The checkerboard's lesson, applied before it costs anything.

    Those two lived in the dark palette's range only, so a light-theme session
    drew a near-black checker under a white window -- found by a screenshot,
    because nothing compared the palettes to each other. A tour highlight is the
    same shape of thing: colour drawn over whatever is on screen, which is a
    different colour in every theme.

    Measured on the *blend*, for ``composite``'s own reason -- the scrim is
    drawn translucent, so the number that matters is the one on the glass. On a
    light palette the token is darker than the text it sits near, which is why
    a plain token-against-token contrast check reads backwards there and says
    nothing about whether the screen actually dimmed.
    """
    from warlock.studio.panes.tour import VEIL_ALPHA

    for name, palette in tokens.PALETTES.items():
        assert "TOUR_VEIL" in palette, f"{name} has no tour scrim"
        assert "TOUR_RING" in palette, f"{name} has no tour ring"
        dimmed = tokens.composite(palette["TOUR_VEIL"], palette["BG"], VEIL_ALPHA)
        assert tokens.luminance(dimmed) < tokens.luminance(palette["BG"]), (
            f"{name}: the scrim does not darken this palette's background"
        )
        assert tokens.contrast(palette["TOUR_RING"], dimmed) >= tokens.CONTRAST_UI, (
            f"{name}: the ring does not clear the control boundary on its own scrim"
        )


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


def test_home_never_draws_the_health_row():
    """It was the third rendering of one fact -- after the rail's badge and the
    doctor banner -- and the suppression that used to hide it behind the banner
    was this argument made halfway."""
    health = landing.Status("health", "!", "2 things need attention", 0)
    queue = landing.Status("queue", "o", "Queue idle", 0)
    assert landing.visible_home_rows([health, queue], shell_issue_visible=True) == [queue]
    assert landing.visible_home_rows([health, queue], shell_issue_visible=False) == [queue]


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


def test_a_typed_slider_value_is_clamped_to_the_range_it_draws():
    """Ctrl+click-to-type always worked; nothing bounded what was typed.

    Injected in ``_field_call``, the one chokepoint every field passes
    through, so no call site changes and none can forget.
    """
    from imgui_bundle import imgui as real_imgui

    args, kwargs = controls._clamp_typed_entry("slider_int", ("##x", 3, 0, 10), {})
    assert kwargs["flags"] == real_imgui.SliderFlags_.clamp_on_input.value
    assert args == ("##x", 3, 0, 10)
    # Drags take it too -- typing past a drag's soft range is the same defect.
    _args, kwargs = controls._clamp_typed_entry("drag_float", ("##x", 1.0), {})
    assert "flags" in kwargs


def test_the_clamp_is_never_always_clamp_and_never_reaches_a_text_field():
    """Two constraints that must not be "simplified" into one.

    ``always_clamp`` includes ``clamp_zero_range``, which clamps a drag whose
    ``v_min == v_max`` -- and that is exactly how ``settings_3d`` spells
    *unbounded* (``SIZE_NO_BOUND = (0.0, 0.0)``), so it would pin the asset's
    size to zero. And ``input_int``'s sixth positional is an
    ``InputTextFlags``, not a ``SliderFlags``: injecting there would switch on
    an unrelated text-field option.
    """
    # The *assignment*, not the prose: this file's style is to state a
    # rejected alternative by name, so a raw scan would fail on its own note.
    body = [
        line
        for line in inspect.getsource(controls._clamp_typed_entry).splitlines()
        if "SliderFlags_" in line
    ]
    assert body and all("clamp_on_input" in line for line in body)
    assert not any("always_clamp" in line for line in body)
    for name in ("input_int", "input_float", "input_text", "checkbox"):
        args, kwargs = controls._clamp_typed_entry(name, ("##x", 1), {})
        assert kwargs == {} and args == ("##x", 1), name
    # A caller that spelled its own flags keeps them, whole.
    _args, kwargs = controls._clamp_typed_entry("slider_int", ("##x", 3, 0, 10), {"flags": 7})
    assert kwargs == {"flags": 7}
    # And so does one that passed them positionally.
    args, kwargs = controls._clamp_typed_entry("slider_int", ("##x", 3, 0, 10, "%d", 7), {})
    assert kwargs == {} and args[-1] == 7


def test_every_slider_says_that_a_value_can_be_typed():
    """One string in one place, covering all 166 sites."""
    assert "Ctrl+click" in controls.TYPED_ENTRY_HINT
    body = inspect.getsource(controls._finish_item)
    assert "TYPED_ENTRY_HINT" in body
    assert '("slider_", "drag_")' in body


def test_ctrl_click_cannot_collide_with_a_keyboard_shortcut():
    """The absence of a collision is exactly what gets broken silently.

    ``main._shortcut`` is reached from a ``pygame.KEYDOWN`` branch; ctrl+click
    is a mouse event and never enters that path. Nothing else in the app may
    claim ctrl+click either -- imgui's own text-entry gesture is the only
    reader.
    """
    from warlock.studio import main

    assert "pygame.KEYDOWN" in inspect.getsource(main.App._events)
    assert "config_drag_click_to_input_text" not in inspect.getsource(theme.apply)
