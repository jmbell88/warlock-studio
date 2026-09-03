"""Flourish's two surfaces: the insert popup and the effect inspector.

Drawn only when they apply. The popup opens from the **Flourish** menu
(``inker_ops`` names it, ``inker_bridge.popups`` opens it -- the same door
every other Inker dialog uses). The inspector draws under the timeline's
transport when the active layer is inside an effect group, and nowhere else:
an ordinary drawing has nothing for it to say, and a panel that appears only
for the thing it describes is the rule the sheet strip already follows.

**No control here writes the document.** Every slider and field writes the
*pending* recipe in ``inker_state`` and starts a short clock; when the value
has rested, ``inker_flourish.tick`` submits one render, and the document is
written once when it lands -- one gesture, one undo step, and the frame loop
never waits on a bake. Because nothing is pushed while dragging, there is no
``fold_undo`` to call: the fold happens by construction.

Parameters are drawn from each primitive's own ``PARAMS`` table, so a new
primitive gets an inspector without touching this file. A curve is shown as
its first and last key -- what it starts at and what it ends at -- which is
the two numbers a user reaches for; the keys in between keep their shape.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from imgui_bundle import imgui

from .. import controls, inker_flourish, inker_mode, inker_ops, widgets
from ..inker.flourish import curves as flourish_curves
from ..inker.flourish import engines, presets, prims
from ..inker.flourish import recipe as flourish_recipe
from ..manual import render as manual_render
from ..tokens import sp

MODE_OPTIONS = (("painterly", "Painterly"), ("pixel", "Pixel art"))
ENGINE_OPTIONS = tuple((e, e) for e in engines.ENGINES)
FLOURISH_POPUP = inker_flourish.FLOURISH_POPUP
SNIPPET_POPUP = inker_flourish.SNIPPET_POPUP
TEXTURE_POPUP = inker_flourish.TEXTURE_POPUP
RESTYLE_POPUP = inker_flourish.RESTYLE_POPUP
DIRECTION_OPTIONS = (("1", "One direction"), ("4", "Four directions"), ("8", "Eight directions"))


# -- the insert popup ------------------------------------------------------------------


def open_popup(ctx: Any, tab: Any) -> None:
    """``inker_bridge.popups``' door: the menu asked, the tab is open."""
    if tab is None or tab.busy:
        return
    imgui.open_popup(inker_flourish.FLOURISH_POPUP)


def popup(ctx: Any, tab: Any) -> None:
    state = inker_mode.ensure(ctx)
    if not imgui.begin_popup(inker_flourish.FLOURISH_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    widgets.section("Insert an effect")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-flourish")
    names = presets.names()
    options = [(name, presets.label(name)) for name in names]
    if state.flourish_preset not in names and names:
        state.flourish_preset = names[0]
    state.flourish_preset = widgets.labeled_combo("Effect", state.flourish_preset, options)
    state.flourish_mode = widgets.labeled_combo("Look", state.flourish_mode, list(MODE_OPTIONS))
    picked = widgets.labeled_combo(
        "Facings", str(state.flourish_directions), list(DIRECTION_OPTIONS)
    )
    state.flourish_directions = int(picked)
    widgets.muted_wrapped(
        "The effect lands as a layer group above the active layer, one tag per "
        "phase. Every parameter can be changed afterwards from the inspector "
        "under the timeline."
    )
    imgui.dummy((0, 4))
    busy = tab is None or tab.busy or ctx.busy(inker_flourish.insert_key(tab))
    if controls.button("Insert", (sp(90), 0), enabled=not busy, reason=inker_flourish.BUSY):
        inker_mode.flourish_insert(
            ctx,
            tab,
            preset=state.flourish_preset,
            mode=state.flourish_mode,
            directions=state.flourish_directions,
        )
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel", (sp(90), 0)):
        imgui.close_current_popup()
    imgui.end_popup()


# -- the engine snippet popup ----------------------------------------------------------------


def open_snippet_popup(ctx: Any, tab: Any) -> None:
    if tab is None or tab.busy:
        return
    imgui.open_popup(inker_flourish.SNIPPET_POPUP)


def snippet_popup(ctx: Any, tab: Any) -> None:
    state = inker_mode.ensure(ctx)
    if not imgui.begin_popup(inker_flourish.SNIPPET_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    widgets.section("Engine snippet")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-flourish")
    tags = inker_flourish.tag_names(tab) if tab is not None else []
    if not tags:
        widgets.muted("This document has no tags.")
        if controls.button("Close", (sp(90), 0)):
            imgui.close_current_popup()
        imgui.end_popup()
        return
    if state.flourish_snippet_tag not in tags:
        state.flourish_snippet_tag = tags[0]
    if state.flourish_snippet_engine not in engines.ENGINES:
        state.flourish_snippet_engine = engines.ENGINES[0]
    state.flourish_snippet_tag = widgets.labeled_combo(
        "Phase", state.flourish_snippet_tag, [(t, t) for t in tags]
    )
    state.flourish_snippet_engine = widgets.labeled_combo(
        "Engine", state.flourish_snippet_engine, list(ENGINE_OPTIONS)
    )
    text = inker_flourish.snippet_text(
        tab, state.flourish_snippet_tag, state.flourish_snippet_engine
    )
    imgui.set_next_item_width(sp(520))
    controls.input_text_multiline(
        "##fl-snippet", text, (sp(520), sp(260)), imgui.InputTextFlags_.read_only.value
    )
    widgets.muted_wrapped(
        "Assumes the per-tag sheet export's filenames and an origin at the canvas "
        "centre, which is where an effect is placed."
    )
    if controls.button("Copy", (sp(90), 0)):
        imgui.set_clipboard_text(text)
        ctx.toast("Snippet copied.", "success")
    imgui.same_line()
    if controls.button("Close", (sp(90), 0)):
        imgui.close_current_popup()
    imgui.end_popup()


# -- the texture popup ----------------------------------------------------------------------


def open_texture_popup(ctx: Any, tab: Any) -> None:
    if tab is None or tab.busy:
        return
    imgui.open_popup(inker_flourish.TEXTURE_POPUP)


def texture_popup(ctx: Any, tab: Any) -> None:
    state = inker_mode.ensure(ctx)
    if not imgui.begin_popup(inker_flourish.TEXTURE_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    widgets.section("Generate a texture")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-flourish")
    imgui.set_next_item_width(sp(320))
    _c, subject = controls.input_text_with_hint(
        "##fl-texture-subject",
        "a skull-shaped ember, a rune, a flame",
        state.flourish_texture_subject,
    )
    state.flourish_texture_subject = subject
    widgets.muted_wrapped(
        "One centred ingredient on black; the black is keyed out into the "
        "texture's alpha. It lands on the layer the inspector is showing when "
        "that layer can take a texture (a sprite, or particles)."
    )
    pending = state.flourish_texture_pending is not None
    busy = tab is None or tab.busy or pending
    reason = inker_flourish.TEXTURE_PENDING if pending else inker_flourish.BUSY
    if controls.button("Generate", (sp(100), 0), enabled=not busy, reason=reason):
        inker_mode.flourish_texture_generate(ctx, tab, subject=subject)
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel", (sp(90), 0)):
        imgui.close_current_popup()
    imgui.end_popup()


# -- the restyle popup ------------------------------------------------------------------------


def open_restyle_popup(ctx: Any, tab: Any) -> None:
    if tab is None or tab.busy:
        return
    imgui.open_popup(inker_flourish.RESTYLE_POPUP)


def restyle_popup(ctx: Any, tab: Any) -> None:
    state = inker_mode.ensure(ctx)
    if not imgui.begin_popup(inker_flourish.RESTYLE_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    widgets.section("Restyle keyframes")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-flourish")
    phases = inker_flourish.phase_names(state, tab) if tab is not None else []
    if not phases:
        widgets.muted("The active layer is not part of an effect.")
        if controls.button("Close", (sp(90), 0)):
            imgui.close_current_popup()
        imgui.end_popup()
        return
    if state.flourish_restyle_phase not in phases:
        state.flourish_restyle_phase = phases[0]
    state.flourish_restyle_phase = widgets.labeled_combo(
        "Phase", state.flourish_restyle_phase, [(p, p) for p in phases]
    )
    imgui.set_next_item_width(sp(320))
    _c, subject = controls.input_text_with_hint(
        "##fl-restyle-subject",
        "a painted fireball, thick brush strokes",
        state.flourish_restyle_subject,
    )
    state.flourish_restyle_subject = subject
    imgui.set_next_item_width(sp(200))
    _c, strength = controls.slider_float(
        "strength##fl-restyle", float(state.flourish_restyle_strength), 0.1, 0.95, "%.2f"
    )
    state.flourish_restyle_strength = float(strength)
    imgui.set_next_item_width(sp(200))
    _c, anchors = controls.slider_int(
        "keyframes##fl-restyle", int(state.flourish_restyle_anchors), 2, 6
    )
    state.flourish_restyle_anchors = int(anchors)
    widgets.muted_wrapped(
        "The keyframes go through the image model one at a time; the frames "
        "between are crossfaded under the effect's own motion. The result is a "
        "new layer inside the group -- the procedural layers stay. Unmeasured: "
        "judge it before you keep it."
    )
    pending = state.flourish_restyle_pending is not None
    busy = tab is None or tab.busy or pending
    reason = inker_flourish.RESTYLE_PENDING if pending else inker_flourish.BUSY
    if controls.button("Start", (sp(100), 0), enabled=not busy, reason=reason):
        inker_mode.flourish_restyle(
            ctx,
            tab,
            phase=state.flourish_restyle_phase,
            subject=subject,
            strength=float(strength),
            anchors=int(anchors),
        )
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel", (sp(90), 0)):
        imgui.close_current_popup()
    imgui.end_popup()


# -- the inspector -----------------------------------------------------------------------


def draw_inspector(ctx: Any, tab: Any) -> None:
    """Under the transport, for the active effect group. Also the frame-thread
    tick that submits rested edits, which is why it runs before the early
    return: a render owed to a group the user has just clicked away from is
    still owed."""
    state = ctx.state.inker
    if state is None or tab is None or tab.doc.anim is None:
        return
    now = inker_flourish.clock()
    inker_flourish.tick(ctx, state, tab, now=now)
    inker_flourish.poll_texture(ctx, state, now=now)
    inker_flourish.poll_restyle(ctx, state, now=now)
    group = inker_flourish.active_group(state, tab)
    if group is None:
        return
    recipe = inker_flourish.current_recipe(state, tab, group)
    if recipe is None:
        return
    widgets.section(f"Flourish: {recipe.name}")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-flourish")
    imgui.same_line()
    _status(ctx, state, tab, group)

    _press(ctx, "flourish_regenerate", "Regenerate")
    imgui.same_line()
    _press(ctx, "flourish_keep_edits", "Keep painted cels")
    imgui.same_line()
    _press(ctx, "flourish_regenerate_all", "Replace painted cels")
    imgui.same_line()
    _press(ctx, "flourish_detach", "Detach")

    imgui.set_next_item_width(sp(300))
    entered, words = controls.input_text_with_hint(
        "##fl-prompt",
        "colder, more sparks, no smoke...",
        state.flourish_prompt_text,
        imgui.InputTextFlags_.enter_returns_true.value,
    )
    state.flourish_prompt_text = words
    imgui.same_line()
    pressed = _press(ctx, "flourish_prompt", "Apply words")
    if entered and not pressed:
        inker_ops.run(ctx, inker_ops.get("flourish_prompt"))
    imgui.same_line()
    config = getattr(getattr(ctx, "svc", None), "config", None)
    widgets.muted("model" if inker_flourish.text_model_available(config) else "keywords")

    changed = False
    imgui.set_next_item_width(sp(120))
    seed_changed, seed = controls.input_int("Seed##fl", int(recipe.seed), 1, 10)
    if seed_changed:
        recipe = replace(recipe, seed=max(0, int(seed)))
        changed = True
    imgui.same_line()
    imgui.set_next_item_width(sp(120))
    fps_changed, fps = controls.slider_int("fps##fl", int(recipe.fps), 1, 60)
    if fps_changed:
        recipe = replace(recipe, fps=int(fps))
        changed = True
    imgui.same_line()
    mode_changed, mode = controls.combo("##fl-mode", recipe.mode, list(MODE_OPTIONS))
    if mode_changed:
        recipe = replace(recipe, mode=mode)
        changed = True
    if recipe.mode == "pixel":
        imgui.same_line()
        imgui.set_next_item_width(sp(120))
        col_changed, colors = controls.slider_int("colours##fl", int(recipe.colors), 2, 64)
        if col_changed:
            recipe = replace(recipe, colors=int(colors))
            changed = True

    recipe, phases_changed = _phases(recipe)
    changed = changed or phases_changed
    recipe, layer_changed = _layer_block(state, group, recipe)
    changed = changed or layer_changed

    if changed:
        inker_flourish.set_pending(state, group, flourish_recipe.clamp(recipe), now=now)


def _status(ctx: Any, state: Any, tab: Any, group: int) -> None:
    if inker_flourish.in_flight(ctx, tab, group):
        progress = ctx.progress(inker_flourish.render_key(tab, group)) or {}
        widgets.muted(f"rendering {progress.get('label', '')}".rstrip())
        return
    if group in state.flourish_pending:
        widgets.muted("edited, render pending")
        return
    flagged = tab.doc.flourish_conflicts(group)
    if flagged:
        widgets.muted(f"{len(flagged)} painted cell(s) flagged")
    else:
        widgets.muted("up to date")


def _press(ctx: Any, name: str, label: str) -> bool:
    state = ctx.state.inker
    op = inker_ops.get(name)
    tab = state.active
    enabled = op.enabled(state, tab)
    reason = inker_ops.reason_for(op, state, tab) if not enabled else ""
    if controls.button(f"{label}##fl-{name}", enabled=enabled, reason=reason, tooltip=op.hint):
        return inker_ops.run(ctx, op)
    return False


def _phases(recipe: Any) -> tuple[Any, bool]:
    changed = False
    phases = list(recipe.phases)
    for i, phase in enumerate(phases):
        imgui.set_next_item_width(sp(110))
        f_changed, frames = controls.slider_int(
            f"{phase.name}##fl-phase-{i}", int(phase.frames), 1, 60
        )
        imgui.same_line()
        l_changed, loop = controls.checkbox(f"loop##fl-loop-{i}", bool(phase.loop))
        if f_changed or l_changed:
            phases[i] = replace(phase, frames=int(frames), loop=bool(loop))
            changed = True
    if changed:
        recipe = replace(recipe, phases=tuple(phases))
    return recipe, changed


def _layer_block(state: Any, group: int, recipe: Any) -> tuple[Any, bool]:
    if not recipe.layers:
        widgets.muted("This effect has no layers.")
        return recipe, False
    uids = [layer.uid for layer in recipe.layers]
    current = state.flourish_layer.get(group)
    if current not in uids:
        current = uids[-1]
    options = [
        (str(layer.uid), f"{layer.name} ({layer.kind})") for layer in reversed(recipe.layers)
    ]
    imgui.set_next_item_width(sp(220))
    _c, picked = controls.combo("##fl-layer", str(current), options, tooltip="Which layer to edit.")
    current = int(picked)
    state.flourish_layer[group] = current
    layer = recipe.layer(current)
    changed = False

    imgui.same_line()
    v_changed, visible = controls.checkbox("visible##fl-vis", bool(layer.visible))
    if v_changed:
        layer = replace(layer, visible=bool(visible))
        changed = True
    imgui.same_line()
    imgui.set_next_item_width(sp(100))
    o_changed, opacity = controls.slider_float(
        "opacity##fl-op", float(layer.opacity), 0.0, 1.0, "%.2f"
    )
    if o_changed:
        layer = replace(layer, opacity=float(opacity))
        changed = True

    specs = prims.params_of(layer.kind)
    held = state.active.doc.flourish_state(group) if state.active is not None else None
    asset_ids = list(held.assets) if held is not None else []
    for name, spec in specs.items():
        if spec.kind == "asset":
            new_value, p_changed = _asset_control(name, layer.params.get(name, ""), asset_ids)
        else:
            new_value, p_changed = _param_control(name, spec, layer.params.get(name, spec.default))
        if p_changed:
            layer = layer.with_param(name, new_value)
            changed = True
    if changed:
        recipe = recipe.replace_layer(layer)
    return recipe, changed


def _param_control(name: str, spec: Any, value: Any) -> tuple[Any, bool]:
    label = f"{name}##fl-p-{name}"
    tip = spec.label or ""
    imgui.set_next_item_width(sp(160))
    if spec.kind == "float":
        changed, got = controls.slider_float(
            label, float(value), float(spec.lo), float(spec.hi), "%.2f", tooltip=tip
        )
        return float(got), bool(changed)
    if spec.kind == "int":
        changed, got = controls.slider_int(
            label, int(value), int(spec.lo), int(spec.hi), tooltip=tip
        )
        return int(got), bool(changed)
    if spec.kind == "bool":
        changed, got = controls.checkbox(label, bool(value), tooltip=tip)
        return bool(got), bool(changed)
    if spec.kind == "choice":
        changed, got = controls.combo(
            label, str(value), [(c, c) for c in spec.choices], tooltip=tip
        )
        return str(got), bool(changed)
    if spec.kind == "color":
        rgba = prims.parse_color(str(value))
        changed, got = controls.color_edit4(
            label, [float(c) for c in rgba], imgui.ColorEditFlags_.no_inputs.value, tooltip=tip
        )
        if changed:
            return "#" + "".join(f"{int(round(c * 255)):02X}" for c in got), True
        return value, False
    # A curve over the phase or over a particle's life: its first and last key.
    curve = flourish_curves.Curve.from_json(value)
    first, last = curve.keys[0][1], curve.keys[-1][1]
    lo, hi = float(spec.lo), float(spec.hi)
    imgui.set_next_item_width(sp(76))
    a_changed, a = controls.slider_float(
        f"##fl-c0-{name}", float(first), lo, hi, "%.1f", tooltip=f"{name} at the start"
    )
    imgui.same_line()
    imgui.set_next_item_width(sp(76))
    b_changed, b = controls.slider_float(
        f"##fl-c1-{name}", float(last), lo, hi, "%.1f", tooltip=f"{name} at the end"
    )
    imgui.same_line()
    widgets.muted(name if not tip else f"{name} ({tip})")
    if not (a_changed or b_changed):
        return value, False
    if len(curve.keys) == 1:
        keys = ((0.0, float(a)),) if not b_changed or a == b else ((0.0, float(a)), (1.0, float(b)))
    else:
        keys = ((curve.keys[0][0], float(a)), *curve.keys[1:-1], (curve.keys[-1][0], float(b)))
    return flourish_curves.Curve(keys, curve.easing).to_json(), True


def _asset_control(name: str, value: Any, asset_ids: list[str]) -> tuple[Any, bool]:
    options = [("", "(none)"), *((a, a) for a in asset_ids)]
    current = str(value or "") if str(value or "") in asset_ids else ""
    imgui.set_next_item_width(sp(160))
    changed, got = controls.combo(
        f"{name}##fl-p-{name}",
        current,
        options,
        tooltip="A texture of this effect: from the selection, or generated.",
    )
    return str(got), bool(changed)
