"""Troupe's left-bottom pane: the form that starts a character.

It submits the *first* link of the chain and nothing else: one cheap pose
reference. The gate is the whole point of the shape -- the user approves the
drawing in Create, and only then is the reconstruction spent -- so this pane
deliberately has no "and then build everything" button. What it does have is the
sheet's options, because they ride along on the reference and are validated at
its door: a bad palette found an hour later, on a row the worker minted, would
be a refusal the user never submitted.
"""

from __future__ import annotations

from typing import Any

from ... import rigging
from ...service import troupe as svc_troupe
from .. import forms, troupe_mode, verbs, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = troupe_mode.ensure(ctx)
    widgets.section("New character")
    manual_render.help_button(ctx, "troupe-settings")

    options = _options(ctx)
    form = _form(state, options)
    # ``errors``/``on_edit``: the chain this form starts refuses by name --
    # ``prompt``, ``pose``, ``variant``, ``palette``, ``outline``,
    # ``reduce_mode`` -- and each is a field here, so the refusal had an
    # address and nothing at the other end of it. See ``main._collect_tasks``.
    with forms.Form(
        "troupe-settings",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
        _changed, form["prompt"] = form_ui.text("prompt", "Describe them", form["prompt"])
        _changed, form["variant"] = form_ui.combo(
            "variant",
            "Build",
            form["variant"],
            [(v, v) for v in options.get("variants") or ()],
        )
        # Beside the build, because the two together are the guide the
        # reference is drawn against and neither means much alone.
        _changed, form["pose"] = form_ui.combo(
            "pose",
            "Reference pose",
            form["pose"],
            [
                (name, troupe_mode.POSE_LABELS.get(name, name))
                for name in options.get("poses") or ()
            ],
            help_text=(
                "The stick figure the first drawing is conditioned on. "
                "A-pose matches the rig template, so the joints are fitted "
                "straight to it. T-pose separates the limbs further, which is "
                "what the reconstruction has the least trouble with -- pick it "
                "if the arms come back fused to the body."
            ),
        )
        _layout(form, form_ui, options)
        _size(form, form_ui, options)
        _palette(ctx, form, form_ui, options)
    imgui.dummy((0, 4))
    _submit(ctx, form)
    imgui.dummy((0, 8))
    _existing_mesh(ctx, form)


#: Where the picker's current choice lives. On ``state.preview`` and not on
#: ``TroupeState``: the mode holds a *selection* (which character and which
#: sheet are on screen) and this is neither -- see :func:`_existing_mesh`.
_PICK_SLOT = "troupe_send_mesh"


def _existing_mesh(ctx: Any, form: dict[str, Any]) -> None:
    """Send a mesh you already have, using the settings above.

    Collapsed and *below* the form, sharing it rather than repeating it: the
    layout, size and palette controls above are visibly the settings this will
    use, which is how ``_rebuild`` already words the same relationship. Two
    competing forms in one 300 px column would be the pane asking the same
    questions twice.

    **It never calls** ``troupe_mode.select``, and that is the trap it is
    written around: ``select`` accepts any job id, and ``sheets()`` returns []
    for a bare mesh -- so pointing the mode at one lands on the blank arrival
    ``open_sheet``'s False return exists to prevent. The picker holds a local
    choice and the button calls the same ``send_to_troupe`` the library item
    calls, so nothing points the mode at anything and "Troupe holds a
    selection" stays intact.
    """
    from imgui_bundle import imgui

    if not widgets.header("Or use a mesh you already have", default_open=False):
        return
    meshes = troupe_mode.sendable_meshes(ctx)
    if not meshes:
        widgets.muted("No finished meshes yet. Anything with a mesh can come in here.")
        return
    current = str(ctx.state.preview.get(_PICK_SLOT) or "")
    if current not in {mesh["id"] for mesh in meshes}:
        current = meshes[0]["id"]
    options = [(mesh["id"], _mesh_label(mesh)) for mesh in meshes]
    picked = widgets.labeled_combo("Mesh", current, options)
    if picked != current:
        ctx.state.preview[_PICK_SLOT] = picked
        current = picked
    chosen = next((mesh for mesh in meshes if mesh["id"] == current), None)
    busy = ctx.busy(f"troupe-send:{current}")
    if busy:
        widgets.busy("Sending")
    if widgets.disabled_button(verbs.send_to("troupe"), not busy, (-1, 0)):
        troupe_mode.send_to_troupe(ctx, chosen, form)
    widgets.cost_note(
        "A mesh that is not rigged is rigged first, as a humanoid. Then the "
        f"{cell_count(form)} cells above are rendered."
    )
    imgui.dummy((0, 4))


def _mesh_label(mesh: dict[str, Any]) -> str:
    """A name a person can pick from, never an empty row."""
    text = str(mesh.get("prompt") or "").strip()
    if not text:
        return str(mesh["id"])[:8]
    return text if len(text) <= 48 else f"{text[:47]}..."


def _options(ctx: Any) -> dict[str, Any]:
    """The door's own answer about what may be asked for, read once.

    Cached on the frame state rather than called per draw: it walks the palette
    directory, and a directory walk sixty times a second is a cost with no
    reader. Keyed on nothing, because the only thing that changes it is a file
    the user dropped in -- which the Refresh below re-reads.
    """
    cached = ctx.state.preview.get("troupe_options")
    if cached is None:
        cached = svc_troupe.troupe_options(ctx.svc)
        ctx.state.preview["troupe_options"] = cached
    return cached


def _form(state: Any, options: dict[str, Any]) -> dict[str, Any]:
    """The request, kept on the mode's own state so a trip to Create and back
    does not lose what was typed."""
    defaults = options.get("defaults") or {}
    if not state.form:
        state.form = {
            "prompt": "",
            "variant": str(defaults.get("variant") or "male"),
            "pose": str(defaults.get("pose") or "apose"),
            "logical_size": int(defaults.get("logical_size") or 32),
            "colors": int(defaults.get("colors") or 64),
            "outline": str(defaults.get("outline") or "outer"),
            "reduce_mode": str(defaults.get("reduce_mode") or "box"),
            "dither": False,
            "palette": "",
            "name": "",
            "layout": {
                "version": 2,
                "columns": 8,
                "movements": [
                    {
                        "key": row.get("name"),
                        "enabled": True,
                        "frames": int(row.get("frames") or 1),
                        "directions": 8,
                    }
                    for row in options.get("animations") or ()
                ],
            },
        }
    elif "layout" not in state.form:
        # Session-state migration for a form created by a pre-v2 build.
        state.form["layout"] = {
            "version": 2,
            "columns": 8,
            "movements": [
                {
                    "key": row.get("name"),
                    "enabled": True,
                    "frames": int(row.get("frames") or 1),
                    "directions": 8,
                }
                for row in options.get("animations") or ()
            ],
        }
    return state.form


def _layout(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    """Per-movement frame and direction controls; the total is always derived."""

    layout = form["layout"]
    # ``check_troupe`` refuses the composed sheet with ``field="layout"``, and
    # the layout is not one control -- it is this whole table. So the message
    # goes above it rather than being rung onto an arbitrary row: the reader
    # needs to be pointed at the *set* of switches and counts that add up to
    # the refusal. ``widgets.field_error`` is the same helper the single-control
    # case uses, which keeps the wording and the colour identical.
    form_ui.note("layout")
    limits = {row["name"]: row for row in options.get("animations") or ()}
    presets = [int(n) for n in options.get("direction_presets") or (1, 4, 8, 16)]
    for movement in layout.get("movements") or ():
        key = str(movement.get("key") or "")
        label = key.replace("_", " ").title()
        _changed, movement["enabled"] = form_ui.switch(
            f"movement_{key}", label, bool(movement.get("enabled", True))
        )
        if not movement["enabled"]:
            continue
        _changed, frames = form_ui.number(
            f"frames_{key}",
            f"{label} frames",
            int(movement.get("frames") or limits.get(key, {}).get("frames") or 1),
            helper=(
                f"{limits.get(key, {}).get('min_frames', 1)}-"
                f"{limits.get(key, {}).get('max_frames', 32)} frames"
            ),
        )
        movement["frames"] = max(
            int(limits.get(key, {}).get("min_frames") or 1),
            min(int(frames), int(limits.get(key, {}).get("max_frames") or 32)),
        )
        _changed, directions = form_ui.combo(
            f"directions_{key}",
            f"{label} directions",
            str(movement.get("directions") or 8),
            [(str(n), f"{n}-direction") for n in presets],
        )
        movement["directions"] = int(directions)


def cell_count(form: dict[str, Any]) -> int:
    return sum(
        int(row.get("frames") or 0) * int(row.get("directions") or 0)
        for row in (form.get("layout") or {}).get("movements") or ()
        if row.get("enabled", True)
    )


def _size(form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]) -> None:
    _changed, size = form_ui.combo(
        "logical_size",
        "Sprite size",
        str(form["logical_size"]),
        [(str(s), f"{s} px") for s in options.get("logical_sizes") or ()],
    )
    form["logical_size"] = int(size)
    _changed, outline = form_ui.combo(
        "outline",
        "Outline",
        form["outline"],
        [(m, m) for m in options.get("outline_modes") or ()],
    )
    form["outline"] = outline
    # Beside the size, because it is a statement about the same act: how the
    # 512px render becomes a sprite of that size. Both modes were validated and
    # tested from the day the mode shipped and neither was ever askable, so
    # ``point`` -- the crisp, every-Nth-sample answer -- was a real code path
    # reachable only by editing a job row.
    _changed, reduce_mode = form_ui.combo(
        "reduce_mode",
        "Reduction",
        form["reduce_mode"],
        [(m, m) for m in options.get("reduce_modes") or ()],
    )
    form["reduce_mode"] = reduce_mode


def _palette(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]
) -> None:
    """A designed palette if one is installed, a colour budget otherwise.

    Two controls rather than one because they are two different answers: a
    named palette is the artist's decision and the budget is the machine's --
    a median cut over the atlas, which is the fallback and says so.
    """
    installed = list(options.get("palettes") or ())
    choices = [("", "Derived from the render")] + [(name, name) for name in installed]
    _changed, palette = form_ui.combo("palette", "Palette", form["palette"], choices)
    form["palette"] = palette
    if not palette:
        _changed, colors = form_ui.combo(
            "colors",
            "Colours",
            str(form["colors"]),
            [(str(n), f"{n} colours") for n in options.get("colors") or ()],
        )
        form["colors"] = int(colors)
    _changed, form["dither"] = form_ui.switch("dither", "Dither", bool(form["dither"]))
    # **Last, and optional.** A sheet has always been able to carry a name --
    # the door validates it, the worker writes it into the sidecar and the
    # chooser reads it back -- and there was no field, so every sheet a
    # character had was "sheet - 32px". Two builds at one size were two
    # identical rows in a list that only appears once there are two.
    _changed, form["name"] = form_ui.text(
        "name",
        "Name this sheet",
        str(form.get("name") or ""),
        hint="optional",
        max_length=rigging.MAX_SHEET_NAME,
        helper="Shown in the sheet chooser. The size and cell count are added for you.",
    )


def _submit(ctx: Any, form: dict[str, Any]) -> None:

    busy = ctx.busy("troupe-start")
    count = cell_count(form)
    ready = bool(form["prompt"].strip()) and 0 < count <= 512
    if busy:
        widgets.busy("Drawing the reference")
    if widgets.disabled_button(
        "Draw the reference",
        not busy and ready,
        (-1, 0),
        reason="Describe the character and select a layout of at most 512 cells."
        if not ready
        else "A reference is already being queued.",
    ):
        troupe_mode.start_character(ctx, form)
    widgets.cost_note(
        f"One image, and then it stops. After approval, the mesh, rig, and "
        f"{count} rendered cells follow."
    )
    if count > 256:
        widgets.muted("Large sheet: over 256 cells can take substantially longer.")
