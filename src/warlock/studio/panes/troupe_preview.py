"""Troupe's centre pane: one sprite, at an integer scale.

**Once it is playing it does not stop itself**, and that is the design the
program spec asks for: a bad frame in a walk cycle is obvious in half a second
of playback and invisible in a contact sheet. A one-shot holds its last frame
rather than freezing the preview, because a preview that stops needs a control
to start it again -- see ``troupe_mode.advance``.

**It opens paused, though**, which this docstring claimed the opposite of until
2026-09-03. The default was overturned on 2026-08-23 and the reasoning is on
``TroupeState.playing``: the first thing anyone does with a new sheet is look
at a *frame*, and a clip already moving when you arrive is one you have to stop
before you can look at anything in it.

**The scale is an integer and the filter is NEAREST.** A sprite drawn at 6.3x
through a linear filter is a blurred sprite, which is precisely the thing the
whole pipeline exists to avoid producing; drawing it that way in the one place
the user judges it would be the pipeline lying about its own output.

The pane is also the mode's heartbeat. There is no per-mode update hook -- the
arrangement ``packwright_mode`` already lives with -- so the pane that draws is
what pumps the clock.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, probe, theme, toolbar, troupe_mode, widgets
from ..manual import render as manual_render
from ..tokens import sp
from ..troupe import qa

#: The playback multipliers the transport offers. A short ladder rather than a
#: slider: the useful speeds for reading a run cycle are a quarter, a half and
#: full, and a continuous control invites 0.87x, which is not a thing anyone
#: wants. Stored as strings because that is what ``controls.combo`` trades in.
_SPEEDS: tuple[tuple[str, str], ...] = (
    ("0.25", "0.25x"),
    ("0.5", "0.5x"),
    ("1.0", "1x"),
    ("2.0", "2x"),
    ("4.0", "4x"),
)


def _speed_key(value: float) -> str:
    """The ladder rung ``value`` sits on -- nearest, never a refusal.

    ``speed`` is a float on the state and a profile written by any other means
    can hold something not on the ladder; a combo that showed a blank for it
    would be a control describing nothing.
    """
    return min((key for key, _ in _SPEEDS), key=lambda key: abs(float(key) - value))


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = troupe_mode.ensure(ctx)
    troupe_mode.advance(ctx, imgui.get_io().delta_time)

    _transport(ctx, state)
    imgui.dummy((0, 6))

    texture = troupe_mode.atlas_texture(ctx)
    record = troupe_mode.active_sheet(ctx)
    if texture is None or record is None:
        widgets.empty_state(
            icons.PERSON_STANDING,
            "No character on screen",
            "Pick one on the left, or describe a new one below it.",
        )
        return
    _scorecard(ctx, state)
    _sprite(ctx, state, texture, record)


#: The heatmap's square, in design pixels. Small: the picture under it is
#: what is being judged, and a matrix of 16 directions by 12 frames has to fit
#: above it without pushing it off the pane.
QA_CELL = 14.0
QA_GAP = 2.0


def _scorecard(ctx: Any, state: Any) -> None:
    """The animation scores as a heatmap over the frame x direction matrix.

    Above the sprite for ``_transport``'s reason. One square per cell of the
    selected movement; ok is the panel's own elevation, warn and bad the two
    status colours, and a bad cell also carries a cross so it says so in shape
    as well as colour (``tokens.STATUS_GLYPHS``' rule). Clicking a square
    points the preview at that cell and stops it. The scores **rank and never
    gate**: nothing here or anywhere refuses a sheet on their account.
    """
    from imgui_bundle import imgui

    score = troupe_mode.scores(ctx)
    movement = troupe_mode.preview_movement(ctx)
    if movement is None:
        return
    if score is None:
        widgets.muted(
            "Could not score this sheet." if troupe_mode.scores_failed(ctx) else "scoring..."
        )
        return
    if score.worst is None:
        widgets.muted("No frame flagged.")
    else:
        metric, value, cell = score.worst
        where = next((c for c in score.cells if c.cell == cell), None)
        place = (
            f" ({where.animation}/{where.direction}, frame {where.frame + 1})" if where else ""
        )
        widgets.muted(
            f"worst: {metric} {value:.2f}{place} - "
            f"{score.flagged} of {len(score.cells)} cells flagged"
        )

    by = score.lookup()
    animation = str(movement.get("key") or "")
    frames = int(movement.get("frames") or 1)
    directions = [str(d.get("key") or "") for d in movement.get("directions") or ()]
    side = sp(QA_CELL)
    gap = sp(QA_GAP)
    draw = imgui.get_window_draw_list()
    fills = {
        qa.LEVEL_OK: imgui.get_color_u32(theme.rgba(theme.ELEV_2)),
        qa.LEVEL_WARN: imgui.get_color_u32(theme.rgba(theme.WARN)),
        qa.LEVEL_BAD: imgui.get_color_u32(theme.rgba(theme.ERR)),
    }
    ring = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    ink = imgui.get_color_u32(theme.rgba(theme.TEXT))
    label_w = max((imgui.calc_text_size(d).x for d in directions), default=0.0) + sp(8)
    for direction in directions:
        origin = imgui.get_cursor_screen_pos()
        draw.add_text(
            (origin.x, origin.y + (side - imgui.get_text_line_height()) * 0.5), ink, direction
        )
        imgui.set_cursor_screen_pos((origin.x + label_w, origin.y))
        for frame in range(frames):
            cell = by.get((animation, direction, frame))
            level = qa.level(cell) if cell is not None else qa.LEVEL_OK
            current = direction == state.direction and frame == state.frame
            a = imgui.get_cursor_screen_pos()
            b = (a.x + side, a.y + side)
            clicked = imgui.invisible_button(f"##troupe-qa/{direction}/{frame}", (side, side))
            tip = _tooltip(cell)
            probe.record(
                label=f"{animation} {direction} frame {frame + 1}",
                kind="heatmap_cell",
                selected=current,
                tooltip=tip,
            )
            draw.add_rect_filled((a.x, a.y), b, fills[level], sp(2))
            if level == qa.LEVEL_BAD:
                inset = side * 0.28
                draw.add_line((a.x + inset, a.y + inset), (b[0] - inset, b[1] - inset), ink, 1.5)
                draw.add_line((b[0] - inset, a.y + inset), (a.x + inset, b[1] - inset), ink, 1.5)
            if current:
                draw.add_rect((a.x - 1, a.y - 1), (b[0] + 1, b[1] + 1), ring, sp(2), 0, 2.0)
            if imgui.is_item_hovered() and tip:
                imgui.set_tooltip(tip)
            if clicked:
                troupe_mode.goto(ctx, direction, frame)
            if frame + 1 < frames:
                imgui.same_line(0.0, gap)
    imgui.dummy((0, sp(4)))


def _tooltip(cell: Any) -> str:
    if cell is None:
        return ""
    lines = [", ".join(cell.flags) if cell.flags else "ok"]
    for name in qa.METRICS:
        value = cell.metrics.get(name)
        if value is not None:
            lines.append(f"{name}: {value:.2f}")
    return "\n".join(lines)


def _transport(ctx: Any, state: Any) -> None:
    """Play/pause, the two clip selectors, and the frame step.

    All on one row and above the sprite rather than below it: the thing being
    judged is the sprite, and controls under it push it up the pane by however
    tall they happen to be, which moves the picture every time the row wraps.
    """
    from imgui_bundle import imgui

    layout = troupe_mode.preview_layout(ctx)
    movements = list(layout.get("movements") or ())
    movement = troupe_mode.preview_movement(ctx)
    manual_render.help_button(ctx, "troupe-preview")

    # ``SQUARE`` for pause: the icon set is lucide 0.525.0 pinned to the
    # codepoints in ``icons.py``, and it has no pause glyph -- a stop square is
    # the nearest thing in it, and inventing a codepoint would draw a box.
    glyph = icons.SQUARE if state.playing else icons.PLAY
    frames = int((movement or {}).get("frames") or 1)

    # **A row, not a same_line chain.** This was four bare ``same_line`` calls,
    # which do not wrap -- they clip, so on a narrow centre pane the zoom field
    # went off the edge with no way to reach it. ``toolbar`` measures first and
    # collapses to icons, then to an overflow menu. The transport is pinned:
    # play/back/forward are the row's reason for existing, and a Play hidden
    # in a menu is not a transport.
    def _trailing() -> None:
        widgets.muted(f"frame {state.frame + 1} of {frames}")
        imgui.same_line()
        # 110, not less: an ``input_int`` spends most of its width on the two
        # step buttons, and at UI scale 1.5 a narrower field squeezes the
        # *value* out from between them -- a number control showing no number.
        imgui.set_next_item_width(sp(110))
        changed, zoom = controls.input_int("##troupe-zoom", int(state.zoom), 1, 2)
        if changed:
            # Clamped rather than validated: this is a view control, and
            # refusing a number the user dragged past the end would be a
            # dialog about nothing.
            state.zoom = max(1, min(int(zoom), 32))
        imgui.same_line()
        imgui.set_next_item_width(sp(90))
        # **``state.speed`` had no reader on screen.** ``advance`` has divided
        # the frame interval by it since the mode was written and nothing has
        # ever been able to change it, so every Troupe preview has played at
        # exactly 1x. Looking at a run cycle a quarter speed is the whole
        # reason the field exists.
        picked, name = controls.combo(
            "##troupe-speed", _speed_key(state.speed), _SPEEDS
        )
        if picked:
            state.speed = float(name)

    row = [
        toolbar.Item(
            key="play",
            label="Pause" if state.playing else "Play",
            icon=glyph,
            pinned=True,
        ),
        toolbar.Item(key="back", label="Previous frame", icon=icons.ARROW_LEFT, pinned=True),
        toolbar.Item(key="fwd", label="Next frame", icon=icons.CHEVRON_RIGHT, pinned=True),
    ]
    clicked = toolbar.toolbar("troupe-transport", row, trailing=(sp(340), _trailing))
    if clicked == "play":
        state.playing = not state.playing
    elif clicked == "back":
        troupe_mode.step(ctx, -1)
    elif clicked == "fwd":
        troupe_mode.step(ctx, 1)

    # A row each, and both wrapping. A 16-direction movement cannot fit on the
    # transport's line, and a clipped radio is a direction nobody can pick.
    _choices(
        [str(row.get("key") or "") for row in movements],
        state.animation,
        "anim",
        lambda name: troupe_mode.set_animation(ctx, name),
    )
    _choices(
        [str(row.get("key") or "") for row in (movement or {}).get("directions") or ()],
        state.direction,
        "dir",
        lambda name: troupe_mode.set_direction(ctx, name),
    )


def _choices(names: list[str], current: str, suffix: str, choose: Any) -> None:
    """One wrapping row of radio buttons.

    Through ``same_line_or_wrap`` rather than a bare ``same_line``, which is
    what puts the last item past the content edge where it cannot be clicked at
    all -- the rule the sidebar's seed row already follows. The width asked for
    is the *radio's*, which is a button's width plus its own glyph and the
    spacing between them, or the last item on a full row wraps one item late.
    """
    from imgui_bundle import imgui

    for name in names:
        if controls.radio_button(f"{name}##troupe-{suffix}", current == name):
            choose(name)
        widgets.same_line_or_wrap(
            widgets.button_width(name)
            + imgui.get_frame_height()
            + imgui.get_style().item_inner_spacing.x
        )
    imgui.new_line()


def _sprite(ctx: Any, state: Any, texture: Any, record: dict[str, Any]) -> None:
    """One cell of the atlas, drawn as a sub-rectangle at an integer scale."""
    from imgui_bundle import imgui

    index = troupe_mode.cell_index(ctx)
    if index is None:
        widgets.muted("That animation and direction are not on this sheet.")
        return
    columns = int(record.get("columns") or 8)
    size = int(record.get("frame_size") or 0)
    if size < 1:
        # A non-square plan writes ``frame_size: 0`` and puts the truth in
        # ``frame_w``/``frame_h`` -- a loud wrong answer rather than a quiet
        # one, which this reads rather than dividing by zero.
        size = int(record.get("frame_w") or 0)
    if size < 1:
        widgets.muted("That sheet does not describe its cell size.")
        return

    width, height = texture.size
    column, row = index % columns, index // columns
    uv0 = (column * size / width, row * size / height)
    uv1 = ((column + 1) * size / width, (row + 1) * size / height)

    drawn = size * state.zoom
    avail = imgui.get_content_region_avail()
    # Centred by hand: the sprite is small and the pane is not, and a sprite
    # pinned to the top-left of a large dark pane reads as a loading state.
    imgui.dummy((0, max((avail.y - drawn) * 0.5, 0)))
    imgui.dummy((max((avail.x - drawn) * 0.5, 0), 0))
    imgui.same_line()
    imgui.image(widgets.texture_ref(texture), (drawn, drawn), uv0, uv1)

    # **The wheel, at last.** The centre pane has carried
    # ``no_scroll_with_mouse`` since the mode was built, on the stated grounds
    # that the wheel belongs to the zoom control -- and no pane in Troupe read
    # the wheel, so it belonged to nothing and turning it over the sprite did
    # not do anything at all. Same clamp as the transport's field, because it
    # is the same setting: a view control, clamped rather than refused.
    io = imgui.get_io()
    if imgui.is_item_hovered() and io.mouse_wheel:
        state.zoom = max(1, min(int(state.zoom + io.mouse_wheel), 32))
