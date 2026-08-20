"""Troupe's centre pane: one sprite, playing, at an integer scale.

**It never stops**, and that is the design the program spec asks for: a bad
frame in a walk cycle is obvious in half a second of playback and invisible in a
contact sheet. Which is also why the mode's default is playing and why a
one-shot holds its last frame rather than freezing the preview -- see
``troupe_mode.advance``.

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

from .. import controls, icons, troupe_mode, widgets
from ..manual import render as manual_render
from ..tokens import sp
from ..troupe import spec as troupe_spec


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
    _sprite(ctx, state, texture, record)


def _transport(ctx: Any, state: Any) -> None:
    """Play/pause, the two clip selectors, and the frame step.

    All on one row and above the sprite rather than below it: the thing being
    judged is the sprite, and controls under it push it up the pane by however
    tall they happen to be, which moves the picture every time the row wraps.
    """
    from imgui_bundle import imgui

    table = troupe_spec.load()
    manual_render.help_button(ctx, "troupe-preview")

    # ``SQUARE`` for pause: the icon set is lucide 0.525.0 pinned to the
    # codepoints in ``icons.py``, and it has no pause glyph -- a stop square is
    # the nearest thing in it, and inventing a codepoint would draw a box.
    glyph = icons.SQUARE if state.playing else icons.PLAY
    if controls.small_button(f"{glyph}##troupe-play"):
        state.playing = not state.playing
    imgui.same_line()
    if controls.small_button(f"{icons.ARROW_LEFT}##troupe-back"):
        troupe_mode.step(ctx, -1)
    imgui.same_line()
    if controls.small_button(f"{icons.CHEVRON_RIGHT}##troupe-fwd"):
        troupe_mode.step(ctx, 1)

    # The frame counter and the zoom ride the transport's own row: they are
    # about *this* frame, and they are short.
    imgui.same_line()
    frames = table.animation(state.animation).frames
    widgets.muted(f"frame {state.frame + 1} of {frames}")
    imgui.same_line()
    imgui.set_next_item_width(sp(110))
    changed, zoom = controls.input_int("##troupe-zoom", int(state.zoom), 1, 2)
    if changed:
        # Clamped rather than validated: this is a view control, and refusing a
        # number the user dragged past the end would be a dialog about nothing.
        state.zoom = max(1, min(int(zoom), 32))

    # A row each, and both wrapping. Five animations and eight directions on
    # the transport's line ran the last of them into the pane edge and clipped
    # its label -- and a clipped radio is a direction nobody can pick.
    _choices(
        [a.name for a in table.animations],
        state.animation,
        "anim",
        lambda name: troupe_mode.set_animation(ctx, name),
    )
    _choices(
        [d.name for d in table.directions],
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
