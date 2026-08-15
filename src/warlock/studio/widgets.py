"""Small imgui pieces used by more than one pane.

Nothing here holds state. Each function draws and returns what the user did,
which keeps every pane's read of "what happened this frame" in one place rather
than spread across callbacks.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any

from imgui_bundle import imgui

from ..vectors import GRADE_MAX, GRADE_MIN
from . import fonts, icons, motion, theme, tokens
from . import state as app_state
from .tokens import sp

# Every artifact the downloads section offers, in the order it offers them:
# what the mesh *is* first, then what it can be turned into.
ARTIFACTS = (
    ("model.glb", "GLB"),
    ("source.glb", "Source GLB"),
    ("model.stl", "STL"),
    ("model_obj.zip", "OBJ (zip)"),
    ("model.fbx", "FBX"),
    ("collision.glb", "Collision"),
    ("textures.zip", "Textures"),
    ("rig.glb", "Rigged GLB"),
    # The pixels the mesh was reconstructed from. Last because it is an input
    # rather than an output, but present because a promoted job copies it and
    # then had no way to give it back: the inspector showed it as a 96px
    # thumbnail and offered no path to the file.
    ("input.png", "Reference image"),
)

# What a finished *reference* can hand over. A separate tuple rather than a
# filtered ARTIFACTS: the two lists have nothing in common but input.png, and a
# reference offered eight greyed mesh buttons -- which is what it used to get --
# reads as a broken asset rather than as a 2D one.
ARTIFACTS_2D = (
    ("icon.png", "Icon PNG"),
    ("sprite.png", "Sprite PNG"),
    ("pixel_32.png", "Pixel 32"),
    ("pixel_64.png", "Pixel 64"),
    ("pixel_128.png", "Pixel 128"),
    ("manifest.json", "Manifest"),
    ("input.png", "Source image"),
)

# And what a finished *tile* can. Almost none of the list above: every cutout
# is the operation of lifting a subject off a background, and a seamless
# texture is background. The texture itself comes first here rather than last,
# because for a tile input.png is the asset and not the input to one.
ARTIFACTS_TILE = (
    ("input.png", "Tile PNG"),
    ("wrap_preview.png", "Wrapped view"),
    # The zip leads the material group because it is what somebody taking this
    # into an engine wants: all four images plus a glTF material fragment in
    # one file. The individual maps follow, for whoever wants one of them.
    #
    # Every one of those three says "est." and that is not modesty. They are
    # derived from the albedo's own contrast and describe nothing about a
    # surface; a button labelled "Normal map" claims a measurement, and the
    # docstring explaining otherwise is in a repository the user does not have.
    ("material.zip", "Material set (zip)"),
    ("material_normal.png", "Normal (est.)"),
    ("material_roughness.png", "Roughness (est.)"),
    ("material_height.png", "Height (est.)"),
    ("manifest.json", "Manifest"),
)


def artifacts_for(job: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """The Export tab's grid for one job.

    Keyed on the stage rather than on which files happen to exist: every entry
    in all three tuples is *derivable*, so a list built from what is on disk
    would hide exactly the exports that have not been produced yet -- which is
    all of them, the first time.

    The two image stages' lists are labels for exactly what
    ``service.files.derived_2d_for`` says each can produce, plus the source
    image every job may take away. They are literals rather than a lookup
    because the *order* is a UI decision the service has no opinion about --
    a tile leads with its own PNG, a reference ends with the image it was
    drawn from -- and the price of that is a second place to edit. What keeps
    the two from drifting is ``test_the_grid_offers_exactly_what_each_stage_
    can_derive``, which fails on a name added to one and not the other: a
    label missing here is not a wrong button, it is no button at all.
    """
    stage = job.get("stage")
    if stage == "tile":
        return ARTIFACTS_TILE
    if stage == "reference":
        return ARTIFACTS_2D
    return ARTIFACTS


def texture_ref(texture: Any) -> Any:
    """A moderngl texture as something ``imgui.image`` will accept.

    Two things happen here, and skipping the second is the bug where every
    image in the UI renders as imgui's font atlas: imgui 1.92 wants an
    ImTextureRef rather than a bare id, *and* the renderer has to be told which
    moderngl object that id belongs to, because it binds through moderngl and
    an unknown id leaves whatever was bound last in place.
    """
    from . import imgui_backend

    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.register_texture(texture)
    return imgui.ImTextureRef(texture.glo)


def text_colored(value: int, text: str, alpha: float = 1.0) -> None:
    imgui.text_colored(imgui.ImVec4(*theme.rgba(value, alpha)), text)


def muted(text: str) -> None:
    text_colored(theme.MUTED, text)


def secondary(text: str) -> None:
    """Meaningful copy that is not the primary line (UX-18).

    The role imgui's own ``text_disabled`` was being used for, and the reason
    it could not stay: that style is MUTED at 0.6 alpha, which reaches the eye
    at 3.20:1 on a dark panel and 2.55:1 on a light one -- under the 4.5:1 body
    copy needs, in both themes. It is the right styling for a control that
    cannot be operated and the wrong styling for a sentence somebody has to
    read, and the two had collapsed into one call.

    So this is deliberately not a new colour: MUTED at full opacity already
    clears the bar on every surface in the ramp (6.67:1 dark, 5.84:1 light on
    PANEL). What was failing was the alpha, and nothing else.
    """
    text_colored(theme.MUTED, text)


def muted_wrapped(text: str) -> None:
    """A muted note that wraps to the pane rather than running off it.

    ``muted`` deliberately does not wrap -- most of its callers are short
    status lines where a wrap would be a second row of chrome. A *sentence*
    explaining what a button does is the other case, and in a 300px sidebar an
    unwrapped one is simply cut off at the pane edge with no scrollbar to reach
    it, which is what ``same_line`` past the content region already taught us
    about controls.
    """
    imgui.push_text_wrap_pos(0.0)
    text_colored(theme.MUTED, text)
    imgui.pop_text_wrap_pos()


def cost_note(text: str) -> None:
    """What pressing the button next to this will cost (S138).

    Its own function rather than a bare ``muted`` so the three submissions that
    carry one look alike and can be found together. The rule for the wording is
    that it says what *this app* knows -- whether the work is queued behind the
    GPU or derived on the spot, and roughly how much of it there is -- and never
    a wall-clock estimate the app has never measured. "About two minutes" would
    be a number, and a wrong one on someone else's card.

    No icon: the atlas is a pinned lucide subset, and a glyph it does not carry
    renders as the missing-glyph box -- the same rule that keeps these strings
    inside imgui's Basic-Latin+Latin-1 range.
    """
    muted(text)


def section(label: str) -> None:
    """A heading, with space above it instead of a rule under it.

    Two changes, one argument (UX.md Phase 2). **The separator is gone**: a
    rule is a line drawn where the rhythm failed, and a section that needs one
    to be legible is a section too close to what precedes it -- so the gap
    above went from ``SP_1`` (4 px, less than one line spacing) to ``SP_6``,
    which is what actually says "a new thing starts here".

    **And the label is sentence-case Medium at body size**, not small-caps
    muted. Field labels stay small-caps -- they earn it in a dense form, where
    a column of them has to read as chrome around the values. A *section*
    heading is the thing they are grouped under, and drawing both registers the
    same way flattened exactly the hierarchy the headings exist to create: on
    the 2D pane, "SUBJECT" and "CATEGORY" were the same size, weight and
    colour, one indent apart.
    """
    # ...except at the very top of a pane, where the gutter is already there
    # and a heading pushed down by both would start a fifth of the way into the
    # scroller. ``get_cursor_pos`` is window-relative, so "still at the padding"
    # is the test for "nothing has been drawn above me".
    if imgui.get_cursor_pos_y() > imgui.get_style().window_padding.y + 1.0:
        imgui.dummy((0, sp(tokens.SP_6)))
    with fonts.label(imgui):
        imgui.text(label)


def pane_title(label: str) -> None:
    """What a whole pane is, at heading size (UX.md Phase 2).

    A rung above :func:`section`, which names a group *inside* a pane -- and
    the reason the ramp needed a fourth step at all: with the scale stopping at
    16 the loudest thing a screen could say was a section heading, so a pane
    that titled itself did it at exactly the weight of the six headings under
    it. One loud thing per screen, and in a full-window pane this is it.

    No gap above, unlike ``section``: this is the first thing in its pane, and
    the window's own gutter is already there.
    """
    with fonts.heading(imgui):
        imgui.text(label)
    imgui.dummy((0, sp(tokens.SP_2)))


#: How many recent files a bridge pane offers. One screenful beside the Save
#: and Export rows; the full history is the mode's own menu.
RECENT_SHOWN = 6


def recent_files(paths: list[str], on_open: Any) -> None:
    """The "recent" block a document mode's bridge pane draws, or nothing.

    The path is in the imgui id and not just in the label: two maps can share a
    basename, and one id between them is one row.
    """
    from pathlib import Path

    if not paths:
        return
    imgui.dummy((0, 8))
    section("recent")
    for path in paths[:RECENT_SHOWN]:
        if imgui.selectable(f"{Path(path).name}##{path}", False)[0]:
            on_open(path)
        if imgui.is_item_hovered():
            imgui.set_tooltip(path)


def _chip(label: str, colour: tuple[float, float, float, float], fill: float) -> None:
    """One rounded chip, sized to its own text and laid out as a single item.

    The layout half matters as much as the paint: the cursor is put back where
    it started and a ``dummy`` of the chip's full size is what advances it, so
    a caller's ``same_line`` lands beside the chip rather than inside it.
    """
    pad_x, pad_y = sp(8), sp(3)
    with fonts.small(imgui):
        size = imgui.calc_text_size(label)
        pos = imgui.get_cursor_screen_pos()
        draw = imgui.get_window_draw_list()
        draw.add_rect_filled(
            pos,
            (pos.x + size.x + pad_x * 2, pos.y + size.y + pad_y * 2),
            imgui.get_color_u32((colour[0], colour[1], colour[2], fill)),
            (size.y + pad_y * 2) * 0.5,
        )
        imgui.set_cursor_screen_pos((pos.x + pad_x, pos.y + pad_y))
        imgui.text_colored(imgui.ImVec4(*colour), label)
        imgui.set_cursor_screen_pos((pos.x, pos.y))
        imgui.dummy((size.x + pad_x * 2, size.y + pad_y * 2))


def status_pill(status: str) -> None:
    """A rounded chip: colour *and* a glyph, because a pill that differs only
    by hue is unreadable to a chunk of people and useless in a screenshot."""
    glyph = theme.STATUS_GLYPHS.get(status, "?")
    _chip(f"{glyph} {status}", theme.status_color(status), 0.16)


# What each stage *is*, as a glyph and a word. The glyph carries the
# distinction -- an image for the two that end at pixels, a cube for the three
# that end at geometry -- because colour is spoken for: the status pill next to
# this one is the only chip on a card allowed to mean anything by its hue, and
# two colour encodings side by side fight rather than add.
STAGE_BADGES: dict[str, tuple[str, str]] = {
    "reference": (icons.IMAGE, "reference"),
    "tile": (icons.GRID, "tile"),
    "model": (icons.BOX, "model"),
    "rig": (icons.BONE, "rig"),
    "sheet": (icons.FILM, "sheet"),
}


def stage_badge(job: dict[str, Any], *, inline: bool = False) -> None:
    """What kind of asset this is, beside what state it is in.

    A card never said: the thumbnail, the "from a reference" note and the
    *absence* of a quality badge were the only tells, so a 2D reference and the
    mesh made from it were told apart by squinting at a 72 px picture. Unlike
    :func:`quality_badge` this always draws something, so the plain ``inline``
    is honest here -- a ``same_line`` issued for it is always spent.
    """
    stage = str(job.get("stage") or "")
    icon, label = STAGE_BADGES.get(stage, (icons.CIRCLE, stage or "asset"))
    if inline:
        imgui.same_line()
    _chip(f"{icon} {label}", theme.rgba(theme.MUTED), 0.10)


# Below this the silhouette audit has found nothing, and that is the whole of
# what it means (P120). Lives here rather than in the inspector because this is
# the lower layer of the two and the inspector imports it; it used to be the
# boundary of a *green* verdict, which is the claim that had to go.
AUDIT_UNINFORMATIVE = 0.02


def quality_badge(job: dict[str, Any], *, inline: bool = False) -> None:
    """The mesh's verdict, from mesh_report if there is one.

    mesh_report wins over mesh_audit because they answer different questions
    and only the report's answer is about whether an importer will accept it;
    the audit is a silhouette check and its thresholds are what the badge falls
    back to when no report exists.

    **``inline`` belongs here and not to the caller**, because this draws
    nothing at all for a job with neither -- and a ``same_line`` issued before a
    call that turns out to draw nothing is not spent, it is inherited by
    whatever is drawn next. On a library card that was the entire action row,
    pulled up onto the status pill's line and 73 px to the right, which pushed
    the favourite star clean off the card. Every reference in the library has
    neither a report nor an audit, so that was most of them.
    """
    params = job.get("params") or {}
    report = params.get("mesh_report")
    # There is deliberately no ``verdict`` branch above this one. There was:
    # it read ``report["verdict"]`` and painted good/usable/anything-else, and
    # nothing has ever written that key -- ``meshreport.build`` writes
    # ``status`` (ready / review / invalid), so every mesh fell through it to
    # the silhouette audit and the branch was dead from the day it was typed.
    # Wiring it to ``status`` rather than deleting it would have been worse
    # than nothing: ``status`` is "does this file have any reason at all",
    # pooling the triangle budget, the UV set, both PBR maps and the pivot into
    # one word, and it would have sat *above* the welded-watertight branch and
    # suppressed it for every mesh. The compound answer is the inspector's job,
    # where the reasons are listed; a card gets the one-word topology tell.
    # The welded flag, never the raw one: `meshreport` loads with process=False
    # so the UV and material checks see the unwelded mesh, which makes every
    # xatlas seam split a boundary edge -- a badge keyed on that says "not
    # watertight" about every textured mesh ever reconstructed.
    if isinstance(report, dict) and isinstance(report.get("welded_watertight"), bool):
        sealed = bool(report["welded_watertight"])
        if inline:
            imgui.same_line()
        text_colored(theme.OK if sealed else theme.WARN, "watertight" if sealed else "open")
        return
    audit = params.get("mesh_audit")
    if isinstance(audit, dict) and audit.get("worst") is not None:
        ratio = float(audit["worst"])
        # **No green branch** (P120). This used to paint ``theme.OK`` below 2%,
        # which is the ``hole_worst`` inversion drawn on a card: AUC(hole_worst
        # -> reject) is 0.115 over the reviewed corpus -- not weakly
        # informative, *backwards* -- because the dominant failure mode is a
        # solid slab, and a slab has no visible openings at all. 48 of 81
        # rejected meshes measured exactly 0.0 and would have worn a green
        # badge; none of the 3 accepted ones did.
        #
        # A high reading is still real evidence of a hole, so the escalation
        # stays. What is gone is the claim in the other direction: below the
        # threshold the badge is muted, which says "nothing seen through" and
        # not "good". ``hole_worst`` is corpus-dependent and is never a quality
        # scale -- see ``docs/measurements/2026-08-09-rebaseline.md`` and
        # ``judge.py``'s module docstring.
        colour = (
            theme.MUTED
            if ratio < AUDIT_UNINFORMATIVE
            else theme.WARN
            if ratio < 0.08
            else theme.ERR
        )
        if inline:
            imgui.same_line()
        text_colored(colour, f"{ratio * 100:.1f}% open")


def progress_bar(percent: float, width: float = -1.0, height: float = 0.0) -> None:
    """A bar, or a marquee when there is no percentage to show.

    A determinate bar sitting at zero reads as "stuck"; the marquee reads as
    "working on something it cannot measure", which is exactly what a cold
    model load is.
    """
    if height <= 0:
        height = sp(5)
    draw = imgui.get_window_draw_list()
    avail = imgui.get_content_region_avail().x if width < 0 else width
    pos = imgui.get_cursor_screen_pos()
    imgui.dummy((avail, height))
    radius = height * 0.5
    draw.add_rect_filled(
        pos, (pos.x + avail, pos.y + height), imgui.get_color_u32(theme.rgba(theme.EDGE)), radius
    )
    fill = imgui.get_color_u32(theme.rgba(theme.ACCENT))
    if percent > 0:
        end = pos.x + avail * min(percent, 100.0) / 100.0
        draw.add_rect_filled(pos, (end, pos.y + height), fill, radius)
    else:
        span = avail * 0.25
        offset = (time.monotonic() * 0.6 % 1.0) * (avail + span) - span
        draw.add_rect_filled(
            (pos.x + max(offset, 0.0), pos.y),
            (pos.x + min(offset + span, avail), pos.y + height),
            fill,
            radius,
        )


def spinner(radius: float = 0.0, thickness: float = 0.0) -> None:
    """An indeterminate arc. Draws in place and advances the cursor."""
    radius = radius or sp(7)
    thickness = thickness or sp(2.5)
    draw = imgui.get_window_draw_list()
    pos = imgui.get_cursor_screen_pos()
    centre = (pos.x + radius, pos.y + radius)
    imgui.dummy((radius * 2, radius * 2))
    start = time.monotonic() * 3.0
    draw.path_clear()
    for i in range(24):
        angle = start + i / 24.0 * math.pi * 1.5
        draw.path_line_to(
            (centre[0] + math.cos(angle) * radius, centre[1] + math.sin(angle) * radius)
        )
    draw.path_stroke(imgui.get_color_u32(theme.rgba(theme.ACCENT)), thickness=thickness)


def combo(label: str, value: str, options: list[tuple[str, str]], width: float = -1.0):
    """A combo over (key, label) pairs. -> the (possibly unchanged) key.

    Keys rather than indices because every one of these is a guidance taxonomy
    whose order is free to change; an index would silently become a different
    option the next time a table gained an entry.

    **``label`` should be ``##``-prefixed here; a visible name goes through
    :func:`labeled_combo`.** imgui draws a combo's label to its *right*, and the
    default width is ``-1``, so a named combo in a sidebar puts its name past
    the content region -- where ``same_line`` clips instead of wrapping, i.e.
    the name is simply not drawn. Nine call sites were doing that (Blend,
    Symmetry, Model, Style LoRA, Budget, Frame, Lighting, From, To), which is
    why this is written down here rather than fixed nine times: the rule was
    already stated in ``settings_3d`` and it was stated somewhere nobody
    reaching for ``combo`` would read it.
    """
    keys = [key for key, _ in options]
    labels = [text for _, text in options]
    current = keys.index(value) if value in keys else 0
    if width:
        imgui.set_next_item_width(width)
    changed, index = imgui.combo(label, current, labels)
    return keys[index] if changed else value


def note_ime_rect() -> None:
    """Put the IME's candidate window on the field just drawn (UX-19).

    Called right after a text widget, and does nothing unless that widget has
    focus -- so the rect follows the caret between fields without any of them
    having to know the others exist. It sits in the two wrappers below rather
    than at ~20 call sites for the same reason.

    Guarded on ``is_item_active`` rather than on ``want_text_input``: the
    latter is true for the whole frame an unrelated field owns, and pointing
    SDL at the wrong control is worse than not pointing it anywhere.
    """
    if not imgui.is_item_active():
        return
    from . import imgui_backend

    lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    imgui_backend.set_ime_rect((lo.x, lo.y), (hi.x - lo.x, hi.y - lo.y))


def input_text(label: str, value: str, *, max_length: int = 1000, hint: str = "") -> str:
    """A single-line field, clamped after the fact.

    imgui's Python binding grows its own buffer, so the cap is applied to what
    comes back rather than to what can be typed -- which also means a paste
    over the cap keeps its first N characters instead of being refused.
    """
    if hint:
        changed, out = imgui.input_text_with_hint(label, hint, value)
    else:
        changed, out = imgui.input_text(label, value)
    note_ime_rect()
    return out[:max_length] if changed else value


def multiline(label: str, value: str, height: float, max_length: int) -> str:
    """A wrapping text area.

    Wrapping is not imgui's default: without the flag a long prompt scrolls off
    to the right in a box three lines tall, which hides most of what was typed
    behind a horizontal scrollbar.
    """
    changed, out = imgui.input_text_multiline(
        label, value, (-1, height), imgui.InputTextFlags_.word_wrap.value
    )
    note_ime_rect()
    return out[:max_length] if changed else value


def field_options(ctx: Any, field: str) -> list[tuple[str, str]]:
    """(key, label) pairs for one taxonomy field, with a blank first entry.

    Blank because every guidance field is optional: an empty select means "say
    nothing about this", which is a different prompt from any of the choices.
    """
    entries = (ctx.guidance.get("fields") or {}).get(field) or []
    return [("", f"{field.replace('_', ' ')}...")] + [
        (e["key"], e["label"]) for e in entries
    ]


# Set only by the pane smoke test, which needs every section's contents built
# rather than a column of collapsed headings. Here rather than as an argument
# because the point is to override what the *user* left closed, which no caller
# of header() can know.
FORCE_SECTIONS_OPEN = False

# The Settings the persisted headers write through; injected at setup because
# widgets stay stateless functions and a pane never carries the settings just
# to draw a heading. None (tests, early frames) means default_open decides.
_SETTINGS: Any = None


def attach_settings(settings: Any) -> None:
    global _SETTINGS
    _SETTINGS = settings


# Sections asked to open on their next draw, by persist_key. Module state
# rather than a field on AppState because ``header`` already reads its other
# half (``FORCE_SECTIONS_OPEN``, ``_SETTINGS``) from here and takes no state
# argument to hang it on.
_OPEN_REQUESTS: set[str] = set()


def thumb_placeholder(size: float, glyph: str) -> None:
    """A framed square with an icon in it, where a thumbnail would be (H72).

    A bare ``dummy`` was the old answer, which reads as a layout bug rather
    than as "there is no picture yet" -- and every queued job, every failure
    and every rig row has no picture. The glyph says what *kind* of thing is
    missing, which is the difference between a card that looks broken and one
    that looks pending.

    Takes exactly the space an image of the same size would, so a card with a
    thumbnail and a card without lay out identically.
    """
    origin = imgui.get_cursor_screen_pos()
    imgui.dummy((size, size))
    draw = imgui.get_window_draw_list()
    draw.add_rect(
        origin,
        (origin.x + size, origin.y + size),
        imgui.get_color_u32(theme.rgba(theme.MUTED, 0.28)),
        sp(4),
    )
    if not glyph:
        return
    # The glyphs live in the same atlas as the text (they are a merged icon
    # range, not a second font), so a bigger one is just a bigger push.
    with fonts.title(imgui):
        extent = imgui.calc_text_size(glyph)
        draw.add_text(
            (
                origin.x + (size - extent.x) * 0.5,
                origin.y + (size - extent.y) * 0.5,
            ),
            imgui.get_color_u32(theme.rgba(theme.MUTED, 0.55)),
            glyph,
        )


def list_filter(ctx: Any, tag: str, count: int, *, minimum: int = 8) -> str:
    """A small search box over a panel list (J86). -> the lowered query.

    Drawn only once the list is long enough to be worth searching, and the
    query is *cleared* when it is not: a box that appears at eight entries and
    vanishes at seven would otherwise leave a filter running with nothing on
    screen to say so, and a panel that has silently hidden half its contents
    looks like a panel that has lost them.

    The value lives on ``AppState.list_filters`` rather than here -- nothing in
    this module holds state -- keyed by ``tag`` so every list has its own.
    """
    store = ctx.state.list_filters
    if count < minimum:
        store.pop(tag, None)
        return ""
    imgui.set_next_item_width(-1)
    store[tag] = input_text(f"##find-{tag}", store.get(tag, ""), max_length=80, hint="Find...")
    return store[tag].strip().lower()


def no_matches(needle: str, shown: int) -> None:
    """The row a filtered list draws when the filter matched nothing.

    A separate call rather than something ``list_filter`` could do on its own:
    every caller owns its loop and its own idea of what "shown" means -- a
    layer, a profile, an outliner node -- so only the caller can count. A
    filtered-to-empty panel with no such row looks exactly like a panel that
    has lost its contents, which is what a search box appearing above nothing
    at all reads as.

    Not an ``empty_state``: that is the centred three-register treatment for a
    region with nothing *in* it, and this list is full -- it is the query that
    matches none of it.
    """
    if needle and shown == 0:
        muted("Nothing matches the filter.")


def request_open(persist_key: str) -> None:
    """Open a collapsible section the next time it is drawn.

    For the case where something arrives in a section the user has collapsed --
    a dropped image landing in the 2D pane's Reference block -- and would
    otherwise be accepted with nothing on screen changing at all.
    """
    _OPEN_REQUESTS.add(persist_key)


# How long a slot stays outlined after something is dropped into it (H70).
# Long enough to catch the eye after the mouse has moved on, short enough not
# to read as a permanent state of the control.
DROP_FLASH_SECONDS = 1.4


def drop_flash(state: Any, key: str) -> float:
    """0..1 -- how strongly the named slot should be outlined right now.

    An eased fade rather than a blink: a control that flashes on and off reads
    as an error, and this is an acknowledgement.
    """
    if getattr(state, "drop_flash_slot", "") != key:
        return 0.0
    age = time.monotonic() - getattr(state, "drop_flash_at", 0.0)
    if age < 0 or age > DROP_FLASH_SECONDS:
        return 0.0
    # Reduce-motion keeps the acknowledgement and drops the ramp: the outline is
    # simply on for its dwell and then off. Collapsing the *dwell* instead would
    # remove the answer rather than the animation.
    if motion.REDUCED:
        return 1.0
    return (1.0 - age / DROP_FLASH_SECONDS) ** 0.6


def shadow(
    low: Any,
    high: Any,
    radius: float,
    elevation: str = "resting",
    *,
    strength: float = 1.0,
    draw: Any = None,
) -> None:
    """The one shadow, under one rectangle (UX.md Phase 2).

    ``low``/``high`` are screen-space corners, in the ``(x, y)``-indexable shape
    :func:`ring` already takes; ``radius`` is the surface's own corner radius,
    which each band grows with so a rounded card does not get a square shadow.
    ``elevation`` names how far off the surface behind it this one sits --
    ``resting`` (a card), ``raised`` (a popup, the palette) or ``overlay`` (a
    modal) -- and is the whole reason this is a helper rather than four lines in
    ``card``: depth is one physical story, and three call sites each inventing a
    pair of alphas is three stories.

    ``strength`` scales the alphas alone, for a surface whose shadow deepens as
    it lifts. It is *not* the elevation: raising a card on hover moves it
    further from the panel, but a hovered card is still resting furniture.

    Every band is **stroked**, not filled -- see ``tokens.SHADOW_STEPS``. That
    is what lets the same recipe serve a card, whose shadow is drawn before the
    surface, and an imgui window, whose background ``begin`` has already
    painted by the time anything else can draw.
    """
    if strength <= 0.0:
        return
    scale = tokens.ELEVATION.get(elevation, 1.0)
    draw = draw if draw is not None else imgui.get_window_draw_list()
    if _sliced_shadow(low, high, radius, scale, strength, draw):
        return
    for inner, outer, alpha in tokens.SHADOW_STEPS:
        mid = sp((inner + outer) * 0.5) * scale
        thick = sp(outer - inner) * scale
        drop = sp(outer * tokens.SHADOW_DROP) * scale
        draw.add_rect(
            (low[0] - mid, low[1] - mid + drop),
            (high[0] + mid, high[1] + mid + drop),
            imgui.get_color_u32((0.0, 0.0, 0.0, alpha * min(strength, 1.0))),
            radius + mid,
            thickness=thick,
        )


def _sliced_shadow(
    low: Any, high: Any, radius: float, scale: float, strength: float, draw: Any
) -> bool:
    """:func:`shadow` from the blurred sprite. -> whether it drew anything.

    Eight quads, never nine: the centre is left alone so this one recipe can be
    drawn *under* a card and *over* an already-painted window background, which
    is the property the stroked bands had and the reason they were stroked.

    Returns False -- and the bands run instead -- in the two cases the slice
    cannot serve. There is no sprite (switched off, no renderer, a build that
    failed once), or the surface is smaller than its own corners: a shadow
    whose corner blocks would overlap has to either shrink the blocks, which
    distorts the blur, or draw them twice, which doubles the alpha exactly at
    the corners. Neither is worth having, and the case is rare -- a control
    smaller than about 40 px square at ``resting``.
    """
    from . import ninepatch, shadows

    spread = sp(shadows.SPREAD) * scale
    built = shadows.sprite(radius, spread)
    if built is None:
        return False
    # The sprite is built from the *rounded* pair, so the reach it actually has
    # is the rounded spread rather than the one asked for. Reading it back off
    # the sprite's own corner (which is radius + spread) rather than trusting
    # the argument is what keeps the drawn rect and the pixels in agreement to
    # the pixel, which is the whole reason a nine-slice looks like a blur.
    reach = float(built.corner) - float(round(radius))
    drop = sp(shadows.SPREAD * tokens.SHADOW_DROP) * scale
    colour = imgui.get_color_u32((0.0, 0.0, 0.0, shadows.PEAK * min(strength, 1.0)))
    return ninepatch.paint(
        draw,
        built,
        (low[0] - reach, low[1] - reach + drop),
        (high[0] + reach, high[1] + reach + drop),
        colour,
        centre=False,
    )


def window_shadow(elevation: str = "raised", *, radius: float | None = None) -> None:
    """:func:`shadow` under the imgui window being drawn right now.

    Called just after ``begin``, and it has to escape the window's clip
    rectangle to do anything at all -- the shadow is by definition outside the
    window. ``push_clip_rect_full_screen`` is the render-level scissor and
    touches no hit-testing, so nothing about where clicks land changes.

    The rect is *this* frame's position and last frame's size, which for an
    ``always_auto_resize`` popup is one frame stale on the frame it appears --
    the frame its alpha is near zero anyway (:func:`popover_enter`).

    ``radius`` has to be passed by any caller that pushed its own rounding for
    this window, because the push is scoped to ``begin`` and is gone by the time
    this runs -- reading the global style would draw a 6 px shadow around a 10
    px surface, which shows as four bright corners.
    """
    if not _has_context():
        return
    try:
        draw = imgui.get_window_draw_list()
        pos, size = imgui.get_window_pos(), imgui.get_window_size()
    except RuntimeError:
        # A context but no *frame*: a headless test driving ``dialogs`` through
        # its own stand-in imgui still reaches the real module here, and after
        # some other test in the session has made a context that is an assert
        # rather than the crash ``_has_context`` catches. Two guards because
        # there are two failure modes, not because one of them is belt-and-
        # braces: neither answers for the other.
        return
    draw.push_clip_rect_full_screen()
    shadow(
        (pos.x, pos.y),
        (pos.x + size.x, pos.y + size.y),
        imgui.get_style().window_rounding if radius is None else radius,
        elevation,
        draw=draw,
    )
    draw.pop_clip_rect()


def surface_fill(
    draw: Any,
    low: Any,
    high: Any,
    radius: float,
    colour: Any,
    *,
    border: Any = None,
) -> bool:
    """A filled surface with continuous-curvature corners. -> whether it drew.

    The caller is expected to have cleared whatever imgui would have painted
    (a child's ``child_bg``, a window's background alpha) *before* asking,
    because this replaces the fill rather than sitting over it -- and to be
    ready for a False, which means the pre-Phase-5 rounded rect should be drawn
    instead: the effect is off, the radius is below the size the difference
    shows at, or the surface is smaller than its own corners.

    The hairline is drawn as the *same* shape one border-width larger rather
    than as an outline. A superellipse fill under a circular-arc stroke would
    disagree with itself by a pixel or two at exactly the four places this
    whole item is about, and imgui's own border is a circular arc.
    """
    from . import ninepatch, surfaces

    built = surfaces.sprite(radius)
    if built is None:
        return False
    low = (low[0], low[1]) if not hasattr(low, "x") else (low.x, low.y)
    high = (high[0], high[1]) if not hasattr(high, "x") else (high.x, high.y)
    if border is None:
        return ninepatch.paint(draw, built, low, high, imgui.get_color_u32(colour))
    edge = sp(tokens.BORDER)
    inner_low = (low[0] + edge, low[1] + edge)
    inner_high = (high[0] - edge, high[1] - edge)
    # Both rectangles are decided before either is painted. The inset one is
    # the smaller, so there is a window of sizes about two border-pixels wide
    # where the border fits and the fill does not -- and painting the border
    # first meant returning False with it already on screen, under the
    # fallback fill the caller then draws on that False.
    if not (ninepatch.covers(built, low, high) and ninepatch.covers(built, inner_low, inner_high)):
        return False
    ninepatch.paint(draw, built, low, high, imgui.get_color_u32(border))
    return ninepatch.paint(draw, built, inner_low, inner_high, imgui.get_color_u32(colour))


def frosted() -> bool:
    """Whether the next window should ask for a blurred backdrop.

    Asked *before* ``begin``, because the caller has to clear the window's own
    background alpha there and cannot un-paint it afterwards. It is the same
    question :func:`window_backdrop` answers again after ``begin`` -- and that
    is why the second call paints a solid fill when the backdrop is not ready:
    a window whose background was cleared on this answer and whose backdrop
    then failed to arrive would be a floating surface with nothing in it.
    """
    from . import vibrancy

    return _has_context() and vibrancy.available()


def window_backdrop(*, radius: float | None = None, tint: float = 0.8) -> bool:
    """Blur what is behind the window being drawn right now. -> did it?

    Called immediately after ``begin``, beside :func:`window_shadow`, and it
    replaces the window's own background rather than sitting under it: imgui
    paints that during ``begin``, so the only way to be *behind* it is to cover
    it. The caller therefore sets the window's background alpha to zero, and
    what lands here is the blurred copy plus a tint of the fill colour -- which
    is what a translucent material is.

    ``tint`` is how much of the surface colour survives over the blur. High
    enough that text stays readable over a bright backdrop, low enough that the
    blur is visibly there: the two failure modes are a panel you cannot read
    and a panel indistinguishable from the solid one it replaced.
    """
    from . import vibrancy

    if not _has_context() or not vibrancy.available():
        return False
    try:
        pos, size = imgui.get_window_pos(), imgui.get_window_size()
    except RuntimeError:
        return False
    ref = vibrancy.backdrop()
    low = (pos.x, pos.y)
    high = (pos.x + size.x, pos.y + size.y)
    rounding = imgui.get_style().window_rounding if radius is None else radius
    draw = imgui.get_window_draw_list()
    if ref is None:
        # Nothing captured yet -- the first frames of a session, or a capture
        # that has not come round since the window was resized. The caller has
        # already cleared this window's background on ``frosted``'s word, so
        # this paints the fill imgui would have.
        draw.add_rect_filled(
            low, high, imgui.get_color_u32(theme.rgba(theme.ELEV_2)), rounding
        )
        return False
    window = imgui.get_io().display_size
    uv_min, uv_max = vibrancy.uv_for(low, high, (window[0], window[1]))
    draw.add_image_rounded(
        ref,
        low,
        high,
        uv_min,
        uv_max,
        imgui.get_color_u32((1.0, 1.0, 1.0, 1.0)),
        rounding,
    )
    draw.add_rect_filled(
        low,
        high,
        imgui.get_color_u32(theme.rgba(theme.ELEV_2, tint)),
        rounding,
    )
    return True


def _has_context() -> bool:
    """Whether there is an imgui context to draw into at all.

    ``fonts`` and ``motion`` both state this rule and both answer it their own
    way; the three helpers below need it because they are the only ones in this
    module reached by a caller that has substituted its *own* imgui. A headless
    test driving ``dialogs`` through a stand-in still lands in here on the real
    module, and ``push_style_var`` with no context is not an exception -- it is
    an access violation, because imgui's null check is an assert compiled out of
    the release build.
    """
    return imgui.get_current_context() is not None


def push_surface_rounding() -> float:
    """Round the *next* window like a surface rather than like a control.

    -> the physical radius pushed, which is what :func:`window_shadow` then has
    to be told (the push is scoped around ``begin`` and gone by then).

    Both rounding vars, because a modal is a popup and imgui rounds a popup by
    ``popup_rounding`` while rounding everything else by ``window_rounding`` --
    a caller that pushed one and got the other would be debugging a corner.
    Paired with :func:`pop_surface_rounding` immediately after ``begin``: window
    rounding latches when the background is painted, so holding the push for the
    whole body would round every child and tooltip inside it too.
    """
    radius = sp(tokens.RADIUS_L)
    if not _has_context():
        return radius
    imgui.push_style_var(imgui.StyleVar_.window_rounding.value, radius)
    imgui.push_style_var(imgui.StyleVar_.popup_rounding.value, radius)
    return radius


def pop_surface_rounding() -> None:
    if _has_context():
        imgui.pop_style_var(2)


def ring(low: Any, high: Any, colour: int, alpha: float, thick: float = 2.0) -> None:
    """An outline around a rectangle, drawn into the current window.

    Shared by the drop flash and by the in-app drag target, so "this is the
    thing you are aiming at" looks the same however the payload arrived.
    """
    if alpha <= 0.0:
        return
    imgui.get_window_draw_list().add_rect(
        (low.x - sp(4), low.y - sp(4)),
        (high.x + sp(4), high.y + sp(4)),
        imgui.get_color_u32(theme.rgba(colour, alpha)),
        sp(4),
        thickness=sp(thick),
    )


class Problem(str):
    """A validation message that knows which control it is about.

    A ``str`` subclass, deliberately: the aggregate block above Generate does
    ``imgui.text_wrapped(problem)`` and the tests compare against plain
    strings, and both keep working unchanged. What it adds is the half that was
    missing when the *keyboard* door refused a submit -- Ctrl+Enter and the
    command palette both call ``generate``/``promote`` directly, where the only
    feedback was the block of red text in a pane the user may not be looking
    at, so a refused Ctrl+Enter did nothing observable at all.

    ``field`` is empty for a problem that names no single control ("Choose a
    reference first" is about the library, not about a widget in the form), and
    :meth:`state.note_field_error` already treats that as "keep going to the
    toast".
    """

    __slots__ = ("field",)

    def __new__(cls, text: str, field: str = "") -> Problem:
        self = super().__new__(cls, text)
        self.field = field
        return self


def field_error(state: Any, field: str) -> bool:
    """Ring the control just drawn, and say why, if a refusal named it (UX.md
    Phase 3). -> whether anything was drawn.

    Called immediately after the control, because it reads ``get_item_rect_*``
    -- which is the whole reason this is a function taking a *name* rather than
    a wrapper around every widget: the panes draw combos, sliders, radio rows
    and hand-built segmented controls, and one shared "just drew a thing" hook
    covers all of them where a wrapper per widget type would not.

    One shared helper so every pane says it the same way. The aggregate block
    above Generate (``settings_2d.validate``) stays: it is the list of
    everything wrong, which is what somebody looking at a disabled button
    wants, and this is the pointer to *which control* -- two views of one fact,
    and the pointer is the half that was missing.
    """
    message = (getattr(state, "field_errors", None) or {}).get(field)
    if not message:
        return False
    ring(imgui.get_item_rect_min(), imgui.get_item_rect_max(), theme.ERR, 0.9, thick=1.5)
    imgui.push_text_wrap_pos(0.0)
    text_colored(theme.ERR, message)
    imgui.pop_text_wrap_pos()
    return True


def header(label: str, default_open: bool = True, persist_key: str | None = None) -> bool:
    """A collapsing section. Open by default, because these *are* the panel.

    The inspector's sections are its content, not extras: an asset opened with
    every section collapsed shows a column of headings and nothing to act on.
    With ``persist_key``, the open state survives a restart -- imgui's own ini
    is disabled, so this rides the app's Settings instead. FORCE_SECTIONS_OPEN
    still wins: the smoke test needs every section built regardless of what
    last session left closed.
    """
    stored: dict[str, Any] = {}
    if persist_key and _SETTINGS is not None:
        stored = _SETTINGS.get("panels_open") or {}
    if FORCE_SECTIONS_OPEN:
        imgui.set_next_item_open(True, imgui.Cond_.always.value)
    elif persist_key and persist_key in _OPEN_REQUESTS:
        # A one-shot ``always``, consumed here. ``once`` cannot serve this: it
        # fires the first time the section is drawn in a session, and the case
        # this exists for -- a dropped file landing in a section the user has
        # collapsed -- happens long after that.
        _OPEN_REQUESTS.discard(persist_key)
        imgui.set_next_item_open(True, imgui.Cond_.always.value)
    elif persist_key and persist_key in stored:
        imgui.set_next_item_open(bool(stored[persist_key]), imgui.Cond_.once.value)
    # allow_overlap: the manual's (?) button right-aligns onto this same header
    # row, and without the flag imgui's hover resolution gives the row priority
    # and the button becomes unclickable.
    flags = imgui.TreeNodeFlags_.allow_overlap.value
    if default_open:
        flags |= imgui.TreeNodeFlags_.default_open.value
    with fonts.label(imgui):
        opened = imgui.collapsing_header(label, flags)
    if (
        persist_key
        and _SETTINGS is not None
        and not FORCE_SECTIONS_OPEN
        and stored.get(persist_key) != opened
    ):
        _SETTINGS.set("panels_open", {**stored, persist_key: opened})
    return opened


def tab_bar(bar_id: str, tabs: list[tuple[str, Any]]) -> None:
    """Labelled tabs over draw callables.

    Under FORCE_SECTIONS_OPEN every tab's content is drawn sequentially, each
    inside its own ``push_id`` -- the smoke test exists to build every section,
    and a tab bar that only builds the selected tab would quietly exempt the
    other two from it.
    """
    if FORCE_SECTIONS_OPEN:
        for label, draw_fn in tabs:
            imgui.push_id(f"{bar_id}/{label}")
            draw_fn()
            imgui.pop_id()
        return
    if not imgui.begin_tab_bar(bar_id):
        return
    for label, draw_fn in tabs:
        opened, _ = imgui.begin_tab_item(label, None, 0)
        if opened:
            draw_fn()
            imgui.end_tab_item()
    imgui.end_tab_bar()


# How far a popover rises as it appears, in design px. Small on purpose: the
# move is there to say "this arrived", and anything a reader can measure is a
# thing sliding around rather than a surface settling.
POPOVER_RISE = 6.0


def popover_enter(key: str, appearing: bool) -> tuple[float, float]:
    """``(alpha, rise)`` for a surface that is appearing. -> ``(1.0, 0.0)`` once
    it has settled, and immediately under reduce-motion.

    One helper because the four floating surfaces in the app -- the confirm
    modal, the text prompt, the command palette, the two popups behind the
    header -- were four hard cuts and would otherwise be four spellings of one
    fade. The caller pushes ``alpha`` as a style var *around* ``begin`` (a
    window's background is painted during ``begin``, so pushing after it fades
    the contents over an already-solid panel) and spends ``rise`` as a cursor
    offset inside it.

    ``appearing`` is passed rather than read from ``is_window_appearing``
    because that answer is only available *after* ``begin``, which is one frame
    too late for the alpha the same ``begin`` paints with.
    """
    motion_key = f"popover/{key}"
    # The sprung version (UX.md Phase 5): stated at 0 on the appearing frame and
    # sprung to 1, so the surface comes to rest by settling rather than by
    # running out of ramp. The alpha is *clamped* and the rise is not -- a
    # spring goes a little past 1, which for an opacity is a value that does not
    # exist and for a position is exactly the overshoot being asked for.
    if appearing:
        motion.seed(motion_key, 0.0)
    t = motion.spring(motion_key, 1.0, duration=tokens.DUR_BASE)
    return min(max(t, 0.0), 1.0), sp(POPOVER_RISE) * (1.0 - t)


# Which plain popups were open last frame. Module state keyed by name, exactly
# as ``motion``'s own is and for the same reason: the alternative was a set on
# ``App``, which put a frame-loop attribute in the path of every pane that
# opens a popup and of every test that drives one with a stand-in App.
_POPUPS_OPEN: set[str] = set()


def popup_enter(name: str) -> tuple[float, float]:
    """:func:`popover_enter` for a popup that has no open flag of its own.

    The header's shortcuts and diagnostics popups are opened by ``open_popup``
    and closed by imgui itself, so there is nothing to read "is this the frame
    it appeared" off. imgui knows -- but only through ``is_window_appearing``,
    which answers *after* ``begin``, and the alpha is needed before it, because
    ``begin`` is where the popup's own background is painted.
    """
    opened = imgui.is_popup_open(name)
    appearing = opened and name not in _POPUPS_OPEN
    if opened:
        _POPUPS_OPEN.add(name)
    else:
        _POPUPS_OPEN.discard(name)
    return popover_enter(f"popup/{name}", appearing)


def _hover_amount(key: str) -> float:
    """Last frame's hover for ``key``, 0..1. Set by :func:`note_hover`.

    The one-frame lag is not a compromise, it is the only order available: a
    button's colour is needed to draw the button, and whether it is hovered is
    known only once it has been. ``card`` has always worked this way.
    """
    return motion.peek(f"hover/{key}")


def note_hover(key: str, hovered: bool) -> None:
    """Record this frame's hover so the next frame can draw the approach."""
    motion.value(f"hover/{key}", 1.0 if hovered else 0.0, duration=tokens.DUR_FAST)


def _button_with_note(
    label: str,
    enabled: bool,
    size: tuple[float, float] = (0, 0),
    *,
    reason: str = "",
    tooltip: str = "",
) -> tuple[bool, bool]:
    """:func:`disabled_button`, also answering whether it was hovered.

    The second value exists because of the ordering hazard the body below
    documents: ``set_tooltip`` renders an item of its own and overwrites
    ``g.LastItemData``, so a caller that asks ``is_item_hovered()`` *after* this
    returns is told about the tooltip's text rather than about the button --
    which reads as "not hovered" and drops the hover animation on precisely the
    frames a reason is being explained. ``primary_button`` and ``ghost_button``
    both animate their own fill from last frame's hover, so both need the
    answer this captured before the tooltip and neither can re-ask for it.
    """
    if not enabled:
        imgui.begin_disabled()
    clicked = imgui.button(label, size)
    if not enabled:
        imgui.end_disabled()
    note = reason if not enabled else tooltip
    hovered = bool(imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value))
    if note and hovered:
        imgui.set_tooltip(note)
    return clicked and enabled, hovered


def disabled_button(
    label: str,
    enabled: bool,
    size: tuple[float, float] = (0, 0),
    *,
    reason: str = "",
    tooltip: str = "",
) -> bool:
    """A button that is visibly unavailable rather than absent.

    Absent controls make a UI feel like it is hiding things; a greyed one with
    a tooltip says why. The tooltip is what this function did *not* draw for
    most of its life: the docstring promised one, the body had none, and the
    result was 92 call sites able to grey out and none able to explain it --
    "Download selected (0)" being the worst of them.

    ``reason`` is shown only while the button is disabled and ``tooltip`` only
    while it is live, so one call can carry both without either being a lie.
    Both default to "", which is why the ~90 sites that pass neither are
    unchanged: this is an addition, not a migration.

    The hover has to be asked for with ``allow_when_disabled`` -- imgui
    swallows hover on a disabled item, which is exactly the state whose
    explanation matters, and is why this cannot be a ``set_item_tooltip``
    one-liner. ``_glyph_button`` has always done it this way.

    The hover is read *before* set_tooltip can run, and stored. SetTooltip
    renders through TextV, which is an item of its own and overwrites
    g.LastItemData -- so after the call, ``is_item_hovered`` and the item-rect
    getters answer about the tooltip's text, not about this button. Asking in
    that order is the bug palette.py had at its own call site: a caller that
    follows this with a rect query would place whatever it draws next at the
    tooltip. Cheap and unconditional, so the ~90 call sites that pass no note
    cannot start depending on the order by accident. ``_glyph_button`` has
    always captured first, for this reason -- and :func:`_button_with_note`
    hands that captured answer back to the two wrappers that animate a fill
    from it.
    """
    return _button_with_note(label, enabled, size, reason=reason, tooltip=tooltip)[0]


def same_line_or_wrap(width: float) -> None:
    """Continue on this line if ``width`` still fits, otherwise start the next.

    ``same_line`` after an item drawn at ``-1`` leaves the cursor *exactly* on
    the content region's right edge, so whatever comes next is drawn past it
    and clipped away -- which is where every findings hint in both generate
    panes, and every ``(?)`` in the 3D pane's Mesh section, was going. The
    panes already knew this in one place (the 2D platform combo is narrowed by
    30 px "to leave room for the marker"); this asks the layout instead of
    remembering it per call site, which is what a full-width control one pane
    over cannot be relied on to do.
    """
    imgui.same_line()
    if imgui.get_content_region_avail().x < width:
        imgui.new_line()


def button_width(label: str) -> float:
    """How wide imgui will draw a text button, for ``same_line_or_wrap``.

    Every call site that wrapped before this measured its own button as
    ``calc_text_size(label).x`` plus a hand-picked ``sp(...)`` fudge for the
    frame padding. Asking the style is the same rule ``grid_width`` was written
    for: a guess that is right at UI scale 1.0 is wrong at 1.6, and 1.0 is the
    scale the smoke suite runs at, so the error is invisible exactly where it
    is checked. ``##``-suffixed ids are stripped, since imgui does not draw
    them.
    """
    return (
        imgui.calc_text_size(label.split("##")[0]).x
        + imgui.get_style().frame_padding.x * 2
    )


def grid_width(columns: int) -> float:
    """The per-button width of an ``n``-across grid laid out with ``same_line``.

    Asks the style for the gap rather than assuming one. Every call site used to
    subtract a literal ``8``, which is ``tokens.SP_2`` *unscaled* -- correct at
    UI scale 1.0 and wrong everywhere else, because ``theme.apply`` sets
    ``item_spacing`` through ``sp()``. On a 1.6x display each gap is 4.8 px
    wider than the arithmetic budgeted for, so a five-across row overruns its
    pane by ~19 px and the last column is clipped in half: the Inker toolbox
    lost the spray can, the marquee and the eyedropper, and Clay's tool row lost
    its fourth button. It is invisible at 1.0, which is the scale the smoke
    suite runs at.

    The same reasoning as ``same_line_or_wrap``: ask the layout, do not
    remember its numbers per call site.
    """
    gap = imgui.get_style().item_spacing.x
    return (imgui.get_content_region_avail().x - gap * (columns - 1)) / columns


# Grade buttons per row. Six, so the eleven-value scale lays out 6/5 rather
# than leaving one button alone on a third line.
GRADES_PER_ROW = 6

# The scale itself, derived from the constants ``vectors.py`` owns rather than
# spelled again here -- widening the scale there must move this row, not leave
# it drawing last week's buttons. ``vectors`` is pure stdlib, so the import
# costs the frame loop nothing.
GRADES = tuple(range(GRADE_MIN, GRADE_MAX + 1))


def grade_buttons(id_prefix: str, enabled: bool) -> int | None:
    """The -5..+5 mesh grade row. -> the grade clicked, or ``None``.

    Eleven buttons, worst on the left, so the control is the scale: a reviewer
    reads the position rather than the label. Eleven do not fit one row in a
    260 px sidebar at any UI scale, so they wrap at a **counted** six per row
    rather than on measured space.

    That count is the fix rather than the shortcut. Wrapping on
    ``same_line_or_wrap`` is what the rest of this module does and it is right
    where the item widths are not known in advance -- but here they are exactly
    ``grid_width(6)``, and six of those plus five gaps comes to the content
    width to within a rounding error, so the measured test failed on the sixth
    button and laid the scale out 5/5/1. Counting gives 6/5 and, more to the
    point, gives *the same* 6/5 at every UI scale and sidebar width, which a
    control the user reads positionally has to.

    Widths still come from ``grid_width`` for the reason it exists: the gap is
    a style value, not the literal 8 that is only right at scale 1.0.

    Zero is drawn as ``0`` and everything else signed, which is
    ``review_mode.grade_text`` and is *imported* rather than re-spelled here --
    the unit list, the inspector's toast and this row must agree about how a
    grade is written.
    """
    from .review_mode import grade_text

    clicked: int | None = None
    width = grid_width(GRADES_PER_ROW)
    for index, grade in enumerate(GRADES):
        if index % GRADES_PER_ROW:
            imgui.same_line()
        if disabled_button(f"{grade_text(grade)}##{id_prefix}-grade{grade}", enabled,
                           (width, 0), tooltip=grade_key_hint(grade)):
            clicked = grade
    return clicked


def grade_key_hint(grade: int) -> str:
    """How this grade is typed. "" when the row has no keyboard behind it.

    A hover rather than a suffix in the label: the row is read *positionally*
    (see :func:`grade_buttons`), the buttons are ``grid_width(6)`` wide, and
    "+1 (1)" in a 260 px sidebar is how a scale stops looking like a scale.

    The negative arm is the half worth saying at all. A magnitude is its own
    digit, but a *minus* is R first -- which nothing on screen said, and which
    is one keypress away from filing the opposite verdict.

    The "" case is unreachable today and is the point of the bound below:
    ``review_mode.GRADE_KEYS`` binds the digits 1..``GRADE_MAX``, so the scale
    is fully typeable only while ``GRADE_MIN == -GRADE_MAX``. Widen one end of
    the scale alone -- and ``docs/measurements`` is where that argument would
    be made -- and every row past the digits would otherwise keep promising a
    keystroke that files nothing. Saying nothing is the honest answer for a row
    that has no key; ``grade_buttons`` already draws a hintless row.
    """
    if grade == 0:
        return "Key: 0"
    if abs(grade) > GRADE_MAX:
        return ""
    if grade > 0:
        return f"Key: {grade}"
    return f"Keys: R then {abs(grade)}"


def tag_toggles(id_prefix: str, pending: list[str], enabled: bool) -> str | None:
    """The two tag vocabularies as toggle rows. -> the tag clicked, or ``None``.

    Two rows rather than one list, because the polarity is the thing a reader
    needs before the word: "good-texture" and "bad-texture" differ by three
    characters and mean opposite things. The order inside each row is the
    vocabulary's own, which is also the order ``Ctrl``/``Shift`` + 1-5 press
    them in -- a hand-ordered row here would silently disagree with the
    keyboard.

    Returns the tag rather than mutating ``pending``: the toggle rule lives in
    ``review_mode.toggle_tag`` (which also drops an unknown tag rather than
    staging a verdict that would be refused), and a second copy of it in a
    drawing function is how the two come to disagree.
    """
    from ..service import verdicts as verdicts_mod

    clicked: str | None = None
    for label, vocabulary, modifier in (("Good", verdicts_mod.GOOD_TAGS, "Ctrl"),
                                        ("Bad", verdicts_mod.BAD_TAGS, "Shift")):
        field_label(f"{label}:")
        width = grid_width(3)
        for index, tag in enumerate(vocabulary):
            if index:
                same_line_or_wrap(width)
            # The chord, on a hover for ``grade_key_hint``'s reason: the row is
            # already three columns of whole words in a 260 px sidebar, and the
            # position in it *is* the digit.
            hint = f"{modifier}+{index + 1}" if index < 9 else ""
            selected = tag in pending
            if selected:
                # ``button_hovered`` as well as ``button``, or a staged tag
                # reverts to the default fill under the pointer -- which is
                # exactly when the user is deciding whether to unstage it, and
                # the one moment the state has to stay legible. ``.value``
                # because every other ``push_style_color`` in this module spells
                # it that way; the enum is an ``int`` subclass, so both work and
                # one spelling is worth more than the choice between them.
                fill = imgui.ImVec4(*theme.rgba(theme.ACCENT))
                imgui.push_style_color(imgui.Col_.button.value, fill)
                imgui.push_style_color(imgui.Col_.button_hovered.value, fill)
            if disabled_button(
                f"{tag}##{id_prefix}-tag-{tag}", enabled, (width, 0), tooltip=hint
            ):
                clicked = tag
            if selected:
                imgui.pop_style_color(2)
    return clicked


def help_marker(text: str) -> None:
    """The hover-for-more mark beside a control.

    A Lucide info glyph rather than the literal ``"(?)"`` it was (UX.md Phase
    2) -- and the *same* glyph the manual's own help button uses, which is the
    half worth stating: the two marks meant "there is more about this" and
    "there is a chapter about this", drawn as three ASCII characters and a
    circled i, so nothing about them said they were relatives.
    """
    same_line_or_wrap(imgui.calc_text_size(icons.INFO).x)
    text_colored(theme.MUTED, icons.INFO)
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


def hint_text(text: str) -> None:
    """A muted note after a control -- beside it if it fits, under it if not.

    Wrapped as well as placed: an evidence line reads
    ``"holes 3% * watertight 71% (21 meshes)"``, which is wider than a cell of
    the 2D pane's two-column guidance grid, so a hint that only moved down a
    line would still be cut off at the column edge rather than at the panel's.
    """
    same_line_or_wrap(imgui.calc_text_size(text).x)
    imgui.push_text_wrap_pos(0.0)
    secondary(text)
    imgui.pop_text_wrap_pos()


def segmented_control(
    seg_id: str,
    options: list[tuple[str, str]],
    current: str,
    *,
    breaks: Any = (),
    compact: list[tuple[str, str]] | None = None,
    max_width: float | None = None,
) -> str:
    """One rounded track with a sliding highlight; the macOS mode switch.

    Each segment is an ``invisible_button`` with a stable id
    (``{seg_id}/{key}``), which is what keeps this drivable from tests. The
    highlight's position animates; the click is honoured immediately.

    ``breaks`` is a set of indices *after* which the track is split (UX.md
    Phase 2). Splitting the track rather than merely spacing the segments is
    the point: a gap inside one continuous track reads as a missing segment,
    where two tracks read as two groups -- which is what the app's thirteen
    modes needed, since five of them are places and eight are workspaces and a
    flat row said they were all the same kind of thing. The pill still slides across
    a break, because it is one selection over all of them.

    ``compact`` is a second labelling of the same segments -- in practice the
    glyph alone -- used when the full one does not fit inside ``max_width``. It
    is **all or nothing**: a switch that abbreviated only the segments that did
    not fit would be saying two different kinds of thing in one control, and
    which two would change as the window is dragged. A compacted segment keeps
    its full label in a tooltip, or the abbreviation is a picture with no name.
    A clipped segment is an unreachable mode, which is why this is a fit
    problem rather than a tidiness one.
    """
    draw = imgui.get_window_draw_list()
    pad_x, pad_y = sp(14), sp(6)
    gap = sp(tokens.SP_2)
    breaks = frozenset(breaks)
    titles: dict[str, str] = {}
    with fonts.label(imgui):

        def measure(entries: list[tuple[str, str]]) -> list[float]:
            return [imgui.calc_text_size(label).x + pad_x * 2 for _, label in entries]

        widths = measure(options)
        if compact is not None and max_width is not None:
            spread = sum(widths) + gap * len(breaks)
            if spread > max_width:
                titles = dict(options)
                options, widths = compact, measure(compact)
        height = imgui.get_text_line_height() + pad_y * 2
        origin = imgui.get_cursor_screen_pos()
        # One offset per segment, with a gap opened at each break.
        offsets: list[float] = []
        cursor = 0.0
        for index, width in enumerate(widths):
            offsets.append(cursor)
            cursor += width + (gap if index in breaks else 0.0)
        total = cursor
        # One track per group. ``start`` walks the group boundaries, which are
        # the breaks plus the end of the list.
        start = 0
        for index in range(len(options)):
            if index not in breaks and index != len(options) - 1:
                continue
            draw.add_rect_filled(
                (origin.x + offsets[start], origin.y),
                (origin.x + offsets[index] + widths[index], origin.y + height),
                imgui.get_color_u32(theme.rgba(theme.ELEV_1)),
                height * 0.5,
            )
            start = index + 1
        # The sliding pill, at last frame's animated position.
        keys = [key for key, _ in options]
        index = keys.index(current) if current in keys else 0
        # Sprung rather than eased (UX.md Phase 5). This is the one control in
        # the app whose target routinely changes *while it is still moving* --
        # Alt+2 then Alt+5 before it has arrived -- and an exponential approach
        # has no velocity to carry through the re-aim, so it decelerated into
        # the first target and then started again towards the second. The
        # duration is the base rather than the fast one because a spring spends
        # part of it settling, and DUR_FAST left the overshoot looking like a
        # twitch instead of a stop.
        x = motion.spring(f"{seg_id}/x", offsets[index], duration=tokens.DUR_BASE)
        w = motion.spring(f"{seg_id}/w", widths[index], duration=tokens.DUR_BASE)
        draw.add_rect_filled(
            (origin.x + x + sp(2), origin.y + sp(2)),
            (origin.x + x + w - sp(2), origin.y + height - sp(2)),
            imgui.get_color_u32(theme.rgba(theme.ELEV_2)),
            (height - sp(4)) * 0.5,
        )
        selected = current
        for (key, label), width, offset in zip(options, widths, offsets, strict=True):
            imgui.set_cursor_screen_pos((origin.x + offset, origin.y))
            if imgui.invisible_button(f"{seg_id}/{key}", (width, height)):
                selected = key
            hovered = imgui.is_item_hovered()
            if hovered and key in titles:
                imgui.set_tooltip(titles[key])
            active = key == current
            alpha = motion.value(
                f"{seg_id}/{key}/text",
                1.0 if active else (0.85 if hovered else 0.55),
                duration=tokens.DUR_FAST,
            )
            text_size = imgui.calc_text_size(label)
            draw.add_text(
                (
                    origin.x + offset + (width - text_size.x) * 0.5,
                    origin.y + (height - text_size.y) * 0.5,
                ),
                imgui.get_color_u32(theme.rgba(theme.TEXT, alpha)),
                label,
            )
        imgui.set_cursor_screen_pos((origin.x, origin.y))
        imgui.dummy((total, height))
    return selected


def toggle(label: str, value: bool, *, tag: str | None = None) -> tuple[bool, bool]:
    """An animated switch. -> (changed, value).

    ``tag`` keys the animation when the visible label alone would collide.
    """
    key = tag or label
    track = sp(18)
    width = sp(32)
    # The *item* is a full frame tall even though the switch drawn inside it is
    # not, and the track is centred in that. ``same_line`` aligns items by their
    # top edge, so an 18px switch between frame-height buttons rides high and
    # drags the whole row out of line -- which is what made the viewport
    # toolbar's Frame and Screenshot icons read as sitting off-centre.
    height = max(track, imgui.get_frame_height())
    origin = imgui.get_cursor_screen_pos()
    top = origin.y + (height - track) * 0.5
    text_w = imgui.calc_text_size(label).x if label else 0.0
    clicked = imgui.invisible_button(
        f"##toggle/{key}", (width + (sp(6) + text_w if label else 0), height)
    )
    if clicked:
        value = not value
    t = motion.value(f"toggle/{key}", 1.0 if value else 0.0, duration=tokens.DUR_FAST)
    draw = imgui.get_window_draw_list()
    on = theme.rgba(theme.ACCENT)
    off = theme.rgba(theme.EDGE)
    fill = tuple(off[i] + (on[i] - off[i]) * t for i in range(3)) + (1.0,)
    draw.add_rect_filled(
        (origin.x, top), (origin.x + width, top + track), imgui.get_color_u32(fill), track * 0.5
    )
    radius = track * 0.5 - sp(2)
    knob_x = origin.x + track * 0.5 + (width - track) * t
    draw.add_circle_filled(
        (knob_x, top + track * 0.5), radius, imgui.get_color_u32(theme.rgba(0xFFFFFF)), 24
    )
    if label:
        draw.add_text(
            (origin.x + width + sp(6), top + (track - imgui.get_text_line_height()) * 0.5),
            imgui.get_color_u32(theme.rgba(theme.TEXT)),
            label,
        )
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
    return clicked, value


def _glyph_button(
    icon: str,
    side: float,
    tooltip: str,
    *,
    danger: bool = False,
    enabled: bool = True,
    borderless: bool = False,
) -> bool:
    """A square button holding one glyph, centred in it.

    The centring is the whole reason this is not four lines inline. imgui gives
    a button's label the rect ``side - 2 * frame_padding.x``, and the theme's
    horizontal padding (``theme.py``: 9 design px) is over half of a square
    that is only ``frame_padding.y`` taller than the text -- so a Lucide glyph,
    which advances about a full em, does not fit. ``RenderTextClipped`` then
    clamps the alignment offset to zero rather than centring into a negative
    gap, and the icon is pinned to the left edge with a sliver of space on the
    right and its own right-hand pixels clipped off. Every icon in the app was
    drawn that way.

    So the padding is pushed to zero *for this button only* -- which hands the
    glyph the whole square -- and the alignment is stated rather than inherited,
    because the whole point is where the glyph lands inside a rect it now fits.
    ``side`` is measured before the push: frame height is a function of the
    padding this is about to change. Neither ``ICON_OFFSET`` (font metrics --
    it is what makes an icon sit right vertically) nor the global padding (it
    shapes every text button and every modal) is any part of the fix.

    The hover is interpolated rather than switched (UX.md Phase 1). imgui's own
    hover is a one-frame jump to a fully saturated accent, which across a
    toolbar of eight icons is the single most "not Apple" thing in the app --
    so ``button`` and ``button_hovered`` are pushed to the *same* interpolated
    colour and the approach is what moves. The state is last frame's, which is
    the only order available: see :func:`_hover_amount`.

    **Where it interpolates to** is REDESIGN.md wave 1's rule, arriving here in
    wave 2 because this button pushes its own colours and so was not reached by
    the style edit: a hover is one step of the elevation ramp (ELEV_2 -> EDGE),
    not a fill of accent. Smoothing the approach fixed the *jump*; the toolbar
    of eight icons was still eight indigo primaries one at a time. ``danger``
    keeps its red, because there the colour is the warning rather than a
    highlight.

    ``borderless`` drops the resting fill entirely and lets the hover bring one
    in from nothing. That is what a *toolbar* wants -- a row of eight filled
    squares is eight boxes drawn around eight glyphs, and the glyphs are the
    content -- while a lone icon beside a text field still wants its frame, so
    this is a flag rather than a new default. The hit area is unchanged: only
    the paint goes.
    """
    key = f"glyph/{icon}/{tooltip}"
    t = _hover_amount(key)
    if borderless:
        wash = theme.ERR if danger else theme.ELEV_2
        fill = theme.rgba(wash, t * (0.25 if danger else 1.0))
    else:
        towards = theme.ERR if danger else theme.EDGE
        fill = theme.mix(theme.ELEV_2, towards, t, 1.0 - (0.2 if danger else 0.0) * t)
    imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(*fill))
    imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.ImVec4(*fill))
    pushed = 2
    if danger:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        pushed = 3
    if not enabled:
        imgui.begin_disabled()
    imgui.push_style_var(imgui.StyleVar_.frame_padding.value, imgui.ImVec2(0.0, 0.0))
    imgui.push_style_var(imgui.StyleVar_.button_text_align.value, imgui.ImVec2(0.5, 0.5))
    clicked = imgui.button(icon, (side, side))
    imgui.pop_style_var(2)
    if not enabled:
        imgui.end_disabled()
    hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value)
    imgui.pop_style_color(pushed)
    note_hover(key, hovered and enabled)
    # Focus as well as hover (UX-02). The tooltip is the *only* place a glyph
    # button says what it does, so showing it on hover alone meant the name
    # existed for pointer users and did not exist for keyboard users -- who
    # reach the same button and are told nothing but a shape.
    #
    # Gated on ``nav_visible`` and not on focus alone. Focus is held by some
    # control from the first frame, so the looser test popped a tooltip beside
    # the header on startup, before anybody had pressed anything -- caught in a
    # screenshot rather than by a test, since nothing headless has a nav
    # cursor. ``nav_visible`` is imgui's own "the user is navigating by
    # keyboard right now", which is exactly the audience this is for.
    named = hovered or (imgui.is_item_focused() and imgui.get_io().nav_visible)
    if tooltip and named:
        imgui.set_tooltip(tooltip)
    return clicked and enabled


def icon_button(
    icon: str,
    tooltip: str,
    *,
    danger: bool = False,
    enabled: bool = True,
    borderless: bool = False,
) -> bool:
    """A square glyph button with its meaning in the tooltip.

    ``tooltip`` is required, and that is the accessible-name rule rather than a
    tidiness one (UX-02): a glyph button carries no visible label, so the
    tooltip is the only thing that ever says what it does. Every call site
    already passed one -- making it mandatory is what stops the twenty-fourth
    from being the exception.
    """
    return _glyph_button(
        icon,
        imgui.get_frame_height(),
        tooltip,
        danger=danger,
        enabled=enabled,
        borderless=borderless,
    )


def small_icon_button(icon: str, tooltip: str, *, borderless: bool = False) -> bool:
    """The same button at ``small_button`` height, for a card's action row.

    Its own entry point rather than a flag, but *not* its own drawing code: the
    two call sites this replaced were bare ``imgui.small_button(icons.…)``,
    which is the un-centred, un-square shape ``_glyph_button`` exists to
    correct, and an idiom that governs everywhere except two cards governs
    nothing. The side is the text line height, which is what ``small_button``
    makes its own height (it draws with zero vertical padding), so an icon
    still lines up with the labelled small buttons beside it.
    """
    return _glyph_button(
        icon, imgui.get_text_line_height(), tooltip, borderless=borderless
    )


def field_label(label: str) -> None:
    """The small caps line above a control; what makes a combo answerable."""
    with fonts.small(imgui):
        text_colored(theme.MUTED, label.upper())


def labeled_combo(label: str, value: str, options: list[tuple[str, str]], width: float = -1.0):
    """A combo that keeps saying what it is after a value is chosen."""
    field_label(label)
    return combo(f"##{label}", value, options, width)


def labeled_slider_int(label: str, value: int, low: int, high: int) -> tuple[bool, int]:
    """A full-width slider that keeps saying what it is. -> (changed, value)

    ``labeled_combo``'s rule, applied to the control it was missing. imgui
    draws a slider's label *outside* the widget, to its right, so a slider set
    to `-1` width has nowhere to put one -- and the nine sliders in the Inker
    tool and layer panes were all drawn that way, which put the user in front
    of a bare `0.850` with nothing to say it was Hardness. The value still
    reads inside the track; only the name was gone.
    """
    field_label(label)
    imgui.set_next_item_width(-1)
    return imgui.slider_int(f"##{label}", value, low, high)


def float_format(low: float, high: float, step: float | None = None) -> str:
    """The printf format a slider over ``low..high`` should draw itself with.

    imgui's default is ``"%.3f"`` for every float slider, which is how the Inker
    layer pane came to draw an opacity of *one* as ``1.000`` and a brush angle
    of 45 degrees as ``45.000``. Three decimals is right for a value that varies
    in thousandths and absurd for one that varies in whole numbers, and which of
    those a slider is, is already stated by the range it was given.

    Derived from the span rather than declared per call site, which is
    ``clay_ops.format_for``'s rule read the other way round: that one widens
    downwards from a step, this one narrows upwards from a range. ``step`` wins
    when it is given, because a slider that moves in halves over 0..100 needs a
    decimal the span alone would talk it out of.
    """
    span = abs(high - low)
    if step:
        return "%.0f" if abs(step) >= 1.0 else ("%.1f" if abs(step) >= 0.1 else "%.2f")
    if span >= 100.0:
        return "%.0f"
    return "%.1f" if span >= 10.0 else "%.2f"


def labeled_slider_float(
    label: str,
    value: float,
    low: float,
    high: float,
    *,
    fmt: str | None = None,
    percent: bool | None = None,
) -> tuple[bool, float]:
    """``labeled_slider_int`` for a float. See it for why this exists.

    Two additions, both about the number the user actually reads.

    ``percent`` draws a 0..1 fraction as 0..100 % and hands back the fraction,
    so the model keeps its natural units and the reader gets the ones they
    think in. **Inferred by default**, because every 0..1 slider in this app is
    a percentage -- opacity, hardness, flow, weight, tolerance -- and a rule
    that has to be remembered at eleven call sites is a rule that will be
    forgotten at the twelfth. Passing it explicitly is for the exception: a
    0..1 range that is genuinely a ratio and not a percentage.

    ``fmt`` overrides the format outright, for the units a number carries that
    a range cannot know about (``"%.2fx"`` for a scale factor). Otherwise
    :func:`float_format` picks one from the range, which is what stops a
    degrees-of-rotation slider reading ``45.000``.
    """
    field_label(label)
    imgui.set_next_item_width(-1)
    if percent is None:
        percent = low == 0.0 and high == 1.0
    if percent:
        changed, shown = imgui.slider_float(
            f"##{label}", value * 100.0, low * 100.0, high * 100.0, fmt or "%.0f%%"
        )
        return changed, shown / 100.0
    return imgui.slider_float(f"##{label}", value, low, high, fmt or float_format(low, high))


def primary_button(
    label: str,
    size: tuple[float, float] = (0, 0),
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> bool:
    """The accent-filled call to action; one per pane.

    Hover lightens by approach rather than by swap (UX.md Phase 1); see
    :func:`_glyph_button` for why ``button`` and ``button_hovered`` are pushed
    to one colour.

    **A disabled primary is not an accent button** (REDESIGN.md wave 2). It used
    to keep the fill and let ``begin_disabled`` fade it, which draws a faded
    indigo slab -- still the loudest thing on the pane, still reading as the one
    thing to press, and now unpressable. Generate is disabled for most of the
    time anybody spends in the 2D pane, so that was the app's most-seen button.
    Neutral ELEV_2 instead: the call to action stops calling when it cannot be
    answered, and it is the *return* of the accent that says the form is ready.
    The fade itself is still imgui's -- ``begin_disabled`` multiplies the global
    alpha by ``style.disabled_alpha`` (``tokens.DISABLED_ALPHA``), which reaches
    a pushed colour exactly as it reaches the label on top of it, so dimming the
    push by hand as well would fade it twice.

    ``reason``/``tooltip`` forward to :func:`disabled_button`'s contract, which
    is what lets a primary explain its own refusal. Nothing in the sweep test
    requires it -- that matcher keys on the name ``disabled_button`` -- so this
    is the same rule reaching the same button under its other name rather than
    a test being satisfied.
    """
    key = f"primary/{label}"
    if enabled:
        fill = imgui.ImVec4(*theme.rgba(theme.ACCENT, 1.0 - 0.15 * _hover_amount(key)))
        pressed = imgui.ImVec4(*theme.rgba(theme.ACCENT, 0.7))
    else:
        fill = imgui.ImVec4(*theme.rgba(theme.ELEV_2))
        pressed = fill
    imgui.push_style_color(imgui.Col_.button.value, fill)
    imgui.push_style_color(imgui.Col_.button_hovered.value, fill)
    imgui.push_style_color(imgui.Col_.button_active.value, pressed)
    with fonts.label(imgui):
        clicked, hovered = _button_with_note(
            label, enabled, size, reason=reason, tooltip=tooltip
        )
    note_hover(key, enabled and hovered)
    imgui.pop_style_color(3)
    return clicked


def ghost_button(
    label: str,
    size: tuple[float, float] = (0, 0),
    *,
    enabled: bool = True,
    reason: str = "",
    tooltip: str = "",
) -> bool:
    """A labelled button with nothing drawn under it until it is pointed at.

    The third register, between :func:`primary_button` (the one thing to press)
    and a plain ``disabled_button`` (a filled ELEV_2 slab). A pane with six
    equally-filled buttons has told the reader nothing about which of them
    matters, and this app's panes routinely have six: Cancel beside Apply,
    Reset beside Save, "View all" under a list. Those are all *available* and
    none of them is the point, which is a thing a fill cannot say and an absence
    can.

    It keeps the standard ``frame_padding``, so :func:`button_width` still
    measures it and a row that mixes ghosts with filled buttons still lines up
    -- the paint is the only difference, which is also why the wrapping and
    overflow arithmetic in :mod:`.toolbar` needs no second measurement path.

    Delegates the reason/tooltip contract rather than restating it: a ghost is
    the register a "not right now" action is most likely to be drawn in, so it
    would be the worst of the three to leave unable to explain itself.
    """
    key = f"ghost/{label}"
    fill = imgui.ImVec4(*theme.rgba(theme.ELEV_2, _hover_amount(key)))
    imgui.push_style_color(imgui.Col_.button.value, fill)
    imgui.push_style_color(imgui.Col_.button_hovered.value, fill)
    with fonts.label(imgui):
        clicked, hovered = _button_with_note(
            label, enabled, size, reason=reason, tooltip=tooltip
        )
    note_hover(key, enabled and hovered)
    imgui.pop_style_color(2)
    return clicked


def destructive_button(
    label: str, size: tuple[float, float] = (0, 0), *, enabled: bool = True
) -> bool:
    """Red where it acts, not where it cancels.

    ``enabled`` greys it exactly as ``disabled_button`` does. It takes the same
    shape deliberately: a caller writing ``if destructive_button(...) and
    enabled:`` draws a live red button that swallows the click, which is the
    one button in the app where "nothing happened" is hardest to tell from
    "something irreversible happened".
    """
    if not enabled:
        imgui.begin_disabled()
    key = f"destructive/{label}"
    fill = imgui.ImVec4(*theme.rgba(theme.ERR, 0.85 + 0.15 * _hover_amount(key)))
    imgui.push_style_color(imgui.Col_.button.value, fill)
    imgui.push_style_color(imgui.Col_.button_hovered.value, fill)
    imgui.push_style_color(
        imgui.Col_.button_active.value, imgui.ImVec4(*theme.rgba(theme.ERR, 0.7))
    )
    with fonts.label(imgui):
        clicked = imgui.button(label, size)
    note_hover(key, enabled and imgui.is_item_hovered())
    imgui.pop_style_color(3)
    if not enabled:
        imgui.end_disabled()
    return clicked and enabled


@contextmanager
def card(card_id: str, size: tuple[float, float]):
    """An elevated child: soft shadow, raised fill, hover lift.

    The shadow is :func:`shadow` at ``resting``, on the parent's draw list and
    drawn *before* the child so it sits beneath it; hover state comes from last
    frame via the motion dict, because this frame's hover is not known until the
    child has been drawn.
    """
    origin = imgui.get_cursor_screen_pos()
    radius = sp(tokens.RADIUS_L)
    # Peeked, not advanced: the target is set after ``end_child``, once hover is
    # known. Reading through ``value`` (with a target of 0.0, which is what this
    # used to do) stepped the easing twice a frame and towards the wrong target
    # first, so a card took visibly longer to lift than to settle.
    lift = motion.peek(f"card/{card_id}/lift")
    shadow(
        (origin.x, origin.y),
        (origin.x + size[0], origin.y + size[1]),
        radius,
        "resting",
        strength=0.6 + 0.4 * lift,
    )
    imgui.push_style_var(imgui.StyleVar_.child_rounding.value, radius)
    # Interpolated rather than switched at 0.5: the shadow fades continuously
    # and the fill did not, so mid-hover the background popped a step while the
    # shadow was still on its way -- one visible artifact on the app's first
    # screen, six times over.
    fill = theme.mix(theme.ELEV_1, theme.ELEV_2, lift)
    # Continuous corners (UX.md Phase 5). The card is where this item earns its
    # keep: it is the app's most numerous surface and the one drawn at
    # ``RADIUS_L``, which is the radius the difference is visible at. Drawn on
    # the *parent's* list before the child begins, exactly as the shadow is, so
    # the child's own content lands over it -- and imgui's fill and border are
    # both cleared, because a circular-arc hairline over a superelliptical fill
    # disagrees with itself at precisely the four corners this is about.
    squircle = surface_fill(
        imgui.get_window_draw_list(),
        origin,
        (origin.x + size[0], origin.y + size[1]),
        radius,
        fill,
        border=theme.rgba(theme.EDGE),
    )
    if squircle:
        imgui.push_style_var(imgui.StyleVar_.child_border_size.value, 0.0)
    imgui.push_style_color(
        imgui.Col_.child_bg.value,
        imgui.ImVec4(0.0, 0.0, 0.0, 0.0) if squircle else imgui.ImVec4(*fill),
    )
    # No scrollbar. A card is a fixed-size tile whose content is laid out to
    # fit, so a scrollbar in one is a symptom rather than an affordance -- and
    # it is a *self-worsening* one, because it takes its width out of the
    # content region and pushes the content it appeared for further over. Every
    # tile on the landing screen was drawing one over a few pixels of overflow:
    # the app's first screen, every launch, six grey slivers.
    visible = imgui.begin_child(
        card_id,
        size,
        imgui.ChildFlags_.borders.value,
        imgui.WindowFlags_.no_scrollbar.value,
    )
    try:
        yield visible
    finally:
        imgui.end_child()
        imgui.pop_style_color()
        # Two vars when the squircle drew its own border, one otherwise: the
        # count has to match what was pushed, and imgui asserts rather than
        # forgives -- which on a release build is an access violation.
        imgui.pop_style_var(2 if squircle else 1)
        hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_blocked_by_active_item.value)
        motion.value(f"card/{card_id}/lift", 1.0 if hovered else 0.0, duration=tokens.DUR_FAST)


# How wide a centred block of prose is allowed to get before it wraps. A hint
# is one or two sentences and a measure of ~60 characters is where a reader
# stops having to travel to find the next line; more to the point, an empty
# state is centred, and centred text that runs the full width of a 1600 px
# viewport has a *ragged left edge* two hundred pixels from where the eye
# expects it. In design px, so it is 360 of them at every UI scale.
PROSE_WIDTH = 360.0


def _centre_cursor(width: float) -> None:
    """Put the cursor where an item of ``width`` starts, to be centred."""
    imgui.set_cursor_pos_x(
        max(imgui.get_cursor_pos_x() + (imgui.get_content_region_avail().x - width) * 0.5, 0.0)
    )


def centred_text(text: str, colour: int = theme.MUTED, *, wrap: bool = False) -> None:
    """One line -- or one wrapped paragraph -- centred in the region.

    ``wrap`` is the whole reason this is not two lines inline at each call site.
    ``calc_text_size`` measures the *unwrapped* string, so centring on it and
    then letting imgui wrap put a two-line hint's first line off-centre and its
    second line anywhere at all: the width the block occupies is not the width
    the measurement was taken of. Measuring at the wrap width and pushing the
    same figure as the wrap position is the only way for the two to agree.
    """
    if not wrap:
        _centre_cursor(imgui.calc_text_size(text).x)
        text_colored(colour, text)
        return
    limit = min(imgui.get_content_region_avail().x, sp(PROSE_WIDTH))
    # ``text_end`` is the second positional in this binding, not the
    # hide-after-## flag -- passing the wrap width third silently measures the
    # unwrapped string in a build that accepts it.
    _centre_cursor(imgui.calc_text_size(text, None, False, limit).x)
    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + limit)
    text_colored(colour, text)
    imgui.pop_text_wrap_pos()


def empty_state(
    icon: str,
    title: str,
    hint: str = "",
    *,
    action: tuple[str, Any] | None = None,
) -> None:
    """A centred nothing-here: what this region is for, and what to do next.

    Three registers rather than one (UX.md Phase 4). It used to be three lines
    of muted text at three sizes, which says "there is nothing here" in the
    typography as well as in the words -- and an empty state is structurally
    everywhere in this app (an empty viewport, an empty library, an empty
    trash, a filter that matches nothing), so it is one of the most-seen
    surfaces there is. The glyph stays muted and grows, the title takes the
    body colour at heading size because it is the one loud thing in an
    otherwise empty region, and the hint -- the half that says what to *do* --
    stays small and muted under it.

    ``action`` is that same half as a *button* rather than as a sentence, for
    the states where there is exactly one thing to do. It is optional because
    most empty states are the answer to a filter or a selection and have no
    action to offer -- "nothing matches this search" is not a button -- and it
    is a single pair rather than a list because an empty region offering three
    equal choices is a menu, which is :func:`nothing_open`'s job.
    """
    avail = imgui.get_content_region_avail()
    imgui.dummy((0, max(avail.y * 0.5 - sp(40), 0)))

    with fonts.display(imgui):
        centred_text(icon)
    imgui.dummy((0, sp(tokens.SP_2)))
    with fonts.heading(imgui):
        centred_text(title, theme.TEXT)
    if hint:
        imgui.dummy((0, sp(tokens.SP_1)))
        with fonts.small(imgui):
            centred_text(hint, wrap=True)
    if action is not None:
        label, on_click = action
        imgui.dummy((0, sp(tokens.SP_4)))
        width = max(button_width(label), sp(160))
        _centre_cursor(width)
        if primary_button(label, (width, 0)):
            on_click()


# How wide the stacked buttons under a "nothing open" screen are drawn. The
# four panes that had their own copy of this screen all chose 240 design px
# independently, which is as close to a considered answer as a copy-paste gets:
# wide enough for "New custom size..." and narrow enough not to read as a
# banner. Kept as the shared number rather than re-derived per label, because a
# stack of buttons at three different widths is a ragged column.
ACTION_WIDTH = 240.0


def nothing_open(
    hint: str,
    actions: Any = (),
    *,
    recent_paths: Any = (),
    on_open: Any = None,
) -> None:
    """The workspace screen for "there is no document here yet".

    Four panes -- Inker, Clay, Plotter and Packwright -- drew this, and drew it
    four times: the same ``dummy(40)``, the same bare ``text("Nothing open")``,
    the same muted sentence, the same 240 px buttons, the same recent list with
    the same full-path-in-the-id trick. Copies drift, and these had: Clay's was
    written from Inker's and *dropped the sp() scaling*, so at 150 % it drew
    240-physical-pixel buttons under 1.5x labels; only Inker's offered recents;
    none of the four used :func:`empty_state`, so the app's four most-seen empty
    screens were the four that did not look like the rest of the app's empty
    screens.

    ``actions`` is ``(label, callback)`` pairs, most important first: the head
    of the list is the primary and the tail are ghosts. That ordering is the
    whole hierarchy claim -- "New map" and "Open a file..." are not two equal
    options, and drawing them as two identical filled buttons said they were.

    ``recent_paths``/``on_open`` are optional together; the path goes in the
    imgui id rather than only in the label, because two documents sharing a
    basename is ordinary and one id between them is one row.
    """
    from pathlib import Path

    empty_state(icons.SQUARE_DASHED, "Nothing open", hint)
    width = sp(ACTION_WIDTH)
    for index, (label, on_click) in enumerate(actions):
        imgui.dummy((0, sp(tokens.SP_2 if index else tokens.SP_4)))
        _centre_cursor(width)
        draw = primary_button if index == 0 else ghost_button
        if draw(label, (width, 0)):
            on_click()
    if not recent_paths or on_open is None:
        return
    imgui.dummy((0, sp(tokens.SP_4)))
    with fonts.small(imgui):
        centred_text("Recent")
    for path in list(recent_paths)[:6]:
        _centre_cursor(width)
        if imgui.selectable(f"{Path(path).name}##{path}", False, 0, (width, 0))[0]:
            on_open(path)
        if imgui.is_item_hovered():
            imgui.set_tooltip(path)


# What a toast's ``action`` draws as. A name with no entry here draws nothing,
# so an action the UI has not learned yet degrades to a plain toast rather than
# to a button that does something unexpected.
TOAST_ACTIONS = {
    "log": "Open log",
    "show": "Show",
    "review": "Review",
    # Forgiveness over interrogation (UX.md Phase 3). The soft-delete trash has
    # existed since migration 9 and the card's Delete already asks nothing --
    # the argument being that the trash *is* the confirmation. What was missing
    # was the other half of that bargain: the trash is a place you have to know
    # about and navigate to, so "it is undoable" was true and invisible. An
    # Undo inside the toast's own dwell is the offer the no-confirm rule was
    # trading on.
    "undo": "Undo",
}

# How many toasts are on screen at once. The rest are counted, not dropped
# (H69): five stacked notices already reach a third of the way up the window,
# and a burst -- a sweep landing, a bulk delete -- would otherwise cover the
# viewport it is reporting about.
TOAST_VISIBLE = 5


def toast_style(level: str) -> tuple[int, str]:
    """``(background colour, leading glyph)`` for a toast level (H68).

    One table rather than a chain of ``if level ==``, because the colour, the
    glyph, the dwell time and the stickiness are four facts about one level and
    three of them already live in ``state``. An unknown level renders as an
    ordinary notice, which is what makes a new level safe to raise before this
    file has heard of it.
    """
    return {
        "success": (theme.OK, icons.CIRCLE_CHECK),
        "warn": (theme.WARN, icons.TRIANGLE_ALERT),
        "error": (theme.ERR, icons.CIRCLE_ALERT),
    }.get(level, (theme.ELEV_2, ""))


def toasts(
    state: Any, viewport_size: tuple[float, float], on_action: Any = None
) -> None:
    """Stacked bottom-right, newest lowest; born sliding up, dying fading out.

    Info and success toasts stay ``no_inputs`` -- they never mattered enough to
    click -- unless they carry an action, which would otherwise be a button the
    mouse passes straight through. ``state.TOAST_STICKY`` levels always take
    input so their close button works: eight seconds is enough to read a
    sentence, not always enough to act on one.

    **Hovering pauses the clock** (H69), by pushing ``born`` forward by the
    frame's own delta rather than by recording a pause timestamp: age is read
    from ``born`` in four places, and a second field saying "but not that age"
    is how two of them come to disagree. Only a toast that takes input can be
    hovered at all, which is the same set that is worth pausing.
    """
    state.expire_toasts()
    if not state.toasts:
        return
    now = time.monotonic()
    delta = imgui.get_io().delta_time
    margin = sp(16)
    y = viewport_size[1] - margin
    dismissed: list[Any] = []
    hidden = max(0, len(state.toasts) - TOAST_VISIBLE)
    for toast in reversed(state.toasts[-TOAST_VISIBLE:]):
        age = now - toast.born
        fade_in = min(age / 0.18, 1.0)
        fade_out = min(max(toast.ttl - age, 0.0) / 0.3, 1.0)
        alpha = min(fade_in, fade_out)
        # The slide-up, through the shared curve rather than through its own
        # copy of it -- this was ``1 - (1 - t)**3`` written out, which is
        # ``ease_out_cubic`` spelled a second time, and the second spelling is
        # what reduce-motion could not reach. Under it the toast still fades,
        # because a fade is not movement; only the travel goes.
        travel = 0.0 if motion.REDUCED else sp(10)
        rise = motion.ease_out_cubic(fade_in) * travel - travel
        sticky = toast.level in app_state.TOAST_STICKY
        colour, glyph = toast_style(toast.level)
        imgui.set_next_window_bg_alpha(0.96 * alpha)
        imgui.set_next_window_pos(
            (viewport_size[0] - margin, y - rise), imgui.Cond_.always.value, (1, 1)
        )
        imgui.set_next_window_size((sp(320), 0))
        imgui.push_style_color(imgui.Col_.window_bg.value, imgui.ImVec4(*theme.rgba(colour)))
        imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
        )
        label = TOAST_ACTIONS.get(toast.action or "") if on_action is not None else None
        if not sticky and not label:
            flags |= imgui.WindowFlags_.no_inputs.value
        if imgui.begin(f"##toast{id(toast)}", None, flags)[0]:
            # Raised, and the alpha rides the toast's own fade: a shadow that
            # stayed solid while the notice faded would be a rectangle of dark
            # left hanging over the viewport for the last 300 ms of its life.
            window_shadow("raised")
            if imgui.is_window_hovered(imgui.HoveredFlags_.child_windows.value):
                # Paused, not extended: the clock resumes where it stopped when
                # the mouse leaves.
                toast.born += delta
            if sticky:
                # A real icon button rather than a small_button holding the
                # letter x (UX.md Phase 2): ``small_icon_button`` is the idiom
                # that centres a glyph in a square, and this was the last
                # hand-spelled close in the app.
                if small_icon_button(f"{icons.X}##toast-close{id(toast)}", "Dismiss"):
                    dismissed.append(toast)
                imgui.same_line()
            if glyph:
                text_colored(theme.TEXT, glyph)
                imgui.same_line()
            imgui.text_wrapped(toast.text)
            # Its own line: the text above it wraps to the toast's full width,
            # so a same_line here would start past the right edge.
            if label and imgui.small_button(f"{label}##toast-action{id(toast)}"):
                on_action(toast.action, toast.action_arg)
                # Acted on, so done with: leaving it up invites a second click
                # that opens a second copy of the same file.
                dismissed.append(toast)
        height = imgui.get_window_height()
        imgui.end()
        imgui.pop_style_var()
        imgui.pop_style_color()
        y -= height + sp(8)
    if hidden:
        # Above the stack, in the direction the older ones went. A count rather
        # than a scrollable list: the full record is the history in the
        # diagnostics popup, and this line exists to say the stack is a window
        # onto something rather than the whole of it.
        imgui.set_next_window_bg_alpha(0.0)
        imgui.set_next_window_pos((viewport_size[0] - margin, y), imgui.Cond_.always.value, (1, 1))
        if imgui.begin(
            "##toast-more",
            None,
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_inputs.value
            | imgui.WindowFlags_.no_focus_on_appearing.value,
        )[0]:
            muted(f"+{hidden} more")
        imgui.end()
    for toast in dismissed:
        if toast in state.toasts:
            state.toasts.remove(toast)
