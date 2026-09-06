"""Troupe's right-top pane: what the selected sheet is, and another one of it.

Two jobs, and they are the same subject from both sides -- this is the pane that
answers "what am I looking at" and "give me it again, differently". The second
is the *direct* door (``service.troupe.create_charsheet``): a second sheet at
another size, or a supplied base mesh that never went through the reference
chain at all.

**The structural verdict lives here and the QA scores do not**, and the wording
keeps them apart on purpose. ``studio/troupe/qa.py`` ranks the drawing -- it is
the heatmap over the sprite, it flags the worst cells, and nothing anywhere
refuses a sheet on its account. ``pipelines/sheetcheck.py`` says a cell is
clipped at the frame edge, empty, or was never rendered: facts about the file.
"Needs repair" is the second one and says so, because a user who reads them as
one thing will either ignore both or re-render over neither.
"""

from __future__ import annotations

import time
from typing import Any

from .. import controls, icons, theme, tokens, troupe_mode, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = troupe_mode.ensure(ctx)
    widgets.section("Sheet")
    manual_render.help_button(ctx, "troupe-sheets")

    record = troupe_mode.active_sheet(ctx)
    if record is None:
        widgets.muted_wrapped("Pick a character on the left.")
        return

    columns = int(record.get("columns") or 0)
    rows = int(record.get("rows") or 0)
    size = int(record.get("frame_size") or 0)
    widgets.muted(f"{columns} x {rows} cells at {size} px")
    tags = (record.get("animation") or {}).get("tags") or []
    widgets.muted(f"{len(tags)} tagged runs")
    line = camera_line(record)
    if line:
        widgets.muted(line)
    _repair(ctx, record)
    _character(ctx, record)

    report = _pixel_report(ctx, state)
    if report:
        palette = report.get("palette_name") or report.get("palette") or ""
        widgets.muted(f"{report.get('colors', '?')} colours ({palette})")
        if report.get("orphans"):
            # Worth a line rather than hidden: a large orphan count is the
            # signal that the reduction found detail the palette could not
            # hold, which is a reason to try a bigger sprite or a wider palette.
            widgets.muted(f"{report['orphans']} stray pixels cleaned")

    imgui.dummy((0, sp(tokens.SP_2)))
    _rebuild(ctx, state)
    _rerender(ctx, state)


def camera_line(record: dict[str, Any]) -> str:
    """What this sheet was framed from, in one line. Empty when it does not say.

    From the ``camera`` block, which every charsheet has carried since it was
    added -- and falling back to the bare ``elevation`` that ``sheet.sidecar``
    has always written, because a sheet rendered before the block existed still
    knows its angle and a pane that showed nothing for it would be describing
    the format rather than the file.

    A pure function of the record so the wording is testable without a window;
    the preset's *label* comes from ``charsheet.CAMERA_PRESETS``, which is the
    one home for that table.
    """
    from ...pipelines import charsheet

    camera = record.get("camera")
    if not isinstance(camera, dict):
        elevation = record.get("elevation")
        return "" if elevation is None else f"framed at {float(elevation):g} degrees"
    labels = {key: label for key, label, _angle in charsheet.CAMERA_PRESETS}
    preset = str(camera.get("preset") or "")
    # A preset the table no longer holds is named rather than hidden: the sheet
    # was rendered at that framing whether or not this build still offers it.
    name = labels.get(preset, preset) or "custom"
    parts = [f"{name}, {float(camera.get('elevation') or 0.0):g} degrees"]
    projection = str(camera.get("projection") or "")
    if projection:
        parts.append(projection)
    pixel = camera.get("pixel_size")
    if pixel:
        parts.append(f"{int(pixel)} px sprite")
    return " -- ".join(parts)


def _repair(ctx: Any, record: dict[str, Any]) -> None:
    """The structural verdict, and it is **not** the QA heatmap.

    Two different questions over one sheet, and a pane that ran them together
    would teach the user that a red square and this pill mean the same thing.
    ``troupe/qa.py`` *ranks*: it compares each cell with its neighbours, flags
    the worst, and nothing anywhere refuses a sheet on its account -- the
    heatmap above the sprite is advice about the drawing.
    ``pipelines/sheetcheck.py`` *checks structure*: a cell is cut off at the
    frame edge, empty, or was never rendered at all. Those are facts about the
    file, and they are what "needs repair" means.

    It still does not gate anything -- the sheet plays, exports and opens in
    Inker exactly as it did. What it does is name the runs worth re-rendering,
    which is the button directly underneath.
    """
    if not troupe_mode.needs_repair(record):
        return
    widgets.pill(f"{icons.TRIANGLE_ALERT} Needs repair", theme.WARN)
    widgets.muted_wrapped(
        "Structural check: cells are missing, empty or cut off at the frame "
        "edge. This is not the QA heatmap, which ranks the drawing and never "
        "refuses anything -- re-render the runs below to fix these."
    )
    for note in troupe_mode.repair_notes(record):
        widgets.muted_wrapped(f"- {note}")


def _character(ctx: Any, record: dict[str, Any]) -> None:
    """Who this is, and the way to make another one like them.

    Only on a sheet the character registry built: one from a supplied mesh, or
    from Troupe's own form, carries no ``character`` block and there is no
    recipe to load -- so the button is absent rather than disabled, because a
    greyed control implies a state the user could reach.
    """
    from imgui_bundle import imgui

    block = record.get("character")
    if not isinstance(block, dict):
        return
    recipe = troupe_mode.recipe_of(record)
    family = str(block.get("family") or recipe.get("family") or "")
    if not family:
        return
    version = block.get("family_version")
    line = family if version is None else f"{family} v{int(version)}"
    theme_name = str(recipe.get("theme") or "")
    if theme_name:
        line += f", {theme_name}"
    widgets.muted(line)
    seed = recipe.get("seed")
    if seed is not None:
        widgets.muted(f"seed {int(seed)}")
    if not recipe:
        return
    imgui.dummy((0, sp(tokens.SP_1)))
    if widgets.disabled_button(
        "Vary in Create",
        True,
        (-1, 0),
        tooltip="Loads this character's recipe into Create as your own "
        "settings, so a new seed, a wider palette or a longer horn makes the "
        "next one. Editing the prompt will not undo them.",
    ):
        troupe_mode.vary_in_create(ctx, record)


def _pixel_report(ctx: Any, state: Any) -> dict[str, Any]:
    """What the pixel-art pass measured about *this* atlas.

    Read off the job row that produced the sheet rather than stored in the
    sidecar: it is a measurement of one run, which is exactly what
    ``DERIVED_PARAMS`` keeps out of a rerun -- and the sidecar is the
    engine-neutral description of the atlas, not a place to file verdicts.

    **Keyed on the sheet id**, because this is a draw: a ``SCAN_LIMIT``-row page
    with per-row JSON parsing, taken under ``JobStore``'s one lock, ran once per
    frame for a measurement that cannot change while the same sheet is selected.
    A rebuild lands as a new sheet with a new id, which is what makes the id the
    whole of the key -- and the interval beside it is what lets a report the
    worker has not written *yet* still appear, without the id having changed to
    say so.
    """
    now = time.monotonic()
    if (
        state.pixel_report_cache is not None
        and state.pixel_report_key == state.sheet_id
        and now < state.pixel_report_next
    ):
        return state.pixel_report_cache
    found: dict[str, Any] = {}
    # Narrowed in SQL: filtering a mixed page in Python meant that past
    # ``SCAN_LIMIT`` newer jobs of any kind this line silently vanished for a
    # sheet that was still perfectly findable.
    for row in ctx.svc.store.list(limit=troupe_mode.SCAN_LIMIT, kind="charsheet"):
        params = row.get("params") or {}
        if params.get("sheet_id") == state.sheet_id:
            found = dict(params.get("pixel_report") or {})
            break
    state.pixel_report_cache = found
    state.pixel_report_key = state.sheet_id
    state.pixel_report_next = now + troupe_mode.SHEETS_REFRESH
    return found


_RERENDER_SLOT = "troupe_rerender_runs"


def _rerender(ctx: Any, state: Any) -> None:
    """Re-render some of this sheet's runs, keeping the rest.

    The other half of the cleanup loop: fix a clip in Poser, re-render only the
    animation it changed, and merge it into the document you have been cleaning
    up in Inker. A whole new sheet would throw those cleanups away.

    Picked as *runs* rather than as cells because a run is the unit a person
    judges and re-authors -- half a walk cycle re-rendered against the other
    half is a seam by construction.
    """
    from imgui_bundle import imgui

    runs = troupe_mode.sheet_runs(ctx)
    if not runs:
        return
    if not widgets.header("Re-render some runs", default_open=False):
        return
    chosen = ctx.state.preview.setdefault(_RERENDER_SLOT, set())
    by_animation: dict[str, list[dict[str, str]]] = {}
    for run in runs:
        by_animation.setdefault(run["animation"], []).append(run)

    for animation, entries in by_animation.items():
        if not imgui.tree_node(f"{animation}##rerender-{animation}"):
            continue
        for run in entries:
            token = f"{run['animation']}/{run['direction']}"
            _clicked, on = controls.checkbox(
                f"{run['direction']}##rr-{token}", token in chosen
            )
            if on:
                chosen.add(token)
            else:
                chosen.discard(token)
        imgui.tree_pop()

    subset = [
        {"animation": token.split("/", 1)[0], "direction": token.split("/", 1)[1]}
        for token in sorted(chosen)
    ]
    key = f"troupe-sheet:{state.job_id}"
    busy = ctx.busy(key)
    # Every run selected is a full render taking the slower path; the door
    # refuses it by name and the button says so before the press rather than
    # after it.
    everything = len(subset) == len(runs)
    if widgets.disabled_button(
        f"Re-render {len(subset)} run(s)",
        bool(subset) and not everything and not busy,
        (-1, 0),
        reason=(
            "A sheet is already being queued for this character."
            if busy
            else "That is every run -- use Build another sheet instead."
            if everything
            else "Tick the runs to re-render."
        ),
    ) and troupe_mode.rerender_runs(ctx, subset):
        chosen.clear()
    widgets.cost_note(
        "Renders only the ticked runs and copies the rest from this sheet, at "
        "this sheet's own settings. The result is a new sheet -- open it in "
        "Inker with Merge re-render to keep hand edits."
    )


def _rebuild(ctx: Any, state: Any) -> None:
    """Another sheet of the same character, at the form's current options."""
    from . import troupe_settings

    if not state.job_id:
        return
    form = state.form or {}
    if not form:
        # The left pane builds it, and it is drawn every frame the mode is --
        # but a headless caller (or the first frame after a restore) can reach
        # this first, and an empty form would submit the door's defaults under
        # the appearance of the user's choices.
        form = troupe_settings._form(state, troupe_settings._options(ctx))
    key = f"troupe-sheet:{state.job_id}"
    count = troupe_settings.cell_count(form)
    valid = 0 < count <= 512
    if widgets.disabled_button(
        "Build another sheet",
        not ctx.busy(key) and valid,
        (-1, 0),
        reason=(
            "A sheet is already being queued for this character."
            if ctx.busy(key)
            else "Select a layout of at most 512 cells."
        ),
    ):
        troupe_mode.build_sheet(ctx, state.job_id, form)
    widgets.cost_note(
        f"{count} rendered cells from the rig that already exists -- minutes of "
        "CPU, no GPU. The settings on the left are what it uses."
    )
    if count > 256:
        widgets.muted("Large sheet: over 256 cells can take substantially longer.")
