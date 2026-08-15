"""Sprite sheets from a single drawing: the form, and the drafts it produces.

The 2D counterpart to :mod:`.sheet_panel`. That one renders a *mesh* from every
direction, which is exact because the geometry already exists; this one asks
SDXL to imagine the three views a flat drawing has never shown, which is a
guess -- so the deliverable is a *pair* the user picks between, and drafts
accumulate rather than overwriting each other. Nothing here throws a candidate
away; the only way one leaves is the Delete button beside it.

Drafts are write-once (``rigging.sprite_draft_path``), which is what makes the
directory-mtime stamp on the listing sound: a record can appear or disappear,
but never change under a cached copy of itself.
"""

from __future__ import annotations

import logging
from typing import Any

from imgui_bundle import imgui

from ... import rigging
from ...service import sprites as svc_sprites
from ...service import validation
from .. import controls, forms, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import model_gate, stamps

#: How wide a candidate thumbnail is drawn, before the integer pixel scale
#: below rounds it down to a whole multiple of the atlas's own size.
THUMB_SIZE = 96

log = logging.getLogger(__name__)


def _pixel_scale(size: tuple[int, int], avail: int) -> int:
    """``inspector.pixel_scale``, repeated rather than imported.

    :mod:`.inspector` imports this module, so importing it back would be a
    cycle -- and the function is one ``max`` whose whole content is its
    docstring's rule. A lazy import inside the draw would work and would also
    mean an import statement executing sixty times a second.
    """
    return max(1, avail // max(size[0], size[1], 1))


def draw(ctx: Any, job: Any) -> None:
    if job.get("stage") != "reference" or job.get("status") != "done":
        return
    if "input.png" not in set(job.get("files") or []):
        return
    if not widgets.header("Sprite sheet", default_open=False):
        return
    manual_render.help_button(ctx, "sprites")

    job_id = job["id"]
    form = _form(ctx, job_id)
    with forms.Form("sprite-settings") as form_ui:
        _controls(ctx, form, form_ui)
        _submit(ctx, job_id, form)
    _running(ctx, job_id)
    _drafts(ctx, job_id)


def _form(ctx: Any, job_id: str) -> dict[str, Any]:
    """The request, kept on the app state so it survives a reselect.

    Rebuilt when the selection moves, for ``sheet_panel._form``'s reason: the
    seeds are this attempt's, and carrying them to another asset would offer a
    "Reroll" that had already happened somewhere else.
    """
    form = ctx.state.preview.get("sprite_form")
    if form is None or form.get("job_id") != job_id:
        defaults = (svc_sprites.sprite_options() or {}).get("defaults") or {}
        form = {
            "job_id": job_id,
            "sheet_type": str(defaults.get("sheet_type") or "turnaround"),
            "logical_size": int(defaults.get("logical_size") or 64),
            "colors": int(defaults.get("colors") or 32),
            "seed_a": validation.random_seed(),
            "seed_b": validation.random_seed(),
        }
        ctx.state.preview["sprite_form"] = form
    return form


def _controls(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    options = svc_sprites.sprite_options()
    types = options.get("sheet_types") or []
    labels = {
        entry["key"]: f"{entry['key']} ({entry['columns']}x{entry['rows']})"
        for entry in types
    }
    _changed, form["sheet_type"] = form_ui.combo(
        "sheet_type",
        "Type",
        form["sheet_type"],
        [(entry["key"], labels[entry["key"]]) for entry in types],
    )
    _changed, logical_size = form_ui.combo(
        "logical_size",
        "Cell size",
        str(form["logical_size"]),
        [(str(s), f"{s} px") for s in options.get("logical_sizes") or ()],
    )
    form["logical_size"] = int(logical_size)
    _changed, colors = form_ui.combo(
        "colors",
        "Palette",
        str(form["colors"]),
        [(str(n), f"{n} colours") for n in options.get("colors") or ()],
    )
    form["colors"] = int(colors)
    for field in ("seed_a", "seed_b"):
        imgui.push_id(field)
        changed, seed = form_ui.number(field, field, int(form[field]))
        if changed:
            form[field] = max(0, seed)
        # Asked rather than assumed, the rule the rest of the sidebar follows:
        # 300px is not enough for a long seed and a button on one line, and a
        # bare same_line() puts the button off the panel edge where it cannot
        # be clicked at all.
        if controls.small_button("Reroll", role=controls.ButtonRole.GHOST):
            form[field] = validation.random_seed()
        imgui.pop_id()


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"sprite:{job_id}"
    busy = ctx.busy(key)
    # Said before the button rather than after it is pressed: a synthesis needs
    # four registry rows and the door refuses one at a time.
    locked = model_gate.draw(ctx, svc_sprites.SPRITE_ROWS, what="A sprite sheet")
    if busy:
        widgets.spinner()
        imgui.same_line()
    if widgets.disabled_button(
        "Generate two drafts",
        not busy and not locked,
        reason="A synthesis is already running for this drawing."
        if busy
        else "The weights this needs are not installed; see the note above.",
    ):
        ctx.submit(
            key,
            svc_sprites.create_sprite_synthesis,
            ctx.svc,
            job_id,
            sheet_type=form["sheet_type"],
            logical_size=form["logical_size"],
            colors=form["colors"],
            seed_a=form["seed_a"],
            seed_b=form["seed_b"],
        )
    widgets.cost_note(
        "Two full image generations, queued behind the GPU. Both land as one "
        "draft you choose between."
    )


def _running(ctx: Any, job_id: str) -> None:
    """The queued job's bar, while it is the one running.

    Best-effort and unlabelled beyond a phase name: the queue pane is the
    authority on where a job is, and this is only here so pressing the button
    visibly does something without leaving the panel.
    """
    active = ctx.state.preview.get("sprite_active")
    if not active:
        return
    if active.get("source_job") != job_id:
        return
    progress = ctx.runtime.progress(active.get("id"))
    if not progress:
        return
    widgets.progress_bar(float(progress.get("percent") or 0.0))
    widgets.muted(str(progress.get("label") or ""))


# --- the drafts -------------------------------------------------------------


def draft_records(ctx: Any, job_id: str) -> list[dict[str, Any]]:
    """Every finished draft for this reference, re-read only when it changes.

    Stamped on the *directory*, which is sound here in a way it would not be
    for a rewritable artifact: a draft is written once under a fresh id and is
    only ever added or deleted, so a stale cached record cannot exist -- the
    only thing a missed stamp can cost is a listing that is one draft behind
    for another 50 ms. ``stamps.storable`` handles that window anyway.

    Split from the drawing half so the caching rule is assertable without a GL
    context, which is the split ``sheet_panel.pixel_record`` makes.
    """
    directory = rigging.sprite_dir(ctx.job_dir(job_id))
    stamp = stamps.stamp_ns(directory)
    cache = ctx.state.preview.setdefault("sprite_drafts", {})
    cached = cache.get(job_id)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    records = rigging.list_sprite_drafts(ctx.job_dir(job_id))
    if stamps.storable(stamp):
        cache[job_id] = (stamp, records)
    return records


def _drafts(ctx: Any, job_id: str) -> None:
    widgets.section("Drafts")
    records = draft_records(ctx, job_id)
    if not records:
        widgets.muted("no drafts yet")
        return
    for record in records:
        _draft(ctx, job_id, record)


def _draft(ctx: Any, job_id: str, record: dict[str, Any]) -> None:
    draft_id = str(record.get("id") or "")
    if not draft_id:
        return
    imgui.push_id(draft_id)
    try:
        seeds = " / ".join(
            str(c.get("seed")) for c in record.get("candidates") or []
        )
        widgets.muted(f"{record.get('sheet_type', 'sheet')} - seeds {seeds}")
        for candidate, letter in zip(
            record.get("candidates") or [], rigging.SPRITE_CANDIDATES, strict=False
        ):
            _candidate(ctx, job_id, draft_id, letter, candidate)
        if controls.small_button("Delete draft"):
            # No confirm, exactly as deleting a rendered sheet has none: a
            # draft is regenerable from the seed recorded beside it, and the
            # listing above refreshes off the directory stamp on its own.
            ctx.submit(
                f"sprite-del:{job_id}:{draft_id}",
                svc_sprites.delete_sprite_draft,
                ctx.svc,
                job_id,
                draft_id,
            )
    finally:
        imgui.pop_id()


def _candidate(
    ctx: Any, job_id: str, draft_id: str, letter: str, candidate: dict[str, Any]
) -> None:
    imgui.push_id(letter)
    try:
        path = rigging.sprite_draft_png_path(ctx.job_dir(job_id), draft_id, letter)
        texture = None
        if ctx.textures is not None:
            # ``nearest`` for the reason ``inspector._pixel`` gives: this is a
            # pixel-art atlas of 32-64px cells and a bilinear filter would show
            # the user a blurred version of the thing they are judging.
            texture = ctx.textures.get(
                f"{job_id}:sprite:{draft_id}:{letter}", path, nearest=True
            )
        if texture is None:
            widgets.muted(f"{letter}: rendering...")
        else:
            # A whole multiple or nothing, for ``inspector.pixel_scale``'s
            # reason: a fractional factor samples some source pixels twice and
            # reads as banding, which is what NEAREST is here to avoid.
            factor = _pixel_scale(texture.size, int(sp(THUMB_SIZE)))
            imgui.image(
                widgets.texture_ref(texture),
                (texture.size[0] * factor, texture.size[1] * factor),
            )
        if candidate.get("front_note"):
            widgets.muted_wrapped(str(candidate["front_note"]))
        for warning in candidate.get("warnings") or []:
            widgets.muted_wrapped(
                f"{warning.get('cell')}: {warning.get('detail')}"
            )
        if controls.small_button("Edit in Inker"):
            from .. import inker_mode

            inker_mode.open_sprite_draft(ctx, job_id, draft_id, letter)
    finally:
        imgui.pop_id()
