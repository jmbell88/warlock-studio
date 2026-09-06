"""Sprite sheets from a single drawing: the form, and the drafts it produces.

The 2D counterpart to :mod:`.sheet_panel`. That one renders a *mesh* from every
direction, which is exact because the geometry already exists; this one asks
SDXL to imagine the three views a flat drawing has never shown, which is a
guess -- so the deliverable is a *pair* the user picks between, and drafts
accumulate rather than overwriting each other. Nothing here throws a candidate
away; the only way one leaves is the Delete button beside it.

A pair up to sixteen cells, that is: past it a draft is one candidate, because
a pair of an eight-direction action is sixteen generations rather than two.
Which is why the button's label and the cost note under it are *derived* --
:func:`submit_label` and :func:`cost_text` over ``sprites.sprite_cost`` -- and
were literals promising "two drafts" and "two generations" until 2026-08-29,
both of which the panel's own Type combo could make false in either direction.

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
from .. import asset_open, controls, forms, theme, verbs, widgets
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
    if not widgets.header(
        "Sprite sheet",
        default_open=False,
        # Two reasons and both matter: it is what ``widgets.request_open``
        # addresses, so arriving from a "Show" toast can open a section the
        # user last left shut -- and it is what makes "shut" a preference
        # rather than a fact reasserted on every reselect.
        persist_key=asset_open.SPRITES_SECTION,
    ):
        return
    manual_render.help_button(ctx, "sprites")

    job_id = job["id"]
    form = _form(ctx, job_id)
    # ``errors``/``on_edit``: ``create_sprite_synthesis`` refuses by name --
    # ``sheet_type``, ``logical_size``, ``seed_b`` -- and every one of those is
    # a field on this form, so the refusal had an address and no ring at the
    # other end of it. See ``main._collect_tasks``.
    with forms.Form(
        "sprite-settings",
        errors=ctx.state.field_errors,
        on_edit=ctx.state.clear_field_error,
    ) as form_ui:
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
        # Cleared beside the rest of the per-job preview state: the marker
        # names a draft in *this* job, so carrying it to another asset would
        # decorate whichever draft happened to share the id.
        ctx.state.preview.pop("sprite_focus", None)
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


def submit_label(plan: dict[str, Any]) -> str:
    """What the button promises, from the door's own count of drafts.

    A literal "two" here was wrong for half the menu above it:
    ``spritesynth.default_candidates`` draws a pair only up to sixteen cells, so
    an eight-direction action is one draft and the button was naming a number
    the press could not produce.
    """
    if not plan.get("drawable"):
        return "Generate"
    drafts = int(plan["candidates"])
    return "Generate 1 draft" if drafts == 1 else f"Generate {drafts} drafts"


def cost_text(plan: dict[str, Any]) -> str:
    """The sentence under the button: generations, wait, and what lands.

    Every number is :func:`svc_sprites.sprite_cost`'s, including the wait --
    the pane states what the door computed, which is the rule the Create form's
    own cost line follows. The old literal claimed "two full image generations"
    for every sheet on the menu; one direction is one generation, so an
    eight-direction action is eight of them per draft.
    """
    if not plan.get("drawable"):
        return str(plan.get("refusal") or "This sheet cannot be drawn at this size.")
    generations = int(plan["generations"])
    count = (
        "One full image generation"
        if generations == 1
        else f"{generations} full image generations"
    )
    if plan["bands"] > 1:
        count += (
            f" ({plan['bands']} directions x {plan['candidates']} drafts)"
            if plan["candidates"] > 1
            else " (one per direction)"
        )
    tail = (
        "Both land as one draft you choose between."
        if plan["candidates"] > 1
        else "A sheet this size draws one draft rather than a pair."
    )
    return f"{count}, queued behind the GPU, {plan['duration']}. {tail}"


def sprite_draft_cap_reason(
    records: list[dict[str, Any]], jobs: list[dict[str, Any]], job_id: str
) -> str | None:
    """Whether one more draft would push this reference past MAX_SPRITE_DRAFTS.

    The 2026-09-06 audit, finding create2-08: this pane graded its button on
    ``busy``/``locked``/``plan["drawable"]`` alone, so
    ``create_sprite_synthesis``'s own cap (``service.sprites``,
    ``MAX_SPRITE_DRAFTS``) was discovered only after a wasted press -- and a
    paid generation -- as a fieldless ``Conflict`` toast.

    A pure function over state the frame already has, matching
    ``retarget_panel.dependent_job_reason``: ``records`` is
    ``draft_records(ctx, job_id)``, the same stamp-cached on-disk listing
    ``_drafts`` already draws (no new query -- the stamp check it costs is
    the one ``_drafts`` already pays every frame), and ``jobs`` is
    ``ctx.cache.jobs``, the frame-thread-safe window every dependent-job
    reason reads. The service counts on-disk drafts plus queued/running
    ``sprite_synthesis`` rows for this reference; this mirrors that exactly
    so the two inventories cannot disagree.
    """
    queued = sum(
        1
        for job in jobs
        if job.get("status") in ("queued", "running")
        and job.get("kind") == "sprite_synthesis"
        and (job.get("params") or {}).get("source_job") == job_id
    )
    if len(records) + queued < rigging.MAX_SPRITE_DRAFTS:
        return None
    return (
        f"This reference already holds {rigging.MAX_SPRITE_DRAFTS} sprite "
        "sheet drafts; delete one first."
    )


def _submit(ctx: Any, job_id: str, form: dict[str, Any]) -> None:
    key = f"sprite:{job_id}"
    busy = ctx.busy(key)
    # Said before the button rather than after it is pressed: a synthesis needs
    # four registry rows and the door refuses one at a time.
    locked = model_gate.draw(ctx, svc_sprites.SPRITE_ROWS, what="A sprite sheet")
    # The plan for what is *currently selected*, so the label and the note below
    # move with the two combos rather than describing the default forever.
    plan = svc_sprites.sprite_cost(form["sheet_type"], form["logical_size"])
    cap_reason = None
    if not busy:
        cap_reason = sprite_draft_cap_reason(draft_records(ctx, job_id), ctx.cache.jobs, job_id)
        if cap_reason:
            widgets.text_colored(theme.WARN, cap_reason)
    if busy:
        widgets.spinner()
        imgui.same_line()
    if widgets.disabled_button(
        submit_label(plan),
        not busy and not locked and not cap_reason and bool(plan["drawable"]),
        reason="A synthesis is already running for this drawing."
        if busy
        else (
            cap_reason
            or (
                "The weights this needs are not installed; see the note above."
                if locked
                else plan["refusal"]
            )
        ),
    ):
        # Last time's rings first: a new submit is judged on its own.
        ctx.state.clear_field_errors()
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
    widgets.cost_note(cost_text(plan))


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
        if ctx.state.preview.get("sprite_focus") == draft_id:
            # Which of several drafts is the one the toast was about.
            # ``rigging.list_sprite_drafts`` is documented oldest-first, so the
            # new one is at the *bottom* of a list the user did not watch grow
            # -- and arriving at the top of it, told the sheet was ready, is a
            # smaller version of the blank screen ``asset_open`` exists to fix.
            widgets.text_colored(theme.ACCENT, "just made")
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
        if controls.small_button(verbs.open_in("inker")):
            from .. import inker_mode

            inker_mode.open_sprite_draft(ctx, job_id, draft_id, letter)
    finally:
        imgui.pop_id()
