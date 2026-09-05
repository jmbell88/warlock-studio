"""Muse's results tray: the takes this mode has made, newest first.

The centre pane. What it draws is *not* a viewport -- Muse is deliberately
absent from ``modes.VIEWPORT_MODES``, because there is no asset to frame -- but a
grid of cards, one per music job row, read from the same ``ctx.cache.jobs`` the
Library reads.

**The card is an audio card**, which is the one place this departs from the
mesh-shaped candidate grid it is otherwise modelled on. Where that one offers
"Make 3D" it offers **Open in Sirens**, and where it shows a thumbnail it shows
a Play/Stop toggle -- because a picture of a waveform tells a listener nothing
that pressing play does not tell them better, and a thumbnail nobody can read is
worse than a control they can.

Filtering to this mode's own rows rather than showing every job: a tray that
listed meshes would be a second Library, and the Library is one implementation
that already exists.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, muse_mode, verbs, widgets
from ..tokens import sp

#: A card's size in design pixels. Wide enough for two lines of tags at a
#: readable width and for the three buttons beneath them -- two on a row and
#: "Make more" under them, because the third would not fit beside the second at
#: this width and a card that wraps its own buttons reads as a broken one.
CARD_W = 260.0
CARD_H = 182.0


def plan_for(ctx: Any) -> list[dict[str, Any]]:
    """This mode's rows, newest first.

    Off ``ctx.cache.jobs`` rather than a query of its own: the cache is already
    refreshed on the app's own schedule, and a second poll would be a second
    answer to "what has finished" that could disagree with the Library's.
    """
    return [job for job in reversed(ctx.cache.jobs) if job.get("kind") == "music"]


def should_draw(ctx: Any) -> bool:
    """Whether there is anything to show. -> False for a first visit."""
    return bool(plan_for(ctx))


def draw(ctx: Any) -> None:
    # Once per frame, before anything reads ``playing_job``: a take that ran to
    # its end stops being the playing one without any card having to notice.
    muse_mode.sync(ctx)
    jobs = plan_for(ctx)
    if not jobs:
        widgets.empty_state(
            icons.MUSIC,
            "No takes yet",
            "Describe the music you want above and press Generate. Each take is "
            "its own row -- keep the ones you like, delete the rest.",
        )
        return
    _grid(ctx, jobs)
    derive_popup(ctx)


def _grid(ctx: Any, jobs: list[dict[str, Any]]) -> None:
    """Cards, wrapped to the pane's width."""
    width = sp(CARD_W)
    gap = imgui.get_style().item_spacing.x
    avail = imgui.get_content_region_avail().x
    per_row = max(1, int((avail + gap) // (width + gap)))
    for index, job in enumerate(jobs):
        if index % per_row:
            imgui.same_line()
        _card(ctx, job, width)


def _card(ctx: Any, job: dict[str, Any], width: float) -> None:
    job_id = str(job["id"])
    state = muse_mode.ensure(ctx)
    with widgets.card(f"muse-take/{job_id}", (width, sp(CARD_H))):
        if imgui.is_item_clicked():
            state.selected_job = job_id
        widgets.stage_badge(job, inline=True)
        imgui.same_line()
        widgets.status_pill(str(job.get("status") or ""))
        prompt = str(job.get("prompt") or "")
        widgets.muted_wrapped(widgets.fit_text(prompt, width) if prompt else "(no tags)")
        params = job.get("params") or {}
        duration = params.get("actual_duration") or params.get("duration")
        line = [f"{float(duration):.0f}s"] if duration else []
        # **Seed and lineage, because comparing two takes is the whole job (W2,
        # 2026-09-05).** Both are already in the row -- the seed is written per
        # take by ``create_music_job`` and the parent is a column -- so this is
        # presentation and not a second query. Without it, telling two
        # near-identical generations apart meant leaving for the Library.
        seed = params.get("seed")
        if seed is not None:
            line.append(f"seed {int(seed)}")
        if line:
            widgets.secondary("  ".join(line))
        parent = job.get("parent_id")
        if parent:
            # The task rather than the parent's id, because a twelve-hex-digit
            # id is not something a person recognises and the *kind* of
            # derivation is what they are comparing against the original.
            task = str(params.get("task") or "derived")
            widgets.muted(f"{task} of {str(parent)[:8]}")
        _actions(ctx, job, job_id)


def _actions(ctx: Any, job: dict[str, Any], job_id: str) -> None:
    """Play/Stop and Open in Sirens, both dead until the WAV exists.

    Gated on the row's *status* rather than on the file: a queued take has no
    audio yet, and a button that reads the disk every frame to find that out
    would be a stat per card per frame for an answer the row already carries.
    """
    ready = str(job.get("status") or "") == "done"
    playing = muse_mode.is_playing(ctx, job_id)
    if widgets.transport(
        f"muse-{job_id}",
        playing,
        enabled=ready,
        reason="" if ready else "this take has not finished yet",
        shortcut="",
    ):
        if playing:
            muse_mode.stop(ctx)
        else:
            muse_mode.play(ctx, job_id)
    imgui.same_line()
    if widgets.ghost_button(
        verbs.open_in("sirens"),
        enabled=ready,
        reason="" if ready else "this take has not finished yet",
        tooltip="Import this track into the tracker as a sample instrument.",
    ):
        muse_mode.open_in_sirens(ctx, job_id)
    _derive_menu(ctx, job_id, ready)
    imgui.same_line()
    stems = muse_mode.has_stems(ctx, job)
    if widgets.ghost_button(
        "Stems" if not stems else "Stems ✓",
        enabled=ready and not stems,
        reason=(
            "this take has already been split"
            if stems
            else "" if ready else "this take has not finished yet"
        ),
        tooltip=(
            "Split this take into drums, bass, vocals and everything else. "
            "Needs a one-off ~320 MiB download."
        ),
    ):
        muse_mode.separate(ctx, job_id)


#: The derive popup's imgui id. One popup for the whole tray, not one per card:
#: ``MuseState.derive_job`` is what says which take it is about, and six sets of
#: controls on screen at once would be five of them about takes the user is not
#: looking at.
DERIVE_POPUP = "muse-derive"

#: The tasks in the order the menu offers them, and the sentence each one is.
#: Ordered by how often it is wanted rather than alphabetically or by the
#: sampler's own naming: "another one like this" is the press a generative mode
#: exists for, and the reference task is the one you reach for last.
#:
#: The labels are deliberately not ACE-Step's words. ``repaint`` and
#: ``audio2audio`` are mechanisms; "Repaint a section" and "Something like this"
#: are what the user is asking for. ``muse_mode.DERIVE_CONTROLS`` is where the
#: keys live, and ``_jobs_music.TASKS`` is the door's half of the same list.
DERIVE_ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        "retake",
        "Another like this",
        "Same brief, same base noise, nudged toward a fresh draw.",
    ),
    ("extend", "Extend", "Make it longer -- the only generation-side way to."),
    ("repaint", "Repaint a section", "Regenerate one window of it in place."),
    ("loop", "Make it loop", "Rewrite the joint between the end and the start."),
    ("edit", "Change the words or tags", "The same piece, to a different brief."),
    (
        "audio2audio",
        "Something like this",
        "A new piece, using this take as the reference.",
    ),
)

#: What each numeric derive control is called on screen, with its bounds and its
#: help. A table for ``_jobs_music._RANGES``' reason -- so adding a control is a
#: row, and so a label and its bound cannot end up in two places disagreeing.
#:
#: The bounds are the *door's*, restated as the widget's clamp. The door stays
#: the gate; this only stops the user reaching a number that was always going to
#: come back as a refusal.
DERIVE_FIELDS: dict[str, tuple[str, float, float, str]] = {
    "retake_variance": (
        "Variation",
        0.0,
        1.0,
        "0 is this take again; 1 is a fresh draw of the same brief.",
    ),
    "extend_left": ("Add before", 0.0, 240.0, "Seconds of new music ahead of it."),
    "extend_right": (
        "Add after",
        0.0,
        240.0,
        "Seconds of new music after it. Neither end may be longer than the"
        " take itself -- extend twice to go further.",
    ),
    "repaint_start": ("From", 0.0, 240.0, "Where the window starts, in seconds."),
    "repaint_end": ("To", 0.0, 240.0, "Where it ends. The rest is left alone."),
    "ref_audio_strength": (
        "Closeness",
        0.0,
        0.9,
        "How near the reference to stay. Past about 0.95 the model takes no"
        " sampling steps at all, which the door refuses.",
    ),
}


def _derive_menu(ctx: Any, job_id: str, ready: bool) -> None:
    """The "Make more" button and the task menu it opens."""
    if widgets.ghost_button(
        "Make more",
        enabled=ready,
        reason="" if ready else "this take has not finished yet",
        tooltip="Derive another take from this one.",
    ):
        muse_mode.ensure(ctx).selected_job = job_id
        imgui.open_popup(f"muse-more/{job_id}")
    if imgui.begin_popup(f"muse-more/{job_id}"):
        widgets.popup_chrome(_imgui=imgui)
        for task, label, note in DERIVE_ITEMS:
            clicked, _ = controls.menu_item(label, "", False)
            if imgui.is_item_hovered():
                imgui.set_tooltip(note)
            if clicked:
                muse_mode.open_derive(ctx, job_id, task)
                imgui.close_current_popup()
        imgui.end_popup()


def derive_popup(ctx: Any) -> None:
    """The one derive popup, drawn once per frame after the grid.

    *After* the grid, not inside a card: it is opened from a menu inside one,
    and a modal begun inside a card's id stack would be torn down the moment
    the grid re-wraps.

    No control here is a ``derive[...]`` key, which is the one-owner rule holding
    on a third surface: the bar is *what to make*, the recipe column is *how*,
    and this is *what to do with one finished take*. A generation setting that
    appeared here would be a fourth place to look for it.
    """
    state = muse_mode.active(ctx)
    if state is None or not state.derive_job:
        return
    if not imgui.is_popup_open(DERIVE_POPUP):
        imgui.open_popup(DERIVE_POPUP)
    opened, keep = imgui.begin_popup_modal(DERIVE_POPUP, True)
    if not keep:
        # The window's own close box. Cleared here rather than left set, or the
        # popup would reopen on the very next frame.
        muse_mode.close_derive(ctx)
    if not opened:
        return
    widgets.popup_chrome(_imgui=imgui)
    # Named after the field it is, not ``form``: ``MuseState`` holds *two*
    # dicts, and the brief's is the one every other Muse pane calls ``form``.
    # A reader arriving from the bar must not have to check which this is.
    derive = state.derive_form
    task = str(derive["task"])
    label, note = next(
        ((one[1], one[2]) for one in DERIVE_ITEMS if one[0] == task), (task, "")
    )
    widgets.popup_title(label)
    if note:
        widgets.muted_wrapped(note)

    for name in muse_mode.DERIVE_CONTROLS[task]:
        _derive_field(ctx, derive, name, task)

    widgets.divider()
    _, derive["count"] = widgets.labeled_slider_int(
        "How many",
        int(derive["count"]),
        1,
        _max_count(),
        help_text="Several cheap candidates to choose between, as on the bar.",
    )
    widgets.field_error(ctx.state, "count")

    if controls.button("Queue it", role=controls.ButtonRole.PRIMARY) and muse_mode.derive(ctx):
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel"):
        muse_mode.close_derive(ctx)
        imgui.close_current_popup()
    imgui.end_popup()


def _derive_field(ctx: Any, derive: dict[str, Any], name: str, task: str) -> None:
    """One control, from :data:`DERIVE_FIELDS` or the edit task's two fields."""
    if name in ("edit_prompt", "edit_lyrics"):
        # Empty means "keep this take's", which is why the hint says so rather
        # than the field arriving pre-filled with the parent's text: a
        # pre-filled field the user did not touch is indistinguishable from one
        # they retyped identically, and the door refuses an edit that changes
        # nothing.
        is_lyrics = name == "edit_lyrics"
        widgets.secondary("New lyrics" if is_lyrics else "New style tags")
        derive[name] = widgets.multiline(
            f"##muse-{name}",
            str(derive[name]),
            sp(62.0 if is_lyrics else 40.0),
            _max_lyrics() if is_lyrics else 400,
        )
        widgets.muted("Leave it empty to keep this take's.")
        widgets.field_error(ctx.state, name)
        return

    title, low, high, help_text = DERIVE_FIELDS[name]
    if task == "loop" and name == "repaint_end":
        # The loop task asks for a *span* -- how much of the joint to rewrite --
        # and the door reads it as a window it then centres on the roll. One
        # params key, two readings, and the reading lives where it is drawn
        # rather than as a second key meaning almost the same thing.
        title = "Joint to rewrite"
        help_text = (
            "How much music either side of the seam to recompose. At most half"
            " the take."
        )
    _, derive[name] = widgets.labeled_slider_float(
        title, float(derive[name]), low, high, help_text=help_text
    )
    widgets.field_error(ctx.state, name)


def _max_count() -> int:
    from ...service._jobs_music import MAX_COUNT

    return MAX_COUNT


def _max_lyrics() -> int:
    from ...service._jobs_music import MAX_LYRICS

    return MAX_LYRICS


__all__ = [
    "CARD_H",
    "CARD_W",
    "DERIVE_FIELDS",
    "DERIVE_ITEMS",
    "DERIVE_POPUP",
    "derive_popup",
    "draw",
    "plan_for",
    "should_draw",
]
