"""The selected asset: what it is, what it cost, and what you can take away.

The downloads section is the interesting part. Each artifact has three states
-- ready, derivable, impossible -- and the difference matters: a greyed "FBX"
with "needs Blender" under it is information, while a missing one is a mystery.
*Which* artifacts appear is a function of the job: a reference was being shown
the mesh grid, so all eight of its buttons were permanently greyed, which reads
as a broken asset rather than as a 2D one.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from ...service import derive as svc_derive
from ...service import files as svc_files
from ...service import jobs as svc_jobs
from ...service import system as svc_system
from .. import theme, widgets
from ..app_ctx import derive_key, pixel_prefs
from ..manual import render as manual_render
from ..state import format_duration
from ..tokens import sp
from . import candidates_panel, pose_panel, retarget_panel, sheet_panel

# Matches the library card's thumbnail, so the two read as the same kind of
# object rather than as two different image widgets. Design px: every use of it
# goes through ``sp()``, as the library's own does -- a raw literal here was
# the one size in the pane that ignored the UI scale.
THUMB_SIZE = 96

# How tall a single reference image may get. The reference now fills the pane's
# width, and at full width a portrait is taller than the pane -- three thumbs is
# enough to read the composition by and leaves the sections under it reachable
# without a scroll to the bottom.
REFERENCE_MAX_THUMBS = 3


def draw(ctx: Any) -> None:
    from .. import icons

    if ctx.state.mode == "3d":
        # Above the header, and above the "select an asset" empty state, on
        # purpose: an undecided candidate group is a question being asked, and
        # its rows are hidden from the library -- so this is the only way back
        # to them, and it has to be visible whatever is (or is not) selected.
        candidates_panel.draw(ctx)
    job = ctx.job()
    if job is None:
        widgets.empty_state(icons.BOX, "Select an asset.", "Its details and exports live here.")
        return

    # Identity stays above the tabs: whatever tab is open, the header answers
    # "what am I looking at". Everything below it moved from one nine-header
    # scroll column into three tabs, because reaching the sprite sheet on a
    # rigged mesh meant scrolling past the whole pose editor.
    _header(ctx, job)
    manual_render.help_button(ctx, "inspector")
    _meta(ctx, job)
    if job.get("status") == "error":
        _error(ctx, job)

    if ctx.state.mode == "3d":
        widgets.tab_bar(
            "inspector-tabs",
            [
                ("Details", lambda: _details_tab(ctx, job)),
                ("Rig && Pose", lambda: _rig_tab(ctx, job)),
                ("Export", lambda: _downloads(ctx, job)),
            ],
        )
    else:
        widgets.tab_bar(
            "inspector-tabs",
            [
                ("Details", lambda: _details_tab(ctx, job)),
                ("Export", lambda: _downloads(ctx, job)),
            ],
        )


def can_edit_in_clay(job: Any) -> bool:
    """Whether "Open in Clay" belongs on this job.

    The library's overflow menu gates on ``model.glb`` being listed and this is
    the same question asked from the same cached row -- no filesystem call,
    because the inspector asks it every frame. ``build.wblk`` is deliberately
    not consulted: it is never listed, and ``clay_mode.edit_asset_in_clay``
    prefers it on the task thread where a stat is free.
    """
    if not isinstance(job, dict):
        return False
    return job.get("status") == "done" and "model.glb" in (job.get("files") or [])


def _edit_actions(ctx: Any, job: Any) -> None:
    """Take this asset somewhere it can be edited, from where it was made.

    Both halves already existed and were reachable only from the library's
    overflow menu -- which is the wrong place for the moment they are wanted,
    the one straight after a generation finishes with the result still on
    screen. Wiring only: the Inker gate and the Clay open are the functions
    those two modes own, called verbatim, so the confirm Clay puts in front of
    a 200k-triangle import still fires here.
    """
    from .. import clay_mode, icons, inker_mode

    if inker_mode.can_edit_job(ctx, job):
        if imgui.button(f"{icons.BRUSH} Open in Inker"):
            inker_mode.open_job_reference(ctx, job)
        widgets.hint_text("Paint over the reference; saving updates this asset.")
    # Independent, not an else: the two gates are disjoint today (a reference
    # has no mesh and a mesh has no editable reference), and an else would hide
    # one of them without saying so if that ever stopped being true.
    if can_edit_in_clay(job):
        if imgui.button(f"{icons.BOX} Open in Clay"):
            clay_mode.edit_asset_in_clay(ctx, job)
        widgets.hint_text("Opens the authored document when there is one, else the mesh.")


def _details_tab(ctx: Any, job: Any) -> None:
    _edit_actions(ctx, job)
    _settings(ctx, job)
    _reference(ctx, job)
    _pixel(ctx, job)
    _seam(ctx, job)
    if ctx.state.mode == "3d":
        _quality(ctx, job)
        _verdict(ctx, job)


def _rig_tab(ctx: Any, job: Any) -> None:
    _weighting(ctx, job)
    retarget_panel.draw(ctx, job)
    pose_panel.draw(ctx, job)
    sheet_panel.draw(ctx, job)


def weighting_verdict(params: Any) -> tuple[int, str] | None:
    """The one line that says how the mesh is bound to its skeleton, or None.

    A pure function for the reason ``seam_verdict`` is one: the wording is the
    feature and it has to be assertable without a GL context. Envelope is a
    *degraded* outcome -- the bone-heat solve failed and a capsule falloff
    stood in for it -- so it is named as one rather than reported as a second
    kind of success, which is how a silently degraded rig reached the user
    looking exactly like a good one.
    """
    if not isinstance(params, dict):
        return None
    weighting = params.get("weighting")
    if not weighting:
        return None
    if weighting == "envelope":
        return (theme.WARN, "weighting: envelope - needs review")
    return (theme.MUTED, f"weighting: {weighting}")


def _weighting(ctx: Any, job: Any) -> None:
    verdict = weighting_verdict(job.get("params") or {})
    if verdict is None:
        return
    widgets.text_colored(*verdict)
    reason = (job.get("params") or {}).get("weighting_reason")
    if reason and imgui.is_item_hovered():
        imgui.set_tooltip(str(reason))
    if verdict[0] == theme.WARN:
        widgets.hint_text("Blender's bone-heat solve did not take; hover for why.")


# --- pieces -----------------------------------------------------------------

# Field text the queue refused, keyed by submit key. Frame thread only, and
# cleared the moment a submit is accepted, so it holds at most one entry per
# field being typed into right now. See ``_write_field``.
_unsent: dict[str, str] = {}


def _write_field(ctx: Any, key: str, job_id: str, payload: Any, typed: str, current: str) -> None:
    """Submit an edited field, retrying until the queue takes it.

    ``TaskRunner.submit`` refuses a key already in flight and nothing re-arms
    it, and ``widgets.input_text`` hands back the *passed* value on a frame
    with no keystroke -- so the frame after a refusal compared the stale cached
    job against itself, found no difference, and the keystrokes typed during
    the write were simply gone. Fast typing committed a prefix.

    The unsent text is held here instead and retried every frame until a submit
    is accepted, which is also what clears it.
    """
    if typed != current:
        _unsent[key] = typed
    pending = _unsent.get(key)
    if pending is None:
        return
    if ctx.submit(key, svc_jobs.update_job, ctx.svc, job_id, payload(pending)):
        _unsent.pop(key, None)


def _header(ctx: Any, job: Any) -> None:
    job_id = job["id"]
    name = widgets.input_text("##name", job.get("name") or "", max_length=120, hint="Name")
    _write_field(
        ctx, f"name:{job_id}", job_id, lambda v: {"name": v}, name, job.get("name") or ""
    )
    tags = widgets.input_text("##tags", job.get("tags") or "", max_length=400, hint="tags, comma")
    _write_field(
        ctx,
        f"tags:{job_id}",
        job_id,
        lambda v: {"tags": v.split(",")},
        tags,
        job.get("tags") or "",
    )
    widgets.status_pill(job["status"])
    imgui.same_line()
    favourite = bool(job.get("favorite"))
    if imgui.small_button("Unfavourite" if favourite else "Favourite"):
        ctx.submit(
            f"fav:{job_id}", svc_jobs.update_job, ctx.svc, job_id, {"favorite": not favourite}
        )


def _meta(ctx: Any, job: Any) -> None:
    widgets.muted(f"{job['id']} - {job.get('kind')} - {job.get('stage')}")
    if job.get("created_at"):
        widgets.muted(str(job["created_at"]))
    if ctx.viewer is not None and ctx.viewer.has_model and ctx.state.mode == "3d":
        stats = ctx.viewer.stats()
        if stats:
            size = stats["size"]
            # Counted off what is actually loaded, not read from mesh_report:
            # the viewer may be showing rig.glb or a baked pose, and the number
            # under the model should describe the model under it.
            widgets.muted(
                f"{stats['triangles']:,} tris - {stats['vertices']:,} verts - "
                f"{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m"
            )


def _error(ctx: Any, job: Any) -> None:
    widgets.text_colored(theme.ERR, job.get("error") or "It failed.")
    if not imgui.tree_node("Details"):
        return
    key = "trellis-log"
    if widgets.disabled_button("Read the trellis log", not ctx.busy(key)):
        ctx.submit(key, svc_system.trellis_log, ctx.svc)
    log_text = ctx.state.preview.get("trellis_log") if ctx.state.preview else None
    if log_text:
        imgui.input_text_multiline(
            "##log", log_text[-4000:], (-1, 160), imgui.InputTextFlags_.word_wrap.value
        )
    if "error.log" in (job.get("files") or []) and imgui.button("Save error.log..."):
        ctx.save_artifact(job["id"], "error.log")
    imgui.tree_pop()


def _settings(ctx: Any, job: Any) -> None:
    params = job.get("params") or {}
    if not params:
        return
    if not widgets.header("Generation settings", default_open=False):
        return
    rows = [
        ("seed", params.get("seed")),
        ("reference seed", params.get("reference_seed")),
        ("mesh seed", params.get("mesh_seed")),
        ("platform", params.get("platform")),
        ("resolution", params.get("resolution")),
        ("size (m)", params.get("size_m")),
        ("model", params.get("base_model")),
        ("style LoRA", params.get("style_lora")),
        ("appearance ref", params.get("ip_adapter")),
        ("appearance strength", params.get("ip_scale")),
        ("structure", params.get("control")),
        ("structure strength", params.get("control_scale")),
        ("background", params.get("bg_removal")),
        ("profile", params.get("profile")),
    ]
    for label, value in rows:
        if value in (None, ""):
            continue
        widgets.muted(f"{label}: {value}")
    composed = params.get("composed_prompt")
    if composed and imgui.tree_node("Prompt sent"):
        imgui.text_wrapped(str(composed))
        imgui.tree_pop()


def _attempt_verdict(attempt: Any) -> str:
    """What one reroll attempt is to be called in the attempts line.

    Three words rather than two, because "not measured" is a third state and
    not a flavour of "refused": the composition report can fail to run at all
    (queue.py records the attempt with measured False, following
    reference.unmeasured), and calling that a refusal would put a verdict in
    the UI that nothing ever reached. Written against .get throughout, so an
    attempt row from before the third state existed still renders.
    """
    if not isinstance(attempt, dict):
        return "?"
    if attempt.get("measured") is False:
        return "not measured"
    return "kept" if attempt.get("ok") else "refused"


def reference_max_height() -> float:
    """The tallest a reference image may be drawn, in scaled pixels."""
    return sp(THUMB_SIZE * REFERENCE_MAX_THUMBS)


def reference_fit(size: tuple[int, int], avail: float) -> tuple[float, float]:
    """The box a reference image is drawn in: fill the width, keep the shape.

    The same fit-to-box arithmetic ``main._draw_reference`` uses for the
    viewport, and pure for the reason ``pixel_scale`` is -- it decides what the
    user sees and it should not need a GL context to assert. It used to be a
    hardcoded ``(THUMB_SIZE, THUMB_SIZE)``, which ignored the aspect ratio
    outright: a 2:1 sheet was drawn squashed into a square, and the image never
    reacted to the width of the pane it was in.

    The height cap is the other half. Filling the width of a 300 px sidebar
    with a 1:4 portrait is a thousand-pixel image, and there are up to four of
    them, which pushes every section under Reference off the screen.
    """
    width = float(max(size[0], 1))
    height = float(max(size[1], 1))
    scale = min(max(avail, 1.0) / width, reference_max_height() / height)
    return (width * scale, height * scale)


def _reference(ctx: Any, job: Any) -> None:
    """What the mesh engine was actually handed, and what it made of it.

    The three images answer the question a bad mesh always raises -- was it
    the reconstruction, or was it what we sent? -- which nothing in the UI
    could answer before: trellis was streamed input.png verbatim and no
    measurement of it was kept.
    """
    params = job.get("params") or {}
    report = params.get("reference_report")
    files = set(job.get("files") or [])
    shown = [n for n in ("input.png", "ref.png", "reference.png", "control.png") if n in files]
    if not isinstance(report, dict) and len(shown) <= 1:
        return
    # Open by default now that the images are legible: the section is only
    # drawn at all when there is a composition report or more than one image,
    # so it is never a collapsed heading over nothing, and the question it
    # answers -- was it the reconstruction or was it what we sent? -- is the
    # first one a bad mesh raises. It carries no persist_key, so this changes
    # the default rather than overriding anything a user has closed.
    if not widgets.header("Reference"):
        return

    if isinstance(report, dict):
        for reason in report.get("reasons") or []:
            widgets.text_colored(theme.ERR, reason)
        for warning in report.get("warnings") or []:
            widgets.text_colored(theme.WARN, warning)
        if report.get("occupancy"):
            widgets.muted(f"the subject fills {float(report['occupancy']) * 100:.0f}% of the frame")
        if report.get("measured") is False:
            widgets.muted("not measured")
        elif report.get("normalised"):
            widgets.muted("recentred and rescaled before upload")

    attempts = params.get("reference_attempts")
    if isinstance(attempts, list) and len(attempts) > 1:
        # Only ever present on a job the reroll actually ran for, so the line
        # doubles as the answer to "why is this not the seed I asked for".
        widgets.muted(
            f"redrawn {len(attempts) - 1} time(s): "
            + "; ".join(f"seed {a.get('seed')} {_attempt_verdict(a)}" for a in attempts)
        )

    hint = params.get("control_hint")
    if isinstance(hint, dict) and hint.get("edge_fraction") is not None:
        # The number that answers "my silhouette lock did nothing": near zero
        # means the detector found no structure, not that it was ignored.
        widgets.muted(f"{hint['kind']} hint: {float(hint['edge_fraction']) * 100:.1f}% edges")

    if ctx.textures is None:
        return
    for name in shown:
        # Keyed by job *and* file: the cache keys on (id, mtime), so sharing
        # the bare job id across four images would show whichever one was
        # decoded first for all of them.
        texture = ctx.textures.get(f"{job['id']}:{name}", ctx.job_dir(job["id"]) / name)
        if texture is None:
            continue
        widgets.muted(name)
        # Measured per image rather than once: the label above each one costs a
        # line, and a scrollbar appearing partway down the list changes what is
        # left for the images under it.
        imgui.image(
            widgets.texture_ref(texture),
            reference_fit(texture.size, imgui.get_content_region_avail().x),
        )


def seam_verdict(report: Any) -> tuple[int, str] | None:
    """The one line that says whether the tile actually tiles, or None.

    A pure function of the stored report, so the wording is assertable without
    a GL context -- and so the number and the word never come from two places.
    The ratio is "how much worse than the picture's own grain the wrap seam is",
    which is why the threshold is quoted alongside it: a bare 3.4 is a number
    nobody can calibrate against, while "3.4, over 2.00" says which way it went
    and by how far.
    """
    if not isinstance(report, dict):
        return None
    worst = float(report.get("worst") or 0.0)
    if report.get("seamless"):
        return (theme.OK, f"seamless -- edge/grain {worst:.2f}")
    threshold = float(report.get("threshold") or 0.0)
    return (theme.WARN, f"visible seam -- edge/grain {worst:.2f}, over {threshold:.2f}")


def _seam(ctx: Any, job: Any) -> None:
    """Whether the tile actually tiles, and what it looks like wrapped.

    The ratio alone is a number nobody can calibrate against by eye, which is
    what the wrapped view is for: rolled by half, what was the wrap seam runs
    through the middle of the frame, where a discontinuity is obvious. It is a
    derived artifact like every other export rather than something this pane
    writes -- the roll reads and rewrites a full-size PNG, which is emphatically
    not frame-thread work, and going through ``derive.get_file`` means the
    button here and the Export tab's produce the same file under the same lock.
    """
    verdict = seam_verdict((job.get("params") or {}).get("seam_report"))
    if verdict is None:
        return
    if not widgets.header("Seam"):
        return
    report = job["params"]["seam_report"]
    widgets.text_colored(*verdict)
    widgets.muted(
        f"left/right {float(report.get('horizontal') or 0.0):.2f} - "
        f"top/bottom {float(report.get('vertical') or 0.0):.2f}"
    )
    if ctx.textures is None:
        return
    job_id = job["id"]
    # The cache answers "is it on disk yet" as a side effect of answering "give
    # me the texture": a missing file is a None, which is exactly the state in
    # which the button should be offered.
    texture = ctx.textures.get(f"{job_id}:wrap", ctx.job_dir(job_id) / "wrap_preview.png")
    key = f"wrap:{job_id}"
    if texture is None:
        busy = ctx.busy(key)
        if busy:
            # The same idiom the Export grid uses beside a busy artifact:
            # rolling a 1024-square PNG is several hundred milliseconds, and a
            # button that only greys out reads as one that did nothing.
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button("Show it wrapped", not busy):
            ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, "wrap_preview.png")
        return
    imgui.image(widgets.texture_ref(texture), (sp(THUMB_SIZE * 2), sp(THUMB_SIZE * 2)))


def pixel_provenance(manifest: Any, name: str) -> str | None:
    """What actually cut the pixel artifact on disk, or None without an entry.

    Read off the manifest rather than off the knob, because the two can
    disagree -- switching sizes shows a file derived under an older palette
    setting -- and the line exists to say what the file *is*. Pure for the
    reason seam_verdict is: the wording is assertable without a GL context.
    """
    if not isinstance(manifest, dict):
        return None
    entry = (manifest.get("artifacts") or {}).get(name)
    if not isinstance(entry, dict):
        return None
    size = svc_files.PIXEL_ARTIFACTS.get(name)
    palette = entry.get("palette")
    if entry.get("palette_file"):
        colours = str(entry["palette_file"])
        if entry.get("dither"):
            colours += " (dithered)"
    else:
        colours = f"{palette} colours" if palette else "full colour"
    parts = [f"{size} px", colours]
    grid = entry.get("grid")
    if isinstance(grid, dict) and grid.get("scale"):
        # Only said when it happened: an ordinary render has no lattice, and
        # claiming one would be a claim about the generator, not the file.
        parts.append(f"{grid['scale']}px source grid")
    from ...pipelines import pixel as pixelmod

    sentence = pixelmod.verdict(entry.get("qa") or {})
    if sentence:
        parts.append(sentence)
    return " - ".join(parts)


def pixel_scale(size: tuple[int, int], avail: int) -> int:
    """The integer factor a pixel artifact is drawn at.

    A whole multiple or nothing: a fractional factor samples some source
    pixels twice and others once, which reads as banding -- the thing NEAREST
    sampling exists to avoid.
    """
    return max(1, avail // max(size[0], size[1], 1))


def _palette_names(ctx: Any) -> list[str]:
    """The palette directory's stems, listed once per version of the directory.

    One stat per frame rather than one walk: this is drawn on the frame thread,
    and a directory that changes when a user drops a file into it moves its own
    mtime, so a file added while the app runs still appears on the next frame.
    A missing directory is an empty list, never an error -- palettes are opt-in
    and there is nothing to report about not having any.
    """
    from ...service import palettes as svc_palettes

    directory = ctx.svc.config.palette_dir
    try:
        key = directory.stat().st_mtime_ns
    except OSError:
        ctx.state.palettes = None
        return []
    cached = ctx.state.palettes
    if cached is not None and cached[0] == key:
        return cached[1]
    names = svc_palettes.available(ctx.svc.config)
    ctx.state.palettes = (key, names)
    return names


def _pixel(ctx: Any, job: Any) -> None:
    """The pixel-art cutout as it will actually export, crisp on screen.

    The knob is derive-time state, not a job param: input.png stays at full
    resolution (a promotion feeds trellis the same pixels it always did) and
    changing it re-derives the artifact, exactly as a hand edit does. The
    re-derive goes through ``derive.get_file`` under its own ``derive:`` key
    and reports busy on either key, so this preview, its spinner and the
    export button can never describe different files -- while the "Saved to"
    toast the app hangs off every ``save:`` key stays with the one control
    that actually opens a dialog.
    """
    if job.get("stage") != "reference" or job.get("status") != "done":
        return
    if "input.png" not in set(job.get("files") or []):
        return
    if not widgets.header("Pixel art", default_open=False):
        return
    job_id = job["id"]
    size, colors, palette_name, dither = pixel_prefs(ctx.settings)
    submit_kwargs = {
        "pixel_colors": colors,
        "pixel_palette": palette_name,
        "pixel_dither": dither,
    }

    sizes = sorted(svc_files.PIXEL_ARTIFACTS.values())
    chosen = int(widgets.labeled_combo("Size", str(size), [(str(s), f"{s} px") for s in sizes]))
    if chosen != size:
        ctx.settings.set("pixel_size", chosen)
    # The name is always one of the three literals: ``chosen`` comes from
    # PIXEL_ARTIFACTS itself, so the MEDIA allowlist property holds.
    name = f"pixel_{chosen}.png"
    key = derive_key(job_id, name)

    picked = int(
        widgets.labeled_combo(
            "Colours",
            str(colors),
            [(str(n), "Off" if n == 0 else f"{n} colours") for n in svc_files.PIXEL_COLOR_CHOICES],
        )
    )
    if picked != colors:
        ctx.settings.set("pixel_colors", picked)
        colors = submit_kwargs["pixel_colors"] = picked
        # Re-derive the size on screen straight away; the other sizes catch up
        # when they are next asked for, which is what _pixel_current is for.
        ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)

    names = _palette_names(ctx)
    if names:
        options = [("", "None")] + [(n, n) for n in names]
        chosen_palette = widgets.labeled_combo("Palette", palette_name or "", options)
        chosen_palette = chosen_palette or None
        changed = chosen_palette != palette_name
        if changed:
            ctx.settings.set("pixel_palette", chosen_palette or "")
            palette_name = submit_kwargs["pixel_palette"] = chosen_palette
        if palette_name:
            toggled, dither = imgui.checkbox("Dither", dither)
            if toggled:
                ctx.settings.set("pixel_dither", dither)
                submit_kwargs["pixel_dither"] = dither
                changed = True
        if changed:
            ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)

    if ctx.textures is None:
        return
    busy = ctx.artifact_busy(job_id, name)
    texture = ctx.textures.get(f"{job_id}:{name}", ctx.job_dir(job_id) / name, nearest=True)
    if texture is None:
        if busy:
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button("Preview pixels", not busy):
            ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)
        return
    factor = pixel_scale(texture.size, int(sp(THUMB_SIZE * 2)))
    imgui.image(widgets.texture_ref(texture), (texture.size[0] * factor, texture.size[1] * factor))
    if busy:
        widgets.spinner()
        return
    manifest = _manifest(ctx, job_id)
    line = pixel_provenance(manifest, name)
    if line:
        widgets.muted(line)
    entry = ((manifest or {}).get("artifacts") or {}).get(name)
    stale = isinstance(entry, dict) and (
        (entry.get("palette_file") or None) != palette_name
        or bool(entry.get("dither")) != dither
        or (
            # The median-cut cap only means anything while no palette file is
            # in play; with one, the entry records ``palette: None`` by design.
            not palette_name and entry.get("palette") != (colors or None)
        )
    )
    # The file on screen was cut under a different palette setting -- switching
    # sizes surfaces one derived before the knob changed.
    if stale and imgui.button("Rebuild with these colours"):
        ctx.submit(key, svc_derive.get_file, ctx.svc, job_id, name, **submit_kwargs)


def watertight_value(report: Any) -> bool | None:
    """Whether the mesh is sealed, read the way the badge reads it.

    ``meshreport`` measures watertightness on a *welded* copy, because it loads
    with ``process=False`` so the UV and material checks see the unwelded mesh
    -- and every xatlas seam split is then a boundary edge, which makes the raw
    flag say "not watertight" about every textured mesh ever reconstructed.
    ``widgets.quality_badge`` was moved onto ``welded_watertight`` and this
    panel was not, so it printed ``watertight: False`` directly under a badge
    saying ``watertight``.

    The fallback is what keeps rows recorded before the change readable, and it
    is the identical expression ``vectors.py`` uses.
    """
    if not isinstance(report, dict):
        return None
    value = report.get("welded_watertight", report.get("watertight"))
    return value if isinstance(value, bool) else None


def watertight_line(report: Any) -> str | None:
    """The one line the panel prints about watertightness, or None.

    The label names which measurement it is: a welded reading says so, and a
    row that only ever carried the raw flag must not be relabelled as one --
    that would be the same overclaim in reverse, and it is exactly the mistake
    the badge made with ``mesh_audit``.
    """
    value = watertight_value(report)
    if value is None:
        return None
    if isinstance(report, dict) and isinstance(report.get("welded_watertight"), bool):
        return f"watertight (welded): {value}"
    return f"watertight: {value}"


def _quality(ctx: Any, job: Any) -> None:
    params = job.get("params") or {}
    report = params.get("mesh_report")
    audit = params.get("mesh_audit")
    if not (report or audit):
        return
    if not widgets.header("Mesh quality"):
        return
    if isinstance(report, dict):
        widgets.quality_badge(job)
        for reason in report.get("reasons") or []:
            widgets.muted(f"- {reason}")
        # Only the report may use the word watertight: the audit is a
        # silhouette check and never proved it.
        for label, key in (
            ("triangles", "triangles"),
            ("materials", "materials"),
        ):
            if key in report:
                widgets.muted(f"{label}: {report[key]}")
        sealed = watertight_line(report)
        if sealed:
            widgets.muted(sealed)
        if "grounded" in report:
            widgets.muted(f"pivot at feet: {report['grounded']}")
    if isinstance(audit, dict) and audit.get("worst") is not None:
        widgets.muted(f"visible openings: {float(audit['worst']) * 100:.1f}%")

    attempts = params.get("mesh_attempts")
    if isinstance(attempts, list) and len(attempts) > 1:
        # Only ever present on a job the remesh actually ran for, so the line
        # doubles as the answer to "why is this not the mesh seed I asked for".
        widgets.muted(
            f"remeshed {len(attempts) - 1} time(s); kept the best of "
            + ", ".join(
                "unmeasured" if a.get("worst") is None else f"{float(a['worst']) * 100:.1f}%"
                for a in attempts
            )
        )


# --- verdicts ---------------------------------------------------------------
#
# The same Accept/Reject a sweep unit gets in Review, on any finished asset --
# which is the point: daily use and a deliberate sweep feed one findings pool,
# and the vast majority of assets anyone ever looks at were never part of a
# sweep. The two halves that are easy to get wrong (when the reject is armed,
# and what a recorded verdict does) are plain functions so they can be asserted
# without a GL context.


def verdict_armed(state: Any, job_id: str) -> bool:
    return state.inspector_reject_armed == job_id


def arm_verdict(state: Any, job_id: str | None) -> None:
    state.inspector_reject_armed = job_id


def record_verdict(ctx: Any, job_id: str, verdict: str, reasons: tuple[str, ...] = ()) -> bool:
    """File a verdict and queue the findings recompute. -> whether it landed.

    Inline on the frame thread for the reason Review's is: one INSERT under the
    store's lock. The recompute that follows reads every verdict and writes a
    file, so it is a task.
    """
    from ...service import verdicts as svc_verdicts
    from ...service.errors import ServiceError
    from .. import review_mode

    try:
        svc_verdicts.record_verdict(ctx.svc, job_id, verdict=verdict, reasons=reasons)
    except (ServiceError, OSError):
        ctx.toast("Could not record that verdict.", "error")
        return False
    arm_verdict(ctx.state, None)
    # Through Review's own request, not a second submit under a copy of its
    # key: two spellings of one task key are two things to keep in step.
    review_mode.refresh_findings(ctx)
    ctx.toast(f"Recorded: {verdict}.")
    return True


def _verdict(ctx: Any, job: Any) -> None:
    from ...service import verdicts as svc_verdicts

    if job.get("status") != "done" or job.get("stage") != "model":
        return
    if not widgets.header("Was this any good?", default_open=False):
        return
    job_id = job["id"]
    widgets.muted("Feeds the same findings a sweep does.")
    if widgets.primary_button("Accept"):
        record_verdict(ctx, job_id, "accept")
    imgui.same_line()
    if imgui.button("Reject"):
        arm_verdict(ctx.state, job_id)
    if verdict_armed(ctx.state, job_id):
        widgets.muted("Why?")
        for reason in svc_verdicts.REASONS:
            if imgui.button(f"{reason}##verdict-{reason}"):
                record_verdict(ctx, job_id, "reject", (reason,))
        if imgui.button("Cancel##verdict-cancel"):
            arm_verdict(ctx.state, None)


def _downloads(ctx: Any, job: Any) -> None:
    """The Export tab: a two-column grid of artifacts.

    A blocked artifact keeps its button and explains itself in a tooltip --
    the old layout wrote the reason as an indented line under each of eight
    disabled buttons, which for a plain reference job was a column of noise.
    """
    from .. import icons

    widgets.section("Downloads")
    job_id = job["id"]
    files = set(job.get("files") or [])
    # Both notes under the grid are about cutouts -- which matte cut them, and
    # what each one came out as -- so they belong to the stage that has any. A
    # tile's manifest holds the recipe and no artifact entries at all.
    cutouts = job.get("stage") == "reference"
    if not imgui.begin_table("downloads", 2):
        return
    for name, label in widgets.artifacts_for(job):
        # job["files"] is the sanctioned answer; a raw exists() check here used
        # to re-enable buttons the service would then refuse.
        ready = name in files
        blocked = _why_blocked(ctx, job, name, ready, _derivable(job, files, name))
        # Either key: the pixel panel can have a derivation of this very name
        # in flight, and it takes the same per-artifact lock this button would.
        busy = ctx.artifact_busy(job_id, name)
        imgui.table_next_column()
        if busy:
            widgets.spinner()
            imgui.same_line()
        if widgets.disabled_button(
            f"{icons.DOWNLOAD} {label}##{name}", not blocked and not busy, (-1, 0)
        ):
            ctx.save_artifact(job_id, name)
        if blocked and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(blocked)
    imgui.end_table()
    if cutouts:
        # Read once and handed to both: the two notes answer halves of the same
        # question, and asking for the manifest twice in a frame would make the
        # cheap-stat property depend on how many readers there happen to be.
        manifest = _manifest(ctx, job_id)
        _matte_note(ctx, manifest)
        _manifest_summary(manifest)


def _derivable(job: Any, files: set[str], name: str) -> bool:
    """Whether this job could produce ``name`` if the button were pressed.

    A 2D export comes from the reference's own pixels, so it is gated on those
    and not on a ``model.glb`` a reference will never have -- which is the old
    gate, and which greyed out every one of them. But it *is* gated on the job
    having finished: the mesh half got that for free, because model.glb is not
    listed until the worker is done with it, whereas ``derivable_2d`` answers a
    question about the name alone and would light six buttons on a queued job
    whose only outcome is an error toast. The conditions are deliberately the
    same three ``service.files.ready`` applies, restated here in terms of the
    listing rather than of the disk, because the frame thread may not stat --
    ``ready`` takes a job dir and would be a stat per artifact per frame. The
    two are pinned equal across the whole stage x status x files matrix by
    ``test_the_pane_agrees_with_the_service_about_every_artifact``, which is
    what stops the restatement drifting.

    Which names each image stage may ask for is deliberately *not* restated: a
    reference has the cutouts and a tile has the wrapped view, and that split is
    a fact about the artifacts rather than about the disk, so it is read
    straight off ``derivable_2d`` at no cost.
    """
    stage = job.get("stage")
    if stage in ("reference", "tile"):
        return (
            job.get("status") == "done"
            and "input.png" in files
            and svc_derive.derivable_2d(name, stage)
        )
    return "model.glb" in files and svc_derive.derivable(name)


def _why_blocked(ctx: Any, job: Any, name: str, ready: bool, derivable: bool) -> str | None:
    if ready:
        return None
    if name == "model.fbx" and not ctx.rigging_available:
        return "needs Blender"
    if derivable:
        return None
    if job.get("status") in ("queued", "running"):
        # Correctly disabled, and it used to be misleadingly explained: every
        # export on an unfinished job read "not available for this asset",
        # which says the asset cannot have the file rather than that it has not
        # got it yet. An errored job keeps the other wording, because there
        # "not yet" would promise something that is not coming.
        return "not finished yet"
    return "not available for this asset"


def _matte_note(ctx: Any, manifest: Any) -> None:
    """Why the cutouts will look the way they do, before any of them exist.

    Not a reason a button is blocked, which is why it sits under the grid
    rather than in ``_why_blocked``: the exports always work. The question a
    user has when the edges come out ragged is whether that is the model or the
    fallback, and nothing in the UI could answer it -- the doctor row says the
    same thing in a place nobody opens mid-export.

    Deliberately only for the case where nothing has been derived yet. Once
    artifacts exist, each manifest entry records the matte that actually cut
    it, and ``_manifest_summary`` says so per artifact -- which is the truthful
    answer where this one is merely the current one: an icon cut by the corner
    fill before the weights were installed would, on this note alone, claim
    nothing at all the moment they were.
    """
    from ...pipelines import matting

    if manifest is not None and (manifest.get("artifacts") or {}):
        return
    if matting.available(ctx.svc.config):
        return
    widgets.muted(
        "Cutouts use the corner fill -- edges are rougher than the matting "
        "model's. See the matting row under Settings for the one-time download."
    )


def _manifest_summary(manifest: Any) -> None:
    """The pivot, the alpha QA and the matte, read off the manifest as written.

    Read from the file rather than recomputed: the manifest is the thing an
    importer will consume, so showing anything else here would let the two
    disagree about the asset. That is also why the matte is reported per
    artifact -- ``derive`` records which one cut each file, and a file on disk
    was cut by whatever was installed when it was made, not by what is
    installed now.

    The hand-edited caveat travels with the recipe for the reason
    ``derive._write_manifest`` records it: the recipe hash names a seed and a
    model, which after a hand edit is no longer the whole story of the pixels.
    """
    if manifest is None:
        return
    if manifest.get("hand_edited"):
        widgets.muted("hand-edited -- the recipe describes the generated image, not these pixels")
    for name, entry in sorted((manifest.get("artifacts") or {}).items()):
        if not isinstance(entry, dict):
            continue
        bits = [name]
        if entry.get("pivot"):
            bits.append(f"pivot {entry['pivot'][0]:.0f},{entry['pivot'][1]:.0f}")
        alpha = entry.get("alpha") or {}
        if alpha.get("islands", 0) > 1:
            bits.append(f"{alpha['islands']} separate pieces")
        if entry.get("matte") == "flood":
            bits.append("corner fill")
        widgets.muted(" - ".join(bits))


def _manifest(ctx: Any, job_id: str) -> dict[str, Any] | None:
    """The job's parsed manifest.json, or None if there isn't a readable one.

    A stat every frame and a parse only when the file has changed. That split
    is the whole point: the export tab redraws sixty times a second, a
    derivation running on the TaskRunner rewrites the manifest underneath it,
    and re-reading on an mtime is how ``ThumbnailCache`` already reconciles
    those two facts. The slot lives on ``AppState`` rather than in a module
    global so the pane itself stays stateless.

    A failure to parse is cached too, as ``(key, None)``. "Unreadable" is an
    answer about that version of the file exactly as a parsed dict is, and a
    truncated or hand-mangled manifest is the one state that would otherwise be
    read and parsed *every* frame, for as long as it stayed mangled -- which is
    the per-frame read this cache exists to remove. A missing file needs no
    sentinel: the stat fails, so nothing is read either way.
    """
    import json

    path = ctx.job_dir(job_id) / "manifest.json"
    try:
        key = (job_id, path.stat().st_mtime_ns)
    except OSError:
        return None
    cached = ctx.state.manifest
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        manifest = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        manifest = None
    if not isinstance(manifest, dict):
        manifest = None
    ctx.state.manifest = (key, manifest)
    return manifest


def duration(job: Any) -> str:
    return format_duration(job.get("duration"))
