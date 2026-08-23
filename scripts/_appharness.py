"""Booting the real app, seeding it, and photographing it --- once.

This is an extraction, not new behaviour. ``screenshot_modes.py`` grew the
whole of it while it was the only script that drove the app, and
``exercise_mode.py`` needs every part: the same boot (so the dispatch under
test is ``_build_ui``'s own), the same warmup and settle (so a capture is not a
picture of a half-cleared crossfade), the same seeding (so a mode has controls
in it rather than an empty state), and the same popup teardown.

Two scripts each booting the app their own way is exactly the drift
``screenshot_modes.py``'s own docstring warns about one level up, where it
derives its mode list rather than writing it out.

Every seeder writes into whatever data directory the process was pointed at, so
run against a throwaway ``WARLOCK_HOME`` / ``WARLOCK_DATA_DIR`` / ``WARLOCK_DB``
--- all three, because ``WARLOCK_DATA_DIR`` alone does not move the sqlite
store.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def boot(scale: float | None = None, size: tuple[int, int] | None = None):
    """The real App, windowed, warmed and told two harness truths. -> ``App``.

    ``recovery_offered`` and ``first_run`` are set for the reason
    ``screenshot_modes.py`` recorded when it learned them the hard way: both
    are surfaces that own the screen ahead of everything else, both are pending
    on exactly the throwaway home a harness run has, and both are raised on the
    first frame that has a Ctx --- which is inside the first capture, so
    answering them afterwards is a frame too late. Nothing is deleted either
    way: declining recovery keeps the files, and this process has no business
    adopting somebody's documents to take a photograph.
    """
    from imgui_bundle import imgui

    from warlock.config import get_config
    from warlock.studio import theme as theme_mod
    from warlock.studio import tokens
    from warlock.studio.main import App
    from warlock.studio.runtime import Runtime

    app = App(Runtime(get_config()))
    app.setup_window(size_override=size)
    if scale is not None:
        # After the window (which samples the monitor) and before the context
        # (which makes textures): the atlas has to be re-baked at the new scale
        # or every icon sits off-centre by a fraction of the difference, which
        # is ``fonts.reload``'s whole reason for existing. Between frames is
        # satisfied trivially here -- there has not been one yet.
        from warlock.studio import fonts

        tokens.set_scale(scale)
        theme_mod.apply(imgui)
        fonts.reload(imgui)
    app.setup_runtime()
    app.setup_context()
    app.app_ctx.state.recovery_offered = True
    app.app_ctx.first_run = False
    return app


# Frames drawn before the read. Three is not a guess: one to build, one for the
# textures asked for on it to upload, one for anything those made visible.
WARMUP = 3

# How many further frames a capture will wait for the app to stop moving. A
# mode change raises a content crossfade (UX.md Phase 1) and three warmup frames
# is 50 ms of a 200 ms one, so without this every capture is a picture of a
# half-cleared veil -- a harness that made the whole screenshot pass useless in
# exactly the phase it exists to review. Bounded rather than a bare ``while``:
# an animation that never settles is a bug this must report by capturing it,
# not hang on.
SETTLE_FRAMES = 40


def capture(app, path: Path) -> None:
    import pygame
    from PIL import Image

    from warlock.studio import motion

    for _ in range(WARMUP):
        app.frame(1.0 / 60.0)
        pygame.display.flip()
    for _ in range(SETTLE_FRAMES):
        if not motion.animating():
            break
        app.frame(1.0 / 60.0)
        pygame.display.flip()
    width, height = pygame.display.get_window_size()
    data = app.ctx.screen.read(components=3, alignment=1)
    # GL's origin is bottom-left and everybody else's is top-left.
    image = Image.frombytes("RGB", (width, height), data).transpose(Image.FLIP_TOP_BOTTOM)
    image.save(path)
    print(f"  {path.name}", flush=True)


def close_popups(app) -> None:
    """Reset every transient surface after an isolated popup capture."""
    import pygame
    from imgui_bundle import imgui

    from warlock.studio import matte_preview, plotter_mode

    ctx = app.app_ctx
    while ctx.confirms.pending is not None:
        ctx.confirms.dismiss()
    while ctx.prompts.pending is not None:
        ctx.prompts.dismiss()
    matte_preview.close(ctx)
    plotter_mode.ensure(ctx).setup_pending = False
    # Guarded. ``close_popup_to_level`` walks the open-popup stack and trips an
    # IM_ASSERT when there is nothing on it, and the last capture of the run
    # reaches here with the stack already empty -- so the whole ``--popups``
    # pass aborted on its final step, *after* writing its images, which is why
    # it read as a crash with a complete-looking output directory. The public
    # any-popup query is the cheap way to ask before walking.
    any_popup = imgui.PopupFlags_.any_popup_id.value | imgui.PopupFlags_.any_popup_level.value
    if imgui.is_popup_open("", any_popup):
        imgui.internal.close_popup_to_level(0, True)
    # Let popup owners observe the close before another owner opens one under
    # the same host window.
    app.frame(1.0 / 60.0)
    pygame.display.flip()


def seed_matte(app) -> None:
    """Put a deterministic, already-computed matte in the model modal."""
    from PIL import Image, ImageDraw

    from warlock.service.matte import Preview, stamp_for
    from warlock.studio import matte_preview

    ctx = app.app_ctx
    job_id = ctx.svc.store.create(
        "text",
        "a hooded adventurer standing",
        {"seed": 7},
        stage="reference",
        status="done",
    )
    job_dir = ctx.svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (320, 320), (90, 110, 150, 255)).save(job_dir / "input.png")

    preview = Image.new("RGB", (320, 320), (180, 180, 180))
    draw = ImageDraw.Draw(preview)
    cell = 16
    for y in range(0, 320, cell):
        for x in range(0, 320, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(140, 140, 140))
    draw.ellipse((72, 24, 248, 306), fill=(82, 103, 148))
    draw.ellipse((118, 42, 202, 132), fill=(198, 171, 145))

    stamp = stamp_for(job_dir / "input.png")
    result = Preview(
        job_id=job_id,
        stamp=stamp,
        width=preview.width,
        height=preview.height,
        rgb=preview.tobytes(),
        source="birefnet",
        approved=False,
        coverage=0.47,
        warnings=("Check the fine edges around the hood before building.",),
    )
    state = matte_preview.open_for(ctx, job_id, {"mesh_seed": 11})
    state.stamp = stamp
    state.preview = result
    state.cache[job_id] = result


def seed(app) -> None:
    """Open a canvas and a model, so the panes that need one are not empty.

    Inker and Clay both draw an empty-state pane with nothing open, which is
    exactly the frame that shows none of the controls Phase 4 added -- the tool
    grid's options, the properties panel's sections, the timeline. Both entry
    points are the ones the buttons call.

    The canvas is **animated**, which this claimed to cover and did not: the
    timeline strip is drawn only for a document with an ``anim``, so a plain
    new canvas left the app's densest row -- the transport, the frame
    operations, the exports and their controls -- out of every capture this
    harness has ever taken. That row was rewritten in the UI redesign, wave 4.2
    precisely because it was clipping at 150 %, which is the defect class the
    scale pass exists to find.

    And a map and an atlas, for the same reason one wave later (the UI redesign,
    wave 6). Plotter and Packwright are four panes each, all four of which
    answer "Open or start a map first" with nothing open -- so the sentence-case
    sweep over their eleven headings, the tool grids, the layer tree and the
    packing controls had never appeared in a capture at any scale. Both
    ``new_document`` calls are synchronous; sprites and tilesets are not,
    because they land through ``ctx.submit`` on a task thread, and a seeder
    that races the capture is worse than one that stops short of it.
    """
    from warlock.studio import clay_mode, inker_mode, packwright_mode, plotter_mode

    inker_mode.new_document(app.app_ctx, 1024, 1024)
    state = inker_mode.ensure(app.app_ctx)
    if state.active is not None:
        inker_mode.animate(app.app_ctx, state.active)
    clay_mode.new_document(app.app_ctx)
    plotter_mode.new_document(app.app_ctx)
    packwright_mode.new_document(app.app_ctx)


def seed_tile(app, png: Path) -> None:
    """A finished tile job holding ``png``, selected, with 2D showing it.

    Enough of a job for the tile-only controls to be on screen. Writes into
    whatever data directory the process was pointed at, so run this against a
    throwaway ``WARLOCK_DATA_DIR`` rather than a real library.
    """
    import shutil

    ctx = app.app_ctx
    job_id = ctx.svc.store.create("text", "cobblestone", {"seed": 11}, stage="tile", status="done")
    job_dir = ctx.svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png, job_dir / "input.png")
    from warlock.pipelines import seam

    ctx.svc.store.merge_params(job_id, {"seam_report": seam.report(job_dir / "input.png")})
    ctx.cache.invalidate()
    ctx.cache.tick()
    ctx.state.select(job_id)


def seed_asset(app) -> None:
    """A finished reference and a rigged mesh promoted from it, mesh selected.

    Create's five stages are four columns and an inspector *about an asset*,
    and with nothing selected four of the five draw an empty state -- so the
    mode pass had never photographed the Rig column, the Pose column, the
    export grid, the lineage links, or a rail with any segment ticked. That is
    the gap ``--seed`` closed for Inker and ``--review`` for the verdict panel,
    and this is the same hole one wave later.

    The GLB is a **real** one, written by the app's own exporter: a stub would
    fail to parse on the frame that shows it and put an error toast in every
    capture.

    Writes into whatever data directory the process was pointed at, so run it
    against a throwaway ``WARLOCK_DATA_DIR`` rather than a real library.
    """
    from PIL import Image

    from warlock.studio.clay import document as bd
    from warlock.studio.clay import primitives as bp
    from warlock.studio.viewer import glbwrite

    ctx = app.app_ctx
    ref_id = ctx.svc.store.create(
        "text", "a hooded adventurer standing", {"seed": 7}, stage="reference", status="done"
    )
    ref_dir = ctx.svc.job_dir(ref_id)
    ref_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (256, 256), (90, 110, 150, 255)).save(ref_dir / "input.png")

    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box()))
    glb = glbwrite.write_glb(bd.to_model(doc))

    mesh_id = ctx.svc.store.create(
        "image",
        "a hooded adventurer standing",
        {"seed": 7, "mesh_seed": 11},
        stage="model",
        status="done",
        parent_id=ref_id,
    )
    mesh_dir = ctx.svc.job_dir(mesh_id)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "model.glb").write_bytes(glb)
    (mesh_dir / "source.glb").write_bytes(glb)
    # rig.glb is what the Rig and Pose stages gate on. Its own GLB rather than
    # a copy of nothing, for the parse reason above.
    (mesh_dir / "rig.glb").write_bytes(glb)
    Image.new("RGBA", (256, 256), (90, 110, 150, 255)).save(mesh_dir / "input.png")
    ctx.cache.invalidate()
    ctx.cache.tick()
    ctx.state.select(mesh_id)


def seed_review(app) -> None:
    """A finished mesh in the recent-unreviewed bucket, open in Review.

    Review's empty state shows none of the verdict panel: no grade row, no tag
    toggles, no recorded line. That is the same gap ``--seed`` closes for Inker
    and Clay, and it matters more here because the grade row is eleven buttons
    wrapping inside a 300 px sidebar -- ``same_line`` past the content region
    clips rather than wrapping, which is the bug that once hid seven controls,
    and it is invisible to the smoke suite because that asserts only that a pane
    builds.

    Writes into whatever data directory the process was pointed at, so run it
    against a throwaway ``WARLOCK_DATA_DIR`` rather than a real library.
    """
    from warlock.studio import review_mode

    ctx = app.app_ctx
    job_id = ctx.svc.store.create(
        "image",
        "a wooden chest",
        {"lora_weight": 0.9, "seed": 42},
        stage="model",
        status="done",
    )
    ctx.svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    ctx.cache.invalidate()
    ctx.cache.tick()
    # Through the scan the Rescan button runs, rather than by building units by
    # hand: a second way to populate this list is a second thing to keep true.
    # It is a *task*, so frames have to be pumped until it lands -- and the
    # staged tag has to be set after that, because the scan's completion opens
    # the bucket and opening disarms. That is the product behaviour (a rescan
    # moves you off the unit, so what you had staged for it goes with it), so
    # the harness waits rather than the rule bending.
    review_mode.scan(ctx)


def seed_troupe(app) -> None:
    """A finished character sheet, selected, so Troupe is not in its empty state.

    Without this the mode exercises nothing it exists for: the Sheet pane, the
    frame table and every handoff in ``sheet_panel`` are drawn only for a
    selected character, so the 2026-08-23 pass covered the *empty* Troupe and
    said so. Everything the preview reads is a file, not a render, which is why
    this can exist at all -- a real character costs an image, a reconstruction,
    a rig and 256 rendered cells behind a GPU.

    **Built through the app's own builders**, not by hand-writing JSON.
    ``clips.expand_clips`` and ``charsheet.plan`` need the shipped clip library
    and nothing else -- no weights, no card, and about a millisecond -- so the
    sidecar this writes is the one the worker would write, tags and runs and
    all. A hand-rolled sidecar would be a second dialect of the format and
    would go stale the first time the real one changed.

    The PNG is drawn here rather than rendered, and deliberately *varies per
    cell*: a flat fill would make every frame of every direction identical, and
    a driver that presses Play, steps a frame and turns the character would
    photograph one unchanging picture and call all three controls dead.
    """
    import json
    import time

    from PIL import Image, ImageDraw

    from warlock import clips, rigging
    from warlock.pipelines import charsheet
    from warlock.pipelines import sheet as sheetlib
    from warlock.service import troupe as svc_troupe
    from warlock.studio import troupe_mode

    ctx = app.app_ctx
    layout = charsheet.resolve_layout()
    # The door's own template, not the string "humanoid": it is pinned there
    # because it is the only one the clip library carries a walk for, and a
    # seeder that names it a second time is a second place for that to be true.
    plan = charsheet.plan(
        clips.expand_clips(svc_troupe.TROUPE_TEMPLATE, layout),
        frame_size=32,
        layout=layout,
    )

    source_id = ctx.svc.store.create(
        "image",
        "a fire guardian",
        {"seed": 7},
        stage="model",
        status="done",
    )
    job_dir = ctx.svc.job_dir(source_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    sheet_id = rigging.new_id()
    png = rigging.sheet_png_path(job_dir, sheet_id)
    image = Image.new("RGBA", (plan.width, plan.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, cell in enumerate(plan.cells):
        x, y = cell.x, cell.y
        size = plan.frame_size
        # A head and a body, with the head tracking the cell index: adjacent
        # frames differ, so stepping and playing are visible in a screenshot.
        sway = (index % 4) - 1.5
        cx = x + size / 2 + sway
        draw.ellipse(
            (cx - size * 0.14, y + size * 0.12, cx + size * 0.14, y + size * 0.40),
            fill=(226, 232, 240, 255),
        )
        draw.rectangle(
            (cx - size * 0.11, y + size * 0.42, cx + size * 0.11, y + size * 0.86),
            fill=(148, 163, 184, 255),
        )
    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png)

    meta = sheetlib.sidecar(
        plan,
        sheet_id=sheet_id,
        source_job=source_id,
        image=png.name,
        created=time.time(),
        name="a fire guardian",
        animation=charsheet.animation_block(layout),
    )
    meta["troupe"] = layout.as_dict()
    # The sidecar is the completion marker and is written last, which is
    # ``list_sheets``' own rule: a sidecar with no PNG beside it is skipped.
    rigging.sheet_path(job_dir, sheet_id).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    ctx.svc.store.create(
        "charsheet",
        "a fire guardian",
        {"source_job": source_id, "sheet_id": sheet_id, "layout": layout.as_dict()},
        stage="sheet",
        status="done",
        parent_id=source_id,
    )
    ctx.cache.invalidate()
    ctx.cache.tick()
    troupe_mode.open_sheet(ctx, source_id, sheet_id)
