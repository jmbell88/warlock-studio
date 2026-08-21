"""Render the real app, mode by mode and palette by palette, to PNGs.

The GL smoke suite asserts that every pane *builds*; it asserts nothing about
what anything looks like, and says so in its own docstring. That left Phase 3's
light palette and Phase 4's new Inker and Clay controls shipped without a human
having seen them. This is the harness that closes that gap without asking
somebody to click through thirteen modes twice.

It is the **real** App: `setup_window`, `setup_runtime`, `setup_context`, and
`App.frame` -- so the dispatch under test is `_build_ui`'s own, the fonts are
the loaded atlas rather than imgui's default bitmap, and the theme is whatever
`theme.apply` actually put in imgui's style. A reimplementation of the mode
dispatch here would be a second dispatch, which is the thing `modes.py` is data
rather than a callback table to avoid.

Frames are read back off the default framebuffer, so the window is genuinely on
screen while this runs. Several are drawn per capture because a pane's first
frame is routinely its emptiest -- textures upload on the frame after they are
asked for, and the job cache ticks once a frame.

Run:  uv run python scripts/screenshot_modes.py --out <dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Modes in switch order. Every one of them, because "which pane did nobody look
# at" is exactly the question this answers -- so it is *derived* rather than
# written out. A hand-kept tuple claiming to be every mode is the drift this
# script's own docstring warns about, and it had already happened once: two
# modes were added and the list still said eight.
from warlock.studio import create_stages  # noqa: E402
from warlock.studio import modes as _modes  # noqa: E402

MODES = _modes.KEYS

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


def _capture(app, path: Path) -> None:
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
    image = Image.frombytes("RGB", (width, height), data).transpose(
        Image.FLIP_TOP_BOTTOM
    )
    image.save(path)
    print(f"  {path.name}", flush=True)


def _close_popups(app) -> None:
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
    any_popup = (
        imgui.PopupFlags_.any_popup_id.value | imgui.PopupFlags_.any_popup_level.value
    )
    if imgui.is_popup_open("", any_popup):
        imgui.internal.close_popup_to_level(0, True)
    # Let popup owners observe the close before another owner opens one under
    # the same host window.
    app.frame(1.0 / 60.0)
    pygame.display.flip()


def _seed_matte(app) -> None:
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


def _capture_popups(app, out: Path, theme_name: str) -> None:
    """Capture every full-size transient container, including model flow."""
    from warlock.studio import create_stages, dialogs, plotter_mode, rail

    ctx = app.app_ctx
    ctx.state.mode = create_stages.MODE
    ctx.state.create_stage = "mesh"

    rail.request("diagnostics")
    _capture(app, out / f"{theme_name}-popup-diagnostics.png")
    _close_popups(app)

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Delete generated model?",
            message=(
                "The generated model and its derived files will be removed. "
                "The source reference is kept."
            ),
            confirm_label="Delete",
            cancel_label="Keep",
        )
    )
    _capture(app, out / f"{theme_name}-modal-confirm.png")
    _close_popups(app)

    ctx.prompts.ask(
        dialogs.Prompt(title="Name generated model", label="Name", value="Hooded adventurer")
    )
    _capture(app, out / f"{theme_name}-modal-prompt.png")
    _close_popups(app)

    _seed_matte(app)
    _capture(app, out / f"{theme_name}-modal-model-matte.png")
    _close_popups(app)

    # Into Plotter first. ``setup_pending`` is drawn by the *plotter* pane, and
    # this function opens on Create -- so the new-map modal was raised on a
    # frame nothing drew it in, and the capture was an ordinary picture of the
    # Create mode under a filename claiming to be a modal. A screenshot that
    # silently shows the wrong screen is worse than a missing one, because
    # nothing about the file says so.
    ctx.state.mode = "plotter"
    plotter_mode.ask_new_document(ctx)
    _capture(app, out / f"{theme_name}-modal-new-map.png")
    _close_popups(app)
    ctx.state.mode = create_stages.MODE


def _seed(app) -> None:
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


def _seed_tile(app, png: Path) -> None:
    """A finished tile job holding ``png``, selected, with 2D showing it.

    Enough of a job for the tile-only controls to be on screen. Writes into
    whatever data directory the process was pointed at, so run this against a
    throwaway ``WARLOCK_DATA_DIR`` rather than a real library.
    """
    import shutil

    ctx = app.app_ctx
    job_id = ctx.svc.store.create(
        "text", "cobblestone", {"seed": 11}, stage="tile", status="done"
    )
    job_dir = ctx.svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png, job_dir / "input.png")
    from warlock.pipelines import seam

    ctx.svc.store.merge_params(job_id, {"seam_report": seam.report(job_dir / "input.png")})
    ctx.cache.invalidate()
    ctx.cache.tick()
    ctx.state.select(job_id)


def _seed_asset(app) -> None:
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


def _seed_review(app) -> None:
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
        "image", "a wooden chest", {"lora_weight": 0.9, "seed": 42},
        stage="model", status="done",
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


def _stage_review_tag(app) -> None:
    """Wait for Review's scan, open the bucket and stage one tag.

    Called after the mode switch and immediately before the capture, because
    both halves of that timing are load-bearing: the scan is a task and the
    units do not exist until it lands, and the scan's completion re-opens the
    bucket, which drops anything staged beforehand.

    What it buys is the one frame where the *selected* toggle is on screen --
    the branch that draws the accent fill, and so the branch that looks
    identical to every other button if it is ever pushed wrongly.
    """
    from warlock.studio import review_mode

    ctx = app.app_ctx
    state = review_mode.ensure(ctx)
    for _ in range(SETTLE_FRAMES):
        app.frame(1.0 / 60.0)
        if not state.scanning and state.units:
            break
    if not state.units:
        return
    review_mode.toggle_tag(state, "sharp-detail")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--themes", default="dark,light,pixel")
    ap.add_argument(
        "--seed",
        action="store_true",
        help="open an Inker canvas and a Clay model before capturing",
    )
    ap.add_argument(
        "--tile",
        type=Path,
        help="seed a finished tile job from this PNG and select it",
    )
    ap.add_argument(
        "--tile-preview",
        action="store_true",
        help="turn the 2D viewport's tiled preview on",
    )
    ap.add_argument(
        "--asset",
        action="store_true",
        help="seed a reference and a rigged mesh made from it, mesh selected, "
             "so Create's five stages draw an asset rather than four empty "
             "states",
    )
    ap.add_argument(
        "--review",
        action="store_true",
        help="seed a finished mesh and open Review on it, so the verdict panel "
             "draws its grade row and tag toggles rather than its empty state",
    )
    ap.add_argument(
        "--floating",
        action="store_true",
        help=(
            "also capture the command palette open over each mode's own "
            "backdrop, which is the only way Phase 5's translucency is visible "
            "in a still: nothing in the mode pass ever opens a floating surface."
        ),
    )
    ap.add_argument(
        "--overlays",
        action="store_true",
        help=(
            "also capture the two surfaces that stopped being modes in "
            "the UI redesign, wave 3 -- the manual and the profile sheet -- plus the "
            "expanded navigation rail. The mode pass is derived from "
            "modes.KEYS, so none of the three is reachable by it: a screen that "
            "is not a mode is a screen nobody would look at."
        ),
    )
    ap.add_argument(
        "--components",
        action="store_true",
        help="also capture the developer component gallery",
    )
    ap.add_argument(
        "--popups",
        action="store_true",
        help=(
            "also capture diagnostics, confirm and prompt containers, the model "
            "matte modal, and Plotter's new-map modal"
        ),
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "UI scale to capture at, overriding the monitor's. The pass is run "
            "at 1.0 and 1.5: three shipped defects were invisible at 1.0, which "
            "is the only scale the smoke suite runs at and the one nobody uses."
        ),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.components:
        os.environ["WARLOCK_DEV_COMPONENTS"] = "1"

    from imgui_bundle import imgui

    from warlock.config import get_config
    from warlock.studio import theme as theme_mod
    from warlock.studio import tokens
    from warlock.studio.main import App
    from warlock.studio.runtime import Runtime

    app = App(Runtime(get_config()))
    app.setup_window()
    if args.scale is not None:
        # After the window (which samples the monitor) and before the context
        # (which makes textures): the atlas has to be re-baked at the new scale
        # or every icon sits off-centre by a fraction of the difference, which
        # is ``fonts.reload``'s whole reason for existing. Between frames is
        # satisfied trivially here -- there has not been one yet.
        from warlock.studio import fonts

        tokens.set_scale(args.scale)
        theme_mod.apply(imgui)
        fonts.reload(imgui)
    app.setup_runtime()
    app.setup_context()
    # A modal drawn over every capture is the harness lying by *commission*
    # (the UI redesign, wave 6): the journal lives in the user's home, not under
    # WARLOCK_DATA_DIR, so a real crashed session on the machine running this
    # put "Recover unsaved work?" across the middle of all thirty-four
    # pictures. Never *offered* rather than dismissed after the fact -- the
    # offer is raised on the first frame that has a Ctx, which is inside the
    # first capture, so answering it afterwards is a frame too late. Nothing
    # is deleted either way: declining keeps the files, and this process has
    # no business adopting somebody's documents to take a photograph.
    app.app_ctx.state.recovery_offered = True
    if args.seed:
        _seed(app)
    if args.asset:
        _seed_asset(app)
    if args.tile:
        _seed_tile(app, args.tile)
    if args.review:
        _seed_review(app)
    app.app_ctx.state.tile_preview = bool(args.tile_preview)
    try:
        for name in args.themes.split(","):
            tokens.set_theme(name)
            theme_mod.apply(imgui)
            print(f"{name}:", flush=True)
            for mode in MODES:
                app.app_ctx.state.mode = mode
                if mode == create_stages.MODE:
                    # One mode, several viewports (the UI redesign, wave 5): a
                    # single capture would show whichever stage happened to be
                    # current and silently leave the rest of the pipeline
                    # unlooked-at, which is the gap --seed closed for Inker.
                    for stage in create_stages.STAGES:
                        app.app_ctx.state.create_stage = stage
                        _capture(app, args.out / f"{name}-{mode}-{stage}.png")
                    continue
                if args.review and mode == "review":
                    # *After* the switch, because entering a mode starts a
                    # rescan and a scan landing re-opens the bucket -- which
                    # correctly drops anything staged against the unit that was
                    # on screen. Seeding transient per-unit state before a loop
                    # over every mode therefore cannot survive to its capture.
                    _stage_review_tag(app)
                _capture(app, args.out / f"{name}-{mode}.png")
            if args.overlays:
                from warlock.studio.manual import render as manual_render
                from warlock.studio.panes import profiles_panel

                state = app.app_ctx.state
                state.mode = create_stages.MODE
                state.create_stage = "mesh"
                manual_render.open_at(app.app_ctx, ("01-overview", None))
                _capture(app, args.out / f"{name}-manual.png")
                manual_render.close(app.app_ctx)
                state.create_stage = "reference"
                profiles_panel.open_sheet(app.app_ctx)
                _capture(app, args.out / f"{name}-profiles.png")
                state.profiles_open = False
                # The rail with its labels, which is a preference rather than a
                # mode -- so the whole mode pass above draws the collapsed one.
                app.layout.set_rail("labels")
                state.mode = "home"
                _capture(app, args.out / f"{name}-rail-expanded.png")
                app.layout.set_rail("icons")
                # The keyboard list, which is a popup rather than a mode and
                # so was the last drawn surface in the app with no picture of
                # it. Wave 5 renamed the modes underneath it and left its
                # "2D / 3D" table naming two that no longer exist; nothing
                # caught that until wave 6 went looking for what the pass did
                # not cover. Requested rather than opened directly -- the flag
                # is what Ctrl+/ and the palette both set, and ``open_popup``
                # has to happen inside the frame that draws it.
                state.shortcuts_requested = True
                _capture(app, args.out / f"{name}-shortcuts.png")
                state.shortcuts_requested = False
                # And shut again, or the light pass would draw every mode
                # under the dark pass's popup. ``close_current_popup`` is only
                # legal inside the popup's own begin/end, so the close comes
                # from the internal API instead -- between frames, where the
                # open-popup stack is nobody's to read.
                imgui.internal.close_popups_except_modals()
            if args.floating:
                # Over the Mesh stage rather than over Home: the backdrop is
                # what is being looked at, and a viewport with a mesh in it is
                # the one screen where "is this a blur or a flat fill" cannot
                # be argued about.
                from warlock.studio.panes import palette as palette_pane

                app.app_ctx.state.mode = create_stages.MODE
                app.app_ctx.state.create_stage = "mesh"
                _capture(app, args.out / f"{name}-mesh-clean.png")
                palette_pane.toggle(app.app_ctx)
                _capture(app, args.out / f"{name}-palette.png")
                palette_pane.close(app.app_ctx)
            if args.components:
                from warlock.studio import component_gallery

                app.app_ctx.state.mode = "settings"
                component_gallery.request()
                _capture(app, args.out / f"{name}-components.png")
                imgui.internal.close_popups_except_modals()
            if args.popups:
                _capture_popups(app, args.out, name)
    finally:
        app.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
