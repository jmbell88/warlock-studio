"""Render the real app, mode by mode and palette by palette, to PNGs.

The GL smoke suite asserts that every pane *builds*; it asserts nothing about
what anything looks like, and says so in its own docstring. That left Phase 3's
light palette and Phase 4's new Inker and Clay controls shipped without a human
having seen them. This is the harness that closes that gap without asking
somebody to click through eight modes twice.

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Modes in switch order. Every one of them, because "which pane did nobody look
# at" is exactly the question this answers -- so it is *derived* rather than
# written out. A hand-kept tuple claiming to be every mode is the drift this
# script's own docstring warns about, and it had already happened once: two
# modes were added and the list still said eight.
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


def _seed(app) -> None:
    """Open a canvas and a model, so the panes that need one are not empty.

    Inker and Clay both draw an empty-state pane with nothing open, which is
    exactly the frame that shows none of the controls Phase 4 added -- the tool
    grid's options, the properties panel's sections, the timeline. Both entry
    points are the ones the buttons call.
    """
    from warlock.studio import clay_mode, inker_mode

    inker_mode.new_document(app.app_ctx, 1024, 1024)
    clay_mode.new_document(app.app_ctx)


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
    ap.add_argument("--themes", default="dark,light")
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
    if args.seed:
        _seed(app)
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
                if args.review and mode == "review":
                    # *After* the switch, because entering a mode starts a
                    # rescan and a scan landing re-opens the bucket -- which
                    # correctly drops anything staged against the unit that was
                    # on screen. Seeding transient per-unit state before a loop
                    # over every mode therefore cannot survive to its capture.
                    _stage_review_tag(app)
                _capture(app, args.out / f"{name}-{mode}.png")
            if args.floating:
                # Over 3D rather than over Home: the backdrop is what is being
                # looked at, and a viewport with a mesh in it is the one screen
                # where "is this a blur or a flat fill" cannot be argued about.
                from warlock.studio.panes import palette as palette_pane

                app.app_ctx.state.mode = "3d"
                _capture(app, args.out / f"{name}-3d-clean.png")
                palette_pane.toggle(app.app_ctx)
                _capture(app, args.out / f"{name}-palette.png")
                palette_pane.close(app.app_ctx)
    finally:
        app.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
