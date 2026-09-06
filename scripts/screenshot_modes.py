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

# Booting, warming, settling, seeding and popup teardown all live in the shared
# harness now: ``exercise_mode.py`` needs every one of them, and two scripts
# each booting the app their own way is the drift the derived mode list below
# exists to avoid.
#
# Modes in switch order. Every one of them, because "which pane did nobody look
# at" is exactly the question this answers -- so it is *derived* rather than
# written out. A hand-kept tuple claiming to be every mode is the drift this
# script's own docstring warns about, and it had already happened once: two
# modes were added and the list still said eight.
from _appharness import (  # noqa: E402
    SETTLE_FRAMES,
    boot,
)
from _appharness import capture as _capture  # noqa: E402
from _appharness import close_popups as _close_popups  # noqa: E402
from _appharness import seed as _seed  # noqa: E402
from _appharness import seed_asset as _seed_asset  # noqa: E402
from _appharness import seed_matte as _seed_matte  # noqa: E402
from _appharness import seed_palette as _seed_palette  # noqa: E402
from _appharness import seed_review as _seed_review  # noqa: E402
from _appharness import seed_sheet_form as _seed_sheet_form  # noqa: E402
from _appharness import seed_tile as _seed_tile  # noqa: E402
from _appharness import seed_troupe as _seed_troupe  # noqa: E402

from warlock.studio import create_stages  # noqa: E402
from warlock.studio import modes as _modes  # noqa: E402

MODES = _modes.KEYS


def _size(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT for deterministic minimum-size captures."""
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return width, height


def _capture_popups(app, out: Path, theme_name: str) -> None:
    """Capture every full-size transient container, including model flow."""
    from warlock.studio import create_stages, dialogs, plotter_mode

    ctx = app.app_ctx
    ctx.state.mode = create_stages.MODE
    ctx.state.create_stage = "mesh"

    # **Settings -> Health is deliberately not captured, and must not be.**
    #
    # It was, briefly. The three ``*-settings-health.png`` it wrote carried
    # ``C:\Users\<name>\...`` about eight times each, the absolute vendor
    # directory of the machine that ran the pass, and the capture's own temp
    # path down to its session GUID -- straight into a public repository. That
    # is the exact defect the doctor-banner comment in ``main`` records: 44 of
    # the 84 images in this repo once shipped with a developer's real home
    # directory, username and all.
    #
    # No environment isolation fixes it. The banner could be dismissed because
    # it is incidental to every mode; printing probed absolute paths is the
    # health page's *entire content*, so an isolated home only swaps one real
    # path for another real path. Nor is deleting the files enough on its own,
    # which is why this comment stands where the capture did: the next
    # regeneration would re-leak.
    #
    # A scrubbing mode -- render, then replace every path-shaped run with a
    # placeholder before the PNG is written -- is the better answer and is the
    # follow-up this refusal is holding a place for. It is not written here
    # because it would have to recognise every path-shaped string the page can
    # print, in pixels, and a scrubber that misses one is worse than no
    # capture: it looks safe. Until then the honest minimum is that a
    # screenshot which cannot be taken without leaking is a screenshot this
    # corpus does not have.
    #
    # ``rail.request("diagnostics")`` used to stand here, from when the page
    # was a popup. It kept working after the popup went, silently: ``request``
    # assigns into ``_wants`` rather than looking a key up, so it set a flag
    # nothing reads and the capture was an ordinary picture of Create.

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


#: The sheet arms of Create's 2D form, and what each one has to be told before
#: it draws its own controls. One capture each, because the arms do not share a
#: layout: the tileset arm's Tile-layout section is a different set of fields in
#: every one of its modes, and the sprite arm has an Action/Directions pair the
#: tileset arm has nothing like.
#:
#: ``tile_mode`` is the only extra: it is the tileset arm's own picker and the
#: default is Materials, so a single tileset capture would leave the terrain
#: fields -- Inside, Outside, Shared setting -- unphotographed, which is where
#: this session's work actually is.
#:
#: The fourth element is whether the arm also gets a second capture with the
#: form scrolled to its end. The Advanced disclosure is where Dimensions,
#: target cell, Palette, Dither and Outline live, and the form is a good deal
#: taller than the window -- so opening the disclosure is necessary and not
#: sufficient: the section opens below the fold and the picture still shows
#: none of it. Only one of the two tileset arms takes that second shot, because
#: their Advanced sections are the same controls.
SHEET_ARMS = (
    ("tileset-materials", "tileset", {"tile_mode": "materials"}, False),
    ("tileset-terrain", "tileset", {"tile_mode": "terrain"}, True),
    # ``walk8`` and not the stored default. The default is ``turnaround``,
    # which is one of ``generation.SPRITE_LEGACY_MODES`` -- and the Directions
    # row is drawn only for the planned kinds, so the default arm photographs
    # the Action combo and nothing beside it.
    ("sprite", "sprite_sheet", {"sheet_layout": "walk8"}, True),
)

#: The scrolling child ``settings_2d.draw`` puts the whole form in.
FORM_CHILD = "2d-form"


def _scroll_form(app, child: str, *, to_end: bool) -> bool:
    """Park a named child window at the top or the bottom of its content.
    -> whether it was found.

    imgui's scroll setters are frame-local -- ``set_scroll_y`` applies to the
    window currently being built -- and this harness works between frames, so
    the only handle available is the retained ``ImGuiWindow``. A child's
    retained name is ``"<parent>/<str_id>_<id>"`` and the id is not knowable
    from out here, hence the scan rather than ``find_window_by_name``.

    Writing ``scroll`` rather than ``scroll_target``: a target is consumed and
    cleared by the next ``Begin``, so it would move the form for one frame and
    the capture's settle loop would draw several more.
    """
    from imgui_bundle import imgui

    suffix = f"/{child}_"
    for window in imgui.get_current_context().windows:
        if suffix in window.name:
            window.scroll = (window.scroll.x, window.scroll_max.y if to_end else 0.0)
            return True
    return False


def _capture_sheet_arms(app, out: Path, theme_name: str) -> None:
    """Create's reference stage once per sheet arm, Advanced open.

    The mode walk photographs Create on whatever ``asset_type`` the form
    remembers, and a fresh home remembers Image -- the one arm with no sheet
    controls on it. So this is not a nicety: without it the corpus has no
    picture of any control on either sheet branch, while appearing to have five
    pictures of the form.
    """
    state = app.app_ctx.state
    before = dict(state.form_2d)
    try:
        for name, arm, extra, advanced in SHEET_ARMS:
            _seed_sheet_form(app, arm=arm)
            state.form_2d.update(extra)
            # Back to the top first: the child is retained, so an arm that
            # scrolled would leave the next arm's own picture halfway down a
            # form whose fields have all changed underneath it.
            _scroll_form(app, FORM_CHILD, to_end=False)
            _capture(app, out / f"{theme_name}-create-{name}.png")
            if advanced and _scroll_form(app, FORM_CHILD, to_end=True):
                _capture(app, out / f"{theme_name}-create-{name}-advanced.png")
    finally:
        # Restored, or the light and pixel passes would draw every remaining
        # capture on the arm the dark pass left behind -- the same
        # wrong-screen-under-the-right-filename failure ``_capture_popups``
        # records twice.
        state.form_2d.clear()
        state.form_2d.update(before)
        state.create_advanced = False
        _scroll_form(app, FORM_CHILD, to_end=False)


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
        "--modes",
        default=None,
        help="comma-separated workspace subset for targeted captures",
    )
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
        "--troupe",
        action="store_true",
        help="seed a finished character sheet and select it, so Troupe shows a "
        "sprite, a populated Sheet pane and live handoffs rather than four "
        "empty-state panes",
    )
    ap.add_argument(
        "--sheets",
        action="store_true",
        help=(
            "also capture Create's reference stage once per sheet arm -- the "
            "two tileset layouts and the sprite sheet -- and seed a palette so "
            "the Palette combo, which only draws when the palette folder has "
            "something in it, is on screen. The mode walk cannot reach any of "
            "this: it draws whichever asset type the form remembers, and a "
            "fresh home remembers Image."
        ),
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
        "--tour",
        action="store_true",
        help=(
            "also capture the guided tour, which is not reachable from the mode "
            "pass for the same reason the manual is not: it is an overlay. Two "
            "captures, one per tour, on the mode the tour's own first anchored "
            "step is about -- so the scrim's hole has a real control in it "
            "rather than framing an empty screen."
        ),
    )
    ap.add_argument(
        "--overlays",
        action="store_true",
        help=(
            "also capture the surfaces the mode pass cannot reach -- the "
            "first-run question and the manual -- plus the expanded "
            "navigation rail. The mode pass is derived from modes.KEYS, so "
            "none of them is reachable by it: a screen that is not a mode "
            "is a screen nobody would look at."
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
            "also capture the confirm and prompt containers, the model matte "
            "modal, and Plotter's new-map modal"
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
    ap.add_argument(
        "--size",
        type=_size,
        default=None,
        metavar="WIDTHxHEIGHT",
        help=(
            "capture at an exact window size, for example 1100x700; combine "
            "with --scale to exercise narrow high-density layouts"
        ),
    )
    args = ap.parse_args()
    selected_modes = MODES
    if args.modes:
        selected_modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
        unknown = [mode for mode in selected_modes if mode not in MODES]
        if unknown:
            ap.error(f"unknown mode(s): {', '.join(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.components:
        os.environ["WARLOCK_DEV_COMPONENTS"] = "1"

    from imgui_bundle import imgui

    from warlock.studio import theme as theme_mod
    from warlock.studio import tokens

    app = boot(args.scale, args.size)
    if args.seed:
        _seed(app)
    if args.asset:
        _seed_asset(app)
    if args.tile:
        _seed_tile(app, args.tile)
    if args.review:
        _seed_review(app)
    if args.troupe:
        # Troupe's four panes all answer "pick a character on the left" with
        # nothing selected, so the set shipped three pictures of a mode with
        # no character in it -- no sprite, no frame table, no Sheet pane, and
        # both handoffs greyed. Everything the preview reads is a file, so
        # this costs no GPU (see ``_appharness.seed_troupe``).
        _seed_troupe(app)
    if args.sheets:
        # Before the theme loop, because the folder listing behind the Palette
        # combo is read once per frame and a palette written mid-pass would
        # appear in some themes and not others.
        _seed_palette(app)
    app.app_ctx.state.tile_preview = bool(args.tile_preview)
    # **The doctor banner, off, before anything is photographed.** It is a
    # property of the machine that ran the capture -- which weights happen to
    # be on its disk -- and not of any mode, and it is a full-width strip that
    # pushes every pane down by its own height in some pictures and not others.
    # Worse, it prints the path it looked in: 44 of the 84 images in this repo
    # shipped with a developer's real home directory, username and all, drawn
    # across the top of them. ``exercise_mode`` dismisses it for the first two
    # reasons; this one has the third.
    app.app_ctx.state.dismiss_errors()
    try:
        for name in args.themes.split(","):
            tokens.set_theme(name)
            theme_mod.apply(imgui)
            print(f"{name}:", flush=True)
            for mode in selected_modes:
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
            if args.sheets:
                _capture_sheet_arms(app, args.out, name)
            if args.tour:
                from warlock.studio.panes import tour as tour_pane
                from warlock.studio.tour import TOURS

                state = app.app_ctx.state
                # The first-run question owns the screen ahead of every other
                # overlay, and a fresh WARLOCK_HOME is exactly what a capture
                # run has -- so without this the whole pass photographs the
                # setup modal and files it under the tour's name. That is the
                # harness's own recurring failure, and the reason
                # ``recovery_offered`` is set above.
                app.app_ctx.first_run = False
                for one in TOURS:
                    # Park on the mode the tour is about before starting it.
                    # The step's anchor is only marked by the pane that draws
                    # it, so a tour photographed from Home rings nothing and
                    # the picture silently shows a plain dimmed screen -- the
                    # same class of lie as the new-map modal captured from
                    # Create.
                    step = next((s for s in one.steps if s.anchor and s.mode), None)
                    state.mode = (step.mode if step else None) or "home"
                    tour_pane.start(app.app_ctx, one.key)
                    if step is not None:
                        state.tour.index = one.steps.index(step)
                    _capture(app, args.out / f"{name}-tour-{one.key}.png")
                    tour_pane.stop(app.app_ctx)
                state.mode = "home"
            if args.overlays:
                from warlock.studio.manual import render as manual_render
                from warlock.studio.panes import first_run

                state = app.app_ctx.state
                # The setup question, which is the first screen a new install
                # shows and the only one nothing else could photograph: it is
                # not a mode, and it is gone for good once answered. Forced on
                # rather than waited for, because a capture run's home may
                # already carry the marker.
                app.app_ctx.first_run = True
                if not app.app_ctx.first_run_info:
                    app.app_ctx.first_run_info = first_run.snapshot(app.app_ctx)
                state.mode = "home"
                _capture(app, args.out / f"{name}-first-run.png")
                app.app_ctx.first_run = False
                state.mode = create_stages.MODE
                state.create_stage = "mesh"
                manual_render.open_at(app.app_ctx, ("20-overview", None))
                _capture(app, args.out / f"{name}-manual.png")
                manual_render.close(app.app_ctx)
                state.create_stage = "reference"
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
                # Three passes down the scroller, not one. The gallery is a
                # popup with a scrolling child and a capture photographs one
                # frame, so a single shot showed only the blocks above the
                # fold -- four of the ten the catalogue grew to on 2026-09-05.
                # A catalogue whose pictures omit most of it is not one.
                for index, fraction in enumerate((0.0, 0.5, 1.0)):
                    component_gallery.scroll_to(fraction)
                    suffix = "" if index == 0 else f"-{index + 1}"
                    _capture(app, args.out / f"{name}-components{suffix}.png")
                imgui.internal.close_popups_except_modals()
            if args.popups:
                _capture_popups(app, args.out, name)
    finally:
        app.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
