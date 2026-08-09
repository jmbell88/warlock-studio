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

# Modes in switch order. Every one of them, because "which pane did nobody
# look at" is exactly the question this answers.
MODES = ("home", "2d", "3d", "inker", "clay", "review", "manual", "settings")

# Frames drawn before the read. Three is not a guess: one to build, one for the
# textures asked for on it to upload, one for anything those made visible.
WARMUP = 3


def _capture(app, path: Path) -> None:
    import pygame
    from PIL import Image

    for _ in range(WARMUP):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--themes", default="dark,light")
    ap.add_argument(
        "--seed",
        action="store_true",
        help="open an Inker canvas and a Clay model before capturing",
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
    app.setup_runtime()
    app.setup_context()
    if args.seed:
        _seed(app)
    try:
        for name in args.themes.split(","):
            tokens.set_theme(name)
            theme_mod.apply(imgui)
            print(f"{name}:", flush=True)
            for mode in MODES:
                app.app_ctx.state.mode = mode
                _capture(app, args.out / f"{name}-{mode}.png")
    finally:
        app.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
