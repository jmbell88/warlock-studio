"""The startup splash: what is on screen while the runtime is starting.

``App.setup`` used to call ``runtime.start()`` *before* ``pygame.init()``, so
the slow half of startup -- the doctor checks, the worker construction, the
out-of-process bpy probe -- happened with no window at all. Splitting it into
``setup_window`` / ``setup_runtime`` / ``setup_context`` puts the window up
first, and this is what fills it meanwhile.

``Startup`` is the part that decides how long the splash stays, and it is
stdlib-only on purpose: no imgui, no moderngl, an injectable clock. Every
lifecycle rule about the splash -- the minimum hold, a load that outruns it, a
quit arriving mid-load, a load that raises -- is therefore assertable
headlessly, which is where the real hazard of this feature lives. The two
drawing helpers below import their GL/imgui dependencies inside the call.

The window is *live* during the splash, which is the whole new risk: the X
button works, so a quit has to be honoured, and it may arrive while
``runtime.start()`` is halfway through opening a store, a loop thread and a
worker. So a quit does not abandon the load -- it drops the minimum hold and
the splash stays up until the load lands, after which teardown unwinds a fully
constructed runtime through exactly the path it always did.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Show the logo for at least this long, however fast the load is. Longer than
# a flash, short enough that it is never the thing being waited for on a warm
# start (which is ~1 s) and invisible on a cold one (which is ~10 s).
MIN_SECONDS = 3.0

# Beside icon.ico, which setup_window already loads from this directory: one
# place for the two files that are pictures of the app rather than data. The
# whole of src/warlock ships in the wheel (hatch's ``packages``), so no
# per-file packaging rule is involved.
LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.png"

MESSAGE = "Starting Warlock Studio..."


class Startup:
    """A load running off the frame thread, and the clock that outlasts it.

    ``finished()`` is the whole contract: the splash comes down once the load
    has landed *and* ``min_seconds`` has passed -- except that an error or a
    quit request drops the minimum, because there is then nothing left to look
    at and nothing that a pause helps.
    """

    def __init__(
        self,
        load: Callable[[], Any],
        *,
        clock: Callable[[], float] = time.perf_counter,
        min_seconds: float = MIN_SECONDS,
    ) -> None:
        self._load = load
        self._clock = clock
        self.min_seconds = min_seconds
        # Whatever the load raised, re-raised on the frame thread by
        # ``raise_if_failed`` so ``App.run``'s existing "could not start"
        # branch reports it. Swallowing it here is how a splash turns a crash
        # into a window that simply never becomes an app.
        self.error: BaseException | None = None
        self.started_at = 0.0
        self._done = threading.Event()
        self._quit = False
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("this Startup has already run")
        self.started_at = self._clock()
        self._thread = threading.Thread(target=self._run, name="warlock-startup", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._load()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the frame thread
            self.error = exc
        finally:
            # Last, unconditionally: ``loaded`` is what lets a quit stop
            # pumping, so a load that raised before setting it would leave the
            # splash spinning forever on a window whose X button "worked".
            self._done.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- what the frame loop asks ------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._done.is_set()

    @property
    def quit_requested(self) -> bool:
        return self._quit

    def request_quit(self) -> None:
        """The X button, during the splash. Never abandons the load."""
        self._quit = True

    def elapsed(self) -> float:
        return self._clock() - self.started_at

    def finished(self) -> bool:
        if not self.loaded:
            return False
        if self.error is not None or self._quit:
            return True
        return self.elapsed() >= self.min_seconds

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise self.error


# -- the picture -----------------------------------------------------------


def load_logo(gl: Any) -> Any:
    """Decode the logo into a texture, or ``None``.

    Cosmetic, so a missing or unreadable file is a warning rather than a
    failure to start -- the same policy the GLB loader applies to a texture it
    cannot decode. The caller must ``release_logo`` it: two megabytes of
    decoded pixels are not worth holding for the session.
    """
    try:
        import moderngl
        from PIL import Image

        with Image.open(LOGO_PATH) as handle:
            image = handle.convert("RGBA")
            size, data = image.size, image.tobytes()
        texture = gl.texture(size, 4, data)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture
    except Exception:
        log.warning("could not load the splash logo from %s", LOGO_PATH, exc_info=True)
        return None


def release_logo(texture: Any, renderer: Any) -> None:
    """Forget it in the backend *before* freeing it.

    The renderer holds moderngl objects by GL name; releasing without the
    forget leaves a dead object under a name the driver will hand to the next
    texture, which then renders as this logo.
    """
    if texture is None:
        return
    try:
        if renderer is not None:
            renderer.forget_texture(texture)
    finally:
        texture.release()


def draw(texture: Any, window_size: tuple[int, int], message: str = MESSAGE) -> None:
    """One borderless window filling the host window: logo, then a line."""
    from imgui_bundle import imgui

    from . import widgets

    width, height = float(window_size[0]), float(window_size[1])
    imgui.set_next_window_pos(imgui.ImVec2(0.0, 0.0))
    imgui.set_next_window_size(imgui.ImVec2(width, height))
    flags = (
        imgui.WindowFlags_.no_decoration.value
        | imgui.WindowFlags_.no_move.value
        | imgui.WindowFlags_.no_saved_settings.value
        | imgui.WindowFlags_.no_bring_to_front_on_focus.value
        | imgui.WindowFlags_.no_nav.value
        | imgui.WindowFlags_.no_scrollbar.value
    )
    imgui.begin("##splash", None, flags)
    text_size = imgui.calc_text_size(message)
    if texture is not None:
        draw_w, draw_h = _fit(texture.size, (width, height))
        imgui.set_cursor_pos(
            imgui.ImVec2((width - draw_w) * 0.5, (height - draw_h - text_size.y * 3.0) * 0.5)
        )
        imgui.image(widgets.texture_ref(texture), imgui.ImVec2(draw_w, draw_h))
        imgui.dummy(imgui.ImVec2(0.0, text_size.y))
        imgui.set_cursor_pos_x((width - text_size.x) * 0.5)
    else:
        imgui.set_cursor_pos(
            imgui.ImVec2((width - text_size.x) * 0.5, (height - text_size.y) * 0.5)
        )
    widgets.muted(message)
    imgui.end()


def _fit(source: tuple[int, int], window: tuple[float, float]) -> tuple[float, float]:
    """The logo at its own size, shrunk to fit 60% of the window. Never grown."""
    src_w, src_h = float(source[0]), float(source[1])
    if src_w <= 0.0 or src_h <= 0.0:
        return (0.0, 0.0)
    scale = min(window[0] * 0.6 / src_w, window[1] * 0.6 / src_h, 1.0)
    return (src_w * scale, src_h * scale)
