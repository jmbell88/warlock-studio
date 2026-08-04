"""Windows DPI awareness, sampled once at startup.

The app draws in physical pixels: with Per-Monitor-V2 awareness the pygame
window's size *is* the framebuffer size, so ``io.display_framebuffer_scale``
stays (1, 1) and the imgui backend's projection and scissor math never change.
Scaling lives entirely in content -- font pixel size and the style values --
via :data:`tokens.SCALE`.

Failure at any step degrades to 1.0: a wrong scale on an exotic setup is a
small UI, not a crash.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)


def make_process_dpi_aware() -> None:
    """Opt in to Per-Monitor-V2 before the window exists.

    Must run before ``pygame.display.set_mode``; once a window is created the
    process awareness is frozen. SDL may have set awareness already, in which
    case the call fails and that is fine -- :func:`window_scale` reads the
    effective DPI regardless of who set it.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        # -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(-4):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        log.warning("could not make the process DPI-aware; UI will scale at 1.0")


def system_scale() -> float:
    """The primary monitor's scale, readable before any window exists.

    Used only to size the initial window; :func:`window_scale` is the value
    the UI is actually built at once the window knows its monitor.
    """
    if sys.platform != "win32":
        return 1.0
    import ctypes

    try:
        dpi = ctypes.windll.user32.GetDpiForSystem()
        return (dpi / 96.0) if dpi else 1.0
    except (AttributeError, OSError):
        return 1.0


def window_scale(pygame_module) -> float:
    """The opened window's monitor scale (1.0 = 96 DPI)."""
    if sys.platform != "win32":
        return 1.0
    import ctypes

    try:
        hwnd = pygame_module.display.get_wm_info()["window"]
        dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        return (dpi / 96.0) if dpi else 1.0
    except (AttributeError, OSError, KeyError):
        return 1.0
