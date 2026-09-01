"""The imgui context the Inker pane tests press controls inside.

The twenty-odd lines below were written four times before they were written
once: ``test_pattern_fill``, ``test_timeline_cel_opacity_input``,
``test_timeline_cel_z_input`` and ``test_timeline_note_input`` each carried a
hand copy, and one of them said so in its docstring ("``...``'s fixture
verbatim"). They are here so a sixth pane test does not make it six.

It is a *real* imgui context with no GL: nothing is rendered, only laid out,
which is what lets a test press a control and read what the press did. Built
and destroyed per test rather than shared, because two imgui contexts over one
GL context take the process down (see ``test_pane_guard``).

**Deliberately not a ``conftest.py``, and deliberately not a fixture.** Two
reasons, both learned the hard way:

* ``tests/`` has no ``__init__.py``, so every ``conftest.py`` under it is
  importable as the bare name ``conftest`` -- and ``test_fakes_match_real_signatures``
  and ``test_gpu_lane_selection`` both do ``from conftest import ...`` meaning
  the one at ``tests/``. A second one in this directory wins that name whenever
  it is imported first, which under xdist it sometimes is, and those two
  modules then fail to *collect*. (``tests/plotter/conftest.py`` had already
  made that latent; a third made it fire every run.)
* Importing a fixture by name into a module whose tests take a ``ui``
  parameter makes every one of those parameters a redefinition, which is fifty
  lint errors saying nothing.

So this is a context manager, and each module keeps a four-line fixture around
it. What is shared is the part that is easy to get wrong.
"""

from __future__ import annotations

import contextlib

from warlock.studio import probe


@contextlib.contextmanager
def imgui_context(monkeypatch):
    """An imgui context with the control census on, torn down after."""
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600.0, 950.0)
    io.delta_time = 1.0 / 60.0
    io.fonts.add_font_default()
    # No renderer backend here, so imgui has to be told not to expect one to
    # have built the font atlas for it.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)
    monkeypatch.setattr(probe, "ENABLED", True)
    try:
        yield imgui
    finally:
        imgui.destroy_context(ctx)
        if previous is not None:
            imgui.set_current_context(previous)
