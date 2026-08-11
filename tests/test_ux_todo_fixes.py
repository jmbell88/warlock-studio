"""Regressions for the UX review's verified bugs (``docs/TODO.md``).

Five promises that were stated in one place and not kept in another: a
space-to-pan hold that only ever latched on, two toasts asking for a level that
does not exist, a "works everywhere" shortcut that died the moment a text field
had focus, two permanent deletes with nothing between them and the file, and a
Clay document model whose ``close`` had no caller at all.

The toast-level and Clay-close checks are source scans on purpose, for
``test_mode_writes``'s reason: the failure they guard is a *call site* getting
it wrong later, which a behaviour test on the current call sites cannot see.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock.studio import plotter_mode
from warlock.studio.state import TOAST_LEVELS

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock"
STUDIO = SRC / "studio"


# --- the toast vocabulary is closed ------------------------------------------

def _toast_levels(path: Path):
    """Every ``x.toast(msg, "level")`` in one file, as (line, level).

    Parsed rather than matched: the third argument is an *action* name, and a
    regex loose enough to skip over an f-string message is also loose enough to
    read that action as the level -- which is what the first version of this
    test did, and it reported four call sites that were all correct.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        level = node.args[1]
        if name == "toast" and isinstance(level, ast.Constant) and isinstance(level.value, str):
            yield node.lineno, level.value


def test_every_toast_level_literal_is_a_level_that_exists():
    """"warning" is not a level; ``warn`` is. A level with no entry in
    ``TOAST_LEVELS`` silently falls back to ``info``, so the two call sites
    that asked for a warning got the grey, glyph-less, non-sticky,
    mouse-pass-through toast -- and nothing anywhere said so."""
    offenders = [
        f"{path.relative_to(SRC).as_posix()}:{n} -> {level!r}"
        for path in sorted(SRC.rglob("*.py"))
        for n, level in _toast_levels(path)
        if level not in TOAST_LEVELS
    ]
    assert offenders == []


# --- space-to-pan is a hold --------------------------------------------------


class _FakeKey:
    """Just enough of ``pygame.key`` for ``handle_key``'s space branch."""

    @staticmethod
    def get_mods() -> int:
        return 0

    @staticmethod
    def name(key: int) -> str:
        return "space"


@pytest.fixture
def plotter_ctx(monkeypatch):
    import pygame

    ctx = SimpleNamespace(state=SimpleNamespace(plotter=None))
    plotter_mode.ensure(ctx)
    monkeypatch.setattr(pygame, "key", _FakeKey)
    return ctx


def test_space_clears_on_the_key_up(plotter_ctx):
    """The latch: ``handle_key`` returned on every non-KEYDOWN event before it
    reached the space branch, so ``space_held`` went True on the first press
    and stayed True for the rest of the session -- every left-drag panning
    instead of drawing, with no way back short of restarting the app."""
    import pygame

    state = plotter_mode.ensure(plotter_ctx)

    down = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_SPACE)
    assert plotter_mode.handle_key(plotter_ctx, down) is True
    assert state.space_held is True

    up = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_SPACE)
    assert plotter_mode.handle_key(plotter_ctx, up) is True
    assert state.space_held is False


def test_a_key_up_that_is_not_space_is_still_ignored(plotter_ctx):
    """The rest of the KEYDOWN filter stays: only space is a hold."""
    import pygame

    up = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_f)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pygame, "key", SimpleNamespace(get_mods=lambda: 0, name=lambda k: "f"))
        assert plotter_mode.handle_key(plotter_ctx, up) is False


# --- the global shortcuts survive a focused text field -----------------------


def _passes(key_name: str, mods: int, monkeypatch) -> bool:
    import pygame

    from warlock.studio.main import App

    # The modifiers go on the *event*, which is where the filter now reads them
    # from: ``event.mod`` is the state at the instant the key went down, and
    # ``pygame.key.get_mods()`` is the state whenever the batched event is
    # finally drained -- so a Ctrl released between the press and the drain used
    # to read as never held, and the chord was silently swallowed. That is
    # ``review_mode.handle_key``'s stated rule, applied here on 2026-08-11.
    # ``get_mods`` is still stubbed so a regression back to it fails loudly
    # rather than reading the real keyboard.
    monkeypatch.setattr(
        pygame, "key", SimpleNamespace(get_mods=lambda: 0, name=lambda k: key_name)
    )
    return App._passes_text_field(SimpleNamespace(type=pygame.KEYDOWN, key=0, mod=mods))


def test_a_modifier_chord_reaches_the_shortcuts_while_typing(monkeypatch):
    """Ctrl+K is documented as the one binding that works in every mode
    (``docs/manual/14-shortcuts.md``, ``panes/settings_2d.py``) and it was
    dead in the 2D prompt box -- which is exactly where you are when you want
    to jump somewhere else."""
    import pygame

    assert _passes("k", pygame.KMOD_CTRL, monkeypatch) is True


def test_the_function_keys_reach_the_shortcuts_while_typing(monkeypatch):
    assert _passes("f1", 0, monkeypatch) is True
    assert _passes("f10", 0, monkeypatch) is True


def test_a_plain_key_still_belongs_to_the_field(monkeypatch):
    """The half that must not change: a letter typed into a field is a letter,
    and Shift is part of typing it."""
    import pygame

    assert _passes("k", 0, monkeypatch) is False
    assert _passes("k", pygame.KMOD_SHIFT, monkeypatch) is False


def test_the_fields_own_ctrl_chords_do_not_leak_out(monkeypatch):
    """imgui binds Ctrl+Z/Y/X/C/V/A inside an input-text widget. Letting them
    through would undo the *document* while the user renamed a layer."""
    import pygame

    for name in ("z", "y", "x", "c", "v", "a"):
        assert _passes(name, pygame.KMOD_CTRL, monkeypatch) is False


# --- a permanent delete is asked about first ---------------------------------


def test_the_pose_and_sheet_deletes_go_through_a_confirm():
    """Both sat one pixel from a save button and deleted on the click. The
    scan is on the source because the alternative is driving two imgui panes;
    what it pins is that neither ``Delete`` submits directly again."""
    for name in ("pose_panel.py", "sheet_panel.py"):
        text = (STUDIO / "panes" / name).read_text(encoding="utf-8")
        assert "_ask_delete" in text, name
        # Either spelling of "a modal is raised": both panes went through
        # ``dialogs.ask_delete`` on 2026-08-11, which is the shared wording for
        # this question -- ``ctx.confirms.ask`` with the Delete/Keep labels --
        # rather than a third copy of it per pane.
        assert ("confirms.ask" in text) or ("dialogs.ask_delete" in text), name


# --- Clay's documents can be closed ------------------------------------------


def test_clay_state_close_has_a_caller():
    """``ClayState.close`` existed, closed to the neighbour, and nothing ever
    called it: Clay could open documents and never shut one, so a dirty-quit
    prompt asked about documents the user had no way to see."""
    from warlock.studio import clay_mode

    assert hasattr(clay_mode, "close_tab")
    text = (STUDIO / "clay_mode.py").read_text(encoding="utf-8")
    assert "state.close(uid)" in text
    # And both routes to it: the tab bar's x and Ctrl+W. The bar is drawn by
    # ``main`` rather than by a pane, because ``main`` owns Clay's centre pane.
    assert "close_tab" in (STUDIO / "main.py").read_text(encoding="utf-8")
    assert 'name == "w"' in text
