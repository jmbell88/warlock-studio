"""Regressions for the UX review's verified bugs.

The review they come from was ``docs/TODO.md``, deleted on 2026-08-11
(`de87838`) along with the roadmap it doubled as; ``git log --diff-filter=D``
finds it. The bugs below are the record now, which is the point of writing a
regression rather than a checkbox.

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
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


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


# --- no new citations of the deleted roadmap ---------------------------------


def test_no_studio_module_cites_the_deleted_roadmap():
    """``docs/INVARIANTS.md``'s rule, made a ratchet over the UI half of the
    tree.

    A ``TODO.md §N`` citation resolves against ``docs/LEFTOVERS.md`` -- deleted
    2026-08-10 -- and never against the ``docs/TODO.md`` that replaced it,
    which had no sections and is itself deleted (`de87838`). Three of them
    survived in ``studio/`` pointing at reasoning that has since been written
    down properly in ``docs/measurements/``, so they now cite the measurement
    documents instead. The invariant says not to mint new ones at all; this is
    what makes that enforceable in the tree most likely to grow them.

    Scoped to ``studio/`` deliberately. Fourteen citations remain elsewhere in
    ``src`` and they are grandfathered: each names reasoning that has no other
    home yet, and rewriting them blind would replace a findable pointer with a
    vaguer one. Narrowing the scope is what makes this a ratchet rather than a
    migration.
    """
    offenders = [
        f"{path.relative_to(SRC).as_posix()}:{n}"
        for path in sorted(STUDIO.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "TODO.md" in line
    ]
    assert offenders == []


#: The plan files retired on 2026-08-15, named here so the scan below can look
#: for them without this module's own prose becoming an offender -- which is
#: also why the scan is scoped to ``src`` and ``scripts`` and never ``tests``.
RETIRED_PLANS = ("REDESIGN" ".md", "INKER_UPDATE" ".md", "NEXT_SESSION" ".md")


def test_no_module_cites_a_retired_plan_file():
    """The sibling of the ratchet above, for the three plans deleted on 2026-08-15.

    ``REDESIGN.md``, ``INKER_UPDATE.md`` and ``NEXT_SESSION.md`` described work
    that is merged; ``docs/INVARIANTS.md`` records their retirement and carries
    the one section worth keeping (the eighteen Aseprite divergences). Their
    eighty-odd citations were rewritten to name the *programme* rather than the
    file -- "the UI redesign, wave N" -- and unlike the ``TODO.md §N`` case
    above that rewrite loses nothing, because a wave's outcome is described in
    ``docs/INVARIANTS.md`` and the filename pointed only at a deleted plan.

    This is a full sweep rather than a narrowed ratchet for exactly that
    reason: there is no grandfathered site left to protect. Scoped to ``src``
    and ``scripts`` because ``tests`` contains this docstring.
    """
    offenders = [
        f"{path}:{n} -> {token}"
        for root in (SRC, SCRIPTS)
        for path in sorted(root.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for token in RETIRED_PLANS
        if token in line
    ]
    assert offenders == []


# --- a greyed button says why, in the places that were swept -----------------

#: The files the ``reason=`` sweep covered. A named list rather than "all of
#: ``studio/``", and the difference is the point: ``disabled_button`` grew the
#: parameter with ~90 call sites already written, so a repo-wide assertion
#: would be a migration disguised as a test and would fail on the first commit.
#: These are the surfaces where a greyed button is the *whole* interaction --
#: the four document bridges, the panels whose one job is a submit, and the two
#: settings panes -- and a new one added here has to explain itself.
REASON_SWEPT = (
    "panes/clay_bridge.py",
    "panes/inker_bridge.py",
    # Added in the Inker UX pass: this pane holds thirteen buttons that can
    # grey out and every one of them was silent about it, which is worse here
    # than on a bridge -- "Grow", "Shrink", "Border", "Feather" and "Crop to
    # selection" all grey out for the *same* reason, and a user meeting five
    # dead buttons at once has no way to learn it is one cause.
    "panes/inker_tools.py",
    "panes/inker_colors.py",
    "panes/packwright_bridge.py",
    "panes/plotter_bridge.py",
    "panes/candidates_panel.py",
    "panes/retarget_panel.py",
    "panes/sheet_panel.py",
    "panes/sprite_panel.py",
    "panes/settings_2d.py",
    "panes/settings_3d.py",
)


def _unexplained_disabled_buttons(path: Path):
    """``disabled_button`` calls that can grey out and cannot say why.

    A literal ``True`` for ``enabled`` is exempt: the button is never disabled,
    so a reason would be text that can never appear. Everything else is a
    button the user can meet greyed out.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "disabled_button":
            continue
        enabled = node.args[1] if len(node.args) > 1 else None
        always_on = isinstance(enabled, ast.Constant) and enabled.value is True
        if always_on:
            continue
        if not any(kw.arg == "reason" for kw in node.keywords):
            yield node.lineno


def test_a_greyed_button_on_a_swept_surface_says_why():
    """Absent controls make a UI feel like it is hiding things and a greyed one
    with no tooltip is worse -- it says "no" and refuses to say what would make
    it yes. ``disabled_button`` has taken a ``reason`` since the parameter was
    added; what it did not have was call sites passing one."""
    offenders = [
        f"{rel}:{n}"
        for rel in REASON_SWEPT
        for n in _unexplained_disabled_buttons(STUDIO / rel)
    ]
    assert offenders == []


def test_the_sweep_list_names_files_that_exist():
    """A guard on the guard: a renamed pane would otherwise drop out of the
    sweep silently, which is the shrink ``test_external_doc_links`` warns
    about in its own domain."""
    missing = [rel for rel in REASON_SWEPT if not (STUDIO / rel).exists()]
    assert missing == []


# --- a toast never forwards a bare exception ---------------------------------


def _bare_exception_toasts(path: Path):
    """Every ``x.toast(str(exc))`` in one file, as line numbers.

    The shape, not the variable name: ``str(<Name>)`` as the whole of a toast's
    message. Nothing else is looked for, because nothing else is the bug --
    ``f"could not read {path.name}: {exc}"`` is the same exception with a
    subject in front of it, which is exactly what this is asking for.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        message = node.args[0]
        if (
            name == "toast"
            and isinstance(message, ast.Call)
            and getattr(message.func, "id", "") == "str"
            and len(message.args) == 1
            and isinstance(message.args[0], ast.Name)
        ):
            yield node.lineno


def test_no_toast_forwards_a_bare_exception():
    """``packwright_mode``'s house rule, made a ratchet.

    A ``ctx.toast(str(exc))`` is library text with no subject in front of it:
    the user reads "axis 'seed' has no values" and has to work out what was
    being attempted, and the two call sites that did this -- planning a sweep
    and keeping a candidate -- were the two where the attempt was least
    guessable. A frame costs one f-string and turns the message into a
    sentence about their action.
    """
    offenders = [
        f"{path.relative_to(SRC).as_posix()}:{n}"
        for path in sorted(STUDIO.rglob("*.py"))
        for n in _bare_exception_toasts(path)
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

    down = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_SPACE, mod=0)
    assert plotter_mode.handle_key(plotter_ctx, down) is True
    assert state.space_held is True

    up = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_SPACE, mod=0)
    assert plotter_mode.handle_key(plotter_ctx, up) is True
    assert state.space_held is False


def test_a_key_up_that_is_not_space_is_still_ignored(plotter_ctx):
    """The rest of the KEYDOWN filter stays: only space is a hold."""
    import pygame

    up = SimpleNamespace(type=pygame.KEYUP, key=pygame.K_f, mod=0)
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
    (``docs/manual/16-shortcuts.md``, ``panes/settings_2d.py``) and it was
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
