"""The rules the four document modes share, once.

Inker, Clay, Plotter and Packwright grew four copies of each of these, and four
copies of a rule are three places for it to change without the fourth. The pins
at the bottom are the point of the file: the modes must reach for *this* object
rather than keeping a fourth copy that happens to agree today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from warlock.studio import docmodes


class _Confirms:
    def __init__(self) -> None:
        self.asked: list[Any] = []

    def ask(self, confirm: Any) -> None:
        self.asked.append(confirm)


class _State:
    def __init__(self) -> None:
        self.inker = None
        self.preview: dict[str, Any] = {}


class FakeCtx:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.state = _State()
        self.confirms = _Confirms()
        self.submitted: list[str] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        return self.accept


class _Tab:
    def __init__(self, dirty: bool = False) -> None:
        self.saving = False
        self.dirty = dirty


class _Docs:
    def __init__(self, *dirty: bool) -> None:
        self.docs = [_Tab(flag) for flag in dirty]

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)


# --- purity -------------------------------------------------------------------


def test_it_imports_no_window_and_no_service_at_module_scope():
    """The ``recents`` pin, second instance. ``docmodes`` is imported from
    ``inker_state`` and ``plotter_state``, which every headless test of a
    document loads -- so a window import here would put imgui behind them.

    Parsed rather than grepped, because this module's own docstring names the
    three things it imports *lazily* and a substring scan cannot tell the two
    apart. numpy is the one third-party name allowed: :func:`decode_rgba`
    returns one.
    """
    import ast

    source = (
        Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio" / "docmodes.py"
    ).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add("." * node.level + (node.module or "").split(".")[0])
    assert roots <= {"__future__", "pathlib", "typing", "numpy"}, roots


# --- start_save ---------------------------------------------------------------


def test_an_accepted_submit_leaves_the_tab_locked():
    ctx, tab = FakeCtx(), _Tab()
    docmodes.start_save(ctx, tab, "mode-save:1", lambda: None)
    assert tab.saving is True
    assert ctx.submitted == ["mode-save:1"]


def test_a_refused_submit_unlocks_the_tab():
    """The runner refuses a key already in flight; leaving the flag set is what
    made a tab read-only forever after a double press."""
    ctx, tab = FakeCtx(accept=False), _Tab()
    docmodes.start_save(ctx, tab, "mode-save:1", lambda: None)
    assert tab.saving is False


# --- title_for ----------------------------------------------------------------


def test_a_title_is_the_file_name_and_nothing_is_untitled():
    assert docmodes.title_for(Path("D:/atlases/hero.wpack")) == "hero.wpack"
    assert docmodes.title_for(None) == "Untitled"


# --- decode_rgba --------------------------------------------------------------


def test_an_image_decodes_to_rgba_uint8(tmp_path):
    from PIL import Image

    path = tmp_path / "one.png"
    Image.new("RGB", (5, 3), (10, 20, 30)).save(path)
    pixels = docmodes.decode_rgba(path)
    assert pixels.shape == (3, 5, 4)
    assert pixels.dtype == np.uint8
    assert tuple(pixels[0, 0]) == (10, 20, 30, 255)


# --- textures -----------------------------------------------------------------


class _Texture:
    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name

    def release(self) -> None:
        self.log.append(f"release:{self.name}")


class _Renderer:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def forget_texture(self, texture: Any) -> None:
        self.log.append(f"forget:{texture.name}")


@pytest.fixture()
def _renderer(monkeypatch):
    log: list[str] = []
    from warlock.studio import imgui_backend

    monkeypatch.setattr(imgui_backend, "current", lambda: _Renderer(log))
    return log


def test_a_texture_is_forgotten_before_it_is_released(_renderer):
    """The other order leaves the backend holding a dead object under a GL name
    the driver will reuse."""
    docmodes.forget_texture(_Texture(_renderer, "a"))
    assert _renderer == ["forget:a", "release:a"]


def test_a_release_sweep_pops_only_the_matching_prefix(_renderer):
    ctx = FakeCtx()
    ctx.state.preview.update(
        {
            "mode_tex:t1:atlas": _Texture(_renderer, "a"),
            "mode_tex:t1:atlas:gen": 4,
            "mode_tex:t2:atlas": _Texture(_renderer, "b"),
            "other:t1": _Texture(_renderer, "c"),
        }
    )
    docmodes.release_prefix(ctx, "mode_tex:t1:")
    assert set(ctx.state.preview) == {"mode_tex:t2:atlas", "other:t1"}
    assert _renderer == ["forget:a", "release:a"]


def test_a_release_sweep_takes_the_whole_prefix_when_asked(_renderer):
    ctx = FakeCtx()
    ctx.state.preview.update(
        {
            "mode_tex:t1:atlas": _Texture(_renderer, "a"),
            "mode_tex:t2:atlas": _Texture(_renderer, "b"),
        }
    )
    docmodes.release_prefix(ctx, "mode_tex:")
    assert ctx.state.preview == {}
    assert _renderer == ["forget:a", "release:a", "forget:b", "release:b"]


# --- guard --------------------------------------------------------------------


def test_the_guard_is_silent_before_the_mode_has_ever_been_opened():
    """``getattr`` rather than ``ensure``: asking which documents are unsaved
    must not create the state that says none is."""
    ctx = FakeCtx()
    calls: list[str] = []
    assert docmodes.guard(ctx, "inker", "drawing", "drawings", "quit", lambda: calls.append("go"))
    assert calls == ["go"] and ctx.state.inker is None and ctx.confirms.asked == []


def test_a_clean_state_proceeds_without_asking():
    ctx = FakeCtx()
    ctx.state.inker = _Docs(False, False)
    calls: list[str] = []
    assert docmodes.guard(ctx, "inker", "drawing", "drawings", "quit", lambda: calls.append("go"))
    assert calls == ["go"] and ctx.confirms.asked == []


def test_one_question_covers_however_many_are_dirty():
    ctx = FakeCtx()
    ctx.state.inker = _Docs(True)
    calls: list[str] = []
    assert (
        docmodes.guard(ctx, "inker", "drawing", "drawings", "quit", lambda: calls.append("go"))
        is False
    )
    assert calls == []
    assert len(ctx.confirms.asked) == 1
    assert "One drawing has unsaved changes" in ctx.confirms.asked[0].message
    assert "if you quit" in ctx.confirms.asked[0].message

    ctx.state.inker = _Docs(True, True, False)
    docmodes.guard(ctx, "inker", "drawing", "drawings", "close", lambda: calls.append("go"))
    assert "2 drawings have" in ctx.confirms.asked[-1].message
    ctx.confirms.asked[-1].on_confirm()
    assert calls == ["go"]


# --- the pins -----------------------------------------------------------------


def test_every_mode_reaches_for_the_shared_helpers():
    """The point of the module. A fourth copy that happens to agree today is
    the drift this replaced."""
    from warlock.studio import clay_mode, inker_mode, inker_state, plotter_io, plotter_state
    from warlock.studio.panes import inker_textures, packwright_textures, plotter_textures

    assert clay_mode._start is docmodes.start_save
    assert inker_mode._start is docmodes.start_save
    assert plotter_io._start is docmodes.start_save
    assert plotter_io._decode is docmodes.decode_rgba
    assert inker_state.title_for is docmodes.title_for
    assert plotter_state.title_for is docmodes.title_for
    for module in (inker_textures, plotter_textures, packwright_textures):
        assert not hasattr(module, "_forget"), f"{module.__name__} kept its own copy"


def test_clay_titles_a_tab_by_stem_on_purpose():
    """Deliberately *not* shared: a Clay tab is named for the document rather
    than for the file it came from."""
    from warlock.studio import clay_state

    assert clay_state.title_for is not docmodes.title_for
    assert clay_state.title_for(Path("D:/x/hero.wblk")) == "hero"
