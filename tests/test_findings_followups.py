"""Regressions for the smaller entries of the 2026-09-02 review as they are
struck: the sentences the code contradicted, and the tables that had drifted.
"""

from __future__ import annotations

import inspect
from importlib import import_module

import pytest


def test_every_recent_kind_has_an_opener_with_an_open_path():
    """``open_row``'s opener table and ``_KIND_MODES`` had drifted: a ``.wsng``
    row did nothing on click, with no toast."""
    from warlock.studio import recents
    from warlock.studio.panes import landing

    assert set(landing.KIND_OPENERS) == set(recents.KINDS)
    for kind, module in landing.KIND_OPENERS.items():
        assert kind in landing._KIND_MODES
        assert callable(import_module(f"warlock.studio.{module}").open_path)


def test_play_refuses_a_stale_buffer(monkeypatch):
    """The docstring and INVARIANTS said so; the code played the old bar."""
    import numpy as np
    from test_sirens_mode import FakeCtx, _tab

    from warlock.studio import sirens_audio, sirens_mode

    played: list = []
    monkeypatch.setattr(sirens_audio, "available", lambda: True)
    monkeypatch.setattr(
        sirens_audio, "play", lambda pcm, **_kw: played.append(pcm) or True
    )
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.pcm = np.zeros((10, 2), dtype=np.float32)
    tab.render_dirty = True
    assert sirens_mode.play(ctx, tab) is False
    assert played == []
    assert any("rendering" in message.lower() for message, _kind in ctx.toasts)
    tab.render_dirty = False
    assert sirens_mode.play(ctx, tab) is True


def test_xray_picks_through_the_surface():
    """Three places say a click in X-ray picks through the surface; the pick
    always passed the surface depth."""
    from warlock.studio import _view_pick

    source = inspect.getsource(_view_pick)
    assert 'getattr(self, "xray", False) else hit.t' in source


def test_the_shortcut_sheet_lists_the_sirens_clipboard():
    from warlock.studio import main

    sections = dict(main.shortcut_sections()) if callable(
        getattr(main, "shortcut_sections", None)
    ) else None
    if sections is None:
        pytest.skip("shortcut_sections is not a module-level function")
    keys = " ".join(key for key, _what in sections["Sirens"])
    assert "Ctrl+C" in keys and "Ctrl+V" in keys


def test_no_document_says_six_workspaces():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = [
        path
        for folder in ("docs", "src")
        for path in (root / folder).rglob("*")
        if path.suffix in {".md", ".py"}
        and "six workspaces" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


# --- Plotter JSON readers: a stored zero is a zero ----------------------------


def test_json_number_keeps_a_stored_zero():
    from warlock.studio.plotter.props import json_number

    assert json_number({"probability": 0}, "probability", 1.0) == 0.0
    assert json_number({"opacity": None}, "opacity", 1.0) == 1.0
    assert json_number({}, "x", 0) == 0.0


def test_a_tmj_keeps_a_stored_zero_opacity_and_origin():
    """``float(entry.get(k, d) or d)`` turned every stored ``0`` into ``d``:
    an invisible layer drew, and an object at the origin moved."""
    import json

    from warlock.studio.plotter import tmx

    payload = {
        "type": "map",
        "version": "1.10",
        "orientation": "orthogonal",
        "renderorder": "right-down",
        "infinite": False,
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "nextlayerid": 5,
        "nextobjectid": 6,
        "layers": [
            {
                "type": "objectgroup",
                "id": 3,
                "name": "Things",
                "opacity": 0,
                "parallaxx": 0,
                "draworder": "topdown",
                "objects": [
                    {
                        "id": 4,
                        "name": "spawn",
                        "x": 0,
                        "y": 0,
                        "opacity": 0,
                        "point": True,
                        "visible": True,
                    }
                ],
            }
        ],
    }
    loaders = {"image_loader": lambda s: None, "tsx_loader": lambda s: None}
    doc = tmx.read_tmj(json.dumps(payload).encode(), **loaders)
    layer = doc.layers[0]
    assert (layer.opacity, layer.parallax_x) == (0.0, 0.0)
    assert (layer.objects[0].x, layer.objects[0].opacity) == (0.0, 0.0)


# --- Sirens: one keystroke, one step --------------------------------------------


def _sirens():
    from test_sirens_mode import FakeCtx, _tab

    from warlock.studio import sirens_mode

    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    pattern = tab.doc.pattern(tab.doc.order[0])
    sirens_mode.set_caret(ctx, pattern=pattern.uid, row=0, channel=0, column=0)
    return ctx, tab, state, pattern


def test_a_typed_note_and_its_instrument_are_one_undo_step():
    from warlock.studio import sirens_mode
    from warlock.studio.sirens import document as D
    from warlock.studio.sirens import notes

    ctx, tab, state, pattern = _sirens()
    state.instrument = 0
    depth = len(tab.doc.history)
    assert sirens_mode.write_note(ctx, 0)
    assert len(tab.doc.history) == depth + 1
    assert int(pattern.cells[0, 0, D.NOTE]) != notes.EMPTY
    assert int(pattern.cells[0, 0, D.INSTRUMENT]) == 0
    assert tab.doc.undo()
    assert int(pattern.cells[0, 0, D.NOTE]) == notes.EMPTY
    assert int(pattern.cells[0, 0, D.INSTRUMENT]) == notes.EMPTY, "half a keystroke came back"


def test_a_two_digit_hex_entry_is_one_undo_step():
    from warlock.studio import sirens_mode
    from warlock.studio.sirens import document as D
    from warlock.studio.sirens import notes

    ctx, tab, state, pattern = _sirens()
    sirens_mode.set_caret(ctx, column=D.PARAM)
    depth = len(tab.doc.history)
    assert sirens_mode.write_hex(ctx, 0x4)
    assert sirens_mode.write_hex(ctx, 0xF)
    assert int(pattern.cells[0, 0, D.PARAM]) == 0x4F
    assert len(tab.doc.history) == depth + 1
    assert tab.doc.undo()
    assert int(pattern.cells[0, 0, D.PARAM]) == notes.EMPTY


def test_shift_up_at_row_zero_does_not_wrap():
    from warlock.studio import sirens_mode

    ctx, tab, state, pattern = _sirens()
    sirens_mode.move_caret(ctx, drow=-1, select=True)
    assert state.row == 0
    sirens_mode.move_caret(ctx, drow=-1)
    assert state.row == pattern.rows - 1, "an unshifted arrow still wraps"


def test_a_paste_ends_the_nibble_being_typed():
    from warlock.studio import sirens_mode
    from warlock.studio.sirens import document as D

    ctx, tab, state, pattern = _sirens()
    sirens_mode.set_caret(ctx, column=D.PARAM)
    assert sirens_mode.write_hex(ctx, 0x4)
    assert state.digit == 1
    sirens_mode.copy_selection(ctx)
    sirens_mode.paste(ctx)
    assert state.digit == 0


# --- first run: "still checking" is not "Ready" ---------------------------------


def test_a_row_still_checking_is_not_ready():
    from warlock.doctor import Check
    from warlock.studio.panes import first_run

    pending = Check("CUDA", True, "still checking in the background", False)
    assert first_run._settled(pending) is False
    assert first_run._settled(Check("CUDA", True, "NVIDIA thing", False)) is True
    assert first_run._settled(None) is False


# --- Inker: the size-3 pixel nib is a plus ---------------------------------------


def test_the_size_three_pixel_nib_is_a_plus():
    from warlock.studio.inker import brush

    assert brush.make_stamp(3, 1.0, "pixel").astype(int).tolist() == [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    assert int(brush.make_stamp(5, 1.0, "pixel").sum()) == 21, "5 is 3-5-5-5-3, as before"
