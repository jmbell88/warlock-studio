"""Regressions for the smaller entries of the 2026-09-02 review as they are
struck: the sentences the code contradicted, and the tables that had drifted.
"""

from __future__ import annotations

import inspect
import re
from importlib import import_module
from pathlib import Path

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


#: Files that record what was true on a date rather than claiming what is
#: true now. A count in one of them is correct as history and must not be
#: "fixed" -- ``FINDINGS.md`` quotes the very sentences it wants corrected.
_HISTORIES = {"CHANGELOG.md", "FINDINGS.md"}

#: Number words this project actually writes, for the workspace-count sweep.
_WORKSPACE_WORDS = {
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

#: "seven workspaces", "seven creative workspaces" -- the two phrasings the
#: docs use. Deliberately *not* "N of the seven workspaces", which is a
#: different sentence and correct with any N.
_WORKSPACE_PHRASE = re.compile(
    r"\b(" + "|".join(_WORKSPACE_WORDS) + r")\s+(?:creative\s+)?workspaces\b",
    re.IGNORECASE,
)


def _prose_files():
    """Every file whose workspace count is a live claim about this build.

    The repo root is walked as well as ``docs``/``src``, and it is the half
    that matters: the two files a new reader opens (``README.md`` and
    ``INSTALL.md``) are both there, and both said "six creative workspaces"
    while this test watched two directories that did not contain them.

    ``CHANGELOG.md``, ``FINDINGS.md`` and ``docs/measurements/`` are left out
    on purpose -- they are records of what was true on a date, and a count in
    them is correct as history. ``FINDINGS.md`` earns its place in that list
    the hard way: it *quotes* the wrong sentences it is asking somebody to fix.
    """
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.glob("*.md")):
        if path.name not in _HISTORIES:
            yield path
    for folder in ("docs", "src"):
        for path in sorted((root / folder).rglob("*")):
            if path.suffix not in {".md", ".py"}:
                continue
            if "measurements" in path.parts or path.name in _HISTORIES:
                continue
            yield path


def test_no_document_miscounts_the_workspaces():
    """The rail's own group is the answer; prose that names another is drift.

    Derived rather than hardcoded, for the reason the CI wheel step's
    ``len(modes.KEYS) == 11`` was deleted: a literal in a guard is a second
    place to update, and the update is exactly what gets forgotten.
    """
    from warlock.studio import modes

    group = modes.RAIL_GROUPS[modes.RAIL_GROUP_LABELS.index("Workspaces")]
    offenders = []
    for path in _prose_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _WORKSPACE_PHRASE.finditer(text):
            if _WORKSPACE_WORDS[match.group(1).lower()] != len(group):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.name}:{line}: {match.group(0)!r}")
    assert offenders == [], (
        f"the rail draws {len(group)} workspaces; these say otherwise: {offenders}"
    )


def test_no_document_still_describes_the_deleted_profiles_feature():
    """``README.md`` told every reader that Profiles "are a sheet over the
    reference form" long after Profiles was deleted (``settings_2d`` still
    carries the epitaph: "Kept when Profiles went"). A deleted feature in the
    file a new user reads first is worse than an undocumented one."""
    offenders = []
    for path in _prose_files():
        if path.suffix != ".md":
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            if re.search(r"\bProfiles\b", line) and "deleted" not in line.lower():
                offenders.append(f"{path.name}:{number}: {line.strip()!r}")
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
