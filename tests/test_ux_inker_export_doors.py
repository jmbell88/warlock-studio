"""Inker's five exports are one record, presented twice.

Until 2026-09-05 the five labels, tooltips and refusal sentences lived in
``panes/inker_timeline.py`` and nowhere else, as a second toolbar row under the
transport. At 1280x800, scale 1.0, that row overflowed: three of the five
collapsed into a ``...`` menu, "Skip empty" was clipped mid-word, and the onion
row below it was cut off by the pane's bottom edge -- so what leaves the app was
the part of the timeline you could not see.

The doors are :data:`inker_export.DOORS` now, drawn by Inker's bridge and
offered as File rows. This file is the pass-2 shared-vocabulary scan extended
onto them: a label spelled in one place, an enabled state answered by one
function, and no refusal without a sentence. It fails against the unfixed code
on the first assertion (the labels were literals in the timeline pane).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock.studio import inker_export, menus
from warlock.studio.panes import inker_generate, inker_timeline

LABELS = (
    "Export sheet...",
    "Export GIF...",
    "Export PNGs...",
    "Export sheet per tag...",
    "Export sheet per layer...",
)


def _src_files() -> dict[Path, str]:
    root = Path(inspect.getsourcefile(menus)).parent
    return {p: p.read_text(encoding="utf-8") for p in root.rglob("*.py")}


def test_each_export_label_is_spelled_in_exactly_one_module():
    """The pass-2 shared-vocabulary scan, extended onto the five doors.

    ``inker_ops.py`` is the one other module allowed to hold a door's label,
    and only for the two ops that predate the doors -- which ``menus`` then
    suppresses, so the File menu still spells each door once. Everything else,
    every pane included, must ask :mod:`inker_export`.
    """
    for label in LABELS:
        owners = sorted(p.name for p, text in _src_files().items() if label in text)
        assert owners[0] == "inker_export.py", (label, owners)
        assert set(owners) <= {"inker_export.py", "inker_ops.py"}, (label, owners)
    from warlock.studio import inker_ops

    for name in menus.SHADOWED_BY_DOORS:
        assert inker_ops.get(name).menu == "File"
    assert set(menus.SHADOWED_BY_DOORS) == {"export_gif", "export_sheet"}


def test_the_timeline_no_longer_draws_an_export_row():
    source = inspect.getsource(inker_timeline)
    assert "inker-timeline-out" not in source
    assert "_output_trailing" not in source
    # The strip is still drawn unconditionally, and the two view toggles that
    # act on playback stayed with it.
    assert "def _view_toggles" in source
    assert 'widgets.toggle("Onion"' in source
    assert 'widgets.toggle("Thumbs"' in source
    # The sheet knobs went with the doors -- none of them is read here.
    for name in ("export_scale", "export_arrange", "export_merge", "export_padding"):
        assert name not in source, name


def test_the_bridge_and_the_menu_both_resolve_the_doors_from_inker_export():
    bridge = inspect.getsource(inker_generate)
    assert "inker_export.doors()" in bridge
    assert "inker_export.door_state(" in bridge
    assert "inker_export.open_door(" in bridge
    menu = inspect.getsource(menus._inker_export_specs)
    assert "inker_export.doors()" in menu
    assert "inker_export.door_state(" in menu
    assert "inker_export.open_door(" in menu


def _tab(*, busy=False, tags=(), splits=1):
    doc = SimpleNamespace(anim=SimpleNamespace(tags=list(tags)))
    return SimpleNamespace(busy=busy, doc=doc, _splits=splits)


@pytest.fixture
def one_layer(monkeypatch):
    from warlock.studio.inker import sheetout

    monkeypatch.setattr(sheetout, "layer_splits", lambda doc: [object()])


def test_every_refused_door_carries_a_sentence(one_layer):
    for door in inker_export.doors():
        for tab in (None, _tab(busy=True), _tab()):
            enabled, reason = inker_export.door_state(door, tab)
            assert enabled or reason, (door.key, tab)


def test_the_two_split_doors_name_what_is_missing(one_layer):
    per_tag = next(d for d in inker_export.doors() if d.key == "per-tag")
    per_layer = next(d for d in inker_export.doors() if d.key == "per-layer")
    assert inker_export.door_state(per_tag, _tab()) == (
        False,
        "This document has no tags to split by.",
    )
    assert inker_export.door_state(per_layer, _tab()) == (
        False,
        "There is only one visible layer, so a split would write "
        "the sheet Export sheet already writes.",
    )
    assert inker_export.door_state(per_tag, _tab(tags=["walk"])) == (True, "")


def test_with_no_document_every_door_says_so():
    for door in inker_export.doors():
        assert inker_export.door_state(door, None) == (False, "No drawing is open.")


def test_open_door_dispatches_the_five_and_nothing_else(monkeypatch):
    seen = []
    for name in ("export_sheet", "export_gif", "export_pngs"):
        monkeypatch.setattr(
            inker_export, name, lambda ctx, tab, _n=name: seen.append(_n)
        )
    for name in ("export_per_tag", "export_per_layer"):
        monkeypatch.setattr(
            inker_export, name, lambda ctx, tab, kind, _n=name: seen.append(_n)
        )
    for door in inker_export.doors():
        inker_export.open_door(None, None, door.key)
    assert seen == [
        "export_sheet",
        "export_gif",
        "export_pngs",
        "export_per_tag",
        "export_per_layer",
    ]


def test_the_manual_says_where_the_five_doors_are():
    text = Path("docs/manual/28-inker.md").read_text(encoding="utf-8")
    assert "Export** block of the right-hand panel" in text
    assert "also a row in the **File** menu" in text
    assert "timeline's second row" not in text
    for label in ("Export sheet per tag", "Export sheet per layer"):
        assert label in text, label
