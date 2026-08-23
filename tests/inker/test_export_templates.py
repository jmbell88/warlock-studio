"""Wave 4 Task 5: filename templates + export presets.

``sheetout.filename_for`` is unit-tested on its own in ``test_sheetout.py``;
this file is the wiring around it -- ``run_pngs``'s own per-frame template,
the two split defaults (``_split_stems``) reproduced through the new
machinery byte for byte, and the per-tab (session) / cross-session export
option memory: what a completed export records onto its tab, what the *next*
export dialog for that tab suggests, what ``persist`` carries between
sessions, and that none of it is anything the journal or ``dirty`` can see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from warlock.studio import inker, inker_mode, inker_state
from warlock.studio.inker_state import InkerDoc, InkerState

RED = (255, 0, 0, 255)


class _Settings:
    """The two methods ``persist`` and ``ensure`` use, over a plain dict."""

    def __init__(self, block: Any = None) -> None:
        self.data: dict = {"inker": block} if block is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


class _AppState:
    def __init__(self, inker: Any = None) -> None:
        self.inker = inker


class _Ctx:
    def __init__(self, state: Any, settings_block: Any = None) -> None:
        self.state = _AppState(state)
        self.settings = _Settings(settings_block)
        self.toasts: list[tuple[str, str]] = []
        self.submitted: list[str] = []
        self.accept = True
        self.run: Any = None

    def toast(self, message: str, kind: str = "info", **_kw: Any) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, run: Any) -> bool:
        self.submitted.append(key)
        self.run = run
        return self.accept


def _clip(frames: int = 3) -> Any:
    doc = inker.Document.blank(4, 4)
    ones = np.ones((2, 2), dtype=np.float32)
    doc.write_colour((0, 0, 2, 2), RED, ones)
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    return doc


def _tagged(frames: int = 4) -> Any:
    doc = _clip(frames)
    assert doc.add_tag("intro", 0, 1)
    assert doc.add_tag("walk", 2, 3)
    return doc


def _open(doc: Any = None) -> tuple[_Ctx, InkerState, InkerDoc]:
    tab = InkerDoc(doc=doc if doc is not None else _clip(), title="walk.ora")
    tab.path = None
    state = InkerState()
    state.add(tab)
    return _Ctx(state), state, tab


def _saved(monkeypatch, dest: Any) -> list[str]:
    """Monkeypatch the save picker to hand back ``dest``, recording every
    ``default_name`` it was opened with."""
    from warlock.studio import dialogs

    calls: list[str] = []

    def fake(title: str, default_name: str, filters: Any = None) -> Any:
        calls.append(default_name)
        return dest

    monkeypatch.setattr(dialogs, "save_file", fake)
    return calls


def _finish(ctx: _Ctx, state: InkerState, limit: int = 200) -> Any:
    """Pump the stepper to completion, then run the task the submit stashed."""
    for _ in range(limit):
        inker_mode.pump_export(ctx)
        if state.export is None:
            return ctx.run()
    raise AssertionError("the export never finished")


def _done(ctx: _Ctx, tab: InkerDoc, result: Any) -> None:
    """What the app does after ``_finish``: hand the result to ``on_task_done``."""
    inker_mode.on_task_done(
        ctx, type("Done", (), {"key": f"inker-export:{tab.uid}", "result": result})()
    )


# --- run_pngs' own template ----------------------------------------------------


def test_run_pngs_default_names_reproduce_the_old_index_naming(monkeypatch, tmp_path):
    ctx, state, tab = _open(_clip(3))
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_pngs(ctx, tab)
    result = _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "walk_0000.png",
        "walk_0001.png",
        "walk_0002.png",
    ]
    assert result["exported"] == tmp_path / "walk_0000.png"


def test_a_custom_template_renames_a_plain_png_sequence(monkeypatch, tmp_path):
    ctx, state, tab = _open(_clip(2))
    state.export_template = "frame-{frame}"
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_pngs(ctx, tab)
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "frame-0000.png",
        "frame-0001.png",
    ]


def test_a_plain_png_templates_missing_frame_key_collides_and_is_refused(
    monkeypatch, tmp_path
):
    ctx, state, tab = _open(_clip(2))
    state.export_template = "{title}"
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_pngs(ctx, tab)
    try:
        _finish(ctx, state)
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert not list(tmp_path.glob("*.png"))


# --- the two split defaults, through the new machinery -------------------------


def test_a_split_png_sequence_nests_the_frame_under_the_split_stem(monkeypatch, tmp_path):
    ctx, state, tab = _open(_tagged(4))
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "pngs")
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "walk_intro_0000.png",
        "walk_intro_0001.png",
        "walk_walk_0000.png",
        "walk_walk_0001.png",
    ]


def test_a_custom_split_template_still_numbers_frames_with_the_default(
    monkeypatch, tmp_path
):
    """The one field customises how outputs of a batch differ from each other
    -- the stem -- and never how frames inside one output differ, which stays
    :data:`sheetout.DEFAULT_FRAME_TEMPLATE` even when the stem's own template
    is a custom one."""
    ctx, state, tab = _open(_tagged(4))
    state.export_template = "{title}-{tag}"
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "pngs")
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "walk-intro_0000.png",
        "walk-intro_0001.png",
        "walk-walk_0000.png",
        "walk-walk_0001.png",
    ]


def test_a_layer_split_sheet_uses_the_layer_template_by_default(monkeypatch, tmp_path):
    ctx, state, tab = _open()
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_layer(ctx, tab, "sheet")
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["walk_Background.png"]


def test_a_custom_template_renames_a_tag_split_sheet(monkeypatch, tmp_path):
    ctx, state, tab = _open(_tagged())
    state.export_template = "{title}-{tag}"
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "walk-intro.png",
        "walk-walk.png",
    ]


# --- what a completed export records --------------------------------------------


def test_a_completed_export_records_the_dest_and_the_option_set(monkeypatch, tmp_path):
    ctx, state, tab = _open(_clip(2))
    state.export_scale = 2
    state.export_merge = True
    state.export_template = "{title}_{tag}"
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_sheet(ctx, tab)
    result = _finish(ctx, state)
    _done(ctx, tab, result)

    assert tab.export_dest == tmp_path / "walk.png"
    assert tab.export_options["scale"] == 2
    assert tab.export_options["merge"] is True
    assert tab.export_options["template"] == "{title}_{tag}"


def test_a_fresh_tab_has_recorded_nothing():
    _, _, tab = _open()
    assert tab.export_dest is None
    assert tab.export_options == {}


def test_a_flat_png_export_records_where_it_wrote_and_leaves_the_options(
    monkeypatch, tmp_path
):
    """The flat export records its own destination since 6.9 -- Repeat Last
    Export needs one to repeat -- and still hands back **no options**, so the
    nine sheet controls a sheet/GIF/PNG-sequence export recorded are not
    clobbered by a flatten that reads none of them."""
    ctx, state, tab = _open()
    tab.export_dest = tmp_path / "old.png"
    tab.export_options = {"scale": 5}
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "flat.png")
    inker_mode.export_png(ctx, tab)
    result = ctx.run()
    _done(ctx, tab, result)

    assert tab.export_dest == tmp_path / "flat.png"
    assert tab.export_kind == "png"
    assert tab.export_options == {"scale": 5}


def test_repeating_an_export_asks_no_dialog_and_writes_where_it_wrote(
    monkeypatch, tmp_path
):
    """The hot-path escape valve: configure once, then one key forever."""
    ctx, state, tab = _open()
    from warlock.studio import dialogs

    asked: list[int] = []

    def save_file(*args, **kwargs):
        asked.append(1)
        return tmp_path / "first.png"

    monkeypatch.setattr(dialogs, "save_file", save_file)
    inker_mode.export_png(ctx, tab)
    _done(ctx, tab, ctx.run())
    assert asked == [1]

    assert inker_mode.repeat_export(ctx, tab) is True
    _done(ctx, tab, ctx.run())
    assert asked == [1], "the repeat opened no dialog"
    assert tab.export_dest == tmp_path / "first.png"


def test_repeating_with_nothing_to_repeat_says_so():
    ctx, state, tab = _open()
    assert inker_mode.repeat_export(ctx, tab) is False
    assert state.tip is not None and "export once" in state.tip.text


# --- the next export dialog for that tab -----------------------------------------


def test_the_save_dialog_opens_in_the_tabs_own_last_folder(monkeypatch, tmp_path):
    ctx, state, tab = _open(_clip(2))
    folder = tmp_path / "prior"
    tab.export_dest = folder / "anything.png"
    calls = _saved(monkeypatch, folder / "untitled.png")

    inker_mode.export_sheet(ctx, tab)
    _finish(ctx, state)

    assert calls == [str(folder / "untitled.png")]


def test_a_tab_with_no_recorded_export_gets_a_bare_suggested_name(monkeypatch, tmp_path):
    ctx, state, tab = _open(_clip(2))
    calls = _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_sheet(ctx, tab)
    _finish(ctx, state)

    assert calls == ["untitled.png"]


def test_a_second_export_of_the_same_tab_keeps_a_live_edit(monkeypatch, tmp_path):
    """Exporting the *same* tab twice in a row must not put back over a
    control the user has just changed -- only a switch to another tab and
    back re-suggests this tab's own recorded settings."""
    ctx, state, tab = _open(_clip(2))
    state.export_scale = 2
    _saved(monkeypatch, tmp_path / "walk.png")
    inker_mode.export_sheet(ctx, tab)
    result = _finish(ctx, state)
    _done(ctx, tab, result)
    assert tab.export_options["scale"] == 2

    state.export_scale = 5  # a live edit, right before exporting again
    _saved(monkeypatch, tmp_path / "walk2.png")
    inker_mode.export_sheet(ctx, tab)
    assert state.export_scale == 5  # not put back to the recorded 2


def test_switching_tabs_and_back_re_suggests_this_tabs_own_settings(monkeypatch, tmp_path):
    ctx, state, tab_a = _open(_clip(2))
    tab_b = InkerDoc(doc=_clip(2), title="b.ora")
    tab_b.path = None
    state.add(tab_b)

    state.export_scale = 3
    state.export_merge = True
    _saved(monkeypatch, tmp_path / "a.png")
    inker_mode.export_sheet(ctx, tab_a)
    result = _finish(ctx, state)
    _done(ctx, tab_a, result)
    assert tab_a.export_options["scale"] == 3

    # A different tab's export changes the shared controls.
    state.export_scale = 1
    state.export_merge = False
    _saved(monkeypatch, tmp_path / "b.png")
    inker_mode.export_sheet(ctx, tab_b)
    _finish(ctx, state)

    # Back to tab A: its own recorded settings are suggested again.
    _saved(monkeypatch, tmp_path / "a2.png")
    inker_mode.export_sheet(ctx, tab_a)
    assert state.export_scale == 3
    assert state.export_merge is True


# --- cross-session: the last-used option set -------------------------------------


def test_the_last_used_export_options_persist_across_a_session():
    ctx, state, _tab = _open()
    state.export_scale = 4
    state.export_arrange = "horizontal"
    state.export_merge = True
    state.export_template = "{title}_{tag}_{frame}"
    inker_mode.persist(ctx)

    reopened = _Ctx(None, settings_block=ctx.settings.get("inker"))
    restored = inker_mode.ensure(reopened)

    assert restored.export_scale == 4
    assert restored.export_arrange == "horizontal"
    assert restored.export_merge is True
    assert restored.export_template == "{title}_{tag}_{frame}"


def test_a_garbage_stored_export_block_is_ignored_not_crashed():
    reopened = _Ctx(
        None,
        settings_block={"export": {"scale": "banana", "arrange": 123, "bogus": "x"}},
    )
    state = inker_mode.ensure(reopened)
    assert state.export_scale == inker_state.EXPORT_OPTION_DEFAULTS["scale"]
    assert state.export_arrange is None


def test_a_stored_block_missing_a_newer_key_leaves_it_at_the_default():
    """A settings file written before ``template`` existed must not crash and
    must not leave the new control at whatever it happened to be."""
    reopened = _Ctx(None, settings_block={"export": {"scale": 2}})
    state = inker_mode.ensure(reopened)
    assert state.export_scale == 2
    assert state.export_template == ""


# --- journal exemption ------------------------------------------------------------


def test_export_dest_and_options_are_invisible_to_the_journals_head():
    """``journal._write_if_due``'s whole decision is whether ``head_of(slot)``
    changed; if it cannot see these two fields, nothing downstream of it can
    either."""
    _, _, tab = _open()
    before = inker_mode.JOURNAL.head_of(tab)

    tab.export_dest = Path("somewhere") / "walk.png"
    tab.export_options = {"scale": 4, "merge": True, "template": "{title}"}

    after = inker_mode.JOURNAL.head_of(tab)
    assert before == after


def test_export_dest_and_options_do_not_touch_saved_head_or_dirty():
    _, _, tab = _open()
    tab.saved_head = tab.doc.history.head
    assert not tab.dirty
    tab.export_dest = Path("x.png")
    tab.export_options = {"scale": 2}
    assert not tab.dirty


# --- 6.9: the packed sheet type and the JSON meta switches -------------------


def test_the_packed_arrange_is_the_squarest_grid():
    """What "packed" means for frames that are all one size -- the bin-packing
    the name suggests would have nothing to solve here."""
    from warlock.studio.inker import sheetout

    for count, columns in ((4, 2), (9, 3), (10, 4), (1, 1)):
        plan = sheetout.plan_frames(count, 16, 16, arrange="packed")
        assert plan.columns == columns, count


def test_packed_is_one_of_the_arranges_the_engine_names():
    from warlock.studio.inker import sheetout

    assert "packed" in sheetout.ARRANGES
    assert "packed" not in sheetout.COUNTED_ARRANGES, "it takes no wrap count"


def test_the_meta_switches_are_part_of_the_recorded_option_set():
    """Or a repeat would write a different sidecar from the export it repeats."""
    state = InkerState()
    snapshot = state.export_options_snapshot()
    for key in ("meta_tags", "meta_slices", "meta_layers"):
        assert key in snapshot
        assert key in inker_state.EXPORT_OPTION_DEFAULTS


def test_turning_the_tag_meta_off_drops_the_tags_from_the_sidecar(monkeypatch, tmp_path):
    import json

    ctx, state, tab = _open(_tagged())
    state.export_meta_tags = False
    _saved(monkeypatch, tmp_path / "walk.png")
    inker_mode.export_sheet(ctx, tab)
    _finish(ctx, state)
    meta = json.loads((tmp_path / "walk.json").read_text(encoding="utf-8"))
    assert not meta["animation"].get("tags")


def test_the_tags_are_written_by_default(monkeypatch, tmp_path):
    import json

    ctx, state, tab = _open(_tagged())
    _saved(monkeypatch, tmp_path / "walk.png")
    inker_mode.export_sheet(ctx, tab)
    _finish(ctx, state)
    meta = json.loads((tmp_path / "walk.json").read_text(encoding="utf-8"))
    assert [tag["name"] for tag in meta["animation"]["tags"]] == ["intro", "walk"]


def test_a_refused_export_does_not_settle_the_document_first():
    """``_settle`` commits the floating buffer and cancels the filter and
    conversion previews. Running it above the span and stem checks meant an
    export that went on to say "cannot export" had already folded a paste into
    the layers on its way to refusing."""
    from types import SimpleNamespace

    import numpy as np

    from warlock.studio import inker, inker_mode, inker_state

    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    tab = inker_state.InkerDoc(doc=doc, title="a.png", saved_head=doc.history.head)
    state = inker_state.InkerState()
    state.add(tab)
    said: list[str] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state),
        toast=lambda message, kind="info": said.append(message),
    )

    # Something floating, which is exactly what a settle would fold away.
    doc.float_pixels(np.full((4, 4, 4), 255, np.uint8), (0, 0))
    assert doc.floating is not None

    # A span that holds no frames: the first of the two refusals.
    inker_mode._begin_export(ctx, tab, "png", span=(99, 99))

    assert said, "it refused"
    assert doc.floating is not None, "and left the document exactly as it found it"
    assert state.export is None
    assert not tab.saving
