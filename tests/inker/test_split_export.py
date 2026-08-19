"""Splitting one export into several files: one per tag, one per layer.

Two verbs, one machine. A split is still *one* export -- one lock taken at the
click, one stepper, one flatten per pump, one task at the end -- because the
alternative is N exports racing each other for the same task key with the tab
locked N times over. What the split changes is how many files come out of the
one task, and what each of them describes.

So what is pinned here is the part that can silently go wrong: that a per-tag
file is exactly the file exporting that tag on its own would have written, that
a per-layer file holds that layer's own composite and not the whole frame's,
that the stepper contract survives a batch, and that two outputs can never
quietly become one file.

``_Ctx`` mirrors ``test_inker_export_steps.py``'s: the three things the export
path asks of a Ctx, with ``submit`` stashing the runner so the test can call it
directly, off the task thread it would really run on.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from PIL import Image

from warlock.studio import inker, inker_mode
from warlock.studio.inker import sheetout
from warlock.studio.inker_state import InkerDoc, InkerState

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


class _Ctx:
    def __init__(self, state: Any) -> None:
        self.state = _AppState(state)
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


class _AppState:
    def __init__(self, inker_state: Any) -> None:
        self.inker = inker_state


def _clip(frames: int = 4) -> Any:
    doc = inker.Document.blank(4, 4)
    ones = np.ones((2, 2), dtype=np.float32)
    doc.write_colour((0, 0, 2, 2), RED, ones)
    doc.add_layer("ink")
    doc.write_colour((2, 2, 4, 4), BLUE, ones)
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    return doc


def _open(doc=None):
    tab = InkerDoc(doc=doc if doc is not None else _clip(), title="walk.ora")
    tab.path = None
    state = InkerState()
    state.add(tab)
    return _Ctx(state), state, tab


def _tagged(frames: int = 4) -> Any:
    doc = _clip(frames)
    assert doc.add_tag("intro", 0, 1)
    assert doc.add_tag("walk", 2, 3)
    return doc


def _saved(monkeypatch, dest) -> None:
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: dest)


def _finish(ctx: Any, state: Any, limit: int = 200) -> Any:
    """Pump the stepper to completion, then run the task the submit stashed."""
    for _ in range(limit):
        inker_mode.pump_export(ctx)
        if state.export is None:
            return ctx.run()
    raise AssertionError("the export never finished")


# --- one file per tag ---------------------------------------------------------


def test_a_split_by_tag_writes_one_sheet_per_tag(monkeypatch, tmp_path):
    ctx, state, tab = _open(_tagged())
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    result = _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "walk_intro.png",
        "walk_walk.png",
    ]
    assert (tmp_path / "walk_intro.json").exists()
    assert (tmp_path / "walk_walk.json").exists()
    # No file under the name the dialog itself was given: every output of a
    # split is named, so an unlabelled one would be a fourth thing on disk
    # nobody asked for.
    assert not (tmp_path / "walk.png").exists()
    assert result["exported"] == tmp_path / "walk_intro.png"
    # The recorded destination is the dialog's own pick, not the split's --
    # so the tab's next export suggests the folder the *dialog* named.
    assert result["dest"] == tmp_path / "walk.png"


def test_each_per_tag_sheet_holds_only_that_tags_frames(monkeypatch, tmp_path):
    doc = inker.Document.blank(4, 4)
    ones = np.ones((2, 2), dtype=np.float32)
    for index in range(4):
        if index:
            doc.add_frame()
        doc.write_colour((0, 0, 2, 2), (10 * index, 0, 0, 255), ones)
    assert doc.add_tag("intro", 0, 1)
    assert doc.add_tag("walk", 2, 3)
    ctx, state, tab = _open(doc)
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    _finish(ctx, state)

    intro = np.asarray(Image.open(tmp_path / "walk_intro.png").convert("RGBA"))
    walk = np.asarray(Image.open(tmp_path / "walk_walk.png").convert("RGBA"))
    assert intro.shape == walk.shape == (4, 8, 4)  # two cells each
    assert tuple(intro[0, 0]) == (0, 0, 0, 255)
    assert tuple(intro[0, 4]) == (10, 0, 0, 255)
    assert tuple(walk[0, 0]) == (20, 0, 0, 255)
    assert tuple(walk[0, 4]) == (30, 0, 0, 255)


def test_a_per_tag_sheet_is_what_exporting_that_tag_alone_writes(monkeypatch, tmp_path):
    """The pin that keeps the two paths one feature: a batch reuses
    ``export_tag``'s span and its looping, so the file it writes has to be the
    file the single-tag verb writes -- pixels byte for byte, and a sidecar that
    differs only in the identity of the file it is beside."""
    ctx, state, tab = _open(_tagged())
    _saved(monkeypatch, tmp_path / "alone.png")
    inker_mode.export_tag(ctx, tab, "sheet", 1)
    _finish(ctx, state)

    ctx, state, tab = _open(_tagged())
    _saved(monkeypatch, tmp_path / "walk.png")
    inker_mode.export_per_tag(ctx, tab, "sheet")
    _finish(ctx, state)

    assert (tmp_path / "walk_walk.png").read_bytes() == (
        tmp_path / "alone.png"
    ).read_bytes()
    batched = json.loads((tmp_path / "walk_walk.json").read_text(encoding="utf-8"))
    alone = json.loads((tmp_path / "alone.json").read_text(encoding="utf-8"))
    for key in ("id", "image", "created"):
        batched.pop(key), alone.pop(key)
    assert batched == alone
    # ``name`` is the document's, not the tag's: it is the pose name every cell
    # carries, so a batch that renamed it per file would write cells the
    # single-tag export does not.
    assert batched["name"] == alone["name"] == "untitled"


def test_each_per_tag_sidecar_renumbers_its_own_tags(monkeypatch, tmp_path):
    ctx, state, tab = _open(_tagged())
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    _finish(ctx, state)

    meta = json.loads((tmp_path / "walk_walk.json").read_text(encoding="utf-8"))
    assert [(t["name"], t["start"], t["end"]) for t in meta["animation"]["tags"]] == [
        ("walk", 0, 1)
    ]
    assert len(meta["animation"]["frames"]) == 2


def test_a_document_with_no_tags_is_refused_before_anything_is_locked():
    ctx, state, tab = _open(_clip())
    inker_mode.export_per_tag(ctx, tab, "sheet")
    assert state.export is None
    assert not tab.saving
    assert ctx.submitted == []
    assert ctx.toasts and ctx.toasts[0][1] == "warn"


def test_two_tags_that_would_share_a_filename_are_refused_by_name(tmp_path):
    """A bumped "walk_2.png" is a file claiming to be a tag called walk_2, and
    an engine looking a clip up by name would take it. Slices bump; tags, which
    an engine addresses by name, refuse."""
    doc = _clip()
    assert doc.add_tag("walk", 0, 1)
    assert doc.add_tag("walk", 2, 3)
    ctx, state, tab = _open(doc)

    inker_mode.export_per_tag(ctx, tab, "sheet")

    assert state.export is None
    assert not tab.saving
    assert ctx.submitted == []
    assert any("walk" in message for message, _kind in ctx.toasts)


def test_a_tag_name_a_filename_cannot_hold_is_sanitised(monkeypatch, tmp_path):
    doc = _clip()
    assert doc.add_tag("A/B*C? swing", 0, 3)
    ctx, state, tab = _open(doc)
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    _finish(ctx, state)

    assert [p.name for p in tmp_path.glob("*.png")] == ["walk_A-B-C-swing.png"]


# --- one file per layer -------------------------------------------------------


def test_a_split_by_layer_writes_one_sheet_per_top_level_row(monkeypatch, tmp_path):
    ctx, state, tab = _open()
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_layer(ctx, tab, "sheet")
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "walk_Background.png",
        "walk_ink.png",
    ]


def test_a_per_layer_sheet_holds_that_layers_own_composite(monkeypatch, tmp_path):
    """Not the whole frame with a name on it: the ink sheet must be missing the
    background's pixels, cell for cell."""
    ctx, state, tab = _open()
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_layer(ctx, tab, "sheet")
    _finish(ctx, state)

    ink = np.asarray(Image.open(tmp_path / "walk_ink.png").convert("RGBA"))
    back = np.asarray(Image.open(tmp_path / "walk_Background.png").convert("RGBA"))
    expected = sheetout.flatten_subset(
        tab.doc, tab.doc.anim.frames[0].uid, {tab.doc.anim.tracks[1].uid}
    )
    assert np.array_equal(ink[:, :4], expected)
    assert tuple(ink[0, 0]) == (0, 0, 0, 0)
    assert tuple(ink[3, 3]) == BLUE
    assert tuple(back[0, 0]) == RED
    assert tuple(back[3, 3]) == (0, 0, 0, 0)


def test_a_per_layer_split_leaves_the_flatten_cache_holding_whole_frames(
    monkeypatch, tmp_path
):
    """The cache is keyed on the frame uid alone. A split that poisoned it
    would hand the next onion-skin draw a frame with layers missing."""
    ctx, state, tab = _open()
    uid = tab.doc.anim.frames[0].uid
    whole = tab.doc.frame_flat(uid).copy()
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_layer(ctx, tab, "sheet")
    _finish(ctx, state)

    assert np.array_equal(tab.doc.frame_flat(uid), whole)
    assert tuple(whole[0, 0]) == RED and tuple(whole[3, 3]) == BLUE


def test_a_hidden_layer_is_not_one_of_the_outputs(monkeypatch, tmp_path):
    ctx, state, tab = _open()
    tab.doc.set_layer_props(1, visible=False)
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_layer(ctx, tab, "sheet")
    _finish(ctx, state)

    assert [p.name for p in tmp_path.glob("*.png")] == ["walk_Background.png"]


def test_a_group_exports_as_one_sheet_of_everything_inside_it(monkeypatch, tmp_path):
    doc = _clip()
    doc.add_layer("glow")
    doc.set_current_frame(0)
    doc.write_colour((0, 2, 2, 4), BLUE, np.ones((2, 2), dtype=np.float32))
    node = doc.group_layers([1, 2], name="fx")
    assert node is not None
    ctx, state, tab = _open(doc)
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_layer(ctx, tab, "sheet")
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "walk_Background.png",
        "walk_fx.png",
    ]
    fx = np.asarray(Image.open(tmp_path / "walk_fx.png").convert("RGBA"))
    assert tuple(fx[3, 3]) == BLUE  # the ink track
    assert tuple(fx[3, 0]) == BLUE  # and the glow track, in one file


def test_a_still_document_is_refused_before_anything_is_locked():
    tab = InkerDoc(doc=inker.Document.blank(4, 4), title="still.png")
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    inker_mode.export_per_layer(ctx, tab, "sheet")
    assert state.export is None
    assert not tab.saving


# --- the stepper contract, across a batch -------------------------------------


def test_one_frame_is_flattened_per_pump_across_the_whole_batch():
    """The contract that made the stepper worth having, and the one a batch is
    most likely to break: four frames of tag one and four of tag two is eight
    pumps, not two clips flattened in two frames."""
    ctx, state, tab = _open(_tagged(6))
    tab.doc.set_tag(0, start=0, end=2)
    tab.doc.set_tag(1, start=3, end=5)

    inker_mode.export_per_tag(ctx, tab, "sheet")
    assert state.export is not None
    read = 0
    for _ in range(20):
        inker_mode.pump_export(ctx)
        if state.export is None:
            break
        read += 1
        assert state.export.read == read
    assert ctx.submitted == [f"inker-export:{tab.uid}"]
    assert read == 5  # the sixth pump reads the last frame and submits


def test_the_tab_is_locked_once_for_the_whole_batch():
    ctx, state, tab = _open(_tagged())
    inker_mode.export_per_tag(ctx, tab, "sheet")
    assert tab.saving and tab.busy
    for _ in range(10):
        inker_mode.pump_export(ctx)
        if state.export is None:
            break
    assert tab.saving  # still locked: the encode is in flight


def test_a_second_click_during_a_batch_is_refused():
    ctx, state, tab = _open(_tagged())
    inker_mode.export_per_tag(ctx, tab, "sheet")
    first = state.export
    inker_mode.export_per_layer(ctx, tab, "sheet")
    inker_mode.export_sheet(ctx, tab)
    assert state.export is first


def test_closing_the_tab_mid_batch_abandons_it_quietly(tmp_path):
    ctx, state, tab = _open(_tagged())
    inker_mode.export_per_tag(ctx, tab, "sheet")
    inker_mode.pump_export(ctx)
    inker_mode.pump_export(ctx)
    state.docs.remove(tab)
    inker_mode.pump_export(ctx)

    assert state.export is None
    assert ctx.submitted == []
    assert ctx.toasts == []
    assert list(tmp_path.glob("*")) == []


def test_a_cancelled_dialog_writes_no_files_at_all(monkeypatch, tmp_path):
    ctx, state, tab = _open(_tagged())
    _saved(monkeypatch, None)

    inker_mode.export_per_tag(ctx, tab, "sheet")
    assert _finish(ctx, state) is None
    assert list(tmp_path.glob("*")) == []


# --- a batch is all-or-nothing ------------------------------------------------
#
# The runner composes every leg before it writes any of them, so a refusal that
# only the Nth leg can reach -- ``skip_empty`` over a tag with nothing drawn in
# it -- leaves the folder as it found it rather than half a batch on disk under
# names a user will believe.


def _half_empty() -> Any:
    """Four frames: 0 and 1 drawn, 2 and 3 blank, one tag over each pair."""
    doc = _clip(2)
    doc.add_frame()
    doc.add_frame()
    assert doc.add_tag("intro", 0, 1)
    assert doc.add_tag("walk", 2, 3)
    return doc


def test_a_leg_with_only_empty_frames_refuses_by_name_and_writes_nothing(
    monkeypatch, tmp_path
):
    ctx, state, tab = _open(_half_empty())
    state.export_skip_empty = True
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    with pytest.raises(ValueError, match="walk"):
        _finish(ctx, state)
    # Not one file, not even the healthy first leg's: the batch is atomic.
    assert list(tmp_path.glob("*")) == []


def test_the_leg_refusal_says_which_kind_of_split_and_what_went_wrong(
    monkeypatch, tmp_path
):
    ctx, state, tab = _open(_half_empty())
    state.export_skip_empty = True
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    with pytest.raises(ValueError) as caught:
        _finish(ctx, state)
    assert "tag" in str(caught.value)
    assert "every frame is empty" in str(caught.value)


def test_a_healthy_split_with_skip_empty_still_writes_every_leg(
    monkeypatch, tmp_path
):
    ctx, state, tab = _open(_tagged())
    state.export_skip_empty = True
    _saved(monkeypatch, tmp_path / "walk.png")

    inker_mode.export_per_tag(ctx, tab, "sheet")
    _finish(ctx, state)

    assert sorted(p.name for p in tmp_path.glob("*")) == [
        "walk_intro.json",
        "walk_intro.png",
        "walk_walk.json",
        "walk_walk.png",
    ]


# --- the naming, which Task 5's templates will replace ------------------------


def test_an_unsplit_export_keeps_the_name_the_dialog_was_given():
    assert inker_mode._split_stems("walk", [""]) == ["walk"]


def test_a_split_stem_is_the_name_plus_the_label():
    assert inker_mode._split_stems("walk", ["intro", "run"]) == [
        "walk_intro",
        "walk_run",
    ]


def test_a_label_that_sanitises_to_nothing_is_refused():
    with pytest.raises(ValueError):
        inker_mode._split_stems("walk", ["///"])


def test_two_labels_that_collide_after_sanitising_are_refused():
    with pytest.raises(ValueError, match="walk_a-b"):
        inker_mode._split_stems("walk", ["a/b", "a*b"])


def test_an_empty_tag_name_falls_back_to_a_placeholder_not_the_bare_stem():
    """A loaded .ase/ORA can carry a tag or a track with an empty name. An
    empty *label* only means "unsplit" when ``kind`` is also empty -- a split
    leg's empty label is a real (if badly named) tag or layer, and must not
    collapse onto the bare stem, which is what the single-file export writes
    and would make this split output masquerade as a whole-document export."""
    assert inker_mode._split_stems("walk", [""], kind="tag") == ["walk_tag"]


def test_an_empty_layer_name_falls_back_to_a_placeholder_not_the_bare_stem():
    assert inker_mode._split_stems("walk", [""], kind="layer") == ["walk_layer"]


def test_two_empty_named_tags_still_collide_and_refuse():
    with pytest.raises(ValueError, match="walk_tag"):
        inker_mode._split_stems("walk", ["", ""], kind="tag")
