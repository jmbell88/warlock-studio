"""Exporting slices as PNGs (C13d).

The engine call is ``inker_mode.export_slices``: per slice, resolve
``at(current_frame_uid)``, crop the flatten, and write one PNG named after the
slice. Three things can go wrong that a round-trip through the encoder would
not obviously show -- a name collision silently overwriting a file, a keyed
slice exporting the wrong frame's rectangle, and a scale that is read after
the dialog rather than before it -- so those are what is pinned here, plus the
ordinary crop-content check.

Mirrors ``test_inker_export_steps.py``'s ``_Ctx``: the three things the export
path asks of a ``Ctx``, with ``submit`` accepting immediately and stashing the
runner so the test can call it directly, off the task thread it would really
run on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from warlock.studio import inker, inker_mode
from warlock.studio.inker.slices import SliceKey
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


def _open(doc=None):
    tab = InkerDoc(doc=doc or inker.Document.blank(16, 16), title="sprite.png")
    state = InkerState()
    state.add(tab)
    return _Ctx(state), state, tab


def _paint(doc, rect, colour) -> None:
    x0, y0, x1, y1 = rect
    weight = np.ones((y1 - y0, x1 - x0), dtype=np.float32)
    doc.write_colour(rect, colour, weight)


def _saved(monkeypatch, dest) -> None:
    """Point ``dialogs.save_file`` at a fixed destination for the run closure."""
    from warlock.studio import dialogs

    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: dest)


def test_each_slice_becomes_its_own_png_cropped_to_its_bounds(monkeypatch, tmp_path):
    doc = inker.Document.blank(16, 16)
    _paint(doc, (0, 0, 8, 8), RED)
    _paint(doc, (8, 8, 16, 16), BLUE)
    doc.add_slice((0, 0, 8, 8), name="red")
    doc.add_slice((8, 8, 16, 16), name="blue")
    ctx, _state, tab = _open(doc)
    _saved(monkeypatch, tmp_path / "sprite.png")

    inker_mode.export_slices(ctx, tab)
    assert ctx.submitted == [f"inker-export:{tab.uid}"]
    result = ctx.run()

    red_png = np.asarray(Image.open(tmp_path / "red.png").convert("RGBA"))
    blue_png = np.asarray(Image.open(tmp_path / "blue.png").convert("RGBA"))
    assert red_png.shape == (8, 8, 4)
    assert tuple(red_png[0, 0]) == RED
    assert tuple(blue_png[0, 0]) == BLUE
    # ``dest`` and ``export_kind`` beside the first file written: Repeat Last
    # Export has to know where this went and which runner to run again.
    assert result == {
        "exported": tmp_path / "red.png",
        "dest": tmp_path / "sprite.png",
        "export_kind": "slices",
    }


def test_duplicate_slice_names_get_a_numeric_suffix_rather_than_overwrite(
    monkeypatch, tmp_path
):
    doc = inker.Document.blank(16, 16)
    doc.add_slice((0, 0, 4, 4), name="Hitbox")
    doc.add_slice((4, 4, 8, 8), name="Hitbox")
    doc.add_slice((8, 8, 12, 12), name="Hitbox")
    ctx, _state, tab = _open(doc)
    _saved(monkeypatch, tmp_path / "sprite.png")

    inker_mode.export_slices(ctx, tab)
    ctx.run()

    assert (tmp_path / "Hitbox.png").exists()
    assert (tmp_path / "Hitbox_2.png").exists()
    assert (tmp_path / "Hitbox_3.png").exists()


def test_a_bumped_name_does_not_collide_with_a_later_slices_own_bump():
    """Reviewer repro: a per-base counter bumps every "Hitbox" independently of
    what the *previous* one landed on, so the second slice claims "Hitbox_2"
    and the third -- also counting from its own base -- claims "Hitbox_2"
    again. Each candidate must be checked against every name already handed
    out, not just against occurrences of its own base."""
    entries = [SimpleNamespace(name=n) for n in ("Hitbox", "Hitbox_2", "Hitbox")]
    names = inker_mode._slice_filenames(entries)
    assert len(set(names)) == len(names)
    assert names == ["Hitbox", "Hitbox_2", "Hitbox_3"]


def test_a_literal_bumped_name_does_not_collide_with_a_repeats_bump():
    """Reviewer repro: two slices literally named "a" bump the second to
    "a_2" -- which collides with a third slice *already* named "a_2" outright,
    an ordering the per-base counter never saw coming because "a" and "a_2"
    are different bases to it."""
    entries = [SimpleNamespace(name=n) for n in ("a", "a", "a_2")]
    names = inker_mode._slice_filenames(entries)
    assert len(set(names)) == len(names)
    assert names == ["a", "a_2", "a_2_2"]


def test_a_name_with_unsafe_characters_is_sanitised(monkeypatch, tmp_path):
    doc = inker.Document.blank(16, 16)
    doc.add_slice((0, 0, 4, 4), name="A/B*C? weapon")
    ctx, _state, tab = _open(doc)
    _saved(monkeypatch, tmp_path / "sprite.png")

    inker_mode.export_slices(ctx, tab)
    ctx.run()

    written = list(tmp_path.glob("*.png"))
    assert len(written) == 1
    assert written[0].name == "A-B-C-weapon.png"


def test_scale_upsamples_the_cropped_pixels_not_the_whole_canvas(monkeypatch, tmp_path):
    doc = inker.Document.blank(16, 16)
    _paint(doc, (0, 0, 4, 4), RED)
    doc.add_slice((0, 0, 4, 4), name="icon")
    ctx, _state, tab = _open(doc)
    ctx.state.inker.export_scale = 3
    _saved(monkeypatch, tmp_path / "sprite.png")

    inker_mode.export_slices(ctx, tab)
    ctx.run()

    png = Image.open(tmp_path / "icon.png")
    assert png.size == (12, 12)


def test_a_keyed_slice_exports_the_rectangle_of_the_current_frame(monkeypatch, tmp_path):
    """The panel beside the slice list resolves ``entry.at(tab.frame_uid)`` to
    describe a slice on the frame the playhead sits on; the export must
    resolve the same key rather than always reading the base rectangle."""
    doc = inker.Document.blank(16, 16)
    _paint(doc, (0, 0, 4, 4), RED)
    _paint(doc, (8, 8, 12, 12), BLUE)
    doc.add_frame(link=True)
    entry = doc.add_slice((0, 0, 4, 4), name="body")
    frame1_uid = doc.anim.frames[1].uid
    doc.set_slice_key(entry.uid, frame1_uid, key=SliceKey((8, 8, 12, 12)))

    ctx, _state, tab = _open(doc)
    tab.doc.set_current_frame(1)
    assert tab.frame_uid == frame1_uid
    _saved(monkeypatch, tmp_path / "sprite.png")

    inker_mode.export_slices(ctx, tab)
    ctx.run()

    png = np.asarray(Image.open(tmp_path / "body.png").convert("RGBA"))
    assert tuple(png[0, 0]) == BLUE


def test_a_document_with_no_slices_exports_nothing_and_submits_nothing():
    ctx, _state, tab = _open(inker.Document.blank(16, 16))
    inker_mode.export_slices(ctx, tab)
    assert ctx.submitted == []


def test_a_cancelled_dialog_writes_no_files(monkeypatch, tmp_path):
    doc = inker.Document.blank(16, 16)
    doc.add_slice((0, 0, 4, 4), name="icon")
    ctx, _state, tab = _open(doc)
    _saved(monkeypatch, None)

    inker_mode.export_slices(ctx, tab)
    result = ctx.run()

    assert result is None
    assert list(tmp_path.glob("*.png")) == []


def test_a_busy_tab_refuses_the_export():
    ctx, _state, tab = _open(inker.Document.blank(16, 16))
    tab.doc.add_slice((0, 0, 4, 4), name="icon")
    tab.saving = True
    inker_mode.export_slices(ctx, tab)
    assert ctx.submitted == []


# --- the two collision policies, pinned against each other ---------------------


def test_the_two_collision_policies_diverge_deliberately_and_by_the_same_question():
    """``inker_mode`` holds two naming helpers with opposite answers to the same
    situation, and Wave 4's "Left open / owed" section named the pair as
    unreconciled. They are not reconciled here either -- they are *pinned*,
    because each is right for its own door and collapsing them onto one policy
    would break whichever door lost.

    The deciding question is whether anything downstream addresses the file
    **by name**:

    * Nothing addresses a slice PNG by name. A human picks it off a folder
      listing, so ``Hitbox_2.png`` is a name they can read and live with, and
      refusing the whole export over it would be obstructive.
    * A tag or a layer *is* addressed by name by whatever consumes the sheet.
      A second ``walk`` quietly becoming ``walk_2.png`` is a file claiming to
      be a clip that does not exist, and refusing is the only answer that
      cannot silently be believed.

    Bumping is friendly where a human disambiguates and dishonest where a
    machine does. A third naming helper answers that question before it picks a
    side; this test is what makes either half drifting onto the other's policy
    a red test rather than a silent behaviour change."""
    import pytest

    entries = [SimpleNamespace(name="Hitbox"), SimpleNamespace(name="Hitbox")]
    # Slices: the second one is bumped, and both files are written.
    assert inker_mode._slice_filenames(entries) == ["Hitbox", "Hitbox_2"]

    # Tags and layers: the same duplicate is refused outright, by name.
    for kind in ("tag", "layer"):
        with pytest.raises(ValueError, match="would both be called"):
            inker_mode._split_stems("sheet", ["Hitbox", "Hitbox"], kind=kind)


def test_the_split_refusal_survives_a_template_that_erases_the_difference():
    """The refusal is about the *rendered* names, not the labels -- two distinct
    tags put through a template that mentions neither collide just as hard, and
    a check that compared labels instead would let that batch overwrite itself
    one file at a time."""
    import pytest

    with pytest.raises(ValueError, match="would both be called"):
        inker_mode._split_stems("sheet", ["walk", "run"], kind="tag", template="{title}")
