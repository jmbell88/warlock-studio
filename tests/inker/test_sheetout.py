"""An Inker animation exported as a sprite sheet.

The grid arithmetic is the interesting half and it is pure, so all of it is
asserted from plain arrays with no ``Document`` and no GPU. The one thing worth
saying out loud about the format: a linked cel appearing in three frames is
three identical cells here, because a sheet is played back by an engine that
knows nothing about links.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from warlock.pipelines import sheet as sheetlib
from warlock.studio.inker import sheetout
from warlock.studio.inker.animation import Tag
from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _frame(width: int, height: int, mark: tuple[int, int, int, int], at=(0, 0)) -> np.ndarray:
    plane = np.zeros((height, width, 4), dtype=np.uint8)
    plane[at[1], at[0]] = mark
    return plane


# --- the grid ----------------------------------------------------------------


def test_one_cell_per_frame_in_row_major_order():
    plan = sheetout.plan_frames(3, 40, 20)

    assert plan.columns == 3 and plan.rows == 1
    assert [(c.index, c.x, c.y, c.frame, c.yaw) for c in plan.cells] == [
        (0, 0, 0, 0, 0.0),
        (1, 40, 0, 1, 0.0),
        (2, 80, 0, 2, 0.0),
    ]
    assert (plan.width, plan.height) == (120, 20)


def test_a_non_square_plan_reports_its_real_cell_size():
    plan = sheetout.plan_frames(2, 40, 20)
    assert (plan.cell_w, plan.cell_h) == (40, 20)
    # Zero, not the width: a square-only importer must fail loudly rather than
    # slice the atlas correctly across and wrongly down.
    assert plan.frame_size == 0


def test_a_row_wraps_before_the_atlas_ceiling():
    """32 frames of a 320px canvas is 10240px across, past what an engine will
    accept as a texture at all -- so wrapping is required, not a nicety."""
    plan = sheetout.plan_frames(32, 320, 100)

    assert plan.columns == sheetlib.MAX_ATLAS_PX // 320
    assert plan.rows == -(-32 // plan.columns)
    assert max(plan.width, plan.height) <= sheetlib.MAX_ATLAS_PX
    assert [c.index for c in plan.cells] == list(range(32))


def test_a_wrapped_grid_positions_its_second_row():
    plan = sheetout.plan_frames(3, 4000, 10)
    assert plan.columns == 2
    assert [(c.row, c.column, c.x, c.y) for c in plan.cells] == [
        (0, 0, 0, 0),
        (0, 1, 4000, 0),
        (1, 0, 0, 10),
    ]


def test_a_frame_too_big_for_any_atlas_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        sheetout.plan_frames(1, sheetlib.MAX_ATLAS_PX + 1, 10)


def test_too_many_rows_is_refused_by_the_shared_guard():
    with pytest.raises(ValueError, match="8192"):
        sheetout.plan_frames(2000, sheetlib.MAX_ATLAS_PX, 100)


def test_an_empty_clip_is_refused():
    with pytest.raises(ValueError):
        sheetout.plan_frames(0, 10, 10)


# --- compositing -------------------------------------------------------------


def test_frame_i_lands_in_cell_i():
    frames = [_frame(4, 4, RED, (0, 0)), _frame(4, 4, BLUE, (1, 1))]
    image, plan, _trims = sheetout.build(frames, [100, 100])
    try:
        assert image.size == (plan.width, plan.height)
        assert image.getpixel((0, 0)) == RED
        assert image.getpixel((5, 1)) == BLUE
    finally:
        image.close()


def test_every_frame_must_be_the_same_size():
    with pytest.raises(ValueError, match="same size"):
        sheetout.build([_frame(4, 4, RED), _frame(8, 8, RED)], [10, 10])


def test_every_frame_needs_a_duration():
    with pytest.raises(ValueError, match="duration"):
        sheetout.build([_frame(4, 4, RED)], [])


def test_trims_are_measured_per_frame():
    frames = [_frame(4, 4, RED, (2, 3)), np.zeros((4, 4, 4), dtype=np.uint8)]
    _image, _plan, trims = sheetout.build(frames, [10, 10])
    assert trims[0] == {"x": 2, "y": 3, "w": 1, "h": 1}
    assert trims[1] is None


# --- the animation block -----------------------------------------------------


def test_durations_and_tags_round_trip_in_timeline_order():
    plan = sheetout.plan_frames(3, 4, 4)
    block = sheetout.animation_block(
        plan, [40, 120, 40], [Tag(name="walk", start=0, end=2, loop=False)]
    )

    assert block["frames"] == [
        {"cell_index": 0, "duration_ms": 40},
        {"cell_index": 1, "duration_ms": 120},
        {"cell_index": 2, "duration_ms": 40},
    ]
    assert block["tags"] == [{"name": "walk", "start": 0, "end": 2, "loop": False}]


def test_every_cell_index_in_the_block_names_a_real_cell():
    plan = sheetout.plan_frames(5, 4000, 10)
    block = sheetout.animation_block(plan, [10] * 5, [])
    known = {c.index for c in plan.cells}
    assert {entry["cell_index"] for entry in block["frames"]} == known


# --- from a document ---------------------------------------------------------


def _animated() -> Document:
    doc = Document.blank(4, 4)
    weight = np.ones((2, 2), dtype=np.float32)
    doc.write_colour((0, 0, 2, 2), RED, weight)
    doc.add_frame(link=True)
    doc.add_frame(link=True)
    return doc


def test_a_still_document_is_refused():
    """``Export PNG`` already covers that case, and a one-frame "sprite sheet"
    is the kind of output a user has to go and check."""
    with pytest.raises(ValueError, match="not animated"):
        sheetout.from_document(Document.blank(4, 4))


def test_a_linked_cel_becomes_one_identical_cell_per_frame():
    """Right, not a bug: an engine playing this back knows nothing about links,
    so the frames have to be there."""
    image, plan, extra = sheetout.from_document(_animated())
    try:
        assert len(plan.cells) == 3
        tiles = [
            image.crop((c.x, c.y, c.x + plan.cell_w, c.y + plan.cell_h)).tobytes()
            for c in plan.cells
        ]
        assert tiles[0] == tiles[1] == tiles[2]
        assert image.getpixel((0, 0)) == RED
    finally:
        image.close()
    assert len(extra["animation"]["frames"]) == 3


def test_the_sidecar_carries_the_animation_and_the_real_frame_size():
    image, plan, extra = sheetout.from_document(_animated(), name="walk")
    image.close()
    meta = sheetlib.sidecar(
        plan,
        sheet_id="s",
        source_job="",
        image="s.png",
        created=0.0,
        trims=extra["trims"],
        animation=extra["animation"],
    )

    assert meta["frame_size"] == 0
    assert (meta["frame_w"], meta["frame_h"]) == (4, 4)
    assert meta["animation"]["frames"][0]["duration_ms"] > 0
    assert all(cell["w"] == 4 and cell["h"] == 4 for cell in meta["cells"])


# --- the purity rule this module is the first exception to -------------------


ENGINE = Path(__file__).resolve().parents[2] / "src/warlock/studio/inker"
PACKAGE = "warlock.studio.inker"

#: Everything under ``studio/inker/`` that reaches outside the package, resolved
#: to absolute module names. Pinned as an exact set rather than as a predicate,
#: so a *new* outward import is a failing test and a deliberate decision rather
#: than something that turns up in a review three months later.
OUTWARD_IMPORTS = {
    ("anim_edits.py", "warlock.studio.undo"),
    ("composite.py", "warlock.native"),
    ("selection.py", "warlock.native"),
    ("sheetout.py", "warlock.pipelines"),
    ("undo.py", "warlock.studio.undo"),
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}


def _outward(path: Path) -> set[str]:
    """Absolute module names this file imports from outside its own package."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            elif node.level >= 2:
                # Level 1 is a sibling inside the package. Level 2+ climbs out
                # of it, which is exactly what this is measuring.
                base = PACKAGE.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    # ``from ... import native``: with no module part the names
                    # are themselves modules, so they are the thing being
                    # reached for and the package alone would say nothing.
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def test_the_engine_never_imports_a_window():
    """``studio/inker/`` stays headless, which is what makes every rule it has
    about pixels assertable in a test like this one. The rule has been true
    since the package was written and was held by a docstring, which is not a
    test."""
    for path in sorted(ENGINE.glob("*.py")):
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in sorted(ENGINE.glob("*.py")):
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    """``sheetout`` is the newest of these and the one with an argument
    attached: it reaches for the *authority* on the sprite-sheet format, so
    ``version: 1`` cannot come to mean two subtly different documents. The
    others predate it -- the shared undo engine and the native kernel loader,
    both of which are as headless as this package is."""
    found = {
        (path.name, name)
        for path in sorted(ENGINE.glob("*.py"))
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_sheetout_reaches_for_the_format_and_nothing_else_in_pipelines():
    """``pipelines`` is a large package that runs inside worker and Blender
    processes. The allowlist above is at module granularity, so this is the
    half that says *which* module."""
    tree = ast.parse((ENGINE / "sheetout.py").read_text(encoding="utf-8"))
    reached = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("pipelines")
        for alias in node.names
    }
    assert reached == {"sheet"}


def test_the_export_is_split_so_only_the_snapshot_reads_the_document():
    """The thread split, stated where it can be checked.

    ``snapshot`` is the whole of the export that reads the document, so that it
    can be the whole of the export that runs on the frame thread: ``frame_flat``
    fills and evicts the flatten cache and ``layers_for`` copies track
    properties down onto cels, which is exactly what the onion-skin draw is
    doing to the same dicts. So ``compose`` must take no document -- a
    parameter it could reach through is a parameter a later edit will reach
    through -- and ``from_document`` must be the two back to back rather than a
    third path that could drift from either.
    """
    assert "doc" not in inspect.signature(sheetout.compose).parameters

    doc = _animated()
    image, plan, extra = sheetout.compose(*sheetout.snapshot(doc))
    image.close()
    direct_image, direct_plan, direct_extra = sheetout.from_document(_animated())
    direct_image.close()
    assert (plan, extra) == (direct_plan, direct_extra)


def test_a_snapshot_survives_the_document_recompositing_underneath_it():
    """The arrays handed out are the ones the encoder keeps. The cache replaces
    entries rather than writing into them, so a later flatten on the frame
    thread cannot rewrite a sheet that is halfway to disk."""
    doc = _animated()
    frames, _durations, _tags = sheetout.snapshot(doc)
    taken = [plane.copy() for plane in frames]

    doc.set_current_frame(1)
    doc.write_colour((0, 0, 4, 4), (0, 255, 0, 255), np.ones((4, 4), dtype=np.float32))
    for frame in doc.anim.frames:
        doc.frame_flat(frame.uid)

    assert all(np.array_equal(a, b) for a, b in zip(frames, taken, strict=True))
