"""The two ways out of the sheet panel that are not a file on disk.

A rendered 8-direction sheet and its pixel restyle both dead-ended at Save PNG,
though everything needed to open them already existed. Neither door below
carries a new algorithm; what is worth testing is the *geometry* they read off
the sidecar and the fact that each lands in the machinery its editor already
has -- unlinked in the Inker, parked behind Packwright's confirm popup.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from PIL import Image

from warlock.studio import inker_mode, packwright_mode


class _Ctx:
    """Runs a submitted callable inline, as the plotter suite's FakeCtx does."""

    def __init__(self) -> None:
        self.svc = object()
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self) -> None:
        self.inker = None
        self.packwright = None
        self.mode = "home"
        self.previous_mode = "home"
        self.mode_observed = "home"
        self.preview: dict[str, Any] = {}


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


# --- sheet_grid ---------------------------------------------------------------


def test_a_square_sidecar_gives_its_frame_size() -> None:
    cell, count = inker_mode.sheet_grid(
        {"frame_size": 64, "columns": 4, "rows": 2, "cells": [{}] * 8}
    )
    assert cell == (64, 64)
    assert count == 8


def test_a_non_square_sidecar_reads_the_pair_not_the_zero() -> None:
    """``frame_size`` is written as 0 on a non-square plan -- a loud wrong
    answer rather than a quiet one -- and the truth is in ``frame_w``/``h``."""
    cell, count = inker_mode.sheet_grid(
        {"frame_size": 0, "frame_w": 48, "frame_h": 64, "cells": [{}] * 3}
    )
    assert cell == (48, 64)
    assert count == 3


def test_the_cell_list_beats_columns_times_rows() -> None:
    """A padded final row would make the product overcount."""
    _cell, count = inker_mode.sheet_grid(
        {"frame_size": 32, "columns": 4, "rows": 2, "cells": [{}] * 7}
    )
    assert count == 7


def test_a_sidecar_with_no_geometry_is_refused() -> None:
    with pytest.raises(ValueError, match="frame size"):
        inker_mode.sheet_grid({"cells": [{}]})
    with pytest.raises(ValueError, match="cells"):
        inker_mode.sheet_grid({"frame_size": 16})


# --- the two doors ------------------------------------------------------------


def _sheet(tmp_path, columns: int = 4, rows: int = 2, size: int = 16) -> dict[str, Any]:
    """A rendered sheet on disk: a PNG plus the sidecar that describes it."""
    atlas = np.zeros((rows * size, columns * size, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    for r in range(rows):
        for c in range(columns):
            atlas[r * size:(r + 1) * size, c * size:(c + 1) * size, :3] = 40 + r * 20 + c
    Image.fromarray(atlas, "RGBA").save(tmp_path / "sheet.png")
    record = {
        "name": "turnaround",
        "columns": columns,
        "rows": rows,
        "frame_size": size,
        "cells": [{"index": i} for i in range(columns * rows)],
    }
    (tmp_path / "sheet.json").write_text(json.dumps(record), encoding="utf-8")
    return record


def _serve(monkeypatch, tmp_path, record: dict[str, Any]) -> None:
    monkeypatch.setattr(
        "warlock.service.sheets.get_sheet", lambda svc, job, sid: record
    )
    monkeypatch.setattr(
        "warlock.service.sheets.sheet_png", lambda svc, job, sid: tmp_path / "sheet.png"
    )
    monkeypatch.setattr(
        "warlock.service.sheets.get_pixel_sheet", lambda svc, job, sid: record
    )
    monkeypatch.setattr(
        "warlock.service.sheets.sheet_pixel_png",
        lambda svc, job, sid: tmp_path / "sheet.png",
    )


def test_editing_a_rendered_sheet_opens_rows_times_columns_frames(tmp_path, monkeypatch):
    ctx = _Ctx()
    _serve(monkeypatch, tmp_path, _sheet(tmp_path))

    inker_mode.open_rendered_sheet(ctx, "j1", "s1")

    assert ctx.submitted == ["inker-open:sheet:s1:render"]
    result = ctx.result
    doc = result["doc"]
    assert len(doc.anim.frames) == 8
    assert doc.size == (16, 16)
    # Unlinked: the sheet on disk is where the document came from, not its file,
    # so the first Ctrl+S must be a Save As.
    assert result["path"] is None
    assert result["format"] == "ora"
    assert result["title"] == "turnaround"


def test_the_pixel_restyle_opens_through_its_own_sidecar(tmp_path, monkeypatch):
    ctx = _Ctx()
    _serve(monkeypatch, tmp_path, _sheet(tmp_path))

    inker_mode.open_rendered_sheet(ctx, "j1", "s1", pixel=True)

    assert ctx.submitted == ["inker-open:sheet:s1:pixel"]
    assert ctx.result["title"] == "turnaround (pixel)"
    assert len(ctx.result["doc"].anim.frames) == 8


def test_a_rendered_sheet_is_sliced_as_a_plain_grid_not_a_directional_layout(
    tmp_path, monkeypatch
):
    """``animation.SHEET_KINDS`` is turnaround/walk only, so
    ``DirectionalLayout.of()`` would return ``None`` for a render's sidecar and
    ``document_from_atlas`` would refuse the whole door."""
    ctx = _Ctx()
    _serve(monkeypatch, tmp_path, _sheet(tmp_path))
    inker_mode.open_rendered_sheet(ctx, "j1", "s1")
    assert ctx.result["doc"].anim.layout is None


def test_adding_a_rendered_sheet_parks_it_with_the_cell_prefilled(tmp_path, monkeypatch):
    ctx = _Ctx()
    _serve(monkeypatch, tmp_path, _sheet(tmp_path, size=24))
    packwright_mode.new_document(ctx)
    tab = packwright_mode.active(ctx)

    packwright_mode.add_rendered_sheet(ctx, "j1", "s1")
    packwright_mode.on_task_done(
        ctx, type("D", (), {"key": f"packwright-tileset:{tab.uid}",
                            "result": ctx.result, "message": ""})()
    )

    state = packwright_mode.ensure(ctx)
    assert state.tileset_import is not None
    assert state.tileset_import[1] == "turnaround"
    assert state.tileset_cell == (24, 24)
    assert state.tileset_import_open is False


def test_a_picked_file_still_keeps_the_last_typed_cell_size(tmp_path, monkeypatch):
    """The cell size persists across imports on purpose, and a door that does
    not know the answer must not overwrite it."""
    ctx = _Ctx()
    packwright_mode.new_document(ctx)
    tab = packwright_mode.active(ctx)
    state = packwright_mode.ensure(ctx)
    state.tileset_cell = (48, 48)

    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    packwright_mode.on_task_done(
        ctx,
        type("D", (), {
            "key": f"packwright-tileset:{tab.uid}",
            "result": {"tileset": ("p.png", "p", pixels), "uid": tab.uid},
            "message": "",
        })(),
    )
    assert state.tileset_cell == (48, 48)


def _charsheet(tmp_path, size: int = 8) -> dict[str, Any]:
    """A Troupe-shaped sidecar: per-cell rects *and* an ``animation`` block.

    The plain ``_sheet`` above is every rendered sheet written before
    ``charsheet.animation_block`` existed -- cells that carry only an index,
    and no animation at all -- which is why the two doors both have to work.
    """
    from warlock.pipelines import charsheet

    block = charsheet.animation_block()
    columns, rows = 16, 16
    atlas = np.zeros((rows * size, columns * size, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    Image.fromarray(atlas, "RGBA").save(tmp_path / "sheet.png")
    return {
        "name": "ranger",
        "columns": columns,
        "rows": rows,
        "frame_size": size,
        "cells": [
            {
                "index": i,
                "x": (i % columns) * size,
                "y": (i // columns) * size,
                "w": size,
                "h": size,
            }
            for i in range(columns * rows)
        ],
        "animation": block,
    }


def test_a_character_sheet_opens_with_its_tags_and_durations(tmp_path, monkeypatch):
    """The handoff the button's tooltip and ``docs/manual/33-troupe.md`` both
    promise: "one tag per animation and direction".

    ``sheetin.document_from_sheet`` was written for this and had no production
    caller at all -- the door said ``document_from_grid``, which is geometry
    only, so a character sheet opened as 256 untagged frames at one default
    duration and every span, loop flag and per-animation frame rate in the
    sidecar was dropped on the floor.
    """
    ctx = _Ctx()
    record = _charsheet(tmp_path)
    _serve(monkeypatch, tmp_path, record)

    inker_mode.open_rendered_sheet(ctx, "j1", "s1")

    doc = ctx.result["doc"]
    assert len(doc.anim.frames) == 256
    names = [tag.name for tag in doc.anim.tags]
    assert len(names) == len(record["animation"]["tags"])
    assert "walk_front" in names and "idle_front" in names
    # The durations are per animation, which is the point of carrying them: a
    # run is faster than an idle, and a single default would flatten both.
    by_index = {
        int(f["cell_index"]): int(f["duration_ms"])
        for f in record["animation"]["frames"]
    }
    assert doc.anim.frames[0].duration_ms == by_index[0]
    fastest = min(by_index.values())
    assert fastest != by_index[0], "this fixture has to contain two rates"
    assert any(frame.duration_ms == fastest for frame in doc.anim.frames)


def test_a_sheet_with_no_animation_block_still_opens_as_a_plain_grid(
    tmp_path, monkeypatch
):
    """The fallback stays: every sheet written before the block existed has
    cells that carry an index and nothing else, and slicing those as rects
    would raise a KeyError on the first one."""
    ctx = _Ctx()
    _serve(monkeypatch, tmp_path, _sheet(tmp_path))
    inker_mode.open_rendered_sheet(ctx, "j1", "s1")
    assert len(ctx.result["doc"].anim.frames) == 8
    assert not ctx.result["doc"].anim.tags


def test_a_pixel_artifact_opens_unlinked_and_derives_itself(tmp_path, monkeypatch):
    """The Create pixel preview's own way into the Inker.

    Unlinked deliberately, and not merely by analogy with the sheets above:
    ``pixel_128.png`` is *derived*, so ``derive.get_file`` rebuilds it whenever
    the knobs make the copy on disk stale -- a linked tab writing back would
    have its edit thrown away by the next re-derive.
    """
    png = tmp_path / "pixel_64.png"
    Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8)).save(png)
    seen: dict[str, Any] = {}

    def _get_file(svc, job_id, name, **kwargs):
        seen.update({"job": job_id, "name": name, **kwargs})
        return png

    monkeypatch.setattr("warlock.service.derive.get_file", _get_file)
    ctx = _Ctx()

    inker_mode.open_pixel_artifact(
        ctx,
        "j1",
        "pixel_64.png",
        title="j1 pixels 64",
        pixel_colors=16,
        pixel_palette="nes",
        pixel_dither=True,
    )

    # The ``inker-open`` prefix is what gets it adopted with no routing change;
    # the ``pixel:`` segment keeps it off ``open_job_reference``'s key for a
    # different document of the same job.
    assert ctx.submitted == ["inker-open:pixel:j1:pixel_64.png"]
    # The knobs are captured on the frame thread and carried in, so the
    # preview, the export and this open describe one file.
    assert seen == {
        "job": "j1",
        "name": "pixel_64.png",
        "pixel_colors": 16,
        "pixel_palette": "nes",
        "pixel_dither": True,
    }
    result = ctx.result
    assert result["title"] == "j1 pixels 64"
    assert result.get("path") is None and result.get("link_kind") is None
    assert result["doc"].size == (8, 8)
