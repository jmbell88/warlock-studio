"""The pixel-sheet restyle's geometry, headless.

Everything here is about the properties that make eight directions agree: one
band per denoise, an exact crop back out of the square it was generated in,
silhouettes taken from the render rather than from the generation, and one
palette across the whole atlas. None of it needs a GPU, which is the point --
the model is the only part that cannot be asserted, so nothing else may depend
on having one.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import pixelsheet


def _meta(frame_size=128, columns=8, rows=1, cells=None):
    return {
        "id": "abcdef123456",
        "name": "sheet",
        "source_job": "job1",
        "frame_size": frame_size,
        "columns": columns,
        "rows": rows,
        "width": frame_size * columns,
        "height": frame_size * rows,
        "elevation": 30.0,
        "lighting": "flat",
        "yaws": [i * 45.0 for i in range(columns)],
        "poses": [{"id": None, "name": "rest"}],
        "cells": cells if cells is not None else [],
    }


def _atlas(meta, colour=(200, 60, 60)):
    """A source render: an opaque blob in each cell on transparency."""
    atlas = Image.new("RGBA", (meta["width"], meta["height"]), (0, 0, 0, 0))
    frame = meta["frame_size"]
    for row in range(meta["rows"]):
        for column in range(meta["columns"]):
            block = Image.new("RGBA", (frame // 2, frame // 2), (*colour, 255))
            atlas.paste(block, (column * frame + frame // 4, row * frame + frame // 4))
    return atlas


# --- banding ------------------------------------------------------------------


def test_eight_yaws_at_128px_is_exactly_one_band():
    """The geometry the whole design rests on: 8 x 128 = 1024, so every
    direction is denoised in one latent under one seed."""
    got = pixelsheet.bands(_meta(rows=1))
    assert len(got) == 1
    assert got[0].width == pixelsheet.BAND_PX
    assert got[0].x == 0


def test_rows_are_grouped_and_never_split():
    """A row is one subject's eight directions. Splitting one across two
    denoises is the flicker the single-latent design removes."""
    got = pixelsheet.bands(_meta(rows=20))
    assert [b.rows for b in got] == [8, 8, 4]
    assert [b.first_row for b in got] == [0, 8, 16]
    assert all(b.height <= pixelsheet.BAND_PX for b in got)


def test_a_narrow_sheet_is_letterboxed_symmetrically():
    band = pixelsheet.bands(_meta(columns=4))[0]
    assert band.width == 512
    assert band.x == (pixelsheet.BAND_PX - 512) // 2


def test_a_sheet_wider_than_one_generation_is_refused_with_a_remedy():
    with pytest.raises(pixelsheet.NotRestylable) as excinfo:
        pixelsheet.check_restylable(_meta(frame_size=256), 32)
    message = str(excinfo.value)
    assert "2048px" in message
    # The refusal has to name what to do instead, not just what is wrong.
    assert "128px or smaller" in message


def test_a_logical_size_that_does_not_divide_the_cell_is_refused():
    with pytest.raises(pixelsheet.NotRestylable, match="divide"):
        pixelsheet.check_restylable(_meta(), 48)
    pixelsheet.check_restylable(_meta(), 32)  # 128 / 32 = 4, fine


# --- the round trip -----------------------------------------------------------


def test_letterbox_and_crop_back_is_exact():
    meta = _meta(columns=4)
    atlas = _atlas(meta)
    band = pixelsheet.bands(meta)[0]

    init = pixelsheet.band_init(atlas, band)
    assert init.size == (pixelsheet.BAND_PX, pixelsheet.BAND_PX)
    assert init.mode == "RGB"

    back = pixelsheet.crop_back(init, band)
    assert back.size == (band.width, band.height)
    # The subject's pixels survive the trip unchanged; only the transparent
    # background became grey, which is what the remask undoes.
    flat = Image.new("RGB", atlas.size, pixelsheet.BAND_BACKGROUND)
    flat.paste(atlas, (0, 0), atlas)
    assert np.array_equal(np.asarray(back), np.asarray(flat))


def test_remask_restores_the_renders_alpha_bit_exactly():
    """The silhouettes are exact by construction: the render is orthographic
    geometry, and whatever the model invented outside it is background."""
    meta = _meta(columns=4)
    source = _atlas(meta)
    styled = Image.new("RGB", source.size, (30, 200, 90))

    out = pixelsheet.remask(styled, source)

    assert np.array_equal(
        np.asarray(out)[:, :, 3], np.asarray(source)[:, :, 3]
    )
    # And the colours are the generation's, not the render's.
    opaque = np.asarray(out)[:, :, 3] > 0
    assert {tuple(p) for p in np.asarray(out)[:, :, :3][opaque]} == {(30, 200, 90)}


def test_remask_refuses_a_mismatched_pair():
    with pytest.raises(ValueError):
        pixelsheet.remask(
            Image.new("RGB", (16, 16)), Image.new("RGBA", (32, 32))
        )


def test_downscale_keeps_every_cell_boundary_on_a_pixel():
    meta = _meta(columns=4, rows=2)
    atlas = _atlas(meta)
    small = pixelsheet.downscale(atlas, 4)
    assert small.size == (128, 64)
    # Cell 1 starts at 128/4 = 32 in the reduced atlas, and its content still
    # begins a quarter of the way into it.
    arr = np.asarray(small)
    assert arr[:, 32:64, 3].any()
    assert not arr[0:8, 32:40, 3].any()


def test_a_non_dividing_factor_is_refused_rather_than_rounded():
    with pytest.raises(ValueError):
        pixelsheet.downscale(Image.new("RGBA", (10, 10)), 3)


# --- the shared palette --------------------------------------------------------


def test_one_palette_is_chosen_across_the_whole_atlas():
    """Per-cell quantization is how the same shirt comes out two shades in two
    directions, which is what reads as flicker when a character turns."""
    meta = _meta(columns=4)
    atlas = Image.new("RGBA", (meta["width"], meta["height"]), (0, 0, 0, 0))
    frame = meta["frame_size"]
    # Four cells, each a *slightly* different red: a per-cell quantize would
    # give each its own entry, a shared one collapses them.
    for column in range(4):
        block = Image.new("RGBA", (frame, frame), (200 + column, 60, 60, 255))
        atlas.paste(block, (column * frame, 0))

    out, palette = pixelsheet.quantize_shared(atlas, 2)

    assert len(palette) <= 2
    arr = np.asarray(out)
    opaque = arr[:, :, 3] > 0
    assert len({tuple(p) for p in arr[:, :, :3][opaque]}) <= 2
    assert all(entry.startswith("#") and len(entry) == 7 for entry in palette)


def test_quantized_alpha_is_binary():
    meta = _meta(columns=2)
    out, _palette = pixelsheet.quantize_shared(_atlas(meta), 8)
    assert set(np.unique(np.asarray(out)[:, :, 3])) <= {0, 255}


def test_an_empty_atlas_says_so_rather_than_quantizing_nothing():
    with pytest.raises(ValueError, match="empty"):
        pixelsheet.quantize_shared(Image.new("RGBA", (32, 32), (0, 0, 0, 0)), 8)


# --- where a palette comes from -------------------------------------------------


RAMP = ((16, 16, 24), (72, 56, 48), (168, 120, 88), (232, 216, 184))


def test_a_designed_palette_is_returned_verbatim_and_never_requantised():
    """The whole point of naming a palette: those colours, unchanged.

    A median cut run over the atlas "as well" would silently substitute the
    render's own muddy colours for the ones the artist chose, which is the
    failure an authored ramp exists to fix.
    """
    entries, source = pixelsheet.resolve_palette(
        _atlas(_meta(columns=2)), colors=8, entries=RAMP
    )
    assert source == "designed"
    assert entries == RAMP


def test_a_designed_palette_is_taken_without_looking_at_the_atlas():
    """No pixels are read on that branch, so an atlas ``quantize_shared`` would
    refuse outright is still fine to map with a palette."""
    empty = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    entries, source = pixelsheet.resolve_palette(empty, colors=8, entries=RAMP)
    assert (entries, source) == (RAMP, "designed")


def test_with_no_palette_the_median_cut_becomes_the_palette_source():
    meta = _meta(columns=2)
    entries, source = pixelsheet.resolve_palette(_atlas(meta), colors=4, entries=None)
    assert source == "derived"
    assert 0 < len(entries) <= 4
    assert all(len(e) == 3 and all(0 <= c <= 255 for c in e) for e in entries)


def test_the_derived_branch_is_exactly_the_hex_list_parsed_back():
    """The collapse ``_q_troupe._quantise`` was refactored onto, pinned: the
    triples are ``quantize_shared``'s own answer and nothing else."""
    atlas = _atlas(_meta(columns=2))
    _mapped, hexes = pixelsheet.quantize_shared(atlas, 8)
    old = tuple(
        (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in hexes
    )
    entries, source = pixelsheet.resolve_palette(atlas, colors=8, entries=None)
    assert (entries, source) == (old, "derived")


def test_an_empty_tuple_is_a_palette_of_no_colours_and_not_an_absent_one():
    """``None`` is how a caller says "I have no palette". ``()`` is a palette
    file somebody emptied, and passing it through is what lets
    ``map_palette``'s "an empty palette maps nothing" be the refusal the user
    sees rather than a silent median cut."""
    entries, source = pixelsheet.resolve_palette(
        _atlas(_meta(columns=2)), colors=8, entries=()
    )
    assert (entries, source) == ((), "designed")


def test_the_shared_property_survives_a_designed_palette():
    """The insight ``resolve_palette``'s docstring carries, as a test.

    What made ``quantize_shared`` shared was never the median cut -- it is that
    one entry set is applied over the whole atlas in one pass. So four cells
    that differ by a hair still collapse onto the same entries when the palette
    is designed, exactly as they do when it is derived.
    """
    from warlock.pipelines import pixel

    meta = _meta(columns=4)
    atlas = Image.new("RGBA", (meta["width"], meta["height"]), (0, 0, 0, 0))
    frame = meta["frame_size"]
    for column in range(4):
        block = Image.new("RGBA", (frame, frame), (200 + column, 60, 60, 255))
        atlas.paste(block, (column * frame, 0))

    entries, _ = pixelsheet.resolve_palette(atlas, colors=8, entries=RAMP)
    mapped = np.asarray(pixel.map_palette(atlas, entries))
    cells = [
        {tuple(p) for p in mapped[:, column * frame : (column + 1) * frame, :3]
         .reshape(-1, 3).tolist()}
        for column in range(4)
    ]
    assert cells[0] == cells[1] == cells[2] == cells[3]


# --- the sidecar ---------------------------------------------------------------


def test_the_sidecar_scales_every_rectangle_by_the_reduction():
    meta = _meta(
        columns=2,
        cells=[
            {
                "index": 0, "row": 0, "column": 0, "x": 0, "y": 0, "w": 128, "h": 128,
                "pose": None, "pose_name": "rest", "yaw": 0.0, "frame": 0,
                "pivot_x": 64.0, "pivot_y": 128.0,
                "trim": {"x": 32, "y": 30, "w": 64, "h": 66},
            },
            {
                "index": 1, "row": 0, "column": 1, "x": 128, "y": 0, "w": 128, "h": 128,
                "pose": None, "pose_name": "rest", "yaw": 45.0, "frame": 0,
                "pivot_x": 64.0, "pivot_y": 128.0, "trim": None,
            },
        ],
    )

    doc = pixelsheet.pixel_sidecar(
        meta,
        image="abc.pixel.png",
        logical_size=32,
        palette=["#000000"],
        recipe={"strength": 0.45, "seed": 7},
        created=1.0,
    )

    assert doc["version"] == pixelsheet.PIXEL_SHEET_VERSION
    assert doc["frame_size"] == 32
    assert (doc["width"], doc["height"]) == (64, 32)
    first, second = doc["cells"]
    assert (first["x"], first["y"], first["w"]) == (0, 0, 32)
    assert (second["x"], second["w"]) == (32, 32)
    assert (first["pivot_x"], first["pivot_y"]) == (16.0, 32.0)
    # Origin floors, extent ceils: a trim that rounded inward would clip a
    # pixel off the silhouette it exists to describe.
    assert first["trim"] == {"x": 8, "y": 7, "w": 16, "h": 17}
    assert second["trim"] is None
    assert doc["restyle"] == {"strength": 0.45, "seed": 7}


def test_the_sidecar_carries_the_render_grids_own_version():
    doc = pixelsheet.pixel_sidecar(
        _meta(), image="a.png", logical_size=32, palette=[], recipe={}, created=0.0
    )
    # Two version numbers in one file, on purpose: this grid is a derivative of
    # that one, and it says which one it was made from.
    assert doc["sheet_version"] == pixelsheet.SHEET_VERSION
    assert doc["version"] == pixelsheet.PIXEL_SHEET_VERSION


# --- the constants the stored artifacts are keyed on ----------------------------


def test_recording_the_lattice_moved_no_sidecar_version():
    """The 2026-08-29 pass added a lattice measurement to four sidecars and
    bumped none of their versions, on purpose.

    A new optional key readers may ignore is not a new format -- ``sheet.py``'s
    sidecar docstring states that rule -- and the default bytes did not move.
    A bump that changes nothing would invalidate every stored benchmark
    comparison for free, which is a real cost paid for a tidier-looking number.
    The literals are the pin: they are what fails if somebody bumps one.
    """
    from warlock.pipelines import spritesynth, tileatlas, tilesheet

    assert pixelsheet.PIXEL_SHEET_VERSION == 1
    # 2, and *not* because of the lattice. The sprite draft's reduction changed
    # from ``reduce_atlas``' single NEAREST resize to ``pixelize.reduce``'s
    # alpha-weighted box on the same day, which moves the published bytes -- and
    # that is the case where the rule cuts the other way: a reader has to be
    # able to tell which compiler drew the sheet. The lattice block still rode
    # in for free; what bought the bump was the pixels.
    assert spritesynth.SPRITE_DRAFT_VERSION == 2
    assert tileatlas.TILE_ATLAS_VERSION == 1
    # 2 is the tile-sheet vocabulary widening, taken before any of this.
    assert tilesheet.TILE_SHEET_VERSION == 2
