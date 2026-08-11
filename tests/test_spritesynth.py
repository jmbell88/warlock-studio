"""The pure half of sprite synthesis: geometry, guides, mattes, arithmetic.

Everything here runs with a Pillow-drawn fixture and no GPU, which is the whole
point of the module being pure -- the parts of this feature that can be wrong
silently (a cell rectangle off by one, a palette that is not shared, a baseline
that clips a head off) are all decided here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from warlock.pipelines import control, reference
from warlock.pipelines import spritesynth as ss

BG = (200, 200, 200)


def _atlas(kind, *, sizes=None, offset=(0, 0), bg=BG):
    """An atlas with one dark blob per cell, standing on the cell's floor."""
    geom = ss.geometry(kind)
    im = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), bg)
    draw = ImageDraw.Draw(im)
    for index, cell in enumerate(geom.cells):
        h = (sizes or {}).get(index, cell.h // 2)
        w = cell.w // 4
        cx = cell.x + cell.w // 2 + offset[0]
        bottom = cell.y + cell.h - cell.h // 10 + offset[1]
        draw.rectangle(
            (cx - w // 2, bottom - h, cx + w // 2, bottom), fill=(20, 30, 40)
        )
    return geom, im


# --- geometry ---------------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_cells_tile_the_atlas_exactly(kind):
    geom = ss.geometry(kind)
    assert geom.columns * geom.cell_w == ss.ATLAS_PX
    assert geom.rows * geom.cell_h == ss.ATLAS_PX
    covered = np.zeros((ss.ATLAS_PX, ss.ATLAS_PX), dtype=int)
    for cell in geom.cells:
        covered[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_the_turnaround_order_is_front_left_right_back():
    cells = ss.geometry("turnaround").cells
    assert [c.name for c in cells] == list(ss.DIRECTION_ORDER)
    assert [(c.row, c.col) for c in cells] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(c.frame == 0 for c in cells)


def test_the_walk_rows_are_one_direction_each_left_to_right():
    geom = ss.geometry("walk")
    assert geom.frames_per_direction == 4
    for row, name in enumerate(ss.DIRECTION_ORDER):
        row_cells = [c for c in geom.cells if c.row == row]
        assert [c.name for c in row_cells] == [name] * 4
        assert [c.frame for c in row_cells] == [0, 1, 2, 3]
        assert [c.col for c in row_cells] == [0, 1, 2, 3]


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_every_cell_carries_its_directions_yaw(kind):
    for cell in ss.geometry(kind).cells:
        assert cell.yaw == ss.DIRECTION_YAWS[cell.name]


def test_an_unknown_sheet_type_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown sprite sheet type"):
        ss.geometry("isometric")


# --- guide templates --------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_shipped_template_covers_every_cell_in_order(kind):
    geom = ss.geometry(kind)
    template = ss.load_guide_template(kind)
    assert [(p.name, p.frame) for p in template.poses] == [
        (c.name, c.frame) for c in geom.cells
    ]


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_shipped_template_has_a_comment_explaining_the_convention(kind):
    raw = json.loads((ss.TEMPLATE_DIR / f"{kind}.json").read_text(encoding="utf-8"))
    assert len(raw["comment"]) > 100


def _raw(kind):
    return json.loads((ss.TEMPLATE_DIR / f"{kind}.json").read_text(encoding="utf-8"))


def test_a_segment_naming_an_unknown_joint_is_refused_at_load():
    raw = _raw("turnaround")
    raw["segments"].append(["neck", "tentacle"])
    with pytest.raises(ValueError, match="missing a point"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_template_with_the_wrong_cell_count_is_refused_at_load():
    raw = _raw("turnaround")
    raw["poses"] = raw["poses"][:3]
    with pytest.raises(ValueError, match="3 poses"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_template_in_the_wrong_order_is_refused_at_load():
    raw = _raw("turnaround")
    raw["poses"][0], raw["poses"][1] = raw["poses"][1], raw["poses"][0]
    with pytest.raises(ValueError, match="wrong order"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_point_outside_its_cell_is_refused_at_load():
    raw = _raw("turnaround")
    raw["poses"][0]["points"]["head"] = [1.4, 0.2]
    with pytest.raises(ValueError, match="outside its cell"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_pose_with_no_head_is_refused_at_load():
    raw = _raw("turnaround")
    del raw["poses"][0]["points"]["head"]
    with pytest.raises(ValueError, match="no 'head' point"):
        ss._parse_template(raw, ss.geometry("turnaround"))


# --- the guide --------------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_guide_draws_structure_in_every_cell(kind):
    geom = ss.geometry(kind)
    guide = ss.render_guide(geom, ss.load_guide_template(kind))
    assert guide.size == (ss.ATLAS_PX, ss.ATLAS_PX)
    assert guide.mode == "RGB"
    for cell in geom.cells:
        assert control.edge_fraction(guide.crop(cell.box)) > 0.001


def test_the_guide_render_is_deterministic():
    geom = ss.geometry("walk")
    template = ss.load_guide_template("walk")
    first = ss.render_guide(geom, template).tobytes()
    second = ss.render_guide(geom, template).tobytes()
    assert first == second


def test_the_guide_never_draws_outside_a_cell():
    # A stroke that crossed a cell boundary would ask the ControlNet for a limb
    # belonging to the neighbouring direction.
    geom = ss.geometry("walk")
    arr = np.asarray(ss.render_guide(geom, ss.load_guide_template("walk")).convert("L"))
    for col in range(1, geom.columns):
        assert not arr[:, col * geom.cell_w - 1 : col * geom.cell_w + 1].any()


# --- splitting --------------------------------------------------------------


def test_split_uses_the_predetermined_rectangles():
    geom, atlas = _atlas("turnaround")
    parts = ss.split(atlas, geom)
    assert len(parts) == len(geom.cells)
    assert all(p.size == (geom.cell_w, geom.cell_h) for p in parts)


def test_a_misregistered_atlas_still_splits_on_the_grid():
    geom, atlas = _atlas("walk", offset=(7, -5))
    parts = ss.split(atlas, geom)
    assert [p.size for p in parts] == [(geom.cell_w, geom.cell_h)] * len(geom.cells)


# --- matting ----------------------------------------------------------------


def test_the_flood_fill_mattes_every_cell():
    geom, atlas = _atlas("turnaround")
    matted_atlas, took = ss.matte_cells(atlas, geom)
    assert took == (True,) * 4
    alpha = np.asarray(matted_atlas)[:, :, 3]
    for cell in geom.cells:
        sub = alpha[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
        assert sub.max() == 255 and sub.min() == 0


def test_a_cell_the_fill_ate_is_left_opaque_and_reported():
    geom = ss.geometry("turnaround")
    # A uniform cell has no subject at all: the fill takes the whole rectangle
    # and the honest answer is "not matted", not a fully transparent cell.
    atlas = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    matted_atlas, took = ss.matte_cells(atlas, geom)
    assert took == (False,) * 4
    assert np.asarray(matted_atlas)[:, :, 3].min() == 255


def test_an_unmatted_atlas_still_quantizes():
    from warlock.pipelines import pixelsheet

    geom = ss.geometry("turnaround")
    atlas = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    matted_atlas, _ = ss.matte_cells(atlas, geom)
    out, palette = pixelsheet.quantize_shared(matted_atlas, 8)
    assert out.size == matted_atlas.size and palette


# --- the baseline -----------------------------------------------------------


def _bottoms(atlas, geom):
    alpha = np.asarray(atlas.convert("RGBA"))[:, :, 3]
    return [ss._cell_bounds(alpha, c) for c in geom.cells]


def test_the_baseline_is_the_median_of_the_cells():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for cell, bottom in zip(geom.cells, (300, 400, 400, 460), strict=True):
        draw.rectangle(
            (cell.x + 100, cell.y + bottom - 200, cell.x + 200, cell.y + bottom),
            fill=(10, 20, 30, 255),
        )
    aligned = ss.baseline_align(atlas, geom)
    bottoms = [b[1] for b in _bottoms(aligned, geom)]
    assert bottoms == [400, 400, 400, 400]


def test_alignment_never_pushes_a_subject_off_its_cell():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    # Three short subjects near the floor and one that fills its cell: the tall
    # one cannot be moved to the median without losing its head.
    for cell, (top, bottom) in zip(
        geom.cells, ((480, 500), (480, 500), (480, 500), (0, 511)), strict=True
    ):
        draw.rectangle(
            (cell.x + 100, cell.y + top, cell.x + 200, cell.y + bottom),
            fill=(10, 20, 30, 255),
        )
    aligned = ss.baseline_align(atlas, geom)
    for cell, bound in zip(geom.cells, _bottoms(aligned, geom), strict=True):
        assert bound is not None
        assert 0 <= bound[0] <= bound[1] <= cell.h - 1
    # And the pixel count is preserved, i.e. nothing was shifted into oblivion.
    before = int((np.asarray(atlas)[:, :, 3] > 0).sum())
    after = int((np.asarray(aligned)[:, :, 3] > 0).sum())
    assert before == after


def test_an_empty_atlas_aligns_to_itself():
    geom = ss.geometry("walk")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    assert ss.baseline_align(atlas, geom).tobytes() == atlas.tobytes()


# --- reduction --------------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
@pytest.mark.parametrize("logical", (32, 48, 64))
def test_reduction_lands_every_cell_boundary_on_a_pixel(kind, logical):
    geom, atlas = _atlas(kind)
    matted, _ = ss.matte_cells(atlas, geom)
    out = ss.reduce_atlas(matted, geom, logical)
    assert out.size == (geom.columns * logical, geom.rows * logical)
    # 48 does not divide 512 or 256, which is exactly why this is one resize of
    # the whole atlas: the *columns* divide, so each output cell is square and
    # sits on an integer boundary regardless.
    for index, cell in enumerate(geom.cells):
        assert (index % geom.columns) * logical == cell.col * logical


def test_reduction_refuses_a_nonsense_size():
    geom, atlas = _atlas("turnaround")
    with pytest.raises(ValueError, match="at least 1"):
        ss.reduce_atlas(atlas, geom, 0)


def test_one_palette_is_shared_across_the_whole_atlas():
    from warlock.pipelines import pixelsheet

    geom = ss.geometry("turnaround")
    atlas = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    draw = ImageDraw.Draw(atlas)
    for index, cell in enumerate(geom.cells):
        draw.rectangle(
            (cell.x + 100, cell.y + 100, cell.x + 400, cell.y + 400),
            fill=(20 + index * 40, 60, 90),
        )
    matted, _ = ss.matte_cells(atlas, geom)
    out, palette = pixelsheet.quantize_shared(ss.reduce_atlas(matted, geom, 64), 8)
    assert len(palette) <= 8
    rgba = np.asarray(out)
    colours = {tuple(int(c) for c in p) for p in rgba[rgba[:, :, 3] > 0][:, :3]}
    assert len(colours) <= 8


# --- warnings ---------------------------------------------------------------


def _codes(warnings):
    return {w["code"] for w in warnings}


def test_an_empty_cell_is_reported():
    geom, atlas = _atlas("turnaround")
    draw = ImageDraw.Draw(atlas)
    back = geom.cells[3]
    draw.rectangle(back.box, fill=BG)
    matted, took = ss.matte_cells(atlas, geom)
    warnings = ss.structural_warnings(matted, geom, took)
    assert "empty" in _codes(warnings) or "unmatted" in _codes(warnings)


def test_a_cell_running_off_its_edge_is_reported():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for index, cell in enumerate(geom.cells):
        top = cell.y if index == 2 else cell.y + 100
        draw.rectangle(
            (cell.x + 100, top, cell.x + 300, cell.y + 400), fill=(1, 2, 3, 255)
        )
    warnings = ss.structural_warnings(atlas, geom, (True,) * 4)
    clipped = [w for w in warnings if w["code"] == "clipped"]
    assert [w["cell"] for w in clipped] == ["right"]
    assert "top" in clipped[0]["detail"]


def test_a_cell_far_off_the_sheets_size_is_reported():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for index, cell in enumerate(geom.cells):
        side = 40 if index == 1 else 300
        draw.rectangle(
            (cell.x + 50, cell.y + 50, cell.x + 50 + side, cell.y + 50 + side),
            fill=(1, 2, 3, 255),
        )
    warnings = ss.structural_warnings(atlas, geom, (True,) * 4)
    assert [w["cell"] for w in warnings if w["code"] == "occupancy"] == ["left"]


def test_an_unmatted_cell_is_reported():
    geom, atlas = _atlas("turnaround")
    matted, _ = ss.matte_cells(atlas, geom)
    warnings = ss.structural_warnings(matted, geom, (True, True, False, True))
    unmatted = [w for w in warnings if w["code"] == "unmatted"]
    assert [w["cell"] for w in unmatted] == ["right"]


def test_every_warning_carries_a_known_code_and_a_sentence():
    geom, atlas = _atlas("turnaround", sizes={0: 40})
    matted, took = ss.matte_cells(atlas, geom)
    for warning in ss.structural_warnings(matted, geom, took):
        assert warning["code"] in ss.WARNING_CODES
        assert warning["detail"].endswith(".")
        assert warning["cell"] in ss.DIRECTION_ORDER


def test_an_unknown_warning_code_cannot_be_minted():
    with pytest.raises(ValueError, match="unknown sprite warning code"):
        ss._warning(ss.geometry("turnaround").cells[0], "vibes", "hm.")


# --- front preservation -----------------------------------------------------


def _report(**kw):
    base = dict(ok=True, bbox=(0, 0, 100, 200), touches=(), components=1)
    base.update(kw)
    return reference.Report(**base)


def test_a_clean_reference_of_matching_proportions_is_pasted_in():
    ok, why = ss.front_fits(_report(), _report(bbox=(0, 0, 105, 200)))
    assert ok is True
    assert "reference drawing itself" in why


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ({"ok": False}, "refused"),
        ({"touches": ("left",)}, "touches the edge"),
        ({"components": 2}, "more than one object"),
        ({"bbox": None}, "could not be measured"),
    ),
)
def test_a_reference_that_does_not_fit_is_refused_with_a_reason(source, expected):
    ok, why = ss.front_fits(_report(**source), _report())
    assert ok is False
    assert expected in why


def test_a_reference_of_the_wrong_shape_is_refused_with_the_ratio():
    ok, why = ss.front_fits(_report(bbox=(0, 0, 400, 200)), _report())
    assert ok is False
    assert "proportions" in why


def test_the_pasted_front_shares_the_sheets_palette():
    from warlock.pipelines import pixelsheet

    geom, atlas = _atlas("turnaround")
    matted, _ = ss.matte_cells(atlas, geom)
    subject = Image.new("RGBA", (120, 240), (0, 0, 0, 0))
    ImageDraw.Draw(subject).rectangle((0, 0, 119, 239), fill=(250, 10, 10, 255))
    pasted = ss.preserve_front(matted, geom, subject)
    # Before quantization, so the reference's red is one of the colours median
    # cut is choosing between rather than a stranger pasted in afterwards.
    out, palette = pixelsheet.quantize_shared(ss.reduce_atlas(pasted, geom, 64), 8)
    front = geom.cells[0]
    cell = np.asarray(out)[0:64, 0:64]
    assert front.name == "front"
    assert (cell[:, :, 3] > 0).any()
    colours = {f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cell[cell[:, :, 3] > 0][:, :3]}
    assert colours <= set(palette)


def test_the_pasted_front_stands_on_the_shared_baseline():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for cell in geom.cells:
        draw.rectangle(
            (cell.x + 200, cell.y + 200, cell.x + 300, cell.y + 460),
            fill=(1, 2, 3, 255),
        )
    subject = Image.new("RGBA", (50, 50), (9, 9, 9, 255))
    pasted = ss.preserve_front(atlas, geom, subject)
    bounds = _bottoms(pasted, geom)
    assert bounds[0][1] == bounds[1][1] == 460


def test_front_preservation_is_a_turnaround_only_step():
    geom = ss.geometry("walk")
    with pytest.raises(ValueError, match="turnaround-only"):
        ss.preserve_front(
            Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX)), geom, Image.new("RGBA", (4, 4))
        )


# --- the sidecar ------------------------------------------------------------


def test_the_sidecar_slices_the_reduced_png_standalone():
    geom, atlas = _atlas("walk")
    matted, _ = ss.matte_cells(atlas, geom)
    reduced = ss.reduce_atlas(matted, geom, 48)
    sidecar = ss.draft_sidecar(
        draft_id="abc123",
        source_job="job1",
        created=1.0,
        geom=geom,
        logical_size=48,
        colors=32,
        candidates=[],
        recipe={},
    )
    assert sidecar["version"] == ss.SPRITE_DRAFT_VERSION
    assert sidecar["cell_w"] == sidecar["cell_h"] == 48
    covered = np.zeros((reduced.height, reduced.width), dtype=int)
    for cell in sidecar["cells"]:
        assert cell["x"] + cell["w"] <= reduced.width
        assert cell["y"] + cell["h"] <= reduced.height
        covered[
            cell["y"] : cell["y"] + cell["h"], cell["x"] : cell["x"] + cell["w"]
        ] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_the_sidecar_cells_are_in_timeline_order_with_their_yaws():
    geom = ss.geometry("walk")
    sidecar = ss.draft_sidecar(
        draft_id="abc123",
        source_job="job1",
        created=1.0,
        geom=geom,
        logical_size=64,
        colors=16,
        candidates=[],
        recipe={"base_model": "sdxl_cfg"},
    )
    assert [(c["name"], c["frame"]) for c in sidecar["cells"]] == [
        (c.name, c.frame) for c in geom.cells
    ]
    assert all(c["yaw"] == ss.DIRECTION_YAWS[c["name"]] for c in sidecar["cells"])
    assert sidecar["recipe"]["base_model"] == "sdxl_cfg"
