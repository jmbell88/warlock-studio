"""The tile sheet's door: every refusal, and what a good request writes.

The tile sizes and the view list here are *copies* -- ``service/`` may not
import ``studio/`` and a form's ceiling does not belong in a pipeline -- so the
drift pins at the bottom of this file are the point of it: if somebody adds a
tile size to ``pipelines.tilesheet`` and forgets this module, the copy goes red
rather than silently refusing a sheet the pipeline would have built.

The other thing this file exists to hold is the **optional** reference. It is
the one input on this door that is allowed to be absent, and three separate
things key on which it was: the IP-Adapter in ``params``, the registry rows a
refusal offers to install, and the VRAM charged at admission.
"""

from __future__ import annotations

import io

import pytest

from warlock import models, vram
from warlock.pipelines import tilesheet
from warlock.service import tilesheets
from warlock.service.errors import Invalid, TooLarge
from warlock.service.validation import DERIVED_PARAMS


def _create(svc, **overrides):
    """A *grid* request, which is what every test in this file is about.

    ``allow_grid`` because the grid mode is refused for a new request now --
    ``docs/measurements/2026-08-18-tile-sheet-grid.md`` -- and every assertion
    below is about the path that still builds it: the sixty-four cell geometry,
    the canny guide, the three views. The seamless modes have their own file.
    """
    kwargs = {
        "prompt": "mossy dungeon",
        "tile_size": 32,
        "view": "top_down",
        "mode": tilesheets.MODE_GRID,
        "allow_grid": True,
    }
    kwargs.update(overrides)
    return tilesheets.create_tile_sheet(svc, **kwargs)


def _png(width: int = 64, height: int = 64) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 90, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# -- the good path -----------------------------------------------------------


def test_a_good_request_writes_a_queued_row_and_no_files(svc):
    made = _create(svc, seed=42)
    row = svc.store.get(made["id"])
    assert row["kind"] == "tile_sheet"
    assert row["stage"] == "tilesheet"
    assert row["status"] == "queued"
    assert row["params"]["seed"] == 42
    assert row["params"]["base_model"] == tilesheets.TILE_SHEET_BASE_MODEL
    assert row["params"]["style_lora"] == models.PIXEL_SHEET_LORA
    assert row["params"]["control"] == "canny"
    # Nothing is on disk: the worker makes the sheet, not the door.
    assert not (svc.job_dir(made["id"]) / "input.png").exists()


def test_the_sheet_block_carries_the_geometry(svc):
    """One stored fact, derived once at the door, so the worker and anything
    reading the row later never re-derive the table and disagree."""
    made = _create(svc, tile_size=64, view="isometric")
    block = svc.store.get(made["id"])["params"]["sheet"]
    assert block["tile_w"] == 64
    assert block["tile_h"] == 32
    assert block["projection"] == "isometric"
    assert block["columns"] == 8
    assert block["rows"] == 8


def test_the_row_is_named_so_a_library_list_says_what_it_is(svc):
    made = _create(svc)
    assert svc.store.get(made["id"])["name"] == "mossy dungeon tile sheet"


def test_the_reply_says_what_it_will_produce(svc):
    made = _create(svc, tile_size=48, view="isometric")
    assert made["tiles"] == 64
    assert made["tile_w"] == 48
    assert made["tile_h"] == 24


def test_an_absent_seed_is_drawn_rather_than_left_unset(svc):
    """A stored sheet has to be reproducible, so the seed is a fact about the
    row from the moment it exists."""
    params = svc.store.get(_create(svc)["id"])["params"]
    assert isinstance(params["seed"], int)
    assert 0 <= params["seed"] <= tilesheet.MAX_SEED


# -- the prompt --------------------------------------------------------------


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_a_sheet_needs_a_prompt(svc, prompt):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, prompt=prompt)
    assert excinfo.value.field == "prompt"


def test_an_overlong_prompt_is_refused(svc):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, prompt="x" * 5000)
    assert excinfo.value.field == "prompt"


# -- the geometry ------------------------------------------------------------


@pytest.mark.parametrize("size", tilesheets.TILE_SIZES)
def test_every_offered_tile_size_is_accepted(svc, size):
    assert _create(svc, tile_size=size)["tile_w"] == size


@pytest.mark.parametrize("size", [0, 24, 128, 4096, -32])
def test_a_tile_size_off_the_menu_is_refused(svc, size):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, tile_size=size)
    assert excinfo.value.field == "tile_size"


def test_a_tile_size_that_is_not_a_number_is_a_refusal_not_a_crash(svc):
    """This door exists to turn every bad request into a sentence with a field
    on it; a bare ``int()`` over a form value would propagate a builtin the
    pane renders as an unhandled crash instead of a red field."""
    with pytest.raises(Invalid) as excinfo:
        _create(svc, tile_size="thirty two")
    assert excinfo.value.field == "tile_size"


@pytest.mark.parametrize("view", tilesheets.VIEWS)
def test_every_offered_view_is_accepted(svc, view):
    made = _create(svc, view=view)
    # The stored key is still ``projection``; only its vocabulary widened.
    assert svc.store.get(made["id"])["params"]["sheet"]["projection"] == view


def test_the_old_orthogonal_spelling_is_accepted_and_stored_as_top_down(svc):
    """A profile or a reroll from before the vocabulary widened carries it, and
    refusing one would make rerolling last week's sheet an error."""
    made = _create(svc, view="orthogonal")
    block = svc.store.get(made["id"])["params"]["sheet"]
    assert block["projection"] == "top_down"
    # 3 now: the block says which shape it describes. Version 2 is what a row
    # written before the seamless modes carries, and it has no ``mode`` key --
    # which is exactly why the number, and not the key, is what tells them
    # apart.
    assert block["version"] == 3
    assert block["mode"] == tilesheets.MODE_GRID


@pytest.mark.parametrize("view", ["hexagonal", "oblique", "", None])
def test_a_view_off_the_menu_is_refused(svc, view):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, view=view)
    # ``field="projection"`` and not ``"view"``: the refusal has to name the
    # control the user is looking at, and the form field kept its name.
    assert excinfo.value.field == "projection"


def test_oblique_is_refused_here_even_though_plotter_knows_it(svc):
    """Deliberate, not an oversight: oblique frames a cell exactly as
    top-down does, so offering it would be a third button that changed
    nothing about the picture."""
    with pytest.raises(Invalid):
        _create(svc, view="oblique")


# -- the negative prompt -----------------------------------------------------


def test_an_absent_negative_prompt_takes_the_pipelines_own_default(svc):
    params = svc.store.get(_create(svc)["id"])["params"]
    assert params["negative_prompt"] == tilesheet.SHEET_NEGATIVE_PROMPT


def test_an_empty_negative_prompt_is_honoured(svc):
    """``None`` means the form said nothing; an empty *string* is a user
    explicitly asking for no negative prompt."""
    params = svc.store.get(_create(svc, negative_prompt="")["id"])["params"]
    assert params["negative_prompt"] == ""


def test_an_overlong_negative_prompt_is_refused(svc):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, negative_prompt="x" * 5000)
    assert excinfo.value.field == "negative_prompt"


# -- the seed ----------------------------------------------------------------


def test_an_out_of_range_seed_is_refused(svc):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, seed=tilesheet.MAX_SEED + 1)
    assert excinfo.value.field == "seed"


# -- the optional reference --------------------------------------------------


def test_without_a_reference_nothing_is_written_and_no_adapter_is_named(svc):
    """The common path. An adapter in params with no image to condition on
    would reach the worker and be silently dropped."""
    made = _create(svc)
    assert "ip_adapter" not in svc.store.get(made["id"])["params"]
    assert not svc.job_dir(made["id"]).exists()


def test_a_reference_is_written_before_the_row_and_names_its_adapter(svc):
    made = _create(svc, reference=_png())
    assert (svc.job_dir(made["id"]) / "ref.png").read_bytes()[:4] == b"\x89PNG"
    assert svc.store.get(made["id"])["params"]["ip_adapter"] == "plus"


def test_an_oversized_reference_is_refused(svc):
    with pytest.raises(TooLarge):
        _create(svc, reference=b"x" * (20 * 1024 * 1024 + 1))


def test_an_undecodable_reference_names_the_control(svc):
    with pytest.raises(Invalid) as excinfo:
        _create(svc, reference=b"not an image at all")
    assert excinfo.value.field == "reference"


def test_a_refused_reference_leaves_no_row_and_no_directory(svc):
    before = len(svc.store.list())
    with pytest.raises(Invalid):
        _create(svc, reference=b"not an image at all")
    assert len(svc.store.list()) == before


# -- the weights -------------------------------------------------------------


def test_a_missing_pixel_lora_is_refused_with_something_to_install(svc):
    """The worker's own tolerance would let this finish: it logs and paints
    bare, so the job would look like one flat picture and write a sidecar
    naming a LoRA that never loaded."""
    from warlock import fetch

    lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    path = svc.config.t2i_model_root / "loras" / lora.filename
    path.unlink()
    assert not fetch.present(svc.config, "lora", lora)
    with pytest.raises(Invalid) as excinfo:
        _create(svc)
    assert excinfo.value.field == "style_lora"
    assert excinfo.value.rows


def test_the_adapter_is_only_required_when_a_reference_is_attached(svc):
    """Making it unconditional would refuse the common prompt-only request on a
    host that has everything that request actually uses."""
    adapter = models.IP_ADAPTERS["plus"]
    weight = (
        svc.config.t2i_model_root
        / adapter.dir_name
        / adapter.subfolder
        / adapter.weight_name
    )
    weight.unlink()
    # Prompt only: unaffected.
    assert _create(svc)["tiles"] == 64
    # With a reference: refused, and it names the adapter.
    with pytest.raises(Invalid) as excinfo:
        _create(svc, reference=_png())
    assert excinfo.value.field == "ip_adapter"


# -- admission ---------------------------------------------------------------


def test_the_kind_is_priced_rather_than_falling_through_to_free(svc):
    """An unlisted kind returns 0.0 from ``estimate_parts``, which makes
    admission control a no-op that refuses nothing."""
    params = svc.store.get(_create(svc)["id"])["params"]
    total, image = vram.estimate_parts("tile_sheet", "tilesheet", params)
    assert total > 0
    assert image > 0


def test_an_attached_reference_costs_its_encoder(svc):
    """The IP-Adapter is params-gated on this kind, so the gate has to actually
    move the number or it is a comment rather than a term."""
    plain = svc.store.get(_create(svc)["id"])["params"]
    conditioned = svc.store.get(_create(svc, reference=_png())["id"])["params"]
    assert vram.estimate("tile_sheet", "tilesheet", conditioned) > vram.estimate(
        "tile_sheet", "tilesheet", plain
    )


def test_a_sheet_never_charges_for_trellis_when_it_has_the_card_alone(svc):
    """A tile sheet has no mesh anywhere near it."""
    params = svc.store.get(_create(svc)["id"])["params"]
    exclusive = vram.estimate("tile_sheet", "tilesheet", params, exclusive=True)
    coexist = vram.estimate("tile_sheet", "tilesheet", params)
    assert exclusive < coexist


# -- the options contract ----------------------------------------------------


def test_the_options_reply_carries_the_menus_and_the_grid(svc):
    options = tilesheets.tile_sheet_options()
    assert options["tile_sizes"] == list(tilesheets.TILE_SIZES)
    assert options["views"] == list(tilesheets.VIEWS)
    assert set(options["view_labels"]) == set(tilesheets.VIEWS)
    assert options["columns"] == 8
    assert options["rows"] == 8
    assert options["tiles"] == 64
    assert options["defaults"]["tile_size"] in tilesheets.TILE_SIZES


def test_the_options_reply_states_the_finished_size_of_every_choice(svc):
    """So the pane can say "a 512x256 sheet" without doing the arithmetic
    twice."""
    options = tilesheets.tile_sheet_options()
    entry = next(row for row in options["sizes"] if row["key"] == 64)
    assert entry["views"]["top_down"] == {
        "tile_w": 64,
        "tile_h": 64,
        "sheet_w": 512,
        "sheet_h": 512,
    }
    assert entry["views"]["three_quarter"] == {
        "tile_w": 64,
        "tile_h": 64,
        "sheet_w": 512,
        "sheet_h": 512,
    }
    assert entry["views"]["isometric"] == {
        "tile_w": 64,
        "tile_h": 32,
        "sheet_w": 512,
        "sheet_h": 256,
    }


# -- the drift pins ----------------------------------------------------------


def test_the_tile_sizes_match_the_pipeline():
    assert tilesheets.TILE_SIZES == tilesheet.TILE_SIZES


def test_the_views_match_the_pipeline():
    assert tilesheets.VIEWS == tilesheet.VIEWS
    assert tilesheets.DEFAULT_VIEW in tilesheets.VIEWS
    # The alias table is restated here too, and for the same reason.
    for stored, canonical in tilesheets.LEGACY_VIEWS.items():
        assert tilesheet.normalize_view(stored) == canonical


def test_every_offered_size_and_view_builds_a_geometry():
    """The door hands its own two lists straight to ``tilesheet.geometry``
    after checking them, so a pair this module accepts and the pipeline refuses
    would be an unhandled ValueError past the point of no return."""
    for size in tilesheets.TILE_SIZES:
        for view in tilesheets.VIEWS:
            assert tilesheet.geometry(size, view).tiles == 64


def test_the_palette_size_is_the_measured_one():
    """docs/measurements/2026-08-17-ground-reduction.md: occupancy at 32 was
    saturated for even a two-material set, and 64 stayed above 95%."""
    assert tilesheets.SHEET_COLORS == 64


def test_the_worker_recorded_report_is_stripped_on_a_rerun():
    """Anything the worker records about its artifacts joins DERIVED_PARAMS, or
    a reroll wears a stale verdict."""
    assert "sheet_report" in DERIVED_PARAMS
