"""``rerender_charsheet``: the door that re-renders some runs of a sheet.

It mints a new sheet rather than rewriting one -- sheets are write-once under a
fresh id -- and it copies its pixel settings from the row that made the sheet
it is re-rendering, so the new cells match the ones they land beside.
"""

from __future__ import annotations

import json
import time

import pytest

from warlock.pipelines import charsheet
from warlock.service import troupe as svc_troupe
from warlock.service.errors import Invalid, NotFound


def _rigged_mesh(svc, template="humanoid"):
    job_id = svc.store.create("image", "a ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


def _published(svc, job_id, *, logical_size=32, troupe=True):
    """A sheet on disk plus the finished row that produced it."""
    made = svc_troupe.create_charsheet(svc, job_id, logical_size=logical_size)
    sheet_id = made["sheet_id"]
    job_dir = svc.job_dir(job_id)
    sheets = job_dir / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    (sheets / f"{sheet_id}.png").write_bytes(b"fake-png")
    layout = charsheet.resolve_layout().as_dict()
    record = {
        "id": sheet_id,
        "image": f"{sheet_id}.png",
        "created": time.time(),
        "cells": [],
    }
    if troupe:
        record["troupe"] = layout
    (sheets / f"{sheet_id}.json").write_text(json.dumps(record), "utf-8")
    svc.store.set_status(made["id"], "done")
    return made["id"], sheet_id


def _runs(n=1):
    return [
        {"animation": animation, "direction": direction}
        for animation, direction, *_ in charsheet.spans()[:n]
    ]


# -- the happy path -----------------------------------------------------------


def test_a_re_render_mints_a_new_sheet_carrying_the_runs_and_the_base(svc):
    job_id = _rigged_mesh(svc)
    row_id, sheet_id = _published(svc, job_id)

    made = svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(2))
    params = svc.store.get(made["id"])["params"]

    assert params["base_sheet"] == sheet_id
    assert len(params["subset"]) == 2
    assert params["sheet_id"] != sheet_id, "a re-render is a new sheet, never a rewrite"
    assert params["source_job"] == job_id
    assert svc.store.get(row_id)["params"]["sheet_id"] == sheet_id, "the old row stands"


def test_the_pixel_settings_are_copied_rather_than_accepted(svc):
    """The new cells have to be reduced, quantised and outlined exactly as the
    ones they will sit beside were. Any option this door took separately would
    be one a user could set to something else."""
    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id, logical_size=64)

    made = svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))
    params = svc.store.get(made["id"])["params"]

    assert params["logical_size"] == 64
    for field in ("colors", "outline", "reduce_mode", "dither", "palette"):
        assert field in params, field


def test_the_previous_runs_answers_about_its_own_output_are_stripped(svc):
    """A fresh row must not wear the last run's report -- ``DERIVED_PARAMS``'
    rule, applied by the door that mints rather than by the rerun path."""
    job_id = _rigged_mesh(svc)
    row_id, sheet_id = _published(svc, job_id)
    svc.store.set_params(
        row_id,
        {
            **svc.store.get(row_id)["params"],
            "cells": 256,
            "rendered_cells": 12,
            "pixel_report": {"palette": "derived"},
        },
    )

    made = svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))
    params = svc.store.get(made["id"])["params"]

    for derived in ("cells", "rendered_cells", "pixel_report"):
        assert derived not in params, derived


def test_the_runs_come_back_named(svc):
    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id)

    made = svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(3))
    assert len(made["runs"]) == 3
    assert all({"animation", "direction"} == set(run) for run in made["runs"])


def test_a_name_may_be_given_and_is_capped(svc):
    from warlock import rigging

    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id)

    made = svc_troupe.rerender_charsheet(
        svc, job_id, sheet_id=sheet_id, subset=_runs(1), name="walk fix"
    )
    assert svc.store.get(made["id"])["params"]["name"] == "walk fix"

    with pytest.raises(Invalid) as caught:
        svc_troupe.rerender_charsheet(
            svc,
            job_id,
            sheet_id=sheet_id,
            subset=_runs(1),
            name="x" * (rigging.MAX_SHEET_NAME + 1),
        )
    assert caught.value.field == "name"


# -- the refusals, each with an address ---------------------------------------


def test_every_refusal_names_the_field_it_is_about(svc):
    """The service layer's rule: a refusal carries a ``field`` so the form has
    somewhere to ring it."""
    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id)

    with pytest.raises(Invalid) as bad_id:
        svc_troupe.rerender_charsheet(svc, job_id, sheet_id="not-an-id", subset=_runs(1))
    assert bad_id.value.field == "sheet_id"

    with pytest.raises(Invalid) as bad_runs:
        svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=[])
    assert bad_runs.value.field == "subset"


def test_a_sheet_that_is_not_on_disk_is_refused(svc):
    from warlock import rigging

    job_id = _rigged_mesh(svc)
    with pytest.raises(NotFound) as caught:
        svc_troupe.rerender_charsheet(
            svc, job_id, sheet_id=rigging.new_id(), subset=_runs(1)
        )
    assert caught.value.field == "sheet_id"


def test_a_plain_sheet_has_no_runs_to_re_render(svc):
    """An ordinary pose sheet carries no ``troupe`` block, so it has no
    animation-and-direction vocabulary to name a subset in."""
    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id, troupe=False)

    with pytest.raises(Invalid, match="not a character sheet"):
        svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))


def test_a_sheet_whose_settings_are_gone_says_so_and_says_what_to_do(svc):
    """Honest rather than guessing at defaults: without the original settings
    the new cells could not be made to match the ones beside them."""
    job_id = _rigged_mesh(svc)
    row_id, sheet_id = _published(svc, job_id)
    svc.store.delete(row_id)

    with pytest.raises(Invalid, match="build a new sheet instead"):
        svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=_runs(1))


def test_every_run_at_once_is_refused_as_a_full_render(svc):
    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id)
    everything = [
        {"animation": animation, "direction": direction}
        for animation, direction, *_ in charsheet.spans()
    ]

    with pytest.raises(Invalid) as caught:
        svc_troupe.rerender_charsheet(svc, job_id, sheet_id=sheet_id, subset=everything)
    assert caught.value.field == "subset"


def test_a_run_the_sheet_does_not_have_is_refused_by_name(svc):
    job_id = _rigged_mesh(svc)
    _row_id, sheet_id = _published(svc, job_id)

    with pytest.raises(Invalid, match="moonwalk"):
        svc_troupe.rerender_charsheet(
            svc,
            job_id,
            sheet_id=sheet_id,
            subset=[{"animation": "moonwalk", "direction": "front"}],
        )
