"""The tile's two UI halves: the 2D pane's switch and the inspector's verdict.

Everything asserted here is a pure function of the form dict or of the stored
report, because the panes draw what these return -- the same split the
composed-prompt helpers use.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.pipelines import seam
from warlock.studio import theme
from warlock.studio.panes import inspector, settings_2d
from warlock.studio.state import default_form_2d


def test_a_new_form_makes_references():
    assert default_form_2d()["output"] == "reference"


def test_the_default_form_submits_a_reference():
    form = default_form_2d()
    form["prompt"] = "a barrel"
    assert settings_2d.submit_kwargs(form)["output"] == "reference"


def test_switching_to_tile_changes_what_is_submitted():
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["output"] = "tile"
    assert settings_2d.submit_kwargs(form)["output"] == "tile"


def test_a_tile_keeps_the_checkpoint_it_is_drawn_with():
    # Model identity: a tile still has to say which base and which style LoRA
    # drew it.
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["base_model"] = "turbo"
    form["output"] = "tile"
    assert settings_2d.submit_kwargs(form)["guidance_fields"]["base_model"] == "turbo"


def test_switching_back_keeps_what_was_typed():
    form = default_form_2d()
    form["base_model"] = "turbo"
    form["output"] = "tile"
    form["output"] = "reference"
    assert settings_2d.submit_kwargs(form)["guidance_fields"]["base_model"] == "turbo"


def test_a_tile_still_needs_a_prompt():
    form = default_form_2d()
    form["output"] = "tile"
    assert settings_2d.validate(form)


# -- the preview the pane asks for -----------------------------------------


class _PreviewCtx:
    """Just enough of AppCtx for ``_preview``'s off-thread request.

    The drawing half needs a GL context; the request does not, and the request
    is the half that can be wrong -- a tile previewed through the object
    template shows framing the job will never use.
    """

    def __init__(self, form):
        self.state = SimpleNamespace(form_2d=form, preview_dirty_at=1e-9, preview=None)
        self.svc = object()
        self.calls: list[tuple[tuple, dict]] = []

    def submit(self, _key, _fn, *args, **kwargs):
        self.calls.append((args, kwargs))
        return True


def test_the_preview_of_a_tile_asks_for_a_tile():
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["output"] = "tile"
    ctx = _PreviewCtx(form)
    settings_2d._preview(ctx)
    assert ctx.calls and ctx.calls[0][1]["tile"] is True


def test_the_preview_of_an_object_does_not():
    form = default_form_2d()
    form["prompt"] = "a barrel"
    ctx = _PreviewCtx(form)
    settings_2d._preview(ctx)
    assert ctx.calls and ctx.calls[0][1]["tile"] is False


# -- the verdict the inspector draws ---------------------------------------


def test_a_job_with_no_seam_report_gets_no_verdict():
    # Every mesh and every ordinary reference: the section is not merely empty,
    # it is absent. A non-dict is the same answer -- params is a JSON blob and
    # a hand-mangled one must not reach a .get().
    assert inspector.seam_verdict(None) is None
    assert inspector.seam_verdict("2.0") is None


def test_a_seamless_tile_says_so_in_the_ok_colour():
    colour, text = inspector.seam_verdict(
        {"worst": 1.1, "seamless": True, "threshold": seam.SEAM_MAX}
    )
    assert colour == theme.OK
    assert "seamless" in text and "1.10" in text


def test_a_failing_tile_quotes_the_threshold_it_went_over():
    # The ratio alone is uncalibratable by eye; the number it was compared
    # against is what makes it a verdict rather than a statistic.
    # Derived from the constant rather than written out: a fixture holding a
    # number that used to be over the threshold and no longer is still passes
    # both assertions below while depicting a verdict the app cannot produce.
    worst = seam.SEAM_MAX + 1.4
    colour, text = inspector.seam_verdict(
        {"worst": worst, "seamless": False, "threshold": seam.SEAM_MAX}
    )
    assert colour == theme.WARN
    assert f"{worst:.2f}" in text and f"{seam.SEAM_MAX:.2f}" in text


def test_a_row_written_before_the_statistic_changed_is_worded_in_its_own_terms():
    """The two tests above are that case, and this one says so out loud.

    Neither carries a ``metric`` field, because no row written before
    2026-08-30 does. Those rows hold an edge-against-mean-grain ratio judged
    against 3.5, and the pane has to keep describing them that way: relabelling
    a stored 4.9 as a seam-against-worst-join number would be reporting a
    measurement that was never taken. ``docs/measurements/2026-08-30-seam-dominance.md``
    R8 -- nothing on disk is reinterpreted.
    """
    _colour, text = inspector.seam_verdict(
        {"worst": 4.9, "seamless": False, "threshold": seam.SEAM_MAX}
    )
    assert "edge/grain" in text and "4.90" in text


def test_a_row_judged_by_dominance_quotes_the_dominance_number():
    """And a new row must not be worded with the old vocabulary either.

    The failure this refuses is quiet and total: both statistics produce values
    in the same range, so a dominance row read as an edge/grain one shows a
    plausible number with the opposite meaning -- 0.94 is an excellent tile by
    one and a suspiciously good one by the other. The ratio stays on the row
    and is deliberately *not* the number quoted.
    """
    row = {
        "worst": 8.6,
        "dominance": 0.86,
        "metric": "dominance",
        "seamless": True,
        "threshold": seam.SEAM_DOMINANCE_MAX,
    }
    colour, text = inspector.seam_verdict(row)
    assert colour == theme.OK
    assert "0.86" in text and "8.6" not in text
    assert "edge/grain" not in text

    over = dict(row, dominance=1.4, seamless=False)
    colour, text = inspector.seam_verdict(over)
    assert colour == theme.WARN
    assert "1.40" in text and f"{seam.SEAM_DOMINANCE_MAX:.2f}" in text


# -- a tile is an editable image, and a repeating one -------------------------


def test_a_tile_can_be_opened_in_the_inker(svc):
    """The albedo is the asset, so it is as editable as a reference's picture.

    The pane asks ``inker_mode.can_edit_job`` and the service enforces
    ``files.EDITABLE_STAGES``; this pins that they agree, because a button the
    service refuses is worse than no button.
    """
    from warlock.service import files as svc_files
    from warlock.studio import inker_mode

    job = {"id": "a" * 12, "stage": "tile", "status": "done", "files": ["input.png"]}
    assert inker_mode.can_edit_job(None, job)
    assert "tile" in svc_files.EDITABLE_STAGES


def test_a_model_job_is_still_not_editable():
    """A model's input.png is the picture it was reconstructed *from*: editing
    it changes nothing about the mesh on disk while invalidating the recipe
    that describes it."""
    from warlock.studio import inker_mode

    job = {"id": "a" * 12, "stage": "model", "status": "done", "files": ["input.png"]}
    assert not inker_mode.can_edit_job(None, job)


def test_editing_a_tile_re_measures_its_seam_rather_than_its_composition(svc):
    """The split ``queue._generate`` already makes, applied to the edit path.

    A composition report is entirely about where a subject sits and a tile has
    none. A seam ratio is the thing an edit is most likely to break -- a brush
    stroke does not wrap -- so the generated verdict must not survive one.
    """
    import io

    import numpy as np
    from PIL import Image

    from warlock.service import files as svc_files

    job_id = svc.store.create("text", "stone", {"seed": 1}, stage="tile", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    # A wrapping triangle wave: seamless, so the generated verdict is "tiles".
    ramp = np.abs((np.arange(64) * 2.0 / 64) - 1.0)
    arr = np.stack([np.tile((ramp * 255).astype(np.uint8), (64, 1))] * 3, axis=-1)
    Image.fromarray(arr, "RGB").save(job_dir / "input.png")
    svc.store.merge_params(job_id, {"seam_report": seam.report(job_dir / "input.png")})
    assert svc.store.get(job_id)["params"]["seam_report"]["seamless"] is True

    # Now hand-edit it into a left-to-right ramp, whose wrap edge is a 251-level
    # cliff against interior steps of 4 -- the seam is the worst join in the
    # frame by a factor of sixty.
    #
    # A half/half block would be the shorter way to write "a hard step" and it
    # is the wrong picture: two blocks make a *stripe*, which repeats without
    # introducing any join it does not already contain, and
    # ``docs/measurements/2026-08-30-seam-dominance.md``'s statistic correctly
    # passes it. The edit this test is about is one that stops the tile
    # wrapping, not one that adds an edge.
    ramp_edit = (np.arange(64) * 255.0 / 64).astype(np.uint8)
    edited = np.stack([np.tile(ramp_edit, (64, 1))] * 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(edited, "RGB").save(buf, "PNG")
    svc_files.save_edited_image(svc, job_id, buf.getvalue())

    params = svc.store.get(job_id)["params"]
    assert params["hand_edited"] is True
    assert params["seam_report"]["seamless"] is False
    # And no composition verdict was invented for something with no subject.
    assert "reference_report" not in params


def test_the_tiled_toggle_is_offered_for_a_tile_at_the_reference_stage_only():
    from types import SimpleNamespace

    from warlock.studio.panes import overlay

    def ctx(stage, mode="create"):
        return SimpleNamespace(state=SimpleNamespace(mode=mode, create_stage=stage))

    done_tile = {"stage": "tile", "status": "done"}
    assert overlay.shows_tiled(ctx("reference"), done_tile)
    # Not at the Mesh stage: the thing on screen there is a mesh.
    assert not overlay.shows_tiled(ctx("mesh"), done_tile)
    # Nor in a mode that is not Create at all, whatever stage was last left.
    assert not overlay.shows_tiled(ctx("reference", mode="inker"), done_tile)
    assert not overlay.shows_tiled(ctx("reference"), {"stage": "reference", "status": "done"})
    assert not overlay.shows_tiled(ctx("reference"), {"stage": "tile", "status": "running"})
    assert not overlay.shows_tiled(ctx("reference"), None)


def test_the_tiled_preview_is_off_by_default():
    """Stated rather than assumed: every other view of an asset shows one cell,
    so a viewport that silently showed four would make the texture look a
    quarter of its size."""
    from warlock.studio.state import AppState

    assert AppState().tile_preview is False
