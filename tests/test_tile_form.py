"""The tile's two UI halves: the 2D pane's switch and the inspector's verdict.

Everything asserted here is a pure function of the form dict or of the stored
report, because the panes draw what these return -- the same split the
composed-prompt helpers use.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.pipelines import prompt as prompt_lib
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


def test_a_tile_form_only_draws_the_surface_guidance_groups():
    form = default_form_2d()
    form["output"] = "tile"
    shown = {f for _title, fields in settings_2d.guidance_groups(form) for f in fields}
    assert shown <= set(prompt_lib.TILE_FIELDS)
    assert "category" not in shown
    assert "material" in shown


def test_an_object_form_draws_every_group():
    form = default_form_2d()
    shown = {f for _title, fields in settings_2d.guidance_groups(form) for f in fields}
    assert "category" in shown and "material" in shown


def test_a_tile_does_not_carry_subject_guidance_it_cannot_use():
    # The fields stay in the form -- switching back must not lose them -- but
    # they must not reach a submit that will ignore them, or the job row claims
    # a taxonomy that did not touch the prompt.
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["category"] = "weapon"
    form["material"] = "stone"
    form["output"] = "tile"

    fields = settings_2d.submit_kwargs(form)["guidance_fields"]

    assert "category" not in fields
    assert fields["material"] == "stone"


def test_a_tile_keeps_the_checkpoint_it_is_drawn_with():
    # Model identity is not subject taxonomy: a tile still has to say which
    # base and which style LoRA drew it, and neither is in TILE_FIELDS.
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["base_model"] = "turbo"
    form["output"] = "tile"
    assert settings_2d.submit_kwargs(form)["guidance_fields"]["base_model"] == "turbo"


def test_a_tile_does_not_submit_the_platform_prompt_fragment():
    # platform is a hint about how much detail to draw *an object* with; the
    # tile prompt compiler discards it, so submitting it would be a claim the
    # picture cannot support.
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["platform"] = "pc"
    form["output"] = "tile"
    assert "platform" not in settings_2d.submit_kwargs(form)["guidance_fields"]


def test_switching_back_keeps_what_was_typed():
    form = default_form_2d()
    form["category"] = "weapon"
    form["output"] = "tile"
    form["output"] = "reference"
    assert settings_2d.submit_kwargs(form)["guidance_fields"]["category"] == "weapon"


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

    # Now hand-edit it into a hard step, which is exactly a seam.
    edited = np.zeros((64, 64, 3), dtype=np.uint8)
    edited[:, 32:] = 255
    buf = io.BytesIO()
    Image.fromarray(edited, "RGB").save(buf, "PNG")
    svc_files.save_edited_image(svc, job_id, buf.getvalue())

    params = svc.store.get(job_id)["params"]
    assert params["hand_edited"] is True
    assert params["seam_report"]["seamless"] is False
    # And no composition verdict was invented for something with no subject.
    assert "reference_report" not in params


def test_the_tiled_toggle_is_offered_for_a_tile_in_2d_and_nowhere_else():
    from types import SimpleNamespace

    from warlock.studio.panes import overlay

    def ctx(mode):
        return SimpleNamespace(state=SimpleNamespace(mode=mode))

    done_tile = {"stage": "tile", "status": "done"}
    assert overlay.shows_tiled(ctx("2d"), done_tile)
    # Not in 3D: the thing on screen there is a mesh.
    assert not overlay.shows_tiled(ctx("3d"), done_tile)
    assert not overlay.shows_tiled(ctx("2d"), {"stage": "reference", "status": "done"})
    assert not overlay.shows_tiled(ctx("2d"), {"stage": "tile", "status": "running"})
    assert not overlay.shows_tiled(ctx("2d"), None)


def test_the_tiled_preview_is_off_by_default():
    """Stated rather than assumed: every other view of an asset shows one cell,
    so a viewport that silently showed four would make the texture look a
    quarter of its size."""
    from warlock.studio.state import AppState

    assert AppState().tile_preview is False
