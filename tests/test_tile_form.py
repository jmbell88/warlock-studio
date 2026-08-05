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
    colour, text = inspector.seam_verdict(
        {"worst": 3.4, "seamless": False, "threshold": seam.SEAM_MAX}
    )
    assert colour == theme.WARN
    assert "3.40" in text and f"{seam.SEAM_MAX:.2f}" in text
