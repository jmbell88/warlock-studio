"""A profile's anchor becomes the job's IP-Adapter reference.

The pane's rule everywhere else is that a manual attachment wins: the anchor
is what the *set* has in common, and the reference the user just dropped is
what this one asset needs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_profiles_anchor import PNG, FakeSettings

from warlock.studio import profiles
from warlock.studio.panes import settings_2d
from warlock.studio.state import default_form_2d


@pytest.fixture
def ctx(tmp_path):
    return SimpleNamespace(
        settings=FakeSettings(),
        svc=SimpleNamespace(config=SimpleNamespace(data_dir=tmp_path)),
    )


def _with_anchor(ctx, scale=0.7):
    profiles.save_profile(ctx.settings, "house", {})
    profiles.set_anchor(ctx.settings, ctx.svc.config, "house", PNG, scale=scale)
    profiles.set_active(ctx.settings, "house")


def test_no_profile_means_the_kwargs_are_untouched(ctx):
    form = default_form_2d()
    kwargs = settings_2d.submit_kwargs(form)
    before = dict(kwargs)

    assert settings_2d.anchor_kwargs(ctx, form, kwargs) == ""
    assert kwargs == before


def test_an_anchor_supplies_the_reference_path_and_the_adapter(ctx):
    _with_anchor(ctx)
    form = default_form_2d()
    kwargs = settings_2d.submit_kwargs(form)

    path = settings_2d.anchor_kwargs(ctx, form, kwargs)

    assert path
    with open(path, "rb") as fh:
        assert fh.read() == PNG
    assert kwargs["guidance_fields"]["ip_adapter"] == profiles.ANCHOR_ADAPTER
    assert kwargs["ip_scale"] == 0.7


def test_a_manual_reference_wins_over_the_anchor(ctx):
    _with_anchor(ctx)
    form = default_form_2d()
    form["ref_path"] = "C:/somewhere/else.png"
    kwargs = settings_2d.submit_kwargs(form)

    assert settings_2d.anchor_kwargs(ctx, form, kwargs) == ""
    assert "ip_adapter" not in kwargs["guidance_fields"]


def test_a_profile_without_an_anchor_changes_nothing(ctx):
    profiles.save_profile(ctx.settings, "house", {})
    profiles.set_active(ctx.settings, "house")
    form = default_form_2d()
    kwargs = settings_2d.submit_kwargs(form)

    assert settings_2d.anchor_kwargs(ctx, form, kwargs) == ""


def test_generate_reads_the_anchor_on_a_task_thread(ctx, monkeypatch):
    # The same contract the manual picker follows: the form is read on the
    # frame thread because it is UI state, the file inside the task because a
    # large one would freeze the window.
    _with_anchor(ctx)
    seen = {}
    monkeypatch.setattr(
        settings_2d.svc_jobs, "create_job", lambda svc, **kw: seen.update(kw) or "id"
    )
    submitted = []
    # A real ``AppState``: ``generate`` also clears the field-error rings
    # (UX.md Phase 3), and a namespace carrying only the three attributes this
    # test happens to need grows a stub every time the pane learns a method.
    from warlock.studio.state import AppState

    ctx.state = AppState()
    ctx.state.form_2d = default_form_2d()
    ctx.submit = lambda key, fn, *a, **k: (submitted.append((key, fn)), True)[1]
    ctx.state.form_2d["prompt"] = "a barrel"

    settings_2d.generate(ctx, ctx.state.form_2d)

    assert seen == {}  # nothing read yet
    _key, run = submitted[0]
    run()
    assert seen["reference"] == PNG
    assert seen["ip_scale"] == 0.7
