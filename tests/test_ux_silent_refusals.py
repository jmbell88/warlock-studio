"""Refusals that happened and said nothing.

Three of them, and they share one shape: an action reachable from more than one
door, where the *button* door renders its refusal as part of the pane and the
other doors return in silence. A disabled button with red text above it is a
good refusal; the same code reached by Ctrl+Enter, by the command palette, or
by a menu item is the same refusal with the explanation removed.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio import widgets
from warlock.studio.panes import settings_2d, settings_3d
from warlock.studio.state import default_form_2d


class _State:
    def __init__(self) -> None:
        self.field_errors: dict[str, str] = {}
        self.toasts: list[tuple[str, str]] = []
        self.prompts: list[str] = []

    def note_field_error(self, field_name: str, message: str) -> bool:
        if not field_name or not message:
            return False
        self.field_errors[field_name] = message
        return True

    def clear_field_errors(self) -> None:
        self.field_errors.clear()

    def remember_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)


class _Ctx:
    def __init__(self) -> None:
        self.state = _State()
        self.submitted: list[str] = []

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.state.toasts.append((text, level))

    def submit(self, key: str, run: Any, *a: Any, **k: Any) -> bool:
        self.submitted.append(key)
        return True


# --- a validation problem knows its control ----------------------------------


def test_a_problem_is_still_a_string():
    """``widgets.Problem`` is a ``str`` subclass so the aggregate block above
    Generate (``imgui.text_wrapped(problem)``) and every existing comparison
    keep working untouched. If it stopped being one, the pane would render
    nothing and no test would notice."""
    problem = widgets.Problem("A prompt is required.", "prompt")
    assert problem == "A prompt is required."
    assert isinstance(problem, str)
    assert problem.field == "prompt"
    assert widgets.Problem("no control").field == ""


def test_every_2d_problem_that_can_name_a_control_does():
    """The field is what turns a toast into a pointer. A problem about the
    prompt that carries no field rings nothing, which is the state all of them
    were in."""
    form = dict(default_form_2d())
    form["prompt"] = ""
    form["count"] = 999
    problems = settings_2d.validate(form)
    assert problems, "the form is deliberately invalid"
    assert {p.field for p in problems} == {"prompt", "count"}


# --- Ctrl+Enter and the palette --------------------------------------------


def test_generate_says_why_it_refused_instead_of_returning_in_silence():
    """Ctrl+Enter and the palette's Generate call straight into ``generate``.
    The red block above the button is the button's own feedback and neither of
    those doors draws it, so a refused Ctrl+Enter did nothing observable."""
    ctx = _Ctx()
    form = dict(default_form_2d())
    form["prompt"] = "   "
    settings_2d.generate(ctx, form)
    assert ctx.submitted == []
    assert ctx.state.toasts and ctx.state.toasts[0][1] == "warn"
    assert "prompt" in ctx.state.toasts[0][0].lower()
    # And the ring, so the user is pointed at the control as well as told.
    assert ctx.state.field_errors.get("prompt")


def test_promote_says_why_it_refused_too():
    """Same door, other pane: Ctrl+Enter in 3D mode with nothing selected used
    to be indistinguishable from a shortcut that is not bound."""
    ctx = _Ctx()
    settings_3d.promote(ctx, None, {})
    assert ctx.state.toasts and ctx.state.toasts[0][1] == "warn"
    assert "reference" in ctx.state.toasts[0][0].lower()


def test_a_refusal_about_the_library_rings_no_widget():
    """"Choose a reference first" is about the library, not about a control in
    the promotion form. Ringing one would point at the wrong thing, which is
    why ``note_field_error`` treats an empty field as toast-only."""
    ctx = _Ctx()
    settings_3d.promote(ctx, None, {})
    assert ctx.state.field_errors == {}


def test_a_refusal_clears_the_previous_one_before_filing_its_own():
    """A new submit is judged on its own: the rings from the last one describe
    a request that no longer exists, and leaving them up has the app pointing
    at a control while it works on the value in it."""
    ctx = _Ctx()
    ctx.state.field_errors["style_lora"] = "stale, from a form ago"
    settings_2d.refuse(ctx, [widgets.Problem("A prompt is required.", "prompt")])
    assert set(ctx.state.field_errors) == {"prompt"}


def test_only_the_first_problem_is_toasted_and_the_rest_are_rings():
    """Four toasts is a wall. The first says what is wrong and the rings mark
    every control that has to change, which is the same pair of facts the
    button path shows as a list plus a highlight."""
    ctx = _Ctx()
    settings_2d.refuse(
        ctx,
        [
            widgets.Problem("A prompt is required.", "prompt"),
            widgets.Problem("References must be between 1 and 8.", "count"),
        ],
    )
    assert len(ctx.state.toasts) == 1
    assert ctx.state.toasts[0][0] == "A prompt is required."
    assert set(ctx.state.field_errors) == {"prompt", "count"}


# --- "Fix matte" -------------------------------------------------------------


class _Recorder:
    """A ``_cut_matte`` that answers as told, and a service it never reaches."""

    def __init__(self, applied: bool) -> None:
        self.applied = applied

    def __call__(self, svc: Any, job_id: str, doc: Any) -> bool:
        return self.applied


@pytest.mark.parametrize("applied", [True, False])
def test_fix_matte_records_what_happened_rather_than_swallowing_it(monkeypatch, applied):
    """``_cut_matte`` log-and-swallows, which is right for *Edit* -- the matte
    is a bonus there and must not cost the user the reference they asked for.
    But "Fix matte" is a command whose entire content is the cutout, so the
    swallow left that menu item opening a tab indistinguishable from the
    ordinary one and saying nothing at all.

    The fix is one pair of keys in the task result, which the completion branch
    turns into a toast. Recorded either way: "it worked" and "it was not asked
    for" are different facts, and a single ``matte_failed`` flag conflates them.
    """
    from warlock.studio import inker, inker_mode

    monkeypatch.setattr(inker_mode, "_cut_matte", _Recorder(applied))
    doc = inker.Document.blank(4, 4)
    monkeypatch.setattr(inker.Document, "load", staticmethod(lambda path: doc))

    from warlock.service import files as svc_files

    monkeypatch.setattr(svc_files, "inker_working_path", lambda svc, job_id: None)
    monkeypatch.setattr(
        svc_files, "inker_working_status", lambda svc, job_id: {"fresh": False}
    )
    monkeypatch.setattr(svc_files, "reference_edit_status", lambda svc, job_id: {})

    class _Svc:
        store = type("S", (), {"get": staticmethod(lambda job_id: {})})()

        @staticmethod
        def job_dir(job_id):
            from pathlib import Path

            return Path(".")

    out = inker_mode._load_job(_Svc(), "job", matte=True)
    assert out["matte_requested"] is True
    assert out["matte_applied"] is applied


def test_a_matte_nobody_asked_for_records_nothing():
    """The plain Edit path must stay exactly as it was: no keys, so the
    completion branch cannot toast about a cutout that was never requested."""
    import inspect

    from warlock.studio import inker_mode

    source = inspect.getsource(inker_mode._load_job)
    body = source.split("if matte:")[0]
    assert "matte_requested" not in body
    branch = inspect.getsource(inker_mode.on_task_done)
    assert 'result.get("matte_requested")' in branch


# --- Generate knows what this host has downloaded ----------------------------


def _rows(*entries):
    return [
        {"row_key": key, "label": label, "present": present}
        for key, label, present in entries
    ]


def test_a_selected_model_that_is_not_downloaded_is_named_before_the_submit():
    """The service refuses this at the door -- and by then the job has queued,
    waited behind whatever else is running, and come back as an error row
    carrying a ``base_model`` the user has to work out was never downloaded.
    Saying it here costs one dict lookup on the frame thread."""
    ctx = _Ctx()
    ctx.model_rows = _rows(("base:sdxl_cfg", "SDXL 1.0", False))
    form = dict(default_form_2d())
    form["prompt"] = "a barrel"
    form["base_model"] = "sdxl_cfg"
    problem = settings_2d.weights_problem(ctx, form)
    assert problem is not None
    assert "SDXL 1.0" in problem
    assert problem.field == "base_model"


def test_an_empty_snapshot_says_nothing_rather_than_everything():
    """``model_gate``'s doctrine. A headless ctx, or the first frame before the
    presence answers land, must not read as "nothing is installed" -- which
    would lock Generate on a fully-installed host."""
    ctx = _Ctx()
    ctx.model_rows = []
    form = dict(default_form_2d())
    form["base_model"] = "sdxl_cfg"
    assert settings_2d.weights_problem(ctx, form) is None


def test_a_row_the_snapshot_has_never_heard_of_is_skipped():
    """Same rule one level down: unknown is not absent."""
    ctx = _Ctx()
    ctx.model_rows = _rows(("base:something_else", "Other", True))
    form = dict(default_form_2d())
    form["base_model"] = "sdxl_cfg"
    assert settings_2d.weights_problem(ctx, form) is None


def test_a_present_model_is_no_problem():
    ctx = _Ctx()
    ctx.model_rows = _rows(("base:sdxl_cfg", "SDXL 1.0", True))
    form = dict(default_form_2d())
    form["base_model"] = "sdxl_cfg"
    assert settings_2d.weights_problem(ctx, form) is None


def test_the_optional_selections_are_checked_too():
    """A style LoRA is the one that fails *silently* in the worker: the loader
    skips a missing style adapter, so the job finishes, looks wrong, and its
    row claims a style that never ran -- and then joins the findings corpus as
    evidence about it."""
    ctx = _Ctx()
    ctx.model_rows = _rows(
        ("base:sdxl_cfg", "SDXL 1.0", True), ("lora:ps1", "PS1 style", False)
    )
    form = dict(default_form_2d())
    form["base_model"] = "sdxl_cfg"
    form["style_lora"] = "ps1"
    problem = settings_2d.weights_problem(ctx, form)
    assert problem is not None and problem.field == "style_lora"


def test_a_form_problem_is_reported_before_a_download_one():
    """An empty prompt is something the user can fix in the box in front of
    them; a missing checkpoint is a trip to Settings. Leading with the second
    would send them away to come back to the same refusal."""
    ctx = _Ctx()
    ctx.model_rows = _rows(("base:sdxl_cfg", "SDXL 1.0", False))
    form = dict(default_form_2d())
    form["prompt"] = ""
    form["base_model"] = "sdxl_cfg"
    settings_2d.generate(ctx, form)
    assert ctx.submitted == []
    assert "prompt" in ctx.state.toasts[0][0].lower()
