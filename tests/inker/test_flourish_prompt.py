"""The prompt field: words become a pending recipe through the keyword mapper,
or through the local text model when one is present -- and every answer of
the model goes through the same clamp."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from types import SimpleNamespace

import pytest

from warlock import doctor, winjob
from warlock.pipelines import recipe_worker
from warlock.studio import inker, inker_flourish, inker_mode, inker_ops, inker_state
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import keywords, presets
from warlock.studio.tasks import Done


class _Ctx:
    def __init__(self, config=None) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.svc = SimpleNamespace(config=config)
        self.toasts: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self._busy: set[str] = set()

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return key in self._busy

    def progress(self, key):
        return None

    def submit(self, key, fn, *args, **kwargs):
        if key in self._busy:
            return False
        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            done = Done(key=key, error=exc)
        inker_mode.on_task_done(self, done)
        return True


def _scene(config=None):
    ctx = _Ctx(config)
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(presets.load("fireball"), width=32, height=32, supersample=2)
    group = tab.doc.insert_flourish(B.bake(rec))
    return ctx, tab, group


def _core(recipe):
    return next(layer for layer in recipe.layers if layer.name == "Core")


def test_the_op_is_greyed_without_an_effect_and_offered_with_one():
    ctx = _Ctx()
    op = inker_ops.get("flourish_prompt")
    assert not op.enabled(ctx.state.inker, None)
    ctx, tab, group = _scene()
    assert op.enabled(ctx.state.inker, tab)


def test_without_a_model_the_words_go_through_the_keyword_mapper():
    ctx, tab, group = _scene()
    state = ctx.state.inker
    state.flourish_prompt_text = "blue, no smoke"
    assert inker_ops.run(ctx, inker_ops.get("flourish_prompt"))
    pending = state.flourish_pending[group]
    assert _core(pending).params["color_outer"] == keywords.COLOURS["blue"][1]
    assert all(not layer.visible for layer in pending.layers if layer.kind == "smoke")
    assert ctx.toasts[-1][0].startswith("[keywords]")
    # The document itself is untouched until the render lands.
    assert tab.doc.flourish_state(group).recipe != pending


def test_words_that_change_nothing_say_so_and_leave_no_pending():
    ctx, tab, group = _scene()
    assert not inker_mode.flourish_prompt(ctx, tab, text="")
    assert inker_mode.flourish_prompt(ctx, tab, text="please sparkle")  # submitted, then dropped
    assert group not in ctx.state.inker.flourish_pending
    assert ctx.toasts[-1][1] == "info"


def _model_dir(tmp_path):
    root = tmp_path / "models"
    base = root / inker_flourish.TEXT_MODEL_DIR
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"\0")
    return SimpleNamespace(t2i_model_root=root)


def test_presence_is_config_json_plus_safetensors(tmp_path):
    config = _model_dir(tmp_path)
    assert inker_flourish.text_model_present(config)
    (config.t2i_model_root / inker_flourish.TEXT_MODEL_DIR / "model.safetensors").unlink()
    assert not inker_flourish.text_model_present(config)
    assert not inker_flourish.text_model_present(None)
    assert inker_flourish.text_model_dir(None) is None


def test_with_a_model_the_answer_is_clamped_through_the_same_funnel(tmp_path, monkeypatch):
    config = _model_dir(tmp_path)
    ctx, tab, group = _scene(config)
    state = ctx.state.inker
    monkeypatch.setattr(inker_flourish, "text_model_available", lambda c: c is config)
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["request"] = json.loads(kwargs["input"])
        answer = {"kind": "ok", "diff": {"layers": {"Core": {"radius": 99999}}, "seed": 7}}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(answer) + "\n", stderr="")

    monkeypatch.setattr(winjob, "run", fake_run)
    assert inker_mode.flourish_prompt(ctx, tab, text="make the core enormous")
    assert seen["argv"][-1] == "warlock.pipelines.recipe_worker"
    assert seen["request"]["request"] == "make the core enormous"
    assert "uid" not in json.dumps(seen["request"]["recipe"])
    pending = state.flourish_pending[group]
    from warlock.studio.inker.flourish import prims

    assert _core(pending).params["radius"] == prims.params_of("core")["radius"].hi
    assert pending.seed == 7
    assert ctx.toasts[-1][0].startswith("[model]")


def test_a_model_that_answers_nonsense_falls_back_to_the_mapper(tmp_path, monkeypatch):
    config = _model_dir(tmp_path)
    ctx, tab, group = _scene(config)
    monkeypatch.setattr(inker_flourish, "text_model_available", lambda c: True)
    monkeypatch.setattr(
        winjob,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="hello there\n", stderr=""),
    )
    assert inker_mode.flourish_prompt(ctx, tab, text="blue")
    pending = ctx.state.inker.flourish_pending[group]
    assert _core(pending).params["color_outer"] == keywords.COLOURS["blue"][1]
    assert ctx.toasts[-1][0].startswith("[keywords]")
    assert "model:" in ctx.toasts[-1][0]


def test_a_model_that_times_out_falls_back_too(tmp_path, monkeypatch):
    config = _model_dir(tmp_path)
    ctx, tab, group = _scene(config)
    monkeypatch.setattr(inker_flourish, "text_model_available", lambda c: True)

    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout", 0))

    monkeypatch.setattr(winjob, "run", slow)
    assert inker_mode.flourish_prompt(ctx, tab, text="faster")
    assert ctx.toasts[-1][0].startswith("[keywords]")


def test_the_worker_protocol_pieces():
    assert recipe_worker.extract_json('Sure: {"a": {"b": 1}} ok') == {"a": {"b": 1}}
    assert recipe_worker.extract_json("{not json} {\"x\": 2}") == {"x": 2}
    assert recipe_worker.extract_json("[1, 2]") is None
    assert recipe_worker.extract_json("") is None
    messages = recipe_worker.build_messages({"name": "fx"}, "colder", keywords.DIFF_SCHEMA)
    assert messages[0]["role"] == "system" and keywords.DIFF_SCHEMA in messages[0]["content"]
    assert messages[1]["content"].endswith("Change: colder")


def test_the_worker_reports_a_bad_request_as_an_error_line(monkeypatch, capsys):
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("[1, 2]"))
    assert recipe_worker.main([]) == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["kind"] == "error" and "ValueError" in out["error"]


def test_the_doctor_row_reports_the_probe_honestly(tmp_path):
    config = SimpleNamespace(t2i_model_root=tmp_path / "models")
    (row,) = doctor._text_checks(config)  # noqa: SLF001
    assert row.ok and not row.fatal
    assert "keyword mapper" in row.detail and "measurement" in row.detail
    (row,) = doctor._text_checks(_model_dir(tmp_path))  # noqa: SLF001
    assert "present" in row.detail


def test_the_prompt_is_one_at_a_time_per_group():
    ctx, tab, group = _scene()
    ctx._busy.add(inker_flourish.PROMPT_KEY + f":{tab.uid}:{group}")
    assert not inker_mode.flourish_prompt(ctx, tab, text="blue")


@pytest.mark.parametrize("text", ["colder", "more sparks", "green flames, bigger"])
def test_ask_words_without_a_model_is_the_mapper(text):
    rec = presets.load("fireball")
    changed, notes, source = inker_flourish.ask_words(rec, text, model_dir=None)
    expected, expected_notes = keywords.apply(rec, text)
    assert changed == expected and notes == expected_notes and source == "keywords"
