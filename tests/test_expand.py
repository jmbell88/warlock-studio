"""Prompt expansion: the field, the templates, the door, and the wiring.

The expander itself (a real GPT-2 sampling pass) is exercised only in the gpu
lane, which is the one place real weights exist; everything here is about the
plumbing around it -- which is where the invariants live: "off" and absent are
one state (no vector re-keys), the expansion is derived and a mode is not, the
scene template never reaches a tile, and a broken expander costs the
enrichment rather than the job.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from warlock import fetch, guidance, models, vectors
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import expand
from warlock.pipelines import prompt as prompt_pipeline
from warlock.queue import Worker
from warlock.service import system as svc_system
from warlock.service.errors import Invalid
from warlock.service.validation import DERIVED_PARAMS, check_weights

# --- clean(): pure text tidying ----------------------------------------------


def test_clean_dedupes_and_collapses():
    assert expand.clean("sharp focus,  sharp focus, dramatic  light") == (
        "sharp focus, dramatic light"
    )


def test_clean_dedupe_is_case_insensitive_but_keeps_first_spelling():
    assert expand.clean("Cinematic, cinematic, epic") == "Cinematic, epic"


def test_clean_strips_stray_punctuation_and_newlines():
    assert expand.clean(" glowing.\n, , intricate ;") == "glowing, intricate"


def test_clean_of_nothing_is_empty():
    assert expand.clean(", ,  ,") == ""


# --- the whitelist -----------------------------------------------------------


class _FakeTokenizer:
    """Just enough of a GPT-2 tokenizer for ``_allowed_ids``."""

    eos_token_id = 0

    def get_vocab(self):
        return {"Ġsharp": 5, "sharp": 6, "Ġblurry": 7, "dragon": 8, ",": 9}

    def __call__(self, text, add_special_tokens=True, **_kw):
        ids = {"sharp": [6], " sharp": [5], ",": [9]}.get(text, [42])
        return {"input_ids": ids}


def test_allowed_ids_take_both_bpe_forms_plus_comma_and_eos():
    allowed = expand._allowed_ids(_FakeTokenizer(), ["Sharp", "", "  "])
    # Both the phrase-initial (Ġ) and mid-phrase forms, the comma, eos -- and
    # never the tokens of words that are not whitelisted.
    assert 5 in allowed and 6 in allowed and 9 in allowed and 0 in allowed
    assert 7 not in allowed and 8 not in allowed


# --- the guidance field ------------------------------------------------------


def test_expand_absent_and_off_are_one_state():
    """Storing "off" would re-key every same-settings vector; absence is the
    canonical spelling of the default."""
    assert "expand" not in guidance.normalize({})
    assert "expand" not in guidance.normalize({"expand": ""})
    assert "expand" not in guidance.normalize({"expand": "off"})


def test_expand_modes_are_stored_and_unknowns_refused():
    assert guidance.normalize({"expand": "asset"})["expand"] == "asset"
    assert guidance.normalize({"expand": "scene"})["expand"] == "scene"
    with pytest.raises(guidance.GuidanceError) as caught:
        guidance.normalize({"expand": "verbose"})
    assert caught.value.field == "expand"


def test_catalog_offers_expand_with_off_as_the_default():
    catalog = guidance.catalog()
    assert [o["key"] for o in catalog["fields"]["expand"]] == ["off", "asset", "scene"]
    assert catalog["defaults"]["expand"] == "off"


def test_the_mode_is_an_input_and_the_expansion_is_derived():
    assert "expand" in vectors.VECTOR_PARAMS
    assert "expanded_prompt" in DERIVED_PARAMS
    assert "expanded_prompt" not in vectors.VECTOR_PARAMS


# --- the scene template ------------------------------------------------------


def test_scene_build_swaps_the_framing_out():
    text = prompt_pipeline.build("a fox in a forest", {}, scene=True)
    assert "detailed illustration" in text
    assert "single subject" not in text
    assert "plain light gray background" not in text
    # The SDXL-failure-mode clauses survive whatever the picture is of.
    assert "no text" in text and "no watermark" in text


def test_tile_wins_over_scene():
    assert prompt_pipeline.build("moss", {}, tile=True, scene=True) == (
        prompt_pipeline.build("moss", {}, tile=True)
    )


# --- the registry row and the door -------------------------------------------


def test_the_expander_is_a_registry_row_with_a_pinned_fetch():
    entry = fetch.find(f"expander:{models.DEFAULT_EXPANDER}")
    assert entry is not None
    assert entry.fetch and entry.fetch[0].revision


def test_present_needs_the_weights_not_only_the_config(svc):
    spec = models.EXPANDER_MODELS[models.DEFAULT_EXPANDER]
    assert fetch.present(svc.config, "expander", spec)
    (svc.config.t2i_model_root / spec.dir_name / "pytorch_model.bin").unlink()
    assert not fetch.present(svc.config, "expander", spec)


def test_the_door_refuses_expansion_without_the_weights(svc):
    spec = models.EXPANDER_MODELS[models.DEFAULT_EXPANDER]
    (svc.config.t2i_model_root / spec.dir_name / "pytorch_model.bin").unlink()
    with pytest.raises(Invalid) as caught:
        check_weights(svc, "text", {"expand": "asset"})
    assert caught.value.field == "expand"
    assert caught.value.rows == (f"expander:{models.DEFAULT_EXPANDER}",)
    # Off costs nothing whatever is on disk.
    check_weights(svc, "text", {})


# --- the preview -------------------------------------------------------------


def test_preview_uses_the_scene_template_even_when_the_expander_cannot_load(svc):
    """The template is the mode's; the appended text is the expander's. A
    placeholder-weights host (this fixture) has no loadable expander, and the
    preview must degrade to the un-expanded prompt in the right template."""
    out = svc_system.prompt_preview(svc, {"expand": "scene"}, "a fox in a forest")
    assert "detailed illustration" in out["prompt"]
    assert out["expanded"] is None


def test_preview_keeps_the_asset_framing_for_asset_mode(svc):
    out = svc_system.prompt_preview(svc, {"expand": "asset"}, "a barrel")
    assert "single subject centered" in out["prompt"]


# --- the worker wiring -------------------------------------------------------

pytestmark_async = pytest.mark.asyncio


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def _wait_done(worker: Worker, job_id: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.store.get(job_id)["status"] in ("done", "error"):
            return
        await asyncio.sleep(0.01)
    pytest.fail("job did not finish before timeout")


@pytest.mark.asyncio
async def test_worker_expands_once_and_records_the_expansion(worker, monkeypatch):
    calls: list[tuple[str, int]] = []

    def fake_expand(text, model_dir, *, seed):
        calls.append((text, seed))
        return f"{text}, ornate, weathered"

    monkeypatch.setattr(expand, "expand", fake_expand)
    job_id = worker.store.create(
        "text", "a barrel", {"seed": 5, "expand": "asset"}, stage="reference"
    )
    worker.start()
    await _wait_done(worker, job_id)
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    # Expanded once, seeded by the reference seed, and the pipeline saw the
    # expanded text -- while the mode stayed an input and the text a derived.
    assert calls == [("a barrel", 5)]
    assert worker._text2image.prompts[-1] == "a barrel, ornate, weathered"
    assert job["params"]["expanded_prompt"] == "a barrel, ornate, weathered"
    assert job["params"]["expand"] == "asset"
    assert worker._text2image.scenes[-1] is False


@pytest.mark.asyncio
async def test_scene_mode_reaches_the_pipeline_as_the_scene_flag(worker, monkeypatch):
    monkeypatch.setattr(expand, "expand", lambda text, model_dir, *, seed: None)
    job_id = worker.store.create(
        "text", "a fox in a forest", {"seed": 5, "expand": "scene"}, stage="reference"
    )
    worker.start()
    await _wait_done(worker, job_id)
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    assert worker._text2image.scenes[-1] is True
    # The gate said "already detailed": nothing recorded, prompt as written.
    assert "expanded_prompt" not in job["params"]
    assert worker._text2image.prompts[-1] == "a fox in a forest"


@pytest.mark.asyncio
async def test_a_broken_expander_costs_the_enrichment_not_the_job(worker, monkeypatch):
    def boom(text, model_dir, *, seed):
        raise RuntimeError("corrupt weights")

    monkeypatch.setattr(expand, "expand", boom)
    job_id = worker.store.create(
        "text", "a barrel", {"seed": 5, "expand": "asset"}, stage="reference"
    )
    worker.start()
    await _wait_done(worker, job_id)
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    assert "expanded_prompt" not in job["params"]
    assert worker._text2image.prompts[-1] == "a barrel"


@pytest.mark.asyncio
async def test_a_tile_is_never_expanded(worker, monkeypatch):
    def forbidden(text, model_dir, *, seed):  # pragma: no cover - the assertion
        raise AssertionError("a tile reached the expander")

    monkeypatch.setattr(expand, "expand", forbidden)
    job_id = worker.store.create(
        "text", "mossy cobblestone", {"seed": 5, "expand": "asset"}, stage="tile"
    )
    worker.start()
    await _wait_done(worker, job_id)
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    assert worker._text2image.tiles[-1] is True
    assert worker._text2image.scenes[-1] is False


# --- the real weights, gpu lane only -----------------------------------------


@pytest.mark.gpu
def test_real_expander_appends_deterministically():
    """CPU inference, but gpu-lane by design: that is the one lane that sees
    the real ~/.warlock and therefore real weights."""
    spec = models.EXPANDER_MODELS[models.DEFAULT_EXPANDER]
    model_dir = Config().t2i_model_root / spec.dir_name
    if not (model_dir / "pytorch_model.bin").exists():
        pytest.skip("expander weights not downloaded")
    first = expand.expand("a sword", model_dir, seed=11)
    again = expand.expand("a sword", model_dir, seed=11)
    assert first == again
    assert first is not None and first.startswith("a sword, ")
    # The gate: a paragraph is left alone.
    detailed = "a " + ", ".join(f"very detail {i}" for i in range(30))
    assert expand.expand(detailed, model_dir, seed=11) is None
