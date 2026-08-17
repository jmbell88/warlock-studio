"""Which style LoRAs reach the pipe, and which adapters get applied.

``Text2Image.__init__`` builds no pipeline, so a fake stands in for one and
both halves are reachable with no torch, no weights and no GPU. What is under
test is the pairing filter in ``_load_loras`` and the predicate that replaced
``_apply_adapters``' family early return -- neither of which any test could
see before, because the family return made both functions no-ops off SDXL.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock import models
from warlock.pipelines.text2image import Text2Image


class FakePipe:
    """Records what a real diffusers pipeline would have been asked to do."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.applied: list[tuple[list[str], list[float]]] = []
        self.deleted: list[list[str]] = []
        self.disabled = 0
        self.enabled = 0
        # The order the pipe was driven in, which is half of what is under test.
        self.calls: list[str] = []
        # Real pipelines carry one; _recipe records its type name.
        self.scheduler = None

    def load_lora_weights(self, _dir, weight_name, adapter_name, **_kw) -> None:
        self.loaded.append(adapter_name)

    def set_adapters(self, names, weights) -> None:
        self.applied.append((list(names), list(weights)))
        self.calls.append("set")

    def enable_lora(self) -> None:
        self.enabled += 1
        self.calls.append("enable")

    def disable_lora(self) -> None:
        self.disabled += 1
        self.calls.append("disable")

    def delete_adapters(self, names) -> None:
        self.deleted.append(list(names))
        self.calls.append("delete")


def _t2i(tmp_path: Path, base_key: str, on_disk: list[str]) -> tuple[Text2Image, FakePipe]:
    loras = tmp_path / "loras"
    loras.mkdir(exist_ok=True)
    for name in on_disk:
        (loras / name).write_bytes(b"")
    t2i = Text2Image(models.BASE_MODELS[base_key], tmp_path)
    pipe = FakePipe()
    t2i._pipe = pipe
    return t2i, pipe


def _all_files() -> list[str]:
    return [lora.filename for lora in models.STYLE_LORAS.values()]


def _fitting(base_key: str) -> list[str]:
    return models.loras_by_base()[base_key]


def _select_all_fitting(t2i, pipe, base_key: str) -> list[str]:
    """Drive one job per fitting style, which is what attaches them now.

    Style adapters are loaded on selection rather than at ``load()`` (MDL-07),
    so a test about *which* adapters a base will accept has to ask for them.
    """
    for key in _fitting(base_key):
        t2i._apply_adapters(pipe, key, 1.0)
    return _fitting(base_key)


@pytest.mark.parametrize("base_key", sorted(models.BASE_MODELS))
def test_only_same_family_adapters_are_loaded(tmp_path, base_key):
    """Every style LoRA in the registry is on disk; only the fitting ones ever
    reach the pipe. Loading a foreign one would raise with the checkpoint
    already resident, which is why this is a filter and not a tolerance.

    The filter moved from ``_load_loras`` to the moment of selection, but it is
    the same filter and this asserts the same thing about the result.
    """
    base = models.BASE_MODELS[base_key]
    files = _all_files() + ([base.base_lora] if base.base_lora else [])
    t2i, pipe = _t2i(tmp_path, base_key, files)
    t2i._load_loras(pipe)
    # Nothing optional is attached before a job asks for it.
    assert t2i._adapters == set()
    fitting = _select_all_fitting(t2i, pipe, base_key)
    # Every fitting style reached the pipe when its job selected it. Only the
    # *last* is still attached -- deselection deletes the rest (MDL-17).
    assert sorted(pipe.loaded) == sorted(
        fitting + ([base.base_lora and models.BASE_LORA_ADAPTER] if base.base_lora else [])
    )
    assert t2i._adapters == ({fitting[-1]} if fitting else set())
    foreign = {
        lora.key for lora in models.STYLE_LORAS.values() if lora.family != base.family
    }
    assert not foreign & set(pipe.loaded)


def test_no_optional_adapter_is_attached_until_a_job_selects_it(tmp_path):
    """MDL-07: every compatible style LoRA on disk used to load during base
    init, so one corrupt file the user had never selected made the whole
    checkpoint unusable -- and the weights of styles nobody asked for sat on
    the device, which ``BaseModel.vram_gib`` does not account for."""
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    assert pipe.loaded == [], "load() must attach no optional adapters"
    assert t2i._adapters == set()

    chosen = _fitting("turbo")[0]
    t2i._apply_adapters(pipe, chosen, 1.0)
    assert pipe.loaded == [chosen], "exactly the one selected, and only it"

    # Idempotent: a warm pipe pays the load once per adapter, not once per job.
    t2i._apply_adapters(pipe, chosen, 0.5)
    assert pipe.loaded == [chosen]


def test_a_corrupt_optional_adapter_fails_only_the_job_that_selects_it(tmp_path):
    """The compounding half of MDL-07: eagerly, this raise happened inside the
    load transaction, so an unrelated corrupt file left the user unable to
    generate at all until they worked out which one to delete."""
    fitting = _fitting("turbo")
    chosen, other = fitting[0], fitting[1]
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())

    def _explode(_dir, weight_name, adapter_name, **_kw):
        if adapter_name == chosen:
            raise RuntimeError("HeaderTooLarge")
        pipe.loaded.append(adapter_name)

    pipe.load_lora_weights = _explode
    t2i._load_loras(pipe)  # the base loads fine, which is the whole point

    with pytest.raises(RuntimeError, match="could not be loaded"):
        t2i._apply_adapters(pipe, chosen, 1.0)
    # A different style still works, and so does generating with none.
    t2i._apply_adapters(pipe, other, 1.0)
    assert other in t2i._adapters
    assert chosen not in t2i._adapters


def test_a_missing_same_family_adapter_is_still_skipped_not_fatal(tmp_path):
    """Loading a base with no style LoRAs on disk at all must still work: they
    are separate optional downloads and a user who fetched none has to be able
    to generate."""
    t2i, pipe = _t2i(tmp_path, "turbo", [])
    t2i._load_loras(pipe)
    assert t2i._adapters == set()
    assert pipe.loaded == []


def test_selecting_a_style_that_is_not_downloaded_refuses_the_job(tmp_path):
    """It used to warn and generate bare -- so the job finished looking
    successful without the style, on a row whose ``style_lora`` claimed one.
    That key is in ``VECTOR_PARAMS``, so the row then joined the findings
    corpus as evidence about a style that never ran (MDL-05)."""
    t2i, pipe = _t2i(tmp_path, "turbo", [])
    t2i._load_loras(pipe)
    with pytest.raises(RuntimeError, match="not downloaded"):
        t2i._apply_adapters(pipe, _fitting("turbo")[0], 1.0)
    assert pipe.applied == []


def test_a_pipe_with_no_adapters_is_never_disabled(tmp_path):
    """The latent case the family early return hid, and it was wrong for SDXL
    too: ``turbo`` has no base LoRA, so on a host with no loras/ directory the
    empty branch called disable_lora() on a pipe with no PEFT state."""
    t2i, pipe = _t2i(tmp_path, "turbo", [])
    t2i._load_loras(pipe)
    assert not t2i._has_adapters
    t2i._apply_adapters(pipe, None, 1.0)
    assert pipe.disabled == 0
    assert pipe.applied == []


def test_a_still_attached_adapter_is_disabled_when_nothing_is_selected(tmp_path):
    """A deselected style is normally *deleted* (MDL-17), which leaves nothing
    to disable. The disable branch survives for the degraded case: a failed
    delete leaves real PEFT state on the pipe, and that state must still be
    switched off for the styleless job."""
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    t2i._apply_adapters(pipe, _fitting("turbo")[0], 1.0)
    assert t2i._has_adapters

    def _explode(names):
        raise RuntimeError("peft says no")

    pipe.delete_adapters = _explode
    t2i._apply_adapters(pipe, None, 1.0)
    assert pipe.disabled == 1


def test_a_selected_adapter_is_applied_at_its_weight(tmp_path):
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    chosen = _fitting("turbo")[0]
    t2i._apply_adapters(pipe, chosen, 0.8)
    assert pipe.applied == [([chosen], [0.8])]


def test_a_foreign_adapter_selected_anyway_is_warned_about_not_applied(tmp_path, caplog):
    """guidance.normalize refuses the pair at the door and queue.py drops it
    for a stored row, so this is the third line of defence -- and it must not
    raise, because the artifact is a job the user can no longer edit."""
    base_key, lora_key = next(
        (b.key, lo.key)
        for b in models.BASE_MODELS.values()
        for lo in models.STYLE_LORAS.values()
        if not models.lora_fits(b, lo)
    )
    base = models.BASE_MODELS[base_key]
    files = _all_files() + ([base.base_lora] if base.base_lora else [])
    t2i, pipe = _t2i(tmp_path, base_key, files)
    t2i._load_loras(pipe)
    with caplog.at_level("WARNING"):
        t2i._apply_adapters(pipe, lora_key, 1.0)
    assert lora_key not in [n for names, _ in pipe.applied for n in names]
    assert any(lora_key in r.getMessage() for r in caplog.records)


def test_a_style_installed_while_the_pipe_is_warm_is_picked_up(tmp_path):
    """MDL-05, the case that motivated lazy loading.

    ``_load_loras`` ran once and froze the adapter set for the pipe's life,
    while the pipe stays warm for the idle timeout. So: generate once, install
    a style in Settings, submit a styled job -- the file passed
    ``check_weights`` at the door, then got dropped here with a "not
    downloaded" warning that was not true, and the job finished looking
    successful with no style on a row that recorded one.

    Attaching on selection means the adapter that arrived after the load is
    simply found when it is asked for.
    """
    base_key = "turbo"
    chosen = _fitting(base_key)[0]
    base = models.BASE_MODELS[base_key]
    others = [lo.filename for lo in models.STYLE_LORAS.values() if lo.key != chosen]
    t2i, pipe = _t2i(
        tmp_path, base_key, others + ([base.base_lora] if base.base_lora else [])
    )
    t2i._load_loras(pipe)

    # Not on disk yet: the job that selects it is refused rather than run bare.
    with pytest.raises(RuntimeError, match="not downloaded"):
        t2i._apply_adapters(pipe, chosen, 1.0)
    assert pipe.applied == []

    # Installed in Settings, with the same pipe still resident.
    (tmp_path / "loras" / models.STYLE_LORAS[chosen].filename).write_bytes(b"")
    t2i._apply_adapters(pipe, chosen, 0.9)
    assert pipe.applied == [([chosen], [0.9])]
    assert chosen in t2i._adapters


def test_a_style_survives_a_run_that_had_none_before_it(tmp_path):
    """The pipe stays resident across jobs, so PEFT state outlives the job that
    set it. ``disable_lora`` sets ``_disable_adapters`` on every layer and
    ``set_adapters`` writes only the scaling, so without an explicit
    ``enable_lora`` one no-style job silently switched every later job in the
    process to no style -- and it reads as working, because the trigger words
    are still prepended and the output does change.
    """
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    chosen = _fitting("turbo")[0]
    # A first styled job, which is what puts PEFT state on the pipe at all now
    # that adapters attach on selection rather than at load().
    t2i._apply_adapters(pipe, chosen, 0.9)
    # Then a job with no style: the adapter is deleted outright now (MDL-17),
    # so there is no PEFT state left to disable.
    t2i._apply_adapters(pipe, None, 1.0)
    assert pipe.deleted == [[chosen]]
    assert pipe.disabled == 0
    # And a third that wants the style back: reloaded from disk and enabled.
    t2i._apply_adapters(pipe, chosen, 0.9)
    assert pipe.loaded == [chosen, chosen]
    assert pipe.enabled == 2, "the adapter was never re-enabled"
    assert pipe.applied[-1] == ([chosen], [0.9])


def test_enabling_precedes_the_weights_it_is_meant_to_restore(tmp_path):
    """Order matters and only one order is right: enable_lora() takes no
    weights, so setting them first and enabling after would apply the previous
    scaling for one call."""
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    t2i._apply_adapters(pipe, _fitting("turbo")[0], 0.5)
    assert pipe.calls.index("enable") < pipe.calls.index("set")


def test_the_fake_defines_nothing_the_real_pipeline_classes_lack():
    """FakePipe fidelity. Every method the fake answers, the real pinned
    pipeline classes must answer too -- otherwise the whole suite passes over
    a call that AttributeErrors on the first real job. The klein pipeline is
    the case that motivated this: _apply_adapters calls enable_lora() on
    whatever _load built, and no test with a fake could notice a family whose
    real class lacked it.

    Class-level hasattr only: no weights load, and importing the classes is
    offline-safe (nothing in diffusers downloads at import; HF_HUB_OFFLINE is
    already 1 via the warlock import above).
    """
    diffusers = pytest.importorskip("diffusers")
    real = [
        # What AutoPipelineForText2Image resolves for the SDXL entries, and
        # the three from_pipe classes _conditioned builds over it.
        diffusers.StableDiffusionXLPipeline,
        diffusers.StableDiffusionXLImg2ImgPipeline,
        diffusers.StableDiffusionXLControlNetPipeline,
        diffusers.StableDiffusionXLControlNetImg2ImgPipeline,
        # And what it resolves for family flux2klein. Verified present on the
        # pinned diffusers (0.39.0) rather than guarded: the registry's klein
        # entries depend on this class existing at all.
        diffusers.Flux2KleinPipeline,
    ]
    fake_methods = [
        name
        for name, value in vars(FakePipe).items()
        if callable(value) and not name.startswith("__")
    ]
    assert fake_methods, "FakePipe stopped defining methods; the test is vacuous"
    for cls in real:
        for name in fake_methods:
            assert hasattr(cls, name), f"{cls.__name__} lacks {name}"


def test_the_recipe_records_a_style_only_when_it_actually_applied(tmp_path):
    """Applied, not requested -- the trigger prepend's own predicate. A style
    that never loaded (not downloaded, or another family's) did not shape the
    image, and a recipe naming it anyway is the pixel-sheet sidecar bug in
    provenance form."""
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    chosen = _fitting("turbo")[0]
    # ``_apply_adapters`` runs before ``_recipe`` in a real generate, and it is
    # what attaches the adapter now -- so the predicate reads the same thing it
    # always did, "did this style actually load", just filled in later.
    t2i._apply_adapters(pipe, chosen, 0.5)
    assert t2i._recipe(1, "p", None, chosen, 0.5, None, 1, False)["style_lora"] == chosen

    bare_root = tmp_path / "bare"
    bare_root.mkdir()
    bare, bare_pipe = _t2i(bare_root, "turbo", [])
    bare._load_loras(bare_pipe)
    recipe = bare._recipe(1, "p", None, chosen, 0.5, None, 1, False)
    assert "style_lora" not in recipe
    assert "lora_weight" not in recipe


# --- stale styles are deleted, not just disabled (MDL-17) ---------------------


def test_a_style_the_next_job_does_not_use_is_deleted(tmp_path):
    """The unbounded-accumulation half of MDL-17: adapters attached lazily and
    were never detached, so every style picked during a pipe's life stayed
    resident in host and VRAM until full unload -- and ``vram._adapter_cost``
    prices only base + selected style, so attached exceeded priced."""
    a, b = _fitting("turbo")[0], _fitting("turbo")[1]
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    t2i._apply_adapters(pipe, a, 1.0)
    t2i._apply_adapters(pipe, b, 1.0)
    assert pipe.deleted == [[a]]
    assert t2i._adapters == {b}
    # Deleted before the surviving set is enabled, so a pipe whose adapters
    # were all just deleted is never enable_lora()'d over nothing.
    last_delete = len(pipe.calls) - 1 - pipe.calls[::-1].index("delete")
    assert last_delete < pipe.calls.index("enable", last_delete)
    assert last_delete < pipe.calls.index("set", last_delete)


def test_the_base_distillation_adapter_is_never_deleted(tmp_path):
    """The base LoRA is step distillation, not style: it lives in
    ``_base_adapter``, never in ``_adapters``, so it is structurally out of
    reach of the stale-set arithmetic."""
    base = models.BASE_MODELS["lightning"]
    files = _all_files() + [base.base_lora]
    t2i, pipe = _t2i(tmp_path, "lightning", files)
    t2i._load_loras(pipe)
    style = _fitting("lightning")[0]
    t2i._apply_adapters(pipe, style, 0.7)
    t2i._apply_adapters(pipe, None, 1.0)
    assert [style] in pipe.deleted
    assert all(models.BASE_LORA_ADAPTER not in entry for entry in pipe.deleted)
    # And a later styled apply still carries it, always at full strength.
    t2i._apply_adapters(pipe, style, 0.7)
    names, weights = pipe.applied[-1]
    assert models.BASE_LORA_ADAPTER in names
    assert weights[names.index(models.BASE_LORA_ADAPTER)] == 1.0


def test_repeating_a_style_neither_deletes_nor_reloads(tmp_path):
    """Same-style repeats are the common case and must stay free: no delete,
    and the idempotence guard in ``_ensure_adapter`` skips the reload."""
    a = _fitting("turbo")[0]
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    t2i._apply_adapters(pipe, a, 1.0)
    t2i._apply_adapters(pipe, a, 0.5)
    assert pipe.deleted == []
    assert pipe.loaded == [a]


def test_a_deleted_style_reattaches_when_selected_again(tmp_path):
    """Alternating styles pay one load_lora_weights from disk per switch --
    seconds, against the minutes of sampling around it."""
    a, b = _fitting("turbo")[0], _fitting("turbo")[1]
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    t2i._apply_adapters(pipe, a, 1.0)
    t2i._apply_adapters(pipe, b, 1.0)
    t2i._apply_adapters(pipe, a, 1.0)
    assert pipe.loaded == [a, b, a]
    assert t2i._adapters == {a}


def test_a_pipe_whose_only_adapter_was_deleted_is_not_disabled(tmp_path):
    """turbo has no base LoRA, so deleting the one style leaves no PEFT state
    at all -- disable_lora() on that pipe is the exact call the derived
    ``_has_adapters`` predicate exists to prevent. The delete must run before
    the predicate is read."""
    a = _fitting("turbo")[0]
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    t2i._apply_adapters(pipe, a, 1.0)
    t2i._apply_adapters(pipe, None, 1.0)
    assert pipe.deleted == [[a]]
    assert pipe.disabled == 0


def test_a_failed_delete_keeps_the_bookkeeping_truthful(tmp_path):
    """Degrades to the old accumulation, loudly: nothing escapes, ``_adapters``
    only shrinks on success (so the reload guard keeps skipping the
    still-attached adapter), and the job's own set is still applied."""
    a, b = _fitting("turbo")[0], _fitting("turbo")[1]
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras(pipe)
    t2i._apply_adapters(pipe, a, 1.0)

    def _explode(names):
        raise RuntimeError("peft says no")

    pipe.delete_adapters = _explode
    t2i._apply_adapters(pipe, b, 0.8)
    assert a in t2i._adapters
    assert pipe.applied[-1] == ([b], [0.8])


# --- load() is transactional -------------------------------------------------
#
# self._pipe used to be assigned straight out of from_pretrained, i.e. before
# the scheduler swap, the device placement and the LoRA attachment had run. A
# failure in any of those left `loaded` True over a half-configured pipe, so
# every later load() returned early and the weights stayed resident with
# nothing able to reach them -- the leak that only shows up as an OOM two jobs
# later. The four failure points below are exactly the four steps.


class _Placed(FakePipe):
    """A FakePipe that also answers the placement calls load() makes."""

    def __init__(self, fail_at: str | None = None) -> None:
        super().__init__()
        self._fail_at = fail_at
        self.placed: list[str] = []

    def to(self, device):
        self.placed.append(str(device))
        if self._fail_at == "placement":
            raise RuntimeError("CUDA out of memory")
        return self

    def enable_model_cpu_offload(self):
        self.placed.append("offload")
        if self._fail_at == "placement":
            raise RuntimeError("CUDA out of memory")

    def load_lora_weights(self, _dir, weight_name, adapter_name, **_kw) -> None:
        if self._fail_at == "lora":
            raise RuntimeError("corrupt safetensors")
        super().load_lora_weights(_dir, weight_name, adapter_name, **_kw)


def _loadable(tmp_path, monkeypatch, base_key, *, fail_at=None, checkpoint_raises=False):
    """A Text2Image whose load() will reach every step but for ``fail_at``."""
    import sys
    import types

    root = tmp_path / "root"
    (root / models.BASE_MODELS[base_key].dir_name).mkdir(parents=True)
    (root / models.BASE_MODELS[base_key].dir_name / "model_index.json").write_text("{}")
    loras = root / "loras"
    loras.mkdir()
    for name in _all_files():
        (loras / name).write_bytes(b"")
    base_lora = models.BASE_MODELS[base_key].base_lora
    if base_lora:
        (loras / base_lora).write_bytes(b"")

    pipe = _Placed(fail_at)

    def from_pretrained(*_a, **_kw):
        if checkpoint_raises:
            raise RuntimeError("no such variant")
        return pipe

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            AutoPipelineForText2Image=types.SimpleNamespace(
                from_pretrained=from_pretrained
            )
        ),
    )
    # Always stubbed, not only on the failure branch: the real one imports the
    # concrete scheduler class out of diffusers, which is the module replaced
    # above. What is under test is load()'s ordering, not the scheduler table.
    def scheduler(*_a, **_kw):
        if fail_at == "scheduler":
            raise RuntimeError("unknown scheduler")
        return "swapped"

    monkeypatch.setattr("warlock.pipelines.text2image._scheduler", scheduler)
    return Text2Image(models.BASE_MODELS[base_key], root), pipe


@pytest.mark.parametrize(
    "fail_at,checkpoint_raises",
    [
        (None, True),      # from_pretrained itself
        ("scheduler", False),
        ("placement", False),
        ("lora", False),
    ],
)
def test_a_failed_load_leaves_nothing_behind(tmp_path, monkeypatch, fail_at, checkpoint_raises):
    pytest.importorskip("torch")
    # `lightning` is the entry that reaches every step: it declares a
    # scheduler, a residency and a base LoRA, and fitting style LoRAs exist for
    # it. A base with `scheduler=None` would skip a step and pass vacuously.
    t2i, _pipe = _loadable(
        tmp_path, monkeypatch, "lightning",
        fail_at=fail_at, checkpoint_raises=checkpoint_raises,
    )
    with pytest.raises(RuntimeError):
        t2i.load()
    assert t2i.loaded is False
    assert t2i._adapters == set()
    assert t2i._base_adapter is None


def test_a_load_that_succeeds_publishes_a_fully_configured_pipe(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    t2i, pipe = _loadable(tmp_path, monkeypatch, "lightning")
    t2i.load()
    assert t2i.loaded is True
    assert t2i._pipe is pipe
    # Published *after* everything: placement happened, the required adapter
    # attached. The optional ones deliberately did not -- they attach when a
    # job selects one, so that a corrupt style nobody asked for cannot take the
    # checkpoint down with it (MDL-07).
    assert pipe.placed == ["cuda"]
    assert t2i._adapters == set()
    assert t2i._base_adapter == models.BASE_LORA_ADAPTER


def test_an_offloaded_base_is_never_also_moved_to_the_device(tmp_path, monkeypatch):
    # accelerate's hooks assume the modules start on the host; a preceding
    # .to("cuda") defeats the whole mechanism. Mutually exclusive, and the
    # transactional rewrite must not have made them both run.
    pytest.importorskip("torch")
    t2i, pipe = _loadable(tmp_path, monkeypatch, "flux_klein")
    t2i.load()
    assert pipe.placed == ["offload"]
