"""Shared fixtures. fake_pipelines replaces the GPU-bound pieces (a real
trellis-server.exe subprocess, real torch/diffusers) with in-process fakes
so Worker's control flow -- cancellation, crash recovery, shutdown -- is
testable without a GPU."""

from __future__ import annotations

import asyncio
import functools
import os
import threading
import time
from pathlib import Path

import pytest

from warlock import memlog, vram
from warlock.db import JobStore
from warlock.models import DEFAULT_LORA_WEIGHT
from warlock.pipelines.text2image import JobCancelled

# Captured before anything can patch it, so ``real_system_memory`` can hand the
# genuine reader back to the one test that is about what it reads.
_REAL_SYSTEM_MEMORY = memlog.system_memory

# The same, for the device side. ``real_device_memory`` hands it back to
# ``test_vram.py``'s reading-never-raises check, which is about the reader
# itself and would otherwise be answered by the pin below.
_REAL_DEVICE_MEMORY = vram.device_memory

# Set by ``pytest_configure`` when the gpu lane is the one selected. That lane is
# the single exception to the WARLOCK_HOME pin below -- see ``_no_migration``.
_GPU_LANE = False


def _selects_gpu(config) -> bool:
    """Whether this invocation is asking *for* the gpu tests.

    Decided with pytest's own ``-m`` expression engine rather than matched as
    text. The negated mention must be skipped -- the default expression is
    ``not gpu and not perf``, so a plain substring test answers yes to every
    ordinary run -- and a lookbehind regex cannot see a negation through a
    parenthesis: ``-m "not (gpu or perf)"`` is a legal spelling of the same
    default, and the old ``(?<!not )\\bgpu\\b`` counted it as selecting the
    lane (refused under ``-n 8``; silently unpinned under ``-n 0``).

    "Asking for gpu" means the expression admits a test marked ``gpu`` while
    refusing an unmarked one: any negated mention fails the first half, and a
    bare exclusion like ``not perf`` -- which admits everything -- fails the
    second. An expression pytest itself cannot parse selects nothing here;
    pytest will refuse it with its own error moments later.
    """
    from _pytest.mark.expression import Expression

    text = (config.getoption("-m") or "").strip()
    if not text:
        return False
    try:
        expr = Expression.compile(text)
    except Exception:
        return False
    gpu_marked = expr.evaluate(lambda name: name == "gpu")
    unmarked = expr.evaluate(lambda name: False)
    return gpu_marked and not unmarked


def pytest_configure(config):
    """Refuse ``-m gpu`` under xdist, and only that combination.

    The default run is ``-n auto``, which is right for 8,700 tests that fake
    every model. It is exactly wrong for the gpu lane: those tests load real
    7 GB checkpoints onto the one card in this machine, so N workers means N
    simultaneous loads and an out-of-memory failure that reads like a bug in
    whatever test happened to lose. Erroring here rather than in a fixture
    because by fixture time the workers already exist.

    The match deliberately skips a negated mention. The *default* expression is
    ``not gpu and not perf``, so a plain substring test fires this refusal on
    every ordinary run -- which it duly did the first time it was written.
    """
    global _GPU_LANE

    if not _selects_gpu(config):
        return
    _GPU_LANE = True
    if (config.getoption("numprocesses", None) or 0) not in (0, None):
        raise pytest.UsageError(
            "the gpu lane must run serially -- real weights, one card. "
            "Use: uv run pytest -m gpu -n 0"
        )


@pytest.fixture(scope="session", autouse=True)
def _no_imgui_ini():
    """No run may leave an ``imgui.ini`` in the working directory.

    imgui writes one on shutdown unless ``set_ini_filename(None)`` is called,
    and a stray one is a cross-process hazard rather than clutter: it restores
    remembered window geometry into the next process, which once failed
    eighteen untouched tests. All four sites that build a context suppress it
    today; this is what makes a fifth site fail loudly instead of quietly
    seeding the next run.
    """
    yield
    stray = Path.cwd() / "imgui.ini"
    assert not stray.exists(), f"a test left {stray} behind; call set_ini_filename(None)"


@pytest.fixture(scope="session", autouse=True)
def _no_migration(tmp_path_factory):
    """Nothing in the suite may move the developer's real library.

    ``config.get_config()`` runs the one-time ``~/.warlock`` migration, and a
    checkout still carrying a legacy ``models/`` or ``palettes/`` is exactly
    what it is looking for -- so without this, the first test to build a Config
    would start copying 95 GB into the user's home directory. Session-scoped and
    autouse rather than a per-test monkeypatch: a single file that forgets it is
    the whole accident, and there is no test in this repo that wants the real
    migration to run (``test_home_migration.py`` clears it explicitly and points
    every path at ``tmp_path``).

    And ``WARLOCK_HOME`` is pinned at a throwaway directory for the second half
    of the same problem. ``WARLOCK_NO_MIGRATE`` stops the suite *moving* the
    developer's library; it does nothing about the suite *reading* it, and every
    root resolves under ``_home()`` (config.py:49) unless its own variable is
    set. So any test that built a ``Config`` without naming ``t2i_model_root``
    got the real ``~/.warlock/models`` -- and with birefnet downloaded there,
    ``matting.mask`` ran a genuine ~12 s CPU inference. Measured on
    2026-08-16: ``test_sprite_synthesis_worker.py`` alone went 160.8 s -> 4.2 s
    with this pin, and ``test_doctor.py``'s child load probes stopped finding
    weights to load.

    Speed is the smaller half. The real defect is that those tests produced a
    *different matte* on a machine with the weights than on one without, which
    is precisely the "a test about the fallback must pin the fallback" rule the
    ``svc`` fixture below states five times over -- stated once here instead,
    because ``WARLOCK_HOME`` moves every root at once and a per-root pin has to
    be remembered by every file that builds its own Config. Thirteen did not.

    A test that wants the real home clears the variable, as
    ``test_home_migration.py`` already does for all seven roots.

    **The gpu lane is exempt, and has to be.** Those tests resolve their
    checkpoints through ``get_config().t2i_model_root`` and ``pytest.skip`` when
    the files are not there -- deliberately, because every weight is a manual
    download. Pinned home, the whole lane would therefore go green as eighteen
    skips: a run that loaded nothing, tested nothing, and said nothing about it.
    That is the same silent-pass this pin exists to prevent, so the one lane that
    genuinely wants the developer's real library keeps it.
    """
    if _GPU_LANE:
        # Real home, real weights. WARLOCK_NO_MIGRATE below still applies: the
        # lane may read the library, never move it.
        previous = os.environ.get("WARLOCK_NO_MIGRATE")
        os.environ["WARLOCK_NO_MIGRATE"] = "1"
        yield
        if previous is None:
            os.environ.pop("WARLOCK_NO_MIGRATE", None)
        else:
            os.environ["WARLOCK_NO_MIGRATE"] = previous
        return

    home = tmp_path_factory.mktemp("warlock-home")
    # The same directories the mkdir loop at the end of ``get_config()`` makes
    # (config.py: ``for d in (cfg.home, cfg.data_dir, cfg.palette_dir,
    # cfg.t2i_model_root)``), because a real ``~/.warlock`` has them and the
    # pin is only supposed to change what is *in* the model root -- not whether
    # the roots exist. Left out, tests that build a bare ``Config``
    # (bypassing ``get_config()`` and its mkdirs) hit
    # ``doctor._disk_check``'s ``shutil.disk_usage`` on a path that is not
    # there: three failed that way before this was added, none of them about
    # disks. Empty, so ``matting.available`` is still False.
    for name in ("assets", "models", "palettes"):
        (home / name).mkdir(parents=True, exist_ok=True)
    previous = {
        name: os.environ.get(name) for name in ("WARLOCK_NO_MIGRATE", "WARLOCK_HOME")
    }
    os.environ["WARLOCK_NO_MIGRATE"] = "1"
    os.environ["WARLOCK_HOME"] = str(home)
    yield
    for name, was in previous.items():
        if was is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = was


@pytest.fixture(autouse=True)
def _no_native_dialogs(monkeypatch):
    """No test may block the suite on a real ``MessageBoxW``.

    ``instance.alert`` and ``instance.alert_fatal`` raise genuine modal native
    dialogs on Windows -- by design, for a user whose GL context is gone. A
    test that drives the crash path (``test_splash.py`` and
    ``test_studio_runtime.py`` both run ``App.run()`` into ``_report_crash``)
    would park the whole suite on a box nobody is there to click; an unattended
    run wedges at that test forever. Autouse for the same reason as
    ``_no_migration``: one test file that forgets the stub is the whole
    accident. A test that wants to assert the dialog's contents can still
    monkeypatch over this with its own recorder.
    """
    from warlock import instance

    shown: list[tuple] = []
    monkeypatch.setattr(instance, "alert", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(
        instance, "alert_fatal", lambda *a, **k: (shown.append(a), False)[1]
    )
    return shown


@pytest.fixture(autouse=True)
def _roomy_host_memory(monkeypatch):
    """No test's verdict may depend on how loaded this machine happens to be.

    ``queue._require_commit_headroom`` refuses to dispatch when host commit is
    at or past ``COMMIT_CEILING``, or when the commit limit has less free than
    the model about to load needs. Both readings come from the real machine
    through ``memlog.system_memory``, so on a busy box the queue tests stop
    testing the queue and start testing the developer's other windows: the
    2026-08-16 baseline run failed nine tests with "needs about 7.0 GiB ... only
    8.0 GiB of the commit limit is free", none of them about memory.

    The suite is its own worst offender, which is what makes this load-bearing
    rather than tidy. A BiRefNet load leaves ~1 GiB resident per the arena note
    in ``doctor._load_probe``, so the slow tests above were inflating the very
    reading that then failed the tests after them -- and running the suite under
    ``-n auto`` multiplies that by the worker count.

    Pinned at ``memlog.system_memory`` rather than at ``queue.commit_fraction``
    because that is the single reader both branches of the check share; a pin on
    the fraction alone would leave the ``need_gib`` branch on real memory, which
    is the branch that actually failed. Roomy rather than realistic: a test that
    wants a refusal patches this itself (``test_queue.py`` does, twice), and
    those patches still win -- they are applied after this one.

    Exempt in the gpu lane, for the same reason ``_no_migration`` is: that lane
    runs against the real card and the real weights, and a real-hardware lane
    that reads a hand-written memory figure is not measuring the hardware.
    """
    if _GPU_LANE:
        return
    monkeypatch.setattr(
        memlog, "system_memory", lambda: memlog.SystemMemory(commit_total=8.0, commit_limit=256.0)
    )
    # The status bar's meter reads *physical* memory, which is a different
    # number from commit and a separate reader -- pinned for the same reason:
    # a status-bar test whose verdict moves with the developer's browser tabs
    # is not testing the status bar.
    monkeypatch.setattr(
        memlog,
        "physical_memory",
        lambda: memlog.PhysicalMemory(total=64.0, available=48.0),
    )


@pytest.fixture
def real_system_memory(monkeypatch):
    """The genuine reader, for the tests that are about what it reads.

    ``_roomy_host_memory`` would otherwise quietly answer
    ``test_memlog.py``'s pages-vs-bytes unit check with a hand-written figure --
    a fabricated reading passing a test whose entire purpose is to catch a
    fabricated-looking one.
    """
    monkeypatch.setattr(memlog, "system_memory", _REAL_SYSTEM_MEMORY)


@pytest.fixture(autouse=True)
def _roomy_device_memory(monkeypatch):
    """``_roomy_host_memory``'s other half: nor may a verdict depend on the card.

    The host pin above closed this on commit and left the device side open, and
    the gap behaved exactly as that fixture's own history predicts. A full
    ``uv run pytest`` failed **a moving set of 10-14 tests** across
    ``test_sprite_synthesis_worker.py``, ``test_tilesheet_worker.py`` and
    ``test_queue.py``, none of them about VRAM, with an admission refusal
    carrying a live driver reading: *"needs about 26.7 GiB of VRAM and only
    26.4 GiB is available"*. Nothing in the suite put that 26.4 there.

    **Two conditions have to meet, which is why it looked like a flake.**
    ``vram.device_memory`` reaches torch through ``sys.modules`` and returns
    None without it, and None means ``vram.plan`` resolves "no CUDA device
    detected" and admission control is off entirely. So a file run on its own
    never sees the card and always passes; a file run in an xdist worker that
    happened to run a torch-importing test first sees it and passes or fails
    according to what else on the machine holds VRAM. The target moves because
    ``--dist loadfile`` does not put the same files in the same worker twice.

    Pinned at ``vram.device_memory`` because that is the single reader the two
    admission checks share -- ``queue._check_resources`` at dispatch and
    ``service.validation._plan`` at the door, through ``vram.plan``. The other
    two device readers are deliberately left alone: ``vram.probe`` is the
    startup import-torch variant that only ``studio/runtime.py`` calls, and
    ``vram.live_memory`` is the status bar's NVML meter, whose tests set
    ``Sampler.reading`` directly and so assert nothing about a live figure.

    A full card rather than a plausible one, and a 32 GiB one rather than an
    enormous one: the budget has to clear ``COEXIST_GIB`` so the auto-selected
    mode stays coexist, which is what the suite was written against, and a
    figure far above any real card would put a fictional number into every
    ``plan.reason`` string. Roomy rather than realistic is ``_roomy_host_memory``'s
    rule and it carries over: a test that wants a refusal patches this itself
    -- ``test_queue.py`` does, six times -- and those patches still win, being
    applied after this one.

    Exempt in the gpu lane for the third time in this file, and for the reason
    it gives twice already: that lane runs against the real card, and a
    real-hardware lane reading a hand-written figure measures nothing.
    """
    if _GPU_LANE:
        return
    monkeypatch.setattr(
        vram,
        "device_memory",
        lambda: vram.DeviceMemory(total_gib=32.0, free_gib=32.0, name="Test GPU"),
    )


@pytest.fixture
def real_device_memory(monkeypatch):
    """The genuine reader, for the one test that is about what it reads.

    ``test_a_reading_never_raises`` puts an exploding stub in ``sys.modules``
    and asserts the reader swallows it; the pin above would answer that with a
    figure taken from no device at all.
    """
    monkeypatch.setattr(vram, "device_memory", _REAL_DEVICE_MEMORY)


@pytest.fixture(autouse=True)
def _unpublished_device_reading(monkeypatch):
    """``vram._published`` is a module global and ``publish`` has no undo.

    Not the pin above but the same rule one level down, and it is what the pin
    made reachable: ``real_device_memory`` hands the genuine reader back, and
    the genuine reader *falls back* to whatever was last published. So
    ``test_t2i_client.py`` publishing a stale ``(1.0, 2.0, "stale")`` reading
    and never taking it away is enough to make ``test_vram.py``'s
    reading-never-raises check see a figure where it asserts None.

    Pre-existing rather than introduced: the two files fail together on
    unmodified code too. It had simply never been observed, because
    ``--dist loadfile`` keeps them in different workers and one file on its own
    is clean -- which is the same reason the device pin above went unnoticed
    for as long as it did.

    ``monkeypatch.setattr`` rather than a bare assignment, so the reset happens
    on the way in *and* the original is restored on the way out. Not exempt in
    the gpu lane: that lane wants a real card, which is the live torch path,
    and that path already wins over a published reading.
    """
    monkeypatch.setattr(vram, "_published", None, raising=False)


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """A WarlockService over a throwaway data dir, with no worker.

    No worker on purpose: wake_worker becomes a no-op and attach_progress
    reports None, which is exactly the shape the service functions have to
    tolerate anyway (the UI reads jobs before the queue has anything to say
    about them). Tests that need dispatch drive the Worker directly.
    """
    import warlock.config as config_mod
    from warlock.config import get_config
    from warlock.service import WarlockService

    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path / "warlock-home"))
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("WARLOCK_DB", str(tmp_path / "assets" / "jobs.sqlite"))
    # Points at a nonexistent exe; nothing here ever runs a job.
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    # And gltfpack is pinned *absent* rather than left to the machine. Its
    # default is PROJECT_ROOT/vendor/gltfpack/gltfpack.exe and vendor/ is
    # gitignored, so whether a named triangle tier is refused depended on
    # whether whoever ran the suite happened to have vendored the binary --
    # which is exactly the "a test about the fallback must pin the fallback"
    # rule docs/INVARIANTS.md states for warlockc.dll. Vendoring gltfpack on
    # 2026-08-07 duly turned two admission tests red without a line of their subject
    # changing. A test that wants the binary *present* writes one.
    monkeypatch.setenv("WARLOCK_GLTFPACK", str(tmp_path / "no-gltfpack.exe"))
    # And the trellis weights directory, for the same reason: the bg_removal
    # default is gated on birefnet.gguf being in it (guidance.default_bg_removal),
    # so leaving it pointed at PROJECT_ROOT/models would make every submitted
    # job's matte depend on which weights this machine happens to have
    # downloaded. Empty here; a test that wants the learned matte writes the file.
    monkeypatch.setenv("WARLOCK_TRELLIS_MODELS", str(tmp_path / "trellis-models"))
    # And the host model root, for the third time and the same reason. This one
    # hid behind a bug: pipelines/matting fed a float32 tensor to an fp16
    # checkpoint, so every "model" matte raised and fell back to the corner
    # fill in milliseconds. With that fixed, leaving this pointed at
    # PROJECT_ROOT/models means any 2D export in the suite does a real ~12 s
    # BiRefNet inference per image on a machine that happens to have the
    # weights -- tests/test_derive_2d.py alone burned 4624 CPU-seconds -- and,
    # worse, produces a *different matte* there than on one that does not.
    # Empty here; a test that wants the model writes the files or patches
    # matting.available, which tests/test_inspector_exports.py already does.
    monkeypatch.setenv("WARLOCK_T2I_ROOT", str(tmp_path / "t2i-models"))
    # And the bench directory, for the fourth time -- but this one is not about
    # a test reading the machine's state, it is about a test *writing* over it.
    # bench_dir defaults to PROJECT_ROOT/bench, and service.findings.refresh
    # writes findings.json under it, so any test that recomputed findings wrote
    # into the real bench/ -- replacing a 299 KB corpus with whatever three
    # verdicts the test had just invented. That file is what the generate panes
    # read for their accept-rate hints, so `uv run pytest` silently blanked the
    # evidence on screen. Derived rather than lost (service.findings.refresh
    # rebuilds it from the verdicts table), which is precisely why it went
    # unnoticed: nothing was destroyed that could not be recomputed, and nothing
    # recomputed it.
    monkeypatch.setenv("WARLOCK_BENCH_DIR", str(tmp_path / "bench"))
    # And the palette directory, for the fifth and last time. Its default is now
    # under the *user's home* rather than the checkout, so a suite that left it
    # unset would read whatever palettes the developer happens to own -- and
    # tests/test_panes_mtime_guard.py works around precisely that today.
    monkeypatch.setenv("WARLOCK_PALETTE_DIR", str(tmp_path / "palettes"))
    monkeypatch.setattr(config_mod, "_config", None)
    config = get_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # And then the *generative* half of that root is populated with empty files
    # at exactly the paths ``fetch.present`` probes (F55). ``create_job`` now
    # refuses a text job whose selected weights are not on the host, which is
    # the right answer for a user and the wrong default for a suite: with the
    # root pinned empty above, every text job in every test would be refused
    # for a reason none of them is about.
    #
    # Deliberately not the matting, pose or metric models, which the guard does
    # not probe and whose *absence* several tests depend on -- the comment above
    # is explicit that a real BiRefNet load is what pinning this root exists to
    # prevent, and materialising one would put a directory back where those
    # tests look for nothing. A test about a missing download deletes what it
    # wants gone; a test about a job that runs no longer has to know any of
    # this exists.
    _materialize_generative_weights(config)
    s = JobStore(config.db_path)
    yield WarlockService(config, s)
    s.close()


@pytest.fixture
def materialize_weights():
    """:func:`_materialize_generative_weights`, for the files that build their
    own ``Config`` instead of taking ``svc`` and so need the same treatment."""
    return _materialize_generative_weights


def _materialize_generative_weights(config) -> None:
    """Placeholder files at every path ``fetch.present`` probes for a
    *selectable* model: the checkpoints, their step-distillation LoRAs, the
    style LoRAs, the IP-Adapters and the ControlNets.

    One byte rather than zero, and the difference is now load-bearing. These
    were empty on the reasoning that nothing here is ever loaded and a probe is
    an existence test -- true until MDL-08, which made a *zero-length* weight
    file mean something specific: a download that did not finish, or a copy that
    lost its contents. ``fetch.suspect_files`` reports those, and a fixture
    writing them would have every model in the suite permanently flagged as
    corrupt. Still nothing that can be loaded; just not a file that claims to be
    broken.

    Driven off the registries rather than a hardcoded list, so adding a model to
    ``models.py`` does not silently start refusing every text job in the suite.
    """
    from warlock import fetch, models

    root = config.t2i_model_root

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"x")

    for spec in models.BASE_MODELS.values():
        base = fetch.base_model_dir(config, spec)
        touch(base / "model_index.json")
        variant = f".{spec.variant}" if spec.variant else ""
        for rel in spec.probe or (f"unet/diffusion_pytorch_model{variant}.safetensors",):
            touch(base / rel)
        if spec.base_lora:
            touch(root / "loras" / spec.base_lora)
    for lora in models.STYLE_LORAS.values():
        touch(root / "loras" / lora.filename)
    for adapter in models.IP_ADAPTERS.values():
        touch(root / adapter.dir_name / adapter.subfolder / adapter.weight_name)
        touch(root / adapter.dir_name / adapter.image_encoder_dir / "config.json")
    for cn in models.CONTROLNETS.values():
        variant = f".{cn.variant}" if cn.variant else ""
        touch(root / cn.dir_name / "config.json")
        touch(root / cn.dir_name / f"diffusion_pytorch_model{variant}.safetensors")
    for expander in models.EXPANDER_MODELS.values():
        # Generative-path like the four above: check_weights probes it when a
        # job turns expansion on, and a submit-shaped test must not be refused
        # for a download it is not about. A test about the refusal unlinks it.
        touch(root / expander.dir_name / "config.json")
        touch(root / expander.dir_name / "pytorch_model.bin")


@pytest.fixture(scope="session")
def gl():
    """A standalone GL 3.3 context, or a skip.

    Session-scoped because context creation is the expensive part and every
    renderer test is read-only about the context itself. Skipped rather than
    failed where there is no GPU (CI, a remote shell): these tests are about
    what the driver draws, and there is nothing to learn without one.
    """
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_context(standalone=True, require=330)
    except Exception as exc:  # no display, no driver, software-only
        pytest.skip(f"no GL 3.3 context: {exc}")
    yield ctx
    ctx.release()


@functools.lru_cache(maxsize=1)
def _tiny_glb() -> bytes:
    """A structurally valid GLB carrying one real cube.

    ART-02. The fakes used to write the markers ``b"fake-glb"`` and
    ``b"fake-png"``, and everything downstream of them is wrapped in
    catch-everything handlers -- ``normalize_glb``, the mesh audit, the report,
    every 2D derive. So a queue test asserting ``status == "done"`` was, for the
    most part, asserting that the *degraded* path completes: the artifact never
    parsed, every post-processing step raised and was swallowed, and the job
    reached ``done`` regardless. Which is exactly what happens when a real
    reconstruction comes back malformed -- so the happy-path tests and the
    degraded-path tests were testing the same thing.

    Real geometry and not merely a parseable header, because that is what the
    steps under test need: ``normalize_glb`` measures extents and a mesh with no
    finite geometry is as unusable to it as a byte string. Built with trimesh
    (a core dependency, so no extra is required) and cached, since it is
    identical for every call and every test.

    Malformed artifacts are still exercised, deliberately, by the tests that ask
    for them -- ``should_raise`` and the explicit degraded-path cases.
    """
    import trimesh

    return trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(file_type="glb")


class FakeTrellisServer:
    """Stands in for TrellisServer: no subprocess, no GPU, no HTTP."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.running = False
        self.last_used = 0.0
        self.on_line = None
        self.stop_calls = 0
        # Which thread each stop() ran on. The real stop() blocks for up to
        # ~20 s (terminate, wait, join), so every call site must dispatch it
        # off the event loop; this is how the tests can tell.
        self.stop_threads: list[int] = []
        self.generate_calls: list[dict] = []
        self.slices = 5
        self.sleep_per_slice = 0.005
        self.should_raise: Exception | None = None
        # When True, stop() no longer stands in for "the subprocess died and
        # unblocked the in-flight request" -- it just records that it was
        # called. Combined with a long generate() run, this simulates a
        # trellis-server.exe that ignores termination, so the only thing that
        # can end the run is Worker.shutdown()'s forced task.cancel() fallback.
        self.ignore_stop = False
        # The launch config, mirrored from the real server's constructor
        # defaults, plus a record of every ensure_config call and the thread it
        # ran on -- ensure_config blocks (it calls stop), so like stop it must
        # never run on the event loop.
        self.tex_res = 512
        self.band: int | None = None
        self.config_calls: list[tuple[int, int | None]] = []
        self.config_threads: list[int] = []
        self.restarts = 0

    def ensure_config(self, *, tex_res: int, band: int | None) -> bool:
        self.config_calls.append((tex_res, band))
        self.config_threads.append(threading.get_ident())
        changed = (self.tex_res, self.band) != (tex_res, band)
        self.tex_res, self.band = tex_res, band
        if changed and self.running:
            self.restarts += 1
            self.stop()
            return True
        return False

    async def generate(
        self,
        image_path: Path,
        output_path: Path,
        *,
        seed: int = 42,
        resolution: int = 1024,
        bg_removal: str | None = None,
    ) -> Path:
        self.generate_calls.append(
            {
                "image_path": image_path,
                "seed": seed,
                "resolution": resolution,
                "bg_removal": bg_removal,
            }
        )
        if self.should_raise is not None:
            exc, self.should_raise = self.should_raise, None
            raise exc
        self.running = True
        for _ in range(self.slices):
            await asyncio.sleep(self.sleep_per_slice)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Stage-and-rename, matching the real client's _atomic_write: every
        # writer of source.glb replaces the directory entry rather than
        # rewriting the inode, and the remesh staging hard-links against that
        # contract (C37) -- a fake that wrote in place would model a writer
        # the app does not have and scribble through the link.
        tmp = output_path.with_suffix(".glb.tmp")
        tmp.write_bytes(_tiny_glb())
        os.replace(tmp, output_path)
        self.last_used = time.monotonic()
        return output_path

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_threads.append(threading.get_ident())
        if self.ignore_stop:
            return
        self.running = False


class FakeText2Image:
    """Stands in for Text2Image: no torch, no diffusers."""

    def __init__(self, spec=None, *_args, **_kwargs) -> None:
        # The real class takes a models.BaseModel; keep it so tests can assert
        # which base the worker constructed after a switch.
        self.spec = spec
        self.loaded = False
        self.last_used = 0.0
        self.unload_calls = 0
        self.trim_calls = 0
        self.unload_threads: list[int] = []
        self.steps = 3
        self.sleep_per_step = 0.005
        self.prompts: list[str] = []
        self.lora_calls: list[tuple] = []
        self.negatives: list[str | None] = []
        self.seeds: list[int] = []
        # Recorded per call, including the Nones: the bit-identity contract is
        # asserted at this boundary -- an unconditioned job must hand the
        # pipeline conditioning=None, not an empty Conditioning.
        self.conditionings: list = []
        self.tiles: list[bool] = []
        self.sheets: list[bool] = []
        self.scenes: list[bool] = []
        self.tilesheets: list[bool] = []
        # The frame each call asked for, including the Nones. A tile sheet is
        # sliced on a grid the caller decided in advance, so "what size did the
        # worker ask for" is the fact its tests turn on.
        self.sizes: list = []
        self.last_prompt = ""
        self.last_recipe: dict = {}

    def generate(
        self,
        prompt,
        output_path,
        *,
        seed=42,
        lora=None,
        lora_weight=DEFAULT_LORA_WEIGHT,
        negative_prompt=None,
        conditioning=None,
        on_state=None,
        on_step=None,
        cancel_event=None,
        tile=False,
        sheet=False,
        scene=False,
        tilesheet=False,
        size=None,
    ):
        self.prompts.append(prompt)
        self.tiles.append(tile)
        self.sheets.append(sheet)
        self.scenes.append(scene)
        self.tilesheets.append(tilesheet)
        self.sizes.append(size)
        self.last_prompt = prompt
        self.lora_calls.append((lora, lora_weight))
        self.negatives.append(negative_prompt)
        self.conditionings.append(conditioning)
        self.seeds.append(seed)
        if on_state is not None:
            on_state("load")
        self.loaded = True
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled
        if on_state is not None:
            on_state("sample")
        for i in range(self.steps):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled
            time.sleep(self.sleep_per_step)
            if on_step is not None:
                on_step(i + 1, self.steps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if tilesheet:
            # The one path whose caller slices what it wrote on a grid decided
            # before the call, so the *size* has to be honoured here or every
            # worker test would fail on a refusal about the frame rather than
            # on what it was actually testing. Sixty-four distinguishable
            # blocks rather than a flat colour: a reduction test that cannot
            # tell one tile from another proves nothing.
            from PIL import Image, ImageDraw

            width, height = size or (1024, 1024)
            image = Image.new("RGB", (width, height), (12, 12, 16))
            draw = ImageDraw.Draw(image)
            cell_w, cell_h = width // 8, height // 8
            for row in range(8):
                for col in range(8):
                    shade = 20 + 3 * (row * 8 + col)
                    draw.rectangle(
                        (
                            col * cell_w + 4,
                            row * cell_h + 4,
                            (col + 1) * cell_w - 5,
                            (row + 1) * cell_h - 5,
                        ),
                        fill=(shade, 255 - shade, (shade * 7) % 256),
                    )
            image.save(output_path, "PNG")
        elif sheet:
            # A sheet restyle's caller reopens and crops what it wrote, so this
            # one path has to produce a decodable image rather than a marker.
            # Deliberately a flat colour no source render uses, so a test can
            # tell "the generation landed here" from "the render did".
            from PIL import Image

            Image.new("RGB", (1024, 1024), (10, 200, 90)).save(output_path, "PNG")
        else:
            from PIL import Image, ImageDraw

            # A real PNG with a real subject on it, not a marker -- see
            # ``_tiny_glb`` for why. A *flat* image would parse and then be
            # refused by ``reference.measure`` ("No subject found against the
            # background"), which is a true answer about a picture of nothing;
            # the happy path needs a picture of something. Plain white ground
            # with one centred dark block: enough for the composition report to
            # find a subject, small enough to cost nothing.
            image = Image.new("RGB", (512, 512), (255, 255, 255))
            ImageDraw.Draw(image).rectangle((160, 160, 352, 352), fill=(40, 40, 48))
            image.save(output_path, "PNG")
        self.last_used = time.monotonic()
        return output_path

    def trim(self) -> None:
        # Releases cached VRAM without unloading: the pipe stays resident, so
        # `loaded` deliberately does not change here.
        self.trim_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1
        self.unload_threads.append(threading.get_ident())
        self.loaded = False


@pytest.fixture
def fake_pipelines(monkeypatch):
    """Patch the GPU pipeline classes at their definition. Worker.__init__
    constructs `TrellisServer(...)` via the name imported into queue.py's
    namespace; Worker._get_text2image does its imports fresh on every call, so
    patching the attribute on the defining module is picked up immediately
    without touching queue.py.

    **Both t2i names, not one.** Since 2026-08-22 the queue builds a
    `Text2ImageClient` -- a handle on a child process -- unless
    `config.t2i_in_process` is set, and patching only `Text2Image` left the
    fake unused: every test that reached this path spawned a real subprocess
    and waited on it. The fixture stands in for *whatever the queue
    constructs*, so it has to cover both sides of that switch."""
    import warlock.pipelines.t2i_client as t2i_client_mod
    import warlock.pipelines.text2image as text2image_mod
    import warlock.queue as queue_mod

    monkeypatch.setattr(queue_mod, "TrellisServer", FakeTrellisServer)
    monkeypatch.setattr(text2image_mod, "Text2Image", FakeText2Image)
    monkeypatch.setattr(t2i_client_mod, "Text2ImageClient", FakeText2Image)


@pytest.fixture(autouse=True)
def _reset_app_settings_session_flags():
    """``app_settings`` caches two once-per-session facts in module flags.

    ``_MEASURED`` (the model store's measured size) and ``_SWEPT`` (the
    cancelled-download staging sweep) are facts about *this process's disk*
    rather than preferences, which is why they are module-level and not ctx
    state. Tests get a fresh service per case and not a fresh module, so
    whichever case ran first decided the answer for every case after it.

    The reset hooks for exactly this existed with no caller -- the pollution
    the authors flagged in their own docstrings and then did not close. This is
    the caller.

    **Read out of ``sys.modules`` rather than imported.** An autouse fixture
    that imports a pane would pull imgui into every test process, including the
    ones that never touch the studio -- which changes module load order for the
    whole session to reset two flags that are already False in a process that
    has not loaded the pane.
    """
    import sys

    def reset() -> None:
        module = sys.modules.get("warlock.studio.panes.app_settings")
        if module is not None:
            module._reset_measure()
            module._reset_sweep()

    reset()
    yield
    reset()
