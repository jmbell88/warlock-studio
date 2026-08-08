"""Shared fixtures. fake_pipelines replaces the GPU-bound pieces (a real
trellis-server.exe subprocess, real torch/diffusers) with in-process fakes
so Worker's control flow -- cancellation, crash recovery, shutdown -- is
testable without a GPU."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

from warlock.db import JobStore
from warlock.models import DEFAULT_LORA_WEIGHT
from warlock.pipelines.text2image import JobCancelled


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

    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("WARLOCK_DB", str(tmp_path / "assets" / "jobs.sqlite"))
    # Points at a nonexistent exe; nothing here ever runs a job.
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    # And gltfpack is pinned *absent* rather than left to the machine. Its
    # default is PROJECT_ROOT/vendor/gltfpack/gltfpack.exe and vendor/ is
    # gitignored, so whether a named triangle tier is refused depended on
    # whether whoever ran the suite happened to have vendored the binary --
    # which is exactly the "a test about the fallback must pin the fallback"
    # rule CLAUDE.md states for warlockc.dll. Vendoring gltfpack on 2026-08-07
    # duly turned two admission tests red without a line of their subject
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
    """Empty files at every path ``fetch.present`` probes for a *selectable*
    model: the checkpoints, their step-distillation LoRAs, the style LoRAs, the
    IP-Adapters and the ControlNets.

    Empty on purpose -- nothing here is ever loaded, and a probe is an existence
    test. Driven off the registries rather than a hardcoded list, so adding a
    model to ``models.py`` does not silently start refusing every text job in
    the suite.
    """
    from warlock import fetch, models

    root = config.t2i_model_root

    def touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")

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
        self.sleep_per_slice = 0.02
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
        tmp.write_bytes(b"fake-glb")
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
        self.sleep_per_step = 0.02
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
        # The framing key travels beside the composed prompt rather than inside
        # it -- it fills PROMPT_TEMPLATE's view slot -- so it is recorded here
        # rather than being readable off ``prompts``.
        self.framings: list[str] = []
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
        framing="",
    ):
        self.prompts.append(prompt)
        self.tiles.append(tile)
        self.sheets.append(sheet)
        self.framings.append(framing)
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
        if sheet:
            # A sheet restyle's caller reopens and crops what it wrote, so this
            # one path has to produce a decodable image rather than a marker.
            # Deliberately a flat colour no source render uses, so a test can
            # tell "the generation landed here" from "the render did".
            from PIL import Image

            Image.new("RGB", (1024, 1024), (10, 200, 90)).save(output_path, "PNG")
        else:
            output_path.write_bytes(b"fake-png")
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
    namespace; Worker._get_text2image does `from .pipelines.text2image
    import Text2Image` fresh on every call, so patching the attribute on
    the text2image module is picked up immediately without touching queue.py."""
    import warlock.pipelines.text2image as text2image_mod
    import warlock.queue as queue_mod

    monkeypatch.setattr(queue_mod, "TrellisServer", FakeTrellisServer)
    monkeypatch.setattr(text2image_mod, "Text2Image", FakeText2Image)
