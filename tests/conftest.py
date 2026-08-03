"""Shared fixtures. fake_pipelines replaces the GPU-bound pieces (a real
trellis-server.exe subprocess, real torch/diffusers) with in-process fakes
so Worker's control flow -- cancellation, crash recovery, shutdown -- is
testable without a GPU."""

from __future__ import annotations

import asyncio
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
    monkeypatch.setattr(config_mod, "_config", None)
    config = get_config()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    s = JobStore(config.db_path)
    yield WarlockService(config, s)
    s.close()


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
        output_path.write_bytes(b"fake-glb")
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
        self.unload_threads: list[int] = []
        self.steps = 3
        self.sleep_per_step = 0.02
        self.prompts: list[str] = []
        self.lora_calls: list[tuple] = []
        self.negatives: list[str | None] = []
        self.seeds: list[int] = []
        self.last_prompt = ""

    def generate(
        self,
        prompt,
        output_path,
        *,
        seed=42,
        lora=None,
        lora_weight=DEFAULT_LORA_WEIGHT,
        negative_prompt=None,
        on_state=None,
        on_step=None,
        cancel_event=None,
    ):
        self.prompts.append(prompt)
        self.last_prompt = prompt
        self.lora_calls.append((lora, lora_weight))
        self.negatives.append(negative_prompt)
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
        output_path.write_bytes(b"fake-png")
        self.last_used = time.monotonic()
        return output_path

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
