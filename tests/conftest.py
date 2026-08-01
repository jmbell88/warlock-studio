"""Shared fixtures. fake_pipelines replaces the GPU-bound pieces (a real
trellis-server.exe subprocess, real torch/diffusers) with in-process fakes
so Worker's control flow -- cancellation, crash recovery, shutdown -- is
testable without a GPU."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from animancer3d.db import JobStore
from animancer3d.pipelines.text2image import JobCancelled


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


class FakeTrellisServer:
    """Stands in for TrellisServer: no subprocess, no GPU, no HTTP."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.running = False
        self.last_used = 0.0
        self.on_line = None
        self.stop_calls = 0
        self.generate_calls: list[dict] = []
        self.slices = 5
        self.sleep_per_slice = 0.02
        self.should_raise: Exception | None = None

    async def generate(
        self, image_path: Path, output_path: Path, *, seed: int = 42, resolution: int = 1024
    ) -> Path:
        self.generate_calls.append(
            {"image_path": image_path, "seed": seed, "resolution": resolution}
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
        self.running = False


class FakeText2Image:
    """Stands in for Text2Image: no torch, no diffusers."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.loaded = False
        self.unload_calls = 0
        self.steps = 3
        self.sleep_per_step = 0.02

    def generate(
        self,
        prompt,
        output_path,
        *,
        seed=42,
        on_state=None,
        on_step=None,
        cancel_event=None,
    ):
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
        return output_path

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


@pytest.fixture
def fake_pipelines(monkeypatch):
    """Patch the GPU pipeline classes at their definition. Worker.__init__
    constructs `TrellisServer(...)` via the name imported into queue.py's
    namespace; Worker._get_text2image does `from .pipelines.text2image
    import Text2Image` fresh on every call, so patching the attribute on
    the text2image module is picked up immediately without touching queue.py."""
    import animancer3d.pipelines.text2image as text2image_mod
    import animancer3d.queue as queue_mod

    monkeypatch.setattr(queue_mod, "TrellisServer", FakeTrellisServer)
    monkeypatch.setattr(text2image_mod, "Text2Image", FakeText2Image)
