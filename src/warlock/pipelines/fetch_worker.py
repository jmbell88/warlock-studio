"""The one process in this project that is allowed to touch the network.

``python -m warlock.pipelines.fetch_worker``, spawned by
``service.downloads``, exactly as ``blender_worker`` is spawned by
``rigging.run_worker`` and for a closely related reason: what it does cannot be
undone inside the process that does it. ``bpy`` is process-global and crashes;
``HF_HUB_OFFLINE`` is read by ``huggingface_hub`` **at import time**, so an
in-process download would mean re-setting the variable and re-importing a
package half the app has already imported. A child sets it in its own
environment, downloads, and dies -- and the app process keeps
``HF_HUB_OFFLINE=1`` for its entire life, which is a property that can be
asserted rather than reviewed.

Two things it does that a bare ``hf download`` would not:

* It downloads into a **staging directory beside the destination** and moves
  the files in only once the fetch has returned. A failure -- no network, a
  gated repo, a full disk -- therefore leaves no half-populated model
  directory, which matters because every presence probe in ``warlock.fetch``
  answers "is this here" from a handful of filenames and a partial directory
  that happened to contain them would read as a finished download forever.
* It reports progress on stdout as one JSON object per line, so the pane can
  draw a bar for a 16 GB fetch. The measurement is bytes-on-disk in the
  staging directory against the declared size, sampled by a helper thread:
  crude, but it needs no ``huggingface_hub`` API beyond ``snapshot_download``
  and therefore cannot break on an upgrade.

Result goes to ``spec["result_path"]`` rather than stdout, following
blender_worker: a stray print from a third-party package can corrupt a stream
and must not be able to corrupt the answer.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Before anything imports huggingface_hub, and this line is the whole point of
# the module existing: warlock/__init__ has already set HF_HUB_OFFLINE=1 in
# *this* process (it is the first thing the package does, and that stays true),
# and this overrides it here and nowhere else. The parent never sees it.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# How often the progress sampler walks the staging directory. Half a second is
# far below what a person notices on a multi-minute download and far above what
# a directory walk costs.
SAMPLE_SECONDS = 0.5


def _emit(**payload: Any) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                # A file huggingface_hub is mid-way through renaming is not an
                # error, it is the normal state of a download. Progress is an
                # estimate; a raise here would fail a fetch that is working.
                continue
    return total


class _Sampler(threading.Thread):
    """Reports staging-directory bytes as a fraction of the declared size."""

    def __init__(self, staging: Path, size_gib: float) -> None:
        super().__init__(daemon=True, name="warlock-fetch-progress")
        self._staging = staging
        self._total = max(size_gib, 0.0) * float(1024**3)
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(SAMPLE_SECONDS):
            got = _dir_bytes(self._staging)
            gb = got / float(1024**3)
            if self._total > 0:
                # Capped below 1.0: the declared size is approximate, and a bar
                # that sits at 100% while the fetch is still running is worse
                # than one that sits at 99%.
                percent = min(99.0, 100.0 * got / self._total)
                label = f"{gb:.1f} of ~{self._total / float(1024**3):.1f} GB"
            else:
                percent = 0.0
                label = f"{gb:.1f} GB"
            _emit(percent=percent, label=label)

    def stop(self) -> None:
        self._stop.set()


def _move_into(staging: Path, dest: Path) -> list[str]:
    """Move the staged tree into ``dest``, creating directories as needed.

    Per file rather than one directory rename, because a destination
    legitimately already exists and is shared: ``loras/`` holds every adapter,
    and a second fetch into it must add files rather than replace the folder.
    """
    moved: list[str] = []
    for src in sorted(staging.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(staging)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        # replace(), not move(): an interrupted earlier attempt can leave a
        # file of the same name, and shutil.move onto an existing file raises
        # on some platforms and silently differs on others.
        os.replace(src, target)
        moved.append(str(rel).replace("\\", "/"))
    return moved


def fetch_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one fetch. Raises on failure, having removed the staging tree."""
    from huggingface_hub import snapshot_download

    dest = Path(spec["dest"])
    # Beside the destination rather than in the system temp directory, so the
    # final move is a rename on the same volume rather than a second full copy
    # of up to 16 GB -- and so the free-space check the host already made
    # against this volume is the one that matters.
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{dest.name}.fetch.part"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    patterns = list(spec.get("filenames") or []) + list(spec.get("allow_patterns") or [])
    ignore = list(spec.get("ignore_patterns") or [])
    sampler = _Sampler(staging, float(spec.get("size_gib") or 0.0))
    sampler.start()
    try:
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            local_dir=str(staging),
            allow_patterns=patterns or None,
            ignore_patterns=ignore or None,
        )
        rename = spec.get("rename")
        if rename:
            src, dst = rename
            staged = staging / src
            if not staged.exists():
                raise FileNotFoundError(
                    f"{spec['repo_id']} did not provide {src}, which this "
                    "download has to rename"
                )
            os.replace(staged, staging / dst)
        # huggingface_hub keeps its resume bookkeeping in a .cache/ subtree of
        # local_dir. Moving it into a model directory would leave a stray
        # directory every presence probe has to learn to ignore.
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        moved = _move_into(staging, dest)
    except BaseException:
        # Every failure, including a KeyboardInterrupt or a kill that unwinds:
        # the promise is that a failed download leaves no half-populated model
        # directory, and the staging tree is the whole of that promise.
        sampler.stop()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    sampler.stop()
    shutil.rmtree(staging, ignore_errors=True)
    return {"ok": True, "dest": str(dest), "files": moved}


def main() -> int:
    raw = sys.stdin.read()
    spec = json.loads(raw)
    result_path = Path(spec["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        result = fetch_one(spec)
    except Exception as exc:  # noqa: BLE001 -- the message is the product here
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result["seconds"] = round(time.perf_counter() - started, 2)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _emit(percent=100.0 if result.get("ok") else 0.0, label="")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
