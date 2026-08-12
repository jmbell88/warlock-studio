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
  the files in only once the fetch has returned -- all of them or none of
  them, the move itself included (``_move_into`` rolls back). A failure -- no
  network, a gated repo, a full disk -- therefore leaves no half-populated
  model directory, which matters because every presence probe in
  ``warlock.fetch`` answers "is this here" from a handful of filenames and a
  partial directory that happened to contain them would read as a finished
  download forever.
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

import contextlib
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
    """``publish.move_into``, plus the backup sweep a self-publishing fetch owes.

    The move itself lives in :mod:`warlock.publish` now, because the *parent*
    performs it when a selection is published as one transaction (MDL-10) and
    two copies of a rollback are two rollbacks. What stays here is the
    single-repository path's own cleanup: it has no journal, so a completed
    publish has nothing to keep the backup tree for.
    """
    from .. import publish

    names = publish.move_into(staging, dest)
    shutil.rmtree(publish.backup_dir(dest), ignore_errors=True)
    return names


def _verify_staged(staging: Path, spec: dict[str, Any]) -> None:
    """Re-hash the staged tree against the hub's own digests. Raises on a mismatch.

    The expected values are ``snapshot_download``'s ``.metadata`` sidecars,
    which is what makes this a check rather than a tautology: they are the
    ETags the *server* sent, recorded per file as it landed, so re-reading the
    bytes off disk and re-hashing them catches a truncated write, a full disk
    that silently short-wrote, and a file another process replaced between the
    download and the publish.

    A file with no sidecar is skipped rather than failed. Sidecars are written
    per download, so a resumed fetch that found a file already complete has no
    new one -- and "nothing to compare against" is not evidence of corruption.
    The failure this exists to catch is a file that *has* a recorded digest and
    does not match it.
    """
    from .. import hashes

    expected = hashes.expected_digests(staging)
    if not expected:
        return
    bad: list[str] = []
    for rel, digest in sorted(expected.items()):
        if not hashes.matches(staging / rel, digest):
            bad.append(rel)
    if bad:
        listed = ", ".join(bad[:4]) + ("..." if len(bad) > 4 else "")
        raise ValueError(
            f"{spec.get('repo_id')} downloaded {len(bad)} file(s) that do not "
            f"match the digests the hub recorded for them ({listed}). Nothing "
            f"was installed; try the download again."
        )


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
        # ``revision`` is the pin, and passing it is the whole of what makes
        # it one: unpinned, this follows whatever the repository's default
        # branch points at *today*, so a Download click a year from now can
        # retrieve different weights -- and, for BiRefNet, different Python,
        # which the matting pipeline then loads with
        # ``trust_remote_code=True`` (MDL-03). ``None`` for an unpinned entry
        # keeps the previous behaviour exactly rather than inventing a default.
        revision = str(spec.get("revision") or "") or None
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            revision=revision,
            local_dir=str(staging),
            allow_patterns=patterns or None,
            ignore_patterns=ignore or None,
        )
        # Verified here: after the download, before the rename, before the
        # .cache subtree that carries the expected digests is deleted, and
        # before anything is published (MDL-08).
        #
        # Mandatory rather than opt-in, and this is the one place it can be:
        # a truncated or corrupt file that reaches ``dest`` reads as a
        # finished model forever, because every presence probe is a handful of
        # ``Path.exists()`` calls. Raising here goes through the same unwind
        # every other failure does -- the staging tree is removed and nothing
        # is installed -- so a failed verify costs the download and nothing
        # else.
        _verify_staged(staging, spec)
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
        if not spec.get("publish", True):
            # No-publish mode (MDL-10). The parent is staging a whole selection
            # and will publish all of them together under one journal, so this
            # child stops with its tree complete, verified and marked. The
            # marker is what distinguishes "finished, waiting" from "was
            # interrupted", which the directory's existence cannot say.
            #
            # Deliberately outside the ``except`` unwind below: the staging
            # tree is the *product* here, not a temporary, so the usual
            # "remove it on any failure" rule would delete the answer.
            sampler.stop()
            return _stage_only(staging, dest, spec)
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
    _write_manifest(dest, spec, moved)
    return {"ok": True, "dest": str(dest), "files": moved}


def _stage_only(staging: Path, dest: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Finish a no-publish fetch: mark the staging tree and report it.

    The file list is computed here rather than by the parent so the child --
    which is the only process that knows what it downloaded -- is the one that
    says. The parent's publish walks the tree anyway; this is what its journal
    records *before* the walk, so a crash mid-publish has something to undo
    against.
    """
    from .. import publish

    names = sorted(
        str(p.relative_to(staging)).replace("\\", "/")
        for p in staging.rglob("*")
        if p.is_file()
    )
    publish.write_json(
        staging / publish.PUBLISH_NAME,
        {
            "repo_id": str(spec.get("repo_id") or ""),
            "revision": str(spec.get("revision") or ""),
            "dest": str(dest),
            "files": names,
        },
    )
    return {"ok": True, "staged": str(staging), "dest": str(dest), "files": names}


def _write_manifest(dest: Path, spec: dict[str, Any], moved: list[str]) -> None:
    """Record what this fetch published, and from which revision.

    Written *last*, after the publish succeeded, which is what makes it a
    completion marker rather than an intention: a fetch killed partway leaves
    no manifest, so "this directory was finished by a Warlock download" is a
    question with a real answer for the first time (MDL-08).

    Merged with whatever is already there rather than replacing it, because a
    destination is legitimately shared -- ``loras/`` accumulates adapters from
    several fetches, and a manifest that named only the last one would be a
    worse record than none.

    Never raises: a manifest that could not be written must not fail a download
    that succeeded. Its absence already means "unknown", which is exactly what
    would be true.
    """
    path = dest / ".warlock-fetch.json"
    from .. import hashes

    record = {
        "repo_id": str(spec.get("repo_id") or ""),
        "revision": str(spec.get("revision") or ""),
        "files": sorted(moved),
        # The digests a later ``fetch.verify_manifest`` compares against. Taken
        # from the *published* files rather than carried over from the staged
        # ones, so what is recorded is what is actually on disk at the path the
        # record names -- a move that silently truncated would otherwise be
        # certified by a hash of the file before it moved.
        "digests": hashes.digest_files(dest, moved),
    }
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        repos = existing.get("repos") if isinstance(existing, dict) else None
        repos = repos if isinstance(repos, dict) else {}
    except (OSError, ValueError):
        repos = {}
    repos[record["repo_id"]] = record
    with contextlib.suppress(OSError):
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps({"repos": repos}, indent=2), encoding="utf-8")
        os.replace(tmp, path)


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
