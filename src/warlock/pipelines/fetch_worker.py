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
import traceback
from pathlib import Path
from typing import Any

from . import download

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


def _pace(rate: float, remaining: float) -> str:
    """" at 12 MB/s, ~4m left" -- or nothing, if there is nothing to say.

    A bar with no speed and no ETA is the difference between "this is working"
    and "this is finished in twenty minutes", which is the only question
    anyone has of a 16 GB fetch. Silent below 64 KB/s: a stalled or
    just-started fetch quoting "~9h left" is worse than quoting nothing.
    """
    if rate < 64 * 1024:
        return ""
    out = f" at {rate / float(1024**2):.0f} MB/s"
    if remaining > 0:
        seconds = int(remaining / rate)
        if seconds >= 3600:
            out += f", ~{seconds // 3600}h {(seconds % 3600) // 60}m left"
        elif seconds >= 60:
            out += f", ~{seconds // 60}m left"
        else:
            out += f", ~{max(seconds, 1)}s left"
    return out


class _Sampler(threading.Thread):
    """Reports staging-directory bytes as a fraction of the declared size."""

    def __init__(self, staging: Path, size_gib: float) -> None:
        super().__init__(daemon=True, name="warlock-fetch-progress")
        self._staging = staging
        self._total = max(size_gib, 0.0) * float(1024**3)
        self._stop = threading.Event()

    def run(self) -> None:
        # The previous sample, for a rate. An EWMA rather than the raw delta:
        # a fetch that stages several files reports in bursts, and an
        # instantaneous rate off one 0.5 s window swings between 0 and 200
        # MB/s -- which is a number nobody can read and an ETA nobody can use.
        last_at = time.monotonic()
        last_got = _dir_bytes(self._staging)
        rate = 0.0
        while not self._stop.wait(SAMPLE_SECONDS):
            got = _dir_bytes(self._staging)
            gb = got / float(1024**3)
            now = time.monotonic()
            span = now - last_at
            if span > 0:
                sample = max(got - last_got, 0) / span
                rate = sample if rate <= 0.0 else rate * 0.7 + sample * 0.3
            last_at, last_got = now, got
            if self._total > 0:
                # Capped below 1.0: the declared size is approximate, and a bar
                # that sits at 100% while the fetch is still running is worse
                # than one that sits at 99%.
                percent = min(99.0, 100.0 * got / self._total)
                label = f"{gb:.1f} of ~{self._total / float(1024**3):.1f} GB"
                label += _pace(rate, self._total - got)
            else:
                percent = 0.0
                label = f"{gb:.1f} GB" + _pace(rate, 0.0)
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
    shutil.rmtree(publish.backup_dir(dest, staging), ignore_errors=True)
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


def _fetch_url(staging: Path, spec: dict[str, Any]) -> None:
    """Download one file by URL into ``staging``, verifying its digest.

    **The digest is not optional**, and refusing without one is the point: this
    transport has no ``revision`` to pin it, so a spec that reached here with an
    empty ``sha256`` would be a download of whatever that URL serves today,
    installed as if it were pinned. Refusing is a bug in a registry entry
    surfacing at the one moment it can still be caught.

    Streamed in chunks and hashed as it goes, so a 320 MiB artifact is never
    held in memory twice, and the ``_Sampler`` watching the staging tree gets a
    growing file to measure exactly as it does for a Hub fetch.
    """
    import hashlib

    digest = str(spec.get("sha256") or "").lower()
    if not digest:
        raise ValueError(f"{spec.get('url')} has no sha256 to verify against")
    name = str(spec.get("filename") or "") or Path(str(spec["url"])).name
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        # It becomes a path. A registry entry is not user input, but this is
        # one join away from the model root and the check costs a line.
        raise ValueError(f"{name!r} is not a filename this fetch may write")

    out = staging / name
    running = hashlib.sha256()
    # Not a bare urlopen: the default agent is banned outright on at least one
    # host this project downloads from. See pipelines/download.py.
    with download.open_url(str(spec["url"]), timeout=60) as response, out.open("wb") as handle:
        while chunk := response.read(1 << 20):
            running.update(chunk)
            handle.write(chunk)
    if running.hexdigest() != digest:
        raise ValueError(
            f"{name} downloaded with digest {running.hexdigest()}, "
            f"which is not the {digest} this build pins"
        )


#: The resume marker, written inside the staging tree. Named for what it is and
#: deliberately *not* ``.warlock-fetch.json`` -- that name is the published
#: manifest in the destination, and one name meaning two things across two
#: directories is how a later reader comes to trust the wrong file.
RESUME_NAME = ".warlock-resume.json"

#: How many times one fetch re-enters the transport before giving up, and how
#: long it waits between attempts. Small and bounded: this is for the reset
#: that happens mid-transfer, not for a host that is down -- a gated repo or a
#: bad revision fails the same way every time and is refused terminally below.
#:
#: Measured need (2026-09-05): on the clean-machine install, one 16.1 GB engine
#: download ran eight minutes and died on a single Xet token refresh. With
#: resume beneath it, one retry would have carried it.
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (2.0, 8.0, 20.0)

#: Failures that must *not* be retried and must *not* leave a tree to resume
#: into. A digest mismatch means the bytes on disk are wrong, so resuming would
#: keep re-verifying the same bad file for ever; a missing rename source and a
#: registry entry with no digest are both faults in the request, identical on
#: every attempt.
#: ``download.AUTHORED`` is the same tuple seen from the other side -- the
#: exceptions this module raises itself, every one already carrying a sentence
#: written for a person, which is why they are both terminal here and passed
#: through untranslated there. One definition, because a divergence would mean
#: an exception that is retried *and* shown raw.
_TERMINAL = download.AUTHORED


def _resume_key(spec: dict[str, Any]) -> dict[str, Any]:
    """What a staging tree has to match before a later fetch may resume into it.

    Everything that decides *which bytes* belong in the tree. A tree staged for
    another revision, another file selection or another URL holds files this
    fetch did not ask for, and publishing it would install a mixture -- so the
    key is compared whole and a mismatch wipes rather than merges.
    """
    return {
        "repo_id": str(spec.get("repo_id") or ""),
        "revision": str(spec.get("revision") or ""),
        "filenames": sorted(str(x) for x in (spec.get("filenames") or ())),
        "allow_patterns": sorted(str(x) for x in (spec.get("allow_patterns") or ())),
        "ignore_patterns": sorted(str(x) for x in (spec.get("ignore_patterns") or ())),
        "url": str(spec.get("url") or ""),
        "sha256": str(spec.get("sha256") or ""),
    }


def _open_staging(dest: Path, spec: dict[str, Any]) -> Path:
    """The staging tree for this fetch, resumed if a matching one is already there.

    **Resuming is the whole of finding F1** (2026-09-05). Until this existed,
    every failure removed the tree, and ``huggingface_hub`` keeps its resume
    bookkeeping in ``.cache/`` *inside* ``local_dir`` -- so a failure discarded
    the ability to resume along with the bytes. On a connection that resets
    every few minutes a 16 GB model could never be downloaded, however many
    times Install was pressed, because every attempt restarted at zero.

    The tree is kept only when its marker matches :func:`_resume_key` exactly.
    An unreadable, absent or differing marker is treated as "not mine" and the
    tree is wiped: a partial download of the *wrong* revision is worse than no
    partial download, and ``_verify_staged`` only catches files the hub wrote a
    digest sidecar for.
    """
    staging = dest.parent / f".{dest.name}.fetch.part"
    want = _resume_key(spec)
    if staging.exists():
        marker = staging / RESUME_NAME
        try:
            found = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            found = None
        if found != want:
            shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        (staging / RESUME_NAME).write_text(json.dumps(want, indent=2), encoding="utf-8")
    return staging


def _with_retries(run: Any, spec: dict[str, Any]) -> None:
    """Call ``run`` until it works, up to :data:`RETRY_ATTEMPTS` times.

    Each attempt re-enters the transport over the *same* staging tree, so a
    retry resumes rather than restarts -- which is what makes a small number of
    attempts worth anything on a 16 GB download.

    ``_TERMINAL`` failures re-raise immediately: retrying them costs the user
    time to arrive at the same refusal. Everything else is retried, deliberately
    including exception types this module cannot name -- the transports raise
    ``httpx`` and ``hf_xet`` errors that are not ``OSError`` subclasses, and the
    reset this exists for (``ConnectError: [WinError 10054]``) is one of them.
    """
    # From the spec rather than an environment variable, so a caller that
    # wants one attempt (every test in this project, which must not sleep
    # through a real backoff) asks for it in the same JSON that carries
    # everything else the child is told.
    attempts = max(1, int(spec.get("retries") or RETRY_ATTEMPTS))
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            run()
            return
        except _TERMINAL:
            raise
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            last = exc
            if attempt == attempts - 1:
                break
            pause = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
            _emit(
                percent=-1.0,
                label=f"connection lost; retrying in {pause:.0f}s",
            )
            time.sleep(pause)
    assert last is not None
    raise last

def fetch_one(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one fetch. Raises on failure, keeping a resumable staging tree.

    The staging tree survives a failed attempt on purpose (F1, 2026-09-05) and
    the next fetch for the same spec resumes into it. What never survives is a
    half-populated *destination*: nothing moves until the whole tree is
    downloaded and verified, which is the promise the presence probes rest on
    and is untouched by this.
    """
    from huggingface_hub import snapshot_download

    dest = Path(spec["dest"])
    # Beside the destination rather than in the system temp directory, so the
    # final move is a rename on the same volume rather than a second full copy
    # of up to 16 GB -- and so the free-space check the host already made
    # against this volume is the one that matters.
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = _open_staging(dest, spec)

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
        if spec.get("url"):
            # The second transport. An artifact that is not on the Hub at all
            # -- today only the separation checkpoint -- and it rides *inside*
            # this child rather than beside it, so nothing else in the app
            # gains a way out to the network. Its digest does what ``revision``
            # does for a repo, and rather more strongly: a revision names an
            # immutable commit, a digest is the artifact. See ``models.Fetch``.
            #
            # It verifies itself, so it does not reach ``_verify_staged``
            # below -- that function reads ``snapshot_download``'s own
            # ``.metadata`` sidecars, which a plain HTTP GET never writes.
            _with_retries(lambda: _fetch_url(staging, spec), spec)
        else:
            revision = str(spec.get("revision") or "") or None
            # Retried over the same ``local_dir``, which is what makes the
            # retry worth having: ``snapshot_download`` reads its own
            # ``.cache/`` bookkeeping there and continues from where the last
            # attempt stopped rather than re-fetching what already landed.
            _with_retries(
                lambda: snapshot_download(
                    repo_id=str(spec["repo_id"]),
                    revision=revision,
                    local_dir=str(staging),
                    allow_patterns=patterns or None,
                    ignore_patterns=ignore or None,
                ),
                spec,
            )
            # Verified here: after the download, before the rename, before the
            # .cache subtree that carries the expected digests is deleted, and
            # before anything is published (MDL-08).
            #
            # Mandatory rather than opt-in, and this is the one place it can
            # be: a truncated or corrupt file that reaches ``dest`` reads as a
            # finished model forever, because every presence probe is a handful
            # of ``Path.exists()`` calls. Raising here goes through the same
            # unwind every other failure does -- the staging tree is removed
            # and nothing is installed -- so a failed verify costs the download
            # and nothing else.
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
        # local_dir, and this module keeps its resume marker beside it. Neither
        # belongs in a model directory: the first would be a stray tree every
        # presence probe has to learn to ignore, and the second would sit next
        # to the published manifest meaning something different.
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        (staging / RESUME_NAME).unlink(missing_ok=True)
        if not spec.get("publish", True):
            # No-publish mode (MDL-10). The parent is staging a whole selection
            # and will publish all of them together under one journal, so this
            # child stops with its tree complete, verified and marked. The
            # marker is what distinguishes "finished, waiting" from "was
            # interrupted", which the directory's existence cannot say.
            #
            # Deliberately outside the ``except`` unwind below: the staging
            # tree is the *product* here, not a temporary.
            sampler.stop()
            return _stage_only(staging, dest, spec)
        moved = _move_into(staging, dest)
    except _TERMINAL:
        # The bytes on disk are wrong or the request is, and both are identical
        # on every attempt -- so this tree must not be resumed into. Dropping
        # it is what stops a corrupt file being re-verified for ever.
        sampler.stop()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except BaseException:
        # Everything else, including a KeyboardInterrupt or a kill that
        # unwinds: **the tree is kept, and the next attempt resumes into it**
        # (F1, 2026-09-05). What the old unconditional ``rmtree`` protected was
        # the destination, and the destination is protected by nothing having
        # moved -- ``_move_into`` is the last statement in the try, and every
        # presence probe answers from ``dest``. A ``.fetch.part`` sibling is
        # invisible to all of them. The disk it holds is reclaimed by
        # ``downloads.sweep_staging`` and by the next mismatching fetch.
        sampler.stop()
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
        # ...and ``describe_failure`` is what finally makes that comment true
        # for a transport error rather than only for the exceptions this module
        # writes itself. See finding F2.
        # The friendly sentence is what the user sees; the raw exception is
        # carried beside it so the parent can put it in ``warlock.log``. Both,
        # because the whole of F2's diagnosis came out of a log file, and a
        # translation that threw the original away would have made this
        # incident harder to read rather than easier.
        traceback.print_exc(file=sys.stderr)
        result = {
            "ok": False,
            "error": download.describe_failure(exc),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    result["seconds"] = round(time.perf_counter() - started, 2)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _emit(percent=100.0 if result.get("ok") else 0.0, label="")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
