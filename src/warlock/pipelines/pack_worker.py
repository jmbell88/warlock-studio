"""The second process allowed to touch the network, and the only one that installs.

``python -m warlock.pipelines.pack_worker``, spawned by ``service.packs``,
following ``fetch_worker`` in every respect that matters: spec on stdin,
progress as one JSON object per line on stdout, the answer in a file, and the
kill-on-close job around it. It exists for the same reason that one does --
what it does cannot be undone inside the process that does it.

Here that reason is sharper than an import-time environment variable. This
child **writes into the application's own ``site-packages``**. Doing that from
inside the running app would mean a process editing the library it is running
out of, with an imgui frame on screen and a GL context live; the failure modes
run from a half-written package to an import that succeeds and returns
something from two versions at once. A child does the writing, dies, and the
app learns what changed by asking ``find_spec`` again -- which is a question
about the filesystem, not about this process's import history.

Two things it does that a bare ``pip install`` would not:

* **It downloads before it installs, and verifies every byte first.** Each
  wheel is streamed to a ``.part`` beside its final name, hashed as it goes,
  and renamed only when the digest matches what the manifest pins. So an
  interrupted download leaves nothing that reads as collected, and the install
  phase begins with every file present and checked -- the point at which
  failing is still free.
* **The install cannot reach the network.** ``--no-index`` and
  ``--find-links <pack dir>`` mean pip resolves against the collected wheels
  and nothing else, and ``--no-deps`` means it installs exactly the list it
  was given. The manifest *is* the resolution -- it came out of the same lock
  the base runtime was built from -- so a resolver asking a second question
  here could only get a different answer, never a better one.

pip rather than uv because pip is what a user's machine has: the installer
stages a uv-managed CPython, which carries pip in its own ``site-packages``,
and ``installer/build.ps1`` already removes the ``EXTERNALLY-MANAGED`` marker
that would otherwise make it refuse. Vendoring uv would be a second binary to
pin, licence and ship for a job the runtime can already do.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from .. import winjob

CHUNK = 1 << 20

# Wall-clock ceiling for the install phase alone. pip unpacking 3 GB of torch
# onto a slow disk is genuinely minutes; a pip parked forever is not.
INSTALL_TIMEOUT = 60 * 60.0


def _emit(**payload: Any) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _pace(rate: float, remaining: float) -> str:
    """" at 12 MB/s, ~4m left", or nothing when there is nothing to say.

    ``fetch_worker._pace``'s rule and its threshold, deliberately the same
    words: these two bars are read by the same person in the same pane, and a
    download that describes itself differently depending on what is being
    downloaded is a worse bar, not a more informative one.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundled(spec: dict[str, Any], name: str) -> Path | None:
    """The staged copy of a wheel that cannot be downloaded, if it is there."""
    where = str(spec.get("bundled_dir") or "")
    if not where:
        return None
    found = Path(where) / name
    return found if found.is_file() else None


def collect(spec: dict[str, Any]) -> list[str]:
    """Download every wheel that is not already present and verified.

    Progress is reported against the *whole* plan rather than per file, because
    what a person watching a 3 GB pack wants to know is how long the pack will
    take, not how long torch will take. A wheel already on disk with the right
    digest counts as done and is never re-fetched, which is what makes a
    resumed install cheap.

    A digest mismatch removes the file and raises. It is never retried here: a
    wheel that arrived wrong twice is a fact about the network or the mirror,
    and quietly trying again is how a corrupt download becomes an install.
    """
    pack_dir = Path(spec["pack_dir"])
    pack_dir.mkdir(parents=True, exist_ok=True)
    wheels = list(spec["wheels"])
    total = sum(int(w.get("size_bytes") or 0) for w in wheels)
    done = 0
    have: list[str] = []
    started = time.monotonic()
    for wheel in wheels:
        name = str(wheel["filename"])
        target = pack_dir / name
        size = int(wheel.get("size_bytes") or 0)
        digest = str(wheel["sha256"]).lower()
        if target.exists() and _sha256(target) == digest:
            done += size
            have.append(name)
            continue
        if wheel.get("bundled"):
            # It publishes no Windows wheel, so the build compiled it and the
            # installer carries it beside the manifest. Copied into the cache
            # rather than installed from where it lies: pip is given one
            # ``--find-links`` directory and the collected set has to be whole
            # in it, and a copy is cheap next to the gigabytes around it.
            #
            # There is nowhere to fetch it from, so its absence means the
            # install is broken rather than merely uncollected -- named as
            # such, rather than failing later with a URL error about an empty
            # string.
            source = _bundled(spec, name)
            if source is None:
                raise ValueError(
                    f"{name} ships with the application and is in neither "
                    f"{pack_dir} nor the directory the installer stages. This "
                    f"installation is incomplete; reinstall Warlock rather "
                    f"than retrying."
                )
            if _sha256(source) != digest:
                # The digest is checked on the *staged* file for the reason it
                # is checked on a downloaded one: what is about to go into
                # site-packages must be what this build pinned, and a wheel
                # from a different build of the app is exactly the file that
                # would otherwise pass unnoticed here.
                raise ValueError(
                    f"{name} ships with the application and does not match "
                    f"the digest this build pins. This installation is "
                    f"damaged; reinstall Warlock."
                )
            staged = target.with_name(target.name + ".part")
            shutil.copyfile(source, staged)
            staged.replace(target)
            done += size
            have.append(name)
            continue
        if spec.get("offline") or not str(wheel.get("url") or ""):
            raise ValueError(f"{name} is not collected and this install may not download")
        staging = target.with_name(target.name + ".part")
        running = hashlib.sha256()
        got = 0
        with (
            urllib.request.urlopen(str(wheel["url"]), timeout=60) as response,
            staging.open("wb") as handle,
        ):
            while chunk := response.read(CHUNK):
                running.update(chunk)
                handle.write(chunk)
                got += len(chunk)
                span = time.monotonic() - started
                rate = (done + got) / span if span > 0 else 0.0
                # Capped below the install phase's own share: a bar that
                # reaches 100% while pip is still unpacking 3 GB is worse than
                # one that sits at 90% and then says what it is doing.
                percent = min(89.0, 90.0 * (done + got) / total) if total > 0 else 0.0
                _emit(
                    percent=percent,
                    label=(
                        f"{(done + got) / float(1024**3):.1f} of "
                        f"~{total / float(1024**3):.1f} GB"
                        + _pace(rate, total - done - got)
                    ),
                )
        if running.hexdigest() != digest:
            staging.unlink(missing_ok=True)
            raise ValueError(
                f"{name} downloaded with digest {running.hexdigest()}, which is "
                f"not the {digest} this build pins. Nothing was installed."
            )
        # Renamed only once the digest matches, so nothing that reads as
        # collected was ever unverified -- the rule ``fetch_worker`` states for
        # a model directory, applied to one file.
        staging.replace(target)
        done += size
        have.append(name)
    return have


def install(spec: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Hand the collected wheels to pip, offline, with the resolution decided.

    ``--no-index`` and ``--find-links`` confine pip to the pack directory;
    ``--no-deps`` gives it the list rather than a problem to solve. Both are
    load-bearing: the pack carries the *delta* over the base runtime, so a pip
    allowed to resolve would go looking for numpy -- which is already
    installed and deliberately not in the pack -- and, given an index, would
    fetch a different one.
    """
    pack_dir = Path(spec["pack_dir"])
    if importlib.util.find_spec("pip") is None:
        # The shipped runtime has pip: `installer/build.ps1` stages a
        # uv-managed CPython, which carries it in its own site-packages, and
        # removes the EXTERNALLY-MANAGED marker that would make it refuse. A
        # *uv venv* does not, which is what a source checkout runs on -- so
        # this fires exactly where the right answer is not "install pip" but
        # "you have uv, use it", and says so rather than dying with
        # ModuleNotFoundError from inside a subprocess.
        raise ValueError(
            "this runtime has no pip, so packs cannot be installed into it. "
            "In a source checkout install the extras with uv instead: "
            "uv sync --extra studio --extra text2image --extra rig --extra music"
        )
    argv = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-input",
        "--disable-pip-version-check",
        "--no-deps",
        "--find-links",
        str(pack_dir),
        *[str(pack_dir / name) for name in names],
    ]
    _emit(percent=92.0, label=f"installing {len(names)} packages")
    # ``winjob.run`` rather than ``subprocess.run``: this worker is itself in
    # the kill-on-close job, and while Windows does put a job'd process's
    # children in the same job, the guarantee here is stated rather than
    # inherited -- which is what ``test_vram.py``'s spawn scan enforces, and
    # what caught this line. A pip stranded halfway through unpacking torch
    # would leave the application's own site-packages half written.
    done = winjob.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=INSTALL_TIMEOUT,
    )
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()
        raise ValueError(
            "pip could not install the pack: " + (tail[-1] if tail else "no output")
        )
    return {"installed": names}


def verify(spec: dict[str, Any]) -> list[str]:
    """Which of the pack's own imports still do not resolve after installing.

    Asked here, in the process that just did the writing, because it is the one
    that knows the install finished -- and asked through ``invalidate_caches``
    because this interpreter has already been told, by its own import system,
    that those modules were absent. An empty list is the pack working; anything
    else is an install that reported success and did not deliver, which is the
    failure a user would otherwise meet later as a greyed-out mode.
    """
    importlib.invalidate_caches()
    missing: list[str] = []
    for name in spec.get("probe") or []:
        try:
            found = importlib.util.find_spec(str(name)) is not None
        except Exception:  # noqa: BLE001 -- a probe must never raise out of a result
            found = False
        if not found:
            missing.append(str(name))
    return missing


def run(spec: dict[str, Any]) -> dict[str, Any]:
    names = collect(spec)
    if spec.get("collect_only"):
        return {"ok": True, "collected": names, "installed": []}
    result = install(spec, names)
    missing = verify(spec)
    if missing:
        raise ValueError(
            "the pack installed but "
            + ", ".join(missing)
            + " still cannot be imported; the application may need restarting"
        )
    return {"ok": True, "collected": names, **result}


def main() -> int:
    spec = json.loads(sys.stdin.read())
    result_path = Path(spec["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        result = run(spec)
    except Exception as exc:  # noqa: BLE001 -- the message is the product here
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    result["seconds"] = round(time.perf_counter() - started, 2)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _emit(percent=100.0 if result.get("ok") else 0.0, label="")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
