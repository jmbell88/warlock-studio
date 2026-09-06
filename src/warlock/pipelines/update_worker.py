"""The third process allowed to touch the network, and the only one that runs nothing.

``python -m warlock.pipelines.update_worker``, spawned by ``service.updates``,
following ``fetch_worker`` and ``pack_worker`` in every respect that matters:
spec on stdin, progress as one JSON object per line on stdout, the answer in a
file, and the kill-on-close job around it. It exists for the reason the other
two do -- the app process stays offline, checkably, and the one thing that
talks to a socket is a child that dies when it is done.

It is the *narrowest* of the three. ``fetch_worker`` writes a model tree,
``pack_worker`` writes into the ``site-packages`` the app is running out of;
this one writes exactly one file, an installer, into a staging directory that
nothing else reads, and it never runs it. Handing the user a verified ``.exe``
and a button that opens it is a deliberate stopping point: an installer that
relaunched itself would mean this process arranging its own termination while
an imgui frame is on screen and a GL context is live, for no gain a
double-click does not already give.

**Nothing is trusted that was not published.** The release's own
``update-manifest.json`` asset carries the version, the installer's filename,
its size and its sha256 -- the same shape ``packs.json`` pins a wheel with --
and the download URL is read off the asset list GitHub actually returned,
never composed from the filename by a naming convention this side happens to
believe in. A release with no such asset is not an error: it predates the
feature, or opted out, and the honest answer is "there is nothing to offer"
rather than a guess at where its installer might live.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

from . import download

CHUNK = 1 << 20

#: The asset a release has to carry for this app to offer it. Named here
#: because ``scripts/make_update_manifest.py`` writes it and this reads it, and
#: a release whose two halves disagree offers nothing at all.
MANIFEST_ASSET = "update-manifest.json"

#: Where "what is the latest release" is asked. A literal rather than a
#: setting: an update feed a user can point elsewhere is an installer a user
#: can be pointed at, and this app has no signing story that would make that
#: safe.
RELEASES_URL = "https://api.github.com/repos/jmbell88/warlock-studio/releases/latest"

# The check is two small GETs. A minute is already generous for that, and a
# check parked on a stalled socket holds a task-pool worker.
CHECK_TIMEOUT = 60.0


def _emit(**payload: Any) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _pace(rate: float, remaining: float) -> str:
    """" at 12 MB/s, ~4m left", or nothing when there is nothing to say.

    ``pack_worker._pace``'s rule and its words, deliberately: these bars are
    read by the same person in the same pane, and a download that describes
    itself differently depending on what is being downloaded is a worse bar.
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


def _get_json(url: str, *, timeout: float = CHECK_TIMEOUT) -> Any:
    # Not a bare urlopen: see pipelines/download.py. GitHub serves the default
    # agent today, which is exactly the reason to spell it once for all three
    # workers rather than per host.
    with download.open_url(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _asset(assets: Any, name: str) -> dict[str, Any] | None:
    """The published asset with this exact name, or None."""
    for entry in assets or []:
        if isinstance(entry, dict) and str(entry.get("name") or "") == name:
            return entry
    return None


def check(spec: dict[str, Any]) -> dict[str, Any]:
    """What the latest release is, if it published a manifest for us.

    ``latest`` is None when the release carries no ``update-manifest.json``.
    That is a normal answer, not a failure: every release published before this
    feature existed is in that state, and the parent turns it into "you're up
    to date" rather than into an error the user has to interpret.
    """
    url = str(spec.get("releases_url") or RELEASES_URL)
    try:
        release = _get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        # GitHub answers ``/releases/latest`` with 404 when a repository has
        # published none -- which was the state of this one the day this was
        # written, and is the permanent state of anybody's fork. It is not a
        # network failure and must not be reported as one: "there is nothing
        # to offer" is the same answer as a release with no manifest.
        return {"ok": True, "latest": None, "release_url": ""}
    if not isinstance(release, dict):
        raise ValueError("the release feed did not describe a release")
    assets = release.get("assets")
    release_url = str(release.get("html_url") or "")
    found = _asset(assets, MANIFEST_ASSET)
    if found is None:
        return {"ok": True, "latest": None, "release_url": release_url}
    manifest = _get_json(str(found.get("browser_download_url") or ""))
    if not isinstance(manifest, dict):
        raise ValueError(f"{MANIFEST_ASSET} is not an object")
    installer = manifest.get("installer")
    if not isinstance(installer, dict):
        raise ValueError(f"{MANIFEST_ASSET} names no installer")
    filename = str(installer.get("filename") or "")
    digest = str(installer.get("sha256") or "").lower()
    if not filename or not digest:
        raise ValueError(f"{MANIFEST_ASSET} names no installer filename or digest")
    # The URL is read off what GitHub published, never composed from the
    # filename: a convention this side believes in is a URL nobody uploaded,
    # and the failure mode of guessing right is worse than of guessing wrong.
    published = _asset(assets, filename)
    if published is None:
        raise ValueError(f"the release publishes no asset called {filename}")
    return {
        "ok": True,
        "latest": str(manifest.get("version") or release.get("tag_name") or ""),
        "installer_name": filename,
        "installer_url": str(published.get("browser_download_url") or ""),
        "size_bytes": int(installer.get("size_bytes") or published.get("size") or 0),
        "sha256": digest,
        "release_url": release_url,
    }


def fetch(spec: dict[str, Any]) -> dict[str, Any]:
    """Stream the installer to a staging name and rename it only if it verifies.

    ``pack_worker.collect``'s loop for one file, and its rule about a mismatch:
    the ``.part`` is deleted and this raises, terminally. A file that arrived
    wrong is a fact about the transfer, and retrying it here is how a corrupt
    download becomes something a user double-clicks.
    """
    dest_dir = Path(spec["dest_dir"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = str(spec["installer_name"])
    digest = str(spec["sha256"]).lower()
    total = int(spec.get("size_bytes") or 0)
    target = dest_dir / name
    staging = target.with_name("." + name + ".part")
    running = hashlib.sha256()
    got = 0
    started = time.monotonic()
    _emit(percent=0.0, label="")
    with (
        download.open_url(str(spec["installer_url"]), timeout=60) as response,
        staging.open("wb") as handle,
    ):
        while chunk := response.read(CHUNK):
            running.update(chunk)
            handle.write(chunk)
            got += len(chunk)
            span = time.monotonic() - started
            rate = got / span if span > 0 else 0.0
            # Capped below 100 until the digest has been checked: a bar that
            # reads finished before anything has been verified is a bar that
            # says "ready" about a file that may be about to be deleted.
            percent = min(99.0, 100.0 * got / total) if total > 0 else 0.0
            _emit(
                percent=percent,
                label=(
                    f"{got / float(1024**2):.0f} of ~{total / float(1024**2):.0f} MB"
                    + _pace(rate, total - got)
                ),
            )
    if running.hexdigest() != digest:
        staging.unlink(missing_ok=True)
        raise ValueError(
            f"{name} downloaded with digest {running.hexdigest()}, which is not "
            f"the {digest} the release publishes. Nothing was kept."
        )
    staging.replace(target)
    return {"ok": True, "path": str(target)}


def run(spec: dict[str, Any]) -> dict[str, Any]:
    mode = str(spec.get("mode") or "check")
    if mode == "check":
        return check(spec)
    if mode == "download":
        return fetch(spec)
    raise ValueError(f"unknown update mode {mode!r}")


def main() -> int:
    spec = json.loads(sys.stdin.read())
    result_path = Path(spec["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        result = run(spec)
    except Exception as exc:  # noqa: BLE001 -- the message is the product here
        # ``resume_note`` overridden because this download does not resume:
        # nothing that failed here was kept, so the fetch worker's "pressing
        # Install again continues from where it stopped" would be a lie about
        # the one thing the sentence is for.
        result = {
            "ok": False,
            "error": download.describe_failure(
                exc, resume_note="Nothing was kept; try again when the connection is better."
            ),
        }
    result["seconds"] = round(time.perf_counter() - started, 2)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _emit(percent=100.0 if result.get("ok") else 0.0, label="")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
