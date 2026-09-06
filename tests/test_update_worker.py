"""The child that asks whether there is a newer Warlock, with the network stubbed.

Nothing in this file reaches the network: ``download.open_url`` is replaced
throughout, which is the same seam ``test_pack_worker``-shaped tests use and
the reason that helper exists as one function rather than a bare ``urlopen``
per call site.

The claims worth pinning are the two ways this could quietly go wrong. A
release whose installer URL is *composed* from a filename rather than read off
the published asset list would point at a URL nobody uploaded -- and might one
day point at one somebody else did. And a download whose digest is not checked
before the rename would leave a corrupt executable sitting under a name the
pane offers as "ready to run".
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from warlock.pipelines import download, update_worker

ASSET = update_worker.MANIFEST_ASSET
INSTALLER = "WarlockSetup-v0.0.37.exe"
INSTALLER_URL = "https://example.invalid/real/download/" + INSTALLER


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def serve(monkeypatch):
    """Map URL -> bytes, and refuse anything not in the map.

    Refusing is deliberate: a worker that composed a URL of its own would fail
    here rather than silently fetching something plausible.
    """
    routes: dict[str, bytes] = {}
    asked: list[str] = []

    def use(**more: bytes) -> dict[str, bytes]:
        routes.update(more)
        return routes

    def fake(url, *, timeout=None):
        asked.append(url)
        if url not in routes:
            raise AssertionError(f"the worker asked for a URL nobody published: {url}")
        return _Response(routes[url])

    monkeypatch.setattr(download, "open_url", fake)
    use.routes = routes  # type: ignore[attr-defined]
    use.asked = asked  # type: ignore[attr-defined]
    return use


def release(*, assets: list[dict], tag: str = "v0.0.37") -> bytes:
    return json.dumps(
        {"tag_name": tag, "html_url": "https://example.invalid/releases/" + tag, "assets": assets}
    ).encode()


def asset(name: str, url: str, size: int = 0) -> dict:
    return {"name": name, "browser_download_url": url, "size": size}


def manifest(*, version: str = "0.0.37", sha: str = "a" * 64, size: int = 10) -> bytes:
    return json.dumps(
        {
            "version": version,
            "installer": {"filename": INSTALLER, "size_bytes": size, "sha256": sha},
        }
    ).encode()


# --- check --------------------------------------------------------------------


def test_the_installer_url_is_read_off_the_published_assets(serve):
    """Not composed from the filename. The whole of the trust story starts here:
    a URL this side invented is a URL nobody uploaded."""
    serve(
        **{
            update_worker.RELEASES_URL: release(
                assets=[
                    asset(ASSET, "https://example.invalid/m.json"),
                    asset(INSTALLER, INSTALLER_URL, size=1234),
                ]
            ),
            "https://example.invalid/m.json": manifest(size=1234),
        }
    )
    out = update_worker.check({})
    assert out["ok"] is True
    assert out["latest"] == "0.0.37"
    assert out["installer_url"] == INSTALLER_URL
    assert out["installer_name"] == INSTALLER
    assert out["size_bytes"] == 1234
    assert out["sha256"] == "a" * 64
    assert out["release_url"].endswith("v0.0.37")


def test_a_release_with_no_manifest_asset_is_not_an_error(serve):
    """Every release published before this feature existed is in this state, and
    so is one that deliberately opts out. The honest answer is "nothing to
    offer", which the parent turns into "you're up to date"."""
    serve(**{update_worker.RELEASES_URL: release(assets=[asset(INSTALLER, INSTALLER_URL)])})
    out = update_worker.check({})
    assert out == {
        "ok": True,
        "latest": None,
        "release_url": "https://example.invalid/releases/v0.0.37",
    }


def test_a_repository_with_no_releases_at_all_is_not_an_error(monkeypatch):
    """GitHub answers ``/releases/latest`` with 404 for a repository that has
    published none -- the state of this one on 2026-09-05, and the permanent
    state of any fork. Reported as a transport failure it would tell a user
    with working internet to check their firewall."""
    import urllib.error

    def refuse(url, *, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(download, "open_url", refuse)
    assert update_worker.check({}) == {"ok": True, "latest": None, "release_url": ""}


def test_any_other_http_error_is_still_a_failure(monkeypatch):
    import urllib.error

    def refuse(url, *, timeout=None):
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(download, "open_url", refuse)
    with pytest.raises(urllib.error.HTTPError):
        update_worker.check({})


def test_a_manifest_naming_an_installer_the_release_never_published_refuses(serve):
    serve(
        **{
            update_worker.RELEASES_URL: release(
                assets=[asset(ASSET, "https://example.invalid/m.json")]
            ),
            "https://example.invalid/m.json": manifest(),
        }
    )
    with pytest.raises(ValueError, match="publishes no asset"):
        update_worker.check({})


def test_a_manifest_with_no_digest_refuses(serve):
    """An installer offered without a digest is an executable downloaded on
    trust, which is the one thing this file exists to prevent."""
    bad = json.dumps({"version": "0.0.37", "installer": {"filename": INSTALLER}}).encode()
    serve(
        **{
            update_worker.RELEASES_URL: release(
                assets=[asset(ASSET, "https://example.invalid/m.json")]
            ),
            "https://example.invalid/m.json": bad,
        }
    )
    with pytest.raises(ValueError, match="digest"):
        update_worker.check({})


# --- download -----------------------------------------------------------------


def spec(tmp_path: Path, payload: bytes, *, sha: str) -> dict:
    return {
        "mode": "download",
        "installer_url": INSTALLER_URL,
        "installer_name": INSTALLER,
        "size_bytes": len(payload),
        "sha256": sha,
        "dest_dir": str(tmp_path / "updates"),
    }


def test_a_verified_download_lands_under_its_real_name(serve, tmp_path, capsys):
    payload = b"MZ" + b"x" * 4096
    serve(**{INSTALLER_URL: payload})
    digest = hashlib.sha256(payload).hexdigest()
    out = update_worker.fetch(spec(tmp_path, payload, sha=digest))
    landed = Path(out["path"])
    assert landed.read_bytes() == payload
    assert landed.name == INSTALLER
    # Nothing is left claiming to be a partial download.
    assert list(landed.parent.iterdir()) == [landed]
    # And the bar never says finished before the digest has been checked.
    percents = [
        json.loads(line)["percent"]
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert max(percents) <= 99.0


def test_a_digest_mismatch_keeps_nothing_and_raises(serve, tmp_path):
    payload = b"MZ" + b"x" * 4096
    serve(**{INSTALLER_URL: payload})
    with pytest.raises(ValueError, match="digest"):
        update_worker.fetch(spec(tmp_path, payload, sha="b" * 64))
    # Neither the final name nor the staging name survives: a partial or wrong
    # installer under either is a file somebody could double-click.
    assert list((tmp_path / "updates").iterdir()) == []


def test_an_unknown_mode_refuses_rather_than_guessing():
    with pytest.raises(ValueError, match="unknown update mode"):
        update_worker.run({"mode": "install"})


def test_a_failed_run_does_not_promise_a_resume_that_does_not_exist(monkeypatch, tmp_path):
    """``fetch_worker``'s transport message ends "pressing Install again
    continues from where it stopped", which is true of a staged model tree and
    false here -- a failed installer download keeps nothing at all."""
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(json.dumps({"mode": "check", "result_path": str(result_path)})),
    )

    def boom(url, *, timeout=None):
        raise ConnectionResetError(104, "reset")

    monkeypatch.setattr(download, "open_url", boom)
    assert update_worker.main() == 1
    said = json.loads(result_path.read_text(encoding="utf-8"))["error"]
    assert "continues from where it stopped" not in said
    assert "Nothing was kept" in said
