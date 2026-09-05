"""The pack worker's half: collecting, refusing, and what it hands pip.

No test here downloads anything or installs anything. What that leaves is
everything that decides whether a download is *accepted* -- the digest, the
already-collected short-circuit, the bundled-wheel refusal -- plus the argv pip
is actually given, which is where the offline guarantee lives.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from warlock.pipelines import pack_worker

BODY = b"a wheel-shaped pile of bytes"
DIGEST = hashlib.sha256(BODY).hexdigest()


def spec(tmp_path, *wheels, **kw):
    out = {"pack_dir": str(tmp_path), "wheels": list(wheels), "probe": []}
    out.update(kw)
    return out


@pytest.fixture
def has_pip(monkeypatch):
    """Stand in for the shipped runtime, which carries pip.

    The dev environment is a uv venv and does not, so without this every test
    below would be exercising the no-pip refusal rather than the thing it
    names -- which is exactly how that refusal came to be written.
    """
    real = pack_worker.importlib.util.find_spec
    monkeypatch.setattr(
        pack_worker.importlib.util,
        "find_spec",
        lambda name: object() if name == "pip" else real(name),
    )


def wheel(name: str, digest: str = DIGEST, **kw) -> dict:
    out = {
        "filename": name,
        "url": f"https://example.invalid/{name}",
        "sha256": digest,
        "size_bytes": len(BODY),
    }
    out.update(kw)
    return out


def test_a_wheel_already_collected_and_verified_is_not_fetched_again(tmp_path):
    """What makes a resumed install cheap, and the reason a 3 GB pack
    interrupted at 90% does not start over."""
    (tmp_path / "thing-1.0-py3-none-any.whl").write_bytes(BODY)
    got = pack_worker.collect(spec(tmp_path, wheel("thing-1.0-py3-none-any.whl")))
    assert got == ["thing-1.0-py3-none-any.whl"]


def test_a_collected_wheel_with_the_wrong_digest_is_re_fetched_not_trusted(tmp_path):
    """It is on disk under the right name and it is not the right file. The
    only safe reading of that is "not collected", and with nothing to fetch
    from here it must refuse rather than install it."""
    (tmp_path / "thing-1.0-py3-none-any.whl").write_bytes(b"something else entirely")
    with pytest.raises(ValueError):
        pack_worker.collect(
            spec(tmp_path, wheel("thing-1.0-py3-none-any.whl"), offline=True)
        )


def test_a_bundled_wheel_that_is_missing_says_the_install_is_incomplete(tmp_path):
    """It publishes no Windows wheel, so the build compiled it and the
    installer was supposed to carry it. There is nowhere to fetch it from, so
    "retry" is the wrong advice and "reinstall" is the right one."""
    with pytest.raises(ValueError) as caught:
        pack_worker.collect(
            spec(tmp_path, wheel("mojimoji-0.0.13-cp313-cp313-win_amd64.whl", bundled=True, url=""))
        )
    said = str(caught.value)
    assert "ships with the application" in said and "reinstall" in said.lower()


def test_a_bundled_wheel_that_is_present_is_simply_collected(tmp_path):
    (tmp_path / "mojimoji-0.0.13-cp313-cp313-win_amd64.whl").write_bytes(BODY)
    got = pack_worker.collect(
        spec(tmp_path, wheel("mojimoji-0.0.13-cp313-cp313-win_amd64.whl", bundled=True, url=""))
    )
    assert got == ["mojimoji-0.0.13-cp313-cp313-win_amd64.whl"]


def test_a_bundled_wheel_is_taken_from_where_the_installer_staged_it(tmp_path):
    """The wiring between ``installer/build.ps1`` and the cache directory.

    The three sdist-only distributions are compiled by the build and staged
    into the application's own ``packs`` directory; the cache the child downloads into is under the
    user's Warlock home, and on a per-user install those are routinely two
    drives. Without this the music pack could only ever fail three wheels from
    the end, on the user's machine, with everything else already downloaded.
    """
    staged = tmp_path / "app-packs"
    staged.mkdir()
    (staged / "mojimoji-0.0.13-cp313-cp313-win_amd64.whl").write_bytes(BODY)
    cache = tmp_path / "cache"
    cache.mkdir()
    got = pack_worker.collect(
        spec(
            cache,
            wheel("mojimoji-0.0.13-cp313-cp313-win_amd64.whl", bundled=True, url=""),
            bundled_dir=str(staged),
        )
    )
    assert got == ["mojimoji-0.0.13-cp313-cp313-win_amd64.whl"]
    # In the cache, because that is the one directory pip is given.
    assert (cache / "mojimoji-0.0.13-cp313-cp313-win_amd64.whl").read_bytes() == BODY


def test_a_staged_bundled_wheel_from_another_build_is_refused(tmp_path):
    """A bundled wheel is pinned by digest exactly as a downloaded one is: it
    is about to go into the site-packages the app is running out of, and a
    file left behind by a different build of Warlock is precisely the one that
    would otherwise be installed without anyone noticing."""
    staged = tmp_path / "app-packs"
    staged.mkdir()
    (staged / "mojimoji-0.0.13-cp313-cp313-win_amd64.whl").write_bytes(b"a different build")
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError) as caught:
        pack_worker.collect(
            spec(
                cache,
                wheel("mojimoji-0.0.13-cp313-cp313-win_amd64.whl", bundled=True, url=""),
                bundled_dir=str(staged),
            )
        )
    assert "digest" in str(caught.value)
    assert not list(cache.iterdir()), "a refused wheel must leave nothing behind"


def test_the_install_cannot_reach_the_network_or_re_resolve(tmp_path, monkeypatch, has_pip):
    """``--no-index`` confines pip to the pack directory and ``--no-deps``
    gives it the list rather than a problem to solve. Both are load-bearing:
    the pack carries the *delta* over the base runtime, so a pip allowed to
    resolve would go looking for numpy -- already installed, deliberately not
    in the pack -- and, given an index, would fetch a different one."""
    seen: dict[str, list[str]] = {}

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **_kw):
        seen["argv"] = list(argv)
        return Done()

    monkeypatch.setattr(pack_worker.winjob, "run", fake_run)
    pack_worker.install(spec(tmp_path), ["thing-1.0-py3-none-any.whl"])
    argv = seen["argv"]
    assert "--no-index" in argv
    assert "--no-deps" in argv
    assert "--find-links" in argv
    assert argv[argv.index("--find-links") + 1] == str(tmp_path)
    assert argv[-1] == str(tmp_path / "thing-1.0-py3-none-any.whl")
    assert "--index-url" not in argv and "--extra-index-url" not in argv


def test_a_runtime_with_no_pip_says_what_to_do_instead(tmp_path, monkeypatch):
    """The shipped runtime has pip -- the installer stages a uv-managed CPython,
    which carries it. A *uv venv* does not, and that is what a source checkout
    runs on, so this fires exactly where the right answer is "you have uv, use
    it" rather than "install pip"."""
    monkeypatch.setattr(pack_worker.importlib.util, "find_spec", lambda _n: None)
    with pytest.raises(ValueError) as caught:
        pack_worker.install(spec(tmp_path), ["thing.whl"])
    assert "no pip" in str(caught.value) and "uv sync" in str(caught.value)


def test_a_pip_that_fails_is_reported_in_its_last_words(tmp_path, monkeypatch, has_pip):
    class Done:
        returncode = 1
        stdout = ""
        stderr = "ERROR: Could not install packages due to an OSError\n"

    monkeypatch.setattr(pack_worker.winjob, "run", lambda *a, **k: Done())
    with pytest.raises(ValueError) as caught:
        pack_worker.install(spec(tmp_path), ["thing.whl"])
    assert "OSError" in str(caught.value)


def test_an_install_that_did_not_deliver_is_a_failure_not_a_success(tmp_path, monkeypatch):
    """The failure a user would otherwise meet later, as a mode that is still
    greyed out after an install that said it worked."""
    monkeypatch.setattr(pack_worker, "install", lambda *a, **k: {"installed": []})
    (tmp_path / "thing-1.0-py3-none-any.whl").write_bytes(BODY)
    with pytest.raises(ValueError) as caught:
        pack_worker.run(
            spec(
                tmp_path,
                wheel("thing-1.0-py3-none-any.whl"),
                probe=["a_module_that_is_not_installed_anywhere"],
            )
        )
    assert "still cannot be imported" in str(caught.value)


def test_collect_only_never_installs(tmp_path, monkeypatch):
    def _never(*_a, **_k):
        raise AssertionError("collect_only must not reach pip")

    monkeypatch.setattr(pack_worker, "install", _never)
    (tmp_path / "thing-1.0-py3-none-any.whl").write_bytes(BODY)
    out = pack_worker.run(
        spec(tmp_path, wheel("thing-1.0-py3-none-any.whl"), collect_only=True)
    )
    assert out == {"ok": True, "collected": ["thing-1.0-py3-none-any.whl"], "installed": []}


def test_the_probe_answers_about_the_filesystem_not_this_process(tmp_path):
    """``verify`` runs in the process that just did the writing, and that
    process has already been told by its own import system that the module was
    absent. Without ``invalidate_caches`` a correct install reads as a failed
    one."""
    assert pack_worker.verify({"probe": ["json", "hashlib"]}) == []
    assert pack_worker.verify({"probe": ["not_a_real_module_xyz"]}) == [
        "not_a_real_module_xyz"
    ]


def test_progress_lines_are_one_json_object_each(tmp_path, capsys):
    pack_worker._emit(percent=12.5, label="hello")
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {"percent": 12.5, "label": "hello"}


@pytest.mark.parametrize(
    "rate, remaining, wanted",
    [
        (0.0, 0.0, ""),  # silent below 64 KB/s: "~9h left" on a stalled fetch is worse
        (32 * 1024, 1024, ""),
        (10 * 1024**2, 0, " at 10 MB/s"),
        (10 * 1024**2, 100 * 1024**2, " at 10 MB/s, ~10s left"),
        (10 * 1024**2, 6000 * 1024**2, " at 10 MB/s, ~10m left"),
    ],
)
def test_the_pace_reads_the_same_as_a_model_download(rate, remaining, wanted):
    """Deliberately ``fetch_worker._pace``'s words and threshold: the two bars
    are read by the same person in the same pane."""
    assert pack_worker._pace(rate, remaining) == wanted


def test_the_two_workers_pace_functions_have_not_drifted():
    """Compared as source rather than by importing the other worker, because
    importing ``fetch_worker`` sets ``HF_HUB_OFFLINE=0`` in whatever process
    does it -- which is precisely the thing that module exists to keep out of
    every process but its own child.

    The claim is that a download's speed and ETA read identically whether the
    bytes are a checkpoint or a wheel. Two copies of a formatter is how that
    stops being true, one release at a time.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "warlock" / "pipelines"

    def body(path: Path) -> str:
        text = (src / path).read_text(encoding="utf-8")
        found = re.search(r"\ndef _pace\(.*?\n(?=\n\ndef |\n\nclass )", text, re.S)
        assert found, f"no _pace in {path}"
        # The docstrings differ (each explains itself in its own module); the
        # code must not.
        return re.sub(r'""".*?"""', "", found.group(0), flags=re.S).strip()

    assert body(Path("pack_worker.py")) == body(Path("fetch_worker.py"))
