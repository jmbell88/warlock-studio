"""Git's racily-clean rule, at the three pane caches that were missing it.

``inspector.rig_meta`` has applied it since it was written; ``palette_names``,
``_manifest`` and ``sheet_panel.pixel_record`` stamped a path's mtime and
remembered it unconditionally, which is the exact documented Windows hazard:
adding a file left a directory's mtime unchanged 155 times in 200 on this
machine (``docs/measurements/2026-08-07-directory-mtime-granularity.md``),
because the mtime comes from the system clock and its tick is 15.6 ms. So a
write landing after the read but inside the stamped mtime's own tick is
invisible to that stamp -- and invisible *forever*, since every later comparison
keeps matching the stale answer.

Each of the three caches exists for a change that is precisely that kind of
write: a palette dropped into the directory while the app runs, a derivation
rewriting manifest.json under an open Export tab, "Pixelate" pressed a second
time. So the failure mode is not theoretical for any of them -- it is the case
the cache was built for.

The tests come in pairs, and the second half of each pair is what makes the
first safe: inside the racy window nothing is remembered at all, so the answer
is re-read every frame at the cost of one stat, and a stamp is only stored once
its mtime is comfortably in the past.
"""

from __future__ import annotations

import json
import types
from typing import Any

from warlock.service.files import MTIME_RACE_NS
from warlock.studio.panes import inspector, sheet_panel, stamps
from warlock.studio.state import AppState


def _settled(monkeypatch, path):
    """Move the clock a tick past ``path``'s mtime, so a stamp may be stored."""
    settled = path.stat().st_mtime_ns + MTIME_RACE_NS * 2
    monkeypatch.setattr(stamps.time, "time_ns", lambda: settled)


def _frozen(monkeypatch, path):
    """Freeze the clock *at* ``path``'s mtime: the worst case the rule exists
    for, and the one where nothing may be remembered."""
    monkeypatch.setattr(stamps.time, "time_ns", lambda: path.stat().st_mtime_ns)


# --- the rule itself ----------------------------------------------------------


def test_a_stamp_from_the_current_tick_is_not_storable(tmp_path):
    (tmp_path / "a").write_text("a", encoding="utf-8")
    assert stamps.storable(stamps.stamp_ns(tmp_path)) is False


def test_an_absent_path_stamps_as_none_and_is_always_storable(tmp_path):
    """None races with nothing: a path appearing turns the stamp from None into
    a number whatever the clock happened to be doing."""
    assert stamps.stamp_ns(tmp_path / "nope") is None
    assert stamps.storable(None) is True


def test_a_backwards_clock_declines_to_cache(monkeypatch, tmp_path):
    """Slower, never wrong, which is the right way round."""
    (tmp_path / "a").write_text("a", encoding="utf-8")
    stamp = stamps.stamp_ns(tmp_path)
    monkeypatch.setattr(stamps.time, "time_ns", lambda: stamp - 10**9)
    assert stamps.storable(stamp) is False


# --- the palette directory ----------------------------------------------------


class _PaletteCtx:
    def __init__(self, svc):
        self.svc = svc
        self.state = AppState()


def _palette_dir(monkeypatch, svc, tmp_path, *, make=True):
    """Point the config at a throwaway palette directory.

    The note this replaces said ``palette_dir`` defaults to PROJECT_ROOT/palettes
    and that ``svc`` does not relocate it. Neither is true any more: the default
    moved under the app's home, and ``svc`` pins WARLOCK_PALETTE_DIR at
    ``tmp_path / "palettes"`` -- the very path this returned.

    That made ``make=False`` a no-op, and quietly. ``get_config()._ensure_dirs``
    creates ``palette_dir``, so the "missing directory" case was handed an
    *existing empty* one and passed for the wrong reason -- ``palette_names``
    answers ``[]`` to both. The absent case therefore names a directory nothing
    creates, which is the only way to tell the two apart.
    """
    directory = tmp_path / ("palettes" if make else "palettes-that-were-never-made")
    if make:
        directory.mkdir(parents=True, exist_ok=True)
    else:
        assert not directory.exists()
    monkeypatch.setattr(svc.config, "palette_dir", directory)
    return directory


def test_a_palette_dropped_in_inside_the_mtime_tick_is_still_offered(
    svc, monkeypatch, tmp_path
):
    """The one that matters: the whole point of stamping the directory is to
    notice a file the user drops into it, and an unguarded stamp hides exactly
    the drop that lands in its own tick -- permanently, from the one control
    whose purpose is to offer it."""
    ctx = _PaletteCtx(svc)
    directory = _palette_dir(monkeypatch, svc, tmp_path)
    (directory / "nes.hex").write_text("000000\nffffff\n", encoding="utf-8")
    _frozen(monkeypatch, directory)

    assert inspector.palette_names(ctx) == ["nes"]

    (directory / "gameboy.hex").write_text("081820\ne0f8d0\n", encoding="utf-8")

    assert inspector.palette_names(ctx) == ["gameboy", "nes"]


def test_a_palette_directory_touched_a_moment_ago_is_not_remembered(
    svc, monkeypatch, tmp_path
):
    ctx = _PaletteCtx(svc)
    directory = _palette_dir(monkeypatch, svc, tmp_path)
    (directory / "nes.hex").write_text("000000\n", encoding="utf-8")
    _frozen(monkeypatch, directory)

    assert inspector.palette_names(ctx) == ["nes"]
    assert ctx.state.palettes is None


def test_a_settled_palette_directory_is_listed_once(svc, monkeypatch, tmp_path):
    """The other half: once the stamp is safely in the past the walk happens
    once, which is what keeps a directory listing off the frame thread."""
    ctx = _PaletteCtx(svc)
    directory = _palette_dir(monkeypatch, svc, tmp_path)
    (directory / "nes.hex").write_text("000000\n", encoding="utf-8")
    _settled(monkeypatch, directory)
    walks: list[Any] = []

    from warlock.service import palettes as svc_palettes

    real = svc_palettes.available
    monkeypatch.setattr(
        svc_palettes, "available", lambda c: (walks.append(c), real(c))[1]
    )

    for _ in range(5):
        assert inspector.palette_names(ctx) == ["nes"]

    assert len(walks) == 1


def test_a_missing_palette_directory_is_no_palettes_and_no_error(svc, monkeypatch, tmp_path):
    ctx = _PaletteCtx(svc)
    _palette_dir(monkeypatch, svc, tmp_path, make=False)
    assert inspector.palette_names(ctx) == []
    assert ctx.state.palettes is None


# --- the export manifest ------------------------------------------------------


class _ManifestCtx:
    def __init__(self, root):
        self._root = root
        self.state = types.SimpleNamespace(manifest=None)

    def job_dir(self, job_id):
        return self._root


def test_a_manifest_rewritten_inside_the_mtime_tick_is_still_read(tmp_path, monkeypatch):
    """"A derivation running on the TaskRunner rewrites the manifest underneath
    it" is exactly a write that can land inside the stamped mtime's own tick,
    after which the Export tab would describe the derivation before last for
    the life of the process."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifacts": {"icon.png": {}}}), encoding="utf-8")
    ctx = _ManifestCtx(tmp_path)
    _frozen(monkeypatch, path)

    assert list(inspector._manifest(ctx, "abc123abc123")["artifacts"]) == ["icon.png"]

    path.write_text(json.dumps({"artifacts": {"sprite.png": {}}}), encoding="utf-8")

    assert list(inspector._manifest(ctx, "abc123abc123")["artifacts"]) == ["sprite.png"]


def test_a_manifest_written_a_moment_ago_is_not_remembered(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    ctx = _ManifestCtx(tmp_path)
    _frozen(monkeypatch, path)

    assert inspector._manifest(ctx, "abc123abc123") is not None
    assert ctx.state.manifest is None


# --- the pixel sheet's sidecar ------------------------------------------------


SHEET_ID = "abc123abc123"


class _SheetCtx:
    def __init__(self, root):
        self._root = root
        self.state = AppState()

    def job_dir(self, job_id):
        return self._root


def _write_sidecar(root, frame_size):
    from warlock import rigging

    path = rigging.sheet_pixel_path(root, SHEET_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"frame_size": frame_size, "palette": ["000000"]}), encoding="utf-8"
    )
    return path


def test_a_re_pixelated_sheet_inside_the_mtime_tick_is_still_read(tmp_path, monkeypatch):
    """"Pixelate" pressed twice rewrites this sidecar in place, so the panel
    would otherwise report the first restyle's cell size against the second
    restyle's PNG."""
    ctx = _SheetCtx(tmp_path)
    path = _write_sidecar(tmp_path, 64)
    _frozen(monkeypatch, path)

    assert sheet_panel.pixel_record(ctx, "job", SHEET_ID)["frame_size"] == 64

    _write_sidecar(tmp_path, 32)

    assert sheet_panel.pixel_record(ctx, "job", SHEET_ID)["frame_size"] == 32


def test_a_sidecar_written_a_moment_ago_is_not_remembered(tmp_path, monkeypatch):
    ctx = _SheetCtx(tmp_path)
    path = _write_sidecar(tmp_path, 64)
    _frozen(monkeypatch, path)

    assert sheet_panel.pixel_record(ctx, "job", SHEET_ID) is not None
    assert ctx.state.preview.get("pixel_sidecars") == {}


def test_a_settled_sidecar_is_parsed_once(tmp_path, monkeypatch):
    ctx = _SheetCtx(tmp_path)
    path = _write_sidecar(tmp_path, 64)
    _settled(monkeypatch, path)

    first = sheet_panel.pixel_record(ctx, "job", SHEET_ID)
    for _ in range(4):
        # Identity, not equality: a second parse produces an equal dict, and
        # parsing a file sixty times a second is what the cache exists to stop.
        assert sheet_panel.pixel_record(ctx, "job", SHEET_ID) is first


def test_a_missing_sidecar_is_no_record(tmp_path):
    assert sheet_panel.pixel_record(_SheetCtx(tmp_path), "job", SHEET_ID) is None


# --- the deformation QA sheet -------------------------------------------------


def _rigged(svc):
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    return job_id


def test_the_qa_sheet_is_probed_once_per_version_of_the_job_directory(svc, monkeypatch):
    """Two stats per frame for as long as the Rig & Pose tab was open. One stat
    on the directory answers both, because a file appearing or going moves the
    directory's own mtime -- ``attach_files``' argument, at two names."""
    job_id = _rigged(svc)
    job_dir = svc.job_dir(job_id)
    job = svc.store.get(job_id)
    (job_dir / "rig_qa.png").write_bytes(b"png-not-really")
    (job_dir / "rig_qa.json").write_text("{}", encoding="utf-8")
    _settled(monkeypatch, job_dir)
    probes: list[Any] = []
    real = inspector._deform_qa_probe
    monkeypatch.setattr(
        inspector, "_deform_qa_probe", lambda d: (probes.append(d), real(d))[1]
    )
    cache: dict = {}

    for _ in range(10):
        assert inspector.deform_qa_path(svc, job, cache=cache) == job_dir / "rig_qa.png"

    assert len(probes) == 1
    assert next(iter(cache)) == job_id


def test_a_qa_sheet_landing_inside_the_mtime_tick_is_still_found(svc, monkeypatch):
    job_id = _rigged(svc)
    job_dir = svc.job_dir(job_id)
    job = svc.store.get(job_id)
    _frozen(monkeypatch, job_dir)
    cache: dict = {}

    assert inspector.deform_qa_path(svc, job, cache=cache) is None
    assert cache == {}

    (job_dir / "rig_qa.png").write_bytes(b"png-not-really")
    (job_dir / "rig_qa.json").write_text("{}", encoding="utf-8")

    assert inspector.deform_qa_path(svc, job, cache=cache) == job_dir / "rig_qa.png"


def test_the_qa_gate_still_answers_without_a_cache(svc):
    """Optional rather than mandatory: the gate is also a plain question a test
    or a one-off caller asks once, and a cache is only worth its subtleties on
    the frame thread."""
    job_id = _rigged(svc)
    job_dir = svc.job_dir(job_id)
    job = svc.store.get(job_id)

    assert inspector.deform_qa_path(svc, job) is None
    (job_dir / "rig_qa.png").write_bytes(b"png-not-really")
    assert inspector.deform_qa_path(svc, job) is None
    (job_dir / "rig_qa.json").write_text("{}", encoding="utf-8")
    assert inspector.deform_qa_path(svc, job) == job_dir / "rig_qa.png"
