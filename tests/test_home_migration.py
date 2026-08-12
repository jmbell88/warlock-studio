"""``~/.warlock`` is the app's home, and getting there is a one-time move.

Two subjects in one file because they are one change: where the roots resolve
(``config``) and what happens to a library still sitting at the old address
(``migrate``). Every test here points ``PROJECT_ROOT`` and the home at
``tmp_path`` -- the suite's autouse ``_no_migration`` fixture exists so that a
file which *forgets* to cannot touch the developer's real library, and each test
below opts back in with an explicit ``delenv``.
"""

from __future__ import annotations

import collections
import shutil
import sqlite3
from pathlib import Path

import pytest

from warlock import migrate
from warlock.config import SETTINGS, Config

# Every variable that would override a default. Cleared wholesale, because the
# machine running the suite is the machine the feature exists for and is
# therefore the machine most likely to have half of these set.
_ROOT_VARS = (
    "WARLOCK_HOME",
    "WARLOCK_DATA_DIR",
    "WARLOCK_DB",
    "WARLOCK_BENCH_DIR",
    "WARLOCK_PALETTE_DIR",
    "WARLOCK_T2I_ROOT",
    "WARLOCK_TRELLIS_MODELS",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake ``Path.home()`` and a clean environment. -> the home directory."""
    for var in _ROOT_VARS:
        monkeypatch.delenv(var, raising=False)
    fake = tmp_path / "home"
    fake.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake))
    return fake / ".warlock"


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    """A checkout with a populated library at the old addresses. -> its root."""
    monkeypatch.delenv("WARLOCK_NO_MIGRATE", raising=False)
    monkeypatch.delenv("WARLOCK_MIGRATE_KEEP", raising=False)
    root = tmp_path / "checkout"
    (root / "assets" / "abcdef012345").mkdir(parents=True)
    (root / "assets" / "abcdef012345" / "model.glb").write_bytes(b"mesh")
    conn = sqlite3.connect(str(root / "assets" / "jobs.sqlite"))
    conn.execute("CREATE TABLE jobs (id TEXT)")
    conn.commit()
    conn.close()
    (root / "bench").mkdir()
    (root / "bench" / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "palettes").mkdir()
    (root / "palettes" / "nes.hex").write_text("000000", encoding="utf-8")
    (root / "models").mkdir()
    (root / "models" / "weights.safetensors").write_bytes(b"w" * 32)
    monkeypatch.setattr(migrate, "PROJECT_ROOT", root)
    return root


# --- where the roots resolve ------------------------------------------------


def test_every_generated_root_defaults_under_the_home(home):
    config = Config()
    assert config.home == home
    assert config.data_dir == home / "assets"
    assert config.db_path == home / "assets" / "jobs.sqlite"
    assert config.bench_dir == home / "bench"
    assert config.palette_dir == home / "palettes"
    assert config.t2i_model_root == home / "models"
    assert config.trellis_models_dir == home / "models" / "trellis2-gguf"


def test_the_vendored_binaries_stay_with_the_checkout(home):
    """They ship *with* the source tree; moving them would break a checkout
    that has them and fix nothing for one that does not."""
    from warlock.config import PROJECT_ROOT

    config = Config()
    assert config.trellis_server_exe.is_relative_to(PROJECT_ROOT)
    assert config.gltfpack_exe.is_relative_to(PROJECT_ROOT)


def test_warlock_home_moves_every_root_at_once(tmp_path, home, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("WARLOCK_HOME", str(elsewhere))
    config = Config()
    assert config.home == elsewhere
    assert config.data_dir == elsewhere / "assets"
    assert config.db_path == elsewhere / "assets" / "jobs.sqlite"
    assert config.bench_dir == elsewhere / "bench"
    assert config.palette_dir == elsewhere / "palettes"
    assert config.t2i_model_root == elsewhere / "models"


def test_a_per_root_variable_still_wins_over_the_home(tmp_path, home, monkeypatch):
    """The point of resolving the home per field rather than once: an existing
    setup that pinned one directory keeps it."""
    monkeypatch.setenv("WARLOCK_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("WARLOCK_T2I_ROOT", str(tmp_path / "big-disk" / "models"))
    config = Config()
    assert config.t2i_model_root == tmp_path / "big-disk" / "models"
    assert config.data_dir == tmp_path / "elsewhere" / "assets"


def test_home_is_a_listed_setting():
    """S140's table is asserted in both directions elsewhere; this is the
    readout actually naming the variable a user would reach for first."""
    assert ("home", "WARLOCK_HOME") in SETTINGS


# --- the move ---------------------------------------------------------------


def test_a_legacy_library_is_copied_verified_and_removed(home, legacy):
    config = Config()
    moved = migrate.run(config)

    assert len(moved) == 4
    assert (home / "assets" / "abcdef012345" / "model.glb").read_bytes() == b"mesh"
    assert (home / "assets" / "jobs.sqlite").exists()
    assert (home / "bench" / "manifest.json").exists()
    assert (home / "palettes" / "nes.hex").exists()
    assert (home / "models" / "weights.safetensors").exists()
    # And the source is gone, so the checkout is a checkout again.
    for name in ("assets", "bench", "palettes", "models"):
        assert not (legacy / name).exists()
    # No staging directory survives a successful move.
    assert not list(home.glob("*.incoming"))


def test_it_leaves_a_note_saying_where_everything_went(home, legacy):
    migrate.run(Config())
    note = (home / migrate.BREADCRUMB).read_text(encoding="utf-8")
    assert str(legacy / "assets") in note
    assert str(home / "assets") in note


def test_a_second_run_does_nothing(home, legacy):
    config = Config()
    migrate.run(config)
    (home / "assets" / "abcdef012345" / "model.glb").write_bytes(b"edited")
    assert migrate.run(config) == []
    assert (home / "assets" / "abcdef012345" / "model.glb").read_bytes() == b"edited"


def test_a_populated_destination_is_left_alone(home, legacy):
    """Two libraries have no join -- two ``jobs.sqlite`` files least of all --
    so a destination that already holds something is not this call's to touch."""
    (home / "assets").mkdir(parents=True)
    (home / "assets" / "jobs.sqlite").write_bytes(b"the newer one")

    moved = migrate.run(Config())

    assert str(home / "assets") not in moved
    assert (home / "assets" / "jobs.sqlite").read_bytes() == b"the newer one"
    assert (legacy / "assets").exists()
    # The others still moved: one occupied root does not veto the rest.
    assert (home / "models" / "weights.safetensors").exists()


def test_a_root_the_user_pinned_is_not_moved(home, legacy, monkeypatch, tmp_path):
    monkeypatch.setenv("WARLOCK_T2I_ROOT", str(tmp_path / "pinned"))
    migrate.run(Config())
    assert (legacy / "models" / "weights.safetensors").exists()
    assert not (tmp_path / "pinned").exists()
    assert (home / "assets").exists()


def test_a_home_pointed_back_at_the_checkout_migrates_nothing(legacy, monkeypatch):
    """The documented escape hatch. ``WARLOCK_HOME=<checkout>`` makes source
    and destination the same directory, and the move has to see that rather
    than copy a tree onto itself."""
    monkeypatch.setenv("WARLOCK_HOME", str(legacy))
    assert migrate.run(Config()) == []
    assert (legacy / "assets" / "jobs.sqlite").exists()


def test_no_migrate_examines_nothing(home, legacy, monkeypatch):
    monkeypatch.setenv("WARLOCK_NO_MIGRATE", "1")
    assert migrate.run(Config()) == []
    assert (legacy / "assets").exists()
    assert not home.exists()


def test_keep_runs_the_copy_but_not_the_delete(home, legacy, monkeypatch):
    monkeypatch.setenv("WARLOCK_MIGRATE_KEEP", "1")
    migrate.run(Config())
    assert (home / "assets" / "jobs.sqlite").exists()
    assert (legacy / "assets" / "jobs.sqlite").exists()


def test_too_little_room_refuses_and_moves_nothing(home, legacy, monkeypatch):
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: usage(1 << 40, 1 << 40, 0))

    with pytest.raises(migrate.MigrationError) as caught:
        migrate.run(Config())

    message = str(caught.value)
    assert "WARLOCK_HOME" in message  # the escape hatch is in the refusal
    assert (legacy / "assets" / "jobs.sqlite").exists()
    assert not (home / "assets").exists()


def test_a_live_writer_refuses_and_moves_nothing(home, legacy):
    """A second Warlock holding the library open. Tested by taking the lock the
    real one holds, rather than by the session marker -- a marker survives a
    crash and would wedge the migration permanently."""
    db = legacy / "assets" / "jobs.sqlite"
    db.unlink()
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE jobs (id TEXT)")
    conn.commit()
    conn.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(migrate.MigrationError) as caught:
            migrate.run(Config())
    finally:
        conn.close()

    assert "another Warlock" in str(caught.value)
    assert (legacy / "assets").exists()
    assert not (home / "assets").exists()


def test_a_copy_that_does_not_match_leaves_the_source_alone(home, legacy, monkeypatch):
    """The verify step, forced. Copy, recount, disagree -- and the rule is that
    nothing is deleted and no half-tree is left at the destination."""
    real = migrate._tree_size

    def miscounting(path):
        files, total = real(path)
        # One file more than the copy will ever contain, and only for the
        # source: this is what a copy that silently dropped something looks
        # like from the verify's side.
        return (files + 1, total) if path == legacy / "assets" else (files, total)

    monkeypatch.setattr(migrate, "_tree_size", miscounting)

    with pytest.raises(migrate.MigrationError):
        migrate.run(Config())

    assert (legacy / "assets" / "jobs.sqlite").exists()
    assert not (home / "assets").exists()
    assert not list(home.glob("*.incoming"))


def test_the_exclusive_hold_survives_the_copy_and_not_only_the_check(home, legacy, monkeypatch):
    """RUN-02: the guarantee has to cover the *copy*, not one instant before it.

    The precondition used to be ``BEGIN EXCLUSIVE`` / ``ROLLBACK`` on the legacy
    store, with the connection closed immediately -- so a second Warlock
    launched during the multi-minute cross-volume copy passed its own identical
    precondition, opened the store, and wrote into ``assets/`` while its
    contents were being copied out from under it. The recount catches a file
    that was *added*; an in-place write of equal size is invisible to it, and
    the legacy root is then deleted on top of those writes.

    Asserted from inside the copy, which is the only place the question is
    interesting: a second writer must be locked out *there*.
    """
    db = legacy / "assets" / "jobs.sqlite"
    db.unlink()
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE jobs (id TEXT)")
    conn.commit()
    conn.close()

    locked_out: list[bool] = []
    real_move = migrate._move

    def _watched(*args, **kwargs):
        other = sqlite3.connect(str(db), timeout=0)
        try:
            other.execute("BEGIN EXCLUSIVE")
            locked_out.append(False)
            other.execute("ROLLBACK")
        except sqlite3.OperationalError:
            locked_out.append(True)
        finally:
            other.close()
        return real_move(*args, **kwargs)

    monkeypatch.setattr(migrate, "_move", _watched)
    migrate.run(Config())

    assert locked_out and all(locked_out), (
        "a second process could take the legacy store while the copy was running"
    )
    # And the move still completed: holding the lock across the copy must not
    # stop the legacy tree being removed afterwards. The delete happens once the
    # hold is released, because Windows will not unlink a file with an open
    # handle -- and jobs.sqlite is inside the tree being deleted.
    assert (home / "assets" / "jobs.sqlite").exists()
    assert not (legacy / "assets").exists()
