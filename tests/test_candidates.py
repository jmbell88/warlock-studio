"""Mesh candidates: the columns, the seed rule, admission, and the picker.

Everything here is headless. What a real N-candidate run costs is two minutes
of GPU per member and nothing in this file exercises that -- what it pins is
the part that can be wrong without a card: that membership is *columns* and
therefore cannot ride a params copy onto a reroll, that candidate 0 keeps the
seed the user pinned, that a refused submit leaves nothing behind at all, and
that Keep dissolves the group rather than leaving rows hidden with nothing to
reach them by.
"""

from __future__ import annotations

import sqlite3

import pytest

from warlock.db import _SCHEMA, MIGRATIONS, JobStore
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Conflict, Invalid, NotFound
from warlock.studio import candidates as candidates_mod
from warlock.studio.state import Filters

# --- the migration -----------------------------------------------------------


def test_the_candidate_columns_are_migration_six(tmp_path):
    """Append-only, and this entry landed first. If it ever stops being index
    5 (version 6), a shipped database has been rewritten rather than added to.
    """
    assert len(MIGRATIONS) == 6
    entry = " ".join(MIGRATIONS[5])
    assert "candidate_group" in entry and "candidate_index" in entry

    store = JobStore(tmp_path / "jobs.sqlite")
    store.close()
    conn = sqlite3.connect(tmp_path / "jobs.sqlite")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
    conn.close()


def test_a_database_that_predates_candidates_gains_the_columns(tmp_path):
    path = tmp_path / "jobs.sqlite"
    conn = sqlite3.connect(path)
    # The jobs table exactly as migration 5 left it: no candidate columns.
    conn.execute(
        "CREATE TABLE jobs ("
        " id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, prompt TEXT,"
        " params TEXT NOT NULL DEFAULT '{}', error TEXT, created_at REAL NOT NULL,"
        " started_at REAL, finished_at REAL, stage TEXT NOT NULL DEFAULT 'model',"
        " parent_id TEXT, name TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',"
        " favorite INTEGER NOT NULL DEFAULT 0, sweep_id TEXT,"
        " sweep_unit TEXT NOT NULL DEFAULT '')"
    )
    conn.execute("PRAGMA user_version = 5")
    conn.execute(
        "INSERT INTO jobs (id, kind, status, prompt, params, created_at)"
        " VALUES ('old', 'text', 'done', 'a barrel', '{}', 1.0)"
    )
    conn.commit()
    conn.close()

    store = JobStore(path)
    job = store.get("old")
    store.close()
    assert job is not None
    assert job["candidate_group"] is None
    assert job["candidate_index"] == 0


def test_a_fresh_database_and_a_migrated_one_agree_about_the_columns(tmp_path):
    """_SCHEMA carries the columns and so does the migration; a fresh DB must
    not end up with a different table shape from an upgraded one."""
    JobStore(tmp_path / "fresh.sqlite").close()
    hand = tmp_path / "hand.sqlite"
    conn = sqlite3.connect(hand)
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    JobStore(hand).close()

    def info(path):
        c = sqlite3.connect(path)
        out = c.execute("PRAGMA table_info(jobs)").fetchall()
        c.close()
        return out

    assert info(tmp_path / "fresh.sqlite") == info(hand)


# --- the columns themselves --------------------------------------------------


def test_the_columns_round_trip_and_the_group_reads_back_in_index_order(store):
    for i in (2, 0, 1):
        store.create("image", "a barrel", {}, f"job{i}", candidate_group="g1", candidate_index=i)
    store.create("image", "unrelated", {}, "plain")

    members = store.candidate_jobs("g1")
    assert [m["id"] for m in members] == ["job0", "job1", "job2"]
    assert [m["candidate_index"] for m in members] == [0, 1, 2]
    assert store.get("plain")["candidate_group"] is None


def test_resolving_a_group_clears_membership_and_keeps_the_index(store):
    for i in range(3):
        store.create("image", "a barrel", {}, f"job{i}", candidate_group="g1", candidate_index=i)
    assert store.resolve_candidates("g1") == 3
    assert store.candidate_jobs("g1") == []
    # The forensic half survives: the row still says which candidate it was.
    assert store.get("job2")["candidate_index"] == 2
    assert store.get("job2")["candidate_group"] is None


# --- the library filter ------------------------------------------------------


def test_the_library_hides_an_undecided_candidate_and_shows_a_decided_one():
    filters = Filters()
    undecided = {"id": "a", "status": "done", "stage": "model", "candidate_group": "g1"}
    decided = {"id": "a", "status": "done", "stage": "model", "candidate_group": None,
               "candidate_index": 1}
    assert filters.matches(undecided) is False
    assert filters.matches(decided) is True


# --- submission --------------------------------------------------------------


def _reference(svc, *, ok: bool = True) -> str:
    job_id = svc_jobs.create_job(
        svc, kind="text", prompt="a wooden chest", output="reference"
    )["id"]
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"x")
    svc.store.merge_params(job_id, {"reference_report": {"ok": ok, "reasons": ["too small"]}})
    svc.store.set_status(job_id, "done")
    return job_id


def test_one_candidate_is_an_ordinary_promotion_in_no_group(svc):
    source = _reference(svc)
    result = svc_jobs.promote_candidates(svc, source, count=1)
    assert result["ids"] == [result["id"]]
    assert result["group"] is None
    assert svc.store.get(result["id"])["candidate_group"] is None


def test_candidate_zero_keeps_the_requested_seed_and_the_rest_draw_fresh_ones(svc):
    source = _reference(svc)
    result = svc_jobs.promote_candidates(svc, source, count=3, mesh_seed=4242)
    rows = [svc.store.get(i) for i in result["ids"]]
    seeds = [r["params"]["mesh_seed"] for r in rows]
    assert seeds[0] == 4242
    assert 4242 not in seeds[1:]
    assert len(set(seeds)) == 3
    # seed follows mesh_seed on a promotion, so the two never disagree.
    assert [r["params"]["seed"] for r in rows] == seeds


def test_every_candidate_is_a_member_of_one_group_numbered_from_zero(svc):
    source = _reference(svc)
    result = svc_jobs.promote_candidates(svc, source, count=3)
    members = svc.store.candidate_jobs(result["group"])
    assert [m["id"] for m in members] == result["ids"]
    assert [m["candidate_index"] for m in members] == [0, 1, 2]
    # Ordinary model jobs otherwise: same stage, same parent lineage.
    assert {m["stage"] for m in members} == {"model"}
    assert {m["parent_id"] for m in members} == {source}


def test_the_count_is_bounded_at_the_door(svc):
    source = _reference(svc)
    for bad in (0, 4, 99):
        with pytest.raises(Invalid):
            svc_jobs.promote_candidates(svc, source, count=bad)
    assert svc.store.list(100) == [] or all(
        j["stage"] != "model" for j in svc.store.list(100)
    )


def test_admission_is_all_or_nothing(svc):
    """A refused group leaves no rows and no directories at all.

    ``promote_to_model`` refuses before it writes anything, and every candidate
    but the seed is the same job -- so a refusal can only land on candidate 0,
    before a single row exists. This is that property, not an implementation
    detail of it: the assertion is on what is in the store afterwards.
    """
    source = _reference(svc, ok=False)
    before = {j["id"] for j in svc.store.list(100)}
    with pytest.raises(Invalid):
        svc_jobs.promote_candidates(svc, source, count=3)
    assert {j["id"] for j in svc.store.list(100)} == before


def test_membership_cannot_leak_onto_a_reroll_or_a_second_promotion(svc):
    """The whole reason membership is columns. A params key would be copied by
    both of these and would have to be added to DERIVED_PARAMS to stop it."""
    source = _reference(svc)
    result = svc_jobs.promote_candidates(svc, source, count=2)
    winner = result["ids"][0]
    job_dir = svc.job_dir(winner)
    (job_dir / "input.png").write_bytes(b"x")
    (job_dir / "model.glb").write_bytes(b"x")
    svc.store.set_status(winner, "done")

    rerun = svc_jobs.rerun_job(svc, winner, mode="remesh")
    assert svc.store.get(rerun["id"])["candidate_group"] is None
    assert svc.store.get(rerun["id"])["candidate_index"] == 0
    assert "candidate_group" not in svc.store.get(rerun["id"])["params"]

    again = svc_jobs.promote_to_model(svc, source)
    assert svc.store.get(again["id"])["candidate_group"] is None


# --- keeping one -------------------------------------------------------------


def test_keep_dissolves_the_group_and_names_the_losers(svc):
    source = _reference(svc)
    result = svc_jobs.promote_candidates(svc, source, count=3)
    kept = result["ids"][1]

    outcome = svc_jobs.keep_candidate(svc, kept)
    assert outcome["id"] == kept
    assert outcome["losers"] == [result["ids"][0], result["ids"][2]]
    # Nothing is deleted: that is a separate, confirmed action.
    assert all(svc.store.get(i) is not None for i in result["ids"])
    # And nothing is left hidden -- every member is an ordinary job now, so a
    # declined delete leaves assets the library can still show.
    assert all(svc.store.get(i)["candidate_group"] is None for i in result["ids"])
    assert svc.store.candidate_jobs(result["group"]) == []


def test_keep_refuses_a_job_that_is_not_a_candidate(svc):
    source = _reference(svc)
    single = svc_jobs.promote_candidates(svc, source, count=1)
    with pytest.raises(Conflict):
        svc_jobs.keep_candidate(svc, single["id"])
    with pytest.raises(NotFound):
        svc_jobs.keep_candidate(svc, "0123456789ab")


# --- the picker's own logic --------------------------------------------------


def _row(job_id, group, index, status="done", created=1.0):
    return {
        "id": job_id,
        "candidate_group": group,
        "candidate_index": index,
        "status": status,
        "created_at": created,
        "files": ["model.glb"] if status == "done" else [],
    }


def test_the_picker_offers_the_newest_undecided_group_in_index_order():
    jobs = [
        _row("b1", "old", 1, created=1.0),
        _row("a0", "new", 0, created=9.0),
        _row("plain", None, 0, created=5.0),
        _row("b0", "old", 0, created=1.0),
        _row("a1", "new", 1, created=9.0),
    ]
    group = candidates_mod.pending(jobs)
    assert group is not None
    assert group.group == "new"
    assert [m["id"] for m in group.members] == ["a0", "a1"]


def test_the_picker_is_silent_when_nothing_is_undecided():
    assert candidates_mod.pending([_row("plain", None, 0)]) is None
    assert candidates_mod.pending([]) is None


def test_a_group_reports_when_it_is_still_running():
    running = [_row("a0", "g", 0, status="running"), _row("a1", "g", 1)]
    group = candidates_mod.pending(running)
    assert group.finished is False
    assert group.done_count == 1
    settled = [_row("a0", "g", 0), _row("a1", "g", 1, status="error")]
    assert candidates_mod.pending(settled).finished is True


def test_the_losers_of_a_group_are_everything_but_the_named_one():
    group = candidates_mod.pending([_row("a0", "g", 0), _row("a1", "g", 1)])
    assert group.losers("a0") == ["a1"]
    assert group.losers("a1") == ["a0"]


# --- the picker's actions ----------------------------------------------------


class _Confirms:
    def __init__(self) -> None:
        self.asked: list = []

    def ask(self, confirm):
        self.asked.append(confirm)


class _Ctx:
    """Enough of ``Ctx`` for the frame-thread half of the picker."""

    def __init__(self, svc) -> None:
        from warlock.studio.state import AppState

        self.svc = svc
        self.state = AppState()
        self.jobs: list = []
        self.toasts: list = []
        self.confirms = _Confirms()
        self.submitted: list = []
        self.invalidated = False
        outer = self

        class _Cache:
            @property
            def jobs(self):
                return outer.jobs

            def invalidate(self):
                outer.invalidated = True

            def get(self, job_id):
                return next((j for j in outer.jobs if j["id"] == job_id), None)

        self.cache = _Cache()

    def toast(self, text, level="info"):
        self.toasts.append((text, level))

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))
        return True


def test_keeping_from_the_picker_settles_the_group_and_only_then_asks(svc):
    from warlock.studio.panes import candidates_panel

    source = _reference(svc)
    result = svc_jobs.promote_candidates(svc, source, count=3)
    ctx = _Ctx(svc)
    ctx.jobs = [svc.store.get(i) for i in result["ids"]]
    group = candidates_mod.pending(ctx.jobs)
    kept = result["ids"][0]

    candidates_panel.keep(ctx, group, kept)

    # Settled immediately; deletion is only *offered*.
    assert svc.store.candidate_jobs(result["group"]) == []
    assert ctx.submitted == []
    assert len(ctx.confirms.asked) == 1
    assert all(svc.store.get(i) is not None for i in result["ids"])

    # And the confirm, when taken, goes through the library's delete path --
    # one submit per loser, under the delete key that pane already owns.
    ctx.confirms.asked[0].on_confirm()
    assert [key for key, *_ in ctx.submitted] == [
        f"delete:{result['ids'][1]}",
        f"delete:{result['ids'][2]}",
    ]


def test_selecting_a_candidate_moves_the_selection_and_nothing_else(svc):
    """The viewer is _sync_viewer's business -- including the pose guard and
    the ``pending`` drop rule -- so the picker must not reach for it."""
    from warlock.studio.panes import candidates_panel

    ctx = _Ctx(svc)
    candidates_panel.select(ctx, "abcdef012345")
    assert ctx.state.selected == "abcdef012345"
    assert ctx.submitted == []


# --- the 3D pane's control ---------------------------------------------------


def test_the_candidate_count_is_clamped_to_what_the_service_admits():
    from warlock.service.validation import MAX_MESH_CANDIDATES
    from warlock.studio.panes import settings_3d

    assert settings_3d.candidate_count({"candidates": 1}) == 1
    assert settings_3d.candidate_count({"candidates": 3}) == 3
    # A persisted settings file written under a higher ceiling, or edited by
    # hand, must not send a number the service refuses.
    assert settings_3d.candidate_count({"candidates": 99}) == MAX_MESH_CANDIDATES
    assert settings_3d.candidate_count({"candidates": 0}) == 1
    assert settings_3d.candidate_count({"candidates": "three"}) == 1
    assert settings_3d.candidate_count({}) == 1


def test_the_count_rides_the_matte_preview_into_the_promotion(svc):
    """Phase 1 put the cutout preview between Make 3D and the queue, and it
    captures the form as it stood at the press. The count is part of that
    capture, or Accept would submit a number the user has since changed."""
    from warlock.studio import matte_preview
    from warlock.studio.panes import settings_3d
    from warlock.studio.state import DEFAULT_FORM_3D

    source = _reference(svc)
    ctx = _Ctx(svc)
    ctx.state.form_3d = {**DEFAULT_FORM_3D, "candidates": 3}
    # Through get_job, because the pane's own validate() reads the attached
    # file list rather than the row.
    row = svc_jobs.get_job(svc, source)
    ctx.jobs = [row]

    settings_3d.promote(ctx, row, ctx.state.form_3d)
    state = ctx.state.matte
    assert state.job_id == source
    assert state.kwargs["count"] == 3

    # The form changing afterwards does not change what Accept submits.
    ctx.state.form_3d["candidates"] = 1
    matte_preview.accept(
        ctx, lambda kwargs, force: settings_3d.submit_promotion(ctx, source, kwargs, force)
    )
    key, fn, args, kwargs = ctx.submitted[0]
    assert key == "submit"
    assert fn is svc_jobs.promote_candidates
    assert kwargs["count"] == 3
