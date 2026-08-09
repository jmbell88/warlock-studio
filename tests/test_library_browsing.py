"""Section J/O: the query syntax, the sorts, the trash, and the window.

The pure halves live in ``state.py`` and ``jobs_cache.py``, so most of this is
ordinary Python. The trash's rules are in the service and are tested against a
real store, because the interesting ones are about what the *worker* can still
do to a row the user has thrown away.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import pytest

from warlock.service import jobs as svc_jobs
from warlock.service.errors import Conflict
from warlock.studio import jobs_cache as cache_mod
from warlock.studio.panes import library
from warlock.studio.state import SORTS, Filters, parse_query


def job(**over: Any) -> dict[str, Any]:
    row = {
        "id": "j1",
        "kind": "text",
        "status": "done",
        "stage": "model",
        "name": "",
        "prompt": "",
        "tags": "",
        "params": {},
        "created_at": 1000.0,
        "started_at": None,
        "finished_at": None,
    }
    row.update(over)
    return row


# --- J87: the query syntax ---------------------------------------------------


def test_a_plain_query_is_still_plain_text():
    """The box has always been plain text; a query with no colon in it must
    behave character for character as it did."""
    assert parse_query("a wooden chest") == (["a", "wooden", "chest"], [])


def test_a_known_prefix_becomes_a_constraint():
    terms, fields = parse_query("tag:wood status:error rusty")
    assert terms == ["rusty"]
    assert fields == [("tag", "wood"), ("status", "error")]


def test_an_unknown_prefix_stays_free_text():
    """Silently reinterpreting a colon somebody typed on purpose is how a
    search starts returning nothing with no explanation."""
    assert parse_query("http://example") == (["http://example"], [])
    assert parse_query("foo:bar") == (["foo:bar"], [])


def test_a_prefix_with_no_value_is_free_text():
    assert parse_query("tag:") == (["tag:"], [])


def test_a_value_may_be_quoted_for_a_space():
    assert parse_query('name:"a wooden chest"') == ([], [("name", "a wooden chest")])


def test_an_unbalanced_quote_does_not_break_the_box():
    """The user is mid-typing; a traceback or an empty list would both be
    wrong answers to a half-written query."""
    assert parse_query('name:"a wooden') == ([], [("name", "a wooden")])


def test_every_word_must_match_not_any():
    filters = Filters(text="wooden chest")
    assert filters.matches(job(name="a wooden chest"))
    assert not filters.matches(job(name="a wooden barrel"))


def test_a_tag_matches_a_whole_entry_and_not_a_substring():
    """``tag:wood`` finding ``driftwood`` makes the prefix useless for the one
    thing it is for."""
    filters = Filters(text="tag:wood")
    assert filters.matches(job(tags="wood,props"))
    assert not filters.matches(job(tags="driftwood"))


@pytest.mark.parametrize(
    ("query", "row", "hit"),
    [
        ("status:error", {"status": "error"}, True),
        ("status:error", {"status": "done"}, False),
        ("kind:model", {"stage": "model"}, True),
        ("kind:model", {"stage": "reference"}, False),
        ("stage:reference", {"stage": "reference"}, True),
        ("id:j1", {}, True),
        ("id:zz", {}, False),
        ("name:chest", {"name": "A Wooden Chest"}, True),
    ],
)
def test_each_field_narrows_what_it_says_it_does(query, row, hit):
    assert Filters(text=query).matches(job(**row)) is hit


def test_a_field_term_adds_to_the_combos_rather_than_overriding_them():
    """A term that quietly overrode a visible control would be worse than a
    contradiction that correctly shows nothing."""
    filters = Filters(text="status:error", status="done")
    assert not filters.matches(job(status="error"))
    assert not filters.matches(job(status="done"))


# --- J85: the sorts ----------------------------------------------------------


def test_every_sort_key_is_offered_and_labelled():
    keys = [key for key, _label in SORTS]
    assert keys[0] == "newest"  # the persisted default
    assert len(set(keys)) == len(keys)
    assert all(label and all(ord(c) < 0x100 for c in label) for _key, label in SORTS)


def test_the_date_sort_is_the_querys_own_order():
    rows = [job(id="a"), job(id="b"), job(id="c")]
    assert Filters(sort="newest").order(rows) == rows
    assert Filters(sort="newest", descending=False).order(rows) == rows[::-1]


def test_the_name_sort_uses_the_name_the_card_shows():
    """Sorting by a column the card does not show would look like no sort at
    all -- the displayed name is the prompt when there is no title."""
    rows = [job(id="a", prompt="zebra"), job(id="b", name="apple")]
    # ``descending`` is each key's own natural order, which for a name is A to Z.
    assert [j["id"] for j in Filters(sort="name").order(rows)] == ["b", "a"]
    assert [j["id"] for j in Filters(sort="name", descending=False).order(rows)] == ["a", "b"]


def test_an_unanswerable_row_sorts_last_in_both_directions():
    """"Unknown" is not a value at one end of a scale, it is the absence of
    one -- filing the whole unmeasured backlog under "smallest" until the
    storage walk lands would be a wrong answer, not a partial one."""
    rows = [job(id="a"), job(id="b"), job(id="c")]
    sizes = {"a": 10, "b": 500}
    for descending in (True, False):
        ordered = Filters(sort="size", descending=descending).order(rows, sizes=sizes)
        assert ordered[-1]["id"] == "c"
    assert [j["id"] for j in Filters(sort="size").order(rows, sizes=sizes)][:2] == ["b", "a"]


def test_the_duration_sort_ignores_a_job_that_never_ran():
    rows = [
        job(id="slow", started_at=0.0, finished_at=100.0),
        job(id="fast", started_at=0.0, finished_at=5.0),
        job(id="queued"),
    ]
    assert [j["id"] for j in Filters(sort="duration").order(rows)] == [
        "slow",
        "fast",
        "queued",
    ]


def test_the_best_sort_keeps_its_old_bucketing():
    rows = [
        job(id="unranked"),
        job(id="good", params={"rank": {"score": 0.9}}),
        job(id="poor", params={"rank": {"score": 0.1}}),
    ]
    assert [j["id"] for j in Filters(sort="best").order(rows)] == ["good", "poor", "unranked"]


def test_every_sort_is_stable_within_a_tie():
    """Two refreshes half a second apart must not reshuffle equal rows."""
    rows = [job(id=f"j{n}", name="same") for n in range(6)]
    for key, _label in SORTS:
        ordered = Filters(sort=key).order(rows, sizes={})
        assert [j["id"] for j in ordered] == [j["id"] for j in rows], key


def test_an_unknown_sort_key_is_the_querys_order():
    """A persisted value from a build that offered a key this one does not."""
    rows = [job(id="a"), job(id="b")]
    assert Filters(sort="kremlin").order(rows) == rows


# --- J91: the trash ----------------------------------------------------------


def test_a_trashed_row_is_in_exactly_one_of_the_two_views():
    trashed = job(deleted_at=123.0)
    live = job()
    assert Filters().matches(live) and not Filters().matches(trashed)
    assert Filters(trash=True).matches(trashed) and not Filters(trash=True).matches(live)


def test_trashing_leaves_the_files_alone(svc):
    """That is the whole feature: a "trash" that had already deleted the mesh
    would be a different, worse thing wearing the name."""
    created = svc_jobs.create_job(svc, kind="text", prompt="a barrel")
    job_id = created["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "model.glb").write_bytes(b"x")
    svc.store.set_status(job_id, "done")

    svc_jobs.trash_job(svc, job_id)
    assert (svc.job_dir(job_id) / "model.glb").exists()
    assert svc.store.get(job_id)["deleted_at"] is not None

    svc_jobs.restore_job(svc, job_id)
    assert svc.store.get(job_id)["deleted_at"] is None


def test_trashing_a_queued_job_cancels_it(svc):
    """A trashed row is still a row. Without the cancel the worker would pick
    up a job the user has thrown away, spend two minutes of GPU on it and write
    a mesh into a directory nothing shows."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    assert svc.store.get(job_id)["status"] == "queued"
    svc_jobs.trash_job(svc, job_id)
    assert svc.store.get(job_id)["status"] == "cancelled"
    assert svc.store.next_queued() is None


def test_a_running_job_cannot_be_trashed(svc):
    """The refusal is about the *filesystem*, which does not care that this
    delete is reversible."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    svc.store.set_status(job_id, "running")
    with pytest.raises(Conflict):
        svc_jobs.trash_job(svc, job_id)


def test_emptying_the_trash_reads_all_of_it_not_a_page(svc):
    """"Empty" that left the older half behind while reporting success would
    be the worst possible reading of the word."""
    source = inspect.getsource(svc_jobs.empty_trash)
    assert "store.trashed()" in source
    assert "list_jobs" not in source

    ids = []
    for _ in range(3):
        job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
        svc.store.set_status(job_id, "done")
        svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        svc_jobs.trash_job(svc, job_id)
        ids.append(job_id)
    # Exact rather than a subset: the count is the claim, and ``kept`` is 0
    # because none of these carries a verdict. A trashed job that does is held
    # back -- see ``test_emptying_the_trash_keeps_a_labelled_image``.
    assert svc_jobs.empty_trash(svc) == {"deleted": 3, "kept": 0}
    for job_id in ids:
        assert svc.store.get(job_id) is None
        assert not svc.job_dir(job_id).exists()


def test_the_trash_query_is_ordered_by_when_it_was_thrown_away(svc):
    ids = []
    for _ in range(3):
        job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
        svc.store.set_status(job_id, "done")
        svc_jobs.trash_job(svc, job_id)
        ids.append(job_id)
        time.sleep(0.002)
    assert [row["id"] for row in svc.store.trashed()] == ids[::-1]


def test_the_card_delete_no_longer_asks_and_the_permanent_one_does():
    """The trash *is* the confirmation, and a better one: an undo the user can
    take an hour later beats a question answered in half a second."""
    source = inspect.getsource(library._overflow)
    body = source.split('menu_item("Delete"', 1)[1]
    assert "delete_asset(ctx, job_id)" in body
    assert "confirms.ask" not in body
    assert "confirms.ask" in inspect.getsource(library._trash_menu)


def test_delete_asset_trashes_and_purge_asset_deletes():
    """Every caller keeps its old name: what the user asked for is "get this
    out of my library", and the reversibility of that is a property of the app
    rather than of each call site."""
    assert "trash_job" in inspect.getsource(library.delete_asset)
    assert "delete_job" in inspect.getsource(library.purge_asset)


def test_quick_open_never_offers_a_trashed_asset():
    """It does not go through the filter bar at all, so without its own rule it
    would be the one surface where a deleted asset still turns up."""
    from types import SimpleNamespace

    from warlock.studio import palette

    rows = [job(id="live", name="chest"), job(id="gone", name="chest", deleted_at=1.0)]
    ctx = SimpleNamespace(cache=SimpleNamespace(jobs=rows))
    assert [j["id"] for j in palette.assets(ctx, "chest")] == ["live"]


# --- J88 / J89 / O119: the window and the rows -------------------------------


def test_filtering_reports_only_the_controls_that_hide_rows():
    """Sort and direction reorder; the trash re-addresses. Neither narrows, so
    neither may trigger the "this is only a window" warning."""
    assert not library.filtering(Filters())
    assert not library.filtering(Filters(sort="name", descending=False, trash=True))
    assert library.filtering(Filters(text="chest"))
    assert library.filtering(Filters(status="error"))
    assert library.filtering(Filters(kind="model"))
    assert library.filtering(Filters(favorites_only=True))


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [(0, "today"), (1, "yesterday"), (3, "this week"), (30, None)],
)
def test_date_grouping_uses_calendar_days(age_days, expected):
    """Something made at 23:50 last night is "yesterday" at 00:10, and an
    elapsed-hours rule would call it "today"."""
    import datetime as dt

    now = dt.datetime(2026, 8, 8, 12, 0).timestamp()
    then = now - age_days * 86400
    group = library.date_group(then, now=now)
    assert group == (expected or dt.date.fromtimestamp(then).strftime("%Y-%m"))


def test_an_undated_row_says_so_rather_than_landing_in_today():
    assert library.date_group(None) == "undated"


def test_the_window_can_be_reset(svc):
    cache = cache_mod.JobsCache(svc)
    assert cache.limit == cache_mod.LIST_LIMIT
    cache.load_more()
    cache.load_more()
    assert cache.limit == cache_mod.LIST_LIMIT * 3
    cache.reset_window()
    assert cache.limit == cache_mod.LIST_LIMIT


def test_resetting_a_window_that_never_grew_does_not_force_a_re_read(svc):
    cache = cache_mod.JobsCache(svc)
    cache._dirty = False
    cache.reset_window()
    assert cache._dirty is False


def test_the_size_sort_notices_a_measurement_landing(svc):
    """The storage walk is deferred and lands on a task thread long after the
    list did; without its own generation the memo would never reorder."""
    cache = cache_mod.JobsCache(svc)
    first = cache._filters_key(Filters(sort="size"))
    cache._dir_sizes = {"a": 1}
    cache._sizes_generation += 1
    assert cache._filters_key(Filters(sort="size")) != first


# --- O116: the prune count ---------------------------------------------------


def test_the_prune_count_is_asked_rather_than_assumed():
    source = inspect.getsource(library._ask_prune)
    assert "input_int" in source
    assert "body=body" in source
    # Reset every time it is asked: a destructive default somebody set once and
    # forgot is exactly the setting that deletes something they wanted.
    assert "PRUNE_KEEP_DEFAULT" in source
    assert "settings.set" not in source


def test_the_prune_message_does_not_promise_a_trash_it_does_not_use():
    """Prune exists to reclaim disk; one that moved two hundred jobs into the
    trash would free nothing while reporting that it had."""
    source = inspect.getsource(library._ask_prune)
    assert "deleted from disk" in source
    assert "cannot be undone" in source
