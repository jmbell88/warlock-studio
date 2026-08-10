"""Home's Resume list is filter-independent and never offers the trash (H11).

``cache.jobs`` is the raw loaded page -- the workshop and the trash together,
which is exactly what lets ``Filters.trash`` be a view -- so Home has to
exclude the trash on the row's own ``deleted_at`` (migration 9) rather than
lean on whatever the library's filter bar happens to be set to. A row that is
trashed simply stops being an answer, and the filter bar reshapes the library
and nothing else.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from warlock.studio import recents
from warlock.studio.panes import landing
from warlock.studio.state import AppState


class FakeSettings:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = dict(data or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


def _ctx(jobs: list[dict[str, Any]]) -> Any:
    return SimpleNamespace(
        state=AppState(),
        settings=FakeSettings({recents.SETTING: []}),
        cache=SimpleNamespace(
            jobs=jobs,
            by_id={j["id"]: j for j in jobs},
            get=lambda job_id: {j["id"]: j for j in jobs}.get(job_id),
            active=None,
            total=len(jobs),
            failures=lambda _filters: 0,
            storage=None,
        ),
        runtime=SimpleNamespace(checks=[]),
        progress=lambda _key: None,
    )


def test_a_trashed_row_never_appears_whatever_the_trash_view_says():
    jobs = [
        {"id": "kept", "status": "done", "stage": "model", "created_at": 2.0},
        {
            "id": "gone",
            "status": "done",
            "stage": "model",
            "created_at": 3.0,
            "deleted_at": 4.0,
        },
    ]
    ctx = _ctx(jobs)
    assert [r.key for r in landing.rows(ctx)] == ["kept"]
    # The trash view is session state the library owns; Home is not the library.
    ctx.state.filters.trash = True
    assert [r.key for r in landing.rows(ctx)] == ["kept"]


def test_the_library_filter_bar_does_not_reshape_resume():
    jobs = [
        {"id": "j1", "status": "done", "stage": "model", "name": "hut", "created_at": 2.0},
        {"id": "j2", "status": "done", "stage": "reference", "name": "oak", "created_at": 1.0},
    ]
    ctx = _ctx(jobs)
    before = landing.rows(ctx)
    assert [r.key for r in before] == ["j1", "j2"]
    ctx.state.filters.status = "error"
    ctx.state.filters.kind = "model"
    ctx.state.filters.text = "nothing-matches-this"
    ctx.state.filters.favorites_only = True
    assert landing.rows(ctx) == before
