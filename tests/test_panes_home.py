"""Home: the Resume list's ordering and activation, and the status block.

The screen this replaced was a grid of nine tiles, seven of which were the mode
switch again, and the test that stood here asserted the *coverage* of that grid
-- that every work mode had a tile. That property is now the mode switch's own
and ``tests/test_mode_keys.py`` has it; what is worth checking here is the two
things Home actually answers. The Resume list is one merged ordering across
five sources, which is the thing four per-mode lists could not produce, and the
status block is four sentences that have each been wrong at some point in this
app's history for want of anything asserting them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import icons, modes, recents
from warlock.studio.panes import landing
from warlock.studio.state import AppState


class FakeSettings:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = dict(data or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


def _cache(jobs: list[dict[str, Any]] | None = None, **extra: Any) -> Any:
    rows = jobs or []
    return SimpleNamespace(
        jobs=rows,
        by_id={j["id"]: j for j in rows},
        get=lambda job_id: {j["id"]: j for j in rows}.get(job_id),
        active=extra.get("active"),
        total=extra.get("total", len(rows)),
        failures=lambda _filters: extra.get("failed", 0),
        storage=extra.get("storage"),
    )


def _ctx(
    jobs: list[dict[str, Any]] | None = None,
    settings: FakeSettings | None = None,
    checks: list[Any] | None = None,
    **cache_extra: Any,
) -> Any:
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(
        state=AppState(),
        settings=settings or FakeSettings({recents.SETTING: []}),
        cache=_cache(jobs, **cache_extra),
        runtime=SimpleNamespace(checks=checks or []),
        progress=lambda _key: None,
        toast=lambda text, level="info": toasts.append((text, level)),
        toasts=toasts,
        svc=SimpleNamespace(),
    )
    return ctx


def _check(name: str, ok: bool, fatal: bool = False) -> Any:
    return SimpleNamespace(name=name, ok=ok, fatal=fatal)


# --- the resume list --------------------------------------------------------


def test_documents_and_assets_merge_into_one_list_newest_first():
    """The property four per-mode ``recent`` lists could not have: an ordering
    *between* the modes, and between them and the library."""
    settings = FakeSettings({recents.SETTING: []})
    recents.remember(settings, "clay", "a.wblk", when=300.0)
    recents.remember(settings, "inker", "b.ora", when=100.0)
    ctx = _ctx(
        jobs=[
            {"id": "j1", "status": "done", "stage": "model", "name": "hut", "created_at": 200.0},
            {"id": "j2", "status": "done", "stage": "model", "name": "old", "created_at": 50.0},
        ],
        settings=settings,
    )
    assert [(r.kind, r.name) for r in landing.rows(ctx)] == [
        ("clay", "a.wblk"),
        ("asset", "hut"),
        ("inker", "b.ora"),
        ("asset", "old"),
    ]


def test_an_unfinished_asset_is_not_offered():
    """Resuming a job that is still queued lands on a pane with nothing in it."""
    ctx = _ctx(
        jobs=[
            {"id": "j1", "status": "queued", "stage": "model", "created_at": 1.0},
            {"id": "j2", "status": "error", "stage": "model", "created_at": 2.0},
        ]
    )
    assert landing.rows(ctx) == []


def test_the_list_is_a_shortlist_rather_than_a_history():
    settings = FakeSettings({recents.SETTING: []})
    for index in range(recents.MAX_RECENT):
        recents.remember(settings, "clay", f"f{index}.wblk", when=float(index))
    jobs = [
        {"id": f"j{i}", "status": "done", "stage": "model", "created_at": 1000.0 + i}
        for i in range(10)
    ]
    assert len(landing.rows(_ctx(jobs=jobs, settings=settings))) == landing.MAX_RESUME


def test_a_rows_glyph_comes_from_the_mode_table():
    """A second copy is how a row comes to open Clay under Plotter's icon."""
    from_modes = {key: icon for key, _label, icon in modes.MODES}
    settings = FakeSettings({recents.SETTING: []})
    for kind in recents.KINDS:
        recents.remember(settings, kind, f"x.{kind}", when=1.0)
    for row in landing.rows(_ctx(settings=settings)):
        assert row.icon == from_modes[row.kind], row.kind


def test_every_row_draws_in_the_default_atlas_range():
    """The glyphs are a pinned lucide subset and every other string here goes
    through imgui's default Basic-Latin+Latin-1 range."""
    known = {value for name, value in vars(icons).items() if name.isupper()}
    settings = FakeSettings({recents.SETTING: []})
    recents.remember(settings, "clay", "a.wblk", when=1.0)
    ctx = _ctx(
        jobs=[{"id": "j1", "status": "done", "stage": "reference", "name": "n", "created_at": 2.0}],
        settings=settings,
    )
    for row in landing.rows(ctx):
        assert row.icon in known
        assert all(ord(c) < 0x100 for c in row.name)


def test_activating_an_asset_row_lands_in_the_pane_that_made_it():
    """The same reference/tile/model split the library filter uses -- and
    through ``state.select`` plus a mode, never ``viewer.load_model``, which
    would bypass the pose guard and the off-thread parse."""
    ctx = _ctx(
        jobs=[
            {"id": "ref", "status": "done", "stage": "reference", "created_at": 2.0},
            {"id": "mesh", "status": "done", "stage": "model", "created_at": 1.0},
        ]
    )
    landing.activate(ctx, 0)
    assert (ctx.state.selected, ctx.state.mode) == ("ref", "2d")
    landing.activate(ctx, 1)
    assert (ctx.state.selected, ctx.state.mode) == ("mesh", "3d")


def test_a_row_whose_file_is_gone_is_dropped_rather_than_failing_silently(tmp_path):
    settings = FakeSettings({recents.SETTING: []})
    missing = tmp_path / "gone.wblk"
    recents.remember(settings, "clay", str(missing), when=1.0)
    ctx = _ctx(settings=settings)

    landing.activate(ctx, 0)

    assert recents.paths(settings, "clay") == []
    assert landing.rows(ctx) == []
    assert ctx.toasts and ctx.toasts[0][1] == "warning"
    # And it did not switch mode on the way: there was nothing to open.
    assert ctx.state.mode == "home"


def test_the_cursor_wraps_over_the_rows_that_are_drawn():
    settings = FakeSettings({recents.SETTING: []})
    for index in range(3):
        recents.remember(settings, "clay", f"f{index}.wblk", when=float(index))
    ctx = _ctx(settings=settings)
    landing.move(ctx, -1)
    assert ctx.state.home_index == len(landing.rows(ctx)) - 1 == 2
    landing.move(ctx, 1)
    assert ctx.state.home_index == 0


def test_moving_over_an_empty_list_is_a_no_op_rather_than_a_zero_division():
    ctx = _ctx()
    landing.move(ctx, 1)
    assert ctx.state.home_index == 0


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (0.0, "just now"),
        (59.0, "just now"),
        (60.0, "1m ago"),
        (3600.0, "1h ago"),
        (90000.0, "yesterday"),
        (400000.0, "4d ago"),
    ],
)
def test_the_relative_stamp_is_coarse_on_purpose(delta, expected):
    """The question is "was that the one from this afternoon"; more precision
    than that invites being read as a measurement."""
    assert landing.ago(1000.0, now=1000.0 + delta) == expected


def test_an_unstamped_row_says_nothing_rather_than_1970():
    assert landing.ago(None) == ""


# --- the status block -------------------------------------------------------


def test_the_status_block_reports_health_queue_and_library():
    ctx = _ctx(
        jobs=[{"id": "j", "status": "done", "stage": "model", "created_at": 1.0}],
        checks=[_check("weights", False, fatal=False)],
        active={"id": "j2", "status": "running", "name": "brass lantern"},
        total=128,
        failed=3,
        storage={"bytes": 4_500_000_000},
    )
    found = {row.key: row for row in landing.status_rows(ctx)}
    assert "1 thing needs attention" in found["health"].text
    assert found["health"].target == "settings"
    assert "brass lantern" in found["queue"].text
    assert "128 assets" in found["library"].text and "3 failed" in found["library"].text
    assert found["library"].target == "library"


def test_a_fatal_check_is_the_error_colour_and_a_warning_is_not():
    from warlock.studio import theme

    fatal = landing.status_rows(_ctx(checks=[_check("trellis", False, fatal=True)]))[0]
    warn = landing.status_rows(_ctx(checks=[_check("weights", False, fatal=False)]))[0]
    assert fatal.colour == theme.ERR
    assert warn.colour == theme.WARN


def test_an_empty_check_list_is_still_checking_rather_than_everything_is_fine():
    """The distinction is the whole reason the row exists: for the first second
    or two after launch nothing has been probed, and "everything checks out" is
    a claim about an install nobody has looked at."""
    row = landing.status_rows(_ctx(checks=[]))[0]
    assert "still checking" in row.text
    assert row.target == ""


def test_the_queue_row_says_idle_rather_than_nothing():
    row = {r.key: r for r in landing.status_rows(_ctx())}["queue"]
    assert "idle" in row.text.lower()


def test_the_unreviewed_row_is_absent_until_a_count_has_come_back():
    """``None`` is "not asked yet" and 0 is "nothing waiting"; neither is a
    sentence worth a row, and drawing "0 unreviewed" from the first would be a
    claim the app has not measured."""
    ctx = _ctx()
    assert "review" not in {r.key for r in landing.status_rows(ctx)}
    ctx.state.home_unreviewed = 0
    assert "review" not in {r.key for r in landing.status_rows(ctx)}
    ctx.state.home_unreviewed = 1
    row = {r.key: r for r in landing.status_rows(ctx)}["review"]
    assert row.text == "1 mesh unreviewed" and row.target == "review"


def test_a_saturated_unreviewed_count_says_so():
    """A count that silently stops at its own limit is a wrong number rather
    than a rounded one."""
    ctx = _ctx()
    ctx.state.home_unreviewed = landing.UNREVIEWED_LIMIT
    row = {r.key: r for r in landing.status_rows(ctx)}["review"]
    assert row.text.startswith(f"{landing.UNREVIEWED_LIMIT}+")


def test_the_unreviewed_count_is_never_read_on_the_frame_thread():
    """A table scan behind the one serialized connection. ``pump`` submits it
    and the block draws the last answer, which is the bargain the storage walk
    already makes."""
    import inspect

    source = inspect.getsource(landing)
    assert "ctx.submit(\"home-unreviewed\"" in source
    assert "unverdicted_models" not in inspect.getsource(landing.status_rows)


def test_the_stamp_only_moves_when_the_submit_was_accepted():
    """``TaskRunner.submit`` refuses a key already in flight and nothing else
    re-arms it, so moving the stamp on a refusal strands the row on a figure
    that will never be recomputed."""
    ctx = _ctx()
    ctx.submit = lambda *_a, **_k: False
    landing.pump(ctx)
    assert ctx.state.home_unreviewed_at == 0.0
    ctx.submit = lambda *_a, **_k: True
    landing.pump(ctx)
    assert ctx.state.home_unreviewed_at > 0.0
    # And a second pump inside the TTL asks for nothing.
    stamp = ctx.state.home_unreviewed_at
    ctx.submit = lambda *_a, **_k: pytest.fail("asked again inside the TTL")
    landing.pump(ctx)
    assert ctx.state.home_unreviewed_at == stamp
