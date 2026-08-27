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


def test_activating_an_asset_row_lands_at_the_stage_that_made_it():
    """The same reference/tile/model split the library filter uses -- and
    through ``create_stages.go``, never ``viewer.load_model``, which would
    bypass the pose guard and the off-thread parse."""
    ctx = _ctx(
        jobs=[
            {"id": "ref", "status": "done", "stage": "reference", "created_at": 2.0},
            {"id": "mesh", "status": "done", "stage": "model", "created_at": 1.0},
        ]
    )
    landing.activate(ctx, 0)
    assert (ctx.state.selected, ctx.state.create_stage) == ("ref", "reference")
    assert ctx.state.mode == "create"
    landing.activate(ctx, 1)
    assert (ctx.state.selected, ctx.state.create_stage) == ("mesh", "mesh")


def test_a_row_whose_file_is_gone_is_dropped_rather_than_failing_silently(tmp_path):
    settings = FakeSettings({recents.SETTING: []})
    missing = tmp_path / "gone.wblk"
    recents.remember(settings, "clay", str(missing), when=1.0)
    ctx = _ctx(settings=settings)

    landing.activate(ctx, 0)

    assert recents.paths(settings, "clay") == []
    assert landing.rows(ctx) == []
    # ``warn``, not "warning": the latter is not in ``state.TOAST_LEVELS`` and
    # fell back to a grey, non-sticky info toast. This test pinned the typo.
    assert ctx.toasts and ctx.toasts[0][1] == "warn"
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


def test_home_draws_the_actionable_rows_and_the_rail_still_gets_all_of_them():
    """The UI redesign, wave 4.3. ``status_rows`` stays the one source -- the rail's
    health badge reads the same data, and a fork would be two answers to "is
    this install healthy". What narrowed is what *Home* draws: the library row
    counted assets on a screen whose lower half is now assets."""
    assert "library" not in landing.HOME_STATUS
    assert set(landing.HOME_STATUS) <= {r.key for r in landing.status_rows(_ctx())} | {
        "review"
    }


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


def test_the_status_rows_are_computed_once_per_draw():
    """C3: ``_status_height`` and ``_status`` are two consumers of one answer,
    and each used to recompute it -- ``draw`` asks once and hands it down."""
    import inspect

    assert inspect.getsource(landing.draw).count("status_rows(") == 1
    for helper in (landing._news, landing._status):
        assert "status_rows(" not in inspect.getsource(helper), helper.__name__


# --- the UI redesign, wave 4.3: the What's New card and the New... menu -----------


def _release(version: str = "0.0.22", bullets: tuple[str, ...] = ("a", "b")):
    from warlock.changelog import Release

    return Release(version=version, date="2026-08-15", bullets=bullets)


def test_the_news_card_shows_once_per_release_and_stays_dismissed():
    assert landing.news_should_show(_release(), "")
    assert landing.news_should_show(_release(), "0.0.21")
    assert not landing.news_should_show(_release(), "0.0.22")


def test_a_release_with_nothing_in_it_is_not_a_card():
    """``changelog.current`` falls back to the newest entry when the running
    version has no section of its own, so "there is a release" is not the same
    question as "there is something to say about it"."""
    assert not landing.news_should_show(None, "")
    assert not landing.news_should_show(_release(bullets=()), "")


def test_the_new_menu_offers_every_creation_type_exactly_once():
    """The seven start_* functions are the whole of what this app can begin
    from nothing, and a menu that lost one would lose the only way into that
    mode that is not "switch there and find its empty state"."""
    keys = [key for key, _label, _icon, _action in landing.NEW_ITEMS]
    assert len(keys) == len(set(keys))
    actions = {action for _key, _label, _icon, action in landing.NEW_ITEMS}
    assert actions == {
        landing.start_2d,
        landing.start_3d,
        landing.start_inker,
        landing.start_clay,
        landing.start_plotter,
        landing.start_packwright,
        landing.start_troupe,
    }


def test_the_version_string_is_asked_for_once_per_process(monkeypatch):
    """C3: ``main._version`` is an importlib.metadata distribution walk, and
    the header and the news block both used to ask every frame. An installed
    version cannot change under a running process."""
    from warlock.studio import main

    calls: list[int] = []

    def counted() -> str:
        calls.append(1)
        return "9.9.9"

    monkeypatch.setattr(main, "_version", counted)
    monkeypatch.setattr(landing, "_VERSION", None)
    assert landing._version() == "9.9.9"
    assert landing._version() == "9.9.9"
    assert calls == [1]


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


# --- what a crashed session left ----------------------------------------------
#
# The pure half of the recovery section: which mode a row opens, and which
# glyph it wears. The drawing itself is smoke-tested with a real context; what
# matters here is that the two lookup tables cannot drift from the journal's
# provider list, which is exactly how a row ends up wearing the "unsupported"
# glyph for a kind this build supports perfectly well.


def test_every_journal_kind_has_a_row_destination():
    """A kind the journal can write and this table has never heard of draws
    with the fallback glyph and navigates nowhere -- which is indistinguishable
    from a build that cannot open it at all."""
    from warlock.studio import journal

    journal.ensure_providers()
    for kind in journal._PROVIDERS:
        assert kind in landing._KIND_MODES, f"{kind} has no row destination"


def test_every_row_destination_is_a_real_mode_or_deliberately_empty():
    """The empty string means "the provider navigates itself" and is the one
    value allowed not to name a mode. Anything else must be switchable to, or
    Recover would move the app somewhere that does not exist."""
    keys = {key for key, _label, _icon in modes.MODES}
    for kind, mode in landing._KIND_MODES.items():
        assert mode == "" or mode in keys, f"{kind} -> {mode!r}"


def test_the_one_self_navigating_kind_is_the_profile_draft():
    """Pinned by name rather than by count: a second empty entry added without
    a provider that navigates would silently make Recover do nothing visible."""
    empty = [kind for kind, mode in landing._KIND_MODES.items() if not mode]
    assert empty == ["profile"]


def test_the_recovery_section_draws_nothing_with_an_empty_snapshot():
    """The common case by far, and the one where a heading would be noise --
    a "Unsaved work" section saying "none" on every launch is a section that
    trains you to stop reading the top of the screen."""
    ctx = _ctx()
    ctx.state.recovery = []
    landing._recovery(ctx)  # no imgui frame needed: it returns before drawing


def test_the_health_row_opens_the_page_that_explains_its_number():
    """Both health rows carry the Settings category, not just the failing one.

    The row counts checks that are listed on exactly one page. Sending the
    reader to Settings and letting the remembered tab decide where they land is
    how a click on "2 things need attention" arrives at the theme picker.
    """
    failing = _ctx(checks=[_check("weights", False, fatal=False)])
    healthy = _ctx(checks=[_check("weights", True)])
    for ctx in (failing, healthy):
        row = {r.key: r for r in landing.status_rows(ctx)}["health"]
        assert row.target == "settings"
        assert row.settings_category == "health"


def test_a_row_with_no_category_leaves_the_remembered_tab_alone():
    """The default stays "": only health has a page it must land on, and a
    queue row that reset the Settings tab on its way to the Library would be
    changing something it has no opinion about."""
    row = landing.Status("queue", "!", "Queue idle", 0, "library")
    assert row.settings_category == ""
