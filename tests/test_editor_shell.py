"""Editor-first host shell: shared commands, status, and compact navigation."""

from __future__ import annotations

from types import SimpleNamespace


def _ctx():
    return SimpleNamespace(
        state=SimpleNamespace(
            mode="home",
            selected=None,
            wireframe=False,
            turntable=False,
            show_fps=False,
            filters=SimpleNamespace(trash=False),
            errors=[],
        ),
        cache=SimpleNamespace(get=lambda _key: None, jobs=[]),
        runtime=SimpleNamespace(checks=[]),
        viewer=None,
    )


def test_workspace_menu_lists_every_named_mode():
    from warlock.studio import menus, modes

    app_ctx = _ctx()
    rows = menus.specs(app_ctx)
    workspace = {
        row.identity.removeprefix("command:go:")
        for row in rows
        if row.identity.startswith("command:go:")
    }
    assert workspace == set(modes.KEYS)


def test_shared_command_specs_keep_palette_state_and_reason():
    from warlock.studio import menus, palette

    app_ctx = _ctx()
    commands = {command.key: command for command in palette.commands(app_ctx)}
    rows = {
        row.identity.removeprefix("command:"): row
        for row in menus.specs(app_ctx)
        if row.identity.startswith("command:")
    }
    for key in ("save", "undo", "wireframe", "manual", "quit"):
        assert rows[key].enabled == bool(commands[key].enabled(app_ctx))
        assert rows[key].shortcut == commands[key].hint
        assert rows[key].disabled_reason == commands[key].why


def test_status_reports_queue_and_health_without_permanent_ok_noise():
    from warlock.doctor import Check
    from warlock.studio import status_bar

    app_ctx = _ctx()
    app_ctx.cache.jobs = [{"status": "queued"}, {"status": "running"}]
    app_ctx.runtime.checks = [Check("CUDA", True, "ready", fatal=False)]
    keys = {item.key for item in status_bar.items(app_ctx)}
    assert "queue" in keys
    assert "health" not in keys
    app_ctx.runtime.checks = [Check("CUDA", False, "missing", fatal=False)]
    assert any(item.key == "health" and item.warning for item in status_bar.items(app_ctx))


def test_a_fresh_layout_prefers_the_44dp_icon_rail():
    from warlock.studio import layout, rail

    class Settings:
        def get(self, key, default=None):
            return default

        def set(self, key, value):
            pass

    assert layout.Layout(Settings()).rail == "icons"
    assert rail.RAIL_W == 44.0


def test_an_inker_row_that_is_a_document_state_reports_its_tick():
    """``MenuSpec.checked`` was hardcoded False for every Inker op, so the one
    row that is a *setting* rather than an action drew no tick and the user had
    no way to see which way it was set."""
    from warlock.studio import inker, inker_state, menus

    app_ctx = _ctx()
    app_ctx.state.mode = "inker"
    state = inker_state.InkerState()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(8, 8), uid="t1", title="Untitled")
    state.add(tab)
    app_ctx.state.inker = state

    def _row():
        return next(
            row for row in menus.specs(app_ctx) if row.identity == "inker:toggle_matte"
        )

    assert _row().checked is False
    assert tab.doc.toggle_matte() is True
    assert _row().checked is True


def test_the_resource_meter_is_not_one_of_the_elided_status_items():
    """It is right-anchored, and that is the whole point.

    ``items`` is drawn left to right with the tail dropped as the window
    narrows, so a meter in that list would be the *first* thing to go --
    backwards for the one figure a user consults while deciding whether to
    start a generation. ``overlay.doctor_banner``'s rule instead: reserve the
    trailing item, then trim the leading detail.
    """
    from warlock.studio import resources, status_bar

    app_ctx = _ctx()
    app_ctx.state.show_resources = True
    app_ctx.resources = resources.Sampler()
    app_ctx.resources.reading = resources.Reading(
        vram_used_gib=9.2, vram_total_gib=32.0, ram_used_gib=23.4, ram_total_gib=64.0, cpu=0.07
    )

    assert "resources" not in {item.key for item in status_bar.items(app_ctx)}
    item = status_bar.resource_item(app_ctx)
    assert item is not None and item.key == "resources"
    assert item.text == "VRAM 9.2/32   RAM 23.4/64   CPU 7%"
    assert not item.warning, "a machine being busy is not a fault"


def test_the_meter_is_an_opt_out_and_omits_what_it_cannot_read():
    """Off costs nothing, and a figure absent is a figure left out."""
    from warlock.studio import resources, status_bar

    app_ctx = _ctx()
    app_ctx.resources = resources.Sampler()
    app_ctx.resources.reading = resources.Reading(ram_used_gib=1.0, ram_total_gib=8.0)

    app_ctx.state.show_resources = False
    assert status_bar.resource_item(app_ctx) is None
    app_ctx.state.show_resources = True
    # No NVIDIA card and no interval yet: RAM alone, rather than a row of
    # dashes pretending to three readings.
    assert status_bar.resource_item(app_ctx).text == "RAM 1.0/8"
    # Nothing readable at all is no item, not an empty one.
    app_ctx.resources.reading = resources.Reading()
    assert status_bar.resource_item(app_ctx) is None


def test_the_sampler_holds_its_cadence_and_its_own_cpu_baseline():
    """One sampler per app: the CPU figure is a delta between calls."""
    from warlock.studio import resources

    sampler = resources.Sampler()
    first = sampler.tick(1000.0)
    # Inside the window: the previous reading, not a fresh syscall.
    assert sampler.tick(1000.0 + resources.TICK_SECONDS / 2) is first
    assert sampler.tick(1000.0 + resources.TICK_SECONDS * 2) is sampler.reading
    assert resources.TICK_SECONDS == 1.0


def test_resources_imports_nothing_from_the_ui():
    """``status_bar.items``' rule: the sampling and the formatting are data."""
    import inspect

    from warlock.studio import resources

    source = inspect.getsource(resources)
    for banned in ("imgui", "moderngl", "pygame"):
        assert banned not in source, banned
