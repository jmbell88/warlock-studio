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
