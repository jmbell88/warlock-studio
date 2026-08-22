"""Where a job row opens.

The bug these hold the line on: a finished sprite sheet toasted "finished" and
its ``Show`` opened a blank screen. Six job kinds -- ``rig``, ``sheet``,
``pixel_sheet``, ``sprite_synthesis``, ``charsheet``, ``retexture`` -- are
*products of another asset*: each is minted with ``params["source_job"]``,
writes its artifacts into that job's directory and never creates its own, so
``job["files"]`` is empty and ``db.Store.create``'s default leaves
``stage='model'``. Routing them by stage sent every one to Create's Mesh stage
holding a row with no mesh.

``route`` is pure, so the rule is a table rather than a window walk.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from warlock.studio import asset_open

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "warlock"


def _row(kind, stage="model", **params):
    return {"id": "self", "kind": kind, "stage": stage, "params": dict(params)}


@pytest.mark.parametrize(
    ("kind", "stage"),
    [
        ("rig", "rig"),
        ("sheet", "pose"),
        ("pixel_sheet", "pose"),
        ("sprite_synthesis", "reference"),
        ("retexture", "mesh"),
    ],
)
def test_a_follow_up_opens_the_asset_that_holds_its_artifacts(kind, stage):
    target = asset_open.route(_row(kind, source_job="SRC"))
    assert target.mode == "create"
    assert target.job_id == "SRC", "the follow-up row holds nothing of its own"
    assert target.stage == stage


def test_a_character_sheet_opens_in_troupe_with_its_sheet_selected():
    target = asset_open.route(_row("charsheet", source_job="MESH", sheet_id="S1"))
    assert target == asset_open.Route("troupe", "", "MESH", "S1", "")


def test_a_sprite_sheet_opens_the_reference_with_the_panel_open():
    target = asset_open.route(_row("sprite_synthesis", source_job="REF", draft_id="D1"))
    assert target.section == asset_open.SPRITES_SECTION
    assert target.detail == "D1"


def test_a_rendered_sheet_opens_the_mesh_with_the_sheet_panel_open():
    target = asset_open.route(_row("sheet", source_job="MESH", sheet_id="S9"))
    assert target.section == asset_open.SHEET_SECTION
    assert target.detail == "S9"


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("reference", "reference"),
        ("tile", "reference"),
        ("tilesheet", "reference"),
        ("model", "mesh"),
    ],
)
def test_an_ordinary_asset_still_routes_by_its_stage(stage, expected):
    """``stage_for``'s answer, unchanged, for a row that is an asset."""
    target = asset_open.route({"id": "A", "kind": "text", "stage": stage, "params": {}})
    assert target == asset_open.Route("create", expected, "A", "", "")


def test_a_follow_up_with_no_source_falls_back_rather_than_going_nowhere():
    """An old row or a hand-edited params blob. The old blank stage is still
    better than a click that does nothing."""
    target = asset_open.route(_row("sheet"))
    assert target.job_id == "self"
    assert target.mode == "create"


def test_every_section_key_is_registered_by_a_header():
    """A ``request_open`` for a key no header declares as its ``persist_key``
    matches nothing and merely accumulates for the life of the process -- a bug
    this codebase has already shipped once, with a comment beside it claiming
    it was why the panel was visible."""
    sources = "\n".join(
        p.read_text(encoding="utf-8") for p in (SRC / "studio").rglob("*.py")
    )
    for key in (asset_open.SPRITES_SECTION, asset_open.SHEET_SECTION):
        assert f"persist_key=asset_open.{_const_name(key)}" in sources, key


def _const_name(value: str) -> str:
    return next(
        name
        for name in ("SPRITES_SECTION", "SHEET_SECTION")
        if getattr(asset_open, name) == value
    )


def test_one_module_decides_where_an_asset_opens():
    """``stage_for`` answers "which stage shows this asset"; ``route`` answers
    "where does this row open", and they stopped being the same question the
    moment a row could be a product of another asset. Four call sites each held
    their own copy of the old ternary and all four were wrong the same way, so
    the copies are pinned to one."""
    offenders = sorted(
        str(path.relative_to(SRC))
        for path in (SRC / "studio").rglob("*.py")
        if "stage_for(" in path.read_text(encoding="utf-8")
        and path.name not in ("asset_open.py", "create_stages.py")
    )
    assert offenders == [], "route through asset_open.route instead"


def test_every_kind_that_writes_into_another_jobs_directory_is_routed():
    """The guard against the *next* follow-up kind silently re-acquiring this
    bug. A ``store.create`` whose params carry ``source_job`` is a row that
    holds nothing of its own, so it has to be in the routing table."""
    routed = set(asset_open.FOLLOWUP_STAGES) | {"charsheet"}
    found = set()
    for path in (SRC / "service").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "create"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            kind = node.args[0].value
            if isinstance(kind, str):
                found.add(kind)
    # Only the kinds that actually carry source_job; the rest are assets.
    expected = {"rig", "sheet", "charsheet", "retexture"} & found
    assert expected <= routed, f"unrouted follow-up kinds: {expected - routed}"
