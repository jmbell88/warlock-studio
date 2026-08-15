"""Poser mode's controller, with tasks run inline and no GL anywhere.

The FakeCtx inline-submit pattern: a submitted callable runs immediately, so
the test sees what the task thread would have done, and the on_task_done half
is driven by hand with the captured result -- which is exactly the seam the
dirty-clears-only-on-landing rule lives on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from warlock import doctor, rigging
from warlock.doctor import Check
from warlock.service import poses as svc_poses
from warlock.studio import poser_mode
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.gltf import Model, Node
from warlock.studio.viewer.pose import PoseEditor


class _Asks:
    def __init__(self) -> None:
        self.asked: list = []

    def ask(self, item) -> None:
        self.asked.append(item)


class FakeCtx:
    def __init__(self, svc=None, accept=True) -> None:
        self.svc = svc
        self.state = SimpleNamespace(poser=None)
        self.submitted: list[str] = []
        self.results: dict = {}
        self.accept = accept
        self.busy_keys: set[str] = set()
        self.confirms = _Asks()
        self.prompts = _Asks()
        self.toasts: list = []
        self.rig_default = "humanoid"
        self.rigging_available = True
        self.viewer = None  # the shared viewer, for pose_panel.guard
        self.poser_viewer = None

    def submit(self, key, fn, *args, **kwargs) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.results[key] = fn(*args, **kwargs)
        return True

    def busy(self, key) -> bool:
        return key in self.busy_keys

    def toast(self, message, level="info", *args) -> None:
        self.toasts.append((message, level))


class FakeViewer:
    """Just the surface poser_mode touches, over a real PoseEditor."""

    def __init__(self, model=None, bones=None) -> None:
        self.editor = PoseEditor()
        self.pose_mode = False
        self.path = None
        self.loaded: list = []
        self.cleared = 0
        self.framed = None
        if model is not None:
            self.editor.bind(model, bones)
            self.pose_mode = True

    def load_model(self, path) -> None:
        self.loaded.append(Path(path))
        self.path = Path(path)

    def enter_pose_authoring(self, bones, mirror_pairs, token) -> bool:
        self.editor.mirror_pairs = [list(p) for p in mirror_pairs]
        self.pose_mode = True
        self.token = token
        return True

    def frame_bounds(self, lo, hi) -> float:
        self.framed = (lo, hi)
        return 1.0

    def clear(self) -> None:
        self.cleared += 1
        self.path = None
        self.editor.clear()
        self.pose_mode = False

    def set_pose(self, bones, *, pose_id=None, dirty=True) -> None:
        self.editor.apply(bones, pose_id=pose_id, dirty=dirty)

    def get_pose(self):
        return self.editor.pose()

    def set_root_translation(self, v, *, dirty=True) -> None:
        self.editor.set_root_translation(v, dirty=dirty)

    def apply_preset(self, preset) -> None:
        self.editor.apply_preset(preset)

    def reset_all(self, *, dirty=True) -> None:
        self.editor.reset_all(dirty=dirty)

    def mirror(self) -> None:
        self.editor.mirror()


def _full_bones(template="humanoid"):
    """Every bone at identity -- validate_record requires the whole skeleton,
    which is what get_pose(), the library's only real writer, produces."""
    return {b["name"]: [0.0, 0.0, 0.0, 1.0] for b in rigging.get_template(template).bones}


def _armature_model() -> Model:
    """One node per humanoid bone under a 'rig' object node -- flat, because
    the editor needs only names and positions here, and a save built off this
    must carry the template's *whole* skeleton (validate_record's bar)."""
    names = [b["name"] for b in rigging.get_template("humanoid").bones]
    nodes = [Node(name="rig", children=list(range(1, len(names) + 1)))]
    for i, name in enumerate(names):
        nodes.append(Node(name=name, translation=m3.vec3(0.0, 0.1 * i, 0.0)))
    return Model(nodes, roots=[0], meshes=[], skins=[])


def _bound_viewer() -> FakeViewer:
    viewer = FakeViewer(
        _armature_model(), [b["name"] for b in rigging.get_template("humanoid").bones]
    )
    viewer.editor.root = "hips"
    return viewer


def _fake_blender(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor, "blender_check", lambda **kw: Check("Blender (rigging)", True, "bpy 5.2.0", False)
    )

    def run_worker(spec, **kwargs):
        Path(spec["out_glb"]).write_bytes(b"armature-glb")
        return {"ok": True}

    monkeypatch.setattr(rigging, "run_worker", run_worker)


# --- entering ----------------------------------------------------------------


def test_enter_refreshes_the_library_and_requests_the_preview(svc, monkeypatch):
    _fake_blender(monkeypatch)
    stored = svc_poses.create_library_pose(
        svc,
        {"name": "Crouch", "template": "humanoid", "bones": _full_bones()},
    )
    ctx = FakeCtx(svc)
    poser_mode.enter(ctx)
    state = poser_mode.ensure(ctx)
    assert state.template == "humanoid"
    assert poser_mode.LIST_KEY in ctx.submitted
    assert f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid" in ctx.submitted

    # The results land through on_task_done, exactly as the app delivers them.
    poser_mode.on_task_done(
        ctx, SimpleNamespace(key=poser_mode.LIST_KEY, result=ctx.results[poser_mode.LIST_KEY])
    )
    assert [p["id"] for p in state.poses] == [stored["id"]]
    assert any(p["name"] == "idle" for p in state.presets), "shipped presets ride along"
    assert state.loading is False

    key = f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid"
    poser_mode.on_task_done(ctx, SimpleNamespace(key=key, result=ctx.results[key]))
    assert state.building is False
    assert state.preview_template == "humanoid"
    assert Path(state.preview_path).exists()


def test_a_refused_submit_clears_loading(svc):
    ctx = FakeCtx(svc, accept=False)
    poser_mode.refresh(ctx)
    assert poser_mode.ensure(ctx).loading is False


def test_a_failed_task_clears_its_flags(svc):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.loading = True
    poser_mode.on_task_failed(ctx, SimpleNamespace(key=poser_mode.LIST_KEY, message=""))
    assert state.loading is False
    state.building = True
    poser_mode.on_task_failed(
        ctx,
        SimpleNamespace(key=f"{poser_mode.PREVIEW_KEY_PREFIX}humanoid", message="no bpy"),
    )
    assert state.building is False
    assert state.error == "no bpy"


# --- the template switch -----------------------------------------------------


def test_a_dirty_editor_guards_the_template_switch(svc):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    ctx.poser_viewer = _bound_viewer()
    ctx.poser_viewer.editor.dirty = True

    poser_mode.set_template(ctx, "fish")
    assert state.template == "humanoid", "nothing moved before the answer"
    assert len(ctx.confirms.asked) == 1

    ctx.confirms.asked[0].on_confirm()
    assert state.template == "fish"
    assert ctx.poser_viewer.cleared == 1, "the old armature must not stay poseable"
    assert poser_mode.LIST_KEY in ctx.submitted


def test_a_clean_editor_switches_immediately(svc):
    ctx = FakeCtx(svc)
    ctx.poser_viewer = _bound_viewer()
    poser_mode.set_template(ctx, "fish")
    assert poser_mode.ensure(ctx).template == "fish"
    assert ctx.confirms.asked == []


# --- applying ----------------------------------------------------------------


def test_apply_pose_loads_the_record_clean(svc):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    viewer = ctx.poser_viewer = _bound_viewer()
    record = {
        "id": "0123456789ab",
        "name": "Leap",
        "bones": {"spine": [0.0, 0.0, 0.7071068, 0.7071068]},
        "root_translation": [0.1, 0.0, 0.2],
    }
    state.poses = [record]
    poser_mode.apply_pose(ctx, "0123456789ab")
    assert viewer.editor.current == "0123456789ab"
    assert viewer.editor.dirty is False, "an applied pose is not an unsaved edit"
    assert viewer.editor.root_translation() == pytest.approx([0.1, 0.0, 0.2])


def test_applying_a_pose_after_another_leaves_no_residue(svc):
    """apply_pose resets first, the apply_preset order: set_pose writes only
    the bones the record lists, so a partial record (pre-completeness saves
    exist on disk) applied over a posed editor would keep stale rotations --
    which get_pose() would then save as authored."""
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    viewer = ctx.poser_viewer = _bound_viewer()
    state.poses = [
        {"id": "0123456789ab", "name": "Twist",
         "bones": {"spine": [0.0, 0.0, 0.7071068, 0.7071068]}},
        {"id": "0123456789ac", "name": "Idle", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
    ]
    poser_mode.apply_pose(ctx, "0123456789ab")
    poser_mode.apply_pose(ctx, "0123456789ac")
    assert viewer.editor.pose()["spine"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert viewer.editor.current == "0123456789ac"


def test_apply_preset_resets_the_root(svc):
    """A preset lists rotations only; the reset-first rule is what makes it
    zero-translation rather than inheriting the last pose's offset."""
    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.set_root_translation([0.0, 0.0, 0.5], dirty=False)
    poser_mode.apply_preset(ctx, {"name": "idle", "bones": {"spine": [0, 0, 0, 1]}})
    assert viewer.editor.root_translation() == [0.0, 0.0, 0.0]


# --- saving ------------------------------------------------------------------


def test_dirty_clears_only_when_the_save_lands(svc):
    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.dirty = True

    poser_mode.save(ctx)  # nothing being edited -> falls through to Save as
    assert len(ctx.prompts.asked) == 1
    ctx.prompts.asked[0].on_accept("Crouch")
    assert poser_mode.SAVE_KEY in ctx.submitted
    assert viewer.editor.dirty is True, "submit is not landing"

    result = ctx.results[poser_mode.SAVE_KEY]
    poser_mode.on_task_done(ctx, SimpleNamespace(key=poser_mode.SAVE_KEY, result=result))
    assert viewer.editor.dirty is False
    assert viewer.editor.current == result["id"]
    # And the library re-reads itself so the new row appears.
    assert ctx.submitted.count(poser_mode.LIST_KEY) == 1


def test_save_over_the_edited_pose_updates_in_place(svc):
    stored = svc_poses.create_library_pose(
        svc,
        {"name": "Crouch", "template": "humanoid", "bones": _full_bones()},
    )
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    state.poses = [stored]
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.current = stored["id"]
    viewer.editor.dirty = True

    poser_mode.save(ctx)
    assert ctx.prompts.asked == [], "an in-place save asks no name"
    result = ctx.results[poser_mode.SAVE_KEY]
    assert result["id"] == stored["id"]
    on_disk = {p["id"]: p for p in svc_poses.list_library(svc)["poses"]}
    assert on_disk[stored["id"]]["updated"] == result["updated"]


def test_deleting_the_edited_pose_clears_current(svc):
    stored = svc_poses.create_library_pose(
        svc,
        {"name": "Crouch", "template": "humanoid", "bones": _full_bones()},
    )
    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.current = stored["id"]
    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(key=f"{poser_mode.DELETE_KEY}:{stored['id']}", result={"ok": True}),
    )
    assert viewer.editor.current is None, "Save must not write to a record that is gone"


# --- the guard ---------------------------------------------------------------


def test_guard_asks_only_for_the_poser_session(svc):
    from warlock.studio.panes import pose_panel

    ctx = FakeCtx(svc)
    ctx.poser_viewer = _bound_viewer()
    ctx.poser_viewer.editor.dirty = True

    ran: list[str] = []
    # The inspector's guard reads the *shared* viewer, which holds nothing --
    # so with a dirty Poser session there is still exactly one question.
    assert pose_panel.guard(ctx, "quit", lambda: ran.append("pose")) is True
    assert poser_mode.guard(ctx, "quit", lambda: ran.append("poser")) is False
    assert ran == ["pose"]
    assert len(ctx.confirms.asked) == 1


def test_guard_proceeds_with_no_session(svc):
    ctx = FakeCtx(svc)
    ran: list[str] = []
    assert poser_mode.guard(ctx, "quit", lambda: ran.append("go")) is True
    assert ran == ["go"]


# --- Mirror, behind the same guard -------------------------------------------
#
# Mirror rewrites every rotation and this mode has no undo at all: it was the
# last control that could discard an unsaved pose with no way back.


def _mirrored_viewer():
    viewer = _bound_viewer()
    viewer.editor.mirror_pairs = [["upper_arm.L", "upper_arm.R"]]
    return viewer


def _left_arm(viewer):
    return list(viewer.editor.pose()["upper_arm.L"])


def test_mirror_over_unsaved_edits_asks_first_and_moves_nothing(svc):
    ctx = FakeCtx(svc)
    viewer = _mirrored_viewer()
    ctx.poser_viewer = viewer
    viewer.editor.apply({"upper_arm.R": [0.0, 0.0, 0.7071068, 0.7071068]})
    before = _left_arm(viewer)

    assert poser_mode.guard(ctx, "mirror the pose", viewer.mirror) is False
    assert len(ctx.confirms.asked) == 1
    assert "mirror the pose" in ctx.confirms.asked[0].message
    assert _left_arm(viewer) == before, "nothing moved before the answer"

    ctx.confirms.asked[0].on_confirm()
    assert _left_arm(viewer) != before


def test_mirror_on_a_clean_editor_just_runs(svc):
    ctx = FakeCtx(svc)
    viewer = _mirrored_viewer()
    ctx.poser_viewer = viewer
    viewer.editor.apply({"upper_arm.R": [0.0, 0.0, 0.7071068, 0.7071068]}, dirty=False)
    before = _left_arm(viewer)

    assert poser_mode.guard(ctx, "mirror the pose", viewer.mirror) is True
    assert ctx.confirms.asked == []
    assert _left_arm(viewer) != before


@pytest.mark.parametrize(
    ("module", "guard_name"),
    [("poser_controls", "poser_mode.guard"), ("pose_panel", "guard")],
)
def test_both_mirror_buttons_go_through_their_pane_s_guard(module, guard_name):
    """Source-scanned rather than clicked: the button lives inside an imgui
    frame, and what is being pinned is that no bare ``viewer.mirror()`` call
    comes back -- which is a fact about the text."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(f"warlock.studio.panes.{module}"))
    assert f'{guard_name}(ctx, "mirror the pose", viewer.mirror)' in source
    assert "viewer.mirror()" not in source


# --- task-key routing --------------------------------------------------------


def test_every_poser_task_key_is_prefixed_poser():
    """The prefix is the routing, so it is not a naming convention: a Poser key
    that did not start with ``poser-`` would be handed to another mode's
    handler by ``main._on_task_done``."""
    keys = [
        poser_mode.LIST_KEY,
        poser_mode.SAVE_KEY,
        poser_mode.DELETE_KEY,
        poser_mode.DUPLICATE_KEY,
        poser_mode.RENAME_KEY,
        poser_mode.PREVIEW_KEY_PREFIX,
    ]
    assert all(k.startswith("poser-") for k in keys), keys


def test_poser_results_are_claimed_before_the_asset_pose_branches():
    """``"poser-save"`` also starts with ``"pose-"``. Nothing but the order of
    two ``if`` statements keeps a Poser result out of the inspector's pose
    handler, and the fix if they were ever swapped is not obvious from either
    site -- so the order is the pin.
    """
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._on_task_done)
    poser = source.index('key.startswith("poser-")')
    assert poser < source.index('key.startswith("pose-library:")')
    assert poser < source.index('key.startswith("pose-")')


# --- the preview binding -----------------------------------------------------


def test_sync_preview_binds_once_and_not_again(svc, tmp_path):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    glb = tmp_path / "humanoid.glb"
    glb.write_bytes(b"glb")
    state.preview_path = glb
    state.preview_template = "humanoid"

    viewer = FakeViewer()
    viewer.editor.model = None
    assert poser_mode.sync_preview(ctx, viewer) is True
    assert viewer.loaded == [glb]
    assert viewer.token == "poser:humanoid"
    assert viewer.editor.root == rigging.get_template("humanoid").root
    assert viewer.framed is not None

    assert poser_mode.sync_preview(ctx, viewer) is True
    assert viewer.loaded == [glb], "already showing it: no reload"


def test_a_preview_for_a_switched_away_template_never_binds(svc, tmp_path):
    ctx = FakeCtx(svc)
    state = poser_mode.ensure(ctx)
    glb = tmp_path / "fish.glb"
    glb.write_bytes(b"glb")
    state.preview_path = glb
    state.preview_template = "fish"  # but state.template is humanoid

    viewer = FakeViewer()
    assert poser_mode.sync_preview(ctx, viewer) is False
    assert viewer.loaded == []


def test_preview_bounds_cover_the_whole_skeleton():
    lo, hi = poser_mode.preview_bounds("humanoid")
    assert all(b > a for a, b in zip(lo, hi, strict=True))
    # The armature is one character-height tall; the glTF vertical axis is Y.
    assert hi[1] - lo[1] >= 1.0
    # And tails are included: the head bone's tail reaches z=1.0 exactly, so a
    # joint-only box would stop short of it.
    assert hi[1] > 1.0 - 1e-6


# --- keys --------------------------------------------------------------------


def test_escape_deselects_the_joint_and_nothing_else(svc):
    import pygame

    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.selected = "spine"
    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    assert poser_mode.handle_key(ctx, event) is True
    assert viewer.editor.selected is None
    assert poser_mode.handle_key(ctx, event) is False
    other = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_f, mod=0)
    assert poser_mode.handle_key(ctx, other) is False


def test_the_mode_is_wired_into_the_switch():
    from warlock.studio import modes

    assert "poser" in modes.KEYS
    assert "poser" in modes.WORK_MODES
    assert "poser" in modes.WORKSPACE_MODES
    # And in the rail's workspace group, which is the navigation the app
    # actually draws (the UI redesign, wave 3). This used to assert that the derived
    # ``GROUP_BREAKS`` still collapsed to two; the grouping is written out now,
    # so what a new mode has to do is *be in a group*.
    assert "poser" in modes.RAIL_GROUPS[1]
