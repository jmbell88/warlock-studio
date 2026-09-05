"""Poser mode's controller, with tasks run inline and no GL anywhere.

The FakeCtx inline-submit pattern: a submitted callable runs immediately, so
the test sees what the task thread would have done, and the on_task_done half
is driven by hand with the captured result -- which is exactly the seam the
dirty-clears-only-on-landing rule lives on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
        # The onion-skin ghosts the clip editor pushes; empty everywhere else.
        self.onion: list = []
        # The real camera, because the view keys are about what it does and a
        # stub would pass whatever they did.
        from warlock.studio.viewer.camera import Camera

        self.camera = Camera()
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


def test_escape_deselects_the_joint(svc):
    import pygame

    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()
    viewer.editor.selected = "spine"
    event = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)
    assert poser_mode.handle_key(ctx, event) is True
    assert viewer.editor.selected is None
    assert poser_mode.handle_key(ctx, event) is False
    # An unbound letter still falls through.
    other = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_g, mod=0)
    assert poser_mode.handle_key(ctx, other) is False


def test_poser_has_clays_view_keys_under_the_same_ctrl(svc):
    """It had none: the only way to look at a pose from the side was to orbit
    there by hand, and no way back once the model was off screen. And once it
    had them it fired them on the *bare* digit, where Clay wants Ctrl -- the
    2026-09-05 consistency pass: one chord, both viewports."""
    import pygame

    from warlock.studio import clay_mode

    ctx = FakeCtx(svc)
    viewer = ctx.poser_viewer = _bound_viewer()

    def press(key, mod=0):
        return poser_mode.handle_key(
            ctx, SimpleNamespace(type=pygame.KEYDOWN, key=key, mod=mod)
        )

    at_rest = (viewer.camera.theta, viewer.camera.phi, viewer.camera.orthographic)
    assert press(pygame.K_1) is False
    assert press(pygame.K_5) is False
    assert (viewer.camera.theta, viewer.camera.phi, viewer.camera.orthographic) == at_rest

    assert press(pygame.K_1, pygame.KMOD_CTRL) is True
    front = (viewer.camera.theta, viewer.camera.phi)
    assert press(pygame.K_1, pygame.KMOD_CTRL | pygame.KMOD_SHIFT) is True
    assert (viewer.camera.theta, viewer.camera.phi) != front

    ortho = viewer.camera.orthographic
    assert press(pygame.K_5, pygame.KMOD_CTRL) is True
    assert viewer.camera.orthographic is not ortho

    assert press(pygame.K_f) is True

    # The table is Clay's, read rather than restated, so the two viewports
    # cannot come to disagree about which number is the front.
    assert set(clay_mode.AXIS_VIEW_KEYS) == {"1", "3", "7"}


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


# --- crash recovery ------------------------------------------------------------


def test_a_recovered_pose_restores_joint_corrections(tmp_path):
    """``_pose_payload`` writes ``moved`` and ``mode``; an adopt that applied
    only bones and root translation recovered a crash mid-joint-placement as a
    rest pose under a success toast, the corrections silently dropped."""
    viewer = _bound_viewer()
    editor = viewer.editor
    editor.enter_joints_mode()
    editor.move_handle("hips", editor.home["hips"] + m3.vec3(0.1, 0.0, 0.2))
    assert editor.moved, "the session holds a correction"
    moved = {name: list(delta) for name, delta in editor.moved.items()}

    slot = poser_mode._PoseSlot(viewer, editor, "poser")
    path = tmp_path / "poser.pose.json"
    path.write_bytes(poser_mode._pose_payload(slot))

    ctx = FakeCtx()
    ctx.poser_viewer = _bound_viewer()
    assert poser_mode._journal_adopt(ctx, path, {}) is True
    recovered = ctx.poser_viewer.editor
    assert recovered.mode == "joints"
    assert recovered.moved == moved
    assert recovered.has_unsaved_edits()


# --- the clip editor ----------------------------------------------------------
#
# The Troupe programme's clip-authoring half. The armature *is* the editor
# here -- a key
# is authored by posing the preview and putting it back -- so what these pin is
# the part that is not the pose editor: the timing model, and the places a clip
# can be edited into something the renderer would refuse.


def _clip_library() -> dict:
    return {
        "template": "humanoid",
        "space": "delta",
        "edited": False,
        "poses": [
            {"name": "A", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "B", "bones": {"spine": [0.0, 0.0, 0.0, 1.0]}},
            {"name": "C", "bones": {"head": [0.0, 0.0, 0.0, 1.0]}},
        ],
        "clips": [
            {
                "name": "walk",
                "keys": ["A", "B", "C"],
                "segments": [2, 2, 2],
                "closed": True,
                "easing": "linear",
                "space": "delta",
            }
        ],
    }


def _clip_ctx():
    ctx = FakeCtx()
    ctx.poser_viewer = _bound_viewer()
    state = poser_mode.ensure(ctx)
    state.clips_dirty_flag = False
    poser_mode.adopt_clips(ctx, _clip_library())
    return ctx, state


def _turned():
    """A rotation that is not the identity, spelled once."""
    return [0.0, 0.3894183, 0.0, 0.9210610]


def test_adopting_a_library_opens_its_first_clip_and_expands_it():
    ctx, state = _clip_ctx()
    assert state.clip == "walk"
    assert state.clips_unsaved is False
    # Three closed segments of two frames each.
    assert len(state.frames) == 6


def test_selecting_a_key_puts_its_pose_on_the_armature():
    """And it goes through ``apply_preset``, so the bones the key does *not*
    name go back to rest -- otherwise loading a walk key after an attack key
    leaves the sword arm up."""
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    editor.apply({"head": _turned()}, dirty=True)
    poser_mode.select_key(ctx, 0)
    # The armature holds an unsaved edit, so the door asks first (the
    # 2026-09-02 review's finding 6); the click lands once it is answered.
    assert len(ctx.confirms.asked) == 1
    ctx.confirms.asked[0].on_confirm()
    assert state.key_index == 0
    assert state.frame == -1
    assert editor.pose()["head"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_capturing_writes_the_armature_back_into_the_selected_key():
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 1)
    editor.apply({"spine": _turned()}, dirty=True)
    poser_mode.capture_key(ctx)

    assert state.clips_unsaved is True
    assert state.key_pose("B")["bones"]["spine"] == pytest.approx(_turned())


def test_capturing_an_in_between_frame_is_refused_by_name():
    """The armature shows an interpolated pose while scrubbing, and storing
    that into a key would silently replace an authored pose with a computed
    one. Refused rather than snapped to the nearest key, which is the same
    mistake with the evidence removed."""
    ctx, state = _clip_ctx()
    poser_mode.scrub(ctx, 3)
    before = dict(state.key_pose("A")["bones"])
    poser_mode.capture_key(ctx)
    assert state.key_pose("A")["bones"] == before
    assert state.clips_unsaved is False
    assert any("in-between" in msg for msg, _kind in ctx.toasts)


def test_scrubbing_pushes_no_undo_steps():
    """It runs once per frame of a slider drag. A step per frame is a stack
    full of one gesture -- the rule ``rotate_selected`` already follows."""
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 0)
    head = editor.history.head
    for frame in range(len(state.frames)):
        poser_mode.scrub(ctx, frame)
    assert editor.history.head == head


def test_closing_or_opening_a_clip_resizes_its_timing():
    """An open clip of N keys has N-1 steps and a closed one has N. Resized
    here rather than left for the save to refuse, because the count is
    *derived*: there is no other value it could take."""
    ctx, state = _clip_ctx()
    poser_mode.set_closed(ctx, False)
    assert state.open_clip()["segments"] == [2, 2]
    poser_mode.set_closed(ctx, True)
    assert state.open_clip()["segments"] == [2, 2, 2]


def test_moving_a_key_carries_its_own_segment():
    """A segment is the step *out of* a key, so reordering without it would
    move the poses and leave the timing behind -- which reads as the reorder
    having corrupted the clip."""
    ctx, state = _clip_ctx()
    state.open_clip()["segments"] = [1, 5, 9]
    poser_mode.move_key(ctx, 0, 1)
    assert state.open_clip()["keys"] == ["B", "A", "C"]
    assert state.open_clip()["segments"] == [5, 1, 9]
    assert state.key_index == 1


def test_removing_a_key_leaves_the_pose_in_the_library():
    """Two different things: another clip may use that pose, and a delete that
    silently reached into the shared list would have a blast radius the button
    does not describe."""
    ctx, state = _clip_ctx()
    poser_mode.remove_key(ctx, 1)
    assert state.open_clip()["keys"] == ["A", "C"]
    assert state.key_pose("B") is not None


def test_a_clip_cannot_be_cut_below_two_keys():
    ctx, state = _clip_ctx()
    poser_mode.remove_key(ctx, 0)
    poser_mode.remove_key(ctx, 0)
    assert len(state.open_clip()["keys"]) == 2
    assert any("at least" in msg for msg, _kind in ctx.toasts)


def test_a_new_key_is_authored_from_the_armature_and_inserted():
    ctx, state = _clip_ctx()
    editor = ctx.poser_viewer.editor
    poser_mode.select_key(ctx, 0)
    editor.apply({"head": _turned()}, dirty=True)
    poser_mode.new_key(ctx, "A crouch")

    assert state.key_pose("A crouch") is not None
    assert state.open_clip()["keys"] == ["A", "A crouch", "B", "C"]
    assert len(state.open_clip()["segments"]) == 4, "a closed clip gains a step"


def test_a_duplicate_key_name_is_refused():
    """Names are what a clip references by, so two of them is an ambiguity
    rather than a duplicate."""
    ctx, state = _clip_ctx()
    poser_mode.new_key(ctx, "B")
    assert any("already exists" in msg for msg, _kind in ctx.toasts)
    assert state.open_clip()["keys"] == ["A", "B", "C"]


def test_a_clip_that_will_not_expand_empties_the_scrubber_instead_of_raising():
    """Mid-edit the segments briefly do not match the keys, which is ordinary.
    The reason is shown; Save is what actually refuses."""
    ctx, state = _clip_ctx()
    state.open_clip()["segments"] = [2]
    poser_mode.rebuild_frames(ctx)
    assert state.frames == []
    assert state.clips_error


def test_a_pump_never_reloads_over_unsaved_edits():
    ctx, state = _clip_ctx()
    state.clips_unsaved = True
    state.clips_dirty_flag = True
    poser_mode.clips_pump(ctx)
    assert poser_mode.CLIPS_KEY not in ctx.submitted
    assert state.clips_dirty_flag is True


def test_unsaved_clears_on_the_landing_and_never_at_submit():
    """A failed write has to leave the guard standing -- the same rule the pose
    save follows."""
    ctx, state = _clip_ctx()
    state.clips_unsaved = True
    poser_mode.on_task_failed(
        ctx, SimpleNamespace(key=poser_mode.CLIPS_SAVE_KEY, message="nope")
    )
    assert state.clips_unsaved is True

    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=poser_mode.CLIPS_SAVE_KEY, result={**_clip_library(), "edited": True}
        ),
    )
    assert state.clips_unsaved is False
    assert state.clips["edited"] is True


def test_a_landing_for_another_template_is_ignored():
    """The ``viewer.path`` idiom: an answer arriving after the user switched
    skeletons simply never binds."""
    ctx, state = _clip_ctx()
    state.clips_unsaved = True
    poser_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=poser_mode.CLIPS_KEY, result={**_clip_library(), "template": "quadruped"}
        ),
    )
    assert state.clips_unsaved is True
    assert state.clips["template"] == "humanoid"


def test_onion_ghosts_are_the_neighbouring_keys_not_the_neighbouring_frames():
    """A keyframe is judged against the keys it steps *between*; the frames
    either side of the playhead are what the scrubber is for."""
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.select_key(ctx, 1)
    poser_mode.set_onion(ctx, True)
    assert ctx.poser_viewer.onion == [
        {"hips": [0.0, 0.0, 0.0, 1.0]},
        {"head": [0.0, 0.0, 0.0, 1.0]},
    ]


def test_a_looping_clip_wraps_its_first_and_last_keys():
    """The comparison a looping walk actually needs, and the one that is
    easiest to get wrong by hand."""
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.set_onion(ctx, True)
    poser_mode.select_key(ctx, 0)
    before, after = ctx.poser_viewer.onion
    assert before == {"head": [0.0, 0.0, 0.0, 1.0]}, "wraps to the last key"
    assert after == {"spine": [0.0, 0.0, 0.0, 1.0]}


def test_an_open_clip_has_no_ghost_before_its_first_key():
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.set_closed(ctx, False)
    poser_mode.set_onion(ctx, True)
    poser_mode.select_key(ctx, 0)
    assert ctx.poser_viewer.onion[0] == {}


def test_scrubbing_clears_the_ghosts_and_coming_back_restores_them():
    """The live skeleton is an in-between pose while scrubbing, so ghosts of
    the neighbouring *keys* around it would be three poses on screen with no
    stated relationship between them."""
    ctx, state = _clip_ctx()
    ctx.poser_viewer.onion = []
    poser_mode.select_key(ctx, 1)
    poser_mode.set_onion(ctx, True)
    assert ctx.poser_viewer.onion

    poser_mode.scrub(ctx, 3)
    assert ctx.poser_viewer.onion == []

    poser_mode.apply_key(ctx)
    assert ctx.poser_viewer.onion


def test_turning_onion_skin_off_clears_the_viewer():
    ctx, state = _clip_ctx()
    poser_mode.set_onion(ctx, True)
    assert ctx.poser_viewer.onion
    poser_mode.set_onion(ctx, False)
    assert ctx.poser_viewer.onion == []


# --- rotation space ---------------------------------------------------------
#
# ``PoseEditor`` is node-local; the shipped clip library is ``space="delta"``.
# Nothing studio-side used to convert, so loading a humanoid key contorted the
# preview and capturing one back wrote node-local values into a delta file.


class _RestEditor:
    """Just enough editor for the two converters: a rest map, XYZW."""

    def __init__(self, rest) -> None:
        self.rest = {name: np.asarray(q, dtype="f8") for name, q in rest.items()}


def _swing(angle):
    """A rotation about local X, the swing axis both converters have to respect."""
    import math

    return [math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]


def test_delta_rotations_round_trip_through_the_node_local_editor():
    editor = _RestEditor({"upper_arm.L": _swing(1.2), "thigh.L": _swing(-0.3)})
    stored = {"upper_arm.L": _swing(-1.0), "thigh.L": _swing(0.4)}

    node = poser_mode._to_node(editor, stored, "delta")
    back = poser_mode._from_node(editor, node, "delta")

    for name, quat in stored.items():
        assert np.allclose(back[name], quat, atol=1e-12)
        # And the trip through the editor genuinely moved the values -- a
        # converter that was quietly a no-op would pass the line above.
        assert not np.allclose(node[name], quat, atol=1e-6)


def test_node_space_libraries_are_passed_through_untouched():
    editor = _RestEditor({"upper_arm.L": _swing(1.2)})
    stored = {"upper_arm.L": _swing(-1.0)}
    want = stored["upper_arm.L"]
    assert np.allclose(poser_mode._to_node(editor, stored, "node")["upper_arm.L"], want)
    assert np.allclose(poser_mode._from_node(editor, stored, "node")["upper_arm.L"], want)


def test_the_shipped_humanoid_library_is_delta_so_the_conversion_is_load_bearing():
    from warlock import rigging

    library = rigging.clip_library("humanoid")
    assert library["space"] == "delta"
