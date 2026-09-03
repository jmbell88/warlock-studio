"""Textures: from the selection, from a generated picture, saved beside the
recipe, and stamped by the sprite and particle primitives."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from warlock.studio import inker, inker_flourish, inker_mode, inker_ops, inker_state
from warlock.studio.inker import flourish, ora
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import presets
from warlock.studio.tasks import Done


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get(self, job_id):
        return self.jobs.get(job_id)


class _Svc:
    def __init__(self, root: Path) -> None:
        self.store = _Store()
        self.root = root
        self.config = None

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id


class _Ctx:
    def __init__(self, root: Path) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.svc = _Svc(root)
        self.toasts: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.pending: list[Done] = []
        self._busy: set[str] = set()

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return key in self._busy

    def progress(self, key):
        return None

    def submit(self, key, fn, *args, **kwargs):
        if key in self._busy:
            return False
        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            done = Done(key=key, error=exc)
        self._busy.add(key)
        self.pending.append(done)
        return True

    def land_all(self):
        while self.pending:
            done = self.pending.pop(0)
            self._busy.discard(done.key)
            inker_mode.on_task_done(self, done)


def _scene(tmp_path):
    ctx = _Ctx(tmp_path)
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=2)
    group = tab.doc.insert_flourish(B.bake(rec))
    return ctx, tab, group


def _select(doc, x0, y0, x1, y1, colour=(200, 40, 40, 255)):
    """Paint a block on the active layer and select it."""
    from warlock.studio.inker.selection import SelectionMask

    doc.stack.active.pixels[y0:y1, x0:x1] = colour
    mask = np.zeros(doc.size[::-1], dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    doc.mask = SelectionMask(mask=mask)


def test_add_and_remove_an_asset_are_steps(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    doc = tab.doc
    tex = np.full((4, 4, 4), 255, dtype=np.uint8)
    asset_id = doc.add_flourish_asset(group, tex)
    assert asset_id == "tex1"
    assert doc.add_flourish_asset(group, tex, stem="gen") == "gen1"
    assert set(doc.flourish_state(group).assets) == {"tex1", "gen1"}
    doc.history.undo(doc)
    assert set(doc.flourish_state(group).assets) == {"tex1"}
    assert doc.remove_flourish_asset(group, "tex1")
    assert doc.flourish_state(group).assets == {}
    assert not doc.remove_flourish_asset(group, "tex1")
    with pytest.raises(ValueError):
        doc.add_flourish_asset(group, np.zeros((0, 4, 4), dtype=np.uint8))


def test_the_selection_becomes_a_texture_on_the_inspectors_layer(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    state = ctx.state.inker
    op = inker_ops.get("flourish_texture_selection")
    assert not op.enabled(state, tab)
    assert inker_ops.reason_for(op, state, tab) == inker_flourish.NO_SELECTION
    # Point the inspector at the sparks (a particles layer: it takes a texture).
    rec = tab.doc.flourish_state(group).recipe
    sparks = next(each for each in rec.layers if each.kind == "particles")
    state.flourish_layer[group] = sparks.uid
    _select(tab.doc, 4, 4, 12, 10)
    assert op.enabled(state, tab)
    assert inker_ops.run(ctx, op)
    held = tab.doc.flourish_state(group)
    assert list(held.assets) == ["tex1"]
    assert held.assets["tex1"].shape == (6, 8, 4)
    assert tuple(held.assets["tex1"][0, 0]) == (200, 40, 40, 255)
    # The layer took it, as a pending edit for the next render.
    pending = state.flourish_pending[group]
    assert pending.layer(sparks.uid).params["texture"] == "tex1"


def test_a_layer_without_a_texture_slot_gets_the_asset_but_no_edit(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    state = ctx.state.inker
    rec = tab.doc.flourish_state(group).recipe
    glow = next(each for each in rec.layers if each.kind == "glow")
    state.flourish_layer[group] = glow.uid
    _select(tab.doc, 0, 0, 4, 4)
    assert inker_flourish.texture_from_selection(ctx, state, tab) == "tex1"
    assert group not in state.flourish_pending


def test_textures_travel_with_the_render(tmp_path, monkeypatch):
    ctx, tab, group = _scene(tmp_path)
    tex = np.zeros((4, 4, 4), dtype=np.uint8)
    tex[..., :3] = 255
    tex[..., 3] = 255
    tab.doc.add_flourish_asset(group, tex)
    seen: dict = {}
    import warlock.studio.inker.flourish.bake as bake_mod

    real = bake_mod.bake

    def spy(recipe, **kw):
        seen.update(kw)
        return real(recipe, **kw)

    monkeypatch.setattr(bake_mod, "bake", spy)
    rec = tab.doc.flourish_state(group).recipe
    assert inker_flourish.submit_render(ctx, tab, group, rec)
    assert list(seen["assets"]) == ["tex1"]


def test_assets_survive_an_ora_round_trip(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    doc = tab.doc
    tex = np.zeros((5, 7, 4), dtype=np.uint8)
    tex[1:4, 2:5] = (10, 200, 30, 255)
    doc.add_flourish_asset(group, tex)
    rec = doc.flourish_state(group).recipe
    sparks = next(each for each in rec.layers if each.kind == "particles")
    doc.set_flourish_recipe(group, rec.replace_layer(sparks.with_param("texture", "tex1")))
    path = tmp_path / "fx.ora"
    ora.write_ora(doc, path)
    again = inker.Document.load(path)
    (guid, state), = again.flourish.items()
    assert list(state.assets) == ["tex1"]
    assert np.array_equal(state.assets["tex1"], tex)
    assert state.recipe.layer(sparks.uid).params["texture"] == "tex1"
    import zipfile

    with zipfile.ZipFile(path) as zf:
        assert any(n.startswith("data/flourish0_tex1") for n in zf.namelist())


def test_key_out_black_makes_alpha_from_brightness():
    picture = np.zeros((2, 2, 4), dtype=np.uint8)
    picture[0, 0] = (255, 255, 255, 255)
    picture[0, 1] = (0, 0, 0, 255)
    picture[1, 0] = (120, 40, 0, 255)
    cut = inker_flourish.key_out_black(picture)
    assert cut[0, 0, 3] == 255 and cut[0, 1, 3] == 0 and cut[1, 0, 3] == 120


def test_the_generate_door_queues_polls_decodes_and_lands(tmp_path, monkeypatch):
    ctx, tab, group = _scene(tmp_path)
    state = ctx.state.inker
    rec = tab.doc.flourish_state(group).recipe
    sparks = next(each for each in rec.layers if each.kind == "particles")
    state.flourish_layer[group] = sparks.uid
    calls: list = []

    def fake_create_job(svc, **kwargs):
        calls.append(kwargs)
        return {"id": "job1"}

    from warlock.service import jobs as svc_jobs

    monkeypatch.setattr(svc_jobs, "create_job", fake_create_job)
    assert inker_ops.run(ctx, inker_ops.get("flourish_texture_generate"))
    assert state.pending_dialog == inker_flourish.TEXTURE_POPUP
    assert inker_mode.flourish_texture_generate(ctx, tab, subject="skull ember")
    assert state.flourish_texture_pending is not None
    assert "skull ember" in calls[0]["prompt"] and calls[0]["output"] == "reference"
    # A second ask while one is pending is refused with a tip.
    assert not inker_mode.flourish_texture_generate(ctx, tab, subject="again")
    ctx.land_all()  # the queue answered with the job id
    assert state.flourish_texture_pending["job_id"] == "job1"
    # Still running: nothing lands.
    ctx.svc.store.jobs["job1"] = {"status": "running"}
    inker_flourish.poll_texture(ctx, state, now=100.0)
    assert state.flourish_texture_pending is not None
    # Done: the picture on black is keyed out and lands as an asset.
    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    picture = Image.new("RGB", (64, 64), (0, 0, 0))
    for x in range(16, 48):
        for y in range(16, 48):
            picture.putpixel((x, y), (255, 200, 50))
    picture.save(job_dir / "input.png")
    ctx.svc.store.jobs["job1"] = {"status": "done"}
    inker_flourish.poll_texture(ctx, state, now=200.0)
    assert state.flourish_texture_pending is None
    ctx.land_all()
    held = tab.doc.flourish_state(group)
    assert list(held.assets) == ["gen1"]
    tex = held.assets["gen1"]
    assert tex.shape == (64, 64, 4)
    assert tex[0, 0, 3] == 0 and tex[32, 32, 3] == 255
    assert state.flourish_pending[group].layer(sparks.uid).params["texture"] == "gen1"
    assert ctx.toasts[-1][1] == "success"


def test_a_failed_job_is_a_warning_and_clears_the_pending(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    state = ctx.state.inker
    state.flourish_texture_pending = {
        "tab_uid": tab.uid, "group": group, "layer": None, "job_id": "j", "next_poll": 0.0
    }
    ctx.svc.store.jobs["j"] = {"status": "failed", "error": "out of memory"}
    inker_flourish.poll_texture(ctx, state, now=1.0)
    assert state.flourish_texture_pending is None
    assert ctx.toasts[-1][1] == "warn"


def test_a_sprite_layer_renders_once_it_has_a_texture(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    doc = tab.doc
    tex = np.zeros((6, 6, 4), dtype=np.uint8)
    tex[1:5, 1:5] = (255, 255, 255, 255)
    asset_id = doc.add_flourish_asset(group, tex)
    rec = doc.flourish_state(group).recipe
    sprite = flourish.Layer(
        uid=flourish.new_uid(), kind="sprite", params={"texture": asset_id, "size": 10.0}
    )
    with_sprite = flourish.clamp(dataclasses.replace(rec, layers=(*rec.layers, sprite)))
    empty = B.bake(with_sprite)  # no assets handed in: the sprite paints nothing
    full = B.bake(with_sprite, assets=doc.flourish_state(group).assets)
    assert empty.facings[0].layers["hit"].get(sprite.uid) is not None
    a = sum(int(c[..., 3].sum()) for c in empty.facings[0].layers["hit"][sprite.uid])
    b = sum(int(c[..., 3].sum()) for c in full.facings[0].layers["hit"][sprite.uid])
    assert a == 0 and b > 0
