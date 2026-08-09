"""``params["bone_count"]`` finally has a reader.

``queue._rig`` records three things about a finished rig side by side --
``weighting``, ``weighting_reason`` and ``bone_count`` -- and the inspector
showed the first two and nothing at all about the third, so a rig could report
*how* it was bound and never *what* it was bound to. The count is also the
cheapest possible sanity check on a fit: a humanoid template that came back
with three bones did not fit.

The two sources are ``rig_weighting``'s two sources, in its order and for its
reason: the **rig** job's row carries the number (a card must not open a file
per asset to draw a badge) and the **model** job the user selects carries
``rig.json`` with the joints themselves (a pose editor needs their names and
positions). They are disjoint in practice, so preferring the row means the
answer is always about the row on screen.
"""

from __future__ import annotations

import json
from typing import Any

from warlock.studio.panes import inspector
from warlock.studio.state import AppState


class FakeCtx:
    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = AppState()

    def submit(self, key: str, run: Any, *args: Any) -> bool:  # pragma: no cover
        raise AssertionError("the rig summary is a frame-thread read, not a task")


def _rigged(svc, bones):
    """A finished mesh with a rig.json beside it, the way the worker leaves one."""
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "rig.json").write_text(
        json.dumps(
            {
                "version": 1,
                "template": "humanoid",
                "weighting": "automatic",
                "bones": [{"name": f"bone{i}"} for i in range(bones)],
            }
        ),
        "utf-8",
    )
    return svc.store.get(job_id)


# --- the pure half ------------------------------------------------------------


def test_a_rig_jobs_row_carries_the_number():
    assert inspector.bone_count_of({"bone_count": 19}) == 19


def test_a_rig_file_carries_the_joints_and_is_counted():
    assert inspector.bone_count_of({"bones": [{"name": "hips"}, {"name": "spine"}]}) == 2


def test_a_row_written_before_the_key_existed_simply_has_no_count():
    """Read with ``.get`` throughout: every asset already on disk predates this
    reader, and a rig job from last week carries neither spelling."""
    assert inspector.bone_count_of({}) is None
    assert inspector.bone_count_of({"weighting": "envelope"}) is None
    assert inspector.bone_count_of(None) is None


def test_a_nonsense_count_is_no_count():
    """Params are JSON off disk. A bool is an ``int`` as far as ``isinstance``
    is concerned, which is exactly the kind of value that would render as
    "1 bone" against a rig with nineteen."""
    assert inspector.bone_count_of({"bone_count": 0}) is None
    assert inspector.bone_count_of({"bone_count": -3}) is None
    assert inspector.bone_count_of({"bone_count": True}) is None
    assert inspector.bone_count_of({"bone_count": "nineteen"}) is None
    assert inspector.bone_count_of({"bones": []}) is None
    assert inspector.bone_count_of({"bones": "hips"}) is None


# --- the two sources ----------------------------------------------------------


def test_the_count_is_read_off_the_rig_beside_the_selected_mesh(svc):
    """The model job the user selects has the file and no params, which is the
    selection the Rig & Pose tab actually opens on."""
    ctx = FakeCtx(svc)
    job = _rigged(svc, 19)

    assert (job.get("params") or {}).get("bone_count") is None
    assert inspector.rig_bones(ctx, job) == 19


def test_a_rig_jobs_own_row_answers_for_itself(svc):
    """Selected in the library, a rig job has the params and no rig.json."""
    ctx = FakeCtx(svc)
    job_id = svc.store.create(
        "rig", "a chest", {"weighting": "automatic", "bone_count": 19},
        stage="model", status="done",
    )
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)

    assert inspector.rig_bones(ctx, svc.store.get(job_id)) == 19


def test_the_row_wins_where_both_exist(svc):
    """``rig_weighting``'s order, for its reason: the answer is about the row
    the user is looking at."""
    ctx = FakeCtx(svc)
    job = _rigged(svc, 19)
    job["params"] = {"bone_count": 3}

    assert inspector.rig_bones(ctx, job) == 3


def test_an_unrigged_mesh_has_no_count_and_no_error(svc):
    ctx = FakeCtx(svc)
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)

    assert inspector.rig_bones(ctx, svc.store.get(job_id)) is None


def test_the_count_is_in_the_rig_tab(svc):
    """Beside the weighting line it belongs with, not in a tab of its own."""
    import inspect as inspect_mod

    source = inspect_mod.getsource(inspector._rig_tab)
    assert "_bones(ctx, job)" in source
    assert source.index("_weighting(ctx, job)") < source.index("_bones(ctx, job)")
