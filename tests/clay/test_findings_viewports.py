"""The 2026-09-02 review's section 6: Clay, Poser, Troupe and the shared viewer.

The entries were struck from the findings file as they were built, per the
repository's rule that a built thing is deleted rather than ticked; this is
what keeps them fixed. Grouped here rather than spread across the mode suites
because several of them are one bug wearing two hats -- the rotation frames and
the two viewports' keyboards especially.
"""

from __future__ import annotations

import numpy as np
import pytest

# --- the two rotation frames have one definition ------------------------------


def test_the_node_delta_conversion_is_one_function():
    """It was written twice -- once against the viewer's rest quaternions and
    once against Blender's -- and neither knew about the other. The multiply is
    one line; the order and which side is conjugated are what drift."""
    from warlock import rigging

    rest = [0.0, 0.0, 0.3826834, 0.9238795]  # 45 degrees about Z
    delta = [0.0, 0.0, 0.3826834, 0.9238795]

    node = rigging.node_from_delta(rest, delta)
    assert node == pytest.approx([0.0, 0.0, 0.7071068, 0.7071068])

    # And back, exactly: the pair are inverses.
    assert rigging.delta_from_node(rest, node) == pytest.approx(delta)


def test_both_ends_of_the_conversion_call_it():
    import inspect

    from warlock.pipelines import blender_worker
    from warlock.studio import poser_mode

    assert "rigging.delta_from_node" in inspect.getsource(blender_worker)
    assert "rigging.node_from_delta" in inspect.getsource(poser_mode)
    assert "rigging.delta_from_node" in inspect.getsource(poser_mode)


# --- what a file is allowed to contain ----------------------------------------


def test_a_clip_librarys_quaternions_are_checked():
    """They were taken verbatim: a 3-element list raised out of the middle of
    ``sheet._blend`` naming neither the file nor the bone, and a NaN was
    interpolated into a clip and written to disk."""
    from warlock import rigging

    raw = {
        "space": "delta",
        "poses": [{"name": "a", "bones": {"spine": [0.0, 0.0, 0.0]}}],
        "clips": [{"name": "c", "keys": ["a"], "segments": [1], "closed": False}],
    }
    with pytest.raises(ValueError, match="spine"):
        rigging.parse_clip_library(raw)

    raw["poses"][0]["bones"]["spine"] = [float("nan"), 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="not numeric"):
        rigging.parse_clip_library(raw)


def test_a_clips_rest_key_may_hold_no_bones_at_all():
    """Which is why ``validate_bones`` is split out rather than being a flag on
    ``validate_pose``: a *saved pose* with no bones is a mistake and a clip's
    rest key is exactly that map."""
    from warlock import rigging

    assert rigging.validate_bones({}) == {}
    with pytest.raises(ValueError):
        rigging.validate_pose({"name": "p", "bones": {}})


def test_a_drifted_quaternion_is_renormalised_rather_than_refused():
    from warlock import rigging

    out = rigging.validate_bones({"spine": [0.0, 0.0, 0.0, 1.0000001]})
    assert sum(v * v for v in out["spine"]) == pytest.approx(1.0)


# --- the viewer's own geometry ------------------------------------------------


def test_skin_weights_are_renormalised_and_a_dead_vertex_is_pinned():
    """The shader sums ``joint * weight`` with no division, so a vertex whose
    weights sum to 0 collapsed onto the origin and the mesh grew a spike to the
    world centre."""
    from warlock.studio.viewer import gltf

    raw = np.array([[0.5, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype="f4")
    weights = raw.copy()
    total = weights.sum(axis=1, keepdims=True)
    dead = total[:, 0] <= 0.0
    weights[dead] = 0.0
    weights[dead, 0] = 1.0
    total = weights.sum(axis=1, keepdims=True)
    out = weights / total

    assert out[0].sum() == pytest.approx(1.0)
    assert list(out[1]) == [1.0, 0.0, 0.0, 0.0]
    # And the module really does it, rather than this test doing it alone.
    assert "weights / total" in gltf.__loader__.get_source(gltf.__name__)


# --- the ops registry ---------------------------------------------------------


def test_shade_auto_can_actually_reach_its_whole_document_branch():
    """It was gated on ``has_objects``, which requires a *selection*, so the
    fallback its own comment describes could never be taken."""
    from types import SimpleNamespace

    from warlock.studio import clay_ops

    empty = SimpleNamespace(selection=set(), objects=[])
    unselected = SimpleNamespace(selection=set(), objects=[SimpleNamespace(uid=1)])

    assert clay_ops.any_object(empty) is False
    assert clay_ops.any_object(unselected) is True
    assert clay_ops.has_objects(unselected) is False

    op = next(o for o in clay_ops.OPS if o.name == "shade-auto")
    assert op.enabled is clay_ops.any_object


def test_select_more_uses_the_one_definition_of_a_selected_face():
    """It reimplemented "a face is selected only when all of its corners are"
    as a Python loop over every face -- a second spelling of the rule those
    verbs rest on being inverses of each other."""
    import inspect

    from warlock.studio import clay_ops

    body = inspect.getsource(clay_ops._sel_from_verts)
    assert "_face_corner_mask" in body
    assert "for face in range(" not in body


def test_dissolve_edges_does_not_rescan_the_mesh_per_edge():
    import inspect

    from warlock.studio.clay import ops_dissolve

    body = inspect.getsource(ops_dissolve.dissolve_edges)
    assert "a.corner_edge == e" not in body
    assert "searchsorted" in body


def test_a_bevel_copies_the_faces_it_does_not_touch():
    import inspect

    from warlock.studio.clay import ops_bevel

    body = inspect.getsource(ops_bevel)
    assert "if not per_face[face]:" in body


# --- the shared viewer's input ------------------------------------------------


def test_the_viewer_orbits_on_alt_drag_like_clay():
    """Alt+drag panned here for two days (a trackpad has no middle button)
    while ``_view_drag`` said the same gesture "must never be reinterpreted"
    and orbited. One rule since 2026-09-05: Alt+drag orbits in both viewports,
    and the pan is the middle button's."""
    import inspect

    from warlock.studio import viewer_embed

    body = inspect.getsource(viewer_embed.Viewer._press)
    assert 'self._grab = "pan" if button == 2 else "orbit"' in body
    assert "KMOD_ALT" not in body


def test_a_bare_hover_does_not_redraw_the_scene():
    """Every event marked the MSAA target dirty, so moving the mouse across the
    pane re-rendered a picture nothing in had changed."""
    import inspect

    from warlock.studio import viewer_embed

    handler = inspect.getsource(viewer_embed.Viewer.handle_event)
    assert handler.count("self._render_dirty = True") == 3  # press, release, wheel
    motion = inspect.getsource(viewer_embed.Viewer._motion)
    assert "if gizmo.hover != was:" in motion
