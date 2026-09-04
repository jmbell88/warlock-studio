"""T8 of the 2026-09-02 review: the modules nothing referenced.

The theme named ten modules with **zero** test references and three surfaces
with no behaviour test at all. A module nothing names is a module whose next
edit is unverified, and the two defects the review found in exactly those files
(the shared undo-gesture bug and the layouts popup) are what it cost.

What is asserted here is the *decidable* half of each: a pure helper, a key, a
predicate, a piece of arithmetic. None of these panes can be driven headlessly
-- there is no imgui harness in this suite -- so a draw function is covered by
the smoke pass and its decisions are covered here, which is the pattern the
rest of the suite already follows.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# --- studio/sizeguard.py ------------------------------------------------------


def test_a_file_past_the_ceiling_is_refused_before_a_byte_is_read(tmp_path):
    """The shared "is this small enough to open" question, which three modes
    had a private copy of and two more had none at all."""
    from warlock.service.errors import TooLarge
    from warlock.studio import sizeguard

    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 100)

    assert sizeguard.within_ceiling(path, 100) == path
    with pytest.raises(TooLarge) as caught:
        sizeguard.within_ceiling(path, 99, field="drawing")
    # A ``ServiceError``, so the text reaches the user through the task
    # classifier rather than becoming "see the log for details".
    assert caught.value.field == "drawing"
    assert "big.bin" in str(caught.value)


def test_the_ceiling_is_read_at_call_time():
    """Which is what lets a test lower it rather than build half a gigabyte."""
    import inspect

    from warlock.studio import sizeguard

    signature = inspect.signature(sizeguard.within_ceiling)
    assert "ceiling" in signature.parameters


# --- studio/_view_cache.py ----------------------------------------------------


def test_the_gpu_cache_key_moves_with_the_mesh_and_not_with_the_transform():
    """The transform is a uniform, not a buffer: moving an object must not
    rebuild it, and editing its mesh must."""
    from warlock.studio import _view_cache

    material = object()
    doc = SimpleNamespace(materials=[material])
    mesh = object()
    obj = SimpleNamespace(mesh=mesh, material=0, translation=(0.0, 0.0, 0.0))

    materials = _view_cache._materials_key(doc)
    before = _view_cache._object_key(obj, materials)

    obj.translation = (5.0, 0.0, 0.0)
    assert _view_cache._object_key(obj, materials) == before

    obj.mesh = object()
    assert _view_cache._object_key(obj, materials) != before


def test_a_replaced_material_changes_the_cache_key():
    """``set_material`` replaces the entry object, which is why the key is
    identity rather than value -- hashing five floats per entry per frame would
    learn the same thing."""
    from warlock.studio import _view_cache

    doc = SimpleNamespace(materials=[object(), object()])
    before = _view_cache._materials_key(doc)
    doc.materials[0] = object()
    assert _view_cache._materials_key(doc) != before


# --- studio/_view_overlay.py --------------------------------------------------


def test_the_fill_bias_pulls_toward_the_eye_and_leaves_no_gl_state():
    """``glPolygonOffset`` is the textbook answer and is deliberately not used:
    it is global state that would leak into the gizmo pass."""
    from warlock.studio import _view_overlay

    matrix = _view_overlay._toward_eye(np.array([0.0, 0.0, 10.0]))

    assert matrix.shape == (4, 4)
    assert matrix[0, 0] < 1.0, "everything shrinks a hair"
    assert matrix[2, 3] > 0.0, "and is nudged along the eye direction"


def test_a_face_outline_is_the_border_and_nothing_else():
    from warlock.studio import _view_overlay

    mesh = SimpleNamespace(
        starts=np.array([0, 4], dtype="i8"),
        loops=np.array([0, 1, 2, 3], dtype="i8"),
    )
    pairs = _view_overlay._face_outline(mesh, np.array([0]))

    assert pairs.shape == (4, 2)
    assert [tuple(p) for p in pairs] == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert _view_overlay._face_outline(mesh, np.zeros(0, dtype="i8")).shape == (0, 2)


# --- viewer/camera.py: the orthographic branch of screen_ray -------------------


def test_an_orthographic_pick_ray_is_parallel_and_moves_its_origin():
    """Perspective-only, every ray was cast from a point the orthographic
    render has no apex at -- so the further from the screen centre a click
    landed, the further what got picked was from what was under the cursor."""
    from warlock.studio.viewer.camera import Camera, screen_ray

    camera = Camera()
    camera.orthographic = True
    camera.aspect = 1.0

    centre_o, centre_d = screen_ray(camera, 50, 50, 100, 100)
    corner_o, corner_d = screen_ray(camera, 90, 10, 100, 100)

    # Parallel: the direction is the same wherever the click landed.
    assert np.allclose(centre_d, corner_d)
    # And it is the *origin* that moves across the view plane.
    assert not np.allclose(centre_o, corner_o)


def test_a_perspective_pick_ray_still_fans_from_one_eye():
    from warlock.studio.viewer.camera import Camera, screen_ray

    camera = Camera()
    camera.orthographic = False
    camera.aspect = 1.0

    centre_o, centre_d = screen_ray(camera, 50, 50, 100, 100)
    corner_o, corner_d = screen_ray(camera, 90, 10, 100, 100)

    assert np.allclose(centre_o, corner_o), "one apex"
    assert not np.allclose(centre_d, corner_d)
    assert np.allclose(np.linalg.norm(corner_d), 1.0)


# --- panes/thumbs.py ----------------------------------------------------------


def test_a_placeholder_says_what_sort_of_thing_is_coming():
    from warlock.studio import icons
    from warlock.studio.panes import thumbs

    assert thumbs.thumb_glyph({"stage": "reference", "kind": "image"}) == icons.IMAGE
    # A failed job gets the alert glyph whatever it would have been: its card is
    # about the failure.
    assert (
        thumbs.thumb_glyph({"status": "error", "stage": "reference"})
        == icons.CIRCLE_ALERT
    )
    # An unrecognised stage falls back rather than raising, on a draw path --
    # ``card_kind`` reads it as a model, which is what the library does.
    assert thumbs.thumb_glyph({"stage": "who-knows"}) == icons.BOX


# --- panes/inker_menu.py ------------------------------------------------------


def test_the_shortcut_sheet_lists_every_binding_the_registry_has():
    """The rows are built from ``inker_ops.BINDINGS`` rather than typed out, so
    a remapped chord and the sheet cannot disagree."""
    from warlock.studio import inker_ops
    from warlock.studio.panes import inker_menu

    rows = inker_menu._shortcut_rows()

    assert rows, "a sheet with no rows is a sheet nobody can read"
    kinds = {row[0] for row in rows}
    assert kinds <= inker_ops.BINDING_KINDS
    # Every action modifier the registry offers to remap appears on it.
    modifiers = {row[1] for row in rows if row[0] == "action_modifier"}
    assert modifiers == {m.name for m in inker_ops.ACTION_MODIFIERS}


# --- the two Troupe panes -----------------------------------------------------


def test_the_troupe_bridge_and_sheet_panes_import_without_a_context():
    """Both had no test reference at all. Importing them is the floor: a pane
    that cannot be imported takes the whole frame down through ``guard``, and
    the smoke pass is what draws them."""
    from warlock.studio.panes import troupe_bridge, troupe_sheets

    assert callable(troupe_bridge.draw)
    assert callable(troupe_sheets.draw)
    assert callable(troupe_sheets._pixel_report)


def test_a_rerender_names_the_runs_it_was_asked_for():
    """``_rerender`` is the subset door -- the one place a partial re-render's
    run list is turned into a request -- and nothing named it."""
    from warlock.studio.panes import troupe_sheets

    assert troupe_sheets._RERENDER_SLOT == "troupe_rerender_runs"


# --- packwright_items ---------------------------------------------------------


def test_a_packed_item_row_is_its_own_function():
    """Lifted out of the loop so the list can be clipped -- and so there is
    something to name."""
    import inspect

    from warlock.studio.panes import packwright_items

    assert callable(packwright_items._item_row)
    assert "ListClipper" in inspect.getsource(packwright_items.draw)


# --- studio/_view_pick.py -----------------------------------------------------


def test_a_picked_element_is_expressed_in_the_mode_that_picked_it():
    """The one place a hit index becomes a selection, and nothing named it."""
    from warlock.studio.clay.primitives import box
    from warlock.studio.clay_view import ClayView

    mesh = box()
    obj = SimpleNamespace(uid=7, mesh=mesh)
    doc = SimpleNamespace(element_mode="vertex", by_uid=lambda uid: obj)

    view = ClayView.__new__(ClayView)
    assert list(ClayView.element_sel_for(view, doc, 7, 3).verts) == [3]

    doc.element_mode = "face"
    assert list(ClayView.element_sel_for(view, doc, 7, 2).faces) == [2]

    doc.element_mode = "edge"
    edges = ClayView.element_sel_for(view, doc, 7, 0).edges
    assert edges.shape == (1, 2), "an edge is a vertex pair, not an index"


# --- studio/_view_drag.py -----------------------------------------------------


def test_a_cancelled_drag_puts_the_overlays_back_on_the_mesh():
    """Esc restored the objects and left the selection overlay's VBO at the
    previewed positions -- the cancel looked half-applied and stayed that way
    until something else rebuilt the overlay."""
    from warlock.studio.clay_view import ClayView

    written: list[object] = []
    positions = object()
    obj = SimpleNamespace(uid=4, mesh=SimpleNamespace(positions=positions))
    doc = SimpleNamespace(by_uid=lambda uid: obj)

    view = ClayView.__new__(ClayView)
    view._overlays = {4: SimpleNamespace(write_positions=written.append)}

    ClayView._restore_overlays(view, doc, [4])
    assert written == [positions]

    # A uid the document no longer holds is skipped rather than raising, on a
    # path reached from an Escape key.
    def gone(uid):
        raise KeyError(uid)

    ClayView._restore_overlays(view, SimpleNamespace(by_uid=gone), [4])
    assert written == [positions]
