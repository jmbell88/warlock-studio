"""The 2026-09-02 review's Inker engine findings, pinned.

Five defects in ``studio/inker/`` that a document could carry silently: a
preview that rewrote the frame under the cursor, a merge that discarded per-cel
properties, a move that repainted an indexed layer in another slot's name, a
flood fill that stopped at pixels nobody could see, and three smaller losses
around the floating buffer and the clipboard. The entries were struck from the
findings file as they were built, per the repository's rule that a built thing
is deleted rather than ticked; this is what keeps them fixed.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import selection
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask


def _animated(frames: int = 2, *, link: bool = True) -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(link=link)
    doc.set_current_frame(0)
    return doc


# --- materialising another frame must not rewrite this one -------------------


def test_previewing_a_frame_does_not_change_what_this_one_composites_to():
    """``Document.stack`` *is* the cel objects -- it has to be, or a stroke
    would land on a copy -- so materialising another frame rewrote the opacity
    of the very layers the current stack is made of."""
    doc = _animated(2)
    track = doc.anim.tracks[0]
    doc.set_cel_opacity(0.25, 0, 1)

    live = doc.stack[0]
    assert live.opacity == 1.0

    # An onion skin of frame 2, which links the same cel at a quarter alpha.
    doc.frame_stack(doc.anim.frames[1]).flatten()

    assert live.opacity == 1.0
    assert doc.anim.cels[(track.uid, doc.anim.frames[0].uid)].opacity == 1.0


def test_a_detached_stack_still_shares_the_pixels_it_reports():
    """It is a shallow copy: a flatten must not cost a plane per layer."""
    doc = _animated(1)
    track = doc.anim.tracks[0]
    cel = doc.anim.cels[(track.uid, doc.anim.frames[0].uid)]

    detached = doc.anim.layers_for(doc.anim.frames[0], doc.size, detach=True)[0]

    assert detached is not cel
    assert detached.pixels is cel.pixels


# --- merge_down carries the per-cel properties -------------------------------


def _two_tracks() -> Document:
    """Two painted rows over two frames. The rows are painted *before* the
    animation exists, because a fresh row on an animated document is a
    placeholder over the shared read-only blank plane."""
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.add_layer()
    doc.stack[1].pixels[:, :] = (0, 0, 255, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.set_current_frame(0)
    return doc


def test_merging_down_honours_a_faded_cel():
    """The effective alpha of a slot is ``track x cel``. Merging at the
    track's number alone brought a faded drawing back at full strength on
    exactly the frames somebody had faded."""
    doc = _two_tracks()
    upper = doc.anim.tracks[1]
    doc.set_cel_opacity(0.0, 1, 0)

    doc.merge_down(1)

    merged = doc.stack[0].pixels
    # The upper cel was invisible on this frame, so the merge is the lower's
    # own red -- not the blue it used to produce.
    assert tuple(merged[0, 0]) == (255, 0, 0, 255)
    assert upper.uid not in {track.uid for track in doc.anim.tracks}


def test_a_merged_cel_does_not_wear_its_dimming_twice():
    doc = _two_tracks()
    lower = doc.anim.tracks[0]
    doc.set_cel_opacity(0.5, 0, 0)

    doc.merge_down(1)

    assert doc.anim.cel_alpha(lower.uid, doc.anim.frames[0].uid) == 1.0


def test_a_merge_across_a_lifted_cel_is_refused_by_name():
    """``cel_z`` reorders rows *within one frame*, so on a frame where a lift
    puts a third row between the pair the two layers this op merges are not
    the two layers that frame composites."""
    import pytest

    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.add_layer()
    doc.stack[1].pixels[:, :] = (0, 0, 255, 255)
    doc.add_layer()
    doc.stack[2].pixels[:, :] = (0, 255, 0, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.set_current_frame(0)
    # Drop the top row below the pair on frame 1 only.
    doc.set_cel_z(-2, 2, 0)

    with pytest.raises(ValueError, match="frame 1"):
        doc.merge_down(2)


def test_an_undisturbed_merge_still_runs_with_zeros_in_the_z_table():
    doc = _two_tracks()
    doc.set_cel_z(0, 1, 0)
    assert doc.merge_down(1) is True


# --- duplicate carries the row and the slots ---------------------------------


def test_duplicating_an_animated_row_keeps_its_per_slot_tables():
    """They are keyed on ``(track uid, frame uid)``, so a duplicate started
    with none of them: a faded cel came back at full strength, a lifted one
    dropped to its track's position, and every cel note was lost."""
    # Linked, so both slots hold a cel: ``set_cel_z`` is refused on an empty
    # one, by its own rule.
    doc = _animated(2)
    source = doc.anim.tracks[0]
    doc.set_cel_opacity(0.5, 0, 0)
    doc.set_cel_z(1, 0, 1)
    assert doc.anim.cel_zindex(source.uid, doc.anim.frames[1].uid) == 1
    source.continuous = True

    doc.duplicate_layer(0)

    copy = doc.anim.tracks[1]
    assert copy.uid != source.uid
    assert copy.continuous is True
    assert doc.anim.cel_alpha(copy.uid, doc.anim.frames[0].uid) == 0.5
    assert doc.anim.cel_zindex(copy.uid, doc.anim.frames[1].uid) == 1


# --- the flood fill sees an erased pixel as empty ----------------------------


def test_two_fully_transparent_pixels_are_the_same_pixel():
    """RGB under alpha 0 is dead data, so comparing it made a fill over erased
    artwork stop at the outline of what used to be drawn."""
    pixels = np.zeros((1, 3, 4), dtype=np.uint8)
    pixels[0, 1] = (200, 30, 30, 0)  # an erased red pixel: invisible, not equal

    distance = selection.colour_distance(pixels, pixels[0, 0])

    assert list(distance[0]) == [0, 0, 0]


def test_an_opaque_seed_still_differs_from_a_transparent_pixel():
    pixels = np.zeros((1, 2, 4), dtype=np.uint8)
    pixels[0, 0] = (200, 30, 30, 255)

    distance = selection.colour_distance(pixels, pixels[0, 0])

    assert distance[0, 0] == 0
    assert distance[0, 1] == 255


# --- the floating buffer and the clipboard -----------------------------------


def test_flipping_a_fresh_buffer_keeps_the_pivot_it_was_given():
    """Seeding ``source`` without ``source_offset`` threw the pivot away, so
    the first flip of a gesture made every later rotate turn about the
    buffer's centre instead."""
    buffer = selection.FloatingBuffer(
        pixels=np.zeros((2, 2, 4), dtype=np.uint8),
        mask=np.full((2, 2), 255, dtype=np.uint8),
        offset=(5, 7),
        layer_uid=1,
    )
    buffer.flip("horizontal")
    assert buffer.source_offset == (5, 7)


def test_a_paste_with_no_selection_lands_where_the_copy_was_made():
    doc = Document.blank(8, 8)
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.select(SelectionMask.from_rect(doc.size, (3, 4, 6, 7)))
    doc.copy()
    assert doc.clipboard.origin == (3, 4)

    doc.select(None)
    doc.paste()

    assert doc.floating is not None
    assert doc.floating.offset == (3, 4)


def test_content_from_outside_the_document_still_pastes_at_the_origin():
    doc = Document.blank(8, 8)
    doc.put_clipboard(np.zeros((2, 2, 4), dtype=np.uint8))
    doc.paste()
    assert doc.floating is not None
    assert doc.floating.offset == (0, 0)


# --- Move permutes the index plane rather than re-resolving it ---------------


def test_moving_an_indexed_layer_permutes_its_slots():
    """Two palette entries holding the same colour are two entries; the patch
    funnel's nearest-match collapsed them, so a drag silently repainted the
    layer in the other slot's name."""
    doc = Document.blank(4, 1)
    doc.palette = [(0, 0, 0, 0), (10, 10, 10, 255), (10, 10, 10, 255)]
    doc.color_mode = "indexed"
    layer = doc.stack[0]
    layer.indices = np.array([[0, 2, 0, 0]], dtype=np.uint16)
    layer.pixels[0, 1] = (10, 10, 10, 255)

    doc.begin_layer_move()
    doc.preview_layer_move(1, 0)
    doc.commit_layer_move()

    assert list(layer.indices[0]) == [0, 0, 2, 0]


def test_cancelling_a_move_puts_the_index_plane_back_too():
    doc = Document.blank(4, 1)
    doc.palette = [(0, 0, 0, 0), (10, 10, 10, 255)]
    doc.color_mode = "indexed"
    layer = doc.stack[0]
    layer.indices = np.array([[1, 0, 0, 0]], dtype=np.uint16)
    layer.pixels[0, 0] = (10, 10, 10, 255)

    doc.begin_layer_move()
    doc.preview_layer_move(2, 0)
    doc.cancel_layer_move()

    assert list(layer.indices[0]) == [1, 0, 0, 0]


# --- the app layer: refusals that say why, and state that is not gesture -----


def test_space_mid_stroke_no_longer_cancels_its_own_pan():
    """Keyboard state is not gesture state. Pressing Space during a stroke
    ends the stroke, and ``clear_drag`` then cleared the very flag the press
    had just set."""
    from warlock.studio.inker_state import InkerState

    state = InkerState()
    state.space_held = True
    state.drag_kind = "paint"

    state.clear_drag()

    assert state.drag_kind == ""
    assert state.space_held is True

    # A tab switch is a different matter: the pane genuinely stops being the
    # one the release would arrive at.
    state.forget_held_keys()
    assert state.space_held is False


def test_the_three_writing_selection_ops_wait_for_a_save():
    """They put pixels into a layer, so a save walking ``doc.stack`` on a task
    thread is exactly the moment they may not run."""
    from types import SimpleNamespace

    from warlock.studio import inker_ops

    doc = Document.blank(4, 4)
    doc.select_all()
    state = SimpleNamespace(transforming=False)
    idle = SimpleNamespace(doc=doc, busy=False)
    saving = SimpleNamespace(doc=doc, busy=True)

    for name in ("fill_selection", "stroke_selection", "shift_selected"):
        op = inker_ops.get(name)
        assert op.enabled(state, idle) is True
        assert op.enabled(state, saving) is False
        assert inker_ops.reason_for(op, state, saving) == inker_ops.BUSY


def test_every_remappable_modifier_has_a_reader():
    """The tuple listed eighteen, the shortcut editor offered all eighteen for
    remapping, and two were read."""
    import inspect
    from pathlib import Path

    from warlock.studio import inker_ops

    canvas = (
        Path(inspect.getfile(inker_ops)).parent / "panes" / "inker_canvas.py"
    ).read_text(encoding="utf-8")
    combine = inspect.getsource(inker_ops)
    for modifier in inker_ops.ACTION_MODIFIERS:
        assert f'"{modifier.name}"' in canvas or f'"{modifier.name}"' in combine, modifier.name
    assert len(inker_ops.ACTION_MODIFIERS) == 9
    # And every one of them still has a default binding to be remapped from.
    targets = {
        binding.target
        for binding in inker_ops.BINDINGS
        if binding.kind == "action_modifier"
    }
    assert targets == {modifier.name for modifier in inker_ops.ACTION_MODIFIERS}


def test_a_lossy_aseprite_write_is_reported_rather_than_called_a_save():
    from warlock.studio.inker import aseout

    plain = Document.blank(4, 4)
    assert aseout.dropped_by_aseprite(plain) == []

    lossy = Document.blank(4, 4)
    lossy.matte = (1.0, 1.0, 1.0, 1.0)
    lossy.stack[0].alpha_lock = True
    lost = aseout.dropped_by_aseprite(lossy)
    assert "the flatten matte" in lost
    assert "layer alpha lock" in lost


def test_a_fractional_wheel_notch_does_not_leave_the_zoom_lattice():
    """A 0.3 notch took the view to 101.5% and carried that fraction forever."""
    from warlock.studio.inker_state import PaintView, zoom_step

    view = PaintView()
    for _ in range(3):
        zoom_step(view, (0.0, 0.0), (0.0, 0.0), 0.3)
    assert view.zoom == 1.0  # nothing whole has accumulated yet

    zoom_step(view, (0.0, 0.0), (0.0, 0.0), 0.3)
    assert round(view.zoom * 100) == 105


def test_a_layer_drag_follows_the_layer_and_not_the_slot():
    """A drag spans frames; an index is a position, and an undo landing while
    the button is held leaves it naming whichever layer moved into that slot."""
    from warlock.studio.panes import inker_timeline

    doc = Document.blank(4, 4)
    doc.add_layer()
    top = doc.stack[1].uid

    assert inker_timeline._row_of_uid(doc, top) == 1
    doc.move_layer(1, 0)
    assert inker_timeline._row_of_uid(doc, top) == 0
    assert inker_timeline._row_of_uid(doc, -1) is None


# --- file IO -----------------------------------------------------------------


def test_a_frame_duration_is_clamped_on_every_write():
    """``__post_init__`` clamped the value a frame was *born* with, so the two
    importers that set it from a file could put a number past the format's
    ``<H`` on one -- and ``aseout`` then died packing it."""
    from warlock.studio.inker.animation import MAX_DURATION_MS, MIN_DURATION_MS, Frame

    frame = Frame()
    frame.duration_ms = 10**9
    assert frame.duration_ms == MAX_DURATION_MS
    frame.duration_ms = -5
    assert frame.duration_ms == MIN_DURATION_MS
    frame.duration_ms = "junk"
    assert isinstance(frame.duration_ms, int)


def test_a_cel_is_built_from_the_copied_down_set_rather_than_a_hand_list():
    import inspect

    from warlock.studio.inker import animation, ora

    assert "track.props()" in inspect.getsource(ora)
    props = animation.Track(name="x", background=True, reference=True).props()
    assert set(props) == set(animation.CEL_PROPS)
    assert props["background"] is True and props["reference"] is True


def test_the_transparent_slot_has_one_answer_for_every_reader():
    """A file naming a slot past its own palette decoded its tilesets against
    a table row that does not exist while its cels decoded against slot 0."""
    import inspect

    from warlock.studio.inker import asein

    body = inspect.getsource(asein)
    assert "_lut(sprite.palette or [], sprite.transparent_index)" not in body
    assert "_lut(sprite.palette or [], _transparent_slot(sprite))" in body


def test_both_document_writers_stage_through_the_one_helper():
    import inspect

    from warlock.studio.inker import aseout, ora

    for module in (aseout, ora):
        body = inspect.getsource(module)
        assert 'with_name(path.name + ".tmp")' not in body
    assert "atomic.staged(path)" in inspect.getsource(ora.write_ora)


def test_a_text_stamp_cannot_ask_for_an_unbounded_surface():
    """The surface is measured from the string in the field at the size in the
    field, and nothing stood between a 4000-point paste and ``Image.new``."""
    import pytest

    from warlock.studio import fonts
    from warlock.studio.inker import textstamp

    font = str(fonts.FONT_DIR / "Inter-Regular.ttf")
    # ``MAX_SIZE`` caps the point size; the *string* was never capped, and at
    # the largest legal size a pasted paragraph is still a surface nothing
    # measured before allocating it.
    with pytest.raises(ValueError, match="past the"):
        textstamp.text_stamp("M" * 40_000, font, textstamp.MAX_SIZE, (255, 255, 255, 255))

    # And the ordinary case still draws.
    assert textstamp.text_stamp("Hi", font, 24, (255, 0, 0, 255)) is not None


# --- UX parity ----------------------------------------------------------------


def test_the_brush_footprint_is_where_the_engine_will_stamp():
    """The cursor was a circle at the raw mouse position, which says how wide
    the brush is and nothing about which pixels it will hit."""
    from warlock.studio.inker import brush

    # A pixel nib anchors on the pixel it is on: odd centred, even down-right.
    assert brush.footprint((4.5, 4.5), 1, "pixel") == (4, 4, 5, 5)
    assert brush.footprint((4.5, 4.5), 3, "pixel") == (3, 3, 6, 6)
    assert brush.footprint((4.5, 4.5), 2, "pixel") == (4, 4, 6, 6)
    # Everything else by the rounding form.
    assert brush.footprint((4.5, 4.5), 4, "soft") == (3, 3, 7, 7)


def test_a_two_frame_clip_draws_one_onion_ghost_and_not_two():
    """The span wraps, so -1 and +1 are the same frame on a two-frame clip and
    it was drawn twice -- two tints and two fades on one picture."""
    from warlock.studio.inker_state import onion_index

    assert onion_index(0, -1, (0, 1)) == 1
    assert onion_index(0, 1, (0, 1)) == 1
    assert onion_index(0, 1, (0, 0)) is None


def test_the_picker_holds_its_own_triple():
    """A grey has no hue, so dragging Hue moved nothing and the slider snapped
    back to 0."""
    from warlock.studio.inker_state import InkerState

    state = InkerState()
    assert state.picker_space is None


# --- performance ---------------------------------------------------------------


def test_an_idle_filter_popup_writes_nothing():
    """The filter was memoised; the blend, the invalidate and the upload behind
    them still ran on every frame the popup was open."""
    doc = Document.blank(8, 8)
    doc.stack[0].pixels[:, :] = (200, 30, 30, 255)
    doc.select_all()
    doc.begin_filter()

    assert doc.preview_filter("invert") is True
    before = doc.rev
    assert doc.preview_filter("invert") is True
    assert doc.rev == before, "an unchanged preview must not invalidate"

    # A different filter is a different answer and does write.
    doc.preview_filter("sharpen")
    assert doc.rev != before


def test_a_selection_change_drops_the_written_filter_signature():
    from warlock.studio.inker.selection import SelectionMask

    doc = Document.blank(8, 8)
    doc.select_all()
    doc.begin_filter()
    doc.preview_filter("invert")
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))
    before = doc.rev
    doc.preview_filter("invert")
    assert doc.rev != before


def test_a_click_that_draws_nothing_does_not_restamp_the_whole_document():
    """``_discard_pending_cel`` went through ``_set_cel``, which stamps every
    frame and recomposites the canvas -- on the most repeated gesture there
    is."""
    doc = _animated(3)
    doc.add_layer()
    doc.set_current_frame(1)
    track = doc.anim.tracks[1]
    frame = doc.anim.frames[1]
    other = doc.anim.frames[2]

    before = doc.frame_stamp(other.uid)
    doc._ensure_cel_for(doc.stack[1].uid)
    assert (track.uid, frame.uid) in doc.anim.cels
    doc._discard_pending_cel()

    assert (track.uid, frame.uid) not in doc.anim.cels
    assert doc.frame_stamp(other.uid) == before, "another frame's cache is untouched"


def test_the_content_box_cache_forgets_a_document_that_is_gone():
    """``id()`` is recycled the moment the object at it is collected, and
    nothing pruned the dict."""
    import gc

    from warlock.studio.panes import inker_canvas

    doc = Document.blank(4, 4)
    doc.stack[0].pixels[1, 1] = (1, 2, 3, 255)
    assert inker_canvas._content_box(doc, doc.stack[0]) == (1, 1, 2, 2)
    ident = id(doc)
    assert any(key[0] == ident for key in inker_canvas._CONTENT_BOX)

    del doc
    gc.collect()
    assert not any(key[0] == ident for key in inker_canvas._CONTENT_BOX)


def test_the_mirror_preview_draws_runs_rather_than_pixels():
    from warlock.studio.panes.inker_canvas import _runs

    assert _runs([1, 2, 3, 7], [0, 0, 0, 0], None) == [
        (1, 0, 3, False),
        (7, 0, 1, False),
    ]
    # The face box splits a run: the two sides are different colours.
    assert len(_runs([1, 2, 3], [0, 0, 0], (2, 0, 3, 1))) == 3


def test_the_gif_palette_map_is_over_distinct_colours():
    import numpy as np

    from warlock.studio.inker import gifout

    palette = [(0, 0, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255)]
    frame = np.zeros((2, 2, 4), dtype=np.uint8)
    frame[..., 3] = 255
    frame[0, 1] = (255, 0, 0, 255)
    frame[1, 0] = (0, 255, 0, 255)

    image = gifout.map_to_palette(frame, palette)
    assert list(np.asarray(image).ravel()) == [0, 1, 2, 0]
