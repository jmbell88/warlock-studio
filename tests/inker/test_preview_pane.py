"""The preview pane's discipline, which is entirely a list of things it must not do.

``panes/inker_preview.py`` carries a second playhead so a clip can run in the
corner while the document is painted on, and ``docs/INVARIANTS.md`` spells out
what makes that free: it never sets ``playing``, ``saving`` or
``set_current_frame``, it draws through the same ``frame_texture`` onion
skinning uses so it adds no GPU state, it ignores ``PaintView``'s rotation and
flip, and it re-flattens at most four times a second.

The whole of that was unasserted -- no test named the module. ``tick_preview``'s
arithmetic was covered; the pane around it was not, and the invariant is about
the pane.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

from warlock.studio.panes import inker_preview

SOURCE = inspect.getsource(inker_preview)
TREE = ast.parse(SOURCE)

#: Every attribute *accessed* in the module, and every name called. Read off the
#: syntax tree rather than grepped out of the text: this module's own docstring
#: names the calls it promises not to make ("never sets ``playing``, ``saving``
#: or ``set_current_frame``"), so a text scan asserts the prose rather than the
#: code and passes whatever the code does.
ATTRS = {node.attr for node in ast.walk(TREE) if isinstance(node, ast.Attribute)}
NAMES = {node.id for node in ast.walk(TREE) if isinstance(node, ast.Name)}


def test_it_never_moves_the_documents_playhead():
    """The one call that would undo the whole point: ``set_current_frame``
    re-materialises the stack and recomposites the canvas."""
    assert "set_current_frame" not in ATTRS


def test_it_never_touches_the_documents_playing_or_saving_flags():
    """``playing`` locks the tab and draws into the canvas; ``saving`` is what
    ``busy`` refuses mutation on. The preview owns neither -- it has
    ``preview_playing`` of its own."""
    for banned in ("playing", "saving"):
        assert banned not in ATTRS, banned
    assert "preview_playing" in ATTRS


def test_it_adds_no_gpu_state_of_its_own():
    """It draws ``Document.frame_flat`` through the same
    ``inker_textures.frame_texture`` onion skinning already uses, so a frame the
    preview shows and a frame the onion skin shows are one texture."""
    assert "frame_texture" in ATTRS
    for banned in ("texture", "release", "forget_texture"):
        assert banned not in ATTRS, banned


def test_it_draws_upright():
    """``PaintView``'s rotation and flip are aids for *drawing*; a preview is a
    check on the result, and following the view would turn the thing being
    checked along with the check."""
    for banned in ("basis", "rotation", "flipped"):
        assert banned not in ATTRS and banned not in NAMES, banned


def test_the_refresh_throttle_is_four_times_a_second():
    assert inker_preview.REFRESH_SECONDS == 0.25


def test_the_speed_key_snaps_to_the_nearest_rung():
    """Nearest rather than exact: the value is a float on the tab and a combo
    key is a string, so a current value matching nothing would silently reset
    the speed to the first entry on the next frame."""
    assert inker_preview._speed_key(1.0) == "1.0"
    assert inker_preview._speed_key(0.9) == "1.0"
    assert inker_preview._speed_key(3.5) == "4.0"
    assert inker_preview._speed_key(0.0) == "0.25"
    assert inker_preview._speed_key(99.0) == "4.0"
    for value, _label in inker_preview.SPEEDS:
        assert inker_preview._speed_key(value) == f"{value}"


def test_every_speed_option_is_offered_by_its_own_key():
    keys = [key for key, _label in inker_preview._speed_options()]
    assert keys == [f"{value}" for value, _label in inker_preview.SPEEDS]


def test_the_throttle_hands_back_the_existing_texture_between_refreshes():
    """The existing texture is returned untouched in between, which is what
    makes the throttle cost nothing -- it is the same object the next call would
    return, one flatten and one upload later."""
    calls: list[int] = []
    monkey = SimpleNamespace(frame_texture=lambda ctx, tab, uid: calls.append(uid) or "fresh")
    real, inker_preview.inker_textures = inker_preview.inker_textures, monkey
    try:
        ctx = SimpleNamespace(viewer=object(), state=SimpleNamespace(preview={}))
        tab = SimpleNamespace(uid="t1")

        assert inker_preview._throttled(ctx, tab, 7) == "fresh"
        assert calls == [7]

        # The cache the pane reads is ``frame_texture``'s own slot.
        ctx.state.preview["inker_tex:t1:frame7"] = "cached"
        assert inker_preview._throttled(ctx, tab, 7) == "cached"
        assert calls == [7], "no second flatten inside the window"
    finally:
        inker_preview.inker_textures = real


def test_the_throttle_answers_nothing_without_a_viewer():
    ctx = SimpleNamespace(viewer=None, state=SimpleNamespace(preview={}))
    assert inker_preview._throttled(ctx, SimpleNamespace(uid="t1"), 1) is None


def test_its_per_tab_key_is_swept_when_the_tab_closes():
    """``inker_preview:{uid}`` is not a texture, so the ``inker_tex:`` prefix
    sweep does not reach it -- it has to be named in ``_PER_TAB_KEYS``."""
    from warlock.studio.panes import inker_textures

    assert "inker_preview:" in inker_textures._PER_TAB_KEYS
