"""A conversion session belongs to a document, and it is settled by name.

``Document._convert`` is per document; ``InkerState`` is one object shared by
every tab; and the palette pane draws whichever tab is in front. A plain "the
popup is up" boolean therefore meant the wrong thing twice over the moment a
second tab existed:

* switching tabs with the popup open ran ``cancel_convert`` on the **new**
  document -- which had no session, so nothing was restored -- while the old one
  kept previewed pixels with no hook left to take them back;
* and a save in that state encoded the preview. The tab then went clean against
  a file holding a dither nobody had approved.

So the flag is a tab uid (``InkerState.convert_uid``) and both halves resolve
the owner through it. This file pins the two paths that must be impossible; the
popup's own drawing is exercised against a live imgui context in
``test_studio_smoke``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from warlock.studio import inker_mode
from warlock.studio.inker.document import Document
from warlock.studio.inker_state import InkerDoc, InkerState

RAMP = [(0, 0, 0, 255), (255, 255, 255, 255)]


def _tab(title: str) -> InkerDoc:
    """A tab whose document is a grey ramp -- a colour set no two-entry
    conversion leaves alone, so "did this document get previewed" is a question
    one row of pixels answers."""
    doc = Document.blank(16, 4)
    row = np.linspace(0, 255, 16).astype(np.uint8)
    doc.stack.active.pixels[..., :3] = row[None, :, None]
    doc.stack.active.pixels[..., 3] = 255
    return InkerDoc(doc=doc, title=title)


def _ctx(*tabs: InkerDoc) -> tuple[SimpleNamespace, InkerState]:
    state = InkerState()
    for tab in tabs:
        state.add(tab)
    return SimpleNamespace(state=SimpleNamespace(inker=state)), state


def _open_on(state: InkerState, tab: InkerDoc) -> np.ndarray:
    """Open a session on *tab* and preview it, as the popup would. -> the pixels
    as they were before the preview."""
    before = tab.doc.stack.active.pixels.copy()
    assert tab.doc.begin_convert() is True
    assert tab.doc.preview_convert(RAMP, "bayer4") is True
    assert not np.array_equal(tab.doc.stack.active.pixels, before)
    state.convert_uid = tab.uid
    return before


# --- the wrong document ------------------------------------------------------


def test_settling_names_the_owner_and_not_whichever_tab_is_in_front():
    first, second = _tab("first"), _tab("second")
    ctx, state = _ctx(first, second)
    before = _open_on(state, first)
    untouched = second.doc.stack.active.pixels.copy()

    # The pane is now drawing the second tab and finds a session open.
    inker_mode.end_convert_session(ctx)

    assert np.array_equal(first.doc.stack.active.pixels, before), "the owner is settled"
    assert np.array_equal(second.doc.stack.active.pixels, untouched)
    assert first.doc._convert is None
    assert state.convert_uid == ""


def test_a_tab_that_owns_no_session_settles_nothing():
    """The other half of addressing by name: a save on the second tab must not
    reach into the first one's open session."""
    first, second = _tab("first"), _tab("second")
    ctx, state = _ctx(first, second)
    _open_on(state, first)
    previewed = first.doc.stack.active.pixels.copy()

    inker_mode.end_convert_session(ctx, second)

    assert np.array_equal(first.doc.stack.active.pixels, previewed)
    assert first.doc._convert is not None
    assert state.convert_uid == first.uid


def test_settling_the_owner_by_name_closes_its_own_session():
    first, second = _tab("first"), _tab("second")
    ctx, state = _ctx(first, second)
    before = _open_on(state, first)

    inker_mode.end_convert_session(ctx, first)

    assert np.array_equal(first.doc.stack.active.pixels, before)
    assert state.convert_uid == ""


def test_a_uid_whose_tab_has_gone_clears_the_flag_and_ends_nothing():
    first = _tab("first")
    ctx, state = _ctx(first)
    _open_on(state, first)
    state.close(first.uid)

    inker_mode.end_convert_session(ctx)

    assert state.convert_uid == ""


def test_settling_with_no_session_open_is_a_no_op():
    first = _tab("first")
    ctx, state = _ctx(first)
    before = first.doc.stack.active.pixels.copy()

    inker_mode.end_convert_session(ctx)
    inker_mode.end_convert_session(ctx, first)

    assert np.array_equal(first.doc.stack.active.pixels, before)


def test_a_mode_that_was_never_opened_is_survived():
    """``ctx.state.inker`` is None until Inker is first used, and a save path
    that ran before then must not raise."""
    inker_mode.end_convert_session(SimpleNamespace(state=SimpleNamespace(inker=None)))


# --- the save path -----------------------------------------------------------


def test_settling_before_a_save_leaves_the_users_own_pixels_to_encode():
    """The bug as the user meets it: Ctrl+S with the popup up wrote the preview,
    and ``mark_saved`` then called the document clean against it."""
    tab = _tab("first")
    ctx, state = _ctx(tab)
    before = _open_on(state, tab)

    inker_mode._settle(ctx, tab)

    assert np.array_equal(tab.doc.stack.active.pixels, before)
    assert tab.doc._convert is None
    assert state.convert_uid == ""


def test_the_settle_still_commits_a_floating_buffer():
    """The other half of what ``_settle`` replaced. A float is *committed* and a
    preview is *cancelled* -- they are opposite mistakes, and the one call has
    to keep doing both."""
    tab = _tab("first")
    ctx, _state = _ctx(tab)
    tab.doc.select_all()
    tab.doc.lift()
    assert tab.doc.floating is not None

    inker_mode._settle(ctx, tab)

    assert tab.doc.floating is None


def test_settling_a_save_on_one_tab_leaves_another_tabs_session_alone():
    first, second = _tab("first"), _tab("second")
    ctx, state = _ctx(first, second)
    _open_on(state, first)
    previewed = first.doc.stack.active.pixels.copy()

    inker_mode._settle(ctx, second)

    assert np.array_equal(first.doc.stack.active.pixels, previewed)
    assert state.convert_uid == first.uid
