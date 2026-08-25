"""Clicking a tag: jump to it, double-click to rename it.

The band was right-click-only, so the one thing a tag is for -- "show me this
animation" -- took a menu it did not have. Aseprite jumps on the click.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.panes import inker_timeline


@pytest.fixture
def doc():
    out = inker.Document.from_pixels(np.zeros((8, 8, 4), dtype=np.uint8))
    out.ensure_animation()
    for _ in range(5):
        out.add_frame()
    out.add_tag("walk", 2, 4)
    return out


class _State:
    tag_editing = -1
    tag_name = ""


def test_clicking_a_tag_moves_the_playhead_to_its_first_frame(doc):
    doc.set_current_frame(0)
    assert inker_timeline.tag_jump(doc, doc.anim.tags[0])
    assert doc.anim.current == 2


def test_a_tag_whose_start_is_out_of_range_is_clamped(doc):
    tag = doc.anim.tags[0]
    tag.start = -3
    assert inker_timeline.tag_jump(doc, tag)
    assert doc.anim.current == 0


def test_double_clicking_a_tag_opens_the_inline_rename(doc):
    state = _State()
    inker_timeline.begin_tag_rename(state, 0, doc.anim.tags[0])
    assert state.tag_editing == 0
    assert state.tag_name == "walk"
