"""Enumerating an Inker document, and why a sprite's key is not its name.

The key is identity and the packer's total order, so it is derived from *where*
a sprite came from. Names legitimately repeat -- two layers called "Layer 1" is
an ordinary document -- and two sprites sharing a key would pack into one slot
with the loser simply missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker.document import Document
from warlock.studio.packwright import sources


def _doc(width: int = 8, height: int = 8) -> Document:
    return Document.blank(width, height)


def _paint(doc: Document, colour: tuple[int, int, int, int]) -> None:
    doc.stack.active.pixels[:] = colour


# --- still documents ----------------------------------------------------------


def test_a_still_document_gives_one_sprite_per_layer_in_stack_order():
    doc = _doc()
    _paint(doc, (255, 0, 0, 255))
    doc.add_layer("top")
    _paint(doc, (0, 0, 255, 255))

    got = sources.sprites_from_document(doc, prefix="art")
    assert len(got) == len(doc.stack)
    assert [s.name for s in got] == [layer.name for layer in doc.stack]
    assert tuple(got[0].pixels[0, 0]) == (255, 0, 0, 255)


def test_a_hidden_layer_is_still_enumerated():
    """The pane chooses what to include. An enumerator that silently dropped
    rows would make "why is my sprite missing" a question about two places."""
    doc = _doc()
    doc.add_layer("hidden")
    doc.stack.active.visible = False
    assert len(sources.sprites_from_document(doc, prefix="art")) == len(doc.stack)


def test_two_layers_of_one_name_get_distinct_keys():
    doc = _doc()
    doc.stack.active.name = "same"
    doc.add_layer("same")
    keys = [s.key for s in sources.sprites_from_document(doc, prefix="art")]
    assert len(set(keys)) == len(keys)


# --- animated documents -------------------------------------------------------


def test_an_animated_document_gives_one_sprite_per_frame():
    doc = _doc()
    _paint(doc, (255, 0, 0, 255))
    doc.add_frame(copy=True)
    _paint(doc, (0, 255, 0, 255))
    doc.add_frame(copy=True)
    _paint(doc, (0, 0, 255, 255))

    got = sources.sprites_from_document(doc, prefix="walk")
    assert [s.key for s in got] == [
        "walk#frame0000",
        "walk#frame0001",
        "walk#frame0002",
    ]


def test_a_packed_frame_is_pixel_identical_to_what_the_timeline_plays():
    """Through ``frame_flat``, which is the same flatten the timeline plays and
    the onion skin draws -- so there is one answer to "what does this frame look
    like" rather than a packer's private one."""
    doc = _doc()
    _paint(doc, (255, 0, 0, 255))
    doc.add_frame(copy=True)
    _paint(doc, (0, 0, 255, 255))

    got = sources.sprites_from_document(doc, prefix="walk")
    for index, frame in enumerate(doc.anim.frames):
        assert np.array_equal(got[index].pixels, doc.frame_flat(frame.uid))


def test_frame_keys_sort_temporally():
    """Zero-padded, because the packer's canonical order is a lexical sort of
    the key and frame 10 arriving before frame 2 would reorder a clip."""
    keys = [sources.frame_key("a", i) for i in (0, 2, 10, 100)]
    assert keys == sorted(keys)


def test_a_linked_cel_appearing_twice_is_two_sprites():
    """A sheet is played by an engine that knows nothing about links, so a cel
    in two frames is two cells -- the rule ``inker.sheetout`` already states."""
    doc = _doc()
    _paint(doc, (255, 0, 0, 255))
    doc.add_frame(link=True)
    got = sources.sprites_from_document(doc, prefix="walk")
    assert len(got) == 2
    assert np.array_equal(got[0].pixels, got[1].pixels)
    assert got[0].key != got[1].key


# --- the sprite type ----------------------------------------------------------


def test_a_sprite_owns_a_read_only_copy_of_its_pixels():
    """The UI keys a texture upload on the array's identity, so an in-place
    edit would leave the cache holding a live key over stale pixels."""
    pixels = np.zeros((4, 4, 4), np.uint8)
    sprite = sources.Sprite(key="a", name="a", pixels=pixels)
    assert sprite.pixels is not pixels
    assert not sprite.pixels.flags.writeable
    pixels[0, 0] = (1, 2, 3, 4)
    assert tuple(sprite.pixels[0, 0]) == (0, 0, 0, 0)


def test_a_non_rgba_sprite_is_refused():
    with pytest.raises(ValueError, match="RGBA"):
        sources.Sprite(key="a", name="a", pixels=np.zeros((4, 4, 3), np.uint8))


def test_a_sprite_needs_a_key():
    with pytest.raises(ValueError, match="needs a key"):
        sources.Sprite(key="", name="a", pixels=np.zeros((4, 4, 4), np.uint8))


def test_sprite_dimensions_come_from_the_array():
    sprite = sources.sprite_from_image(np.zeros((5, 9, 4), np.uint8), key="a")
    assert (sprite.width, sprite.height) == (9, 5)
    assert sprite.name == "a"
