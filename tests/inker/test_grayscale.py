"""Grayscale mode: a constraint at the funnel, over the storage we already have.

Aseprite stores ``(value, alpha)``; this stores ``(v, v, v, a)`` and enforces
``r == g == b`` where every other write constraint is enforced. The claim that
makes that sound -- rather than merely enforced at the door -- is that **all
nineteen blend modes preserve grayness**, so the composite of a grayscale
document is grey too. That is the property test at the bottom of this file, and
it is the one worth reading first.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import Document
from warlock.studio.inker import composite as cp
from warlock.studio.inker import indexed as ix

RED = (200, 20, 20, 255)
GREEN = (20, 200, 20, 255)


def _gray(doc) -> np.ndarray:
    plane = doc.stack.active.pixels
    return plane[..., :3][plane[..., 3] > 0]


def _is_gray(rgb: np.ndarray) -> bool:
    return bool((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all())


# --- the constraint ---------------------------------------------------------


def test_the_luma_is_the_one_the_rest_of_the_app_uses():
    """Rec. 709, shared with the palette sort and ``dither.build_palette``.
    Converting a document to grayscale and then sorting its palette by
    brightness has to produce the order the drawing actually has."""
    from warlock.studio.inker import dither

    px = np.zeros((1, 1, 4), np.uint8)
    px[0, 0] = RED
    value = int(ix.grayscale(px)[0, 0, 0])
    assert value == int(np.floor(dither.luma(RED) + 0.5))


def test_alpha_rides_through_and_a_transparent_pixel_is_left_alone():
    """``indexed.snap``'s rules, restated: a palette -- or a luma -- is a
    statement about colour and never about opacity, and a transparent pixel has
    no colour to convert. Rewriting its dead RGB would make a no-op write look
    like an edit to the funnel."""
    px = np.zeros((1, 3, 4), np.uint8)
    px[0, 0] = RED
    px[0, 1] = (7, 8, 9, 0)
    px[0, 2] = (10, 20, 30, 128)

    out = ix.grayscale(px)
    assert tuple(out[0, 1]) == (7, 8, 9, 0), "returned verbatim"
    assert out[0, 0, 3] == 255 and out[0, 2, 3] == 128
    assert out[0, 2, 0] == out[0, 2, 1] == out[0, 2, 2]


def test_the_caller_s_array_is_not_touched():
    px = np.zeros((1, 1, 4), np.uint8)
    px[0, 0] = RED
    ix.grayscale(px)
    assert tuple(px[0, 0]) == RED


# --- the funnel -------------------------------------------------------------


def test_every_write_path_lands_grey():
    """The funnel is the single door, which is what makes this a property of the
    model rather than a list somebody remembered."""
    doc = Document.blank(8, 8)
    doc.convert_to_grayscale()

    doc.begin_stroke((2, 2), RED, size=3)
    doc.end_stroke()
    assert _is_gray(_gray(doc))

    doc.fill((5, 5), GREEN, thresh=255)
    assert _is_gray(_gray(doc))

    doc.shape("ellipse", (1, 1), (6, 6), GREEN, 1, filled=True)
    assert _is_gray(_gray(doc))

    doc.gradient((0, 0), (7, 7), RED, GREEN)
    assert _is_gray(_gray(doc)), "an antialiased gradient is still constrained"


def test_converting_flattens_the_whole_document_in_one_step():
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.add_layer()
    doc.stack.active.pixels[:, :] = GREEN
    head = doc.history.head

    assert doc.convert_to_grayscale() is True
    assert doc.color_mode == "grayscale" and doc.is_grayscale
    assert all(_is_gray(layer.pixels[..., :3]) for layer in doc.stack)

    doc.undo()
    assert doc.history.head == head
    assert doc.color_mode == "rgb"
    assert tuple(doc.stack[0].pixels[0, 0]) == RED
    assert tuple(doc.stack[1].pixels[0, 0]) == GREEN


def test_converting_to_grayscale_twice_pushes_nothing():
    doc = Document.blank(4, 4)
    doc.convert_to_grayscale()
    head = doc.history.head
    assert doc.convert_to_grayscale() is False
    assert doc.history.head == head


def test_leaving_grayscale_keeps_the_grey_pixels():
    """Lossless in the only direction that matters: the colour is gone and there
    is nothing to restore it from, so ``convert_to_rgb`` restores the *mode* and
    says so rather than inventing hues."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.convert_to_grayscale()
    grey = doc.stack.active.pixels.copy()

    assert doc.convert_to_rgb() is True
    assert doc.color_mode == "rgb"
    assert np.array_equal(doc.stack.active.pixels, grey)

    doc.begin_stroke((1, 1), RED, size=1, nib="pixel")
    doc.end_stroke()
    assert tuple(doc.stack.active.pixels[1, 1]) == RED, "the constraint is lifted"


def test_an_indexed_document_converting_to_grayscale_drops_its_planes():
    """An index plane and a luma constraint are two answers to "what is a
    pixel", and a document cannot hold both."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.convert_to_indexed([(0, 0, 0, 255), RED, GREEN])

    assert doc.convert_to_grayscale() is True
    assert doc.color_mode == "grayscale"
    assert doc.stack.active.indices is None
    assert doc.palette is None
    assert _is_gray(_gray(doc))
    doc.check_materialized()

    doc.undo()
    assert doc.is_indexed and doc.stack.active.indices is not None
    doc.check_materialized()


def test_geometry_and_undo_keep_a_grayscale_document_grey():
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = RED
    doc.convert_to_grayscale()

    doc.flip("horizontal")
    doc.rotate90(1)
    doc.scale((4, 4), resample="smooth")
    assert _is_gray(_gray(doc))
    doc.undo()
    doc.undo()
    doc.undo()
    assert _is_gray(_gray(doc))


# --- the argument -----------------------------------------------------------


@pytest.mark.parametrize("mode", sorted(cp.BLEND_MODES))
def test_every_blend_mode_preserves_grayness(mode):
    """**The load-bearing claim.** The channelwise modes preserve it trivially --
    each channel is computed from equal inputs by one formula -- and the HSL
    family does because a grey has zero saturation, so a hue or saturation
    transfer from a grey leaves a grey and a luminosity transfer is grey by
    definition.

    Without this, grayscale would be a constraint the *storage* enforced and the
    *compositor* broke: two grey layers under `hue` would put colour on screen
    that no layer contains and no export could explain.
    """
    rng = np.random.default_rng(11)
    values = rng.integers(0, 256, size=(6, 6), dtype=np.uint8)
    alphas = rng.integers(0, 256, size=(2, 6, 6), dtype=np.uint8)

    planes = []
    for i in range(2):
        plane = np.zeros((6, 6, 4), np.uint8)
        plane[..., 0] = plane[..., 1] = plane[..., 2] = np.roll(values, i)
        plane[..., 3] = alphas[i]
        planes.append(plane)

    out = cp.to_uint8(
        cp.stack_region(
            [(planes[0], 1.0, "normal"), (planes[1], 1.0, mode)], (0, 0, 6, 6)
        )
    )
    visible = out[..., 3] > 0
    assert _is_gray(out[..., :3][visible]), f"{mode} put colour into a grey composite"


def test_a_grayscale_document_composites_grey_end_to_end():
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = RED
    doc.add_layer()
    doc.stack.active.pixels[:, :] = GREEN
    doc.stack.active.blend = "hue"
    doc.stack.active.opacity = 0.6
    doc.convert_to_grayscale()

    flat = doc.flatten(matte=False)
    visible = flat[..., 3] > 0
    assert _is_gray(flat[..., :3][visible])


def test_a_grayscale_document_with_a_palette_gets_both_constraints():
    """The rule ``_constrained`` exists to hold in one place: grayscale and the
    palette snap are two ``if``s and not an ``if``/``elif``, because a snap onto
    a table of greys is not the same array as a snap onto the table the document
    actually has. It was written out at three sites, each with its own comment
    explaining it."""
    import numpy as np

    from warlock.studio import inker

    doc = inker.Document.blank(4, 4)
    doc.color_mode = "grayscale"
    doc.palette = [(0, 0, 0, 255), (128, 128, 128, 255), (255, 255, 255, 255)]

    region = np.zeros((2, 2, 4), np.uint8)
    region[...] = (200, 30, 30, 255)
    out = doc._constrained(region)

    r, g, b, a = (int(v) for v in out[0, 0])
    assert r == g == b, "flattened to luma"
    assert (r, g, b, a) in [tuple(int(v) for v in c) for c in doc.palette], "and snapped"


def test_the_constraint_rule_is_written_out_only_once():
    import pathlib

    root = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "warlock"
        / "studio"
        / "inker"
    )
    # The *pairing* is the rule, not the bare mode check -- ``to_grayscale`` asks
    # the same question to answer "already grayscale", which is a different
    # thing. A file that calls both primitives is applying the constraint.
    wrote = []
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "ix.grayscale(" in source and "ix.snap(" in source:
            wrote.append(path.name)
    assert wrote == ["document.py"], wrote
