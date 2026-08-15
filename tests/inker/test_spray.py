"""The spray tool: a distribution over positions, and nothing else.

The seam this file guards is the one a random tool always has. The *engine* is
deterministic given a seed and a call sequence, and the UI is what supplies
entropy -- a fresh 32-bit seed at every press. Get that backwards and a spray is
either untestable or the same cloud every time.

Everything else about a dab -- symmetry, the selection clip, the alpha lock,
tiled wrapping, one undo step per press -- comes free because ``spray`` emits
through ``_dab``, and each of those is asserted here rather than assumed.
"""

from __future__ import annotations

import numpy as np

from warlock.studio import inker
from warlock.studio.inker import brush

SIZE = (40, 40)
RED = (255, 0, 0, 255)


def _sprayed(*, seed=7, count=60, scatter=8.0, doc=None, calls=1, **kw):
    doc = doc if doc is not None else inker.Document.blank(*SIZE)
    doc.begin_stroke(
        (20.0, 20.0), RED, size=1, nib="pixel", scatter=scatter, seed=seed, **kw
    )
    for _ in range(calls):
        doc.spray_at((20.0, 20.0), count)
    doc.end_stroke()
    return doc.stack.active.pixels


def test_the_same_seed_and_the_same_calls_give_the_same_pixels():
    assert np.array_equal(_sprayed(seed=7), _sprayed(seed=7))


def test_a_different_seed_gives_a_different_cloud():
    assert not np.array_equal(_sprayed(seed=7), _sprayed(seed=8))


def test_the_stream_follows_the_call_sequence_rather_than_the_chunking():
    """Two calls of thirty dabs and one of sixty draw from the same stream in
    the same order, so a spray does not depend on the frame rate it was drawn
    at -- the same property the spacing carry gives an ordinary stroke."""
    assert np.array_equal(
        _sprayed(seed=3, count=60, calls=1), _sprayed(seed=3, count=30, calls=2)
    )


def test_each_stroke_starts_from_its_own_generator():
    """Not the global numpy stream: two documents sprayed in one session must
    not draw from each other's sequence, or "same seed, same picture" holds
    only for the first one."""
    first = inker.Document.blank(*SIZE)
    _sprayed(seed=5, doc=first)
    second = inker.Document.blank(*SIZE)
    _sprayed(seed=5, doc=second)
    assert np.array_equal(first.stack.active.pixels, second.stack.active.pixels)


def test_the_dabs_land_inside_the_scatter_disc():
    pixels = _sprayed(seed=11, count=400, scatter=6.0)
    ys, xs = np.nonzero(pixels[..., 3])
    assert xs.size > 0
    distance = np.hypot(xs - 20.0, ys - 20.0)
    # A pixel nib of diameter 1 is anchored on the pixel it is on, so a dab at
    # radius r marks a pixel whose centre is within one pixel of it.
    assert distance.max() <= 6.0 + 1.5


def test_the_density_does_not_pile_up_at_the_centre():
    """``sqrt`` of a uniform sample. With a uniform *radius* the inner disc --
    a quarter of the area -- would take half the dabs, and a spray would read as
    a dot with a halo."""
    pixels = _sprayed(seed=13, count=1200, scatter=12.0)
    ys, xs = np.nonzero(pixels[..., 3])
    distance = np.hypot(xs - 20.0, ys - 20.0)
    inner = int((distance <= 6.0).sum())
    total = int(distance.size)
    # Uniform over the disc puts a quarter of the marks inside half the radius.
    assert 0.15 < inner / total < 0.40


def test_a_zero_scatter_spray_is_a_single_dab():
    pixels = _sprayed(seed=2, count=20, scatter=0.0)
    assert int(np.count_nonzero(pixels[..., 3])) == 1


def test_a_count_of_nothing_emits_nothing():
    doc = inker.Document.blank(*SIZE)
    doc.begin_stroke((20.0, 20.0), RED, size=1, nib="pixel", scatter=5.0, seed=1)
    before = doc.stack.active.pixels.copy()
    doc.spray_at((20.0, 20.0), 0)
    assert np.array_equal(doc.stack.active.pixels, before)


def test_one_press_to_release_is_one_undo_step():
    doc = inker.Document.blank(*SIZE)
    blank = doc.stack.active.pixels.copy()
    _sprayed(seed=4, calls=5, doc=doc)
    assert doc.history.can_undo
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, blank)
    assert not doc.history.can_undo


def test_a_selection_clips_the_cloud():
    doc = inker.Document.blank(*SIZE)
    doc.select(inker.SelectionMask.from_rect(SIZE, (0, 0, SIZE[0], 20)))
    pixels = _sprayed(seed=6, count=300, scatter=12.0, doc=doc)
    assert int(pixels[20:, :, 3].max()) == 0
    assert int(pixels[:20, :, 3].max()) > 0


def test_symmetry_mirrors_every_dab_without_the_spray_knowing():
    """Emission is at the *position* level, so the mirrors apply to a sprayed
    dab exactly as they apply to a walked one."""
    pixels = _sprayed(seed=9, count=80, scatter=4.0, symmetry="x")
    left = int(np.count_nonzero(pixels[:, : SIZE[0] // 2, 3]))
    right = int(np.count_nonzero(pixels[:, SIZE[0] // 2 :, 3]))
    assert left > 0 and right > 0


def test_a_spray_at_the_edge_wraps_when_the_document_is_tiled():
    doc = inker.Document.blank(*SIZE)
    doc.begin_stroke(
        (0.0, 20.0), RED, size=1, nib="pixel", scatter=6.0, seed=15, wrap="x"
    )
    doc.spray_at((0.0, 20.0), 200)
    doc.end_stroke()
    assert int(doc.stack.active.pixels[..., 3][:, -3:].max()) > 0


def test_the_scatter_and_seed_default_to_the_stroke_the_editor_always_drew():
    state = brush.StrokeState(
        layer_uid=1, size=SIZE, before=np.zeros((SIZE[1], SIZE[0], 4), np.uint8),
        colour=RED,
    )
    assert state.scatter == 0.0 and state.seed == 0
