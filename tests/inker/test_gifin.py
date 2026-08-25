"""Reading a GIF back in as a document.

Inker could write a GIF and not open one, which made the export a one-way door:
the file a user shared was a file this editor refused. The round trip is the
test that matters, because both halves are here -- what ``gifout`` writes is
exactly what this has to be able to read.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import gifin, gifout

Image = pytest.importorskip("PIL.Image")


def _plane(size, colour, *, box=None):
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box or (0, 0, size[0], size[1])
    out[y0:y1, x0:x1] = colour
    return out


RED = (220, 30, 30, 255)
BLUE = (30, 30, 220, 255)
GREEN = (30, 220, 30, 255)


def _clip(tmp_path, durations=(100, 200, 300)):
    dest = tmp_path / "clip.gif"
    frames = [
        _plane((8, 6), RED, box=(0, 0, 4, 6)),
        _plane((8, 6), BLUE, box=(4, 0, 8, 6)),
        _plane((8, 6), GREEN, box=(2, 0, 6, 6)),
    ][: len(durations)]
    gifout.write_gif(dest, frames, list(durations))
    return dest


def test_a_written_clip_reads_back_with_its_frames_and_size(tmp_path):
    doc = gifin.read_gif(_clip(tmp_path))
    assert doc.size == (8, 6)
    assert doc.anim is not None
    assert len(doc.anim.frames) == 3


def test_the_durations_survive_the_round_trip(tmp_path):
    doc = gifin.read_gif(_clip(tmp_path, (100, 200, 300)))
    assert [f.duration_ms for f in doc.anim.frames] == [100, 200, 300]


def test_each_frame_keeps_its_own_picture(tmp_path):
    doc = gifin.read_gif(_clip(tmp_path))
    cels = list(doc.anim.cels.values())
    assert tuple(int(v) for v in cels[0].pixels[1, 1])[:3] == RED[:3]
    assert tuple(int(v) for v in cels[1].pixels[1, 5])[:3] == BLUE[:3]


def test_a_one_frame_gif_is_a_still_document(tmp_path):
    """One frame is a picture, not a clip: a timeline with one row in it would
    be a claim about the file that the file does not make."""
    dest = tmp_path / "one.gif"
    gifout.write_gif(dest, [_plane((8, 6), RED)], [100])
    doc = gifin.read_gif(dest)
    assert doc.anim is None
    assert doc.size == (8, 6)


def test_document_load_routes_a_gif_here(tmp_path):
    doc = inker.Document.load(_clip(tmp_path))
    assert doc.anim is not None and len(doc.anim.frames) == 3
    assert doc.path == _clip(tmp_path)


def test_a_gif_is_an_image_the_studio_will_open():
    from warlock.studio import filetypes, inker_mode

    assert ".gif" in filetypes.IMAGE_SUFFIXES
    assert ".gif" in inker_mode.OPENABLE
