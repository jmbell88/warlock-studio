"""The three-way comparison, and only that.

``sheetmerge`` is pure and imports nothing outward, so this file needs no
document, no window and no worker -- which is the point of the module being
separate from the op that uses it.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import sheetmerge


def _cell(value: int = 0, size: int = 4) -> np.ndarray:
    cell = np.zeros((size, size, 4), dtype=np.uint8)
    cell[..., 3] = 255
    cell[..., 0] = value
    return cell


# -- the digest ---------------------------------------------------------------


def test_the_same_pixels_digest_the_same_and_different_ones_do_not():
    assert sheetmerge.cell_digest(_cell(7)) == sheetmerge.cell_digest(_cell(7))
    assert sheetmerge.cell_digest(_cell(7)) != sheetmerge.cell_digest(_cell(8))


def test_a_view_digests_as_its_own_contents_rather_than_its_base():
    """A cel's plane can be a view after a geometry op, and a view's ``tobytes``
    is its *base's* buffer -- so without the contiguity copy a cropped cell
    would digest as the thing it was cropped out of."""
    whole = _cell(3, size=8)
    view = whole[2:6, 2:6]
    assert not view.flags["C_CONTIGUOUS"]
    assert sheetmerge.cell_digest(view) == sheetmerge.cell_digest(np.array(view))
    assert sheetmerge.cell_digest(view) != sheetmerge.cell_digest(whole)


def test_the_shape_is_part_of_the_digest():
    """A canvas resized between the import and the merge must read as changed,
    not as a coincidence. Same bytes, different shape, different answer."""
    flat = np.zeros((4, 8, 4), dtype=np.uint8)
    tall = np.zeros((8, 4, 4), dtype=np.uint8)
    assert flat.tobytes() == tall.tobytes()
    assert sheetmerge.cell_digest(flat) != sheetmerge.cell_digest(tall)


# -- the table ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "current", "incoming", "verdict"),
    [
        # Nobody touched it and the renderer produced the same thing.
        ("a", "a", "a", "agreed"),
        # Untouched, and the render moved: this is the whole point of merging.
        ("a", "a", "b", "take"),
        # Painted, and the render did not move: the edit stands.
        ("a", "b", "a", "keep"),
        # Both moved, differently. The one case a person has to look at.
        ("a", "b", "c", "conflict"),
        # Both moved to the *same* place -- the user painted what the renderer
        # has now caught up to. Asking anyone to arbitrate identical pictures
        # would be a conflict in name only.
        ("a", "b", "b", "agreed"),
    ],
)
def test_the_classification_table(base, current, incoming, verdict):
    assert sheetmerge.classify(base, current, incoming) == verdict


@pytest.mark.parametrize("base", [None, ""])
def test_no_base_is_unknown_rather_than_a_guess(base):
    """A document written before this feature has no block, so there is no
    third point to triangulate from. Two-way is not three-way and must not be
    read as if it were: the op refuses rather than overwriting on a coin flip."""
    assert sheetmerge.classify(base, "b", "c") == "unknown"
    assert sheetmerge.classify(base, "b", "b") == "unknown"


def test_every_verdict_the_table_can_produce_is_declared():
    produced = {
        sheetmerge.classify(base, current, incoming)
        for base in (None, "a")
        for current in ("a", "b")
        for incoming in ("a", "b", "c")
    }
    assert produced <= set(sheetmerge.VERDICTS)
    assert produced == set(sheetmerge.VERDICTS)


# -- what the toast says ------------------------------------------------------


def test_a_clean_merge_says_only_what_happened():
    counts = sheetmerge.MergeCounts(taken=48, agreed=208)
    assert sheetmerge.counts_sentence(counts) == "Took 48 cells."


def test_a_conflict_is_never_hidden_in_a_total():
    counts = sheetmerge.MergeCounts(taken=48, kept=3, conflicts=2)
    sentence = sheetmerge.counts_sentence(counts)
    assert sentence == "Took 48 cells, kept 3 edits, flagged 2 conflicts."


def test_singulars_read_as_english():
    counts = sheetmerge.MergeCounts(taken=1, kept=1, conflicts=1)
    assert sheetmerge.counts_sentence(counts) == "Took 1 cell, kept 1 edit, flagged 1 conflict."


def test_a_merge_that_changed_nothing_says_so_rather_than_nothing():
    assert "Nothing to merge" in sheetmerge.counts_sentence(sheetmerge.MergeCounts(agreed=256))


def test_the_counts_report_whether_anything_was_written():
    assert sheetmerge.MergeCounts(taken=1).wrote is True
    assert sheetmerge.MergeCounts(kept=9, conflicts=3).wrote is False
    assert sheetmerge.MergeCounts(taken=2, kept=1, agreed=3, conflicts=1).total == 7


# -- the payload round trip ---------------------------------------------------


def _base(uids, digests, conflicts=()):
    return sheetmerge.SheetBase(
        digests=dict(zip(uids, digests, strict=True)),
        conflicts=set(conflicts),
        source={"job": "abc", "sheet": "def"},
    )


def test_the_block_round_trips_through_indices():
    """Uids in memory, indices in the file. A uid is minted per process and
    means nothing in a file; an index means nothing across a reorder."""
    uids = [11, 22, 33]
    base = _base(uids, ["a1", "b2", "c3"], conflicts=[22])
    payload = base.payload({uid: i for i, uid in enumerate(uids)})

    back = sheetmerge.base_from_payload(payload, uids)
    assert back is not None
    assert back.digests == base.digests
    assert back.conflicts == base.conflicts
    assert back.source == base.source


def test_the_block_is_written_in_a_deterministic_order():
    """``.ora`` writing is pinned byte-identical, and a set's iteration order is
    not a promise."""
    uids = [30, 10, 20]
    base = _base(uids, ["c", "a", "b"], conflicts=[20, 10])
    payload = base.payload({uid: i for i, uid in enumerate(uids)})

    assert [cell["frame"] for cell in payload["cells"]] == [0, 1, 2]
    assert payload["conflicts"] == sorted(payload["conflicts"])


def test_an_empty_base_writes_no_block_at_all():
    """Additive and written only when set -- ``groups``' rule, which is what
    keeps an ordinary document's ``animation.json`` byte-identical."""
    assert sheetmerge.SheetBase().payload({}) is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "not a mapping",
        {"algorithm": "md5", "cells": [{"frame": 0, "digest": "a"}]},
        {"algorithm": sheetmerge.DIGEST_ALGORITHM, "cells": "not a list"},
        {"algorithm": sheetmerge.DIGEST_ALGORITHM, "cells": [{"frame": 9, "digest": "a"}]},
        {"algorithm": sheetmerge.DIGEST_ALGORITHM, "cells": [{"frame": 0}]},
        {"algorithm": sheetmerge.DIGEST_ALGORITHM, "cells": []},
    ],
)
def test_a_block_we_cannot_use_leaves_no_base_rather_than_raising(payload):
    """A base digest is metadata *about* a picture that is already fully and
    correctly built, so a bad block must not cost the user the document.

    An unknown algorithm is in this class deliberately: digests we cannot
    recompute are worse than none, because every cell would classify as edited
    and the merge would refuse to take anything."""
    assert sheetmerge.base_from_payload(payload, [11, 22]) is None


def test_a_conflict_index_off_the_end_is_dropped_and_the_rest_survives():
    payload = {
        "algorithm": sheetmerge.DIGEST_ALGORITHM,
        "cells": [{"frame": 0, "digest": "a"}],
        "conflicts": [0, 99],
    }
    back = sheetmerge.base_from_payload(payload, [11, 22])
    assert back is not None
    assert back.conflicts == {11}


def test_a_copy_does_not_alias_the_live_base():
    """An undo step holds one of these, and a step whose snapshot moves with
    the document restores nothing."""
    base = _base([11], ["a"], conflicts=[11])
    snapshot = base.copy()
    base.digests[11] = "changed"
    base.conflicts.clear()

    assert snapshot.digests == {11: "a"}
    assert snapshot.conflicts == {11}
