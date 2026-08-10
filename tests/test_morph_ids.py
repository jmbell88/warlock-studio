"""The morph op table against the C header that owns its numbers.

``_MORPH_OPS`` maps the two combine functions onto the kernel's operation ids,
and native/morph.c's MORPH_MAX / MORPH_MIN are the authority on what the
numbers mean -- the composite._MODE_IDS arrangement. The pair is the smallest
table that can be swapped without anything raising: a max handed to the kernel
as a min still returns a well-formed mask, just the opposite operation, so the
tests here are the ones that would catch exactly that -- the header parse pins
the numbers, and the asymmetric mask tells a dilate from an erode by result.

The parity half runs only where the DLL exists; everything else runs both ways
(WARLOCK_NATIVE=0 included), because the table is consulted before the kernel
is and must answer the same on a machine with no compiler.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from warlock import native
from warlock.studio.inker import selection as sel

MORPH_C = Path(__file__).resolve().parents[1] / "native" / "morph.c"

needs_dll = pytest.mark.skipif(not native.available(), reason="warlockc.dll not built")


def _header_ids() -> dict[str, int]:
    text = MORPH_C.read_text(encoding="utf-8")
    found = dict(re.findall(r"#define\s+(MORPH_MAX|MORPH_MIN)\s+(\d+)", text))
    assert set(found) == {"MORPH_MAX", "MORPH_MIN"}, "morph.c stopped defining its ops"
    return {name: int(value) for name, value in found.items()}


def test_the_table_spells_the_c_headers_numbers():
    """The C source owns the numbers; the Python table restates them. A swap
    here is invisible to every type check and turns grow into shrink."""
    ids = _header_ids()
    assert sel._MORPH_OPS[np.maximum] == ids["MORPH_MAX"]
    assert sel._MORPH_OPS[np.minimum] == ids["MORPH_MIN"]


def test_the_table_covers_exactly_the_supported_combines():
    """Two entries, keyed by the two functions the callers actually pass.
    Anything else must miss the table and fall back to numpy -- growing the
    kernel's vocabulary means editing morph.c first."""
    assert set(sel._MORPH_OPS) == {np.maximum, np.minimum}
    assert sorted(sel._MORPH_OPS.values()) == [0, 1]


def _lone_pixel(shape=(9, 9)) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    mask[4, 4] = 255
    return mask


def test_the_reference_tells_the_two_ops_apart_on_a_lone_pixel(monkeypatch):
    """The discriminating input, proven discriminating on the numpy body
    first: dilating a lone pixel spreads it, eroding one erases it. A test
    corpus where the two ops gave equal answers could never catch a swap."""
    monkeypatch.setattr(native, "available", lambda: False)
    mask = _lone_pixel()
    grown = sel._morph(mask, 1, np.maximum)
    shrunk = sel._morph(mask, 1, np.minimum)
    assert grown.sum() > mask.sum()
    assert shrunk.sum() == 0
    assert not np.array_equal(grown, shrunk)


@needs_dll
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_the_kernel_agrees_with_the_reference_in_both_directions(radius, monkeypatch):
    """Grow and shrink through the kernel against the numpy body, on the mask
    the test above proved asymmetric -- so a swapped pair fails loudly here
    rather than shipping erode under the grow button."""
    mask = _lone_pixel()
    for combine in (np.maximum, np.minimum):
        monkeypatch.setattr(native, "available", lambda: True)
        fast = sel._morph(mask, radius, combine)
        monkeypatch.setattr(native, "available", lambda: False)
        slow = sel._morph(mask, radius, combine)
        assert np.array_equal(fast, slow), combine.__name__


def test_a_combine_outside_the_table_falls_back_to_numpy(monkeypatch):
    """The written-out table's other half: an unlisted operation is answered
    by the reference, never rendered as whichever of max/min it resembles."""
    monkeypatch.setattr(native, "available", lambda: True)
    called: list[int] = []
    monkeypatch.setattr(native, "morph_u8", lambda *a: called.append(1))
    assert sel._morph_native(_lone_pixel(), 1, np.add) is None
    assert called == []
