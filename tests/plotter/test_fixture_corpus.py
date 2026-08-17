"""Real Tiled files, read both ways and written back.

The corpus is the half of the compatibility contract that we did not write.
Everything else in ``tests/plotter/`` builds a document in Python and asserts
our reader agrees with our writer, which cannot catch the case that matters:
Tiled spelling something in a way we never thought to emit. These files are
authored in Tiled 1.12.2 itself -- ``fixtures/tiled/FIXTURES.md`` records the
build and the steps -- and the gates below are deliberately narrow:

- the two readers agree, so ``.tmx`` and ``.tmj`` are one map read twice;
- export and re-read is semantically identical, so nothing is lost in the
  round trip even when the bytes differ.

Byte-level determinism is *not* asserted here. Our writer is not trying to
reproduce Tiled's bytes; it is trying to preserve Tiled's document. The
byte-identity rule applies to our own output only, and lives in
``test_tmx.py`` and ``test_wmap.py`` where it always has.
"""

from __future__ import annotations

import pytest

from warlock.studio.plotter import tmx

from ._corpus import FIXTURE_DIR, MANIFEST, loaders_for, pairs
from ._semantics import doc_facts


def test_the_fixture_directory_and_its_recipe_exist():
    """The recipe is what makes a fixture reproducible. A corpus nobody can
    regenerate is a corpus that rots the first time Tiled changes."""
    assert FIXTURE_DIR.is_dir()
    assert (FIXTURE_DIR / "FIXTURES.md").is_file()


def test_every_required_fixture_is_present():
    """``MANIFEST`` is the shopping list and ``pairs()`` is what is on the
    shelf. Comparing them is what stops this file passing over an empty
    directory -- a corpus test that only iterates what it finds is not a gate.

    Checked both ways: a stem ``pairs()`` has that ``MANIFEST`` does not is
    just as much a defect as the reverse. ``FIXTURES.md`` says it plainly --
    "a file in this directory that nothing lists is a file nothing tests" --
    and a one-directional check would let seven fixtures ship with six listed
    and never notice the seventh was going untested.
    """
    on_shelf = pairs()
    missing = [stem for stem in MANIFEST if stem not in on_shelf]
    assert not missing, f"missing fixture pairs: {missing}"
    unlisted = [stem for stem in on_shelf if stem not in MANIFEST]
    assert not unlisted, f"fixture pairs present but not in MANIFEST: {unlisted}"


@pytest.mark.parametrize("stem", MANIFEST)
def test_both_readers_see_the_same_map(stem):
    """The strongest cheap statement about the two readers: they are one
    reader written twice, not two that happen to agree today."""
    loaders = loaders_for(FIXTURE_DIR)
    from_xml = tmx.read_tmx((FIXTURE_DIR / f"{stem}.tmx").read_bytes(), **loaders)
    from_json = tmx.read_tmj((FIXTURE_DIR / f"{stem}.tmj").read_bytes(), **loaders)
    assert doc_facts(from_xml) == doc_facts(from_json)


@pytest.mark.parametrize("stem", MANIFEST)
def test_a_tiled_map_survives_our_own_round_trip(stem):
    """Read Tiled's file, write ours, read ours back. Semantic identity, not
    byte identity: our writer emits CSV and external tilesets whatever the
    input did, and that is a choice rather than a loss."""
    loaders = loaders_for(FIXTURE_DIR)
    original = tmx.read_tmx((FIXTURE_DIR / f"{stem}.tmx").read_bytes(), **loaders)
    files = tmx.tmx_export(original)
    again = tmx.read_tmx(files["map.tmx"], **loaders_for(FIXTURE_DIR, extra=files))
    assert doc_facts(again) == doc_facts(original)


@pytest.mark.parametrize("stem", MANIFEST)
def test_the_json_writer_agrees_with_the_xml_writer(stem):
    """Both exporters describe the same document, so a map exported as JSON
    and read back is the map exported as XML and read back."""
    loaders = loaders_for(FIXTURE_DIR)
    original = tmx.read_tmx((FIXTURE_DIR / f"{stem}.tmx").read_bytes(), **loaders)
    files = tmx.tmj_export(original)
    again = tmx.read_tmj(files["map.tmj"], **loaders_for(FIXTURE_DIR, extra=files))
    assert doc_facts(again) == doc_facts(original)


@pytest.mark.parametrize("stem", MANIFEST)
def test_a_tiled_map_survives_our_own_save_format(stem):
    """The third round trip, and the one the studio actually uses: a Tiled
    file opened, saved as ``.wmap``, and reopened. ``doc_facts`` is uid-free
    by construction, which is what makes it the right comparator here --
    ``.wmap`` stores indices and mints fresh uids on read, so a comparator
    that saw uids would fail this on every document."""
    from warlock.studio.plotter import wmap

    loaders = loaders_for(FIXTURE_DIR)
    original = tmx.read_tmx((FIXTURE_DIR / f"{stem}.tmx").read_bytes(), **loaders)
    again = wmap.read_wmap(wmap.wmap_bytes(original))
    assert doc_facts(again) == doc_facts(original)


@pytest.mark.parametrize("stem", MANIFEST)
def test_two_exports_of_a_fixture_are_byte_identical(stem):
    """The determinism rule, applied to the corpus rather than to a synthetic
    document, and to every fixture rather than only the first -- a writer
    that is deterministic on one map and not another is a writer with a
    conditional in it somewhere, and the first fixture alone cannot find it.

    ``MANIFEST`` now names four stems, so this collects four cases. It said the
    corpus was empty, which was true of the milestone that wrote it and has not
    been since. What the corpus proves is bounded in a different way now, and
    the bound is documented rather than in a docstring here: every fixture is
    *synthesized* by this editor rather than authored in Tiled, so a green run
    is a statement about our own encoder's determinism and not about Tiled.
    See ``fixtures/tiled/FIXTURES.md``."""
    loaders = loaders_for(FIXTURE_DIR)
    doc = tmx.read_tmx((FIXTURE_DIR / f"{stem}.tmx").read_bytes(), **loaders)
    assert tmx.tmx_export(doc) == tmx.tmx_export(doc)
