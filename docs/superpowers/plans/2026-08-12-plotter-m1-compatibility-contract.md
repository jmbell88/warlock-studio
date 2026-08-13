# Plotter M1 — Compatibility Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Plotter↔Tiled boundary stated and machine-checked *before* it moves — a semantic comparator, a corpus of real Tiled 1.12.2 fixtures, a published feature matrix cross-checked against the source's own refusal strings, and the version-metadata story settled.

**Architecture:** Three additions, all test-side or docs-side except one constant. `tests/plotter/_semantics.py` gives `doc_facts(doc)` — everything a map *is*, minus process uids and byte encodings — which every later milestone gates on. `tests/plotter/_corpus.py` + `tests/plotter/fixtures/tiled/` load real Tiled files through the engine's host-supplied loader callbacks (the engine stays pure). `docs/PLOTTER_COMPAT.md` is the published matrix, and `test_compat_matrix.py` fails when a `TiledUnsupported` string in `src/` has no row, or a row names a fixture that does not exist. **No behaviour change to what loads or refuses.**

**Tech Stack:** Python 3.13, pytest, numpy, `xml.etree.ElementTree`, `ast` (for the source scan), hashlib. No new dependencies.

**Spec:** `D:\Projects\Warlock\PLOTTER_PLAN.md` § "Milestone 1 — Compatibility contract". Program context, sequencing and the per-milestone ritual live there; this plan implements that section only.

## Global Constraints

- Tiled compatibility target is **1.12.2**, pinned until a deliberate bump.
- The engine package `src/warlock/studio/plotter/` imports no imgui/moderngl/pygame/`service`, and no Pillow at module scope. **M1 adds no engine module** — everything new is under `tests/` or `docs/` except one constant in `tsx.py`. So `tests/plotter/test_plotter_imports.py:131` needs no new entry this milestone.
- Fixtures are authored in real Tiled 1.12.2 by the user. Never synthesize a file and check it in as a Tiled golden — the corpus exists precisely to be something we did not write.
- Tests supply loader callbacks against the fixture directory; the engine never reads a file itself.
- Never edit `src/` while pytest runs (several tests read module source).
- Full suite: `uv run pytest`. Install: `uv sync --extra studio --extra text2image --extra rig`.
- Commit convention: `Warlock v0.0.21` as the subject (current version; do **not** bump unless the user asks).
- Baseline at plan time: `b7fd3fc`, suite **6705 passed, 17 skipped, 18 deselected**.

## File Structure

| File | Responsibility |
|---|---|
| `tests/plotter/_semantics.py` (create) | `doc_facts(doc)` — the uid-free semantic fingerprint of a `MapDoc`. The substrate M2–M8 extend. Not a test module; a helper imported by them. |
| `tests/plotter/test_semantics.py` (create) | Proves the comparator's own properties: uid-blind, float-tolerant, and sensitive to every field it claims to cover. |
| `tests/plotter/_corpus.py` (create) | Fixture-directory plumbing: `loaders_for(dir)` returning the `image_loader`/`tsx_loader` pair, `pairs()` discovering `<name>.tmx`/`<name>.tmj` siblings. |
| `tests/plotter/fixtures/tiled/FIXTURES.md` (create) | The authoring recipe: Tiled build, per-fixture steps, and the manifest the gate reads. |
| `tests/plotter/test_fixture_corpus.py` (create) | The corpus gates: both readers agree, and export→re-read is semantically identical. |
| `docs/PLOTTER_COMPAT.md` (create) | The published feature matrix. One row per Tiled feature, exactly one state each. |
| `tests/plotter/test_compat_matrix.py` (create) | The ledger gate: matrix rows ↔ `TiledUnsupported` strings in `src/`, both directions. |
| `src/warlock/studio/plotter/tsx.py:42-43` (modify) | `TILED_VERSION` → `"1.12.2"`; `TSX_VERSION` stays `"1.10"` with the bump rule stated in a comment. |
| `docs/manual/09-plotter.md` (modify) | Cross-link the refusal section to the matrix. |

**Task order rationale.** Tasks 1–3 are fully self-contained and land first, so the harness and the gate exist before the fixtures do. Task 4 is **user-gated** — it wires the corpus once you have authored the files. Task 5 is **verification-gated** — the version bump only happens after Tiled 1.12.2 has actually opened one of our exports, which is the honest order the spec insists on.

---

### Task 1: The semantic comparator

**Files:**
- Create: `tests/plotter/_semantics.py`
- Test: `tests/plotter/test_semantics.py`

**Interfaces:**
- Consumes: `warlock.studio.plotter.tilemap.MapDoc`, `_map_model.TileLayer`/`ObjectLayer`/`MapObject`, `tileset.TilesetRef`/`Tileset`/`TerrainSpec`, `tsx.Prop`.
- Produces: `doc_facts(doc: MapDoc) -> dict[str, Any]` — a plain, JSON-shaped, uid-free dict. Every later task and milestone compares documents **only** through this function.

**Why a hash for pixels and cells:** a fixture atlas is 4 KiB of ints and a layer is a few hundred; inlining them turns a failing assert into an unreadable wall. Shape travels beside the digest so a size mismatch still reads as a size mismatch.

- [ ] **Step 1: Write the failing test**

Create `tests/plotter/test_semantics.py`:

```python
"""The comparator's own properties, tested before anything gates on it.

``doc_facts`` is the substrate every later milestone compares documents
through, so the thing that must be true of it is not "it works on this map"
but that it is *blind to what it promises to be blind to* (uids, dict order,
float spelling) and *sensitive to everything else*. A comparator that quietly
ignored a field would turn every gate built on it into a test that passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter import gid, tsx
from warlock.studio.plotter.tilemap import MapDoc, MapObject, new_uid
from warlock.studio.plotter.tileset import Tileset

from ._semantics import doc_facts


def _pixels(w: int = 64, h: int = 64) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    array[0, 0] = (7, 8, 9, 255)
    return array


def _doc() -> MapDoc:
    doc = MapDoc(6, 4, 16, 16)
    doc.add_tileset(Tileset(name="terrain", pixels=_pixels(), tile_w=16, tile_h=16))
    tiles = doc.add_tile_layer("Ground")
    cells = np.zeros((4, 6), gid.DTYPE)
    cells[0, 0] = gid.compose(1)
    cells[1, 1] = gid.compose(2, flip_h=True)
    doc.write_region(tiles.uid, 0, 0, cells)
    doc.set_layer_props(tiles.uid, opacity=0.5)
    objects = doc.add_object_layer("Things")
    doc.add_object(
        objects.uid,
        MapObject(uid=new_uid(), name="spawn", kind="point", x=17.5, y=3.0,
                  properties={"team": tsx.Prop("int", 2)}),
    )
    doc.properties = {"theme": tsx.Prop("string", "cave")}
    doc.backgroundcolor = "#ff112233"
    return doc


def test_two_documents_built_the_same_way_agree():
    assert doc_facts(_doc()) == doc_facts(_doc())


def test_the_facts_are_blind_to_uids():
    """The point of the whole function: a uid is minted per process and means
    nothing across a save, so two readings of one file must compare equal."""
    first, second = _doc(), _doc()
    uids = {layer.uid for layer in first.layers} | {layer.uid for layer in second.layers}
    assert len(uids) == 4, "the two documents really do carry different uids"
    assert doc_facts(first) == doc_facts(second)


def test_the_facts_survive_json_and_so_can_be_diffed():
    import json

    assert json.loads(json.dumps(doc_facts(_doc()))) == doc_facts(_doc())


def test_a_float_written_two_ways_compares_equal():
    """0.1 + 0.2 is not 0.3, and a document that went through a text format
    must not fail the gate over the last bit of a coordinate."""
    left, right = _doc(), _doc()
    obj = right.layers[1].objects[0]
    obj.x = 17.5 + 1e-12
    assert doc_facts(left) == doc_facts(right)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: setattr(d, "projection", "isometric"), id="projection"),
        pytest.param(lambda d: setattr(d, "renderorder", "left-up"), id="renderorder"),
        pytest.param(lambda d: setattr(d, "backgroundcolor", "#ffffffff"), id="background"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, name="Other"), id="layer-name"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, opacity=0.25), id="opacity"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, visible=False), id="visible"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, locked=True), id="locked"),
        pytest.param(
            lambda d: d.write_region(d.layers[0].uid, 0, 0, np.full((1, 1), gid.compose(9), gid.DTYPE)),
            id="cells",
        ),
        pytest.param(
            lambda d: setattr(d.layers[1].objects[0], "name", "other"), id="object-name"
        ),
        pytest.param(lambda d: setattr(d.layers[1].objects[0], "x", 99.0), id="object-x"),
        pytest.param(
            lambda d: setattr(d.layers[1].objects[0], "obj_class", "Spawn"), id="object-class"
        ),
        pytest.param(
            lambda d: d.__setattr__("properties", {"theme": tsx.Prop("string", "forest")}),
            id="map-properties",
        ),
    ],
)
def test_every_field_the_comparator_claims_to_cover_actually_moves_it(mutate):
    """One case per field. A comparator is only as good as its worst blind
    spot, and the blind spot is invisible until something that should have
    failed passes."""
    before = doc_facts(_doc())
    doc = _doc()
    mutate(doc)
    assert doc_facts(doc) != before


def test_a_different_atlas_moves_the_facts():
    """Pixels are hashed rather than inlined, so this is the test that the
    hash is of the pixels and not of, say, the shape alone."""
    doc = _doc()
    other = _pixels()
    other[5, 5] = (1, 2, 3, 255)
    swapped = MapDoc(6, 4, 16, 16)
    swapped.add_tileset(Tileset(name="terrain", pixels=other, tile_w=16, tile_h=16))
    assert doc_facts(swapped)["tilesets"] != doc_facts(doc)["tilesets"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plotter/test_semantics.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'tests.plotter._semantics'` (or `ImportError` on the relative import).

- [ ] **Step 3: Write the implementation**

Create `tests/plotter/_semantics.py`:

```python
"""What a map *is*, minus everything that is not the document.

The comparator every Plotter↔Tiled gate is built on. Two documents are the
same map when these facts match, and the three things deliberately absent are
the three that are not properties of the document:

- **uids**, minted per process and meaningless across a save;
- **byte encodings**, so a CSV layer and a zlib layer compare equal -- which
  is the entire point of reading the same fixture twice, once as ``.tmx`` and
  once as ``.tmj``;
- **float spelling**, because a coordinate that went out through a text format
  and came back must not fail a gate over its last bit.

Everything else is in, and the test module beside this one has one case per
field to keep it that way: a comparator with a blind spot turns every gate
built on it into a test that cannot fail.

In ``tests/`` rather than in the engine because it is a *test* vocabulary --
the engine has no reason to be able to fingerprint itself, and putting it in
``plotter/`` would add a module to a package whose import set is pinned.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Coordinates and opacities are compared to this many decimals. Six is well
# inside what a Tiled file writes and well outside float noise.
_PLACES = 6


def _num(value: Any) -> float:
    """A float as the comparator sees it: rounded, and never negative zero."""
    return round(float(value), _PLACES) + 0.0


def _digest(array: np.ndarray) -> str:
    """A short, stable fingerprint of an array's bytes.

    Hashed rather than inlined because an atlas is thousands of ints and a
    failing assert has to stay readable. ``shape`` travels beside every call
    site so a size mismatch still reports as a size mismatch.
    """
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def _prop(value: Any) -> Any:
    """One custom property. Handles both spellings the codebase carries.

    ``properties`` is typed ``dict[str, Any]`` and holds ``tsx.Prop`` in
    practice; a plain value is accepted rather than refused so this never
    becomes the reason a test cannot express a document.
    """
    kind = getattr(value, "type", None)
    if kind is None:
        return ["untyped", value if not isinstance(value, float) else _num(value)]
    raw = getattr(value, "value")
    return [kind, _num(raw) if kind == "float" else raw]


def _props(properties: dict[str, Any]) -> dict[str, Any]:
    return {name: _prop(properties[name]) for name in sorted(properties)}


def _terrain_facts(spec: Any) -> list[Any]:
    return [spec.name, list(spec.fill), list(spec.outline)]


def _tileset_facts(ref: Any) -> dict[str, Any]:
    tileset = ref.tileset
    return {
        "firstgid": int(ref.firstgid),
        "source": ref.source,
        "name": tileset.name,
        "tile_w": int(tileset.tile_w),
        "tile_h": int(tileset.tile_h),
        "spacing": int(tileset.spacing),
        "margin": int(tileset.margin),
        "columns": int(tileset.columns),
        "rows": int(tileset.rows),
        "image_shape": list(np.asarray(tileset.pixels).shape),
        "image": _digest(tileset.pixels),
        "properties": _props(tileset.properties),
        "terrains": [_terrain_facts(spec) for spec in tileset.terrains],
    }


def _object_facts(obj: Any) -> dict[str, Any]:
    return {
        "name": obj.name,
        "kind": obj.kind,
        "x": _num(obj.x),
        "y": _num(obj.y),
        "w": _num(obj.w),
        "h": _num(obj.h),
        "obj_class": obj.obj_class,
        "visible": bool(obj.visible),
        "properties": _props(obj.properties),
    }


def _layer_facts(layer: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "name": layer.name,
        "visible": bool(layer.visible),
        "opacity": _num(layer.opacity),
        "locked": bool(layer.locked),
        "properties": _props(layer.properties),
    }
    objects = getattr(layer, "objects", None)
    if objects is None:
        facts["type"] = "tile"
        facts["shape"] = [int(layer.height), int(layer.width)]
        facts["cells"] = _digest(layer.data)
    else:
        facts["type"] = "object"
        facts["objects"] = [_object_facts(obj) for obj in objects]
    return facts


def doc_facts(doc: Any) -> dict[str, Any]:
    """Everything a :class:`MapDoc` is, as a JSON-shaped dict.

    Two documents compare equal exactly when they are the same map. Layer and
    tileset *order* is significant and preserved -- paint order and firstgid
    allocation are both facts about the document -- while property order is
    not, and is sorted away.
    """
    return {
        "projection": doc.projection,
        "width": int(doc.width),
        "height": int(doc.height),
        "tile_w": int(doc.tile_w),
        "tile_h": int(doc.tile_h),
        "renderorder": doc.renderorder,
        "backgroundcolor": doc.backgroundcolor,
        "properties": _props(doc.properties),
        "tilesets": [_tileset_facts(ref) for ref in doc.tilesets],
        "layers": [_layer_facts(layer) for layer in doc.layers],
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/plotter/test_semantics.py -v`
Expected: PASS, every case.

If `test_a_different_atlas_moves_the_facts` fails, the digest is being taken of the wrong thing. If a `test_every_field...` case fails, that field is missing from `doc_facts` — add it rather than deleting the case.

- [ ] **Step 5: Confirm nothing else moved**

Run: `uv run pytest tests/plotter -q`
Expected: all existing plotter tests still pass; the new module adds tests and changes none.

- [ ] **Step 6: Commit**

```bash
git add tests/plotter/_semantics.py tests/plotter/test_semantics.py
git commit -m "Warlock v0.0.21

Add doc_facts, the semantic comparator Plotter's Tiled gates are built on.
"
```

---

### Task 2: The corpus harness and the authoring recipe

**Files:**
- Create: `tests/plotter/_corpus.py`
- Create: `tests/plotter/fixtures/tiled/FIXTURES.md`
- Test: `tests/plotter/test_fixture_corpus.py`

**Interfaces:**
- Consumes: `doc_facts` from Task 1; `tmx.read_tmx`, `tmx.read_tmj`, `tmx.tmx_export`, `tmj_export` (all `-> dict[str, bytes]`, keyed on relative path, the map itself under `map.tmx` / `map.tmj`).
- Produces:
  - `FIXTURE_DIR: pathlib.Path` — `tests/plotter/fixtures/tiled/`.
  - `loaders_for(directory: Path) -> dict[str, Callable]` — the `{"image_loader": ..., "tsx_loader": ...}` pair `read_tmx`/`read_tmj` take as keyword arguments, resolving relative to `directory`.
  - `pairs() -> list[str]` — stems having both a `.tmx` and a `.tmj` in the fixture dir, sorted.
  - `MANIFEST: tuple[str, ...]` — the stems the gate *requires*. **Lands empty in this task** and is filled in Task 4, so the suite is green before the fixtures exist and the gate becomes real the moment they do.

**Why `MANIFEST` is separate from `pairs()`:** a corpus test that only iterates what it finds passes triumphantly over an empty directory. The manifest is the shopping list; discovery is what is actually there; the gate is that they match. Landing it empty is honest — it says "nothing is required yet" — and Task 4 is the commit where it starts meaning something.

- [ ] **Step 1: Write the failing test**

Create `tests/plotter/test_fixture_corpus.py`:

```python
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
    directory -- a corpus test that only iterates what it finds is not a gate."""
    missing = [stem for stem in MANIFEST if stem not in pairs()]
    assert not missing, f"missing fixture pairs: {missing}"


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


def test_two_exports_of_a_fixture_are_byte_identical():
    """The determinism rule, applied to the corpus rather than to a synthetic
    document. Skipped rather than failed while the corpus is empty, because
    'no fixtures yet' is a state this milestone passes through on purpose."""
    if not MANIFEST:
        pytest.skip("no fixtures authored yet")
    loaders = loaders_for(FIXTURE_DIR)
    doc = tmx.read_tmx((FIXTURE_DIR / f"{MANIFEST[0]}.tmx").read_bytes(), **loaders)
    assert tmx.tmx_export(doc) == tmx.tmx_export(doc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plotter/test_fixture_corpus.py -v`
Expected: collection error — no module `tests.plotter._corpus`.

- [ ] **Step 3: Create the fixture directory and the recipe**

Create `tests/plotter/fixtures/tiled/FIXTURES.md`:

```markdown
# The Tiled fixture corpus

Files in this directory are authored in **Tiled 1.12.2 itself** and checked in
as goldens. That is the whole point of them: every other test in
`tests/plotter/` builds a document in Python and asserts our reader agrees
with our writer, which cannot catch Tiled spelling something in a way we never
thought to emit.

**Never synthesize a file here.** A fixture written by our own exporter and
checked in as a golden records our assumptions and calls them Tiled's.

## The build

- Tiled **1.12.2**, official download, default settings.
- Save each map twice: once as `.tmx` (File → Save As) and once as `.tmj`
  (File → Export As → JSON map file). Both go in this directory, same stem.
- Tilesets are **external** (`.tsx` + `.png` in this directory), which is what
  our exporter writes and what the loaders below resolve.
- Keep atlases tiny — 2×2 tiles at 16×16 is plenty. These are checked into git
  and read on every suite run.
- In Preferences → General, leave "Export files as read-only" off.

## The manifest

`_corpus.py` carries `MANIFEST`, the stems the gate requires. Adding a fixture
means adding its stem there in the same commit — a file in this directory that
nothing lists is a file nothing tests.

## The fixtures

Each entry is one map, saved as both `.tmx` and `.tmj`.

### `basic-ortho`
Orthogonal, 8×8, 16×16 tiles. One tileset (`basic.tsx`, a 2×2 atlas of 16×16
tiles). Two tile layers, `Ground` and `Detail`; paint a handful of tiles on
each, and on `Detail` include at least one horizontally flipped tile, one
vertically flipped, and one diagonally flipped (the `X`/`Y`/`Z` keys while
stamping). Set `Detail`'s opacity to 0.5. Flips are the reason this fixture
exists: they live in the top three bits of every cell and a lost bit is
invisible until the map is in an engine.

### `basic-iso`
Isometric, 8×8, 32×16 tiles, one tileset, one tile layer with a few tiles
painted. Isometric is the projection that left the refusal list, so it needs a
golden that is not ours.

### `two-tilesets`
Orthogonal, 6×6. Two external tilesets with different tile sizes (16×16 and
32×32). Paint from **both** onto one layer. This is the firstgid fixture: the
second set's ids start above the first's, and getting that wrong is silent.

### `objects-rect-point`
Orthogonal, 6×6. One tile layer and one object layer holding: a named
rectangle with a class set, a point, and a rectangle with `visible` unchecked.
Give one object a custom property. Do **not** add ellipses, polygons,
polylines, text, tile objects, or any rotation — those are refused today, and
they arrive as fixtures in M3 when they stop being.

### `typed-props`
Orthogonal, 4×4, one tile layer. Custom properties of every type Plotter
models today — `string`, `int`, `float`, `bool`, `color` — at all three
levels: on the map, on the layer, and on the tileset. Include a `float` with a
long decimal expansion (e.g. `0.30000000000000004`) — that value is why the
comparator rounds.

### `locked-layers`
Orthogonal, 4×4, two tile layers, one of them locked and one hidden. Lock and
visibility are document state, not view state, and this proves we read both.

### `blob-terrain`
Orthogonal, 8×8, one tileset carrying **one** Wang set shaped exactly as
Plotter's blob preset (47 tiles for one terrain colour). Paint a few cells
with the terrain brush. This is the fixture that pins the recognise-or-refuse
boundary at the shape we actually accept — a second colour or a differently
shaped set is an M4 fixture, not this one.
```

- [ ] **Step 4: Write the harness**

Create `tests/plotter/_corpus.py`:

```python
"""Loading the Tiled fixture corpus, without giving the engine a filesystem.

``read_tmx`` and ``read_tmj`` take an ``image_loader`` and a ``tsx_loader``
rather than opening anything themselves, which is what keeps the engine
package pure -- it never learns where a file lives. This module is the host
side of that arrangement for tests: it resolves a relative reference against
the fixture directory, and, when a test is re-reading our own export, against
an in-memory mapping of what that export produced.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from warlock.studio.plotter import tsx
from warlock.studio.plotter.tileset import Tileset

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiled"

# The fixture stems the corpus gate requires, each present as both a ``.tmx``
# and a ``.tmj``. Empty until the fixtures are authored in Tiled 1.12.2 --
# see ``fixtures/tiled/FIXTURES.md``. Adding a file means adding its stem
# here in the same commit, or nothing tests it.
MANIFEST: tuple[str, ...] = ()


def pairs() -> list[str]:
    """Stems in the fixture directory having both spellings, sorted."""
    if not FIXTURE_DIR.is_dir():
        return []
    stems = {path.stem for path in FIXTURE_DIR.glob("*.tmx")}
    return sorted(stems & {path.stem for path in FIXTURE_DIR.glob("*.tmj")})


def _read(source: str, directory: Path, extra: dict[str, bytes]) -> bytes:
    """One reference, resolved. ``extra`` wins, and is how a test re-reads an
    export that was never written to disk.

    Both the bare name and a ``tilesets/`` prefix are tried, because that is
    the layout ``tmx_export`` writes and a fixture authored in Tiled keeps its
    tilesets beside the map.
    """
    for key in (source, f"tilesets/{source}", Path(source).name):
        if key in extra:
            return extra[key]
    candidate = directory / source
    if not candidate.is_file():
        candidate = directory / Path(source).name
    return candidate.read_bytes()


def loaders_for(
    directory: Path = FIXTURE_DIR, *, extra: dict[str, bytes] | None = None
) -> dict[str, Callable[[str], Any]]:
    """The keyword pair both readers take.

    ``read_tsx`` takes a ``.tsx``'s bytes *and its decoded image*, not a
    loader -- so resolving the nested reference is this side's job:
    ``tsx.tsx_source`` reports the ``<image source=...>`` path the tileset
    names, and that gets decoded and handed in. ``tests/plotter/test_tmx.py``
    does the same thing against an in-memory export.

    Pillow is imported inside the loader rather than at module scope, matching
    the engine's own rule: nothing should pay for a PNG decoder by importing a
    test helper.
    """
    files = dict(extra or {})

    def image_loader(source: str) -> np.ndarray:
        from PIL import Image

        with Image.open(io.BytesIO(_read(source, directory, files))) as image:
            return np.asarray(image.convert("RGBA"), dtype=np.uint8)

    def tsx_loader(source: str) -> Tileset:
        raw = _read(source, directory, files)
        return tsx.read_tsx(raw, image_loader(tsx.tsx_source(raw)))

    return {"image_loader": image_loader, "tsx_loader": tsx_loader}
```

**Signatures this task depends on** (verified at plan time, `tsx.py:427`,
`wmap.py:224`/`:281`): `read_tsx(data: bytes, image: np.ndarray) -> Tileset`,
`tsx_source(data: bytes) -> str`, `wmap_bytes(doc) -> bytes`,
`read_wmap(data: bytes) -> MapDoc`. `tests/plotter/test_tmx.py:70-86` is the
working reference for the loader pair.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/plotter/test_fixture_corpus.py -v`
Expected: PASS — two real assertions (`test_the_fixture_directory_and_its_recipe_exist`, `test_every_required_fixture_is_present` over an empty manifest), one skip, and the three parametrized gates collecting zero cases because `MANIFEST` is empty. That is the intended state until Task 4.

- [ ] **Step 6: Commit**

```bash
git add tests/plotter/_corpus.py tests/plotter/test_fixture_corpus.py \
        tests/plotter/fixtures/tiled/FIXTURES.md
git commit -m "Warlock v0.0.21

Add the Tiled fixture corpus harness and its authoring recipe.

The manifest lands empty: the gate becomes real when the fixtures are
authored in Tiled 1.12.2, and until then says so rather than passing over
an empty directory.
"
```

---

### Task 3: The feature matrix and its ledger gate

**Files:**
- Create: `docs/PLOTTER_COMPAT.md`
- Test: `tests/plotter/test_compat_matrix.py`
- Modify: `docs/manual/09-plotter.md` (the refusal section, around lines 283–298 — cross-link only)

**Interfaces:**
- Consumes: nothing from Tasks 1–2. Fully independent; can be built in parallel.
- Produces: `docs/PLOTTER_COMPAT.md` as the published boundary, and the invariant that **every `TiledUnsupported` feature string in `src/warlock/studio/plotter/` appears as a `refused` row**. Later milestones move rows in the same commit that flips the refusal.

**The scan.** Refusal strings are the first argument of `raise TiledUnsupported(...)`. Some are plain literals (`"group layers"`); several are f-strings (`f"a {orientation} map"`, `f"{kind} layers"`). The scan walks the AST and renders both into one normal form: a literal becomes itself, an f-string becomes its text with every interpolation replaced by `{}`. The matrix row carries that exact normal form, so `a {} map` is one row covering staggered and hexagonal — which is honest, because it is one refusal.

- [ ] **Step 1: Write the failing test**

Create `tests/plotter/test_compat_matrix.py`:

```python
"""The refusal ledger, machine-checked against the code that does the refusing.

``docs/PLOTTER_COMPAT.md`` is a promise about what Plotter does with a Tiled
file, and a promise in a markdown table drifts the moment somebody adds a
refusal without opening it. So the table is not prose: the ``refused`` rows
are checked, both ways, against every ``TiledUnsupported`` raised in the
engine.

Both ways is the part that matters. A missing row means a refusal the docs do
not admit to; a stale row means a feature the docs still claim we refuse after
somebody taught us to load it -- and that second one is exactly what happens
during a parity milestone, which is when this file earns its keep.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

MATRIX = Path(__file__).resolve().parents[2] / "docs" / "PLOTTER_COMPAT.md"
ENGINE = Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio" / "plotter"

STATES = {"round-trips", "refused", "preserved-verbatim"}


def _normal_form(node: ast.expr) -> str | None:
    """A refusal's feature argument as one comparable string.

    A literal is itself; an f-string keeps its text and writes every
    interpolation as ``{}``, so ``f"a {orientation} map"`` is ``a {} map`` --
    one row for one refusal, rather than a row per value it can take.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append("{}")
        return "".join(out)
    return None


def _raised_features() -> set[str]:
    found: set[str] = set()
    for path in sorted(ENGINE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            call = node.exc
            if not isinstance(call, ast.Call):
                continue
            name = call.func.id if isinstance(call.func, ast.Name) else None
            if name != "TiledUnsupported" or not call.args:
                continue
            feature = _normal_form(call.args[0])
            assert feature is not None, (
                f"{path.name}: a refusal whose feature is neither a literal nor an "
                "f-string cannot be checked against the matrix -- make it one"
            )
            found.add(feature)
    return found


# Rows under this heading describe things Tiled itself does not have, so
# there is no refusal in the source to check them against and they are not
# part of the two-way ledger. Skipped by name rather than by a fourth state,
# because "we will never do this" is a different kind of statement from the
# three states and giving it one would blur them.
_UNCHECKED_SECTION = "Permanent non-goals"


def _rows() -> list[tuple[str, str, str]]:
    """``(feature, state, note)`` for every checked row of the matrix table."""
    rows = []
    section = ""
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        if section == _UNCHECKED_SECTION:
            continue
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        feature = cells[0].strip("`")
        rows.append((feature, cells[1], cells[2]))
    return rows


def test_the_matrix_exists_and_parses():
    assert MATRIX.is_file()
    assert _rows(), "the matrix has no rows"


def test_every_row_is_in_exactly_one_state():
    for feature, state, _ in _rows():
        assert state in STATES, f"{feature!r} is in unknown state {state!r}"


def test_no_feature_is_listed_twice():
    features = [feature for feature, _, _ in _rows()]
    assert len(features) == len(set(features)), "a feature appears in two rows"


def test_every_refusal_in_the_source_has_a_row():
    """The direction that catches a refusal added without a doc change."""
    refused = {feature for feature, state, _ in _rows() if state == "refused"}
    assert _raised_features() - refused == set()


def test_every_refused_row_still_refuses():
    """The direction that catches a row left behind by a milestone. When a
    feature starts loading, its row moves to ``round-trips`` in the same
    commit -- that is the ritual, and this is what enforces it."""
    refused = {feature for feature, state, _ in _rows() if state == "refused"}
    assert refused - _raised_features() == set()


def test_a_row_that_claims_to_round_trip_names_a_fixture_that_exists():
    """A ``round-trips`` claim is only worth what backs it. The note column
    carries the fixture stem in backticks; the file has to be there."""
    from ._corpus import FIXTURE_DIR

    for feature, state, note in _rows():
        if state != "round-trips":
            continue
        stems = re.findall(r"`([a-z0-9-]+)`", note)
        assert stems, f"{feature!r} claims to round-trip but names no fixture"
        for stem in stems:
            assert (FIXTURE_DIR / f"{stem}.tmx").is_file(), (
                f"{feature!r} names fixture {stem!r}, which is not in the corpus"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plotter/test_compat_matrix.py -v`
Expected: FAIL — `test_the_matrix_exists_and_parses` fails on the missing `docs/PLOTTER_COMPAT.md`.

- [ ] **Step 3: Generate the true refusal list**

Before writing the matrix by hand, print what the scan actually finds, so the
document is written from the source rather than from memory:

```bash
uv run python -c "import sys; sys.path.insert(0, 'tests'); from plotter.test_compat_matrix import _raised_features; [print(repr(f)) for f in sorted(_raised_features())]"
```

Use that exact output as the `refused` rows. If the import path fights you,
run the same thing via `uv run pytest tests/plotter/test_compat_matrix.py::test_every_refusal_in_the_source_has_a_row -v` against a stub matrix and read the assertion diff — either way, **the source is the authority, not this plan.**

- [ ] **Step 4: Write the matrix**

Create `docs/PLOTTER_COMPAT.md`. The rows below are the refusals present at
plan time; reconcile them against Step 3's output and add any the scan finds
that are not here.

```markdown
# Plotter ↔ Tiled compatibility

**Target: Tiled 1.12.2.** One row per feature, each in exactly one state:

- **round-trips** — read, modelled, and written back without loss. The note
  names the corpus fixture that proves it.
- **refused** — the reader stops by name and says what to remove. Never
  half-loaded: a feature silently dropped on read is a feature deleted on the
  next save, and the user finds out when the map is already gone.
- **preserved-verbatim** — carried through a round trip but not honoured by
  the editor. Written back exactly as it arrived.

This table is checked by `tests/plotter/test_compat_matrix.py`, in both
directions, against the `TiledUnsupported` strings in
`src/warlock/studio/plotter/`. A refusal with no row fails the suite, and so
does a row for a refusal that no longer exists — which is how a parity
milestone is forced to update this file in the same commit that changes the
behaviour.

Feature names are the refusal's own words, normalised: an interpolated part of
the message is written `{}`, so one row covers one refusal rather than one row
per value it can name.

## Maps

| Feature | State | Notes |
|---|---|---|
| `a {} map` | refused | Staggered and hexagonal. Orthogonal and isometric are drawn; see M5. |
| `an infinite map` | refused | Fixed-size maps only; see M5. |
| `chunked (infinite) layer data` | refused | The JSON spelling of the same thing. |
| `hexagonal 120-degree tile rotation` | refused | The gid bit that only a hex map can set. |

## Layers

| Feature | State | Notes |
|---|---|---|
| `group layers` | refused | Flatten in Tiled first; see M2/M3. |
| `image layers` | refused | See M2/M3. |
| `{} layers` | refused | Any layer kind the JSON reader does not model. |
| `layer pixel offsets` | refused | See M2/M3. |
| `layer data encoded as {}` | refused | CSV and base64 are read; anything else is refused. |
| `{}-compressed layer data` | refused | zlib and gzip are read; zstd is not. |

## Objects

| Feature | State | Notes |
|---|---|---|
| `object templates` | refused | See M7. |
| `tile objects` | refused | See M3. |
| `rotated objects` | refused | An unrotated outline drawn for a rotated object is a wrong picture. |
| `{} objects` | refused | Ellipse, polygon, polyline and text; see M3. |

## Tilesets

| Feature | State | Notes |
|---|---|---|
| `an image-collection tileset` | refused | One sliced atlas per tileset; see M4. |
| `an embedded tileset image` | refused | An `<image source=…>` path is required. |
| `an external .tsj tileset` | refused | Re-save as `.tsx`; see M4. |
| `Wang sets / terrain brushes` | refused | One blob-shaped set is recognised; anything else is refused. See M4. |
| `terrain types` | refused | Tiled's pre-1.5 spelling. |
| `per-tile animation` | refused | See M4. |
| `per-tile collision shapes` | refused | See M4. |
| `per-tile custom properties` | refused | See M4. |

## Properties

| Feature | State | Notes |
|---|---|---|
| `a custom property of type {}` | refused | `file`, `object`, `class` and `list`; see M2. |

## Preserved but not honoured

| Feature | State | Notes |
|---|---|---|
| `renderorder` | preserved-verbatim | Written back as it arrived; the renderer draws right-down. M5 honours it. |
| `backgroundcolor` | preserved-verbatim | Round-tripped; not painted. |

## Permanent non-goals

| Feature | State | Notes |
|---|---|---|
| `oblique projection` | refused | Not a Tiled feature. Tiled 1.12.2 has four orientations and oblique is not one of them, so there is nothing to be compatible with. |
```

**Note on the last table:** `oblique projection` is not a Tiled feature, so nothing in the source raises for it and a `refused` row would fail `test_every_refused_row_still_refuses`. That is why `_rows()` skips the `## Permanent non-goals` section by name — the heading text in the matrix must match `_UNCHECKED_SECTION` in the test exactly. Keep every other feature out of that section: it is for things Tiled does not have, not for things we have not done yet.

- [ ] **Step 5: Cross-link the manual**

Open `docs/manual/09-plotter.md` and find the refusal paragraph (around lines
283–298). Add one sentence pointing at the matrix, keeping every line at or
under 120 characters and not renumbering anything:

```markdown
The full list of what Plotter reads, refuses and preserves is kept in
`docs/PLOTTER_COMPAT.md`, checked against the code on every test run.
```

Run the manual gates afterwards: `uv run pytest tests/manual -q`.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/plotter/test_compat_matrix.py tests/manual -v`
Expected: PASS. If `test_every_refusal_in_the_source_has_a_row` fails, the
assertion prints the exact strings missing from the matrix — add those rows
verbatim rather than paraphrasing them.

- [ ] **Step 7: Commit**

```bash
git add docs/PLOTTER_COMPAT.md tests/plotter/test_compat_matrix.py docs/manual/09-plotter.md
git commit -m "Warlock v0.0.21

Publish the Plotter/Tiled feature matrix and check it against the source.

Both directions: a refusal with no row fails, and so does a row for a
refusal that no longer exists -- which is what forces a parity milestone to
move a row in the commit that changes the behaviour.
"
```

---

### Task 4: Wire the authored corpus (user-gated)

**Blocked on:** the user authoring the seven fixtures in Tiled 1.12.2 per
`FIXTURES.md` and placing them in `tests/plotter/fixtures/tiled/`. Do not
start this task before the files exist, and **do not create them yourself** —
a synthesized golden defeats the entire milestone.

**Files:**
- Modify: `tests/plotter/_corpus.py` (`MANIFEST`)
- Modify: `docs/PLOTTER_COMPAT.md` (fixture names in the notes of any row that becomes `round-trips`)

**Interfaces:**
- Consumes: `MANIFEST`, `pairs()`, `loaders_for()`, `doc_facts` — all from Tasks 1–2.
- Produces: a live corpus gate. Every stem in `MANIFEST` is exercised by four parametrized tests.

- [ ] **Step 1: Confirm what was delivered**

```bash
ls tests/plotter/fixtures/tiled/
uv run python -c "import sys; sys.path.insert(0,'tests'); from plotter._corpus import pairs; print(pairs())"
```

Expected: the seven stems from `FIXTURES.md`, each with both spellings, plus
the `.tsx` and `.png` files they reference.

- [ ] **Step 2: Fill the manifest**

In `tests/plotter/_corpus.py`, replace the empty tuple with exactly what
`pairs()` reported:

```python
MANIFEST: tuple[str, ...] = (
    "basic-iso",
    "basic-ortho",
    "blob-terrain",
    "locked-layers",
    "objects-rect-point",
    "two-tilesets",
    "typed-props",
)
```

- [ ] **Step 3: Run the corpus gate**

Run: `uv run pytest tests/plotter/test_fixture_corpus.py -v`
Expected: PASS on every stem.

**When one fails, it is evidence, not noise.** A `test_both_readers_see_the_same_map`
failure means the two readers genuinely disagree about a real Tiled file — that is
a bug this milestone was built to find, and the fix belongs in `tmx.py`, in its own
commit, with the failing fixture named in the message. A `TiledUnsupported` raised
by a fixture means either the fixture used a feature it should not have (re-author
it per `FIXTURES.md`) or we refuse something ordinary (a finding: record it, and
decide whether it is M1 scope or belongs to the milestone that owns that feature).
Do not weaken the assertion to get to green.

- [ ] **Step 4: Move the rows that are now proven**

Any matrix row whose feature these fixtures exercise gains a fixture name in
its note. Concretely, add to `docs/PLOTTER_COMPAT.md` a `## Maps`-level set of
`round-trips` rows for what the corpus now proves, each naming its stem in
backticks — for example:

```markdown
| `orthogonal maps` | round-trips | `basic-ortho`, `two-tilesets` |
| `isometric maps` | round-trips | `basic-iso` |
| `tile flip flags` | round-trips | `basic-ortho` |
| `multiple external tilesets` | round-trips | `two-tilesets` |
| `rectangle and point objects` | round-trips | `objects-rect-point` |
| `string/int/float/bool/color properties` | round-trips | `typed-props` |
| `layer lock and visibility` | round-trips | `locked-layers` |
| `one blob-shaped Wang set` | round-trips | `blob-terrain` |
```

`test_a_row_that_claims_to_round_trip_names_a_fixture_that_exists` is what
keeps these honest.

- [ ] **Step 5: Full suite**

Run: `uv run pytest`
Expected: 6705 + the new tests, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add tests/plotter/fixtures/tiled tests/plotter/_corpus.py docs/PLOTTER_COMPAT.md
git commit -m "Warlock v0.0.21

Add the Tiled 1.12.2 fixture corpus and turn on its gate.

Seven maps authored in Tiled itself, each saved as .tmx and .tmj: the
readers are now checked against files we did not write.
"
```

---

### Task 5: The honest version bump (verification-gated)

**Blocked on:** Task 4 green, **and** a human confirming Tiled 1.12.2 opens one of our exports.

**Why gated:** `TILED_VERSION` is a claim about which Tiled release we target. Writing `1.12.2` into every exported file before any 1.12.2 has opened one is a claim we have not earned. The check is thirty seconds of work and it is the difference between a version string and a tested one.

**Files:**
- Modify: `src/warlock/studio/plotter/tsx.py:42-43`
- Modify: whichever tests pin the emitted `tiledversion` bytes (find them; do not guess)

**Interfaces:**
- Consumes: nothing.
- Produces: `tsx.TILED_VERSION == "1.12.2"`. `TSX_VERSION`/`MAP_VERSION` stay `"1.10"` — the *format* version moves only when we first emit a 1.12-only construct, which is M2's class/list properties.

- [ ] **Step 1: Produce an export and open it in Tiled**

```bash
uv run pytest tests/plotter/test_fixture_corpus.py -q
```

Then, in a Python shell, write one fixture's export to a scratch directory:

```python
import sys; sys.path.insert(0, "tests")
from pathlib import Path
from plotter._corpus import FIXTURE_DIR, loaders_for
from warlock.studio.plotter import tmx

out = Path("C:/Users/ILWT/AppData/Local/Temp/claude/D--Projects-Warlock/export-check")
out.mkdir(parents=True, exist_ok=True)
doc = tmx.read_tmx((FIXTURE_DIR / "two-tilesets.tmx").read_bytes(), **loaders_for())
for name, blob in tmx.tmx_export(doc).items():
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
print(out)
```

**Ask the user to open `map.tmx` from that directory in Tiled 1.12.2** and
confirm it loads with both tilesets and the painted cells intact. Record the
answer in the commit message. If it does **not** open, stop: that is a real
interop bug and it outranks the version string.

- [ ] **Step 2: Find what pins the version bytes**

```bash
uv run pytest -q -k "tiled or version" tests/plotter
grep -rn "1.10.2" tests/ src/
```

- [ ] **Step 3: Make the change**

In `src/warlock/studio/plotter/tsx.py`, lines 40–43:

```python
# What this build writes. Tiled accepts anything it recognises; these are the
# values a current Tiled writes and so the values least likely to surprise it.
#
# The two move for different reasons and must not be bumped together.
# ``TSX_VERSION`` is the *format* version and moves only when this writer
# first emits a construct that older Tiled cannot read -- the next such move
# is class and list properties. ``TILED_VERSION`` is the release we target and
# test against, and it moved to 1.12.2 once a 1.12.2 had opened our export.
TSX_VERSION = "1.10"
TILED_VERSION = "1.12.2"
```

- [ ] **Step 4: Update the pinned expectations**

Change every test the grep found, from `1.10.2` to `1.12.2`, **only** where the value being pinned is `tiledversion`. Leave `version="1.10"` expectations alone — that is `TSX_VERSION`, and it is deliberately not moving.

- [ ] **Step 5: Full suite**

Run: `uv run pytest`
Expected: 0 failures. Any remaining failure mentioning `1.10.2` is a pin you missed.

- [ ] **Step 6: Commit**

```bash
git add src/warlock/studio/plotter/tsx.py tests/
git commit -m "Warlock v0.0.21

Target Tiled 1.12.2 in exported metadata.

TILED_VERSION only; TSX_VERSION stays 1.10, because the format version
moves when we first emit a 1.12-only construct and not before. Verified by
opening an exported map in Tiled 1.12.2.
"
```

---

## Milestone close

Per `PLOTTER_PLAN.md`'s per-milestone ritual, before calling M1 done:

1. **Refusal ledger** — no refusals were flipped this milestone (M1 changes no behaviour), so no rows moved from `refused`. Rows *added* as `round-trips` in Task 4 name their fixtures.
2. **Import pin** — `tests/plotter/test_plotter_imports.py:131` untouched: M1 added no engine module. Confirm this is still true before closing.
3. **`docs/INVARIANTS.md`** — no invariant sentence changed. `:111`/`:113` untouched.
4. **Manual** — `09-plotter.md` gained one cross-link; no chapter renumbering; `tests/manual` green.
5. **Fixtures** — seven pairs plus `FIXTURES.md`.
6. **Full suite** — `uv run pytest`, never editing `src/` while it runs.
7. **Commit** — `Warlock v0.0.21` throughout; no version bump unless the user asks.

**What M1 hands to M2:** `doc_facts` (M2 proves `.wmap` v3 lossless with it), the corpus and its loaders (M2 adds `wmap/v1`+`v2` fixtures beside it), and the matrix (M2's property flip is the first row that moves).
