# The `.wmap` back-compatibility fixtures

Two checked-in archives, `v1.wmap` and `v2.wmap`, each written by the **actual
historical writer** at the version it names. They exist for one reason: every
other case in `tests/plotter/test_wmap.py` builds a document in Python and
asserts that our reader agrees with our writer, and that cannot catch a reader
which has quietly come to require a key the old writer never emitted. A format
that reads three versions needs bytes from all three, and only one of them is
bytes this build can produce.

**Never regenerate these from the current writer.** A fixture written by
today's encoder and checked in as a golden records today's assumptions and
calls them version 1's — which is the same argument `fixtures/tiled/FIXTURES.md`
makes about synthesizing a Tiled file.

## Provenance

| file | writer | commit |
| --- | --- | --- |
| `v1.wmap` | `wmap.wmap_bytes` with `VERSION = 1` | `05ba731^` (the last commit before the version-2 bump) |
| `v2.wmap` | `wmap.wmap_bytes` with `VERSION = 2` | `d13f0a1` (the last commit before the version-3 bump) |

Both writers still *run* against today's document model — neither reaches for
anything `tilemap`, `tileset` or `tsx` has since removed — which is what made
this possible without a checkout of the old tree.

## Regenerating them

Only ever needed if a fixture is lost or the recipe below is deliberately
changed. Extract the two historical modules, load each with its `__package__`
set so its relative imports resolve against the installed package, and run it
over the document in the next section:

```python
import importlib.util, sys
from pathlib import Path

def load(path, name):                      # after:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "warlock.studio.plotter"   # so ``from . import gid`` resolves
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# git show 05ba731^:src/warlock/studio/plotter/wmap.py > old_v1.py
# git show d13f0a1:src/warlock/studio/plotter/wmap.py  > old_v2.py
v1, v2 = load(Path("old_v1.py"), "_legacy_v1"), load(Path("old_v2.py"), "_legacy_v2")
assert (v1.VERSION, v2.VERSION) == (1, 2)
Path("v1.wmap").write_bytes(v1.wmap_bytes(build("orthogonal", locked=False)))
Path("v2.wmap").write_bytes(v2.wmap_bytes(build("isometric", locked=True)))
```

## The document both fixtures hold

Deliberately small — a 4×3 map over a two-tile atlas — and deliberately the
*same* document at two settings, so that a difference between the two files is
a difference between the two writers rather than between two maps:

```python
def build(projection: str, locked: bool) -> MapDoc:
    pixels = np.zeros((16, 32, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[4, 4] = (10, 20, 30, 255)
    pixels[8, 20] = (40, 50, 60, 255)

    doc = MapDoc(4, 3, 16, 16, projection=projection)
    doc.add_tileset(Tileset(name="terrain", pixels=pixels, tile_w=16, tile_h=16,
                            properties={"kind": tsx.Prop("string", "outdoor")}))
    tiles = doc.add_tile_layer("Ground")
    cells = np.zeros((3, 4), gid.DTYPE)
    cells[0, 0] = gid.compose(1)
    cells[2, 3] = gid.compose(2, flip_h=True, flip_d=True)
    doc.write_region(tiles.uid, 0, 0, cells)
    doc.set_layer_props(tiles.uid, opacity=0.5)
    if locked:
        doc.set_layer_props(tiles.uid, locked=True)
    things = doc.add_object_layer("Things")
    doc.add_object(things.uid, MapObject(uid=new_uid(), name="spawn", kind="point",
                                         x=1.5, y=2.5,
                                         properties={"team": tsx.Prop("int", 1)}))
    doc.add_object(things.uid, MapObject(uid=new_uid(), name="zone", kind="rect",
                                         x=3.0, y=4.0, w=5.0, h=6.0,
                                         obj_class="Trigger", visible=False))
    doc.properties = {"theme": tsx.Prop("string", "cave")}
    doc.backgroundcolor = "#ff102030"
    doc.renderorder = "right-up"
    return doc
```

`projection` and `locked` are what version 2 could store and version 1 could
not, which is why the two files differ in them: `v1.wmap` proves the *defaults*
(orthogonal, unlocked) and `v2.wmap` proves that tolerance is not amnesia — a
value version 2 really did store still comes back.

Neither file carries a layer or object `id`, a class, a tint, an offset, a
parallax factor, a draw order, a shape record or the `next_*` counters. Every
one of those is what the reader has to default, and
`test_an_older_fixture_opens_with_identity_decorations_and_fresh_ids` is where
each is named.
