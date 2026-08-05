# Build

Build is the top-level modelling mode: primitives, transforms and a material palette, in the same
window as everything else. It exists because the pipeline has one weak spot that no prompt fixes —
if you already know the shape you want, describing it to a diffusion model and hoping is a poor way
to get it. Build lets you say it directly.

It is a mode, not a takeover. Switching away leaves every open document exactly where it was, and a
reconstruction started before you switched keeps running with its progress card floating over the
viewport. Only quitting the app can lose unsaved work, and it asks first.

The layout follows the rest of the app: tools and the selected object's properties on the left, the
viewport in the middle, the outliner and the document panel on the right. Several documents stay
open at once.

## Adding a primitive

The **add** row has one button per primitive: box, plane, cylinder, cone, UV sphere and torus.
Clicking one places it at the origin and selects it. Hovering a button names it.

A placed object remembers *how it was made*. Its generator and the parameters it was built with are
kept, so the properties panel offers those parameters — a cylinder's radius, height and segment
count — and changing one rebuilds the mesh. That is a single undo step, so `Ctrl+Z` takes the
object back to the shape it had rather than to some intermediate state.

## Transforming

Four tools, on `Q`, `W`, `E` and `R`:

| Tool | Key | What it does |
| --- | --- | --- |
| Select | `Q` | Click an object in the viewport to select it. |
| Move | `W` | Three arrows; drag one to slide along that axis. |
| Rotate | `E` | Three rings; drag one to turn about that axis. |
| Scale | `R` | Three handles plus a centre handle for uniform scale. |

A drag is one undo step, recorded when you let go — not one step per frame of the drag, which would
bury everything else in the history.

The **Move**, **Rotate** and **Scale** values are also typed directly in the properties panel, which
is the better way to place something exactly. Rotation is shown as a quaternion in `XYZW` order,
which is what every file this app writes uses.

Two operations act on the whole selection. **Duplicate** (`Ctrl+D`) makes a copy under a new name,
counting up — `Box`, `Box.001`, `Box.002`. **Bake** folds an object's position, rotation and scale
into its geometry and resets the transform to identity, which is what you want before measuring
something or exporting it into a frame that has to match.

**Mirror X / Y / Z** reflects the object across a plane through its own origin. It is baked into the
mesh rather than expressed as a negative scale, and that is deliberate: glTF readers disagree about
whether a negative scale flips the winding order, so an asset that used one would render correctly
here and inside out in some engines, with nothing in the file to explain it. Mirroring here rebuilds
the geometry and reverses the faces, which is true under every reader.

## Snapping

**Snap** quantises a drag to a grid — a distance for moving, an angle for rotating. Both are set
beside the toggle, and setting either to zero turns that half off rather than snapping everything to
the origin.

Snapping applies to gizmo drags only. A number typed into the properties panel is used exactly as
typed, because you already said what you meant.

## Materials

Every object points at a slot in the document's material palette, chosen in the properties panel.
The slot's **base colour**, **metallic** and **roughness** are edited there too, and the change
reaches every object using that slot at once — which is the point of a palette rather than a
material per object.

Build Phase 1 has no textures. The material factors are what the exported asset carries, and they
are enough for a blockout: what the pipeline is for is turning that blockout into something with a
surface.

## Saving

`Ctrl+S` saves the document as a `.wblk` — a zip holding `scene.json` (the objects, their
transforms, their generator parameters and the palette) plus one compressed mesh per object. The
JSON half is sorted and indented so it is readable and diffable, and two saves of an unchanged
document produce byte-identical files.

A saved document keeps every object's identity, so an undo recorded before the save still lands on
the object it was made against after you reopen it.

## The two ways out

Build has two output paths and they do genuinely different things. Choosing between them is the
whole reason both exist.

**Export to library** puts the *exact* geometry in the library as an ordinary asset. It is a
finished model row from the moment it lands, so it inherits everything the rest of the app does to a
mesh: rigging, posing, sprite sheets, the triangle retarget, and the STL, OBJ, FBX, collision and
texture exports. Use it when the shape you modelled is the shape you meant.

**Send to 3D** renders the document flat, on a plain background, with no grid and no gizmos, and
hands that picture to the reconstruction stage. What comes back is *not* your geometry — it is a
reconstruction that used your geometry as a suggestion, with surface detail and irregularity nobody
modelled. Use it when the blockout is a silhouette and proportion study rather than a final shape.

A built asset cannot be rerolled or remeshed. There is no generator behind it: a new seed would
change nothing, and there is no reference image to reconstruct from. The way to get a different mesh
is to open the document and change it.

## Where the files go

An exported asset is an ordinary job directory, and the document that produced it is stored beside
the mesh as `build.wblk`. That copy is never served or downloadable — it exists so that reopening a
built asset brings its objects back instead of one frozen mesh — and it goes away with the job when
the job is deleted.

Dropping a `.wblk` on the window while Build is on screen opens it. Dropping a `.glb` does not:
reading a mesh back into editable objects is not something Build does yet, and quietly making a
frozen single-object document out of it would be a different feature wearing this one's name.

Every binding is listed in [Keyboard shortcuts](09-shortcuts.md).
