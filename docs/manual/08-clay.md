# Clay

Clay is the top-level modelling mode: primitives, transforms and a material palette, in the same
window as everything else. It exists because the pipeline has one weak spot that no prompt fixes —
if you already know the shape you want, describing it to a diffusion model and hoping is a poor way
to get it. Clay lets you say it directly.

It is a mode, not a takeover. Switching away leaves every open document exactly where it was, and a
reconstruction started before you switched keeps running with its progress card floating over the
viewport. Only quitting the app can lose unsaved work, and it asks first.

The layout follows the rest of the app: tools and the selected object's properties on the left, the
viewport in the middle, the outliner and the document panel on the right. Several documents stay
open at once.

## Starting a document

With nothing open, the middle column offers **New model** and **Open a file...**, and lists the
documents you had open recently — clicking one reopens it, and hovering it shows the full path. The
document panel on the right offers the same two buttons. Once a document is open, `Ctrl+N` and
`Ctrl+O` do the same two things from the keyboard.

Choosing **Clay** from the Home screen opens an empty document for you when there is nothing open
already. When there is, it leaves your documents exactly as they were — the documents *are* the
work, and entering the mode is not a reason to disturb them.

## Adding a primitive

The **add** row has one button per primitive: box, plane, cylinder, cone, UV sphere and torus.
Clicking one places it at the origin and selects it. Hovering a button names it.

A placed object remembers *how it was made*. Its generator and the parameters it was built with are
kept, so the properties panel offers those parameters — a cylinder's radius, height and segment
count — and changing one rebuilds the mesh. That is a single undo step, so `Ctrl+Z` takes the
object back to the shape it had rather than to some intermediate state.

## Element modes

Every object starts as one thing you can move about. Press `1`, `2` or `3` and it becomes a mesh you
can take apart: vertices, edges, faces. `4` goes back to object mode. The buttons above the add row
say the same thing, and highlight whichever mode the document is in.

| Key | Mode | What clicking selects |
| --- | --- | --- |
| `1` | Verts | One vertex |
| `2` | Edges | One edge |
| `3` | Faces | One face |
| `4` | Object | The whole object |

The mode belongs to the *document*, not to the app, so switching tabs does not reinterpret what you
had selected in the other one.

Clicking replaces the selection, `Shift`+click adds to it and `Ctrl`+click removes from it. Clicking
the object but missing everything on it clears that object; clicking empty space with the **Select**
tool (`Q`) starts a marquee, and a marquee that ends where it started clears everything. A marquee
takes a vertex inside the rectangle, an edge only when *both* ends are inside, and a face only when
*all* its corners are — and it selects through the mesh, back face included, because that is what a
rectangle dragged over a blockout means.

`Alt`+drag always orbits, in every mode. That is the one gesture that is never reinterpreted, since
it is how you look at what you are about to click.

Right-click opens the context menu, listing exactly the operations that apply in the current mode
with the ones that cannot run greyed out. The same list drives the buttons in the tools column, so
neither can offer something the other refuses. Operations that take a number — bevel, inset, weld,
loop cut, smooth — open a small dialog with the fields and an **Apply** button, and remember what you
last used.

`Ctrl+A` selects everything in the current mode's sense of everything, `Ctrl+Shift+I` inverts it, and
`Esc` steps back: first it drops the element selection, then it leaves the element mode, then it
clears the object selection. `Delete` in an element mode deletes *faces*, never the object. Neither
`Ctrl+A` nor `Ctrl+Shift+I` reaches a **hidden** object, in either sense of everything: hiding
something takes it out of what you are working on, so nothing you select can act on it by accident.

**Element selection is transient.** It is not saved with the document, and an undo that changes
geometry drops it — the indices it named describe a mesh that no longer exists. An undo that only
moves or renames something keeps it, because those cannot invalidate it.

### The operations

| Mode | Operation | What it does |
| --- | --- | --- |
| Faces | Extrude (`E`) | Pulls the selected faces off the surface and walls in the gap. It moves nothing — drag the returned faces with `W`. |
| Faces | Inset | Shrinks each face in place and rings it with the rim it vacated. |
| Faces | Subdivide | Splits each face into quads without changing the shape. |
| Faces | Flip Normals | Reverses the winding of the selected faces. |
| Edges | Bevel | Replaces each edge with a flat quad, mitring the corners where several meet. |
| Edges | Loop Cut | Rings a strip of quads with a new edge loop. |
| Edges | Fill Hole | Caps the boundary ring the selected edge belongs to. |
| Edges, Faces | Collapse | Pulls the selection down to a single point. |
| Verts | Weld | Merges vertices closer together than a distance you give. |
| Any | Dissolve | Removes the selection and merges what it separated, rather than leaving a hole. |
| Any | Smooth | Catmull-Clark subdivision over the whole object. Each level multiplies the face count by four. |

An operation that cannot do what you asked says so in a toast naming the element and what to do
instead, and changes nothing. Bevel refuses a boundary edge; dissolve refuses a selection that rings
a face it does not include; fill hole refuses a pinched boundary. Those are refusals, not failures:
the alternative is geometry that looks right and is not.

The first operation that changes an object's topology **freezes** it. A box that has been extruded is
no longer describable as "box, size 1", so the properties panel switches from the generator's
parameters to a vertex and face count.

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

The gizmos work on elements too. In an element mode they sit at the centre of what is selected
*inside* the objects rather than at the object's own centre, and dragging one moves those vertices.
That is one undo step per object per drag, and a drag that ends where it started records nothing at
all. **Select** (`Q`) shows no gizmo in an element mode, which is what leaves the left button free
for the marquee.

The **Move**, **Rotate** and **Scale** values are also typed directly in the properties panel, which
is the better way to place something exactly. Rotation is shown as a quaternion in `XYZW` order,
which is what every file this app writes uses.

Two operations act on the whole selection. **Duplicate** (`Ctrl+D`) makes a copy under a new name,
counting up — `Box`, `Box.001`, `Box.002`. **Bake** folds an object's position, rotation and scale
into its geometry and resets the transform to identity, which is what you want before measuring
something or exporting it into a frame that has to match.

## Merging objects

**Merge Objects...** (`Ctrl+J`, object mode, two or more selected) turns several shapes into one.
The survivor is the **topmost selected object in the outliner** — it keeps its name, its transform
and its default material — and everything else is carried into its frame, appended to its geometry
and removed from the document. That is one undo step: a `Ctrl+Z` that put one absorbed object back
while the survivor still carried the merged geometry would show you a shape existing twice.

Only objects you can see are merged. A hidden object stays hidden and untouched even when it is
selected, and **Merge Objects...** greys out unless two visible objects are selected — hiding
something means it is not part of what you are working on, and a merge is the one operation where
absorbing an unseen object would leave no trace of having done so.

The dialog asks for a **weld distance**, in metres of world space. Vertices closer together than
that are merged into one at their centroid, which is what makes two shapes that *touch* come out as
a single continuous surface rather than as two shells sharing a plane. Setting it to zero keeps
every vertex, which is the honest answer for parts that are meant to stay separate inside one
object. The distance means the same thing whatever the survivor is scaled to: 1 mm is 1 mm on the
ruler, not 1 mm in whatever units the survivor's own transform happens to work in.

The weld is applied to the **whole merged result**, not only where the shapes meet. That is what
lets three objects touching at one point come out joined, and it costs nothing for ordinary work —
texture coordinates are stored per face corner, so a weld carries UV seams through untouched, and
nothing else in Clay leaves two vertices sitting at one position. The exception worth knowing:
merge at zero to keep two parts as separate shells, then merge *that* object with a third at a
non-zero distance, and the shells you kept apart are welded together. Merge the third one first, or
keep the parts as separate objects until last.

It is a weld and not a solid union. Geometry inside an overlap is kept rather than cut away,
because deciding which faces are inside another solid means classifying every face against every
other one, and a wrong classification quietly deletes a surface you can see. If you want the
interior gone, delete those faces in face mode before merging.

A merged object is no longer what a generator would build, so its generator claim is dropped along
with the merge — the properties panel stops offering the size field that would have rebuilt a
pristine box over your work. That drop is part of the same undo step.

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

## Checking a mesh

The properties panel has a **mesh check** under the generator's parameters. It measures the selected
object against the defects a mesh can carry without anything noticing: holes, non-manifold edges
(three or more faces on one edge), inconsistently wound faces, duplicate faces and vertices no face
uses. Each finding is a button — clicking it switches to the element mode the defect lives in and
selects exactly the offenders, so "3 non-manifold edges" becomes three edges you are looking at.

It runs when you press the button and not before. Building the tables it needs is proportional to
the size of the mesh, which is fine once and unacceptable sixty times a second, so the answer is
kept and marked *edited since the last check* the moment anything changes the geometry.

None of it is a verdict. An open sheet is a perfectly good mesh and so is a plane with no thickness;
what the check tells you is what is there, and whether that matters depends on where the asset is
going. A game engine will usually want a closed mesh; a decal will not.

## Materials

Every object points at a slot in the document's material palette, chosen in the properties panel.
The slot's **base colour**, **metallic** and **roughness** are edited there too, and the change
reaches every object using that slot at once — which is the point of a palette rather than a
material per object.

Clay paints no textures — but it **carries** them. A material that arrived with an imported asset
keeps its baked maps: they render in the viewport, they are stored in the `.wblk`, and they are
written back into an exported GLB. The properties panel shows which slots a material carries as a
read-only line, because there is nothing here that could replace one and offering a control that
looked like it could would be promising a feature that does not exist.

For an asset modelled here from scratch, the material factors are what the exported asset carries,
and they are enough for a blockout: what the pipeline is for is turning that blockout into something
with a surface.

## Saving

`Ctrl+S` saves the document as a `.wblk` — a zip holding `scene.json` (the objects, their
transforms, their generator parameters and the palette) plus one compressed mesh per object. The
JSON half is sorted and indented so it is readable and diffable, and two saves of an unchanged
document produce byte-identical files.

A saved document keeps every object's identity, so an undo recorded before the save still lands on
the object it was made against after you reopen it.

## The two ways out

Clay has two output paths and they do genuinely different things. Choosing between them is the
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

## Importing an asset

Dropping a `.glb` on the window while Clay is on screen imports it, and the library card's overflow
menu has **Edit in Clay** for any finished model.

**Edit in Clay** prefers the document you authored. If the asset was exported from Clay, its
`build.wblk` sidecar is reopened — objects, names, generator parameters and all. If it was not, the
served `model.glb` is imported instead: that is the optimized, grounded mesh, not the raw
reconstruction.

An imported mesh comes in as one object per material, with its vertices merged back together
bitwise. An exporter splits a vertex wherever a normal or a texture coordinate disagrees, and those
split copies are bit-identical in position, so merging them is exact — there is no tolerance to
choose and no chance of welding two features that are a hair apart. If you *want* tolerance welding,
that is what **Weld** is for.

Texture coordinates survive, because Clay stores them per face corner: a seam is two corners at one
vertex, which is exactly what the exporter split produced. Smoothing is a heuristic — a face whose
corner normals agree with its own geometric normal was flat-shaded, and one where they do not was
smooth.

Two things are refused rather than half-done. A **rigged** GLB, because Clay has no skinning and
editing it would drop the rig; open it in 3D mode instead. And a mesh past two million triangles,
because the editor holds every mesh twice per undo step. Past two hundred thousand it asks first,
since every edit rebuilds the whole mesh and you should know that before you press Extrude.

## Where the files go

An exported asset is an ordinary job directory, and the document that produced it is stored beside
the mesh as `build.wblk` (the on-disk name predates the rename). That copy is never served or
downloadable — it exists so that reopening a built asset brings its objects back instead of one
frozen mesh — and it goes away with the job when the job is deleted.

Dropping a `.wblk` on the window while Clay is on screen opens it, and dropping a `.glb` imports it
— see [Importing an asset](#importing-an-asset).

Every binding is listed in [Keyboard shortcuts](12-shortcuts.md).
