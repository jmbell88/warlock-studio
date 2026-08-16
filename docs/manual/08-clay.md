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

The **add** row has one button per primitive: box, plane, grid, cylinder, cone, UV sphere,
icosphere, capsule and torus. Clicking one places it at the origin and selects it. Hovering a button
names it.

Three of those are near-duplicates of others and are worth telling apart. **Grid** is a plane cut
into squares; **plane** is the single quad, which is what a decal or a backdrop wants, and a grid is
what you need the moment you want to bend the sheet, because only interior vertices can move.
**Icosphere** and **UV sphere** are both balls: the icosphere's triangles are all much the same size,
which is what makes it the one to sculpt, bevel or bake to, and the UV sphere is laid out in
latitude and longitude, which is what makes it the one to wrap an equirectangular texture round.
**Capsule**'s `height` is its cylindrical middle alone, so the whole shape is that plus a radius at
each end.

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
| Any | Extrude (`E`) | Pulls the selection off the surface and walls in the gap. It moves nothing — drag what it hands back with `W`. |
| Faces | Inset | Shrinks each face in place and rings it with the rim it vacated. |
| Faces | Subdivide | Splits each face into quads without changing the shape. |
| Faces | Flip Normals | Reverses the winding of the selected faces. |
| Edges | Bevel | Replaces each edge with a flat quad, mitring the corners where several meet. |
| Edges | Loop Cut | Rings a strip of quads with a new edge loop. |
| Edges | Bridge Loops | Joins two selected boundary loops with a strip of quads. |
| Edges | Fill Hole | Caps the boundary ring the selected edge belongs to. |
| Edges, Faces | Collapse | Pulls the selection down to a single point. |
| Verts | Weld | Merges vertices closer together than a distance you give. |
| Any | Dissolve | Removes the selection and merges what it separated, rather than leaving a hole. |
| Faces | Merge Faces | The same operation as Dissolve in face mode, under the name most people look for. |
| Any | Smooth | Catmull-Clark subdivision over the whole object. Each level multiplies the face count by four. |

An operation that cannot do what you asked says so in a toast naming the element and what to do
instead, and changes nothing. Bevel refuses a boundary edge; dissolve refuses a selection that rings
a face it does not include; fill hole refuses a pinched boundary. Those are refusals, not failures:
the alternative is geometry that looks right and is not.

**Merge Faces** and **Dissolve** in face mode are one operation with two names, and the duplication
is deliberate: "dissolve" is the modelling word and is what the vertex and edge modes do too, but
somebody who wants two faces to become one searches for "merge", finds nothing, and concludes the
editor cannot do it. Two things about what it merges are worth knowing. A selection that falls into
several disconnected blocks becomes one face *per block*, not one face overall — a face with a hole
in it is not something this editor's meshes can hold, so there would be nothing to make. And a
single face on its own is refused: there is no neighbour to merge it with.

**Extrude** is one operation in all three modes, because it means the same thing in all three. In
edge mode it grows a quad from each selected *boundary* edge — an edge with a face on each side has
no open side to grow into, and it says so. In vertex mode it extrudes the border edges between the
vertices you selected, which is the only reading available: a mesh here stores faces, not loose
wires, so a vertex on its own has nothing to extrude and says that too.

**Bridge Loops** is the one to reach for when two things need joining: select the boundary edges of
both openings and it skins a strip of quads between them, which is what makes two tubes one tube and
what closes the gap left by deleting a band of faces. Both openings must be *boundaries* — bridging
two interior rings means deleting the faces between them first, and doing that for you would remove
geometry you did not ask to lose. It also needs exactly two loops, of the same length, both open or
both closed; anything else it refuses by name, because there is no pairing to guess at.

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
which is what every file this app writes uses. Under them is a read-only **size** row: the object's
world-space width, depth and height in metres, after its transform — the number a scale of 2 on a
generator whose radius is 0.35 does not tell you.

### Locking an axis, and typing a number

The keyboard joins a drag already under way. While a gizmo is held:

| Key | What it does |
| --- | --- |
| `X` / `Y` / `Z` | Lock the drag to that axis. The same key again clears the lock. |
| digits, `.`, `-` | Type the value outright — metres for a move, degrees for a rotation, a factor for a scale. |
| `Backspace` | Take back the last character. |
| `Enter` | Commit and end the drag. |
| `Esc` | Cancel it: everything goes back where it was and nothing is recorded. |

A readout beside the cursor says what the drag currently amounts to, so `X` then `2` is "two metres
along X" with no dragging left in it. The two compose, and they are different in kind: a lock says
which *direction*, leaving the mouse in charge of the amount, while a number is the amount. That is
why a number on its own still means something — it sets the distance along whichever way you were
already dragging, and the size of a uniform scale.

### Proportional editing

**Soft falloff**, in the snap section, makes an element drag carry the geometry *around* the
selection with it, fading out over the radius, so the surface bends instead of tearing. The radius
is metres of world space and is measured from the nearest selected vertex, not from the middle of
the selection — dragging one end of a long strip fades out away from that end rather than along the
strip. Setting the radius to zero is the same as switching it off.

Two operations act on the whole selection. **Duplicate** (`Ctrl+D`) makes a copy under a new name,
counting up — `Box`, `Box.001`, `Box.002`. **Bake** folds an object's position, rotation and scale
into its geometry and resets the transform to identity, which is what you want before measuring
something or exporting it into a frame that has to match.

## The outliner

Every object in the document, newest at the top. Click to select, `Ctrl`-click to toggle one and
`Shift`-click to take a range. The filter box above narrows the list by name.

The eye on each row hides an object, and a hidden object does not render, does not export and cannot
be clicked in the viewport. **Solo** above the list hides everything *except* what is selected and
**Show all** brings them back — each is a single undo step, so `Ctrl+Z` is a third way out.

Rows are dragged to reorder them, which matters because display order is the order the objects come
out in an exported GLB. Reordering is switched off while the filter box has something in it: the
rows on screen are then a subset, so there is no honest answer for where a drop between two of them
lands in the real list.

Right-clicking a row selects it and offers **Rename**, **Duplicate**, **Solo** and **Delete**.
Double-clicking a name renames it in place.

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

It is a weld and not a solid union. Geometry inside an overlap is kept rather than cut away — for
that, use **Union Objects...** below.

A merged object is no longer what a generator would build, so its generator claim is dropped along
with the merge — the properties panel stops offering the size field that would have rebuilt a
pristine box over your work. That drop is part of the same undo step.

## Union objects

**Union Objects...** (object mode, two or more visible objects selected) is the other way of turning
several shapes into one, and it answers a different question. A merge keeps everything: two
interpenetrating cubes come out as one object still carrying both sets of interior walls, which
z-fight, get exported, and mean the result is not a closed solid. A union cuts those walls away, so
what you get back is the single solid the two shapes look like they describe.

Everything about *which* object survives is the same as a merge: the topmost selected object in the
outliner keeps its name, transform and default material, everything else is carried into its frame,
hidden objects are left alone, and the whole thing is one undo step with the generator claim dropped
inside it. There is no weld distance, because a union has nothing to weld — it recomputes the
surface rather than joining two of them.

Three things it costs, which are why the merge is still here:

- **Texture coordinates are lost.** A union recuts every face it touches, so no corner of the result
  corresponds to any corner of the inputs and there is nothing honest to carry across. Unwrap the
  result afterwards.
- **Quads become triangles.** The solver works in triangles and hands back triangles, so a union of
  two clean quad boxes is a triangle soup — worse to subdivide and worse to edit by hand.
- **Every object must be a closed solid.** "Inside" is only meaningful for something that encloses a
  volume, so an object with holes or loose faces is refused by name rather than guessed at. Fill the
  holes first, or merge instead.

Objects that do not touch at all are a perfectly good union: you get one object holding two separate
shells, exactly as a merge at weld distance zero would give you.

**Mirror X / Y / Z** reflects the object across a plane through its own origin. It is baked into the
mesh rather than expressed as a negative scale, and that is deliberate: glTF readers disagree about
whether a negative scale flips the winding order, so an asset that used one would render correctly
here and inside out in some engines, with nothing in the file to explain it. Mirroring here rebuilds
the geometry and reverses the faces, which is true under every reader.

## Axis views

`Ctrl+1`, `Ctrl+3` and `Ctrl+7` snap the camera to the front, right and top views — the numbers
Blender puts them on, so the muscle memory carries over. Hold `Shift` for the opposite view, which
is how three keys cover six. `Ctrl+5` toggles an **orthographic** projection, where parallel edges
stay parallel and there is no perspective foreshortening; it is what you want for lining two things
up. The buttons **F**, **R**, **T** and **Ortho** in the tool panel do the same.

An axis view changes the *angle* only. It keeps the distance and whatever you were looking at,
because reframing would lose the part of the model you were about to line up. Switching to
orthographic keeps the scale at the point you are looking at, so it reads as a change of projection
rather than a jump cut.

These are Clay's keys, not the app's — which is why switching mode is a click on the rail or a
command in the palette rather than a digit. A global binding is checked above Clay and would take
the key from it permanently.

## Snapping

**Snap** quantises a drag to a grid — a distance for moving, an angle for rotating. Both are set
beside the toggle, and setting either to zero turns that half off rather than snapping everything to
the origin.

Snapping applies to gizmo drags only. A number typed into the properties panel is used exactly as
typed, because you already said what you meant.

**Snap to vertex** is a separate switch beside it, and the two answer different questions: the grid
puts things on round numbers, this puts them exactly *there*. While moving, the drag lands on the
vertex under the cursor. The vertices being moved are never candidates, so a drag cannot snap onto
itself; and locking an axis or typing a value during the drag overrides it, because at that point
you have said where the thing goes.

## Texture coordinates

Every primitive comes with texture coordinates already on it — a box's six faces, a cylinder's band
with its two caps tucked into the corners, a sphere laid out pole to pole, a torus wrapped both
ways. They are laid out so that no two parts of one shape sit on top of each other in the square,
because a texture cannot be baked onto a layout whose pieces overlap.

**Box Unwrap** recomputes them for whatever is selected, projecting each face along whichever axis
it points along most. It is the same projection the box primitive uses, and it is the right answer
for the blockout geometry Clay makes: instant, predictable, and with no way to fail. It is
deliberately not a conformal unwrap — that is the right tool for an organic surface and the wrong
one for a shape made of flat panels, and it fails in ways a modelling panel has nothing useful to
say about.

Two things follow from it being a *projection*. Faces pointing opposite ways share the same square,
so a texture applied to a box appears on both the front and the back (mirrored, so lettering reads
the right way round on each). And unwrapping does not freeze a generator: coordinates are not
geometry, so an unwrapped box is still a box and editing its size still rebuilds it.

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

**Add** appends a new slot and **Remove** drops one — but only a slot no face is using, and the
panel says how many faces are in the way when it will not. Reassigning those faces to some other
slot is the alternative, and it is a silent change to how part of the model looks. A slot is an
index that every face names, so adding always appends rather than inserting; removing one renumbers
the slots above it, and an undo puts the numbering back.

**Shade Smooth** and **Shade Flat** set how faces are shaded — the whole object in object mode, the
selected faces in face mode. **Shade Auto** decides per face from the angle between neighbours: a
sphere or a torus comes out smooth, a box stays flat.

One thing about Shade Auto is worth knowing before it surprises you: **a capped cylinder comes out
entirely flat.** Shading here is a per-face flag, so a face is smooth only when it has no sharp edge
at all — and every side quad of a cylinder meets a cap at a right angle. That is also the right
answer for this renderer rather than a gap: smoothing the band while the caps stayed flat would
average the cap normals into the rim and round the very edge the caps are there to define.

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
the object it was made against after you reopen it. It also keeps the **camera**, so reopening a
document puts you back where you were looking rather than framing it afresh. A file written before
that key existed still opens, and simply gets framed.

## The two ways out

Clay has two output paths and they do genuinely different things. Choosing between them is the
whole reason both exist.

**Export to library** puts the *exact* geometry in the library as an ordinary asset. It is a
finished model row from the moment it lands, so it inherits everything the rest of the app does to a
mesh: rigging, posing, sprite sheets, the triangle retarget, and the STL, OBJ, FBX, collision and
texture exports. Use it when the shape you modelled is the shape you meant.

**Make 3D** renders the document flat, on a plain background, with no grid and no gizmos, and
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
editing it would drop the rig; open it in Create instead. And a mesh past two million triangles,
because the editor holds every mesh twice per undo step. Past two hundred thousand it asks first,
since every edit rebuilds the whole mesh and you should know that before you press Extrude.

## Where the files go

An exported asset is an ordinary job directory, and the document that produced it is stored beside
the mesh as `build.wblk` (the on-disk name predates the rename). That copy is never served or
downloadable — it exists so that reopening a built asset brings its objects back instead of one
frozen mesh — and it goes away with the job when the job is deleted.

Dropping a `.wblk` on the window while Clay is on screen opens it, and dropping a `.glb` imports it
— see [Importing an asset](#importing-an-asset).

Every binding is listed in [Keyboard shortcuts](14-shortcuts.md).
