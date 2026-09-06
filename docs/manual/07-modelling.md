# Modelling

Clay is a polygon modeller: primitives, element editing, booleans and materials, saving to its own
`.wblk` format and exporting into the library like anything else.

It needs no GPU and no weights. It is also the answer to a question the generators raise — what to
do when reconstruction gets a shape *nearly* right, or when the thing you want is a crate and asking
a diffusion model for a crate is the long way round.

## Getting a document

`Ctrl+N` starts one, `Ctrl+O` opens a `.wblk`. A finished mesh in the library has **Open in Clay** in
its menu, and a `.glb` dropped onto the window is imported.

Two limits on import, both about memory rather than taste: past about 200,000 triangles Clay asks
first, and past two million it refuses. The editor holds two copies of a mesh per undo step, so a
model that fits comfortably in a viewer does not necessarily fit comfortably in an editor.

A rigged `.glb` is refused outright. Clay has no skinning, and silently dropping the rig on import
would be worse than saying so.

## Primitives

Twelve, in two groups. The primitives: box, plane, cylinder, cone, UV sphere, icosphere, torus,
capsule and grid. The structures — shapes you would otherwise build out of several primitives —
pyramid, arch and column. Place one and its
parameters — radius, height, segments — stay live in the properties panel, so a cylinder can become
a thinner cylinder without being rebuilt by hand.

**Until it freezes.** The first edit that changes topology — an extrude, a bevel, a dissolve —
discards those parameters permanently, and the panel switches to a plain vertex and face count. It
has to: once you have extruded a face, there is no "regenerate this cylinder" that could keep your
extrusion.

Nothing warns you, because it is not a mistake. But it is the reason a cylinder whose height you
wanted to tweak *after* modelling on it cannot be tweaked.

## Selecting and transforming

`4`, `1`, `2`, `3` switch between Object, Vertex, Edge and Face modes. Selection is not undoable, on
the reasoning that clicking a different object should not dirty a document.

The tools are `Q` select, `W` move, `E` rotate, `R` scale. During a drag you can press `X`, `Y` or
`Z` to lock to an axis and *type a number* to set the amount exactly — the two compose, so `X` then
`2` moves two metres along X. `Esc` cancels the drag with nothing recorded.

One undo step per drag, committed on release, not one per mouse-move.

Two snapping switches, independent: snap to grid, and snap to vertex. Snap-to-vertex is what you
want for making things actually touch.

**Proportional editing** gives a selection a soft falloff, measured from the nearest selected vertex
rather than from the centre of the selection — which is what makes it behave sensibly when you have
selected a whole edge rather than a point.

Camera: `Alt`-drag always orbits, `Ctrl+1`, `Ctrl+3` and `Ctrl+7` snap to axis views (add `Shift` for
the opposite side), and `Ctrl+5` toggles orthographic.

## Editing the mesh

The mesh operations live in one registry, which is why the context menu, the tools column and the
keyboard always offer exactly the same list. Right-click is the easiest way in.

The ones you will use: **Extrude** (`E`), **Inset Faces**, **Bevel Edges**, **Loop Cut**, **Bridge
Loops**, **Subdivide**, **Smooth**, **Fill Hole**, **Weld**, **Collapse**, **Dissolve**, **Flip
Normals** and **Delete**.

## Merge versus Union

Two operations turn several objects into one, they sit on adjacent shortcuts, and they do different
things. This is worth doing once by hand rather than reading about.

**Merge Objects** (`Ctrl+M`) welds them into a single object. Vertices within a distance of each
other are joined. Everything else survives — including, if the shapes interpenetrated, the interior
walls now buried inside your model, which will z-fight and which no light will ever reach.

**Union Objects** (`Ctrl+Shift+M`) is a boolean. It removes the interior. What it costs is stated in
the operation's own hint rather than discovered afterwards: **UVs are lost** (no corner
correspondence survives a recut), n-gons become triangles, and **every input must be a closed
volume** — an object with a hole in it is refused by name.

Union originally shipped with no keyboard shortcut at all, and the result was that people found the
weld, hit the z-fighting, and never discovered the boolean existed. Hence the shifted spelling of
the same chord: same question, answered the other way.

## Checking a mesh

**Check mesh** in the properties panel runs on request rather than continuously, and reports holes,
non-manifold edges, inconsistent winding, duplicate faces and unused vertices. Each finding is
clickable and selects the offending elements for you.

It is a report, not a verdict. An open sheet is a perfectly valid mesh; it is only a problem if you
wanted a solid. Booleans, on the other hand, genuinely do require closed volumes, so this is the
panel to visit when a Union refuses.

## Materials and UVs

One material palette per document, with slots referenced per face. Base colour, metallic and
roughness are editable. Texture slots are **read-only**: Clay carries baked maps through import,
the viewport, `.wblk` and export without damaging them, but it does not paint them. That is Inker's
job.

Adding a material always appends and never inserts, because inserting would renumber every face
assignment in the document. Removing one is only allowed when nothing uses it.

Every primitive already has sensible UVs. **Box Unwrap** re-projects an object planar-per-face by
dominant axis — quick and not conformal, which is the right trade for a blockout and the wrong one
for a hero asset.

**Shade Smooth**, **Flat** and **Auto** control normals. Auto splits on an angle. Be aware that a
capped cylinder shades entirely flat under some settings, which looks like a bug and is the angle
rule doing what it was told.

## The two ways out

This distinction matters as much as Merge versus Union.

**Export to the library** mints an ordinary finished asset from your exact geometry. Nothing is
reinterpreted. It unlocks everything downstream — rigging, posing, sheets, every export format —
none of which needs to know Clay exists.

**Make 3D** renders your model flat and feeds that render to the reconstruction engine as a
reference image. What comes back is a **different mesh** with surface detail nobody modelled. It is
not a higher-fidelity export of your blockout; it is a generation that used your blockout as the
prompt.

One consequence: an asset built in Clay can never be rerolled or remeshed, because there is no
reference image behind it. The app knows and the buttons are absent rather than broken.

Note also what is *not* in Clay: there is no decimate or retopology operation. Triangle budgets are
applied downstream, on the finished asset, in the retarget panel.

## Round trips

Export to the library, then use **Open in Clay** on the resulting card, and you get your objects back —
names, generator parameters and all. That works because the export keeps a `.wblk` beside the mesh.
Opening a mesh that was *not* authored in Clay instead gives you one frozen object per material,
which is the honest answer to "what were the objects in this file" for a file that never had any.

## Try it

1. `Ctrl+N`, and place a box and a sphere so they overlap.
2. Run **Merge Objects** with a small weld distance, orbit into the overlap, and find the interior
   wall.
3. Undo, run **Union Objects** on the same pair, and confirm the interior is gone.
4. Now delete a face from one object and try Union again — read the refusal, run **Fill Hole**, and
   retry successfully.
5. In Face mode, select the top of a box, **Inset**, then **Extrude** upward to make a chimney —
   and watch the properties panel drop the box's parameters the moment the inset lands.
6. **Export to the library**, then **Open in Clay** from the new card, and confirm your objects came
   back.

## What to read next

[Rigging and posing](08-rigging-and-posing.md) — giving a mesh a skeleton, and the two traps that
catch everyone.
