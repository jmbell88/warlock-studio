# Generating meshes

The mesh stage is where the reconstruction engine runs. It costs roughly two minutes of GPU per
attempt, so everything in this chapter is arranged around deciding what to spend that on. All of it
lives in the **3D asset** mode, whose settings pane holds no prompt controls whatsoever.

## Starting from a reference

The normal path is promotion: take a finished 2D asset and run the mesh stage from its image.

Select a finished reference in the library — its card offers **Make 3D**, and selecting a
promotable reference also makes it the 3D pane's source automatically. The **Source** section at the
top of the 3D pane names what the job will start from, and **Make 3D** at the bottom submits it.

The new job is an ordinary image job whose input image is the reference's, recorded as a child of
the reference so the library can show them as one lineage rather than two unrelated rows.

Everything the 3D pane holds is an **override**. Omitting one means "keep what the reference
recorded", which is not the same as sending the reference's value back — so the selects offer
"keep the reference's" as their first entry and that is where they start.

Two settings are exceptions and are always sent explicitly: the rig checkbox and the
normalise-the-reference checkbox. Both are the 3D pane's own decisions, and an omission would let
the promotion inherit whatever the reference happened to record.

Derived values never carry across. Anything the worker recorded about the *source* run's artifacts
— the composed prompt, the mesh report, the applied transform — is stripped, so a new job never
wears a quality verdict about a mesh that does not exist yet. See
[Rerun and promotion](09-library-and-jobs.md#rerun-and-promotion).

## Checking the cutout

**Make 3D** does not submit straight away. It opens **Check the cutout**: the subject cut out of the
reference, drawn over a checkerboard so transparency reads as transparency, with the panel naming
which matte produced it — the reference's own alpha, the BiRefNet model, or the corner fill that
answers when BiRefNet's weights are not installed.

The matte is the single decision that most often turns a good reference into a solid slab, and it
used to be made inside the reconstruction engine two minutes after you had committed. The panel is
where you see it first. It also carries the reference's own quality report: the reasons it may not
reconstruct, and the milder warnings — edge contact, a very thin subject — that are worth knowing
before the spend rather than after.

Three buttons:

- **Accept** queues the mesh job. When the report refused the reference the button reads
  **Build anyway** instead, and it submits with the refusal overridden — a confirm rather than a
  refusal, because the rules are heuristics about composition and you can see the image they are
  arguing about. What you must not do is spend two minutes of GPU by accident.
- **Fix matte** opens the reference in Inker with the cutout already folded into its alpha, as one
  undoable step. The eraser and the brush then edit the matte directly; see
  [Inker](07-inker.md#fixing-a-matte).
- **Cancel** leaves everything as it was.

A matte you edited and saved travels to the engine as the image's own alpha, and the job records
that it was approved — the engine is told to keep the alpha rather than cut its own.

## Candidates

The reconstruction engine is deterministic in its seed, and its failure mode is a lottery: the same
reference comes back clean at one seed and with a hole through the shoulder at another. **Candidates**,
directly above **Make 3D**, is how many attempts one press buys — 1, 2 or 3. The cost line under it
changes with the choice, because this is the one control in the pane that multiplies what the button
spends.

Each candidate is an ordinary mesh job: same validation, same VRAM admission, same worker. The first
keeps the mesh seed you pinned, so a pinned seed still reproduces; the rest draw fresh ones.

While a group is undecided its members are **hidden from the library** — three near-identical cards
are not a workshop — and the **Candidates** picker at the top of the 3D inspector is where they live
instead. Selecting one shows it in the viewport exactly as selecting any other asset does. Once every
attempt has finished, **Keep this one** settles the group: the one you kept and the ones you did not
all become ordinary assets, and only then are you *asked* whether to delete the ones you did not keep.
Nothing is ever deleted on your behalf, and declining leaves you with ordinary assets rather than
hidden ones.

Verdicts work on a candidate like any other mesh, so judging the group feeds the same findings pool.
See [Review](11-review.md).

The count applies to **Make 3D** only. An upload queues one mesh job, as it always has.

## Starting from an upload

You can skip the reference stage entirely. Press **Open an image...** in the **Source** section, or
drop an image file onto the window, and the app queues a mesh job directly from it.

Uploads are bounded at the door, and both limits are checked before anything is written:

- **20 MB** on the file itself, checked before the image is decoded.
- **16 megapixels**, read from the image header before any pixel is decoded — a flat 20-megapixel
  PNG is a few hundred kilobytes on disk and hundreds of megabytes decoded.

Anything larger is refused rather than allocated. Whatever format you supply is re-encoded to PNG,
since the reconstruction engine only decodes PNG and JPEG; transparency is preserved when the
source had it, because a pre-matted upload lets the engine's background detection do less work.

A good upload is the same thing a good generated reference is: one complete subject, not cropped,
on a plain background.

## Mesh parameters

The **Mesh** section holds the reconstruction settings.

**Detail** is the geometry resolution handed to the engine, supplied by a platform preset:

| Platform | Resolution |
| --- | --- |
| 2D | 512 |
| 3D | 1024 |

The question the select is asking is what the asset is *for*: a 2D asset is going to be seen flat
and small, a 3D one in a scene. This is a different control from the 2D pane's platform, which is a
phrase in the prompt. Higher resolutions cost more VRAM and more time, and on a card that cannot
hold both models at once they may need the exclusive VRAM mode (`WARLOCK_VRAM_EXCLUSIVE=1`), which
stops the reconstruction engine while the image model runs.

**Budget** picks the triangle-reduction tier. Only "Raw (no decimation)" is offered here — see
[Triangle budget](#triangle-budget) for why, and for where a budget can actually be chosen today.

**Size (m)** is the physical size the finished GLB is scaled to, along its largest dimension, from
0.01 m to 100 m. Zero means "keep whatever the reference recorded", which in turn falls back to the
category's typical size, or 1 m if no category was chosen.

Scaling is optional, but **grounding is not**. Every finished mesh is centred on X and Z and has its
lowest point put at Y = 0, whether or not a size was asked for. A pivot sitting at the centre of the
reconstruction volume is a manual fixup on every Godot or Unity import, so the app does it for you
on every job.

**Background** chooses how the engine mattes the input image: `auto`, `birefnet` or `threshold`.
`auto` is the default and is right almost always; the other two exist for images `auto` gets wrong.

**Mesh seed** is the reconstruction's own seed, separate from the image seed, with its own **Reroll**
button. Leave it at zero to let the job pick one.

**Normalise the reference** recentres the subject and scales it to fill the frame before the engine
sees it. It is off by default: the engine does its own cropping, and whether doing it twice helps or
hurts has not been measured. Treat it as an experiment rather than an improvement.

The **Rig** section, present only when Blender is installed, holds **Rig when the mesh lands** and a
skeleton picker. See [Rigging and posing](05-rigging-and-posing.md).

## Triangle budget

The reconstruction is kept, permanently, as `source.glb`, and nothing ever overwrites it. The
`model.glb` you actually use is derived from it by optimising and then grounding. That is what makes
changing your mind about triangle count cheap: rebuilding at a different budget is a couple of
seconds of mesh processing rather than another two minutes of reconstruction.

The control is in the inspector, on the **Rig & Pose** tab, under the collapsed **Triangle budget**
header. It appears only on jobs that have a `source.glb` — older jobs and rig jobs do not.

Five tiers exist in the code: Raw (full density), Draft (20k), Standard (50k), Detailed (100k) and
Custom. `gltfpack` — the vendored binary every decimating tier runs through — is present now, so
this panel offers the whole list, and Custom gains a triangle-count field with its own valid range.
**The generate form still offers Raw alone**, because none of the decimating tiers has been
qualified yet: a tier is only exposed there once it has been run against a chest, a sword and a rock
and shown to keep UVs, both PBR maps and material assignment. This panel is where that qualifying
happens, on a mesh that already exists rather than on a job you are about to wait two minutes for.

Two things the panel will not hide from you. A retarget refuses to run on a job that is still queued
or running, because its write would collide with the worker's. And a retarget makes a rig, its saved
poses and its rendered sheets describe a mesh that no longer exists — those are minutes of your
work, so they are reported rather than deleted, and the warning is shown *before* the button rather
than after. Everything else derived from the mesh (STL, OBJ, FBX, collision, textures) is deleted,
because those describe the old geometry exactly.

Press **Rebuild mesh** to apply. The served `model.glb` is never written in place: the new mesh is
staged and swapped, so nothing reading the old one sees a truncated file.

## Mesh audit and mesh report

The app measures a finished mesh in two deliberately separate ways, and they answer different
questions. Both appear in the inspector's **Details** tab under **Mesh quality**.

The **mesh report** answers *will an importer accept this, and will it sit on the floor*. It is the
topology-and-metadata check: triangle count, material count, whether the surface is **watertight**,
and whether the pivot is at the model's feet. It also carries the pass/fail badge and its reasons.
Only this measurement may use the word watertight, because only this one proves it.

The watertight figure is measured on a **welded** copy of the mesh — vertices at the same position
merged first. The file itself is read unwelded, because the UV and material checks need to see the
split vertices, but a UV atlas splits a vertex at every seam and each of those splits reads as a
boundary edge. Unwelded, the check was mostly counting texture seams and calling almost every mesh
open. When the report says a mesh is not watertight it names the boundary edges and components it
found *after* welding; the raw unwelded counts are still recorded, because how badly a file is split
is its own question for a rig or an exporter.

The **mesh audit** answers a different question: *can you see through it*. It is a silhouette check
— render the mesh from several angles and measure how much of the subject is holes — reported as
"visible openings" and a percentage. That is what a player actually notices, and it is not the same
property as watertightness at all. A mesh can be watertight and still look wrong, and vice versa.

**Read that percentage in one direction only.** A high reading means a hole, and it means it
reliably. A *low* one means no hole was seen, which is not the same as a good mesh — the most common
way reconstruction fails is a solid, featureless slab, and a slab has no openings at all. Measured
against 84 reviewed meshes, the accepted ones had *more* visible openings than the median discarded
one, so a near-zero reading is close to no information. The app says so where it shows one: nothing
in the interface paints a low figure as a pass, and the inspector adds "a solid, featureless mesh
scores this too" underneath it.

Neither measurement can fail your job. If either cannot be computed, the failure is logged and the
job still completes: the GLB is already on disk, and a missing verdict is better than a lost mesh.

## Exports

The inspector's **Export** tab lists everything you can take away, as a two-column grid of buttons:

| Button | File | Notes |
| --- | --- | --- |
| GLB | `model.glb` | The finished asset: optimised, grounded, textured. |
| Source GLB | `source.glb` | The raw reconstruction at full density, before optimisation and grounding. |
| STL | `model.stl` | Geometry only. |
| OBJ (zip) | `model_obj.zip` | OBJ plus its material and texture files. |
| FBX | `model.fbx` | Needs Blender; the button says so when it is missing. |
| Collision | `collision.glb` | A simplified collision shape. |
| Textures | `textures.zip` | The texture images on their own. |
| Rigged GLB | `rig.glb` | Present once the mesh has been rigged. |
| Reference image | `input.png` | The picture the mesh was reconstructed from. |

Only `model.glb` and `source.glb` come out of the job itself. Everything else is a pure function of
`model.glb` and is produced the first time you ask for it, then cached — which is why the first STL
of a large mesh takes a moment and the second is instant. Rebuilding the mesh at a new triangle
budget deletes all of them, since they describe the old geometry.

A button you cannot press keeps its place and explains itself in a tooltip: "needs Blender" for FBX
without Blender installed, "not available for this asset" for a mesh export on a plain reference
job. A missing button would be a mystery; a disabled one with a reason is information.

This table is the *mesh* half. A finished reference has its own Export tab offering the cutouts,
the pixel-art reductions and the manifest — see
[2D exports](03-generating-references.md#2d-exports).

For bulk export of several assets at once, and for the storage those files occupy, see
[The library and jobs](09-library-and-jobs.md#storage-and-pruning).
