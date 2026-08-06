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
[Rerun and promotion](08-library-and-jobs.md#rerun-and-promotion).

If the reference's own quality report says it may not reconstruct, promoting it opens a confirm
naming the reasons, with **Build anyway** as the affirmative. It is a confirm rather than a refusal
because the rules are heuristics about composition — what you must not do is spend two minutes of
GPU by accident.

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
| Mobile / VR | 512 |
| Indie desktop | 1024 |
| Hero asset | 1536 |

This is a different control from the 2D pane's platform, which is a phrase in the prompt. Higher
resolutions cost more VRAM and more time; 1536 in particular may need the exclusive VRAM mode
(`WARLOCK_VRAM_EXCLUSIVE=1`), which stops the reconstruction engine while the image model runs.

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
skeleton picker. See [Rigging and posing](04-rigging-and-posing.md).

## Triangle budget

The reconstruction is kept, permanently, as `source.glb`, and nothing ever overwrites it. The
`model.glb` you actually use is derived from it by optimising and then grounding. That is what makes
changing your mind about triangle count cheap: rebuilding at a different budget is a couple of
seconds of mesh processing rather than another two minutes of reconstruction.

The control is in the inspector, on the **Rig & Pose** tab, under the collapsed **Triangle budget**
header. It appears only on jobs that have a `source.glb` — older jobs and rig jobs do not.

Five tiers exist in the code: Raw (full density), Draft (20k), Standard (50k), Detailed (100k) and
Custom. **Today only Raw is offered.** Every decimating tier needs `gltfpack`, a vendored binary
that is not yet shipped, and the panel says so on screen rather than presenting a button that can
only fail. When the binary is present, the whole list appears and Custom gains a triangle-count
field with its own valid range.

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

The **mesh audit** answers a different question: *can you see through it*. It is a silhouette check
— render the mesh from several angles and measure how much of the subject is holes — reported as
"visible openings" and a percentage. That is what a player actually notices, and it is not the same
property as watertightness at all. A mesh can be watertight and still look wrong, and vice versa.

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
[2D exports](02-generating-references.md#2d-exports).

For bulk export of several assets at once, and for the storage those files occupy, see
[The library and jobs](08-library-and-jobs.md#storage-and-pruning).
