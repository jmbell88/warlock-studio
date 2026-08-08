# Pipelines

A job is a row in the store and a directory on disk, and what happens between the two is this
chapter. It follows one asset from the text you typed to the files an engine can import, and it
explains the orderings that look arbitrary from outside and are not.

## From prompt to reference

A reference job composes a prompt, encodes it and denoises an image. The composition happens in
`guidance.py` and `pipelines/prompt.py`, both of which are pure and torch-free — you can see the
result before spending any GPU on it, which is what the prompt preview in the 2D pane shows.

Three things go into the composed prompt, in this order: any trigger words the chosen style LoRA was
trained on, then your own text with the guidance fragments appended in a fixed order that reads like
a sentence, then the framing clause of the prompt template — one object, plain background, no
cropping — that biases the image toward something the reconstruction engine handles well. The
guidance fragments are two to four words each. That is not a limit imposed by the encoder any more,
but a longer fragment still dilutes cross-attention across everything else you asked for.

Then the encoding, which is where the interesting constraint lives. CLIP's text encoders accept 77
tokens — a begin marker, up to 75 content tokens and an end marker — and with a dozen optional
guidance fields the composed prompt routinely exceeds that. Truncating would silently drop whatever
came last, which is the guidance you chose most deliberately.

So `prompt.chunk()` splits the composed text into as many 77-token chunks as it needs, always on a
comma boundary. Commas are safe because every join in the composition uses `", "`, so a break never
lands mid-phrase. Packing by phrase rather than by raw token slice also guarantees that SDXL's two
text encoders — CLIP-L and CLIP-G — produce the *same* chunk count for the same text, which matters
because their hidden states are concatenated per chunk on the feature axis; a mismatch would
misalign every chunk after the first.

Each chunk is then encoded separately and the results concatenated on the sequence axis. The UNet's
cross-attention has no sequence-length limit — only the text encoders do — so N encoded chunks are
simply a longer conditioning sequence, which is the same workaround the wider ecosystem uses. The
pooled embedding cannot be concatenated, being one vector rather than a sequence, so it is taken
from the second encoder's output on the first chunk only, matching what the stock pipeline does.

A prompt that fits in one chunk — the common case — takes a path that is bit-identical to the old
direct-string one. Chunking removed the hard ceiling, not the soft one. See
[The prompt](03-generating-references.md#the-prompt).

## From reference to mesh

The reconstruction engine writes one file, `source.glb`, and nothing ever overwrites it. Everything
else about the mesh is derived from that file, which is what makes rebuilding at a different
triangle budget cost seconds instead of another two-minute reconstruction.

`model.glb` — the file you actually export and preview — is produced by optimising `source.glb` and
then normalising the result. That order is load-bearing in one direction. The optimiser rewrites the
node graph, and normalisation inserts a node carrying the scale and translation, so optimising after
normalising would throw the grounding transform away. A retarget at a new budget therefore reapplies
the transform itself, for exactly that reason. See
[Triangle budget](04-generating-meshes.md#triangle-budget).

Normalisation does three things and only the first is conditional. It scales the mesh, if and only
if you asked for a real-world size. Then it centres the model in X and Z and puts its lowest point
at Y equals zero, on *every* job — because a pivot at the reconstruction volume's centre is a manual
fixup on every Godot or Unity import regardless of whether a size was requested. The measurement
goes through a mesh library but the write does not: only the glTF JSON chunk is rewritten, inserting
one node below the scene root, so buffers and textures are copied through byte for byte rather than
re-encoded.

The transform node goes *below* the scene root rather than on it, because a scene root's transform
is the one thing several importers and mesh libraries discard.

Because normalisation now runs on meshes it previously never touched, a failure there is logged and
swallowed rather than failing the job — the GLB is already on disk and usable, and the same rule
applies to the mesh audit and the mesh report.

A retarget refuses a job that is still queued or running, with a conflict rather than a wait. Its
half of the write takes a lock the worker's own optimise, scale and audit steps do not, and two
writers on one mesh is not a race worth having. A cancelled job has both `source.glb` and
`model.glb` deleted, not just the second — otherwise a cancelled reconstruction could be retargeted
back into existence.

## Derived artifacts

Everything except the two GLBs is a pure function of `model.glb`: `model.stl`, the OBJ zip,
`model.fbx`, `collision.glb` and `textures.zip`. None of them is produced when the job runs. Each is
built the first time something asks for it and then left on disk.

Two mechanisms make that safe. Each artifact of each job has its own lock, so a double-clicked
download button cannot start two conversions of the same file — and the lock is a plain thread lock,
not an asyncio one, because every caller is a pool thread and the protected work blocks anyway.
Keeping those waits off the event loop is the point.

And every write onto a file that is already being served is staged: the new bytes go to a uniquely
named temporary file in the same directory, which is then renamed onto the destination. A rename
within one filesystem is atomic on Windows and POSIX alike, so a concurrent reader sees either the
old complete file or the new complete file, never a truncated one. That matters most for a retarget,
which runs on a job that is already finished and whose `model.glb` the viewer may be reading at that
moment.

A retarget deletes every derived artifact, because each of them describes the old mesh. It does not
delete the rig — it reports that the rig is now stale, and the retarget control warns you before the
button rather than after. Destroying a rig that took minutes to solve, in order to change a triangle
count, is not a trade the app makes on your behalf.

## Pixel-art exports

The pixel exports are derived like everything above, but their processing is its own pure module
(`pipelines/pixel.py`) rather than part of the export code, because the bench measures with exactly
the same functions the export runs. Two implementations of "is this image on a pixel grid" would
drift, and the metric would then be measuring something the export does not do.

Three things happen there and the order is load-bearing. A grid is detected on the *whole frame* —
its cell size and its phase — because the lattice belongs to the generated image and any crop moves
the phase before it can be measured. If one is found, the frame is reduced at cell centres and only
then cropped to the subject; a cell's transparency is decided by majority coverage rather than by
its centre sample, since the centre of an edge cell is as likely to land just outside the subject as
just inside, and one wrong cell on a 32-pixel sprite is a visible bite. Palette mapping, if a
palette file was chosen, runs afterwards in Oklab, with alpha carried around it untouched.

Everything about that is off by default and byte-identical to the export that existed before it,
which is what a test in `tests/test_asset2d.py` pins: every asset already on disk was cut by the
crop-then-scale path, and a manifest claiming so is only true while that holds.

## Blender out of process

Rigging, pose baking and sprite-sheet rendering all need Blender, and Blender's Python module never
runs inside the app process.

There are two reasons and either alone would be enough. `bpy` is process-global and not thread-safe,
which is incompatible with a four-thread pool. And it hard-*crashes* rather than raising on some
non-manifold geometry — which is exactly what a reconstruction frequently produces. A crash in a
library that cannot be caught takes the window, the queue and the store with it.

So it runs out of process, as `python -m warlock.pipelines.blender_worker`, mirroring the pattern
the reconstruction engine already established. `rigging.py` is the host side and stays importable
with no Blender anywhere, which is why the app runs perfectly well on a machine with no `bpy`
installed and simply hides the rig controls. `pipelines/blender_worker.py` is the only module that
imports `bpy`, and it imports `rigging.fit_template` from the host side rather than reimplementing
it, so the two ends can never disagree about where a joint goes.

The threading rules are untouched by any of this: launching the worker is a blocking call, and every
caller dispatches it to the task pool.

A rig writes into the *source* job's directory, next to the `model.glb` it was fitted to, rather
than into the rig job's own — the rig belongs to the mesh. The worker never writes the served names
directly. It is pointed at temporary names, and those are renamed into place only on success, GLB
first so that `rig.json` — which is what the app treats as the completion marker — never advertises
a mismatched GLB. On a re-rig the temporary names are the whole point: the previous `rig.json`
already satisfies the completion check while Blender is still writing, so without them a half-built
rig would look finished. A cancelled rig deletes only the temporaries, leaving any previous
successful rig intact.

Blender's glTF importer also invents objects worth knowing about. It parks bone-shape widgets for an
imported armature in a collection excluded from the view layer, so nothing renders and nothing
re-exports — but the objects are still in the scene, and a unit icosphere among them is enough to
treble a computed bounding box and frame every rigged sprite sheet's subject at a third of its size.
Every import in the worker is followed by a purge of those helpers.

## The pose contract

A pose is a map from joint name to a quaternion, and that is the entire contract between the viewer
and Blender. The viewer never sees a Blender bone; the worker never sees a viewer joint node. They
agree because both talk about the same thing: the joint node's *local* rotation in the glTF sense.

The identity behind that is one line of maths. Blender composes a posed bone as
`pose(b) = pose(parent) @ (rest(parent)^-1 @ rest(b)) @ basis(b)`, and a glTF joint's local
transform is `pose(parent)^-1 @ pose(b)`. Substituting one into the other gives
`node_local == rest_node_local @ basis` exactly — so the worker recovers Blender's pose basis as
`rest_node_local^-1 @ node_local` and nothing else. There is no per-bone axis correction anywhere,
and the Z-up to Y-up conversion cancels because the exporter applies it once at the root rather than
per joint.

Two flags keep that identity true, and both are non-default:

- The worker imports with the bone heuristic set to `BLENDER`. The default heuristic re-aims bones
  at their children for editing comfort, which changes the very rest matrix the identity is
  reconstructed from — a re-aimed bone silently rotates the pose.
- The worker exports with rest-position armature output disabled. The default writes every joint at
  its rest pose and demotes the actual pose to an animation track, which produces a file that looks
  correct in a timeline and unposed in a still.

Quaternions are stored in XYZW order, which is what the browser viewer's array conversion produced,
kept because every pose already on disk is in it; the worker converts to Blender's WXYZ on the way
in. Mirroring a pose is a single shared function rather than two implementations, because the sign
convention is exactly the kind of error a still image cannot show.

Poses are files, not rows: one JSON file per pose in the job's `poses/` directory, with its baked
GLB beside it. One file per pose means saving needs no read-modify-write and therefore no lock.
A caller-supplied pose id becomes a path in exactly one function, which rejects anything that is not
twelve hexadecimal characters. Overwriting a pose deletes its cached GLB, because that bake depicts
the rotations you just replaced. The bake itself is derived on demand under the same per-artifact
lock as the mesh exports rather than being queued: it is roughly a one-second subprocess, and
putting it behind the serial GPU queue would make it wait on a reconstruction. See
[Posing](05-rigging-and-posing.md#posing).

## Sheet planning

A sprite sheet's grid is decided on the host and never in Blender. `pipelines/sheet.py` is pure:
`plan()` works out which cells exist, `pack()` composites the rendered frames with Pillow, and
`sidecar()` writes the JSON. Blender's only job is to render one square transparent PNG per cell
into a scratch directory.

That split is what makes the grid testable without a GPU, and it is why the in-app preview, the
renderer and the sidecar cannot disagree about what a cell contains — there is one planner and all
three read it. The preview's own camera maths has to match the renderer's, down to the constants:
yaw zero puts the camera on `+Z` in the exported Y-up frame, and the framing extent is
`max(hypot(sx, sz), sy) * 1.12` — the same twelve percent margin the renderer frames with.

Three details are easy to undo by accident:

- The camera is framed **once**, from the rest pose's bounding box. Reframing per pose makes the
  subject jump in size between rows.
- The sidecar carries a flat list of cells rather than a nested grid, so an animated clip becomes
  more cells with a frame index above zero instead of a new sidecar format.
- Cells arrive grouped by row, so the worker re-poses once per pose rather than once per frame.

Column zero is the front view. Yaw zero looks along `+Y` in Blender, and every skeleton template
puts the subject's forward direction at `-Y`. See [The grid](06-sprite-sheets.md#the-grid) and
[The sidecar](06-sprite-sheets.md#the-sidecar).
