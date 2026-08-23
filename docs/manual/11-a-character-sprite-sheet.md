# A character sprite sheet

Troupe makes animated character sprite sheets by building a 3D character, rigging it, animating it,
rendering it from eight directions and reducing every frame to pixels. Two hundred and fifty-six
cells from one description.

It is the most ambitious thing in the app, and it is the one where being straight with you about what
is proven matters most. This chapter says which half is solid, which half is untested, and which
features do not exist yet at all.

## How it actually works

Troupe is **four ordinary jobs and one human gate**, not a single orchestrated process. Knowing that
explains everything about how it behaves — including why it stops in the middle and waits.

1. **A reference.** One image job draws your character standing in a fixed pose. The pose is not
   hoped for: a guide figure is drawn and handed to the image model as structural conditioning.
   **A-pose is the default** and T-pose is the alternative — see
   [Making a character](33-troupe.md#making-a-character).
2. **The gate.** You approve that drawing in Create, exactly as in
   [Your first asset](02-your-first-asset.md). Nothing expensive runs until you do. This is the same
   reasoning as the main pipeline, for the same reason: the reconstruction is the costly step and it
   should not run on a drawing nobody wanted.
3. **The mesh.** An ordinary reconstruction.
4. **The rig.** Fitted automatically, with joints measured off the mesh rather than assumed —
   see [the A-pose trap](08-rigging-and-posing.md#the-a-pose-trap), which is exactly the failure this
   avoids.
5. **The sheet.** Every animation, from every direction, rendered and reduced to your chosen pixel
   size.

## The other door, and why you might prefer it

There is a second way in that skips the first three steps entirely: **Build another sheet** takes a
mesh that is *already rigged* and renders a sheet from it.

That door is worth knowing about for a practical reason. The sheet render itself is **Blender on the
CPU — no GPU, no VRAM, no model weights.** So if you bring your own rigged character, the whole of
Troupe's actual output is available to you on a machine that cannot run the generators at all. It
costs minutes of CPU and nothing else.

To use it: import your mesh, rig it in Poser against the humanoid template, then Build another sheet.

Your mesh needs to meet a contract the app cannot enforce, only state. GLB or glTF. T-pose or
A-pose — a dynamically posed mesh degrades both the joint fit and the automatic weights. **+Z up, −Y
forward.** Bone names mapping onto the nineteen-bone template if it is already rigged. **No very
short bones** — Blender deletes them silently along with their children, and fingers and toes are the
usual casualties. Under about 300,000 faces. And a licence that permits you to ship what comes out.

## What a sheet contains

By default, five animations across eight directions:

| Animation | Frames | Loops | Frame time |
| --- | --- | --- | --- |
| Idle | 4 | yes | 150 ms |
| Walk | 8 | yes | 100 ms |
| Run | 8 | yes | 60 ms |
| Attack | 6 | no | 80 ms |
| Jump | 6 | no | 100 ms |

Eight directions clockwise from front in 45° steps. Five animations by eight directions is 256
cells.

That layout is configurable. Each movement can be turned off or given a different frame count, and
the direction count can be 1, 4, 8 or 16. A sheet warns above 256 cells and refuses above 512.

Pixel sizes are 16, 24, 32, 48, 64, 96 and 128. Only 16, 32, 64 and 128 divide the render size
evenly; the other three go through a documented resize.

There are male and female builds, differing in shoulder width, arm length and the stance of the
guide figure.

One implementation detail with a visible consequence: each rendered frame is reduced to its final
pixel size *before* the cells are packed, not after. A 256-cell sheet packed at render resolution
would exceed the maximum atlas size, so per-frame reduction is the only route rather than an
optimisation. It also keeps the smooth resize used for previews away from your pixel art.

## Watching it

The preview plays the sheet. `Space` starts and stops; `Left` and `Right` step one frame and pause.
That is the entire keyboard — there is nothing else.

The preview is a clock rather than a frame counter, so it plays at real durations and loops rather
than falling behind. It draws at integer scale with nearest-neighbour filtering only, because a
pixel-art preview that resampled would be lying about the thing you are inspecting.

Troupe is the one workspace that holds no document. There is nothing to save and no undo stack;
entering it creates nothing. Sheets are ordinary library assets.

## Getting the sheet out

**Open in Inker** brings the sheet in as an animation with its tags already made, for hand cleanup.
**Add the sheet to Packwright** contributes one sprite per cell to an atlas.

That is the intended shape of the work: the pipeline produces frames that are close, and you fix
them by hand.

## What is proven, and what is not

Read this part before building expectations on top of it.

**Proven.** The whole chain runs. Every stage is real code with tests, and sheets have been rendered
from real meshes through real Blender. The supplied-base-mesh path in particular works today and
needs no GPU.

**Untested.** Reconstructing a *humanoid* from a single generated image has never been judged for
quality. The mechanism runs; whether it gives you clean limb separation and a good silhouette on a
character is an open question, not a promise. Related and permanent: reconstruction works from one
image, so **the back of a generated character is invented**, not observed.

**Provisional.** The shipped animation keyframes are placeholders — enough to prove the pipeline,
not finished animation. Expect to author your own in Poser's clip editor.

**Not built.** These do not exist, and no amount of looking will find them:

- Propagating a correction across frames, directions or animations. Fix a cell and you have fixed
  one cell.
- Mirror-assisted cleanup.
- Re-rendering one animation without discarding hand edits made to the others.
- Swappable or layered equipment.
- AI restyling of a rendered sheet, or a learned pixel refiner.
- Any animation beyond the five above.

The cleanup loop in particular is the honest gap: exporting is solid, and the workflow of *fixing a
256-cell sheet efficiently* is not built. For now, cleanup is Inker and patience.

## Try it

Without a GPU, with Blender installed:

1. Import a rigged character of your own, or rig one in Poser.
2. **Build another sheet** at 32 px. Note that it costs CPU minutes and no GPU at all.
3. Play it with `Space`, step through the walk with the arrow keys.
4. Open the sheet in Inker and fix one cell by hand.
5. Add the sheet to a Packwright atlas.

With a GPU and weights:

1. Describe a character and let Troupe draw the pose reference.
2. Approve it in Create — and notice that nothing expensive happened until you did.
3. Watch the mesh, rig and sheet stages go by in the in-progress list.
4. Judge the result honestly against the "untested" note above, especially from behind.

## What to read next

[Tuning what you get](12-tuning-what-you-get.md) — profiles, seeds, LoRAs and the other controls that
change what the generators produce.
