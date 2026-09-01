# Your first asset

This chapter makes one 3D model from one sentence, and explains the thing about Warlock's pipeline
that surprises nearly everyone: **a prompt does not produce a mesh.** It produces a picture, and
stops, and waits for you.

That is not a limitation to work around. It is the single most important design decision in the app,
and understanding it in the first ten minutes will save you a great deal of confusion later.

You need SDXL and the reconstruction engine installed for the whole of this chapter. If they are not
there yet, read it anyway — the forms all work, the refusals are informative, and
[Before you begin](01-before-you-begin.md#what-works-before-the-downloads-finish) says which parts
you can drive today.

## Why there are two stages

Reconstructing a mesh costs roughly two minutes of GPU per attempt. The single biggest factor in how
good that mesh turns out to be is the picture it was made from — the engine can only be as good as
the image it is handed.

So the pipeline is split, and the split is visible in the app rather than hidden inside it:

1. **The reference stage** draws an image and stops. It takes a few seconds, and you can ask for
   several candidates from different seeds at once.
2. **The mesh stage** runs only once you have looked at a reference and approved it.

A text job never falls through into a mesh by accident. Going straight from a prompt to a mesh would
spend two minutes of GPU on an image nobody had looked at, and most of the time you would throw the
result away because the picture was wrong, not because the reconstruction was.

So: if you type a prompt, press Generate, and then wonder why no 3D model appeared — nothing is
broken. You are at the end of stage one, and stage two is a button.

## Stage one: the reference

Open **Create** in the rail on the left. The window splits into three: settings on the left, a
viewport in the middle, and the library on the right. Above the settings is a row of five
segments — Reference, Mesh, Rig, Pose, Export — which is the path an asset takes through the app.
You are on Reference.

Three controls matter for a first run, and everything else has a sensible default.

**Asset type.** Leave it on *3D Model*. This one choice quietly sets several things at once, which
is why it is a single combo rather than three: what the job produces, how the prompt is composed,
and which follow-up work is offered. The other four entries make plain images, seamless materials,
tilesets and sprite sheets.

**Description.** The prompt. Write a subject, not a scene:

```text
a mossy stone well, weathered, fantasy game prop
```

One object, described plainly. The reference stage is trying to draw a thing a reconstruction engine
can turn into a model, and a picture with two objects in it produces a mesh with two objects fused
into one. Composition matters more than adjectives here.

**Model.** Leave it on *Automatic*, which means the app picks the checkpoint and tells you which
one underneath. For a first asset it resolves to SDXL 1.0 at full CFG — thirty steps, the default
because it was measured against the alternatives rather than chosen for speed. Name a checkpoint
yourself when you want a specific one; the four-step entries are for hunting, not for a picture you
mean to reconstruct.

Press **Create image** — the button in the bar, whose label names whatever you are making. The job appears in the
library on the right, immediately, as a queued row with a
progress bar. A few seconds later it is a picture.

## Looking at it

This is the moment the two-stage pipeline exists for.

Select the finished row and the reference fills the viewport at full size. What you are judging is
not whether it is pretty — it is whether it will *reconstruct*. That means: one subject, complete
and not cut off by the frame, filling a decent part of it, on a plain background.

The app forms its own opinion of the same question and records it. If it thinks the image will not
reconstruct, the inspector's **Reference** section says so. Those rules are heuristics about
composition, and you can see the picture they are arguing about, so treat them as a second opinion
rather than a verdict.

If you do not like it, press **Reroll** for another seed, or edit the prompt and generate again.
This is the cheap half of the pipeline; spend time here rather than on the expensive half.

If the image is *nearly* right, you can fix it by hand instead of rerolling: **Open in Inker** on
the viewport toolbar opens the reference as a layered drawing, and saving writes it back in place.

## Stage two: the mesh

When the picture is right, press **Make 3D** on the card.

A panel opens showing a **cutout** — the subject with its background removed. This is not a preview
of the model; it is the actual image the reconstruction will be run against, and it is shown before
anything is spent because the cutout is where most bad reconstructions come from. A halo of leftover
background becomes geometry. A subject with its feet cut off reconstructs without feet.

Four ways out of that panel:

| Button | What it does |
| --- | --- |
| **Accept** | Queue the reconstruction against this cutout. |
| **Build anyway** | The same thing, shown instead of Accept when the composition gate has an objection. It is styled as a destructive action because you are overriding a refusal, not dismissing a warning. |
| **Fix matte** | Change how the background was removed and look again. |
| **Cancel** | Nothing is spent. |

Press Accept. Now the two minutes happen.

## What comes back

When the mesh job finishes you get a model in the viewport that you can orbit, and rather more files
than you might expect. Two of them matter enough to name:

- **`source.glb`** is the raw reconstruction, exactly as the engine produced it.
- **`model.glb`** is derived from it — optimised, then re-centred, scaled and stood on the ground.

**Everything downstream uses `model.glb`.** Rigging, posing, sprite sheets, every export, and
"Open in Clay" all take the derived file, never the raw one. `source.glb` is kept because it is the
evidence — the thing every derived file can be rebuilt from — not because you are meant to use it.
Editing it directly would be invisible to everything else in the app.

The grounding step always runs, even if you asked for no particular size. A model whose pivot sits
in the middle of the reconstruction volume is a manual fix-up on every single import into a game
engine, so the app does it once, here, instead.

The rest — STL, OBJ, FBX, a convex-hull collision mesh, texture archives — are derived on request and
cached, so they cost nothing until you ask.
[Exports](23-generating-meshes.md#exports) lists them.

## If it went wrong

Reconstruction is not reliable, and a bad result is ordinary rather than exceptional. The two useful
responses:

**Remesh** runs the reconstruction again from the same reference with a fresh seed. Use it when the
picture was good and the mesh was not — geometry varies a lot between seeds.

**Reroll** goes back to stage one for a different picture. Use it when, looking again, the picture
was the problem. It usually was.

The inspector's mesh report carries measurements about what came back, and it is worth reading
[Mesh audit and mesh report](23-generating-meshes.md#mesh-audit-and-mesh-report) before trusting any
single number in it — several of them mean less than their names suggest, and the chapter says which.

## What you now know

- A prompt produces a picture, and the mesh is a second, deliberate press.
- The cutout is shown before the GPU is spent, because that is where reconstructions go wrong.
- `model.glb` is the one everything uses; `source.glb` is the evidence behind it.
- Rerolling the picture is cheap and remeshing is not, so judge the picture hard.

## What to read next

[Finding your work again](03-finding-your-work.md) — where all of this went, how to get back to it,
and the four different things the app means by "delete".
