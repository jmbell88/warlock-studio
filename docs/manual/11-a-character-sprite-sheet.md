# A character sprite sheet

A character sprite sheet is a grid of one creature, animated, seen from eight directions and reduced
to pixels. Warlock makes them three ways, and this chapter puts the one that works first.

The short version: **Create → Character → describe the creature → Generate.** No graphics card, no
downloaded weights, no reference image to approve. Warlock builds the body itself, rigs it, animates
it from an authored clip library and renders the sheet. The other two routes — a mesh you already
have, and a reconstruction from a generated drawing — are further down, with what has been measured
about each.

## The route worth starting on

1. Open **Create** and set **Generation type** to *Character*.
2. Type a brief. `fire ogre, 3/4 top down sprite sheet` is a complete one.
3. **Read the column back.** It has filled itself in from your words: the species is *Ogre*, the
   look is *Fire*, the camera is *3/4 top-down*, and anything it did not understand is listed under
   **Not interpreted** rather than silently dropped. Change anything you disagree with; a control
   you touch becomes yours and stops following the prompt.
4. Press **Generate**.

What happens next has no gate in it. The body is built in the app, minted as a finished mesh asset,
and a rig is queued behind it; when the rig lands the sheet follows, and the finished character
plays in [Troupe](33-troupe.md). Create moves to the **Mesh** stage rather than the Reference stage,
because there is no drawing behind a character to look at.

**None of it needs a GPU.** The mesh is generated in-process, the rig and the render are Blender on
the CPU, and the pixel reduction is arithmetic. Rigging does need Blender installed — the press is
refused before anything is built if it is missing, because a body with no skeleton is half an asset.

### Why it works when reconstruction does not

Because nothing is being *recovered* from a picture. A character comes from a registry: **four body
plans** — humanoid, quadruped, winged, amorphous — and **thirty-one species** across them, each a row
of generator parameters with its own palette themes. The body plan decides the skeleton, which clip
library animates it, and which appearance sliders that creature has; an ogre and a wolf do not have
the same ones.

That is also the honest limit of it. The registry is a fixed vocabulary, and **Warlock never
substitutes**. Ask for a phoenix and it does not quietly hand you a dragon: the species stays empty,
Generate is refused, and the refusal says *"Warlock has no phoenix yet. The closest it makes is a
dragon"* with three presses under it — take the offer, switch to the experimental sprite-sheet type,
or take the brief to Troupe. The offer is always the same body plan, and taking it is your press.
[Generating references → Characters](22-generating-references.md#characters) has the whole of that
screen.

## What a sheet contains

A Character sheet defaults to three movements across eight directions:

| Animation | Frames | Loops | Frame time |
| --- | --- | --- | --- |
| Idle | 4 | yes | 150 ms |
| Walk | 8 | yes | 100 ms |
| Attack | 6 | no | 80 ms |

Eighteen frames in eight directions is 144 cells, at 64 pixels and 32 colours. Switch on the other
two — **Run** and **Jump** — and you have the full five, 32 frames per direction and 256 cells,
which is what Troupe's own form defaults to.

Eight directions clockwise from front in 45° steps. Each movement can be turned off or given a
different frame count, and the direction count can be 1, 4, 8 or 16. A sheet warns above 256 cells
and refuses above 512.

Pixel sizes are 16, 24, 32, 48, 64, 96 and 128. Only 16, 32, 64 and 128 divide the render size
evenly; the other three go through a documented resize.

One implementation detail with a visible consequence: each rendered frame is reduced to its final
pixel size *before* the cells are packed, not after. A 256-cell sheet packed at render resolution
would exceed the maximum atlas size, so per-frame reduction is the only route rather than an
optimisation. It also keeps the smooth resize used for previews away from your pixel art.

If your species carries a fire theme, the flame is composited in that same gap — after the reduce,
before the pack — so its oranges go through the same colour cut as the character's skin. *Troupe →
[Characters that are on fire](33-troupe.md#characters-that-are-on-fire)* is why.

## Watching it

The preview plays the sheet. `Space` starts and stops; `Left` and `Right` step one frame and pause.
`Up` and `Down` turn the character to the next direction and hold the frame you were on, `PageUp`
and `PageDown` move between animations, and `Home` and `End` jump to the ends of the run. That is
the entire keyboard.

The preview is a clock rather than a frame counter, so it plays at real durations and loops rather
than falling behind. It draws at integer scale with nearest-neighbour filtering only, because a
pixel-art preview that resampled would be lying about the thing you are inspecting.

Above the sprite is a **heatmap**: one square per cell of the selected animation, directions down
and frames across. A square goes amber where a frame's silhouette, position or colours jump from
the one before it, where a cycle's last frame does not meet its first, or where one direction has
drifted in size from the rest; red where it is well past that, or empty. Hover a square for the
numbers; click one and the preview jumps to that cell and stops. The scores rank cells for you to
look at and never reject a sheet -- nothing downstream reads them.

Separately from the heatmap, the sheet panel reports what the render **checked** about itself —
cells that came back clipped at the frame edge, cells that came back empty, and whether the sheet
needed a second, wider render to fit its poses. That is a structural note rather than a judgement,
and it never refuses a sheet either.

Troupe is the one workspace that holds no document. There is nothing to save and no undo stack;
entering it creates nothing. Sheets are ordinary library assets.

## Getting the sheet out

**Open in Inker** brings the sheet in as an animation with its tags already made, for hand cleanup.
**Add the sheet to Packwright** contributes one sprite per cell to an atlas.

That is the intended shape of the work: the pipeline produces frames that are close, and you fix
them by hand.

## The other routes

### A mesh you already have

**Build another sheet** takes a mesh that is *already rigged* and renders a sheet from it — no
species, no registry, your own model. It costs the same minutes of CPU and no GPU at all, so if you
bring your own rigged character the whole of Troupe's output is available to you.

To use it: import your mesh, rig it in Poser against one of the templates that has clips authored
for it, then Build another sheet.

Your mesh needs to meet a contract the app cannot enforce, only state. GLB or glTF. T-pose or
A-pose — a dynamically posed mesh degrades both the joint fit and the automatic weights. **+Z up, −Y
forward.** **No very short bones** — Blender deletes them silently along with their children, and
fingers and toes are the usual casualties. Under about 300,000 faces. And a licence that permits you
to ship what comes out.

A rig it arrives with is **discarded**, not adopted, so bone names do not have to match anything.
Warlock fits its own nineteen-bone skeleton, because a supplied rig is not evidence about where the
template's joints go — CesiumMan has nineteen bones like the template and still splits them
differently, three per arm and four per leg against the template's four and three. The mesh is
unbound and the old armature removed before a single measurement is taken.

This is the route where joints are *measured off your mesh*, and it is why
[the A-pose trap](08-rigging-and-posing.md#the-a-pose-trap) matters here and not on the Character
route — a built character ships its own joints, so there is nothing to guess.

### A reconstruction from a generated drawing

The third route is the original one, and it is in Troupe's own form: describe a character, let the
app draw a pose reference, approve that drawing, and let it be reconstructed into a mesh which is
then rigged and rendered. See [Making a character](33-troupe.md#making-a-character).

It is still there, and it is still the only route that will draw you a creature the registry does
not model. What it is not is reliable, and the next section is the measurement.

## What is proven, and what is not

Read this part before building expectations on top of it.

**Proven.** The whole chain runs. Every stage is real code with tests, and sheets have been rendered
from real meshes through real Blender. The supplied-base-mesh path in particular works today and
needs no GPU.

**Built, awaiting the render benchmark.** The Character route — the registry, the four body plans,
the rig, the clips, the render, the effects — is built, tested and structurally checked, and no
human has yet sat down and judged the pixels. That judgement is four separate verdicts, not one: a
convincing humanoid walk tells you nothing about whether a quadruped's four-beat gait reads at 64
pixels. Until that sitting happens this route is *built* rather than *proven*, and this sentence is
what will change, per body plan, when it does.

**Measured, and the answer is no — for now.** Reconstructing a *humanoid* from a single generated
image was judged on a graded corpus on 2026-08-30, and it came back with limbs bent and stretched.
The references themselves were fine; it is the mesh that is lost, because a character asks the
reconstruction for separable limbs from one view and it does not deliver them. That verdict is on
today's default reconstruction, not on the idea. Related and permanent either way: reconstruction
works from one image, so **the back of a generated character is invented**, not observed. What that
verdict has stopped meaning, since the Character route shipped, is "you cannot get a character from
a prompt" — you can, and the sentence you type is read rather than drawn.

**Provisional.** The shipped animation keyframes are placeholders — enough to prove the pipeline,
not finished animation. There are four libraries of them now, one per body plan. Expect to author
your own in Poser's clip editor.

**Built, for the cleanup.** Open the sheet in Inker and a strip under the transport sends one
correction to the same frame in every direction, to a whole direction, an animation or the sheet,
and offers a fix drawn on one side to its mirror with the face kept. *Inker -> Sheet corrections*
has the whole of it.

**Built, for the re-render.** *Re-render some runs* in the sheet panel rebuilds only the animations
and directions you tick and copies the rest from the sheet you started with, at that sheet's own
settings. The result is a new sheet, and Inker's *Merge re-render* brings it into the document you
have been cleaning up: untouched cells take the render, cells you painted keep your work, and cells
where both changed are flagged for you rather than overwritten. *Inker -> Merging a re-render* has
the whole of it.

**Not built.** These do not exist, and no amount of looking will find them:

- Swappable or layered equipment.
- AI restyling of a rendered sheet, or a learned pixel refiner.
- Any animation beyond idle, walk, run, attack and jump.
- A species the registry does not carry. There are thirty-one, and the resolver will tell you when
  yours is not one of them rather than approximating it.

## Try it

Without a GPU, with Blender installed:

1. Create → **Character**. Type `fire ogre, 3/4 top down sprite sheet` and read the column back.
2. Press **Preview character** to see the body before you commit to a sheet.
3. **Generate**, and watch the mesh, rig and sheet rows go by in the in-progress list.
4. Play it with `Space`, step through the walk with the arrow keys, and look at it from behind.
5. Ask for something the registry does not make — `phoenix` — and read the refusal. Then take the
   offer and see what you get.
6. Open the sheet in Inker and fix one cell by hand, then add the sheet to a Packwright atlas.

If you have your own rigged character:

1. Import it, rig it in Poser, and use **Build another sheet** at 32 px.
2. Note that it costs CPU minutes and no GPU at all.

## What to read next

[Tuning what you get](12-tuning-what-you-get.md) — seeds, LoRAs and the other controls that
change what the generators produce.
