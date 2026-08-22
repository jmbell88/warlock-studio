# Animating

Inker's timeline turns a drawing into an animation. It is a grid: layers down the side, frames
across the top, and a **cel** — one layer's pixels on one frame — at each crossing.

Like the last chapter, none of this needs a GPU or any weights.

## Turning a drawing into an animation

A new Inker document is a still. **Animate this drawing** turns it into a two-frame animation — your
drawing becomes frame one and a second, empty frame is added after it — and the timeline appears
along the bottom. Nothing about your existing layers changes; they become the first column of the
grid.

The transport under the timeline adds frames, and there are two ways to add one, which is the single
most important distinction in this chapter.

## Copy versus link

**Copy a frame** and you get independent pixels. Paint on the copy and the original is untouched.

**Link a frame** and you get *the same cel in two places*. It is one object appearing twice in the
grid, not two objects that happen to match. Paint on either and both change, because there is only
one thing there.

Linking is how you hold a background still across thirty frames without thirty copies of it, and how
you make a two-frame flap that repeats without editing it twice. Copying is how you make the next
drawing in a sequence.

Getting this backwards is the classic first-hour mistake: link a frame, paint the next pose onto it,
and watch the previous pose change underneath you.

There is a third behaviour, set per layer: a **continuous** track fills an empty cel by looking
backwards along its own row to the nearest drawn one and taking a *copy* of it — not a link. It is
for layers that mostly hold still and occasionally change.

## Tags

A **tag** names a span of frames: `walk` is frames 2–9, `idle` is 10–13. Tags are what an engine
imports, and what the sheet exporter writes into its sidecar.

Each tag carries three things:

- **Loop** — whether it repeats.
- **Direction** — forward, reverse, or ping-pong.
- **Repeat** — a finite count, where zero means "let the loop flag decide".

A finite repeat does not run on past its own span. When it is done, it is done.

Note where playback settings do and do not live. Loop, direction and repeat belong to **the tag**,
which is a property of the document that travels with it. Playback *speed* and "play once" do not —
they belong to the preview pane, below.

## Onion skinning

Onion skinning ghosts the frames either side of the one you are drawing, so you can see what you are
animating between.

By default the ghosts are the whole frame, composited. **Current layer only** is a toggle beside the
depth controls, off until you ask for it, and it is the one to reach for on a twelve-layer character:
ghosting every layer of one turns the canvas into soup at exactly the moment you need to see a single
limb clearly.

The controls are depth before, depth after, and fade. Behind a **More** popover: separate tints for
past and future, a falloff exponent, whether ghosts draw in front of or behind your work, and
whether the ghosting wraps inside the current tag rather than running off its end.

## Watching it

Two independent ways to play, and they are not the same thing.

**The transport's play** moves the document's own playhead. `Enter` starts and stops it, and `.` and
`,` step one frame forward and back. What you see is the document at real speed.

**The preview pane** has a *second* playhead of its own, with a speed ladder from 0.25× to 4× and a
scope of either the whole clip or just the active tag. It never touches the document's playhead, so
you can leave a loop running in the preview at half speed while you draw on a frame in the middle of
it.

## Exporting

The transport's bottom row is the way out:

| Export | What you get |
| --- | --- |
| **Sheet** | One PNG with the frames in a grid, plus a JSON sidecar describing them. |
| **GIF** | An animated GIF, using the document's own palette where that is exact. |
| **PNGs** | One file per frame. |
| **Sheet per tag** | One sheet for each tag. |
| **Sheet per layer** | One sheet for each layer. |

`Ctrl+Shift+X` repeats the last export you did with the same settings, which is what you want while
iterating.

Inker's sheets are row-wrapped grids. That is on purpose and it is not a limitation to complain
about — real packing, with trimming and tight layout, is [Packwright's](10-packing-an-atlas.md) job,
and an atlas packer inside the drawing tool would be a second, worse one.

## Getting animation in from elsewhere

The **Open in Inker** buttons on generated sheets bring an existing sheet in as an animation, sliced
on the rectangles the generator recorded rather than on rectangles guessed from the pixels. A
character sheet arrives with its tags already made.

That is the intended shape of the work: let the pipeline produce the frames, then fix them by hand
here.

## Try it

1. Open the drawing from the last chapter and press **Animate this drawing**.
2. Add three frames with **copy**, and draw a slightly different pose on each.
3. Add a fourth frame with **link**, pointing at the first. Paint on it and watch frame one change —
   then undo.
4. Tag frames 1–4 as `walk`, set the direction to ping-pong, and leave loop on.
5. Turn onion skinning on with two frames back and two forward, and fix whichever pose looks wrong.
6. Press `Enter` to play. Then open the preview pane and run the same tag at 0.25× while you keep
   drawing.
7. Export a sheet, and read the JSON sidecar beside it.

## What to read next

The workspace chapters are independent — take whichever you need. [Modelling](07-modelling.md) is
Clay, which builds meshes by hand rather than by prompt.
