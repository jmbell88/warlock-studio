# Troupe

Troupe is the character-sprite mode: a prompt goes in, a 3D character comes back, and what you get
out is a 256-cell sprite sheet — five animations in eight directions — that plays in the middle of
the window while you watch it.

It exists because the two things this app was already good at do not, on their own, make a
character sheet. A sprite synthesised from one drawing has to *imagine* the other three sides, and
guesses differently every time; a mesh rendered from eight angles is exact, but nothing was
animating it. Troupe is the second of those, with a clip library and a pixel-art reduction on the
end of it.

The output is an ordinary sprite sheet — the same PNG-and-sidecar pair chapter
[Sprite sheets](27-sprite-sheets.md) describes — plus an `animation` block that carries the frame
durations and one tag per animation and direction. Everything that already reads a sheet reads
this one.

## What a character sheet contains

Five animations, in eight directions, at a fixed frame count each:

| Animation | Frames | Loops | Frame time |
| --- | --- | --- | --- |
| idle | 4 | yes | 150 ms |
| walk | 8 | yes | 100 ms |
| run | 8 | yes | 60 ms |
| attack | 6 | no | 80 ms |
| jump | 6 | no | 100 ms |

That is 32 frames per direction and 256 cells in total, laid out eight to a row in the order
`animation → direction → frame`. The directions are the eight compass points of a turn, starting
at `front` and going clockwise in 45° steps.

The counts are not settings. A sheet is a *contract* with whatever imports it — an engine that
knows a Troupe sheet knows where `walk_left` starts without reading anything — and a per-character
frame count would make that impossible to state.

## Making a character

The left column has the cast at the top and the form at the bottom. Describe the character, pick a
build and a reference pose, and press **Draw the reference**.

That queues **one image and then stops**. The reference is drawn against a pose guide, which is a
stick figure fed straight to the ControlNet — legs straight, feet on one line, and the arms held out
at whichever of the two poses you picked. The constrained pose is not an aesthetic choice: a
single-view reconstruction has to get limb separation right, and a folded arm is the failure it
cannot recover from.

**A-pose or T-pose.** A-pose is the default and holds the arms 45° down; T-pose holds them straight
out. The difference is not cosmetic and it shows up two steps later, at the rig. The shipped humanoid
rig template is itself an A-pose, so an A-posed mesh is fitted straight to it. A T-posed mesh is not
— fitting that template to one runs the arm chain down through the ribcage — so its joints are
*measured* off the mesh instead, which needs the pose model
[Installation](38-installation.md) covers.
T-pose separates the limbs further, which is the one thing a single view has the most trouble with,
so it is the one to reach for if the arms come back fused to the body.

You approve that drawing in [Create](22-generating-references.md), the same way you approve any
reference. Only then does the rest run: the reconstruction, then an automatic rig, then the sheet.
Approving is the gate, and it is deliberate — the reconstruction is the expensive step and it should
not be spent on a drawing you would not have kept.

If the character is a disaster, the drawing is still a row you can reroll or edit. That is the point
of the two-step shape.

### From a mesh you already have

The form above starts from nothing. If you already have a character — one you generated, uploaded,
or built in Clay — **Send to Troupe** takes it in directly. It is on the mesh's right-click menu in
the [library](34-library-and-jobs.md), on the inspector under the asset, and inside Troupe itself
under **Or use a mesh you already have**, below the form, which is collapsed until you open it.

There is no reference and no gate on this route: the mesh already exists, so the only decision left
is what the sheet should look like — and the settings in the form above are the ones it uses, which
is why the picker sits under them rather than repeating them.

If the mesh is not rigged, it is rigged first, as a humanoid, with the joints measured off the mesh
rather than fitted to its bounding box. That is a real cost: rigging is minutes of CPU before a
single cell is rendered, which is why the button says so. You get two rows in the queue — the rig,
then the sheet — and either can be cancelled on its own; cancelling the rig simply means no sheet.

A mesh that is already rigged as something else is refused, immediately and before anything is
queued: the clips are a humanoid's, and a walk cycle means nothing to a quadruped. Re-rig it as a
humanoid from [Create's Rig stage](25-rigging-and-posing.md) and send it again.

While either row is running the character appears in the cast under **In progress**, saying which
step it is on. Nothing on this route ever waits for you.

### The options

| Setting | What it does |
| --- | --- |
| Build | Which guide conditions the reference: male or female. They differ in shoulder width, arm length and stance. |
| Reference pose | A-pose or T-pose. A-pose matches the rig template and is the default; T-pose separates the limbs further. Both draw the same figure — only the arms move. |
| Sprite size | How many pixels tall one cell is. 16, 24, 32, 48, 64, 96 or 128. |
| Outline | `outer` grows the silhouette by a dark pixel, `inner` recolours the sprite's own edge, `none` leaves it alone. |
| Palette | A palette file if you have installed one, or a palette derived from the render by median cut. |
| Colours | The budget for a derived palette. Ignored when a palette file is named. |
| Dither | Ordered dithering when colours are mapped. Off by default; at sprite sizes it is usually noise. |

Every one of these is checked when you press the button, not when the sheet is finally rendered —
so an unreadable palette costs you the click rather than an hour.

Sizes divide evenly out of the 512-pixel render at 16, 32, 64 and 128; the other three go through a
documented resize instead. Neither is wrong, but the exact ones are crisper.

## Watching it

The middle of the window shows one sprite at a whole-number scale with no filtering. A sprite drawn
at 6.3× through a smoothing filter is a blurred sprite, which is the one thing the whole pipeline
exists not to produce.

**It opens paused, on the first frame.** Press play, or `Space`, to run it. A bad frame in a walk
cycle is obvious in half a second of playback and invisible in a contact sheet — but the first thing
anyone does with a new sheet is look at a *frame*: a hand, a silhouette, which way the feet point.
A clip already moving when you arrive is one you have to stop before you can look at anything.

The row above the sprite carries the transport and the two selectors:

- The **stop/play** button pauses and resumes. So does `Space`.
- The **arrows** step one frame, and stepping pauses — stepping means you are looking at something.
  `Left` and `Right` do the same.
- The **animation** and **direction** buttons choose what plays. Changing animation restarts the
  clip; changing direction does not, so you can turn the character mid-stride and see the same
  frame from the other side.
- The **zoom** field is how many screen pixels one sprite pixel is drawn as.

A looping animation loops. A one-shot — attack, jump — holds its last frame rather than stopping,
because a preview that stops needs a control to start it again.

## The sheet panel

The top of the right column says what you are looking at: the grid, the cell size, how many tagged
runs the sidecar carries, and what the pixel-art pass measured — how many colours came out, which
palette they came from, and how many stray pixels were cleaned up.

A large stray-pixel count is worth noticing. It means the reduction found detail the palette could
not hold, and the usual answer is a bigger sprite or a wider palette rather than a different
character.

**Build another sheet** renders the same character again at whatever the form on the left currently
says. It re-uses the rig that already exists, so it is minutes of CPU and no GPU at all — which is
what makes "try it at 64 as well" a reasonable thing to do.

## Taking it somewhere

Both ways out are bridges the app already had, because a character sheet is an ordinary sheet with
an animation block on it.

**Open in Inker** opens the sheet sliced on its own grid, one tag per animation and direction, in
the [Inker timeline](29-inker-animation.md). It opens *unlinked*: the first `Ctrl+S`
is a Save As, so cleaning up frames cannot overwrite the render they came from.

**Add to Packwright** contributes one sprite per cell to an open atlas, alongside everything else
being packed.

The sheet and its sidecar are on disk beside the mesh either way, in that job's directory, and the
[library](34-library-and-jobs.md)'s export list is where the files themselves are.

## When it goes wrong

**"A character sheet needs a rigged mesh."** The automatic rig failed, or you cancelled it. Rig the
mesh in [Poser](26-poser.md) — joints measured off the mesh's own vertices, which is what the
automatic pass tries first — and then use **Build another sheet**.

**The arms are welded to the chest.** The shipped humanoid template is an A-pose, and fitting it to
a mesh standing in a T-pose puts the arm chain inside the ribcage. The automatic rig asks for
measured joints for exactly this reason; if it still happens, correct the joints in Poser.

**Everything is one pale colour.** The mesh has no texture. The reference chain is what puts colour
on a character; a supplied base mesh with no material has nothing for the palette to quantise, and
the sheet will come back in whatever few greys the render produced.

**The character is lying down.** A rotation stored in the wrong frame. Clips ship authored as deltas
from the skeleton's rest pose, which is the only frame that survives the rig being fitted to a
different mesh — a clip authored against one skeleton's node-local orientations means something else
entirely against another's.
