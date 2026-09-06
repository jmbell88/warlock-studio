# Troupe

Troupe is the character-sprite mode: a prompt goes in, a 3D character comes back, and what you get
out is a 256-cell sprite sheet — five animations in eight directions — that plays in the middle of
the window while you watch it.

It exists because the two things this app was already good at do not, on their own, make a
character sheet. A sprite synthesised from one drawing has to *imagine* the other three sides, and
guesses differently every time; a mesh rendered from eight angles is exact, but nothing was
animating it. Troupe is the second of those, with a clip library and a pixel-art reduction on the
end of it.

Troupe carries an **Experimental** chip on the rail, and the chip points here for the reason. Two
things are behind it. The shipped animation keyframes are provisional — they are authored placeholders
rather than an animator's work, and a walk that reads as a walk is not the same as a walk you would
ship. And building a character *by reconstruction from a drawing* is measured not to work yet: on a
graded corpus run at the shipped default on 2026-08-30, humanoids came back with limbs bent and
stretched, which is the one failure a character sheet cannot survive — the reconstruction is being
asked for separable limbs from a single view and does not deliver them. The reference stage is fine;
it is the mesh that is lost.

What follows from that is a recommendation rather than a refusal. **Two routes avoid the
reconstruction entirely**, and either is worth reaching for before the form below:

- **[Create → Character](22-generating-references.md#characters)** builds the body from an authored
  registry — four body plans, thirty-one species — instead of recovering one from a picture. It is
  the route that turns a typed sentence into a sheet, it needs no graphics card, and
  [chapter 11](11-a-character-sprite-sheet.md) walks it end to end.
- **[A mesh you already have](#from-a-mesh-you-already-have)** — generated and kept, uploaded, or
  built in Clay — is the route for a character that is yours.

Everything downstream of the mesh (the rig, the clips, the render, the reduction) is the same code on
all three routes. The form below is left in place because it is how you get a reference for a
creature the registry does not model, and because the verdict is on today's default reconstruction,
not on the idea.

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

That is the default: 32 frames per direction and 256 cells in total, laid out eight to a row in the
order `animation → direction → frame`. The directions are the eight compass points of a turn,
starting at `front` and going clockwise in 45° steps.

That layout is configurable, and the form is where you change it. Each animation can be switched off
or given a different frame count, and each can be rendered in 1, 4, 8 or 16 directions. A sheet
warns above 256 cells and refuses above 512.

What does not change is the *shape* of the contract. Eight cells to a row, and the order
`animation → direction → frame`, so an engine that reads the sidecar knows where `walk_left` starts
without guessing. The sidecar carries the layout the sheet was actually built with, which is what
makes a per-character frame count safe to offer.

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
[Installation](39-installation.md) covers.
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
the [library](36-library-and-jobs.md), on the inspector under the asset, and inside Troupe itself
under **Or use a mesh you already have**, below the form, which is collapsed until you open it.

There is no reference and no gate on this route: the mesh already exists, so the only decision left
is what the sheet should look like — and the settings in the form above are the ones it uses, which
is why the picker sits under them rather than repeating them.

If the mesh is not rigged, it is rigged first, as a humanoid, with the joints measured off the mesh
rather than fitted to its bounding box. That is a real cost: rigging is minutes of CPU before a
single cell is rendered, which is why the button says so. You get two rows in the queue — the rig,
then the sheet — and either can be cancelled on its own; cancelling the rig simply means no sheet.

A mesh already rigged on a skeleton that has no clips authored for it is refused, immediately and
before anything is queued: a sheet is animated from a clip library, and a walk cycle means nothing to
a skeleton nobody wrote one for. Four of the eight templates ship with clips — `humanoid`,
`quadruped`, `bird` and `blob` — and each carries all five movements. `fish`, `insect`, `serpent` and
`biped_tail` have none, so a mesh rigged on one of those means re-rigging it on one of the four from
[Create's Rig stage](25-rigging-and-posing.md) and sending it again.

While either row is running the character appears in the cast under **In progress**, saying which
step it is on. Nothing on this route ever waits for you.

### The options

| Setting | What it does |
| --- | --- |
| Build | Which guide conditions the reference: male or female. They differ in shoulder width, arm length and stance. |
| Reference pose | A-pose or T-pose. A-pose matches the rig template and is the default; T-pose separates the limbs further. Both draw the same figure — only the arms move. |
| Camera | One of four **presets**, and the helper under the picker states the chosen one's elevation in degrees. **3/4 top-down** (35°) is the default and the angle most 2D games with depth are drawn at; **Isometric** (30°) matches what tilesets call isometric; **Side** (0°) is straight on; **Top-down** (60°) is as far over as a humanoid still reads — a true overhead figure is a pair of shoulders and a hat brim. The degrees are in the helper rather than left to the name because the number is the thing that transfers: if you are matching these sprites to a Plotter map you already know what elevation that map is drawn at, and "isometric" does not answer that while 30° does. The same four presets are on Create's Character column, and a sheet records which one it was rendered at in its sidecar. |
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

**The heatmap** sits above the sprite: one square per cell of the selected animation, directions
down and frames across. Amber means the frame's silhouette, position or colours jump from the one
before it, a cycle's last frame does not meet its first, or that direction has drifted in size from
the rest; red with a cross means well past that, or an empty cell. Hover for the numbers, click to
land the preview on that cell. It is computed once when you pick a sheet -- "scoring..." while it
runs -- and it ranks cells for you to look at. Nothing downstream reads it; a red square refuses
nothing.

The row above the sprite carries the transport and the two selectors:

- The **stop/play** button pauses and resumes. So does `Space`.
- The **arrows** step one frame, and stepping pauses — stepping means you are looking at something.
  `Left` and `Right` do the same.
- The **animation** and **direction** buttons choose what plays. Changing animation restarts the
  clip; changing direction does not, so you can turn the character mid-stride and see the same
  frame from the other side. `Up` and `Down` turn the character, `PageUp` and `PageDown` change
  animation, and `Home` and `End` jump to the first and last frame of the run.
- The **zoom** field is how many screen pixels one sprite pixel is drawn as.

Two more buttons on that row are about *reading* the sprite rather than playing it, and they are the
first to collapse into the overflow menu when the pane is narrow:

- **Checkerboard** (`C`) puts a checker behind the sprite, so a transparent pixel looks transparent
  instead of looking like the panel's colour. It is **off** by default: a character sheet is judged
  on its colours first, and a pattern behind every frame is noise until the question you are asking
  is "where is the transparency".
- **Pivot** (`P`) draws the point the sidecar records as this sprite's pivot — where an engine will
  place its feet. It is **on** by default, because it is the one thing on the picture that is not in
  the picture, and a user who does not know the mark exists will never switch it on to find out.
  Nothing is drawn on a sheet that records no pivot, which is any sheet rendered before pivots were.

A looping animation loops. A one-shot — attack, jump — holds its last frame rather than stopping,
because a preview that stops needs a control to start it again.

## The sheet panel

The top of the right column says what you are looking at: the grid, the cell size, how many tagged
runs the sidecar carries, and what the pixel-art pass measured — how many colours came out, which
palette they came from, and how many stray pixels were cleaned up.

A large stray-pixel count is worth noticing. It means the reduction found detail the palette could
not hold, and the usual answer is a bigger sprite or a wider palette rather than a different
character.

### Needs repair, and why it is not the heatmap

A sheet that failed its structural check carries a **Needs repair** pill — in the panel, and again
on the row in the **Sheets** list, so that when you are comparing a 32 px attempt with a 64 px one
you can see which came back broken without selecting each in turn. Under the pill is what was found,
as sentences: how many cells are cut off at the frame edge, how many came back empty, how many the
plan named and nothing rendered, anything the sidecar says that disagrees with itself, and whether
the sheet had to be rendered a second time at a wider margin to fit its poses.

**Two different questions, and blurring them would be the mistake.** The heatmap above the sprite is
*advice about the drawing*: it ranks each cell against its neighbours and flags the worst, and
nothing anywhere refuses a sheet on its account. Needs repair is *facts about the file*: a cell is
cut off, empty, or was never rendered. An amber square and this pill do not mean the same thing.

Needs repair gates nothing either. The sheet plays, exports, and opens in Inker exactly as any other
does — an intentionally edge-to-edge portrait counts as clipped by this measure and is exactly the
sheet its author wanted. What the pill buys you is knowing which runs are worth re-rendering, which
is the button directly underneath it.

The camera is framed to hold every pose in the sheet before anything renders, so a jump apex or an
overhead attack wind is inside the frame rather than cut off at the top; if a silhouette still
touches the edge, the sheet is rendered once more with more room around it, once, and the sidecar
records that it happened.

### Making another one

**Build another sheet** renders the same character again at whatever the form on the left currently
says. It re-uses the rig that already exists, so it is minutes of CPU and no GPU at all — which is
what makes "try it at 64 as well" a reasonable thing to do.

**Vary in Create** appears on a sheet the character registry built, and loads that character's whole
recipe into Create's Character column as *your own* settings — so a new seed, a wider palette or a
longer horn makes the next one, and editing the prompt afterwards will not undo them. A sheet
rendered from a supplied mesh, or from Troupe's own form above, has no recipe behind it, so the
button is absent rather than greyed: a disabled control implies a state you could reach.

## Characters that are on fire

Some species carry themes that do more than repaint them. A fire ogre, a fire dragon and a fire
elemental each declare an *effect* as well as a palette, and the sheet draws it: a flame at the
socket the body plan puts it on — the elemental's core, the dragon's and the ogre's crown — in
every cell, animated with the movement it belongs to.

This only happens for a character the app built from a species. A mesh you generated from a
photograph or uploaded has no species behind it, so there is nothing declaring an effect and the
sheet is the sheet it has always been.

Three things are worth knowing about how it is drawn.

The flame is composited **before** the pixel-art pass, not after, so its oranges go through the same
colour cut as the character's skin. That is why a 16-colour sheet of a burning elemental spends
some of those sixteen on fire — and why the flame looks like it belongs to the sprite rather than
sitting on top of it. If you want more of the palette back for the body, ask for more colours.

The flame **rises**, whichever way the character is facing. Fire goes up in the world, not up
relative to the camera, so the eight directions of a turn all show a flame going the same way.

And it is drawn **behind** the body when the socket is on the far side of it. A back-mounted effect
seen from the front is hidden by the character, which is what makes turning around read as turning
around.

One known limitation. **A short loop does not loop seamlessly.** The flame's shape comes from a
scrolling noise field, and that field does not come back round to where it started — so the last
frame of a four-frame idle does not hand cleanly back to the first. At sprite sizes it reads as a
flicker in the tips of the flame rather than a visible jump, and the longer movements (an
eight-frame walk or run) hide it almost entirely. If it bothers you on an idle, a longer idle is
the answer: give the movement more frames in the form.

## Taking it somewhere

Both ways out are bridges the app already had, because a character sheet is an ordinary sheet with
an animation block on it.

**Open in Inker** opens the sheet sliced on its own grid, one tag per animation and direction, in
the [Inker timeline](29-inker-animation.md). It opens *unlinked*: the first `Ctrl+S`
is a Save As, so cleaning up frames cannot overwrite the render they came from.

**Add to Packwright** contributes one sprite per cell to an open atlas, alongside everything else
being packed.

**Export package...** is the third way out and the only one that produces *files*. It copies the PNG
and its JSON sidecar together into a folder you choose — or straight into your configured export
folder, if you have one — and it copies them as a pair on purpose: the PNG is the atlas and the JSON
is what says which cell is `walk` facing south-east, so a folder holding one without the other holds
an asset nothing can interpret. Either both land or neither does.

The sheet and its sidecar are on disk beside the mesh either way, in that job's directory, and the
[library](36-library-and-jobs.md)'s export list is where the files themselves are.

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
