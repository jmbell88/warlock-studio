# Casting a spell

Flourish is the effect generator inside Inker: a fireball, a shockwave, a heal, a portal — about
thirty presets — rendered from a *recipe* rather than drawn or generated frame by frame. A recipe is
a seed, a few phases and a stack of ingredients (a core, a flame, a glow, sparks, smoke, a trail,
heat shimmer), each with numbers you can change. The same recipe renders the same pixels every
time, which is what makes the result an animation rather than sixteen pictures that hopefully
agree. Like Inker itself it needs no GPU and nothing downloaded.

This walks one path: a fireball into a blank document, a colour changed, a cell painted by hand and
kept through a re-render, an export per phase. Ten minutes.

## A blank canvas

Open **Inker** from the rail and make a new 128×128 document (`Ctrl+N`). Anything works — a
sprite you are already animating, too — but a blank canvas shows the effect on its own.

## Insert a fireball

**Flourish → Insert effect...** Pick **Fireball**, leave the look on **Painterly** and the facings on
**One direction**, and press **Insert**.

A moment later a layer group called *Fireball* appears above the background, one layer per
ingredient — *Smoke*, *Trail*, *Outer flame*, *Core*, *Embers*, *Shockwave*, *Flash*, *Fire burst*,
*Sparks*, *Ash*, *Glow*, *Heat* — and the timeline has five tags: *cast*, *projectile*, *impact*,
*explosion*, *dissipate*. Press **Space**. The projectile tag loops on its own, the way a tag does
([Tags](29-inker-animation.md#tags)); scrub past it to see the impact and the burst. It is one undo
step: `Ctrl+Z` takes the whole group away, `Ctrl+Y` brings it back.

## Change a colour

Click any layer of the group. The **inspector** appears under the timeline's transport: seed,
frame rate, look, the frames and loop flag of every phase, and a row of buttons. Under them, a
layer picker and every parameter the picked ingredient has.

Pick **Core** and set *color_outer* to a cold blue. Let go. About a quarter of a second after the
slider rests the effect renders again in the background, the timeline updates when it lands, and
that render is one undo step. The frame loop never waits on it: keep scrubbing while it runs.

Try **turbulence** on *Outer flame*, **count** on *Sparks*, and the two sliders a curve shows —
what a value starts at and what it ends at across the phase. Change the **seed** for a different
arrangement of the same sparks.

Or type it: put *colder, more sparks, no smoke* in the prompt field under the buttons and press
Enter. The toast lists what each word did. With no language model on the machine that is a
fixed vocabulary of colours and adjectives, which is enough for most of what you would say to a
fireball; a local instruct model, when one is present, reads the sentence instead — and its
answer is clamped exactly like a slider.

## Paint on a cell, then regenerate

Go to a frame in the *explosion* tag, pick the **Sparks** layer, and paint a few pixels with the
pencil. Now change any parameter and let it render.

The inspector says *1 painted cell(s) flagged* and the toast says how many cells were rendered and
how many were kept. Every render remembers what it put in each cell; a cell that still holds the
last render takes the new one, a cell you painted keeps your paint and is flagged. That is the
same rule a re-rendered character sheet is merged with
([Sheet corrections](29-inker-animation.md#sheet-corrections)), for the same reason: a cell
wrongly kept is one click to re-take, a cell wrongly taken is your afternoon.

**Keep painted cels** clears the flag and leaves your pixels. **Replace painted cels** renders over
them. Pick one.

## A texture of your own

Draw a small skull on the background layer, select it with the marquee, and choose **Flourish →
Use selection as texture** with the *Sparks* layer picked in the inspector. The sparks are now
skulls — scaled, tinted along the spark colour ramp, turned by **spin** — and the texture is saved
inside the `.ora` beside the recipe. **Generate texture...** does the same from a few words through
the image model, keying the black background out.

## Pixel art and facings

Insert the same effect again with the look on **Pixel art**. This time the group holds one layer —
the finished composite, hard alpha, one palette across every frame — because the pixel pass runs on
the composite and a stack of separately quantised layers would not add up to it. The palette is
derived from every frame of the effect at once, so the cast's glow and the dissipate's ash share
one set of colours; an indexed document's own palette is used instead when it has one.

With **Four directions** or **Eight**, the simulation itself is turned — the same spread, the same
gravity, facing down — and each phase gets a tag per facing: `impact/E`, `impact/S`, and so on.

## Export

A phase is a tag, and a tag is what the sheet export writes
([Exporting part of a clip](29-inker-animation.md#exporting-part-of-a-clip)): one `fireball_cast.png`,
one `fireball_projectile.png`, one `fireball_explosion.png`, each with its sidecar naming the frame
size, the count and the frame rate. Aseprite export keeps the layers and the tags and drops the
recipe, which that format has nowhere to hold; `.ora` keeps the recipe, so the effect still
regenerates next week.

## Where to go next

[Effects](29-inker-animation.md#effects) for the reference; [Putting it in a
game](13-putting-it-in-a-game.md) for the engines.
