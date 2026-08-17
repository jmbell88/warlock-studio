# Generating references

A reference is the picture the mesh will be reconstructed from. Everything in this chapter lives in
the **2D reference** mode's settings pane, in the left sidebar.

## The prompt

The large text box under **Prompt** is where you describe the object. Write the subject and nothing
else — "a weathered wooden crate bound with iron", "a compact energy rifle with panel seams". You do
not need to ask for a plain background, a single object, or a studio render: the app wraps whatever
you write in a fixed template that already asks for all of that, because those are the properties
that make an image reconstruct cleanly.

The prompt is capped at 1000 characters and a counter under the box shows how much you have used.
The counter turns amber inside the last hundred characters.

Beside the box, a **Recent** button opens your last twenty prompts, most recent first and
deduplicated — it appears once you have generated at least one reference, so there is history to
show. Picking one replaces what is in the box. The history is per session and per prompt
text only — if you want a whole recipe back, use **Copy settings to form** from a job's overflow
menu instead, which is described in [Rerun and promotion](11-library-and-jobs.md#rerun-and-promotion).

Under **Negative prompt**, further down the pane, is a second box listing what the image must not
contain. It is pre-filled with the things that most often ruin a reconstruction:

```text
blurry, low quality, multiple objects, cropped, cut off,
text, watermark, signature, busy background, human hands
```

A second subject, or a subject cut off by the frame edge, is the single most common cause of a mesh
that comes out as nonsense — which is why this is a filled-in default rather than an empty field
you have to discover. You can edit it freely, or empty it deliberately. Note that a negative prompt
only has an effect on a model that runs with real classifier-free guidance; the two four-step
distilled defaults ignore it. See [Models and style LoRAs](#models-and-style-loras).

Expand **Prompt actually sent** to see the complete composed prompt — your text inside the fixed
template — along with its token count and how many chunks it was split into. It updates about a
third of a second after you stop typing, computed on a background thread because counting tokens
means loading a tokenizer.

The composed prompt has no hard length ceiling. CLIP's text encoders stop at 77 tokens, but the app
splits a longer prompt into several chunks on comma boundaries — never mid-phrase — encodes each
separately and joins them, which the image model's cross-attention accepts without complaint. That
is why the preview reports chunks rather than warning about truncation. The soft limit still
applies, though: a longer conditioning sequence dilutes attention, so your prompt is best kept to a
sentence.

### Prompt enrichment

Under the prompt box, the **enrich** select turns on a local prompt expander — a small GPT-2 model
(the one Fooocus ships) that appends aesthetic detail to short prompts before the image model sees
them. This is the offline form of what every hosted image service does silently: a language model
rewrites "a sword" into a dense descriptive caption, and the image model's output improves because
the description did. It is **off** by default and needs its own download (about 700 MB, listed in
Settings → Models as "Prompt expander"); with expansion selected and the weights absent, the
submit is refused with the download command.

Two modes, because the right enrichment depends on what the picture is for:

- **3D asset** enriches the prompt and keeps the single-subject, plain-background framing — the
  right choice when the image is a reference for a mesh.
- **General 2D** enriches the prompt *and* swaps the fixed template for one that allows
  composition, backgrounds and scene lighting. Use it for pictures that will stay pictures: a
  reference generated this way will usually be refused at promotion, because it is no longer a
  single centred object.

Three things the expander deliberately does not do. It never touches a prompt that is already
detailed (roughly forty tokens or more) — appending tag soup to a paragraph dilutes it. It never
rewrites your subject: generation is constrained to a fixed whitelist of aesthetic vocabulary, so
it can only add phrases like lighting and quality terms, not change what the picture is of. And it
never runs on a seamless tile, whose prompt describes a surface rather than a subject.

The expansion is deterministic in the job's seed and is recorded on the finished job's params as
`expanded_prompt`, so a job's provenance always shows the text the image model actually received —
**Prompt actually sent** previews it with the mode on.

## The pane at a glance

The form is one flat column of sections: **Output** (Object or Seamless tile), **Profile**,
**Prompt**, **References** (conditioning on an image), **Seed** (how many and which seed),
**Model**, **LoRA** and **Negative prompt**. There are no folds — everything is on screen, and the
Generate button stays pinned below the column.

Earlier versions carried a twelve-select creative taxonomy (category, genre, material and the
rest) behind a "More options" reveal. It was retired on 2026-08-17: no taxonomy axis ever measured
a quality win, and your prompt is the brief. Assets generated under it are unaffected — rerolling
or promoting one simply composes without the retired fragments.

In the **Profile** row, beside **Save as...**, is **Reset...**, which puts the whole 2D form back
to its first-launch defaults after a confirm — the prompt, the negative prompt, the model and LoRA,
the reference and the run controls, with a freshly rolled seed. It touches nothing outside this
pane: saved profiles are kept, and the 3D form is left alone.

## Models and style LoRAs

The **Model** section holds the model choice. Nine base models ship in the registry:

| Model | What it is | Runs at |
| --- | --- | --- |
| SDXL-Turbo (fast) | Small and quick, at a quarter of the resolution. Its own separate download. | 512 px, 4 steps, guidance 0 |
| SDXL 1.0 + Hyper-SD (best LoRA response) | Full SDXL weights with a step-distillation LoRA fused on. | 1024 px, 4 steps, guidance 0 |
| SDXL 1.0 + Lightning (4-step) | The same weights and the same idea, distilled a different way — an alternative to compare Hyper-SD against. | 1024 px, 4 steps, guidance 0 |
| Playground v2.5 (highest fidelity, slow) | The best-looking output, and correspondingly slow. | 1024 px, 25 steps, guidance 3.0 |
| SDXL 1.0 (full CFG, structural control) | **The default.** The same weights as the Hyper-SD entry, run the way the checkpoint was trained — the negative prompt and ControlNet both work here. | 1024 px, 30 steps, guidance 7.0 |
| SDXL 1.0 + LCM (pixel art) | The same weights again, under a consistency adapter — the recipe the pixel-art LoRA was trained against. | 1024 px, 8 steps, guidance 1.0 |
| Juggernaut XL v9 (photoreal) | A photoreal SDXL finetune, at its own documented recipe. | 1024 px, 35 steps, guidance 4.0 |
| DreamShaper XL (stylised) | The stylised counterpart to Juggernaut. | 1024 px, 25 steps, guidance 7.0 |
| FLUX.2 klein-base 4B (full CFG) | A different architecture entirely, and the slowest thing here. | 1024 px, 50 steps, guidance 4.0 |

The sampler settings travel with the checkpoint and are not yours to set. They are part of the
model's identity: a four-step distilled model run at 25 steps with guidance produces mush, and both
Hyper-SD and Lightning degrade silently without the right timestep spacing.

Three consequences are worth knowing. A **negative prompt** is only encoded when guidance is above
1.0, so it does nothing on the three four-step entries — pick one of the full-CFG models if you want
it honoured. **Structure control** (see [Conditioning on an image](#conditioning-on-an-image)) is
only offered on the SDXL models that run with real guidance, because a ControlNet at guidance 0
fights the hint instead of following it. And **style LoRAs, conditioning and seamless tiles are
SDXL-only**: they are all built against SDXL's internals, so on FLUX.2 the Style LoRA picker is
disabled with a note saying so, and asking for a tile is refused rather than quietly producing one
whose edges do not line up. The negative prompt does work on FLUX.2 — that is why the undistilled
`klein-base` variant is the one that ships.

Only one base model is resident on the card at a time — a 32 GB card holds the reconstruction engine
plus a single image pipeline, not two — so switching between models between jobs costs a reload of
several seconds. FLUX.2 is the exception to *how* it is held: it is large enough that it is streamed
onto the card one piece at a time rather than kept there whole, which makes each job slower but means
it still does not displace the reconstruction engine. Any model whose weights are not on disk is
still listed, with "— weights missing" appended to its name, so you learn at pick time rather than at
job-failure time. Run `warlock doctor` for the exact download command.

**Style LoRAs** are the opposite: they are adapters on whatever pipeline is already resident and
switch for free, with no reload. Five ship:

- **3D render** — a general 3D-render look.
- **3D render (Redmond)** — a second, differently trained take on the same idea.
- **PS1 / low-poly game** — chunky untextured geometry, which is among the easiest things to
  reconstruct cleanly.
- **Pixel art (pixel-art-xl)** — generates on a pixel grid rather than producing a smooth image that
  is later downscaled into one. It defaults to a strength of 1.2 rather than the usual 0.9: below
  that the output keeps SDXL's anti-aliased gradients, and no downscale recovers a clean grid from
  them. It pairs naturally with the LCM base, the recipe it was trained against.
- **Pixel art (FLUX.2 klein)** — the same idea for the other architecture, and the only adapter here
  that is not an SDXL one. It is offered on the two FLUX.2 klein entries and on nothing else. Its
  default strength is 0.0625, far below every other entry, because the adapter declares a trained
  scale of 16 where an ordinary one declares about 2 — the slider means the same thing it always
  did, and this adapter's own strength is simply much larger. Anything above about 0.13 smears and
  the range its model card recommends produces black frames.

An adapter is fitted to one architecture, so the picker offers a model only the styles that fit it.
Choosing a model no style fits disables the picker with a note; choosing one that some styles fit
lists those and says so. A style you picked under a different model stays visible and marked rather
than vanishing, and changing the model clears it with an explanation.

Choosing one reveals a **Strength** slider, from 0 to 1.5, defaulting to the LoRA's own tuned
weight — 0.9 unless the entry says otherwise. The slider is hidden entirely when no LoRA is chosen.
The SDXL LoRAs are trained against full SDXL at 20 to 25 steps with guidance, so they land
noticeably stronger on the SDXL entries than on Turbo.

## Seeds and candidates

Generation is deterministic in its seed: the same form and the same seed produce the same image
every time. The **Seed** section is where you control that.

**References** picks how many candidates one submit queues: 1, 2, 4 or 8. Each is a real job holding
a place in the serial queue, which is why eight is the ceiling. Because each one gets its own seed,
a fan-out of four is the cheapest way to find out whether an idea works at all.

**Seed** is the number itself. **Reroll** replaces it with a fresh random one. **Lock** decides what
happens when you press Generate:

- Unlocked (the default): the seed is rerolled on every submit, so pressing Generate twice on an
  unchanged form gives you two different images rather than the same one twice.
- Locked: the seed is reused, so an unchanged form reproduces exactly. Lock it when you want to
  change one guidance field and see only that field's effect.

Seeds are whole numbers from 0 to 2147483647. The seed shown when the app opens is rolled fresh at
startup and is deliberately not remembered between sessions — otherwise every launch would open on
the same seed and a first Generate would reproduce last week's image.

The mesh has its own separate seed, at the Mesh stage. See
[Mesh parameters](04-generating-meshes.md#mesh-parameters).

## Conditioning on an image

Beyond the prompt, you can steer the image with a picture. In the **References** section, either
**Choose an image...** or drop a file onto the window. Everything below the picker stays
hidden until there is an image, because a control with nothing to act on is a control that cannot
do anything.

With a reference loaded, two independent kinds of conditioning become available.

**Appearance** attaches an IP-Adapter — currently one entry, *Appearance reference (IP-Adapter
Plus)*. It conditions on sixteen patch tokens rather than a single pooled embedding, which is the
difference between "the same kind of object" and "this object". Its **Strength** runs from 0 to 1.5
and defaults to 0.6.

**Structure** attaches a ControlNet — currently one entry, *Edge / silhouette lock (Canny)*, which
traces the reference's edges and holds the new image to that shape. It has two controls.
**Strength** (0 to 2.0, default 0.65) is how hard it pulls. **Until** (0 to 1.0, default 0.8) is how
far into the drawing it keeps acting: ending early lets the last steps add detail the reference
never had, while 1.0 holds the shape to the final step and tends to look traced.

Structure needs a base model that runs with real guidance. If yours does not, the section says so
and points you at Advanced instead of offering a control that cannot work.

Clearing the reference clears both selections with it, since neither can be submitted without an
image.

## Approving a reference

When a reference job finishes it appears in the library and, when selected, fills the viewport at
full size. This is the moment the two-stage pipeline exists for: look at the picture before paying
for the mesh.

What to look for is what the reconstruction engine needs. One subject, complete and not cropped by
the frame, filling a good part of it, against a plain background. The app measures this itself and
records a reference report — if it thinks the image will not reconstruct, it says so in the
inspector's **Reference** section, and promoting it asks you to confirm rather than refusing
outright. The rules are heuristics about composition, and you can see the picture they are arguing
about.

If the image is nearly right, you can fix it by hand: **Open in Inker** on the viewport toolbar
opens the reference as a layered document, and saving writes it back in place. See
[Pipeline bridges](07-inker.md#pipeline-bridges).

When you are happy, press **Make 3D** on the card (or select the reference and step to **Mesh**).
That carries the reference and everything it recorded into the mesh stage, where you can override
the mesh-side settings before committing. The next chapter,
[Generating meshes](04-generating-meshes.md), picks up from there.

## 2D exports

A finished reference is an asset in its own right, not only an input to the mesh stage. Its
inspector's **Export** tab offers, alongside the source image:

| Button | File | What it is |
| --- | --- | --- |
| Icon | `icon.png` | A 512-square transparent cutout, centred with a small margin. |
| Sprite | `sprite.png` | The subject alone at native resolution, trimmed, with a recorded pivot. |
| Pixel art | `pixel_32.png`, `pixel_64.png`, `pixel_128.png` | Nearest-neighbour reductions, optionally palette-limited. |
| Manifest | `manifest.json` | What every artifact above is, measured — see below. |

All of them are cut from the same `input.png` the mesh stage uses, and all are derived the first
time you ask for one, then cached — the same rule the mesh exports follow. Editing the reference in
Inker, or rerolling it, makes every derived file stale, and the next request rebuilds it.

The icon is **fitted** inside its square rather than stretched to it, so a tall sword and a round
shield keep their proportions and an icon set stays readable. The sprite is not resized at all: it
records a bottom-centre pivot in the manifest, because that is where an engine puts a standing
character's feet, and an importer that guesses is wrong for half a set.

### Pixel art

The **Pixel art** section of the inspector's Details tab is where the pixel exports are set up and
previewed. **Size** picks which of the three artifacts you are looking at; **Colours** limits the
palette to 8, 16, 32 or 64, or leaves it off; **Palette** maps the export onto a palette file you
supplied, and **Dither** (offered only with one) mixes two nearby entries where a flat map would
pick one.

A palette is a file you drop into the palette directory (`~/.warlock/palettes/` by default — see
[Configuration](16-configuration.md)), in either of the two formats palette sites publish: Lospec's
`.hex`, one `rrggbb` per line, or GIMP's `.gpl`. Nothing ships with the app, because a palette is
art direction rather than a default. Colours are matched perceptually (in Oklab) rather than by raw
RGB arithmetic, which is what stops a dark grey being mapped to black and a whole shadow being
eaten. A palette file supersedes the **Colours** cap entirely: the cap is a median cut of the
picture's own colours, and a palette is a decision about which colours exist.

Editing a palette in place re-derives every export that used it — freshness is keyed on the file's
*contents*, not its name, because editing one is the normal way to work on it.

The preview is drawn crisp, at a whole multiple of the artifact's own size — a fractional scale
samples some source pixels twice and others once, which reads as banding and is exactly what the
export avoids. A line underneath says what the file on disk actually is, read from the manifest
rather than from the controls, because the two can disagree: switching size shows a file cut under
an earlier palette setting, and a **Rebuild with these colours** button appears when that is the
case.

Both settings are app preferences rather than properties of the job, so they persist across
sessions and apply to whichever reference you are looking at. Changing them re-derives; it never
touches `input.png`, so promoting the reference afterwards feeds the mesh engine the same pixels it
always would have.

The reduction is nearest-neighbour, never a smooth resample: a filtered downscale puts a ramp of
in-between colours along every edge, and hard edges are the one property that makes the result read
as pixel art rather than as a small photograph.

There are two reductions, and which one runs is decided by the *image* rather than by a setting.
A pixel-art model draws logical pixels as square blocks — typically eight screen pixels across — and
that lattice has a phase: the first block rarely starts at the very edge of the frame. When one is
detected, the export takes one output pixel per block, sampled at the block's centre, across the
whole frame, and crops to the subject afterwards; the result is exactly the pixels the model
authored. Cropping first — which is what an ordinary export does — would move the origin off the
lattice by however wide the subject happens to be, and shear the art instead of reducing it. An
ordinary render has no lattice to find and takes the plain path unchanged, which is what every asset
already on disk was cut with. The provenance line says which happened. Palette reduction runs on colour only, with
transparency carried around it, so the cutout survives the quantization exactly.

If you are generating for the pixel exports, ask your prompt for flat shading and a bold
silhouette — the things that survive being made small — and avoid the words "pixel art": at 512 or
1024 the image model draws fake chunky pixels that then alias under the real reduction. The
pixel-art LoRA is the exception, because it authors a genuine pixel grid.

### The manifest

`manifest.json` is the sidecar an importer reads. It records, per artifact, the size, the trim box,
the pivot, the palette it was cut to, and which matte produced it — plus the recipe (seed, model,
prompt) behind the image itself. Per artifact, because a file on disk was cut by whatever was
installed when it was made, not by what is installed now.

If the reference was hand-edited in Inker, the manifest says so beside the recipe: the recipe names
a seed and a model, which after a hand edit is no longer the whole story of the pixels.

## Seamless tiles

The **Object / Seamless tile** control at the top of the 2D settings pane switches the whole pane
between two kinds of output. A tile is a repeating texture rather than a subject: it is drawn with
wrapping convolutions, so its left edge continues into its right and its top into its bottom.

The rest of the form means the same thing for a tile as for an object: the prompt describes the
surface ("mossy cobblestone"), and the model, LoRA and negative prompt choose the machinery that
draws it.

A tile cannot be made into a mesh, and the app does not offer to: there is no subject to
reconstruct. It also cannot produce the cutout exports, because every one of them lifts a subject
off its background and a tile *is* background — an icon of one would be the whole frame with a matte
guessed over it.

### Whether it actually tiles

When a tile finishes, the app measures its own wrap seam and reports it in the inspector's **Seam**
section. The number is a ratio: how much sharper the difference across the wrap edge is than the
picture's own grain. Around 1 means the seam is indistinguishable from ordinary texture detail;
above 3.5 the app calls it a visible seam and says so. Both directions are measured and reported
separately, because an image that wraps one way and not the other is not a tile.

A ratio is hard to calibrate against by eye, so the section also offers a wrapped view: the image
rolled by half, which puts what was the wrap edge through the middle of the frame where a
discontinuity is obvious. It is `wrap_preview.png`, a derived export like any other, and the
Export tab offers it too.
