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

Under **Advanced** there is a second box, **Negative**, listing what the image must not contain. It
is pre-filled with the things that most often ruin a reconstruction:

```text
blurry, low quality, multiple objects, cropped, cut off,
text, watermark, signature, busy background, human hands
```

A second subject, or a subject cut off by the frame edge, is the single most common cause of a mesh
that comes out as nonsense — which is why this is a filled-in default rather than an empty field
you have to discover. You can edit it freely, or empty it deliberately. Note that a negative prompt
only has an effect on a model that runs with real classifier-free guidance; the two four-step
distilled defaults ignore it. See [Models and style LoRAs](#models-and-style-loras).

Expand **Prompt actually sent** to see the complete composed prompt — your text plus every guidance
fragment — along with its token count and how many chunks it was split into. It updates about a
third of a second after you stop typing, computed on a background thread because counting tokens
means loading a tokenizer.

## More options

The pane opens on the short path: the preset picker, the Object / Seamless tile switch, the prompt,
and **Run** — how many references and which seed. Everything else lives behind **More options**, one
reveal that remembers whether you left it open. Nothing is removed by that; while it is closed the
line under it names what is inside and how many of those fields are set, so a form carrying a style
and a conditioning image does not look like an empty one. A refusal that names a control inside the
fold opens it for you.

## Guidance fields

Under **More options** are twelve optional selects, grouped by what they describe:

| Group | Fields |
| --- | --- |
| Subject | category, genre, setting, silhouette, framing |
| Style | era style, palette, mood, rarity |
| Surface | material, condition, emissive |

Each one is a small closed vocabulary — genre offers Fantasy, Sci-fi, Modern, Post-apocalyptic,
Horror and Cartoon; condition offers Pristine, Worn, Damaged, Ancient, Rusted, Overgrown and Burned;
and so on. Every entry maps to a short prompt fragment of two to four words, and the fragments are
appended to your prompt in a fixed order that reads like a sentence: category, silhouette,
material, condition, rarity, emissive, setting, genre, mood, era style, palette, platform.

Leaving a field unset simply omits its fragment. Nothing is filled in for you except the platform
default.

**Framing** is the exception to all of that: it is not appended to your prompt but substituted into
the framing template's view clause, and it always has a value. *3/4 view* is the default and is the
wording every reference generated before this control existed was drawn under. *Front orthographic*
swaps in a straight-on plate — a front view, facing the camera, one view only — which is worth
trying for a character you intend to promote to a mesh, and is a question the sweep tools exist to
answer rather than one with a settled answer.

Two of these have effects beyond the text. **Category** supplies a default physical size when you
do not give one — a prop is 0.4 m, a weapon 1 m, a character 1.8 m, a vehicle 4.5 m, an environment
piece 8 m, a consumable 0.15 m. And **detail brief**, which sits on its own below the groups,
is a hint about how much fine detail to draw — 2D or 3D, defaulting to 3D. It is a *brief*, not a
measurement: it goes into the prompt and the sampler may or may not honour it. How much geometry
the mesh gets is the 3D pane's **Mesh resolution**, which is a different control entirely.

At the top of the pane is **preset**, a picker of four complete shipped recipes
(hand-painted fantasy prop, PS1 low-poly character, sci-fi hero weapon, modern consumable pickup).
Choosing one fills in the prompt and every field it names and then gets out of the way — everything
it set stays visible and editable, and the picker shows "Custom" the moment you change anything. A
preset is a starting point, not a mode.

Beside **Save as...** at the top of the pane is **Reset...**, which puts the whole 2D form back to
its first-launch defaults after a confirm — the prompt, the negative prompt, every guidance select,
the model and LoRA, the reference and the run controls, with a freshly rolled seed. It touches
nothing outside this pane: saved profiles and presets are kept, and the 3D form is left alone.

The composed prompt has no hard length ceiling. CLIP's text encoders stop at 77 tokens, but the app
splits a longer prompt into several chunks on comma boundaries — never mid-phrase — encodes each
separately and joins them, which the image model's cross-attention accepts without complaint. That
is why the preview reports chunks rather than warning about truncation. The soft limit still
applies, though: a longer conditioning sequence dilutes attention, so every shipped fragment is
kept to a few words and your own prompt is best kept to a sentence.

## Models and style LoRAs

The **Advanced** section holds the model choice. Nine base models ship in the registry:

| Model | What it is | Runs at |
| --- | --- | --- |
| SDXL-Turbo (fast) | The default. Small, quick, good enough for most props. | 512 px, 4 steps, guidance 0 |
| SDXL 1.0 + Hyper-SD (best LoRA response) | Full SDXL weights with a step-distillation LoRA fused on. | 1024 px, 4 steps, guidance 0 |
| SDXL 1.0 + Lightning (4-step) | The same weights and the same idea, distilled a different way — an alternative to compare Hyper-SD against. | 1024 px, 4 steps, guidance 0 |
| Playground v2.5 (highest fidelity, slow) | The best-looking output, and correspondingly slow. | 1024 px, 25 steps, guidance 3.0 |
| SDXL 1.0 (full CFG, structural control) | The same weights as the Hyper-SD entry, run the way the checkpoint was trained. | 1024 px, 30 steps, guidance 7.0 |
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
switch for free, with no reload. Four ship:

- **3D render** — a general 3D-render look.
- **3D render (Redmond)** — a second, differently trained take on the same idea.
- **PS1 / low-poly game** — chunky untextured geometry, which pairs naturally with the PS1-era art
  style and is among the easiest things to reconstruct cleanly.
- **Pixel art (pixel-art-xl)** — generates on a pixel grid rather than producing a smooth image that
  is later downscaled into one. It defaults to a strength of 1.2 rather than the usual 0.9: below
  that the output keeps SDXL's anti-aliased gradients, and no downscale recovers a clean grid from
  them. The **Pixel-art sprite** preset picks it together with the LCM base and the NES-era art
  style, whose flat shading and bold silhouette are what survive a reduction.

Choosing one reveals a **Strength** slider, from 0 to 1.5, defaulting to the LoRA's own tuned
weight of 0.9. The slider is hidden entirely when no LoRA is chosen. LoRAs are trained against full
SDXL at 20 to 25 steps with guidance, so they land noticeably stronger on the SDXL entries than on
Turbo — and they do not apply at all on FLUX.2, where the whole picker is disabled.

## Seeds and candidates

Generation is deterministic in its seed: the same form and the same seed produce the same image
every time. The **Run** section, directly above the Generate button, is where you control that.

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

The mesh stage has its own separate seed, on the 3D pane. See
[Mesh parameters](04-generating-meshes.md#mesh-parameters).

## Conditioning on an image

Beyond the prompt, you can steer the image with a picture. Open the **Reference** section, then
either **Choose an image...** or drop a file onto the window. Everything below the picker stays
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

When you are happy, press **Make 3D** on the card (or select the reference and switch to 3D mode).
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

A palette is a file you drop into the palette directory (`palettes/` by default — see
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

The 2D pane's art-style select names console eras rather than abstract styles — **NES era**,
**SNES era**, **PS1 era**, **PS2 era**, **PS3/360 era**, **PS5 era** — and the retro end of that
ladder is what pairs with the pixel exports. NES and SNES deliberately do not put the words "pixel
art" into the prompt: at 512 or 1024 the image model draws fake chunky pixels that then alias under
the real reduction. What they ask for is flat shading and a bold silhouette — the things that
survive being made small.

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

Choosing it changes what the pane offers. The object taxonomy — category, silhouette, rarity and
the rest — describes a *thing*, and a tile has none, so those selects are hidden. What remains is
the surface half: material, condition, palette, setting, genre and era style. The 3D pane's platform
detail is hidden for the same reason.

A tile cannot be made into a mesh, and the app does not offer to: there is no subject to
reconstruct. It also cannot produce the cutout exports, because every one of them lifts a subject
off its background and a tile *is* background — an icon of one would be the whole frame with a matte
guessed over it.

### Whether it actually tiles

When a tile finishes, the app measures its own wrap seam and reports it in the inspector's **Seam**
section. The number is a ratio: how much sharper the difference across the wrap edge is than the
picture's own grain. Around 1 means the seam is indistinguishable from ordinary texture detail;
above 2 the app calls it a visible seam and says so. Both directions are measured and reported
separately, because an image that wraps one way and not the other is not a tile.

A ratio is hard to calibrate against by eye, so the section also offers a wrapped view: the image
rolled by half, which puts what was the wrap edge through the middle of the frame where a
discontinuity is obvious. It is `wrap_preview.png`, a derived export like any other, and the
Export tab offers it too.
