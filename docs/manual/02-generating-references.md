# Generating references

A reference is the picture the mesh will be reconstructed from. Everything in this chapter lives in
the **2D reference** mode's settings pane, in the upper half of the left sidebar.

## The prompt

The large text box under **Prompt** is where you describe the object. Write the subject and nothing
else — "a weathered wooden crate bound with iron", "a compact energy rifle with panel seams". You do
not need to ask for a plain background, a single object, or a studio render: the app wraps whatever
you write in a fixed template that already asks for all of that, because those are the properties
that make an image reconstruct cleanly.

The prompt is capped at 1000 characters and a counter under the box shows how much you have used.
The counter turns amber inside the last hundred characters.

Beside the box, a **Recent** button opens your last twenty prompts, most recent first and
deduplicated. Picking one replaces what is in the box. The history is per session and per prompt
text only — if you want a whole recipe back, use **Copy settings to form** from a job's overflow
menu instead, which is described in [Rerun and promotion](07-library-and-jobs.md#rerun-and-promotion).

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

## Guidance fields

Under the prompt are eleven optional selects, grouped by what they describe:

| Group | Fields |
| --- | --- |
| Subject | category, genre, setting, silhouette |
| Style | art style, palette, mood, rarity |
| Surface | material, condition, emissive |

Each one is a small closed vocabulary — genre offers Fantasy, Sci-fi, Modern, Post-apocalyptic,
Horror and Cartoon; condition offers Pristine, Worn, Damaged, Ancient, Rusted, Overgrown and Burned;
and so on. Every entry maps to a short prompt fragment of two to four words, and the fragments are
appended to your prompt in a fixed order that reads like a sentence: category, silhouette,
material, condition, rarity, emissive, setting, genre, mood, art style, palette, platform.

Leaving a field unset simply omits its fragment. Nothing is filled in for you except the platform
default.

Two of these have effects beyond the text. **Category** supplies a default physical size when you
do not give one — a prop is 0.4 m, a weapon 1 m, a character 1.8 m, a vehicle 4.5 m, an environment
piece 8 m, a consumable 0.15 m. And **platform detail**, which sits on its own below the groups,
is a hint about how much fine detail to draw. It is *not* the mesh resolution; that is the 3D
pane's own platform control, and the marker beside this one says so.

Below the guidance groups you will also find **preset**, a picker of four complete shipped recipes
(hand-painted fantasy prop, PS1 low-poly character, sci-fi hero weapon, modern consumable pickup).
Choosing one fills in the prompt and every field it names and then gets out of the way — everything
it set stays visible and editable, and the picker shows "Custom" the moment you change anything. A
preset is a starting point, not a mode.

The composed prompt has no hard length ceiling. CLIP's text encoders stop at 77 tokens, but the app
splits a longer prompt into several chunks on comma boundaries — never mid-phrase — encodes each
separately and joins them, which the image model's cross-attention accepts without complaint. That
is why the preview reports chunks rather than warning about truncation. The soft limit still
applies, though: a longer conditioning sequence dilutes attention, so every shipped fragment is
kept to a few words and your own prompt is best kept to a sentence.

## Models and style LoRAs

The **Advanced** section holds the model choice. Four base models ship in the registry:

| Model | What it is | Runs at |
| --- | --- | --- |
| SDXL-Turbo (fast) | The default. Small, quick, good enough for most props. | 512 px, 4 steps, guidance 0 |
| SDXL 1.0 + Hyper-SD (best LoRA response) | Full SDXL weights with a step-distillation LoRA fused on. | 1024 px, 4 steps, guidance 0 |
| Playground v2.5 (highest fidelity, slow) | The best-looking output, and correspondingly slow. | 1024 px, 25 steps, guidance 3.0 |
| SDXL 1.0 (full CFG, structural control) | The same weights as the Hyper-SD entry, run the way the checkpoint was trained. | 1024 px, 30 steps, guidance 7.0 |

The sampler settings travel with the checkpoint and are not yours to set. They are part of the
model's identity: a four-step distilled model run at 25 steps with guidance produces mush, and
Hyper-SD degrades silently without the right timestep spacing.

Two consequences are worth knowing. A **negative prompt** is only encoded when guidance is above
1.0, so it does nothing on the two four-step entries — pick "full CFG" or Playground if you want it
honoured. And **structure control** (see [Conditioning on an image](#conditioning-on-an-image)) is
only offered on those same two, because a ControlNet at guidance 0 fights the hint instead of
following it.

Only one base model is resident on the card at a time — a 32 GB card holds the reconstruction engine
plus a single SDXL-class pipeline, not two — so switching between models between jobs costs a
reload of several seconds. Any model whose weights are not on disk is still listed, with
"— weights missing" appended to its name, so you learn at pick time rather than at job-failure
time. Run `warlock doctor` for the exact download command.

**Style LoRAs** are the opposite: they are adapters on whatever pipeline is already resident and
switch for free, with no reload. Three ship:

- **3D render** — a general 3D-render look.
- **3D render (Redmond)** — a second, differently trained take on the same idea.
- **PS1 / low-poly game** — chunky untextured geometry, which pairs naturally with the low-poly art
  style and is among the easiest things to reconstruct cleanly.

Choosing one reveals a **Strength** slider, from 0 to 1.5, defaulting to the LoRA's own tuned
weight of 0.9. The slider is hidden entirely when no LoRA is chosen. LoRAs are trained against full
SDXL at 20 to 25 steps with guidance, so they land noticeably stronger on the SDXL entries than on
Turbo.

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
[Mesh parameters](03-generating-meshes.md#mesh-parameters).

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

If the image is nearly right, you can fix it by hand: **Open in Paint** on the viewport toolbar
opens the reference as a layered document, and saving writes it back in place. See
[Pipeline bridges](06-paint.md#pipeline-bridges).

When you are happy, press **Make 3D** on the card (or select the reference and switch to 3D mode).
That carries the reference and everything it recorded into the mesh stage, where you can override
the mesh-side settings before committing. The next chapter,
[Generating meshes](03-generating-meshes.md), picks up from there.
