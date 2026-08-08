# Extending Warlock Studio

Most of the things you might want to add — another image model, another style, another skeleton,
another chapter of this manual — are data rather than code, and the places they are declared are
deliberately the only places they are declared. This chapter is a tour of those extension points and
the rules that keep them from leaking into the rest of the app.

## Adding an image model

An image model is a registry entry, not a directory on disk. `models.py` owns `BASE_MODELS`, and a
`BaseModel` carries both the checkpoint's directory name and the settings it has to be run at:
image size, step count, guidance scale, weight variant, scheduler, and any always-on
step-distillation LoRA fused on at load.

Those settings live with the checkpoint because they are properties of the checkpoint. A four-step
distilled model run at twenty-five steps with classifier-free guidance produces mush, and Hyper-SD
degrades quietly unless its scheduler uses trailing timestep spacing. Neither is a preference, and
neither is something the person writing a prompt should have to know.

Adding one means adding an entry. The fields worth thinking about:

- `dir_name` — resolved under `WARLOCK_T2I_ROOT`, so the model is found by name rather than by path.
- `image_size`, `steps`, `guidance_scale` — the sampler settings the checkpoint was distilled or
  trained for.
- `scheduler` — a key into the scheduler table in `pipelines/text2image.py` (`ddim_trailing` for
  Hyper-SD, `lcm` for a consistency adapter), or left unset to keep whatever the checkpoint's own
  config specifies. A name that table does not know raises — with the weights already in VRAM,
  which is why every shipped entry's name is covered by a test.
- `base_lora` — a step-distillation adapter loaded under a reserved adapter name, so it can never
  collide with a style LoRA key.
- `controlnet` — stated explicitly rather than inferred from the guidance scale, so a future
  checkpoint does not silently become "controllable" by clearing a threshold nobody qualified it
  against.
- `download` — the exact one-time `hf download` command, which is what the diagnostics show when
  the weights are absent.

What you should not do is hardcode any of those numbers in `pipelines/text2image.py`. The pipeline
reads them off the spec, which is what lets a job name a model and get the right sampler settings
without the UI knowing anything about either.

Only one base model is resident at a time. Selecting a different one unloads the previous pipeline
before building the next, because the card holds the reconstruction engine plus one SDXL-class
pipeline and not two. See [VRAM modes](14-configuration.md#vram-modes).

A registry entry is also the right answer when `WARLOCK_T2I_DIR` is not — that variable redirects
where the built-in `turbo` entry loads from and changes nothing about how it is run. See
[Using a different image model](14-configuration.md#using-a-different-image-model).

## Adding a style LoRA

A style LoRA is a `STYLE_LORAS` entry plus a `.safetensors` file under the `loras/` subdirectory of
the model root. The entry carries the filename, a label, a default weight and the trigger words the
adapter was trained with, along with its own download command.

Style LoRAs are the opposite of base models in the way that matters most: they are adapters applied
to whatever pipeline is already resident, switched per job without a reload, so changing style
between jobs is free where changing base model is not.

The trigger words are the detail most likely to be put in the wrong place. They are prepended to the
composed prompt alongside the prompt template, and they are deliberately absent from the guidance
module's prompt fields. A trigger is model-facing scaffolding — it exists because the adapter was
trained to answer to it — not creative direction, and the taxonomy in `guidance.py` is meant to hold
only the latter. Model keys and LoRA keys are validated in `guidance.py` so that a bad value
produces one kind of error from one place, but they contribute nothing to the composed prompt: its
text is byte-identical with and without any of them.

A missing LoRA file is skipped at load time rather than failing the job, and the diagnostics name it.
See [Models and style LoRAs](03-generating-references.md#models-and-style-loras) and
[Optional image models and style LoRAs](13-installation.md#optional-image-models-and-style-loras).

## Adding a palette

There is nothing to add. A palette is a file in the palette directory (`palettes/`, or wherever
`WARLOCK_PALETTE_DIR` points), in Lospec's `.hex` or GIMP's `.gpl` format, and the export's palette
control lists whatever is there — no registry entry, no code, no restart. That is deliberate: a
palette is art direction, and the registry pattern the models use exists for things that have to be
downloaded, checked for and reported on.

The one rule worth knowing is that freshness is keyed on a palette's *contents* and not its
filename, so editing one in place re-derives every export that used it, which is what makes working
on a palette feel like working on a file.

## Adding a skeleton

A skeleton is a JSON file in `src/warlock/templates/`, and adding one is the entire procedure — no
bone list in `blender_worker.py`, no branch anywhere that names a template.

Each file declares a key, a label, a root bone, a list of bones — each with a name, a parent and a
head and tail position — and a list of `mirror_pairs`. The positions are normalised landmarks in a
unit bounding box, expressed in
Blender's axes: `+X` is the subject's left, `-Y` is forward, `+Z` is up. The `x` and `y` components
span `-0.5` to `0.5` about the box centre, and `z` spans `0` at the floor to `1` at the top.

`rigging.fit_template` scales those landmarks onto the measured bounding box of the mesh being
rigged. The fit is bbox-proportional and deliberately approximate — a joint lands where the
proportions say it should, not where anatomy says it should. That is why the fitted positions are
written into `rig.json`: a later adjustment pass can correct a joint without re-solving the rig, and
the record of where each joint actually ended up is the input it needs.

For the `humanoid` template there is a second source of landmarks, and it does not replace the
fitter — it replaces the *template*. When a pose model is installed, `pipelines/pose2d.refit` reads
the subject's joints off the reference image and returns them in exactly the template's own format:
the same names, the same parentage, still normalised into a unit box. `fit_template` then scales
those onto the mesh exactly as it scales the shipped ones, so bounding-box scaling stays owned by
one function and nothing downstream learns a second way a joint can be placed. Depth is always the
template's: one view fixes `x` and `z` and says nothing about `y`.

Any doubt refuses the whole measurement rather than part of it — a landmark below the confidence
floor, one the detector never produced, a figure whose knees come out above its hips, a landmark
outside the subject's silhouette. A skeleton half-measured and half-assumed is not partly better; it
is internally inconsistent in a way nothing downstream can detect. `rig.json` records which source
was used in its `fit` field.

Extending this to another template means a detector for that anatomy and a mapping onto its
landmarks: `pose2d.POSE_FIT_TEMPLATES` is the list, and it names `humanoid` alone because COCO-17 is
a human keypoint set. A quadruped needs an AP-10K model and its own mapping. Adding a template
without one is entirely normal — it simply gets the bbox fit, which is what every template got
before this existed.

Mirroring is not inferred from the geometry. `mirror_pairs` is an explicit array of two-element
`[left, right]` name pairs, and it is the only thing that makes the pose editor's Mirror control do
anything: `rigging.mirror_pose` copies each posed bone onto its named partner reflected, and a bone
that appears in no pair is left exactly as it is, on the assumption that it sits on the mirror plane.
The list is carried into `rig.json`, which is where the viewer reads it from, and the Mirror button
is hidden entirely when it is empty — so a template that omits the field loads and rigs perfectly
well and simply cannot be mirrored. The field is optional in the parser and defaults to empty, which
means forgetting it costs you a feature rather than an error. Omit it deliberately, as `serpent.json`
does, or list every symmetric limb, as `humanoid.json` does.

Two conventions are worth honouring for consistency with the templates already there. Forward is
`-Y`, which is what makes column zero of a sprite sheet the front view. And limbs you intend to pair
should be placed mirror-symmetrically about `X`, because the reflection `mirror_pose` applies assumes
that plane.

See [Templates](05-rigging-and-posing.md#templates).

## The derived-params rule

A job's parameters mix two kinds of thing: what you asked for, and what the app worked out. The
second kind is listed in `DERIVED_PARAMS` in `service/validation.py`, and that list is the single
place a rerun or a promotion consults when deciding what to strip.

The rule is short. Anything the worker records about a *finished* job's artifacts — the composed
prompt, the applied transform, the scale factor, the mesh audit, the mesh report, the optimiser
result, the weighting method, the bone count, the sheet id and its cells, the reference report, the
control hint and the recipe — belongs on that list. If it is not there, a rerolled job inherits it,
and you get a fresh mesh wearing a quality verdict about a mesh that no longer exists.

There is a deliberate counterpart. The conditioning selection — the IP-Adapter, the ControlNet and
their strengths — is an *input*, so it survives a reroll. It does not survive a promotion or a
remesh, because those are image jobs that never run the image model at all, and a row claiming an
adapter that cannot have run is a lie about provenance.

Inputs are bounded at the door rather than deep in the pipeline: an upload is size-checked before it
is decoded and pixel-checked from its header before pixels are allocated, prompts are length-capped,
and every service entry point that accepts a seed range-checks it. See
[Rerun and promotion](09-library-and-jobs.md#rerun-and-promotion).

## Pure-module boundaries

Several parts of the app are pure by rule, and the rule is always the same: nothing below the line
imports imgui, moderngl, pygame or the service layer. Three places do this, for three versions of
the same reason.

**The Inker engine.** `studio/inker/` holds blend arithmetic, layers with stable uids, typed undo
edits, selection masks, brush stamps, gradients and OpenRaster I/O, and none of it knows a window
exists. `studio/inker_mode.py` is the only layer that knows about jobs and task threads. That is
what makes every rule about pixels assertable headlessly — and there are a lot of such rules, since
undo is addressed by layer uid rather than index precisely so that an undo issued after a reorder
still lands on the layer the edit was made to.

**Sheet planning.** As described in [Sheet planning](18-pipelines.md#sheet-planning), the grid is
decided in a module with no Blender and no GPU, so the layout can be tested exhaustively and the
preview cannot drift from the render.

**This manual.** `studio/manual/loader.py` finds chapters and reads them; `studio/manual/parser.py`
turns markdown into typed blocks. Neither imports imgui, so every rule about what a chapter may
contain is a headless test. The renderer that draws those blocks in the app is a separate thing
entirely and holds no opinions about syntax.

The pattern generalises. If a rule is worth enforcing, put the thing it governs somewhere a test can
reach without a display.

## Writing manual chapters

The chapters are markdown files in `docs/manual/`, named `NN-name.md`. They are readable on GitHub
as ordinary markdown and rendered in the app by the parser above; the loader prefers a packaged copy
if one is installed and otherwise reads the repository's copy directly.

A chapter's first line is its H1, and that H1 is its title everywhere — the index, the in-app
navigation, the help button. There is exactly one per file.

The markdown accepted is a strict subset:

| Construct | Accepted |
| --- | --- |
| Headings | `#` through `####` |
| Inline | `**bold**`, `*italic*`, `code`, `[text](target)` |
| Code | fenced blocks, with an optional language |
| Lists | `-` and `1.`, nested by two spaces |
| Tables | pipe rows with a `---` separator row |

Everything else is rejected, including images, raw HTML and blockquotes. Two consequences catch
people out. A bare angle bracket followed by a letter reads as HTML, so a placeholder like a job id
in angle brackets has to sit inside a code span. And a table cell cannot contain a pipe character,
because the row is split on pipes before anything else is parsed.

The strictness is a bargain rather than a preference. A construct outside the subset raises an error
at parse time, and `tests/manual/test_docs.py` parses every chapter — so a chapter that drifts
outside the subset fails the test suite rather than rendering wrong for a reader who has no way to
know it was ever meant to look different. The same test resolves every cross-link and every anchor
in every file, and checks that the index links each chapter exactly once. A link to a heading you
renamed is a failing test, not a dead link discovered a year later.

Anchors follow the GitHub convention, which the parser reproduces: lowercase the heading, drop
punctuation, turn runs of whitespace into single hyphens. Link within a chapter with `#anchor`,
across chapters with `file.md#anchor`.

The one piece of wiring outside the files themselves is `HELP_TARGETS` in
`studio/manual/targets.py`. It maps a pane's key to the chapter and heading that documents it, which
is what lets the help affordance in a panel open the manual at the relevant place instead of at the
top. Adding a chapter that documents a pane means adding its entry there; that map is the only
coupling between the UI's structure and the manual's, and keeping it in one table is what stops
chapter names being scattered through pane code.
