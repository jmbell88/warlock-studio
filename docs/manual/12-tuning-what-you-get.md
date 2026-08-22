# Tuning what you get

[Your first asset](02-your-first-asset.md) used defaults for everything. This chapter is about the
controls under **Advanced**, what each is for, and which of them are worth your attention.

The short version: prompt and seed do most of the work, style profiles save you retyping, and the
conditioning controls are the ones that reward learning properly.

## Seeds

A seed makes generation repeatable. Same prompt, same settings, same seed, same picture.

Next to the seed field are **Reroll**, which picks a fresh random one, and **Lock seed**. Unlocked —
the default — every submit rerolls, which is what you want while exploring. Locked, the seed is
reused, which is what you want when you are changing *one other thing* and need the comparison to
mean something.

That is the whole discipline: unlock while hunting, lock while comparing.

**How many** asks for several candidates at once from different seeds. Generating four and picking
one is usually a better use of the same time than generating one and rerolling it four times,
because you see the spread rather than a sequence.

The mesh stage has its own separate seed, and its own **Candidates** control offering up to three
reconstruction attempts to pick the best from. Mesh geometry varies a great deal between seeds — more
than most people expect — so this is often the more valuable of the two.

## Style profiles

A profile saves the *look* half of the form: base model, style LoRA and its weight, negative prompt,
and optionally a style anchor image and its strength. It does not save your prompt, because the
prompt is the subject and the profile is the style.

Save one from the current form, set it active, and every new job starts from it. It is the answer to
"how do I keep forty props looking like they belong together".

The profile manager opens as a sheet over whatever you are doing, from Create's reference settings or
from the command palette.

## Models and LoRAs

The default base model is SDXL 1.0 run at **full CFG**, and it is the default because it measured
better than the alternatives as a source for reconstruction, not because it is fastest. The faster
distilled recipes — Hyper-SD, LCM, Lightning — are the same weights run differently and cost only a
small adapter each. Turbo, Playground, Juggernaut, DreamShaper and FLUX.2 klein are separate
checkpoints.

Style LoRAs are filtered to those that fit the chosen base's architecture, so you cannot pick an
incompatible pair.

**Leave the style strength alone unless you have a reason.** Selecting a LoRA seeds its own tuned
weight, and those weights are not decorative — they were measured per adapter. One of them runs at
0.0625, because it was trained in a way that makes peft restore a sixteen-fold scale internally; at a
"normal" 0.9 it produces black frames. A flat default across all LoRAs would be wrong for most of
them.

## Conditioning on an image

Two named ways to hand the model a picture, and they do different jobs.

**Appearance** takes the *look* of a reference — colour, material, feel — without copying its shape.

**Structure** takes the *shape*. It is only offered on base models that run real guidance, because a
model at guidance zero has nothing for a structural signal to steer. It has a strength and an
**until** value, the latter being how far through sampling the guidance keeps applying — releasing
it early gives the model room to add detail the control image did not have.

Both need a reference image loaded, and both refuse clearly if you have not.

## The negative prompt

Worth stating plainly: **the negative prompt does nothing unless the base model runs guidance above
1.0.** On a distilled, guidance-zero recipe it is inert. The pane says so beside the field. If you
have carefully tuned a negative prompt and it seems to have no effect, check which base you are on
before concluding anything about the words.

## Prompt enrichment

The expander appends aesthetic vocabulary to your prompt using a small local language model —
CPU-only, and constrained to a fixed token whitelist so it can only add descriptive words and can
never rewrite your subject.

Three settings: off, *3D asset* (keeps single-subject framing, right for references), and
*General 2D* (drops it, for pictures that are not going to be reconstructed).

Tiles and tile grids are never expanded, whatever this is set to — the composition rules for a
lattice are not the ones the expander knows.

There is a preview of the prompt actually sent, with its token count, which is the fastest way to see
what the expander did and whether you are near a length limit.

## Mesh-side settings

**Mesh resolution** sets the reconstruction's geometry resolution. **Size** sets the real-world scale
in metres, where zero means "keep whatever the reference implied". **Background removal** chooses the
matting method. **Normalise the reference** recentres and rescales the subject before upload.

**Triangle budget** exists but currently offers only one tier, so you will usually not see it. The
alternatives have not qualified against the test corpus — zero of twenty meshes passed — so they are
not offered rather than being offered and disappointing.

## Re-texturing

A finished mesh can be repainted. Describe the surface you want, and six sampling passes around two
Blender renders produce a new texture on the same geometry.

Two settings matter. **Strength** — from 0.30 to 0.85, defaulting to 0.65 — is how far it departs
from what is there. And **Anchor to geometry**, which is **on by default and should stay on**: it
depth-tests each texel so colour cannot smear through an overhang onto a surface that has no line of
sight to it. Turning it off was measured as smearing colour into essentially every hidden region;
with it on, almost none.

Re-texturing invalidates only the exports that carry the skin. Your rig, poses and sheets survive.

## One thing to watch out for

There is a known defect worth knowing about while you are experimenting with base models: loading an
image model does not return all of its host memory when it unloads. Switch base models repeatedly in
one long session and system commit climbs until the app starts refusing its own work.

It is not subtle when it happens — the refusals say so. The fix is to restart the app, which people
tend to do unprompted anyway. It is on the list.

## Measuring instead of guessing

Everything above changes what comes out, and it is very easy to convince yourself that a setting
helped. If it matters, the app has the machinery to check: launch a sweep varying one axis, judge the
results blind, and read the Axis verdicts.
[Judging what you made](04-judging-what-you-made.md#sweeps-and-what-verdicts-add-up-to) covers it.

## What to read next

[Putting it in a game](13-putting-it-in-a-game.md) — exports, formats and the honest state of
interoperability with other tools.
