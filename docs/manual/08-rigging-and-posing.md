# Rigging and posing

A rig is a skeleton fitted to a mesh and bound to its surface, so that rotating a bone bends the
model. Warlock fits one automatically, and once a mesh is rigged you can pose it, save those poses,
and render sheets from them.

**This chapter needs Blender.** That means the `rig` extra, which means Python 3.13 — see
[Before you begin](01-before-you-begin.md#what-you-need). Without it every control in this chapter is
hidden rather than greyed out, and `warlock doctor` says why.

## Rigging a mesh

Select a finished mesh and press **Rig this mesh**. It runs as a Blender subprocess, never inside
the app, and produces a `rig.glb` beside your `model.glb`.

Two things happen: a skeleton is fitted, then the mesh is bound to it.

**The skeleton** comes from one of seven templates — humanoid, quadruped, bird, fish, insect,
serpent and biped_tail. A template is a list of named joints at proportions inside a unit box, and
fitting it is *bbox-proportional scaling*. Nothing is learned, nothing is detected. That is worth
knowing because it explains both why it usually works and how it fails: it assumes your mesh is
shaped roughly like the template's idea of that creature.

**The binding** is Blender's automatic bone-heat weighting, with one preparation step — vertices
split at UV seams are welded first, because bone heat treats a split seam as a hole. If heat
weighting fails outright it falls back to envelope weights, and that fallback is recorded rather than
hidden: `rig.json` says `automatic`, `automatic-welded`, or `envelope - needs review`. If you read
the third one, look at the result before trusting it.

For humanoids there is also a deformation-review sheet: a fixed battery of poses — squat, arms
overhead, bent elbow and knee, torso twist — rendered for you to look at. Nothing scores it. It is a
picture, and you are the judge.

## The A-pose trap

Here is the failure that catches nearly everyone, and it looks like a bug.

**The shipped humanoid template is an A-pose** — arms angled down. If your mesh is standing in a
T-pose with arms straight out, fitting that template runs the arm chain diagonally down through the
ribcage. Automatic weights then bind the arms to the chest, and every animation you play barely
moves them.

The symptom is "the arms are welded to the body". The fix is to stop using the template's guess:

- **Measured joints** fits the arm and leg lines to your mesh's own vertex cloud instead of assuming
  proportions. This is the right answer most of the time.
- **Adjust Joints** lets you move joints by hand. Hand-corrected joints always win over both of the
  above.

With the ViTPose weights installed, humanoid joints can also be measured from the reference image's
actual landmarks — all-or-nothing on confidence, so it either uses them or falls back cleanly.

## Posing one mesh

A rigged asset gains a **Pose** panel. Click a joint, a rotation gizmo appears, drag to rotate.
Reset one joint, reset all, or mirror the pose to the other side.

Poses here are **forward-kinematic only**. No IK, no per-joint translation, no scaling. A pose is
exactly a map from joint name to local rotation, and that simplicity is what makes a pose portable
between meshes at all.

Undo is `Ctrl+Z`, and one step is one *gesture* — a whole drag, a preset applied, a mirror, a
reset — not one per frame of mouse movement. A drag that ends where it started records nothing.

The pose editor's history is dropped when you change skeleton or leave the mode, because a step
stores rotations by bone name and replaying it onto a different armature could silently rotate the
wrong joint.

**Save GLB** bakes the posed mesh to a file. It runs on the spot rather than through the queue — a
second or so of Blender, cached afterwards.

## Poser: poses that outlive one mesh

The **Poser** workspace is the same editor over a *bare skeleton*, and what you author there applies
to every mesh that shares that skeleton.

The first thing to know is that **the preview has no mesh in it.** You will see a skeleton floating
in space, exactly one character-height tall, and nothing else. That is correct and not a loading
failure — the point of Poser is that the pose is not about any particular model, so no particular
model is shown.

Poses are saved into a global library per skeleton: save, rename, duplicate, apply, delete. A handful
of presets ship read-only; apply one, adjust it, and **Save as** promotes it into your own library.

Note that **deleting a pose here is permanent.** There is no trash in the pose library, unlike the
asset library.

Applying a library pose to a specific asset **snapshots** it into that asset's own list, marked as
coming from the library. Editing or deleting the library original afterwards does not disturb the
asset. That is deliberate: an asset you posed last month should not change because you tidied your
pose library today.

## Root offset

Selecting the root joint and ticking **Move root** swaps its gizmo from rotation to translation
arrows. This is how you author a crouch or a hop — moving the whole figure rather than bending it.

Two things to know. The offset is stored in **character-height units**, not metres, so the same
crouch applies sensibly to a gnome and a giant. And applying a library pose to an asset **previews
the rotations only** — the offset shows up in the baked GLB and in rendered sheet rows, but not in
the live preview. If you test a crouch by looking at the preview alone, you will conclude it did not
work.

## Poses and clips are stored differently

There are two things in this workspace that both look like "a pose", and they are not
interchangeable.

A **pose** — in the library or on an asset — records each joint's rotation in its own node's frame.
That frame already carries the joint's rest orientation, so a pose is bound to the fitting it was
authored against. Applied to a rig fitted differently, it will not do what it did.

A **clip key** — the animation keyframes used by character sheets — records rotation as a *delta
from the bone's rest orientation*. That is the form that survives a re-fit, which is exactly why
clips are stored that way and poses are not.

You will not usually cross the streams, because the two live on separate surfaces. But it explains
why a saved pose can look wrong on a re-rigged mesh while an animation clip on the same mesh is
fine.

## Editing clips

The clip editor is a section of Poser's sidebar, and it is the tool for authoring the animations
character sheets use. Pick a key, pose the skeleton, **Update key from pose**. Timing is per segment:
frames after this key, looping, easing. Onion skin ghosts the neighbouring keys.

Play scrubs the *interpolated* frames exactly as the renderer will produce them. A scrubbed frame
sits between keys and has nowhere to store an edit, so Update-key refuses while scrubbing rather
than quietly writing to the nearest key.

Saving clips writes to your own data folder and never touches the shipped template, so **Revert** is
always safe.

Two honest notes about the shipped clips. They are **provisional placeholders** rather than finished
animation — good enough to prove the pipeline, not good enough to ship. And easing currently does
nothing at their segment lengths: it needs a step of three frames or more to show, and every shipped
step is one or two. If you tweak easing and see no change, the control is fine and the clip is short.

## Try it

1. Rig a mesh, then read the `weighting:` line and look at the deformation sheet.
2. If the arms look welded to the chest, re-rig with measured joints and compare.
3. In the asset's Pose panel, rotate an arm, mirror it, and save the pose.
4. Open Poser on the same skeleton. Notice the bare armature. Author a wave and **Save as**.
5. Back on the asset, apply that pose from **Library poses**, then delete the library original and
   confirm the asset's copy is untouched.
6. Author a crouch with **Move root**, apply it, and note that the preview shows the rotations but
   not the offset — then **Save GLB** and see the offset appear.

## What to read next

[Building a map](09-building-a-map.md) — Plotter, tile maps, and Tiled interoperability.
