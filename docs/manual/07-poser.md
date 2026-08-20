# Poser

The Poser is where a pose is authored **once, against a skeleton**, rather than against any
particular mesh. Every other way of posing in the app starts from an asset: you rig a mesh, you open
its pose editor, and the pose you save belongs to that mesh. The Poser inverts that. You pick a
skeleton, pose the bare armature, and what you save is offered on every rigged asset that shares it.

That is the whole reason the mode exists. Fitting puts a template's joints in the same relative
place on every mesh it is fitted to, so a rotation authored against the humanoid skeleton means the
same thing on a gnome and on a giant. A pose is a map of joint names to local rotations and nothing
else — see [Posing](06-rigging-and-posing.md#posing) for that contract — which is exactly what makes
it portable.

The Poser needs Blender, for the reason given under [When Blender is missing](#when-blender-is-missing).

## Opening the Poser

Pick **Poser** in the rail. The left panel holds the skeleton picker; choosing a skeleton loads a
**preview armature** into the viewport — the bare bones, with no mesh around them.

The preview is built by the same Blender code path that builds a real rig, and it is exactly one
character-height tall. Both halves matter: what you pose is built the way a rig is built, and it is
scaled the way every bake will scale it, so nothing about the preview is a stand-in for something
you cannot see until later.

It is built once per skeleton and cached, so the first time you open a template takes a moment and
every later time is immediate.

**New pose** clears the editor to the rest skeleton and starts a fresh pose. Over unsaved work it
asks first.

## Posing a skeleton

Posing works exactly the way the pose editor on an asset does:

- **Click a joint** to select it. A rotation gizmo appears on it; drag to rotate. The panel names
  the selected joint, or says "Click a joint to rotate it" when there is none.
- **Reset joint** returns the selected joint to its rest rotation. It is disabled until you have
  selected one, and says why on hover.
- **Reset all** returns every joint.
- **Mirror** copies the pose across the body's centre line. It is hidden for skeletons with no
  mirror pairs — the serpent and the fish — where it could only do nothing.

Poses are forward-kinematic only: no inverse kinematics, no translation on ordinary joints, no
scaling. The one exception is the root, which is [its own section](#moving-the-root) below.

## Undo and redo

`Ctrl+Z` undoes and `Ctrl+Y` — or `Ctrl+Shift+Z` — redoes. This works in the Poser and in an asset's
pose editor alike, because both are the same editor underneath.

**The unit of undo is the gesture, not the frame.** One whole gizmo drag is one step however many
frames of mouse movement it took, and so is applying a preset, a mirror, a reset and a joint move. A
drag that ends where it started records nothing at all, so an accidental nudge-and-return does not
leave a step you have to undo past.

Undoing back to the point you last saved from leaves the session **clean** rather than still asking
about unsaved changes — the history knows where the last save was, so retreating to it is genuinely
a return to saved state.

The history belongs to the editing session. It is dropped when you leave the mode or load a
different skeleton, and deliberately: a step holds rotations by bone name, so replaying one onto a
different armature would find whichever bones happened to share a name and silently mean something
else.

Your editing session itself survives switching modes, the way a document left open in the
[Inker](09-inker.md) does. Only quitting, switching skeletons, or loading another pose over it asks
about unsaved changes.

## Moving the root

Select the root joint and tick **Move root** to swap its rotation gizmo for translation arrows.
Dragging them offsets the whole pose — a crouch that actually lowers, a leap that leaves the ground.

The offset is stored **in character heights**, not in world units, which is what makes it portable
the way the rotations are: a half-height offset lifts a gnome by half a gnome and a giant by half a
giant.

Two limits come with it, both about where the offset shows up:

- Applying a library pose to an asset previews the **rotations only**. The offset is real, but it
  appears in the baked GLB and in sprite sheet rows rather than in the inspector's preview.
- An animated sheet clip cannot interpolate a root offset yet. A clip whose endpoint poses carry one
  is refused by name rather than rendered subtly wrong — see
  [Sprite sheets](08-sprite-sheets.md).

## The pose library

**Save** writes over the pose you are editing; **Save as** asks for a name and adds a new one. Both
write into the **global** pose library, per skeleton, rather than into any asset's own saved poses.

The library is listed in the left panel under the skeleton picker, and the pose currently loaded in
the editor is drawn in the accent colour so you can tell which row you are working on. Each row
carries four actions:

| Action | What it does |
| --- | --- |
| Apply | Loads the pose into the editor, over whatever is there. |
| Rename | Renames it in place. |
| Duplicate | Copies it under a new name, so a variant does not cost you the original. |
| Delete | Removes it permanently. |

A filter box appears above the list once it is long enough to need one.

**Delete is permanent.** The pose library has no trash — unlike the asset library, which does — so
that is the one action here that asks first.

Poses saved here are offered on every rigged asset with a matching skeleton: the inspector's
**Pose** panel grows a **Library poses** section listing them. **Apply** there copies the pose into
that asset's own saved list as a snapshot, marked `(library)`, which then behaves exactly like a
pose you saved by hand on that asset.

**The snapshot is the point.** Editing or deleting the library pose afterwards never changes what an
asset already carries, so a bake you liked stays reproducible forever.

## Shipped presets

Below the library is **Shipped presets** — poses that ship with the app for the current skeleton.
They are read-only by design. A preset is a starting point, and the way to keep a version of one is
to apply it, adjust it, and **Save as**, which promotes your edit into the library beside your own
poses.

The section is absent entirely for a skeleton with no presets, rather than drawn empty.

## When a pose file goes wrong

A pose file that has gone wrong on disk — truncated, hand-edited into the wrong shape, or simply not
JSON any more — costs itself and nothing else.

It stays in the list, so there is a row to act on. Applying, renaming or duplicating it says the
record could not be read. And **Delete always works**, because a pose you cannot read is exactly the
one you most need to be able to remove. One broken file never takes the library down with it.

## When Blender is missing

The Poser builds its preview armature with Blender, so without it the mode has nothing to show. It
says "Posing needs Blender, which is not installed" and offers nothing, rather than drawing controls
that could not do anything.

Blender ships CPython 3.13 wheels only, so on any other Python version the optional extra installs
nothing at all. See [When rigging is
unavailable](06-rigging-and-posing.md#when-rigging-is-unavailable) for the full list of what that
takes with it, and [Installation](18-installation.md) for how to get it.
