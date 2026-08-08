# Rigging and posing

Rigging fits a skeleton to a finished mesh and skins it, so the mesh can be posed. Posing sets each
joint's rotation and saves the result under a name. Both are optional, both need Blender, and both
are covered here.

## Templates

Warlock Studio does not invent a skeleton for your mesh. It fits one of seven shipped **templates**:

| Template | For |
| --- | --- |
| humanoid | Two arms, two legs, upright. |
| quadruped | Four legs, horizontal spine. |
| bird | Wings and a two-legged stance. |
| fish | A spine and fins, no limbs. |
| insect | Six legs. |
| serpent | A long chain of spine joints, no limbs. |
| tailed biped | A humanoid with a tail. |

Each template is a small JSON file listing named joints at normalised positions inside a unit
bounding box. Fitting scales those positions onto the measured bounding box of your mesh — nothing
is learned and nothing is detected. That makes it fast and completely predictable, and it also makes
it **approximate on purpose**: a subject whose proportions differ from the template's will get
joints that are close rather than correct.

Because the fitted positions are approximate, they are written into `rig.json` alongside the rig, so
they can be corrected later without starting over. That is what **Adjust joints** does, described
under [Posing](#posing).

## Rigging a mesh

There are three ways to rig:

- Press **Rig** on a finished mesh's card in the library.
- Open the inspector's **Rig & Pose** tab on an unrigged mesh and press **Rig this mesh** in the
  **Pose** panel.
- Tick **Rig when the mesh lands** in the 3D pane's **Rig** section before generating, and pick a
  skeleton. The rig is then queued automatically as soon as the mesh finishes.

Rigging is a queued job like any other, and it takes minutes of CPU rather than seconds — skinning
is not cheap. It runs Blender as a separate process, never inside the app: Blender's Python module
is process-global and can take the whole interpreter down on the kind of non-manifold geometry
reconstruction sometimes produces.

That same geometry breaks Blender's bone-heat weighting outright. When it does, the app catches the
failure and falls back to envelope weights rather than failing your job, and records which method
was used in `rig.json`. Envelope weights are cruder — expect more distortion at joints — so the
record is worth checking if a pose deforms strangely.

You do not have to open the file for that. The **Rig & Pose** tab names the method at the top —
`weighting: automatic`, `weighting: automatic-welded`, or `weighting: envelope - needs review` in
warning colour, with the reason on hover. It reads the rig beside the selected mesh, so it answers
for the asset you have selected rather than for the rig job that produced it.

Under it, when the rig has one, is the **deformation review** sheet: the skeleton put through a
battery of test poses — a squat, arms overhead, elbow and knee at 90°, a torso twist — and rendered
as one image. It is there to be looked at, and nothing scores it: what you are looking for is skin
tearing or collapsing at the joints, which is a judgement no measurement in the app has earned yet.
The battery ships for the humanoid skeleton only.

The result is `rig.glb` and `rig.json`, written **beside the `model.glb` they were fitted to**, not
into the rig job's own directory. The rig belongs to the mesh. Rigging a mesh a second time replaces
the previous rig; cancelling a re-rig leaves the previous one intact.

### Where the joints come from

By default a skeleton is fitted by proportion: the template's landmarks are scaled onto the mesh's
bounding box, so a shoulder lands where the template says a shoulder is in a body of that height.
That is right for a subject standing in a T-pose and increasingly wrong the further from one it is.

For **humanoid** rigs there is a better first guess, and it is on by default when the pose model is
installed. Warlock measures the subject's joints off the reference image the mesh was reconstructed
from — the premise being that the mesh *is* that picture in three dimensions — and uses those as
the template. They are still scaled onto the mesh by the same fitter, so nothing else about rigging
changes; what changes is where the fitter starts from.

Two things about it are worth knowing:

- **It is all or nothing.** If any landmark is unconvincing — low confidence, missing, outside the
  subject's silhouette, or a figure whose knees came out above its hips — the whole measurement is
  discarded and the template is used. A skeleton with a measured arm and an assumed shoulder is not
  half-right; it is inconsistent in a way nothing later can detect.
- **Depth is always the template's.** One image fixes left–right and up–down and says nothing about
  front–back.

`rig.json` records which was used, so a rig can be told apart afterwards. Without the pose model the
bbox fit is used and nothing about your rigs changes — see
[Model weights](13-installation.md#model-weights) for the download. Set `WARLOCK_POSE_FIT=off` to
force the template everywhere, whatever is installed; it is a kill switch rather than an opt-in,
because the measurement already refuses itself whenever it is unsure.

Joints you have corrected by hand always win over both. See **Adjust joints** under
[Posing](#posing).

## Posing

Once a mesh is rigged, the inspector's **Pose** panel offers **Edit pose**. Pressing it swaps the
viewport from the mesh to the rig and enters pose mode.

In pose mode:

- **Click a joint** to select it. A rotation gizmo appears on it; drag the gizmo to rotate.
- **Reset joint** returns the selected joint to its rest rotation; **Reset all** resets every joint.
- **Mirror** copies the pose across the body's centre line. It is hidden for skeletons with no
  mirror pairs, such as the serpent and the fish, where it could only do nothing.
- A **preset** picker offers the shipped pose library for this skeleton, when one exists. Because
  fitting puts a template's joints in the same relative place on every mesh, a pose authored against
  one humanoid applies to every other humanoid.

Poses are **forward-kinematic only**. A pose is exactly a map of joint names to local rotations —
there is no inverse kinematics, no translation, and no scaling. That is the whole contract, and it
is what lets a pose be a small file that applies to any mesh sharing the skeleton.

**Save pose...** asks for a name and stores it. Saving while a saved pose is loaded **replaces**
that pose rather than adding a near-duplicate, so refining an "idle" leaves you with one idle rather
than two that differ by a shoulder.

Saved poses are listed below, each with three actions:

- **Apply** loads it into the viewport.
- **Save GLB...** writes a posed GLB — the mesh frozen in that pose — wherever you choose. It is
  baked by Blender the first time you ask and cached afterwards, so the first request takes about a
  second and later ones are immediate. It runs off the job queue deliberately: a one-second job
  should not have to wait behind a two-minute reconstruction.
- **Delete** removes the pose, and its cached GLB with it.

A pose lives only in the viewer until it is saved. Every route out of pose mode — leaving edit mode,
switching to another asset, starting a comparison, closing the window — asks before discarding
unsaved changes. The banner at the top of the panel says "unsaved changes" whenever there are any.

**Adjust joints** is the second mode over the same markers, and it exists because fitting is
approximate. It shows the rest skeleton, lets you drag joints to where they should actually be,
counts how many you have moved, and **Apply joint positions** re-rigs the mesh with the corrected
skeleton. That is a real re-rig — minutes of CPU, queued like the first one — not an edit of the
existing rig. **Revert** undoes your unapplied moves; **Back to posing** returns. Entering joints
mode asks first if you have an unsaved pose, since a leftover rotation would put the markers where
the posed bones are rather than where the rest skeleton is.

Rigged meshes are also what [Sprite sheets](06-sprite-sheets.md) render rows from.

## When rigging is unavailable

Rigging, posing, sprite sheets and FBX export all need Blender, which the app installs as an
optional extra.

Blender's Python module ships **CPython 3.13 wheels only**. On any other Python version the extra
installs nothing at all. When that happens:

- `warlock doctor` reports rigging as unavailable.
- The app hides the rig controls entirely rather than greying them out — a greyed control implies it
  could be switched on from where you are standing, and this one cannot.
- The **Pose** panel says "Posing needs Blender, which is not installed" instead of telling you to
  press a button that is not on screen.
- The FBX export button explains itself with "needs Blender".

Everything else in the app works unchanged. Nothing about generating references or meshes depends
on Blender.
