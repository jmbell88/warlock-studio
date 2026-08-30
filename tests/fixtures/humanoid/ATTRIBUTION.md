# CesiumMan, vendored as a test fixture

## What this is

`cesium_man.glb` is **third-party art**: the `CesiumMan` sample from the Khronos
glTF sample models, added on 2026-08-30 as the repository's first textured,
rigged humanoid.

| | |
|---|---|
| Upstream | <https://github.com/KhronosGroup/glTF-Sample-Models> |
| Model | `CesiumMan` (`2.0/CesiumMan/glTF-Binary/CesiumMan.glb`) |
| Author | Cesium (<https://cesium.com/>) |
| Licence | CC-BY 4.0 |
| Added | 2026-08-30 |

**Attribution is a licence condition, not a courtesy.** CC-BY 4.0 permits
commercial use and redistribution — including of sprites rendered from it —
provided the author is credited. Anything published that was rendered from this
mesh credits Cesium.

This file exists because the alternative was repeating a mistake the repository
is still paying for. `examples/` held Nintendo material and ULPC art with *"no
attribution file anywhere"*, and undoing that needs a history rewrite and a
force-push of every commit. A licensed asset enters this tree with its
attribution or it does not enter it.

## Why it is here rather than in the app

It is a **test fixture**, and `tests/test_rig_supplied_mesh.py` is what consumes
it. It is not sample content the app ships and not the tutorial's base mesh: it
is a low-poly test asset (3,273 vertices, one 512-ish JPEG) and the art verdict
it can support is correspondingly weak.

What it *is* good for is the question no other file in this repository could
ask: **what the rig path does with a mesh that arrives already skinned.** Every
mesh the rig had ever seen was a TRELLIS reconstruction — no armature, no
vertex groups — and two defects lived in that blind spot until this file was
added. See that test.

## What it is, measured

| | |
|---|---|
| Geometry | 3,273 vertices, 4,672 polygons after `_import_glb`'s join |
| Material | one, with a base-colour JPEG; no metallic/roughness, no normal map |
| UVs | one set |
| Rig | 19 bones, 19 vertex groups, one animation (`Anim_0`, a walk) |
| Orientation | +Z up in world space; 1.507 m tall, feet at z ≈ 0 |
| Pose | approximately A-pose — arms ~29° below horizontal, slightly bent legs |

**Its 19 bones are not our 19.** The matching count is a coincidence and the
topology differs: the shipped `humanoid` template is 5 spine + 4 per arm +
3 per leg, while CesiumMan is 5 spine + **3** per arm + **4** per leg
(`leg_joint_L_1,2,3,5` — the numbering has a gap). No one-to-one mapping table
is possible, which is why the rig path builds its own armature and discards
this one rather than trying to adopt it.

Its shortest bone is 0.052 against a `rigging.MIN_BONE_FRACTION` floor of 0.003,
so none of Blender's silent short-bone deletion applies here.

## Do not treat this as game art

It is a specification sample. A Troupe verdict taken on it answers "does the
mechanism work", never "does the palette ramp work on a character someone would
ship". The second question still needs the authored or commissioned mesh the
plan file asks for, in male and female variants.
