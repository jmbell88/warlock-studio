# V2 parity checklist

What the browser build did and where it now lives, plus the things only a human
at the window can confirm. Everything with a test name beside it is pinned;
everything under **By hand** is not, because it is about feel rather than
values.

## Where things went

| Was | Is |
|---|---|
| `app.py` route bodies | `warlock/service/*` — sync functions, `ServiceError` instead of `HTTPException` |
| FastAPI lifespan | `studio/runtime.py` (`Runtime.start` / `.shutdown`) |
| `GET /api/progress` polling | `runtime.progress()` read directly, every frame |
| `POST /api/jobs/{id}/thumb.png` | `service.files.save_thumbnail`, handed the viewer's own capture |
| `static/app.js` viewer | `studio/viewer/*` (ModernGL) behind `studio/viewer_embed.Viewer` |
| `static/index.html` panes | `studio/panes/*` |
| localStorage | `studio/settings.py` → `<data_dir>/studio_settings.json` |
| three.js `OrbitControls` | `viewer/camera.py` |
| three.js `TransformControls` | `viewer/gizmo.py` (rotate + translate only) |
| `GLTFLoader` | `viewer/gltf.py` |
| `PMREMGenerator` + gradient env | `viewer/env.py` (CPU SH9 + prefiltered equirect) |

## Pinned by tests

- **Colour pipeline** — the shader's ACES + sRGB output matches a CPU reference
  of the same formulas to within 2/255 (`test_viewer_gl.py`), and the viewport
  background is the literal `0x14151a` rather than a tone-mapped version of it.
- **Ambient from below** — the SH probe's irradiance for a downward normal is
  well above zero, which is the entire reason the environment exists.
- **Skinning** — a rest-pose `rig.glb` renders the same as the unskinned
  `model.glb`; posing a joint moves pixels.
- **Pose contract** — XYZW on the wire, mirror imported from
  `rigging.mirror_quaternion`, joint deltas converted `[x, -z, y]`, a bone's
  tail following its first child's head (`test_viewer_pose.py`).
- **Camera framing** — the exact `d = r/sin(fov/2) · 1.25` and `(0.62, 0.47,
  0.62)` the browser opened every model at (`test_math3d.py`,
  `test_viewer_pose.py`).
- **Sheet preview** — yaw 0 on +Z, extent `max(hypot(sx, sz), sy) · 1.12`, one
  transparent cell per yaw (`test_viewer_sheet.py`).
- **Service behaviour** — the whole of the old HTTP suite, re-pointed
  (`test_api.py`, `test_rig_api.py`, `test_poses_api.py`, `test_sheet.py`,
  plus `test_service.py`'s parity pins).
- **Every pane builds** — one imgui frame containing all eight, with every
  collapsing section forced open (`test_studio_smoke.py`).

## By hand

Open the app (`uv run warlock`) and check:

- [ ] **Orbit damping** — a flick keeps gliding and settles; it should feel the
      same at 60 Hz and 144 Hz (the damping is frame-rate corrected, which the
      browser's was not).
- [ ] **Turntable** — `S`, or the toolbar button. One orbit per 30 seconds.
- [ ] **Pan** — right-drag moves the model with the pointer at any zoom; it
      must not accelerate as you zoom out.
- [ ] **Frame** — `F` re-frames a lost camera without reloading.
- [ ] **Wireframe** — `W`. Desktop-only (`glPolygonMode`), by design.
- [ ] **Compare** — two meshes, one camera. Both halves must move together
      exactly, and both must sit on the grid the same way.
- [ ] **Rotate gizmo** — the visible red ring turns the bone about *its* X.
      Drag a shoulder: the arm should swing, not corkscrew.
- [ ] **Joint drag** — in joints mode a marker follows the pointer and stays
      where it is dropped; the other markers do not move.
- [ ] **Unsaved-pose guard** — with an edited pose, try switching jobs,
      starting a comparison, leaving edit mode and closing the window. All four
      must ask.
- [ ] **Progress card** — a real generation shows a stage, a bar that eases
      rather than jumping, an elapsed clock and (after ~5 s and 10%) an
      estimate that does not flicker between wildly different numbers.
- [ ] **Cancel** — mid-run, from the progress card. The bar must stop and the
      card go away without a modal in front of it.
- [ ] **Drop an image** on the window: it should switch to 3D mode and queue a
      mesh job from it.
- [ ] **Downloads** — a cold STL shows a spinner, then a save dialog. FBX
      without bpy is greyed with "needs Blender" rather than absent.
- [ ] **Restart** — mode, form values, prompt history, filters and window size
      come back; the seed does **not** (that is deliberate).
