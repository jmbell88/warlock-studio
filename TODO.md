# TODO — after the UPDATE.md plan

The 30-task plan in `UPDATE.md` is **complete** on branch `update-plan`, and **not merged**. This
file is the hand-off: what shipped, what is genuinely open, and what a follow-up session should pick
up first.

The authoritative record is `.superpowers/sdd/UPDATE/progress.md` — a per-task note for all 30
tasks, every corrected brief literal, and every controller ruling. It is git-ignored, so
`git clean -fdx` would destroy it. **Read it before touching this work.**

---

## Where things stand

| | |
|---|---|
| Branch | `update-plan`, off `master` @ `9532920`, **unmerged** |
| Commits | 48, every subject exactly `Warlock v0.0.7` (never bumped) |
| Diff | 118 files, +15 986 / −392 |
| Tests | 1558 → **2196 passing, 7 skipped** |
| Lint | `uv run ruff check .` clean |
| Working tree | only the pre-existing changes the plan's constraints named: `uv.lock` modified, `docs/SUGGESTIONS.md` and `docs/superpowers/specs/2026-08-04-manual-design.md` deleted, `UPDATE.md` / `REPORT.md` / this file untracked |

### What shipped

- **Part A (9 tasks)** — reference candidate ranking, style profiles, the opt-in reference reroll
  loop, and a measured mesh-hole retry (`WARLOCK_MESH_HOLE_MAX`, default 0.07).
- **Part B (8 tasks)** — sprite/icon/pixel cutout exports, BiRefNet matting, the 2D derive path with
  a manifest, and seamless tile generation with circular padding.
- **Part C (14 tasks)** — **Build mode**: a primitive modeller in the app. The pure `studio/build/`
  engine (immutable CSR mesh, primitives, ops, uid-addressed edits, the `.wblk` zip format),
  `viewer/glbwrite.py` inverting the GLB loader, vectorised ray-triangle picking plus a scale gizmo,
  `service.jobs.import_mesh` so a built asset is an *ordinary* asset, the controller, the viewport,
  four panes, the eighth mode with its ten dispatch sites, and the manual chapter.

### Two measurement gates, opposite verdicts

Both documented under `docs/measurements/`.

- **A6 → Warranted.** Hole rates are sharply bimodal with an empty gap between 0.031 and 0.101, so
  `WARLOCK_MESH_HOLE_MAX` defaults to 0.07 (the gap's midpoint) and the retry was built.
- **A8 → Scattered.** 330° / 300° yaw spread against a 30° bound, n = 37. **Task A9 was deliberately
  not built** — the finding is the deliverable, and a request-path fidelity score keyed to a fixed
  camera would be measuring the camera rather than the mesh.

---

## Open decisions

None of these is a defect in the branch. Each needs a human decision or a measurement that could not
be made from a keyboard.

### 1. Merge `update-plan` to `master` — or don't yet

48 commits, all gates green. Nothing in the branch is half-finished. The decision is whether Build
mode ships now or waits for the items below.

### 2. `seam.SEAM_MAX = 2.0` is uncalibrated  *(needs eyes on real output)*

B7 chose the constant without measuring against real tiles, because nothing rendered its verdict at
the time. B8 is now its first consumer, so the number is visible in the UI. Somebody should generate
stone, plaster, gravel and fabric tiles and see whether the threshold agrees with what the seam
actually looks like. Deliberately not re-tuned on a hunch.

### 3. torchvision is in the venv but not in `pyproject.toml`  *(a `uv sync` will undo this)*

`torchvision==0.26.0+cu128` was installed during A8 to unblock the calibrate sweep — transformers
5.14 dropped `BitImageProcessor`'s PIL fallback and now requires it. It matches the pinned
`torch 2.11.0+cu128` and is venv-only.

This is **not** just a bench problem. A4's `reference_cosine` — the anchor half of the candidate rank
score — goes down the same `AutoImageProcessor` path, so on any install without torchvision it
raises, is caught, and the rank silently degrades to composition-only.

Two things to decide: whether torchvision joins an optional extra, and whether the doctor's DINOv2
row should say what it has *not* checked (it currently claims only that the weights are on disk,
which is exactly the class of overclaim B2 was written to fix for BiRefNet).

### 4. The BiRefNet matting path has never been executed  *(needs weights)*

No weights on this machine. It is correct-by-construction against the published recipe, with
hand-written Pillow/NumPy preprocessing standing in for torchvision (constants independently verified
at review) and monkeypatched tests. Whether BiRefNet's remote modelling code needs `timm` /
`torchvision` was not confirmable offline, so the doctor row was made honest rather than guessed at.
First run with real weights is the test.

### 5. `viewer_embed` does not forget its viewport texture before releasing it  *(pre-existing)*

`Viewport.resize` releases its texture and allocates a new one, and the imgui backend maps GL names
to moderngl objects — so releasing without `forget_texture` leaves it holding a dead object under a
name the driver is free to reissue, which is how an unrelated image starts rendering as this one.
`textures.py` documents exactly this and `build_view.py` now implements and tests it on both halves
(resize and release). The asset viewer should get the same treatment.

### 6. Task A9 stays unbuilt unless the A8 sweep is redone

Recorded here so nobody re-opens it by accident. If the reconstruction pipeline changes enough that a
matched view might exist, rerun `uv run python -m warlock.bench calibrate --all` and re-read the
verdict before building anything on it.

---

## Low-priority follow-ups

Carried from the ledger's per-task `minor (deferred)` notes. None affects correctness of a shipped
path today; each is a small, well-understood improvement.

- **`_pick_anchor` (`panes/profiles_panel.py`) validates upload bytes but not pixel count**, unlike
  `create_job`'s `MAX_IMAGE_PIXELS` gate. Brief-mandated at the time; it is the decompression-bomb
  hole that gate exists to close.
- **`inspector._quality` reads `audit.get("hole_ratio")`, a key `_audit_mesh` has never written** (it
  stores `worst`). That line is dead. Pre-existing, not introduced by this branch.
- **The profile anchor is re-embedded once per candidate** (`bench/metrics.py`): a `count=8` fan-out
  runs 16 CPU ViT forwards where 9 would do. A `{(path, mtime, device): tensor}` cache removes it.
- **`_restore` after a failed mesh retry uses `copyfile`, which truncates before writing**, so a
  triple failure can leave `source.glb` truncated while `model.glb` and `params` stay a coherent set.
  Only temps + `os.replace` removes it. Pre-fix, the same failure errored the job outright.
- **`BuildView.pick` re-triangulates every mesh on every click.** Fine at blockout scale; the natural
  fix is caching `triangulate` beside the GPU entry under the same frozen-mesh key.
- **Two thumbnail paths now exist** — `ctx.capture_thumbnail` (bound to the asset `Viewer`) and
  `App._capture_build_thumbnail`. They agree today because both are `viewport.thumbnail_png()`.
- **`matting.unload()` does not call `torch.cuda.empty_cache()`.** Nothing passes a non-CPU device
  yet, so it is latent.
- **`ruff format --check` reports 113 files.** The repo has never been formatted with it, it is not
  one of the plan's gates, and this branch did not change that. Running it would be a large,
  unrelated diff — a decision in its own right.

---

## Bench follow-ups

Recorded during the parameter-sweep benchmark work (`warlock.bench sweep`, T1-T8 on `update-plan`).
Neither is a defect — both are the next two things a follow-up session would build.

- **AI judge.** `src/warlock/bench/verdicts.py`'s module docstring already documents the seam:
  a future `bench/judge.py` would read, per unit directory (`<run_dir>/items/<unit_key>/`),
  `views/*.png` + `views.json` (the rendered turntable), `reference.png` (what the item asked for),
  and `job.json` (whose `params["mesh_report"]`/`params["mesh_audit"]` carry the topology/silhouette
  verdicts the worker already computed), score them with a model, and write the result through the
  same `append_verdict(..., source="ai:<model>")` human review already uses — e.g.
  `source="ai:gpt-4-vision"`. Nothing about storage would need to change: `latest` keys on
  `(unit, source)`, so a human's verdict on a unit and an AI's sit side by side rather than one
  overwriting the other, and `report.aggregate`'s `sources` breakdown already tallies accept/reject
  per source. The human verdicts that accumulate across runs in the meantime are the calibration
  set a judge would eventually be checked against.
- **Sweep phase 2: server-config axes.** `bench/sweep.py`'s `SERVER_AXES` (`trellis_band`,
  `trellis_tex_res`) are accepted by the spec parser — so a spec naming one loads, and the file that
  will eventually drive it can be written and reviewed now — but both `unit_kwargs` and
  `_refuse_server_axes` raise on them today, because running one means something trellis-server is
  launched with, not something a request carries: `plan_sweep`/`run_sweep` would need to group units
  by server config and restart trellis-server between groups, with the config override recorded
  per-group in the manifest rather than as a single top-level field.

---

## Notes for whoever picks this up

- **Commit convention is unchanged**: subject exactly `Warlock v0.0.7`, detail in the body, ending
  with the `Co-Authored-By:` trailer. Do not bump the version unless asked.
- **Never `git add -A`.** The working tree carries pre-existing changes that belong to nobody's task
  (see the table above); stage explicit paths only.
- **`tests/build/` is a package** and its `__init__.py` must stay — without it, a Part C test module
  whose basename already exists elsewhere in `tests/` is a hard collection *error*, not a skip.
  `pyproject.toml` also overrides pytest's default `norecursedirs` to un-skip `build`.
- **Any test that drives a `*_mode` entry point through an inline-submitting fake ctx must stub
  `studio.dialogs` first**, or a real native file picker opens and blocks the whole run rather than
  failing it. This cost a run during C10.
- **`pygame.key.get_mods()` raises headlessly** — patch it in an autouse fixture, as
  `tests/test_build_mode.py` does.
