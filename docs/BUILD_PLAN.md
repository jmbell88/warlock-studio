# The overnight build plan

**This is a record, not a queue.** It is the plan that Warlock's Night 1 and
Night 2 build runs executed, transcribed on 2026-08-08 from the session it was
pasted into so that it stops living only in a chat transcript. **`TODO.md` at the
repo root is the live roadmap** — if this document and `TODO.md` disagree about
what is left to do, `TODO.md` is right and this one is history.

Keeping it is a deliberate exception to the rule stated in `CLAUDE.md` that a
finished plan is deleted rather than ticked. That rule exists because a plan
whose checkboxes disagree with the tree is worse than no plan — and it still
applies to the *unticked* half of this document, which is exactly why nothing
here is ticked at all. What this file is kept for is the half a diff cannot
carry: the reasoning behind each package, the file-contention map that decided
the wave split, and the three corrections in the Context section that later
phases were built on top of. Read it for *why*, never for *what is outstanding*.

## What has happened since it was written

- **Night 1 shipped fourteen packages** — 0a–0f and 9a–9d, 9h, 9i — merged to
  master on 2026-08-08.
- **Night 2 shipped four** — Phase 1 (matte preview + Inker hand-off), Phase 2
  (mesh candidates), Phase 4 (weld-before-heat rigging) and 9g (the model
  downloader), merged to master at `84b7e4f`, 3464 passed / 7 skipped.
- **9g's offline-invariant amendment was authorised by the user** and is now
  written into `CLAUDE.md`: the app process stays offline for its whole life, and
  one explicit user-initiated fetch runs in a child process that sets
  `HF_HUB_OFFLINE=0` in its own environment only.
- **Phase 2 took migration 6**, so the DINOv2 judge's `verdicts.stage` column —
  described below as migration 6 — is now **migration 7**. The plan text is left
  as it was written; this note is the correction.
- **Phases 3, 5, 6, 7 and 8 are not started.** They are carried, with everything
  learned since, in `TODO.md` §2, §3, §7–§10, §11 and §12 respectively.

Two known defects in the text below, both left in place rather than silently
edited: the numbering collides with `TODO.md`'s own `§` numbers (they are
different schemes, and this document's "Phase 1" is not `TODO.md`'s "§1"), and
several file:line citations were accurate when written and have since drifted —
locate by symbol, not by line number. One thing *was* repaired in transcription:
the source text repeated Phase 8's heading followed by a second copy of Phase 7's
bullets, a paste artifact with no meaning, and that duplicate block is removed.

---

## Warlock — MASTER PLAN

Reconstruction-reliability program + UI/UX pass + the open-work inventory, in one
phased plan structured for overnight parallel execution.

---
## Context

Three inputs are merged here.

1. An audit of everything open in the repo, verified against the working tree.
Its headline: the codebase has no debt. No FIXME/XXX/HACK anywhere (the
five TODO hits in src/ are cross-references to TODO.md item numbers); no
unimplemented stubs (the only NotImplementedErrors are abstract base methods in
studio/undo.py); all 14 test skips environment-gated; tree clean; the one other
branch (v2-studio) fully merged and 137 behind. What is open is empirical work
— measurements nobody has run — plus unbuilt features and a little actively
misleading documentation.

2. An external review of the image→3D pipeline, fact-checked against the code.
Its core claims held: the BiRefNet matting signal is real, hole_worst is
inverted, promotion silently inherits the 2D pane's resolution so recent
reconstructions ran at 512, host matting fails to load while doctor reports green,
recent rig jobs silently fell back to envelope weights, and meshreport's
watertight figure mostly measures xatlas UV-seam splits (process=False). It
missed that vendor/gltfpack/gltfpack.exe landed 2026-08-07, that the
view-calibration doc rules out fixed-view reference-similarity ranking, and that
the multi-candidate machinery largely exists.

3. Nine UI/UX requests, researched to exact call sites (Phase 9).

The intended outcome: first repair the measurement instruments and input
hygiene — so the TODO.md §0 GPU sessions measure the right things — then the
user-facing reliability features, then the larger subsystems, with the UI pass
running in parallel throughout because it touches disjoint code.

## Decisions already taken with the user

- Framing (character = front-ortho A/T-pose vs global 3/4) enters as a measured
sweep axis, not a default flip.
- Matte UX is preview + Inker hand-off — reuse the raster editor, do not build
a second brush UI.
- Reliability mode is an opt-in Candidates control (1/2/3 seeds) on promote,
never default-on.
- Ship with SDXL-Turbo only; everything else becomes click-to-download.

## Three corrections that later phases depend on

1. bench/findings.json on disk is a 196-byte v1 stub with fabricated data
("key": "abc123", legacy "platform": "pc"); bench/findings.md is a stale
lora_weight dump. The matting signal is genuine but its evidence is the DB
— assets/jobs.sqlite holds 100 verdicts, 4 accept / 96 reject, surviving
prune_jobs with only 2 job rows left, exactly as the denormalized vector
column intends. Any figure from that file must be re-derived via
service.findings.refresh before it is leaned on. The DB shows 4 accepts
where TODO.md §0 says 3 — reconcile in 0f.
2. hole_worst is inverted, not weakly informative — AUC 0.115, because a slab
has no holes and meshaudit scores the dominant failure mode as perfect.
Anything reading a low hole fraction as quality is wrong. This invalidates
docs/measurements/2026-08-04-hole-rate-baseline.md, superseded in Phase 3.
3. An in-app downloader is a deliberate, scoped exception to a hard invariant.
CLAUDE.md says "fully offline — no runtime network calls, ever", and
__init__.py:9-10 sets HF_HUB_OFFLINE=1 / HF_HUB_DISABLE_TELEMETRY=1 as the
first thing the package does. 9g does not relax that: the app process stays
offline forever and fetching happens in a separate subprocess. The invariant
text must be amended to say so explicitly — a rule changing is a decision, not a
silent edit.

## Standing constraints from CLAUDE.md that bind every phase

Fully offline in-process; the service layer is the only business logic; single
sqlite connection under the lock; every subprocess in the kill-on-close job
(tests/test_vram.py::test_every_subprocess_spawn_is_in_the_kill_on_close_job
scans for this); source.glb immutable and model.glb derived; derived values
join DERIVED_PARAMS; VECTOR_PARAMS is an allowlist; corpus-keyed constants get
a docs/measurements/ doc before they change; Blender only ever in the subprocess
worker; nothing may block the frame loop; a texture must be forget_textured
before release; UI strings stay inside imgui's Basic-Latin+Latin-1 atlas range
(· is safe, ≥/Δ are not).

Commit convention: Warlock v0.0.11 — no version bump unless asked.

---
## Overnight dispatch map

Phase 0 and Phase 9 are both GPU-free and touch mostly disjoint files, so they run
together. File contention is the scheduling constraint, not logical dependency:

```
NIGHT 1 — WAVE A (parallel worktrees, disjoint files)
  0a  matting + honest doctor + torchvision    doctor.py, matting.py, pyproject
  0b  kill the 512 promote trap                service/jobs.py
  0d  surface rig-weighting outcome            blender_worker, rigging, queue, inspector
  0e  framing as a prompt axis                 guidance.py, prompt.py, settings_2d.py
  9b  FPS/RAM/VRAM bar                         main.py, state.py, overlay.py
  0c  weld-by-position mesh analysis           meshreport.py, vectors.py, widgets.py

NIGHT 1 — WAVE B (each waits on a Wave A package sharing its files)
  9a  centre icons in icon_button   widgets.py, library.py        <- after 0c
  9c  library 2D/3D badge           library.py, widgets.py        <- after 9a
  9e  open generated image in Inker inker_mode, overlay, library  <- after 9c
  9f  open generated model in Clay  clay_mode, library, inspector <- after 9e
  9h  LoRA auto-deselect            settings_2d.py                <- after 0e
  9d  bigger reference image        inspector.py                  <- after 0d
  9i  splash screen                 main.py, splash.py            <- after 9b
  0f  doc debt, version, memory     CLAUDE.md, docs/, __init__.py <- after 0a

NIGHT 2 (needs Wave A/B)
  9g  model downloader          (needs 0a's deps, 9b's progress plumbing)
  1   matte preview + Inker      (needs 0a, 9e)
  2   mesh candidates            (needs 0b)
  4   rigging: weld-before-heat  (needs 0d)

USER-GATED
  3   GPU sessions + framing measurement   (needs all of Phase 0)
  5   gltfpack tier qualification          (code anytime; corpus from 3)
  6   DINOv2 judge                         (needs 3's corpus + a labelling pass)
  7   retopo + bake prototype              (needs 4)
  8   external backend A/B                 (own spec first)
```

Three file-contention rules, because they are the whole reason for the wave
split. widgets.py is touched by 0c (quality_badge), 9a (icon_button) and 9c
(a new badge) — run that chain serially in one worktree, in that order.
library.py is touched by 9a, 9c, 9e and 9f, and inspector.py by 0d, 9d, 9e and
9f, which is why Wave B is a chain rather than a fan-out. And pyproject.toml
belongs to 0a alone — 0f owns the torchvision decision, 0a makes the edit.

Worktrees need the three WARLOCK_* env vars (vendor/ and models/ are
gitignored) or the suite reports 427/6-skipped instead of 433/0.

---
## Phase 0 — Instrument & input repairs (GPU-free)

### 0a. Make host matting operational and honestly reported

- Add einops and kornia to the extra carrying the matting stack in
pyproject.toml; verify timm/torchvision resolve there (the doctor hint at
doctor.py:391-412 names them). uv sync, then verify
pipelines/matting.py:_load succeeds against models/birefnet.
- Upgrade doctor._matting_checks (doctor.py:375-412): beyond weights presence,
import-probe einops, kornia, timm, transformers and name any missing one
in the detail string. Surface "last load failed: <err>" — matting.py already
remembers a failed load for the session (~`:173`); persist it where doctor can
read it (a module-level marker; doctor runs in-process).
- Files: pyproject.toml, src/warlock/doctor.py,
src/warlock/pipelines/matting.py, tests.

### 0b. Kill the 512 promote trap

- In service/jobs.py:promote_to_model (~776-892): strip inherited resolution
from the copied params unless the promote kwargs carry it explicitly;
re-derive from the model-side platform (default 3d → 1024) through
guidance.normalize. Mirror studio/review_mode.py:capture_base (~420-423),
which already applies this for sweeps: 3D platform wins, explicit override
survives.
- The sharpest correctness bug in the set — it silently halved recent
reconstructions.
- Tests: a platform=2d reference promotes to resolution absent-or-1024; explicit
override survives; sweep behaviour unchanged.

### 0c. Weld-by-position mesh analysis in meshreport

- meshreport.py:49 loads process=False, so every xatlas seam split counts as
boundary. Keep the raw load (UV/material checks need it); run the
watertight/boundary/component analysis on a welded analysis copy — merge by
position at a bbox-relative tolerance (e.g. 1e-5 × diagonal), constant
commented. No measurement doc needed (no stored corpus keys on it yet); note it
in the report output.
- mesh_report keeps its shape; add fields additively (readers are
.get-based): welded watertight verdict, welded boundary-edge and component
counts. The "not watertight" reason and the badge (studio/widgets.py:146-170)
switch to the welded verdict; vectors.observation_metrics picks up the welded
flag. Preserves the invariant that only meshreport may say watertight.
- Tests: synthetic seam-split cube — raw not-watertight, welded watertight.

### 0d. Surface the rig-weighting outcome

- blender_worker._skin (pipelines/blender_worker.py:123-153) already returns
"envelope"; the reason is only a subprocess print. Return it in rig_meta
(weighting_reason), write into rig.json (additive), store as
params["weighting_reason"] beside params["weighting"] at
queue.py:1406-1411, and add both to DERIVED_PARAMS (weighting is already
there per validation.py:79 — verify).
- UI: weighting: envelope - needs review with the reason on hover, in the
inspector's rig section. Envelope is a degraded outcome, not success. Use
widgets.hint_text; stay in the Latin-1 atlas range (hyphen, not —).
- Tests: host-side rig_meta JSON round-trip (no bpy needed).

### 0e. framing as a first-class prompt axis (default byte-identical)

- Add framing to guidance.py: three_quarter (default), front_ortho.
Refactor PROMPT_TEMPLATE (pipelines/prompt.py:47-51) so the view clause is
injected from the field and the default composition is byte-identical — same
clause, same position, same commas — leaving prompt_hash and PROMPT_VERSION
(stays 4) untouched for every existing and default job. front_ortho swaps in a
front-view orthographic contract; for category=character it must not fight the
T-pose fragment at guidance.py:147.
- Add framing to vectors.VECTOR_PARAMS. Expose as a select in the 2D pane
(it composes the reference prompt → 2D owns it, per the one-owner rule). Legacy
params normalize to the default through _canonical/normalize; no alias needed
since no key is renamed.
- Tests: pin byte-identity of the default composition (verify red against a
deliberately altered clause).

### 0f. Packaging, documentation debt, stale artifacts

- torchvision is in the venv but not pyproject.toml. A routine uv sync
removes it and candidate ranking silently degrades to composition-only via a
caught ImportError (bench/metrics.py's AutoImageProcessor path). Open since
2026-08-05. Fold into 0a's dependency work; decide whether doctor's DINOv2 row
should say what it has not checked.
- Five places still say gltfpack is not vendored — verified by grep:
CLAUDE.md:43, src/warlock/config.py:94,
docs/manual/11-configuration.md:25,
docs/manual/03-generating-meshes.md:108, and
src/warlock/studio/panes/settings_3d.py:25-26 ("every named tier needs a
gltfpack that is not vendored yet"). The retarget panel gates on the file at
runtime, so it now shows the full tier list while the manual says it will not.
The downstream half stays true (tiers unqualified, UI offers only raw); the
stated reason is wrong. tests/manual/test_docs.py gates the manual edits.
Phase 5 rewrites the settings_3d.py one again when a tier qualifies — leave the
PROFILES list itself alone here, fix only the comment.
- __init__.py:12 says __version__ = "0.0.9" while pyproject.toml says
0.0.11. One-line fix.
- Regenerate bench/findings.json / .md via service.findings.refresh;
reconcile the 3-vs-4 accept discrepancy.
- docs/superpowers/plans/2026-08-07-library-pane-relocation.md has 11 unticked
checkboxes but the work landed (main.py:1060, tests/test_studio_smoke.py:603).
Tick or delete; reconcile CLAUDE.md's claim that those trees "have been
deleted".
- CLAUDE.md's App Settings wording — the pane is a real 156-line
implementation, read-only by design; "placeholder that exists in the switch
before its feature does" overstates it. (9g changes this pane anyway.)
- Correct two stale memory files — MEMORY.md line 3 and
warlock-rogue-sweep-2026-08-07.md both claim the 83 meshes are unjudged; the
review ran 2026-08-07 (11.3 h, 3 accepts) and the blocker moved to Phase 3.
- Delete the v2-studio branch — merged, 137 behind, a dead pointer.

Phase 0 exit: uv run pytest green (and with WARLOCK_NATIVE=0); doctor shows
matting operational; a promote from a platform=2d reference reconstructs at 1024
(check trellis.log); mesh_report reports welded topology; a rig job's weighting
outcome is visible.

---
## Phase 9 — UI/UX pass (GPU-free, runs alongside Phase 0)

### 9a. Centre the icons in icon buttons

Root cause, measured. theme.py:86 sets frame_padding = (sp(9), sp(6)).
widgets.icon_button (widgets.py:515-535) sizes a square button to
get_frame_height() = fs + 2·6 = fs + 12, but imgui's inner text rect is
side − 2·padding.x = fs − 6, while a Lucide glyph advances ~`fs. The text does not fit, RenderTextClipped clamps the align
offset to zero, so **the glyph is pinned left with ~3 px on the right and is clipped on the right edge**. Vertical is
already correct (fonts.py:33 ICON_OFFSET` handles the Lucide-vs-Inter baseline).

- Fix inside icon_button only: push frame_padding = (0, 0) and
button_text_align = (0.5, 0.5) around the imgui.button call, pop after. Do
not touch ICON_OFFSET (it is derived from font metrics and would de-centre
every icon in the app) and do not change the global frame_padding (it would
reshape every text button and modal).
- Fold the two bare imgui.small_button(icons.…) call sites in
library.py:279,294 onto the helper (or a small_icon_button sharing the same
push/pop), so one idiom governs.
- Tests: extend
test_no_two_of_a_panes_icon_buttons_are_drawn_on_top_of_each_other
(tests/test_studio_smoke.py:1360) with a rect-vs-glyph-centre assertion, and
keep test_no_pane_continues_a_line_that_has_no_room_left (:319) green.

### 9b. FPS / RAM / VRAM bar next to the mode switch

There is no imgui menu bar in the app (theme.py:113 sets menu_bar_bg and
nothing uses it). The de-facto menu is App._mode_switch (main.py:1752-1822),
whose right end is already a mini toolbar: same_line(width − sp(64)) at :1808,
then the ? button and the health dot.

- Add the readout to that right-hand strip, widening the sp(64) reservation
to fit. This is precisely the audit's past-the-edge hazard — a same_line past
the content region does not wrap, it clips and the control vanishes — so use
widgets.same_line_or_wrap and measure the text first.
- Sources, all pure and already None-safe: App.fps (already exists —
main.py:341 uses self.fps.frames), memlog.process_memory() /
memlog.system_memory() (memlog.py:89,124), vram.device_memory()
(vram.py:180).
- Two honesty constraints. vram.device_memory() reaches torch through
sys.modules and returns None when torch is not yet imported — render --
rather than 0, and never call vram.probe() (it imports torch, costing seconds).
And throttle to ~2 Hz behind a timestamp, cached in AppState: the frame loop
must never block, and cudaMemGetInfo per frame is real work.
- Format compactly in Latin-1: "58 fps · 1.9/32 GB · VRAM 21.4/32". Tooltip
carries the long form. Keep the whole strip inside the reservation at every DPI
scale.
- Reconcile with the existing meter. overlay.fps_meter (overlay.py:83+) is
a bottom-left readout gated behind F10, with its own documented reasons for
position and fixed width (toasts stack the other corner; Inter is proportional so
an auto-sized box breathes with every digit). The new bar makes it redundant —
decide deliberately: either retire F10 and its shortcut-popup entry, or keep it
as the detailed view and have the bar be the always-on summary. Do not ship two
unrelated FPS numbers.
- Files: src/warlock/studio/main.py, src/warlock/studio/state.py (cache),
panes/overlay.py.

### 9c. Library: tell a 2D image from a 3D model at a glance

Today the card (panes/library.py:201-271) never reads job["stage"]. The
only tells are the thumbnail itself, "| from a reference" (:261-263, which
marks the derived job), and the incidental absence of quality_badge — whose
docstring (widgets.py:157-159) says "every reference in the library has neither a
report nor an audit", i.e. an accident, not a design.

- Add a stage/kind badge beside status_pill / quality_badge at
library.py:244-245, drawn in the status_pill idiom (widgets.py:122-142):
an icon + short word per kind — reference / tile / model / rig / sheet — with the
icon carrying the distinction (image vs cube) and colour reserved for status so
the two encodings do not fight.
- Honour quality_badge's inline contract (widgets.py:152-159): a widget
that may draw nothing must own its own same_line, or the orphaned same_line
is inherited by whatever comes next. The new badge always draws, so it may take
the simple path — but it must be ordered so the optional badge stays last.
- stage is already used for behaviour in the same file (:331, :380, :400,
:428, :447) and by widgets.artifacts_for (widgets.py:63-88), so no new
data is needed.
- Tests: the two standing layout guards, plus a card-content assertion per kind.

### 9d. Make the reference image fill the inspector's width

inspector.py:29 has THUMB_SIZE = 96 as a raw literal, not wrapped in sp()
(unlike the library's sp(THUMB_SIZE)), and :275 draws
imgui.image(..., (THUMB_SIZE, THUMB_SIZE)) — a hardcoded square, so the aspect
ratio is ignored and the image is squashed, and it never reacts to the pane width.

- Size from imgui.get_content_region_avail().x and preserve aspect. The
fit-to-box idiom already exists at main.py:2015-2025 (_draw_reference:
scale = min(width/tex.w, height/tex.h)); the aspect-correct precedent in this
very file is pixel_scale (inspector.py:376-384). Reuse rather than reinvent.
- Wrap the constant in sp() while there, and cap the height so a tall reference
cannot push the rest of the inspector off-screen.
- The Reference header is default_open=False (:235) — consider defaulting it
open now that it is worth looking at; that is a persist_key behaviour change,
so make it deliberate.

### 9e. Open a newly generated image in Inker

The gate and the loader already exist: inker_mode.can_edit_job
(inker_mode.py:147-158 — stage == "reference", status == "done",
"input.png" present, cache-row only so it is frame-safe) and
inker_mode.open_job_reference (:161-177, which reuses an open tab via
state.find_job/activate and otherwise submits inker-open:<job_id>).

- The only entry point today is the 2D viewport toolbar button
(panes/overlay.py:31-36). Add: an "Open in Inker" item in the library
overflow menu (mirroring Clay's, library.py:336-344, gated on can_edit_job),
and a post-generation affordance in the 2D inspector so a just-finished image is
one click from editing.
- No new machinery — this is wiring existing entry points to more places. Keep the
_save_linked write-order contract untouched (input.png first, paint.ora
second, :314-321).

### 9f. Open a newly generated model in Clay

Same shape, and the entry point already exists: clay_mode.edit_asset_in_clay
(clay_mode.py:150-176), reached from the library overflow menu
(library.py:336-344) gated on "model.glb" in files, preferring the build.wblk
sidecar over model.glb and never source.glb.

- Add the equivalent affordance in the 3D inspector for a just-finished model,
reusing edit_asset_in_clay verbatim.
- Preserve the SLOW_TRIANGLES = 200_000 confirm (clay_mode.py:131,
_adopt_import :190-209) — trellis output is 177k–299k triangles, so this
dialog will fire on nearly every reconstruction. That is correct, not a bug;
do not quietly raise the threshold.

### 9g. Click-to-download models and LoRAs (Settings pane)

Ship with SDXL-Turbo only; everything else downloadable. This is the scoped
exception described in the Context — design it so the app process never becomes
online-capable.

What already exists. Every registry entry (BaseModel models.py:79-130,
StyleLora :133-143, IPAdapter :146-173, ControlNet :176-189,
MetricModel :512-524, PoseModel :544-558, MattingModel :582-603)
carries a download: str. Doctor already reports presence per row and formats
"... download with:\n  {spec.download}" (doctor.py:274-406).

Why the string cannot simply be executed — and what to add. download is
free-form prose, not data: several entries are multi-command joined by "\n  "
(sdxl :223-229), and some carry non-shell steps — pixel :291-302
includes a rename (pytorch_lora_weights.safetensors →
lcm-lora-sdxl.safetensors) and birefnet :616-628 includes a uv pip install.
Executing them blindly is wrong.

- Add a structured Fetch record per entry: repo_id, optional filenames,
allow_patterns/ignore_patterns (the --include/--exclude sets),
local_dir, optional post step (the rename), and an approximate size for the
progress UI (the README carries these as prose: ~7 GB, 787 MB, 394 MB, ~6.9 GB,
~16 GB). Derive the existing download string from that structure so
doctor's text and the README stay identical and cannot drift further.
- Dedupe on (repo_id, local_dir), not on model key — dir_name is not
unique: sdxl, sdxl_cfg, pixel and lightning all point at
sdxl-base-1.0 (:217, :258, :285, :317).
- Resolve paths through the existing rule — doctor._base_model_dir
(doctor.py:230-233) is canonical because only turbo honours
WARLOCK_T2I_DIR; everything else is config.t2i_model_root / dir_name. Lift it
into a shared helper rather than re-deriving. Note the download strings
hardcode relative --local-dir models/..., which WARLOCK_T2I_ROOT can
relocate — the structured record fixes that too.

Execution — out of process, always.

- A python -m warlock.pipelines.fetch_worker child, spawned through
winjob.run/winjob.assign (winjob.py:131,167). This is mandatory:
test_every_subprocess_spawn_is_in_the_kill_on_close_job scans the package and
fails on any spawn without a nearby assign.
- The child sets HF_HUB_OFFLINE=0 in its own environment only. The parent
keeps __init__.py:9's setdefault("HF_HUB_OFFLINE", "1") untouched — and
because huggingface_hub reads it at import time, an in-process download would
require re-setting the var and re-importing, which is exactly the fragility the
subprocess avoids.
- Amend the CLAUDE.md offline invariant to state the exception precisely: the
generation pipeline never touches the network; an explicit, user-initiated
fetch runs in a separate process.

Progress. TaskRunner has no progress mechanism — Done arrives only at
completion (tasks.py:37-64; _Pending.started is set but never advanced), and
the only affordance is a spinner keyed off is_busy. A multi-GB download needs a
bar, so add a small thread-safe progress dict on TaskRunner written by the worker
and read on the frame thread — the same shape 9b's throttled readout uses. Key
downloads download:<key> so any_busy("download:") gates the whole pane.

UI (panes/app_settings.py:126-151). Replace the read-only list with rows
carrying a checkbox + Download button. Today missing-ness is inferred from the
literal substring "missing" in the label (_row, :154-156), fed by
main.py:273-284 — brittle, and ctx.style_loras (:282-284) carries no missing
marking at all. Give each row a real status flag instead.

- On completion, re-run doctor.run_checks the way the "health" task already
does (main.py:454-459, which replaces runtime.checks wholesale) and
recompute ctx.base_models / ctx.style_loras, which are built once at startup
and documented as immutable (app_ctx.py:88-89). That comment must change.
- Add failure routing in _collect_tasks (main.py:422-446) so a failed fetch
toasts rather than vanishing.
- Update the pane docstring (app_settings.py:8-11) and its footer, which both
currently state the read-only rationale.
- README and models.py already drift — four repo ids exist only in
models.py (IP-Adapter h94/IP-Adapter, ControlNet
diffusers/controlnet-canny-sdxl-1.0, DINOv2 facebook/dinov2-base, BiRefNet
ZhengPeng7/BiRefNet), and the README uses PowerShell backtick continuations
against models.py's one-liners. Generating both from Fetch closes this.

Refuse rather than half-download: check free disk against the declared size
before spawning, and treat a partial directory as absent (the probe tuple already
answers "is this really here").

### 9h. Deselect LoRAs when the model cannot use them

Today this is a dead end. Picking a non-SDXL base keeps form["style_lora"]
and merely disables the combo (settings_2d.py:568-574, :594-596);
validate (:692-693) then appends "Style LoRAs need an SDXL model.", which
_submit (:647-664) renders in theme.ERR and uses to disable Generate. The
only recovery is switching the base model back, because the control that would fix
it is disabled.

- On a base-model change (settings_2d.py:563), if the new base is absent from
ctx.guidance["lora_bases"] (models.lora_bases(), models.py:653-663, surfaced
at guidance.py:651), clear style_lora to "" and reset lora_weight to
models.DEFAULT_LORA_WEIGHT. Do it only on change, never every frame, or a
restored form is silently rewritten on open.
- Say so, rather than clearing invisibly: keep lora_note (:533-546) and add a
one-line widgets.muted noting the style was cleared because this model cannot
use one. The existing rationale for disabled-not-hidden (:570-573 — the form may
hold a style picked under another base and hiding it would make the selection
vanish unexplained) argues for an explicit message, not against clearing.
- Apply the same rule to the sibling gates while there — structure_note
(:549-557, controlnet_bases) hides the group, and negative_prompt_note
(:516-530, cfg_bases); check whether either leaves a stale value that
validate will reject. Do not merge lora_bases() and tile_bases()
(models.py:666-675); their docstring explains why the duplication is
deliberate.
- Tests: switching base to flux2klein with a LoRA set clears it and leaves
Generate enabled; switching back does not resurrect it; a restored form is not
rewritten on open.

### 9i. Splash screen on startup

docs/logo.png exists (1.9 MB). Requirement: show it while loading or 3
seconds, whichever is longer.

The ordering problem. App.setup (main.py:160+) calls
self.runtime.start() before pygame.init() — the comment at main.py:313-315
says so explicitly ("it starts the runtime before it touches pygame or GL"). So
today there is no window at all during the slow part (doctor checks, the worker,
and doctor._probe_blender, which CLAUDE.md notes spawns a python.exe during
Runtime._start).

- Split setup() in two. setup_window() — DPI awareness, pygame.init, GL
attributes, set_mode, moderngl, fonts, theme (all fast, all main-thread,
GL-bound). Then setup_runtime() — runtime.start() plus Ctx construction.
- Draw the splash after setup_window(), run setup_runtime() on a plain worker
thread, and pump splash frames until it completes and max(3 s, load time)
has elapsed. Ctx construction must happen back on the main thread (it needs
textures, which need GL).
- The window now exists during startup, so the X button is live: handle a quit
during the splash and make teardown() safe when Ctx was never built. That is
the whole risk of this change and it is where the tests belong.
- The logo is a texture like any other: load once, register through
widgets.texture_ref, and forget_texture before release — the backend keeps
a dead object under a GL name the driver will reuse otherwise
(imgui_backend.py:151). Release it after the splash; do not hold 1.9 MB of
decoded pixels for the session.
- Keep it honest: if startup fails, the splash must give way to the existing
failure path (main.py:330-343 distinguishes a window that never appeared from
one that vanished — preserve that distinction, since the splash changes which
case a user sees).
- Files: src/warlock/studio/main.py, a small studio/splash.py, and
docs/logo.png (consider moving it under studio/resources/ — docs/ is
described in CLAUDE.md as holding only the manual, measurements and
REPORT.md, and a runtime asset there contradicts that).

---
## Phase 1 — Matte preview + Inker hand-off (needs 0a, 9e)

- Preview. In the promote flow (panes/settings_3d.py), before submit, compute
the host-BiRefNet cutout on the TaskRunner, never the frame thread, and show
it over a checkerboard with Accept / Cancel / Fix matte. Cache by
(job id, input.png mtime) under the MTIME_RACE_NS racily-clean rule.
- Fix matte opens Inker on the reference through open_job_reference
(inker_mode.py:161) with the matte as an editable alpha layer. Saving keeps the
_save_linked order — input.png first, paint.ora second (:314-321) — and
the input.orig.png backup precedent.
- Approved alpha travels as image alpha. When the reference carries non-opaque
alpha at promote, the upload preserves it and the job records matte: approved
(joining VECTOR_PARAMS so findings can compare approved vs auto). Verify
against the actual trellis-server.exe — per CLAUDE.md, confirm against the
exe, not upstream docs — what bg_removal should be when alpha is present.
Expected: the server preserves existing alpha (matting.mask() tier 1 is the
same rule host-side); if it re-mattes anyway, send the preserving mode or omit
the field.
- This revises the reference.py:11-14 design rule ("mask drives geometry only,
never written back as alpha"). Update that comment explicitly.
- Preflight. Surface reference.measure's warnings (edge contact, occupancy,
multi-object) in the promote preview via reference_report, so the user sees
them before spending GPU minutes.

---
## Phase 2 — Reliability mode: mesh candidates (needs 0b)

- Columns, not params — the sweeps precedent. A migration adds
candidate_group / candidate_index to jobs; Filters.matches hides
non-winning candidates as it hides sweep units. Membership cannot leak onto
rerolls because columns are not copied by rerun_job/promote_to_model.
Coordinate the migration number with Phase 6 — migrations are append-only and
never edited once shipped; whichever lands first takes 6.
- Submission. A Candidates control (1/2/3, default 1) in the 3D pane.
service/jobs.py gains a candidate-group promote: N ordinary jobs through
create_job, per-job VRAM admission unchanged. The existing count refusal for
output="model" (jobs.py:136-139) stays for the generic path — candidates are
the deliberate exception with their own entry point. Candidate 0 keeps the
requested seed; the rest draw fresh ones (the reference-batch idiom at
jobs.py:272-289).
- Picker. When a group finishes, the 3D inspector shows the candidates;
selecting one loads it in the shared viewer (_sync_viewer rules apply,
including viewer.pending and the pose-mode guard). Keep marks the winner and
offers to delete the losers through the existing delete path — never
automatic. Verdicts go through the ordinary path and feed findings.

---
## Phase 3 — GPU sessions + framing measurement (user-run; needs Phase 0)

Code work is sweep-spec preparation; the GPU time is the user's.

- Build the TODO.md §0 specs: (a) the blinded 8–12 unit birefnet-vs-auto
confirm — the clean 2×2 alone is Fisher p=0.14 and the review was unblinded and
single-reviewer; (b) the re-baseline render sweep with bg_removal=birefnet
as baseline, now also carrying the framing axis for character subjects, which
0e made expressible.
- The rule §0 records: establish a baseline that produces acceptable output at a
workable rate before fanning out. Sweep B spent roughly half the GPU time
measuring nothing because bg_removal was pinned at its bad value while four
axes varied — a floor effect, not a null result.
- After verdicts: write the framing measurement doc. If front_ortho wins for
characters, flip the per-category default then — that is the
PROMPT_VERSION 4→5 moment, and the findings-corpus split is the documented
cost. Supersede 2026-08-04-hole-rate-baseline.md.

Also settled by this corpus, cheaply:

- Rig handedness (TODO.md §6). Whether trellis reconstructs with the same
handedness the COCO→template +X mapping assumes is unverified and invisible
if wrong — the image half is test-pinned, but the mesh half used a symmetric box
as a stand-in. On the first mesh out of the re-run, rig an asymmetric subject
and check which side the .L bones landed on. A flip is a one-line sign change.
- art_style=snes vs the colour brief (TODO.md §7). ART_STYLES["snes"]
contributes "vivid saturated colours" against a "black and silver and blue"
brief; Sweep B's palette axis returned all ties under the floor effect. If
mono/muted wins clearly, the era fragments over-specify colour and should
describe shape and shading only.

---
## Phase 4 — Rigging: weld-before-heat (needs 0d for the UI half)

- Hypothesis: xatlas seam-split vertices make trellis meshes non-manifold for
bone-heat's Laplacian solve; welding first should let ARMATURE_AUTO succeed.
Same root cause 0c addresses in measurement, addressed here in the solve.
- In blender_worker._skin: weld by distance before the heat attempt
(bbox-relative epsilon; Blender's merge-by-distance keeps per-loop UVs and
materials), recalc normals, bind. Record the method in rig_meta
(automatic-welded), joining the vocabulary 0d surfaces.
- The render mesh's appearance must be unchanged — compare exported texture byte
hashes and a seam-visible render before/after.
- Automatic fallback chain: weld → verify → else unwelded heat → else envelope.
Envelope stays the floor.
- Deformation QA, human-reviewable first: a pose battery (squat, arms overhead,
elbow/knee 90°, torso twist) rendered through the existing sheet pipeline and
attached to the rig job. Poses as JSON data under templates/, following "a
skeleton is a JSON template". Scoring waits for Phase 6.

---
## Phase 5 — gltfpack tier qualification (binary present; corpus from Phase 3)

- Harness: run draft/standard/detailed through the existing
pipelines/optimize.py + service.jobs.optimize_job path against a chest, a
sword and a rock. Automated per-tier checks — UVs survive, both PBR maps survive,
material assignment survives — plus a before/after render sheet.
- The corpus must be Phase 3's accepted meshes, not the rejected sweep: 80 of
the existing 83 were rejects, and preservation cannot be judged on broken output.
- On pass, expose the tiers in panes/settings_3d.py:27 (PROFILES) and verify the
retarget panel already shows the full list. Leave Config.mesh_profile at raw
— the default flip is a separate decision.

---
## Phase 6 — DINOv2 judge (TODO.md §8–11 verbatim)

Verified absent: no src/warlock/judge.py, no verdicts.stage, no migration 6
(there are 5), no db.unlabelled_references. Implement the written design — do
not re-design it.

- Migration: verdicts.stage ('reference' | 'blank' | 'model', existing rows
backfill to 'model'); db.unlabelled_references() — unverdicted_models cannot
be reused, it filters out errored jobs and sweep units, the two things a
labelling pass most needs. findings.py filters by stage inside _marginals,
so the per-subject prompts section is covered too. (Coordinate the number with
Phase 2.)
- judge.py pure, in the vram.py/memlog.py sense: stdlib + numpy, no
service/queue/studio imports, None rather than raising, offline with
local_files_only=True, and a judge failure never fails a job. Three linear
probes over frozen DINOv2 CLS embeddings — image-as-product, image-as-blank,
mesh. The mesh probe pools max or mean over 8 views, never single-view
(docs/measurements/2026-08-04-view-calibration.md: no fixed matched view exists;
argmax scattered 330°/300° against a STABLE_YAW_SPREAD of 30). Advisory-only.
The .npz carries corpus size, label count and schema version — the
vendor/warlockc staleness hazard exactly.
- Labelling UI in Review: thumbnail grid, A/R keys, no reason step; texture
uploads paced one per step() (the viewer/sheet.StripRender lesson); retrain as
a pumped flag, never a direct ctx.submit (the findings_dirty /
pump_findings pattern); sort rather than filter; training on TaskRunner.
- Phase 2 of the judge is unbuildable on today's corpus — 4 positives against 96
negatives; a probe fitted to that learns "reject" and scores 96% doing it. The
gate is a positive count in the tens. The existing labels are a clean negative
set, and the matched pairs (identical input.png, opposite verdict) are the most
useful rows in it.
- Acceptance (§11): a measurement doc reporting false-reject and false-accept
rates, agreement with reference.py's rules on the blank probe, a
per-prompt_hash breakdown, and beating a coin flip — hole_worst's 0.115
AUC is no floor. Four questions decided with data: max vs mean pooling; binary vs
five-class REASONS; per-prompt_hash or global; whether an ai: verdict feeds
findings.json or only sorts Review.
- Labels must be human. Training on the gate's own refusals teaches the probe to
imitate reference.py including its blind spots.

---
## Phase 7 — Retopo + bake prototype (needs Phase 4; characters only)

- A new blender_worker op producing a deformation mesh from the immutable
source: remesh (voxel or QuadriFlow, explicitly labelled a preview/proxy), UV
unwrap (smart-project first), bake base colour / metallic-roughness / tangent
normals from the source, then rig it through the Phase 4 path.
- Output is a new labelled artifact (e.g. deform.glb), derived on demand under
_convert_locks, never replacing model.glb. UI labels: Reconstruction /
Static game-ready / Preview riggable / Deformation-ready — so a decimated
reconstruction is never presented as real retopology.
- Prototype on 2–3 accepted character meshes; a QA note on bake fidelity before any
UI default.

---
## Phase 8 — External backend A/B (last; own spec first)

- SkinTokens/TokenRig as an isolated out-of-process worker (the
trellis-server/blender_worker pattern; kill-on-close job; weights by one-time
manual download — or via 9g's fetcher — with doctor reporting absence
non-fatally). Run in existing-skeleton mode against Warlock's template so bone
names and animation compatibility hold; A/B against Phase 4's welded heat weights
on a fixed rig corpus using the Phase 4 pose battery, verdicts through Review.
- Hunyuan3D 2.1 as an optional isolated reconstruction backend, same isolation
rules, benchmarked on the curated Phase 3 references — human acceptance,
silhouette, runtime, VRAM. It must fit the coexist/exclusive vram.plan
machinery: a new backend declares its GiB cost in the cost table.
- Explicitly out: MeshAnything V2 as a retopo path (sub-1600-face target).

---
## Deferred by decision (not open work)

Item: seam.SEAM_MAX = 2.0 uncalibrated (pipelines/seam.py:27)
Status: Open, low priority — needs stone / plaster / gravel / fabric tiles eyeballed. Corpus-keyed, so it owes a measurement

doc before it moves.
────────────────────────────────────────
Item: docs/measurements/2026-08-06-pixel-art-xl.md — "run not yet taken"
Status: Unblocked today: all three recipes and all weights verified present. Three-arm run settling which arm pixel_sprite
names and where GRID_RESIDUAL_MAX belongs (0.05; the doc says "that number is a guess"). Independent of everything — good
use of idle GPU time alongside Phase 3.
────────────────────────────────────────
Item: Fused brush dab kernel (warlockc_dab_u8)
Status: Deferred on purpose; the gate is "the brush shows up in a profile first". ABI 5, four kernels shipped.
────────────────────────────────────────
Item: studio/clay/ops_topo.py:567 hole-fill UV
Status: Documented, deliberate approximation.
────────────────────────────────────────
Item: A9 (view-matched ranking)
Status: Not built on purpose; the Scattered verdict is the deliverable.

---
## Verification

- Every package: uv run pytest and uv run ruff check ., plus a run with
WARLOCK_NATIVE=0 (no native kernels are touched, so both must pass). Baseline
3110 passed, 7 skipped, lint clean. In a worktree set the three WARLOCK_* env
vars or expect 427/6-skipped.
- Standing UI guards must stay green for every Phase 9 package:
test_no_pane_continues_a_line_that_has_no_room_left
(tests/test_studio_smoke.py:319) and
test_no_two_of_a_panes_icon_buttons_are_drawn_on_top_of_each_other (:1360).
The second exists because the first cannot see a control drawn on top of
another — both matter for 9a/9b/9c.
- Phase 0 end-to-end (via the /run skill): promote a 2d-platform reference
and confirm resolution 1024 in trellis.log; doctor shows matting green after
uv sync; rig a mesh and see the weighting outcome.
- 0c: the synthetic seam-split cube reports raw not-watertight, welded
watertight — verify red before the fix.
- 0e: the byte-identity test on the default prompt composition is the entire
safety argument for leaving PROMPT_VERSION at 4 — verify it red against a
deliberately altered clause.
- 9a: screenshot a pane of icon buttons before/after; assert the glyph's drawn
centre against the button rect's centre.
- 9g: with the network disabled, a download must fail cleanly and leave no
half-populated model directory; confirm the parent process still has
HF_HUB_OFFLINE=1 after a fetch; confirm the child appears in the kill-on-close
job and dies with the app.
- 9i: cold start shows the logo for ≥3 s; a slow start holds it until ready;
pressing X during the splash exits cleanly with no stranded python.exe (the bpy
probe runs in this window).
- Phase 3+: each measurement lands as a docs/measurements/ document before
any corpus-keyed default moves.
- The ledger at .superpowers/sdd/UPDATE/progress.md is git-ignored — a
git clean -fdx destroys it.
