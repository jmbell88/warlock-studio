# TODO.md — everything still owed, and who has to do it

Written 2026-08-21, consolidating the four plan files that had accumulated at
the root and in `docs/`. `MY_TODO.md`, `LPC_ALT.md` and `EXE_PLAN.md` were deleted
in that change, along with the two interop ledgers whose owed passes are now in
§4; git holds every one of them and `git log --all --diff-filter=D` finds them.

**This file has two kinds of entry and no others.**

1. **Work only a human can do**: art direction, authoring keyframes, opening a
   file in real Aseprite or real Tiled, running a card. None of it is derivable
   from the tree and none of it can be closed by writing code.
2. **Work that is fully specified and deliberately unstarted** — today that is
   the installer, the open Troupe phases, and the three host-commit defects in
   §8. Each is here with the argument that makes it actionable, not as a title.

**The moment an item could be built, it is built and struck out rather than
tracked.** A plan whose boxes disagree with the tree is worse than no plan, and
that is why every other roadmap file in this repository's history was deleted
rather than ticked. Nothing here is blocked on finding time.

**This file has no `§N` API.** The old `TODO.md §N` citations resolved against
a *different*, long-deleted file; all fourteen were rewritten on 2026-08-21 to
name the programme or the measurement document they meant, and
`tests/test_ux_todo_fixes.py` now refuses any citation of this filename from
`src/` or `scripts/`. Do not mint new ones. What a module needs to explain, it
explains where it is — or in `docs/INVARIANTS.md`, or in a
`docs/measurements/` document, both of which outlive any plan.

---

## 1. Art direction — blocks the whole Troupe pipeline

**The single highest-value thing on this list.** Everything downstream of it is
built and waiting.

- [x] ~~**Two or three pixel-art references** whose look you want.~~ **Supplied
      2026-08-21**, in `examples/`: `light_world.png` (the LTTP light world map
      — 4110×5136, indexed, 204 colours, the only one of the four that carries a
      real palette), `zelda_1.jpg` and `player_male_spritesheet.png` (the
      character bar: four directions, five rows, chunky limbs, dark outline).
      **The two character references are colour-destroyed and can only ever be
      look references** — the JPEG carries 19,909 colours and bakes transparency
      in as a checkerboard, and the sheet is a smooth upscale at 53,088. A ramp
      pulled from either would be JPEG ringing and bicubic interpolation
      wearing the name of art direction. A clean PNG rip is a minute's work if a
      character ramp is wanted.
- [x] ~~**Palette ramps**~~ **Installed 2026-08-21 — the directory had been
      empty**, which is *why* this blocked: `colors.gpl` sat in `examples/`
      where nothing reads it, so every pixel export and every Troupe sheet to
      date ran on median cut. `~/.warlock/palettes/` now holds **`cosmos`** (The
      Cosmos, 64, the chosen bar — pass it as Troupe's `palette` param) and
      **`light_world`** (64, derived from the map's own index table). Both
      verified through `service.palettes.load` and `pixel.map_palette`.
      They stay *user data and out of git*, per the standing rule the manual
      states: nothing ships with the app, because a palette is a decision about
      which colours exist.
      One finding worth keeping, recorded in the derived file's own header: a
      coverage rule alone would have kept the map rip's **canvas** — a flat grey
      covering 14.29% of the sheet, ranking *first* by coverage — so the
      derivation drops any colour filling complete scanlines before it ranks
      anything. No map feature spans 4110 px; padding does.
- [ ] **A textured base mesh** (or a run through the Phase 4 reference chain).
      Both `examples/*_base.obj` carry no texture, so every frame still
      quantises into the pale end of whatever ramp it is given — **and now that
      the ramps exist, this is the only thing between them and a verdict.** The
      palette is provable on 2D today (a tile sheet has real colour) and stays
      unproven *on Troupe* until this lands.

## 2. Author the 22 keyframes — the editor is built for exactly this

Open Poser → **Clips** in the left sidebar. Pick a key, pose the skeleton with
the normal gizmos, **Update key from pose**. Onion skin ghosts the keys either
side; **Play** scrubs the real interpolation. **Save clips** writes to your own
data folder and never touches what the build ships, so **Revert** is always
available and an update cannot overwrite you. Manual: *Poser → Editing clips*.

- [ ] **Author the keys.** The shipped 22 are provisional. This is the most
      important art task in the programme, and the one thing that cannot be
      automated away — moving authoring from frames to keyframes does not make
      animation good, it makes it cheap to fix. A bad clip reproduces exactly
      the "stiff posing" flaw being escaped.
- [ ] Decide **who does it** — you, or an animator. Still a scheduling
      question.

Two things worth knowing before you start:

- **Easing does nothing at the current segment lengths.** It reshapes *where
  inside a step* frames land, so it needs a step of ≥3 frames. Every shipped
  step is 1 or 2, and `ease` is a smoothstep whose value at the only interior
  sample is exactly 0.5 — so `idle`'s `ease` renders identically to `linear`
  right now. `ease_in`/`ease_out` do differ. The panel says so; it is mentioned
  here because otherwise you would change it, see nothing, and assume the field
  is ignored.
- **The arms hang slightly forward** on the shipped keys (noted at the Troupe
  0d spike, still true). Worth an art pass while you are in there.

## 3. Decisions owed

- [ ] **Where does the "judge clips as pixels" preview live?** The Troupe plan
      asked for a live low-res *sprite* preview in Poser. **It cannot go
      there**: `template_preview` builds an armature-only GLB over the canonical
      unit box, so Poser's preview is a *meshless armature* and there is nothing
      to pixelise — a sprite preview would show reduced bone lines. Either Poser
      learns to load a rigged asset for preview, or the pixel verdict stays in
      Troupe where the mesh is. The scrubber shipped instead as the fast loop,
      which is right either way. **Your call which direction.**
- [ ] **Troupe Phase 6 — the cleanup workflow.** See §6 below. It is three
      features and the hard one is a genuine design problem, not an
      implementation task. Worth a conversation before anyone writes code.

## 4. The interop passes — need an app this repo does not have

Both follow the standing rule, now stated once at the top of `docs/COMPAT.md`:
a green test proves this app's reader and this app's writer agree with *each
other*, and a round trip through our own two halves cannot catch an error both
halves make together. **The claim only strengthens once a human with the app
has looked.**

- [ ] **Open a Warlock-written `.aseprite` in real Aseprite.**
      `tests/inker/fixtures/aseprite/FIXTURES.md` names the four fixtures worth
      authoring first. **Start with the tilemap ones** — `tilemap-rgb`,
      `tilemap-indexed`, `spare-tileset`. Their chunk field order was written by
      inverting the *reader*, field for field, and has never been checked
      against a file Aseprite itself wrote. That is the highest-value five
      minutes on this whole list.
      In the same sitting: every RGB and grayscale file now carries a palette
      chunk derived from the art's own colours (divergence #23) — including a
      1-entry transparent palette on a blank document. Check Aseprite is happy
      with that rather than replacing its own default with a single swatch.
- [ ] **Author a `.tmx`/`.tsx` fixture in real Tiled.** Every map under
      `tests/plotter/fixtures/tiled/` was produced by this editor, so every
      `round-trips` row in `docs/COMPAT.md`'s Tiled part is currently a round
      trip *against ourselves*. `tests/plotter/fixtures/tiled/FIXTURES.md` lists what
      authoring is owed and in what order.
- [ ] **Move `tsx.TILED_VERSION` from `1.10.2` to `1.12.2`** — but only after
      the above. The constant is pinned below the 1.12.2 target deliberately;
      nothing in this repo can satisfy the gate, and it has been wrongly bumped
      once already (the bump was reverted rather than the gate satisfied).
- [ ] **Re-check a grid pack's `.tsx` geometry.** pow2 rounding is off by
      default now, so the standing verification is stale.

## 5. GPU runs — need your card

- [x] ~~**`uv run pytest -m gpu -n 0`.**~~ **Run 2026-08-21: 21 passed, 5
      errors**, and the five were one stale attribute rather than a defect on
      the card — the three-views change renamed `SheetGeometry.projection` to
      `.view` and `tests/test_tilesheet_gpu.py` was the one caller it did not
      reach, because that file only runs when someone asks for it. Fixed the
      same day. Serial is still enforced, and this is still the only lane that
      sees real weights and the real `~/.warlock`.
- [x] ~~**Re-run the one file the errors ate**~~ **Run 2026-08-21 as part of a
      full `uv run pytest -m gpu -n 0`: 26 passed, 0 errors, 104 s**, with
      `tests/test_tilesheet_gpu.py` contributing all five. Its two load-bearing
      claims — the model puts its seams where the guide drew them, and the
      reduction keeps a cell's contrast — are asserted again for the first time
      since the view vocabulary widened. Top-down only; the isometric diamond
      guide and the 3/4 clause remain unproven on a card, and that is still a
      separate ask.
- [ ] **Run a `charsheet` job end to end against real Blender.** Troupe Phase
      4's job has *never* been run on a card. The pieces either side of it have
      (Phase 0d), and the render call is `rigging.sheet_spec` + `run_worker`
      exactly as `_sheet` makes it — but the end-to-end run is owed, and it is
      how you would find out that the clip edits from §2 actually reach a
      rendered sheet.

## 6. Troupe — the open phases

Phases 0a–0d, 1, 2, 3, 4 and 5 are **implemented and verified**; what they
established lives in `docs/INVARIANTS.md`, and the measured ULPC facts the
programme rested on are passing oracles in `studio/troupe/ulpc.py`. What
follows is what is left.

### Phase 0e — humanoid reconstruction from a single image

Untested. Run a prompt → reference → TRELLIS → `fit_template` pass on a humanoid
and judge **limb separation and silhouette** — at sprite scale, face-level
fidelity barely matters and those two matter enormously. This is the largest
unproven assumption in the automated chain, and it is compounded by a known
property rather than a bug: Warlock is single-image-only for reconstruction, so
**the back is hallucinated**, and a humanoid with separable limbs is a harder
subject than a prop.

It matters **only for the generated-character path**. The supplied-base-mesh
path works without it, which is why the programme did not stop for it.

### Phase 6 — the cleanup workflow

Export mostly exists (`inker/sheetout.py`, `aseout.py`, `packwright/tsxout.py`).
The work is the loop:

- **Propagate a correction** across frames / direction / animation — fix a pixel
  once, apply it everywhere compatible. Inker's ranged ops (`_doc_ranges.py`)
  are most of the machinery. Must go through the write funnel (`_commit_patch`)
  and address by uid, per `docs/INVARIANTS.md`.
- **Mirror-assisted cleanup.** The W/E mirror property was measured on the
  reference sheets: mirroring leaves 36–37 differing pixels, confined to the
  face, and every non-zero shift is far worse (443px at ±1) — so it is real
  facial asymmetry rather than a centring offset. A fix on one side can be
  offered on the other, face excluded.
- **Re-render one animation without discarding hand edits.** The hardest
  workflow problem in the programme and the one most likely to be discovered
  too late. **Design it deliberately rather than on contact** — that is the
  standing instruction, and it is why this phase is a conversation before it is
  a commit.

### Phase 7 — layered equipment (deferred)

Swappable gear, once whole-character generation works.

- **Multi-GLB scene composition.** `op_sheet` takes one `source_glb`, and
  equipment items are separate assets by construction — so the task is
  *composing* N GLBs under a shared camera, not splitting one. `op_rig` joins
  every mesh into a single object, which is exactly why splitting is a dead end
  and composition is the right shape.
- **Per-part passes with depth**, giving correct per-direction occlusion for
  free — no z-order table to maintain. The depth machinery is proven in
  `op_views._depth_material` and can move onto the sheet path.
- **Garment fitting**: skin-weight transfer from the weighted body by proximity
  (Blender `Data Transfer`). Tractable for hugging garments; capes and long
  skirts are a separate problem — scope to hugging first.

### Phase 8 — reconsider only against a working system

- **AI restyle** — `create_pixel_sheet` with `structure_lock` over a rendered
  sheet. Note `structure_lock` is only a Canny-ControlNet toggle; what actually
  keeps silhouettes exact is `pixelsheet.remask()` stamping the render's own
  alpha back unconditionally. Also note `check_restylable` refuses
  `frame_size × columns > 1024`, so a 512-render 8-column sheet cannot be
  restyled without banding. Opt-in, measured, never default.
- **A learned pixel refiner.** Once cleanup is routine, `(render, hand-cleaned)`
  pairs accumulate for free, perfectly registered, over a fixed palette. A
  well-posed supervised problem — and it automates the Dead Cells cleanup step
  rather than trying to relearn pose transfer, which the geometry already
  solved.
- **More animations.** The spec is extensible by construction; hurt, death, cast
  and climb are additive.
- **Natural-language character description** — only over a working catalog, only
  local weights through `fetch_worker`, following the `expand.py` precedent.

### Inputs the repo cannot produce

Beyond the art direction in §1, a base mesh must satisfy: **GLB/glTF**
(`blender_worker._import_glb` is the only importer on this path); **T-pose or
A-pose** (both `fit_template`'s bbox-proportional fit and automatic-weight
skinning degrade badly on a dynamically posed mesh); **+Z up, −Y forward**;
bone names mapping onto the 19-bone template if it ships rigged (Mixamo or
Rigify naming is fine but needs a mapping table); **no very short bones** —
Blender silently deletes a bone below a minimum fraction of the mesh's largest
dimension *and takes its children with it*, a quiet failure, with fingers and
toes the usual casualties; **under ~300k faces**, because automatic weights on
a 300k-face mesh is minutes of CPU; **male and female variants**; and a
**licence permitting commercial redistribution of derived rendered sprites**.

The reference sheets in `examples/` are ULPC-derived and CC-BY-SA/GPL:
**reference and validation only** — keep them out of shipped art and out of any
training set, and the question stays closed.

## 7. The installer — fully specified, unstarted

Written 2026-08-16 and unchanged since. Warlock is source-checkout-only today by
explicit decision (DST-01: `config.source_checkout()` gates startup in
`studio/main.py`, and all native binaries resolve against `PROJECT_ROOT`). The
goal is a Windows installer so the app can be installed on a laptop.

Decisions already taken:

- **The installer ships only the runtime** — a bundled Python 3.13 env from
  `uv.lock`, `vendor/` binaries (~836 MB trellis + gltfpack + warlockc), the
  source package, assets and the manual. **No model weights** (~6.4 GB payload,
  ~3–4 GB compressed).
- **SDXL (~7 GB) and the TRELLIS GGUFs (~16 GB) become first-run in-app
  downloads** through the existing fetch pipeline.
- **LoRAs stay curated/registry-only** — the existing Settings → Models UX, no
  drop-in folder.
- **Hardware preflight at first run** (the laptop GPU is unknown, and no CPU
  fallback exists).
- **Personal now, shareable later** → one scripted, repeatable build command. No
  code signing, no auto-update (upgrade = run the newer installer).

### The core design: the installed layout is checkout-shaped

The installer lays down `pyproject.toml`, `src\warlock\`, `vendor\`,
`docs\manual\` at the install root plus a bundled CPython under `python\`.
`PROJECT_ROOT = Path(config.py).parents[2]` then resolves to `{app}`,
`source_checkout()` returns True **unchanged**, and every vendor default
resolves correctly with **zero path-code changes**. A bare `pip install` wheel
still fails the probe — that refusal survives intact. Only docstrings and
`docs/INVARIANTS.md` change: the contract becomes "a checkout-shaped layout",
produced by either a git clone or the installer.

```
{app}\
  pyproject.toml  CHANGELOG.md  README.md  install.json
  src\warlock\**                    <- package, byte-identical
  docs\manual\*.md
  vendor\trellis\**  vendor\gltfpack\**  vendor\warlockc\**
  python\                           <- full CPython 3.13 (python-build-standalone via uv)
    Lib\site-packages\warlock_app.pth   <- one line: ../../../src
  bin\warlock-doctor.cmd
```

`sys.executable` is a real `{app}\python\python(w).exe`, so all four subprocess
workers (`[sys.executable, "-m", "warlock.pipelines.<worker>"]`) work unchanged;
the `.pth` (relative to site-packages) makes the tree relocatable with no
launcher env vars.

### Phase 1 — gate rework + entry point (small, first)

- `src/warlock/config.py` — the probe stays **identical**; rewrite the
  docstring so the contract is a checkout-shaped layout (clone or installer),
  wheels still refused. Optionally add `installed_build()` reading
  `install.json` for a log/About line — nothing gates on it.
- `src/warlock/studio/main.py` — fix the stale `uv run warlock-studio` →
  `uv run warlock`; mention the installer; update the comment.
- **New** `src/warlock/__main__.py` — enables `pythonw.exe -m warlock` for the
  Start Menu shortcut, with the pythonw null-stdio shim (under pythonw,
  `sys.stdout`/`stderr` are `None` and pygame's import banner would raise):
  ```python
  import os, sys
  if sys.stdout is None: sys.stdout = open(os.devnull, "w", encoding="utf-8")
  if sys.stderr is None: sys.stderr = open(os.devnull, "w", encoding="utf-8")
  from warlock.cli import main
  main()
  ```
- `docs/INVARIANTS.md`, the DST-01 block — rewrite to the checkout-shaped
  contract.

### Phase 2 — a TRELLIS GGUF fetch record (new `engine` kind)

A new kind, not a row in an existing table: every existing table is a
torch/diffusers artifact under `t2i_model_root`, while the GGUFs are the native
engine's weights at `config.trellis_models_dir` (`WARLOCK_TRELLIS_MODELS`).
`fetch.KINDS` being the single vocabulary means Settings → Models and
`downloads.rows()` pick it up with no pane edits.

- `src/warlock/models.py`: an `EngineModel` dataclass + `ENGINE_MODELS` with one
  entry `trellis_gguf` — Fetch(repo_id `ilintar/trellis2-gguf`, revision
  `a57397bd3d351599d9729fc144b3f87c3f87d65b`, allow_patterns `("*.gguf",)`,
  ignore_patterns `("q4/*","q8/*")`, size_gib 16.1); `probe` = the 10 named
  ggufs (ss_flow, ss_dec, shape_flow_512/1024, shape_dec, tex_flow_512/1024,
  tex_dec, birefnet, dinov3), so a partial fetch reads absent.
- `src/warlock/fetch.py`: `KINDS` gains
  `Kind("engine", models.ENGINE_MODELS, "engine: ", "Reconstruction engine")`
  **first** (it tops doctor order and the pane). Branch `engine` in
  `destination()` (→ `config.trellis_models_dir`), `present()` (all 10 probe
  files exist), `claims()` (→ `trellis_models_dir`, so the containment refusal
  in `removal_plan` applies) and `suspect_files()`.
- `src/warlock/doctor.py`: `_gguf_check` keeps the row name
  `"TRELLIS GGUF weights"` verbatim and `fatal=True` (the banner path is
  unchanged) but probes and renders its remedy via the registry record; delete
  the `TRELLIS_GGUF_HINT` literal so the pin gets one owner, and rewrite the
  "belongs in install instructions" comment. `_t2i_checks` deliberately does
  **not** add an engine row (the fatal row is its row) — say so in a comment.
- The fetch worker, the two-phase transaction and the manifest machinery need
  **no changes**: staging beside the destination, ETag verify, publish rename,
  and `.warlock-fetch.json` is dot-prefixed so the exe's `*.gguf` scan never
  sees it.

### Phase 3 — first-run overlay + hardware preflight

- Marker `config.home / "first-run.json"`, written on dismissal;
  `ctx.first_run` decided once at startup. A fresh `WARLOCK_HOME` ⇒ the overlay
  shows.
- **New** `src/warlock/studio/panes/first_run.py`, a modal overlay while
  `ctx.first_run`, formatted entirely from data already computed at startup
  (doctor rows plus the resolved `vram_plan` — no new probes):
  1. **Hardware**: GPU name + VRAM; verdicts for *3D reconstruction* (CUDA row
     ok AND VRAM row ok → Ready, else "requires an NVIDIA GPU with ~N GiB free;
     no CPU fallback"), *Image generation* (`vram.fits` on the default base) and
     *Rigging* (`ctx.rigging_available`).
  2. **Required downloads**: rows `engine:trellis_gguf` + `base:sdxl_cfg`, a
     deduped total (~23 GB), and `fetch.disk_refusal`'s message if the volume is
     short.
  3. Buttons: **"Download models (~23 GB)"** → `model_gate.request_install(ctx,
     rows)` (the existing pattern — pre-ticks and jumps to Settings → Models);
     **"Not now"**. Both write the marker.

### Phase 4 — build script + InnoSetup script (new files, additive)

**New** `installer/build.ps1`, one command (`pwsh installer\build.ps1`):

1. Version from `uv version --short`; assert it matches
   `src/warlock/__init__.py`.
2. Clean `build\stage\`; copy uv's managed CPython 3.13
   (python-build-standalone, built relocatable) whole into `stage\python`.
3. `uv export --frozen --no-dev --no-emit-project --extra studio
   --extra text2image --extra rig -o build\requirements.txt`, then
   `uv pip sync` it into the staged interpreter. **Hard assert**:
   `stage python -c "import torch; assert torch.version.cuda == '12.8'"`.
   Fallback if uv pip ignores `[tool.uv.sources]`: add
   `--index https://download.pytorch.org/whl/cu128
   --index-strategy unsafe-best-match`.
4. Write `warlock_app.pth` (`../../../src`); copy `src\warlock` (minus
   `__pycache__`), `pyproject.toml`, `CHANGELOG.md`, `README.md`,
   `docs\manual`, `vendor\{trellis,gltfpack,warlockc}`; write `install.json` and
   `bin\warlock-doctor.cmd`.
5. `compileall` the staged tree, so nothing writes `__pycache__` into the
   install directory at runtime.
6. **A staged smoke test** under a fresh temp `WARLOCK_HOME`: `stage doctor`
   runs, finds `trellis-server.exe` at the staged path and reports GGUF/SDXL
   missing with pinned commands; `import warlock.studio.main` succeeds.
7. `iscc /DAppVersion=$v /DStageDir=... installer\warlock.iss` →
   `dist\WarlockSetup-vX.Y.Z.exe`.

**New** `installer/warlock.iss` — a fixed AppId GUID; `PrivilegesRequired=lowest`
with a per-user default of `{localappdata}\Programs\Warlock Studio` (no UAC;
Program Files remains possible via override, since nothing writes to the install
directory); `Compression=lzma2` + `SolidCompression`;
`DiskSpanning=yes`/`DiskSliceSize=2100000000` (try single-exe first, span if
ISCC balks at the ~4 GB payload); `SetupIconFile=src\warlock\assets\icon.ico`;
an `[InstallDelete]` that clean-slates `python\Lib\site-packages` and `src` on
upgrade; `[Icons]` — Start Menu "Warlock Studio" =
`{app}\python\pythonw.exe -m warlock` (WorkingDir `{app}`), "Warlock Doctor" =
the cmd, plus an optional unchecked desktop icon; `[UninstallDelete]` removing
`src`/`python` strays. Uninstall leaves `~/.warlock` in place with an
informational message ("your assets and downloaded models remain at
%USERPROFILE%\\.warlock").

### Phase 5 — tests, docs, version

- `pyproject.toml`: declare `huggingface_hub` explicitly (`fetch_worker` imports
  it directly; today it is transitive); `uv lock`.
- Existing tests auto-cover the new record — the 40-hex pin test, size > 0 and
  doc-pin sync. The "docs name every repository" test **forces** adding
  `ilintar/trellis2-gguf` and a pinned `uvx hf download` line to
  `docs/MODELS.md`. Do it.
- New tests: `warlock.__main__` imports and names `cli.main`; the refusal text
  no longer says `warlock-studio`; engine-kind coverage (destination honours
  `WARLOCK_TRELLIS_MODELS`; `present()` false on partial and true on 10 one-byte
  fixtures; `removal_plan`'s containment refusal; `plan()` for both first-run
  rows totalling ~23 GiB deduped); and a first-run pane test in the existing
  headless-ctx pane pattern (hidden once the marker exists; Download unions the
  picks and switches mode).
- Manual — **edit in place, no new chapters** (numbering is test-gated):
  `docs/manual/19-installation.md` puts the installer path first and the GGUF
  section becomes "downloaded in-app"; `docs/manual/20-configuration.md` notes
  that `WARLOCK_TRELLIS_MODELS` also moves the in-app download;
  `docs/manual/21-app-settings.md` gains the "Reconstruction engine" group and
  the first-run overlay; `docs/manual/22-troubleshooting.md` gains a SmartScreen
  note and a partial-GGUF-reads-missing note. (The plan named 17/18/19/20 when
  it was written on 2026-08-16; Troupe's chapter shifted every one of them, and
  the link checker is what caught it.) Update `docs/INVARIANTS.md`'s record counts.
- Version → the next patch, in the three lockstep places (`pyproject.toml`,
  `src/warlock/__init__.py`, the `CHANGELOG.md` heading).

### Phase 6 — verification (build machine first, then laptop)

1. Checkout: `uv run pytest` and `uv run ruff check .` green.
2. `pwsh installer\build.ps1` — smoke test and cu128 assert pass; record the
   size and compile time.
3. Install (per-user default; once with `/DIR=C:\Temp\WarlockApp /SILENT`). In a
   test shell with `$env:WARLOCK_HOME = "C:\Temp\wh-test"`: the shortcut
   launches under pythonw, the gate passes, the fatal banners list the two
   missing-weights rows, and the first-run overlay shows correct GPU verdicts,
   ~23 GB and the disk check.
4. The fetch pipeline with the bundled python: download one **small** row (e.g.
   dinov2, 0.4 GB) end to end — that proves worker spawn, verify, publish and
   banner refresh. Then SDXL and one reference generation. For TRELLIS, copy an
   existing `~/.warlock/models/trellis2-gguf` into the test home to prove the
   engine launches from `{app}\vendor` and reconstructs; separately start the
   engine download and cancel mid-flight to prove staging cleanup. That avoids a
   second 16 GB pull.
5. Upgrade over-install (a scratch version) — clean slate, data intact.
   Uninstall — `{app}` gone, `~/.warlock` intact.
6. The laptop.

### Risks and notes

- Inno single-exe versus DiskSpanning for the ~4 GB compressed payload is
  empirical; settle it at the first compile.
- The pythonw null-stdio shim runs before any pygame/imgui import; grep studio
  for a bare `print(` as a belt-and-braces audit.
- Keep the doctor row name `"TRELLIS GGUF weights"` verbatim, and check no test
  asserts the old hint literal.
- An old hand-downloaded partial GGUF set reads "missing" under the named-10
  probe. That is correct by the partial-reads-absent invariant, and it is
  documented in troubleshooting.
- An unsigned exe means SmartScreen's "More info → Run anyway". Accepted, and
  documented.

## 8. Host commit — a model load never gives its host memory back

Measured 2026-08-21 on a live session, from the app's own `warlock.log` plus
`Get-CimInstance Win32_Process`. It is here rather than built because the first
defect is a **design question** with three possible shapes, and the other two
are the instrumentation that decides which one is right — worth landing in the
same pass, because the figure that would judge the fix is currently wrong by a
third.

**The session that produced this.** Two reference generations, ten minutes
apart, on a 63.46 GiB machine with a 12.6 GiB pagefile — a 76.06 GiB commit
limit:

| time | event | app private | commit |
| --- | --- | --- | --- |
| 18:17:54 | idle at startup | **2.3 GiB** | 52.3/73.0 (72%) |
| 18:18:48 | `sdxl-base-1.0` done, VRAM 6.58 → 0.01 | **11.3 GiB** | 61.3 (84%) |
| 18:19:43 | `dreamshaper-xl` done, VRAM 6.74 → 0.02 | **13.6 GiB** | 63.6 (84%) |
| 18:25:54 | idle, six minutes later | **13.5 GiB** | 70.9 (93%) |

The card was empty throughout the tail (1.9 of 32.6 GiB). The session ended in
the app refusing its own work:

```
CRITICAL warlock.queue: host commit at 92% before unloading dreamshaper
  -- at or past the ceiling the 2026-08-03 crash hit; further jobs will be refused
RuntimeError: host memory is 92% committed before loading SDXL 1.0
  (full CFG, structural control), at or past the 90% ceiling.
```

- [ ] **D1 — the t2i load path pays an unreturnable host cost in the app
      process.** The `host_peak_gib` comment in `models.py` states the
      assumption this breaks: *"under RESIDENT the weights are read and handed
      to the device, so the host charge is **transient** and roughly the
      checkpoint's size."* It is not transient. `unload()` returns the VRAM
      exactly as it claims (6.58 → 0.01 GiB) and the host figure does not move
      — +9.0 GiB for the first checkpoint, +2.3 for the second, flat across six
      minutes idle. **Each distinct checkpoint pays its own**, so the ceiling is
      reached by switching models, not by running many jobs.

      The mechanism is already measured and already written down:
      `docs/measurements/2026-08-08-load-probe-memory.md` found that dropping
      every reference plus `gc.collect()` recovered 422 MB of BiRefNet's 1475 —
      **1053 MB, 71%, held by the allocator's arenas.** That document is *why*
      `loadprobe` and `matting_worker` are child processes. The t2i loader is
      the one path paying the same cost in the process that has to keep
      running, where nothing but exit reclaims it.

      **The decision owed is which fix**, and it is a conversation before it is
      a commit:
      1. **A t2i child process** — the trade `matting_worker` and `loadprobe`
         already made, and the only option that genuinely returns the memory.
         It costs the resident-pipe design: the pipe object the idle sweep
         deliberately keeps would live across a process boundary, and every
         LoRA / ControlNet / IP-Adapter handle with it. Large; do not start it
         casually.
      2. **Raise `host_peak_gib` to the truth and let admission refuse
         earlier.** Cheap and honest. Makes the app *correct* rather than
         better — the second checkpoint of a session becomes a refusal instead
         of a crash. Worth doing regardless of 1, because the shipped figures
         price a load that is released, and this one never is.
      3. **Accept it and recycle the process** — an explicit "restart to
         reclaim" affordance, which is what the user does today unprompted.

      **Do not rank these until D2 lands.** The app's reading of what it is
      charging is wrong by a third, and choosing on a bad number is how the
      wrong shape gets built.

- [ ] **D2 — every child pid the app records is the wrong process, so the
      idle-tick reports `children 0.0 GiB` against a 6.56 GiB child.**
      `sys.executable` under this uv venv is a **trampoline**, not an
      interpreter: it spawns the real CPython as its own child and stays alive
      as a ~0.8 MB parent. Demonstrated directly —

      ```
      sys.executable : D:\Projects\Warlock\.venv\Scripts\python.exe
      Popen.pid      : 3144
      child says pid : 2960      <- not the same process
      ```

      So `winjob.track(proc.pid)` records the shim. `memlog.children_private`
      dutifully opens it, reads 0.8 MB and rounds to zero — while the matting
      worker beside it holds **6.56 GiB**, which is the exact figure
      `children_private`'s own docstring cites as its reason for existing.
      Admission is **not** affected (`_require_commit_headroom` reads
      system-wide commit, which counts the grandchild), so this is a reporting
      defect and not a safety one. It is also the instrument D1 will be judged
      with, which is why it goes first.

      The fix is to record the pid that holds the memory rather than the one
      `Popen` returned, and it touches every worker spawn — `blender_worker`,
      `fetch_worker`, `matting_worker`, `loadprobe` — not just matting. Note
      that **the installer removes this by construction**: §7's layout gives
      `sys.executable` as a real `{app}\python\python.exe`, so the fix must not
      assume a trampoline is always there.

- [ ] **D3 — `matting.unload()` would orphan the process it means to kill.**
      Same root cause as D2, latent rather than observed. `unload()` calls
      `proc.kill()` on the trampoline, and Windows `TerminateProcess` does not
      cascade to children, so the 6.56 GiB grandchild would survive — reparented
      — until the app exits and the kill-on-close job takes it. That is the
      precise opposite of the module's stated purpose, that *"only a process
      that ends can"* return the memory. It has not fired in a session yet (both
      pids were alive when this was measured), so there is no incident to point
      at; it is waiting for the first idle sweep that calls it. The
      kill-on-close guarantee is unaffected either way — job objects *do*
      cascade — so nothing outlives the app.

**Not a repo item, but it is what the machine looked like.** `mysqld` held
8.40 GiB of commit with a *zero* working set, untouched since 2026-08-17; the
21.4 GiB standby cache is reclaimable and was never the problem; and the commit
limit is only 76.06 GiB because the pagefile is 12.6 GiB on a 63 GiB machine.
Warlock's own ~20 GiB was the largest movable share. Any measurement of a fix
has to name the other tenants, or it will credit itself with their departure.

## 9. Not on this list on purpose

These are decisions with arguments beside them, not backlog items waiting for
time.

- **Scale and crop of a tilemap layer** stay refused, permanently. They
  *resample*, and a tileset cannot follow a resample — there is no permutation
  to teach, only a re-cut, which is a different operation.
- **A hexagonal 120° tile rotation** stays refused. A 120° rotation of a square
  raster is not a permutation of the pixel grid, and the standing bar for a tile
  transform is that it invents no colour. `docs/COMPAT.md` carries the full
  argument, including what would have to change to re-open it.
- **Pen/tablet pressure, ICC colour, per-frame palettes, per-cel opacity** and
  the rest of the Aseprite parity programme's named non-goals, recorded in
  `docs/INVARIANTS.md`.
- **The Aseprite P1 backlog** stays unscheduled *by design* — items are pulled
  into sessions individually, never waved. It lives in `docs/INVARIANTS.md`.
- **An LLM director for Troupe.** Warlock has no LLM infrastructure, an
  OpenAI-compatible endpoint over local HTTP would be the first socket in the
  app besides the trellis subprocess client, and it would break
  `HF_HUB_OFFLINE=1`. Under the chosen flow it is also unnecessary: the user
  approves a *picture*, which is a better interface than a manifest.
