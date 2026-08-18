# EXE_PLAN.md — Warlock Studio InnoSetup Installer

Written 2026-08-16. Status: **planned, not started.**

## Context

Warlock is source-checkout-only today by explicit decision (DST-01: `config.source_checkout()` gates startup at `studio/main.py:5134`, all native binaries resolve against `PROJECT_ROOT`). The goal is a Windows installer so the app can be installed on a laptop. Decisions taken:

- **Installer ships only the runtime**: bundled Python 3.13 env (from uv.lock), `vendor/` binaries (~836 MB trellis + gltfpack + warlockc), source package, assets, manual. **No model weights.** (~6.4 GB payload, ~3–4 GB compressed.)
- **SDXL (~7 GB) and TRELLIS GGUF (~16 GB) are first-run in-app downloads** through the existing fetch pipeline. TRELLIS GGUF needs a new Fetch record (it deliberately has none today — doctor.py:243 rationale inverts for an installed app).
- **LoRAs stay curated/registry-only** — existing Settings → Models UX, no drop-in folder.
- **Hardware preflight at first run** (laptop GPU unknown; no CPU fallback exists).
- **Audience: personal now, shareable later** → one scripted, repeatable build command. No code signing, no auto-update (upgrade = run newer installer).

## Core design: the installed layout is checkout-shaped

The installer lays down `pyproject.toml`, `src\warlock\`, `vendor\`, `docs\manual\` at the install root plus a bundled CPython under `python\`. `PROJECT_ROOT = Path(config.py).parents[2]` then resolves to `{app}`, `source_checkout()` returns True **unchanged**, and every vendor default resolves correctly with **zero path-code changes**. A bare `pip install` wheel still fails the probe — that refusal survives intact. Only docstrings/INVARIANTS change: the contract becomes "checkout-shaped layout", produced by either a git clone or the installer.

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

`sys.executable` is a real `{app}\python\python(w).exe`, so all four subprocess workers (`[sys.executable, "-m", "warlock.pipelines.<worker>"]`) work unchanged; the `.pth` (relative to site-packages) makes the tree relocatable with no launcher env vars.

## Phase 1 — Gate rework + entry point (small, first)

- `src/warlock/config.py:14-39` — probe stays **identical**; rewrite the docstring: contract is a checkout-shaped layout (clone or installer); wheels still refused. Optionally add `installed_build()` reading `install.json` for a log/About line (nothing gates on it).
- `src/warlock/studio/main.py:5134-5151` — fix stale `uv run warlock-studio` → `uv run warlock`; mention the installer; update comment.
- **New** `src/warlock/__main__.py` — enables `pythonw.exe -m warlock` for the Start Menu shortcut, with the pythonw null-stdio shim (under pythonw, `sys.stdout`/`stderr` are `None`; pygame's import banner would raise):
  ```python
  import os, sys
  if sys.stdout is None: sys.stdout = open(os.devnull, "w", encoding="utf-8")
  if sys.stderr is None: sys.stderr = open(os.devnull, "w", encoding="utf-8")
  from warlock.cli import main
  main()
  ```
- `docs/INVARIANTS.md` (~line 278, DST-01 block) — rewrite to the checkout-shaped contract.

## Phase 2 — TRELLIS GGUF Fetch record (new `engine` kind)

New kind, not a row in an existing table: every existing table is a torch/diffusers artifact under `t2i_model_root`; the GGUFs are the native engine's weights at `config.trellis_models_dir` (`WARLOCK_TRELLIS_MODELS`). `fetch.KINDS` being the single vocabulary means Settings → Models and `downloads.rows()` pick it up with no pane edits.

- `src/warlock/models.py`: `EngineModel` dataclass + `ENGINE_MODELS` with one entry `trellis_gguf` — Fetch(repo_id=`ilintar/trellis2-gguf`, revision=`a57397bd3d351599d9729fc144b3f87c3f87d65b`, allow_patterns=`("*.gguf",)`, ignore_patterns=`("q4/*","q8/*")`, size_gib=16.1); `probe` = the 10 named ggufs (ss_flow, ss_dec, shape_flow_512/1024, shape_dec, tex_flow_512/1024, tex_dec, birefnet, dinov3) so a partial fetch reads absent.
- `src/warlock/fetch.py`: `KINDS` gains `Kind("engine", models.ENGINE_MODELS, "engine: ", "Reconstruction engine")` **first** (tops doctor order and the pane). Branch `engine` in `destination()` (→ `config.trellis_models_dir`), `present()` (all 10 probe files exist), `claims()` (→ trellis_models_dir; containment refusal in `removal_plan` then applies), `suspect_files()`.
- `src/warlock/doctor.py`: `_gguf_check` keeps row name `"TRELLIS GGUF weights"` and `fatal=True` (banner path at main.py:555 unchanged) but probes/renders remedy via the registry record; delete the `TRELLIS_GGUF_HINT` literal (pin gets one owner). Rewrite the 243–249 "belongs in install instructions" comment. `_t2i_checks` deliberately does NOT add an engine row (the fatal row is its row) — comment saying so.
- Fetch worker / two-phase txn / manifest machinery: **no changes** (staging beside destination, ETag verify, publish rename, `.warlock-fetch.json` is dot-prefixed → invisible to the exe's `*.gguf` scan).

## Phase 3 — First-run overlay + hardware preflight

- Marker `config.home / "first-run.json"`, written on dismissal; `ctx.first_run` decided once at startup. Fresh `WARLOCK_HOME` ⇒ overlay shows.
- **New** `src/warlock/studio/panes/first_run.py`, modal overlay while `ctx.first_run`, formatted entirely from data already computed at startup (doctor rows + resolved `vram_plan` — no new probes):
  1. **Hardware**: GPU name + VRAM; verdicts — *3D reconstruction* (CUDA row ok AND VRAM row ok → Ready, else "requires an NVIDIA GPU with ~N GiB free; no CPU fallback"), *Image generation* (`vram.fits` on the default base), *Rigging* (`ctx.rigging_available`).
  2. **Required downloads**: rows `engine:trellis_gguf` + `base:sdxl_cfg`, deduped total (~23 GB), `fetch.disk_refusal` message if the volume is short.
  3. Buttons: **"Download models (~23 GB)"** → `model_gate.request_install(ctx, rows)` (existing pattern, pre-ticks + jumps to Settings → Models); **"Not now"**. Both write the marker.

## Phase 4 — Build script + InnoSetup script (new files, additive)

**New** `installer/build.ps1` — one command, `pwsh installer\build.ps1`:
1. Version from `uv version --short`; assert it matches `src/warlock/__init__.py`.
2. Clean `build\stage\`; copy uv's managed CPython 3.13 (python-build-standalone — built relocatable) whole into `stage\python`.
3. `uv export --frozen --no-dev --no-emit-project --extra studio --extra text2image --extra rig -o build\requirements.txt`; `uv pip sync` it into the staged interpreter. **Hard assert**: `stage python -c "import torch; assert torch.version.cuda == '12.8'"` (fallback if uv pip ignores `[tool.uv.sources]`: add `--index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match`).
4. Write `warlock_app.pth` (`../../../src`); copy `src\warlock` (minus `__pycache__`), `pyproject.toml`, `CHANGELOG.md`, `README.md`, `docs\manual`, `vendor\{trellis,gltfpack,warlockc}`; write `install.json` + `bin\warlock-doctor.cmd`.
5. `compileall` the staged tree (no runtime `__pycache__` writes into the install dir).
6. **Staged smoke test** under a fresh temp `WARLOCK_HOME`: `stage doctor` runs, finds trellis-server.exe at the staged path, reports GGUF/SDXL missing with pinned commands; `import warlock.studio.main` succeeds.
7. `iscc /DAppVersion=$v /DStageDir=... installer\warlock.iss` → `dist\WarlockSetup-vX.Y.Z.exe`.

**New** `installer/warlock.iss` — key settings: fixed AppId GUID; `PrivilegesRequired=lowest` with per-user default `{localappdata}\Programs\Warlock Studio` (no UAC; Program Files possible via override since nothing writes to the install dir); `Compression=lzma2` + `SolidCompression`; `DiskSpanning=yes`/`DiskSliceSize=2100000000` (try single-exe first; span if ISCC balks at the ~4 GB payload); `SetupIconFile=src\warlock\assets\icon.ico`; `[InstallDelete]` clean-slates `python\Lib\site-packages` and `src` on upgrade; `[Icons]` — Start Menu "Warlock Studio" = `{app}\python\pythonw.exe -m warlock` (WorkingDir `{app}`), "Warlock Doctor" = the cmd, optional unchecked desktop icon; `[UninstallDelete]` removes `src`/`python` strays; uninstall leaves `~/.warlock` with an informational message ("your assets and downloaded models remain at %USERPROFILE%\.warlock").

## Phase 5 — Tests / docs / version

- `pyproject.toml`: declare `huggingface_hub` explicitly (fetch_worker imports it directly; today it's transitive); `uv lock`.
- Existing tests auto-cover the new record: 40-hex pin test, size>0, doc-pin sync — the "docs name every repository" test **forces** adding `ilintar/trellis2-gguf` + pinned `uvx hf download` line to `docs/MODELS.md`. Do it.
- New tests: `warlock.__main__` imports and names `cli.main`; refusal text no longer says `warlock-studio`; engine-kind coverage (destination honours `WARLOCK_TRELLIS_MODELS`; present() false on partial / true on 10 one-byte fixtures; removal_plan containment refusal; plan() for both first-run rows totals ~23 GiB deduped); first-run pane test in the existing headless-ctx pane pattern (hidden once marker exists; Download unions picks + switches mode).
- Manual — **edit in place, no new chapters** (numbering is test-gated): `17-installation.md` installer path first, GGUF section → "downloaded in-app"; `18-configuration.md` note `WARLOCK_TRELLIS_MODELS` also moves the in-app download; `19-app-settings.md` "Reconstruction engine" group + first-run overlay; `20-troubleshooting.md` SmartScreen note + partial-GGUF-reads-missing note. Update INVARIANTS.md record counts ("21 records over 20 repositories").
- Version → 0.0.24 in the three lockstep places (`pyproject.toml`, `src/warlock/__init__.py`, `CHANGELOG.md` heading). Commit subject per convention: `Warlock v0.0.24`.

## Phase 6 — Verification (build machine first, then laptop)

1. Checkout: `uv run pytest`, `uv run ruff check .` green.
2. `pwsh installer\build.ps1` — smoke test + cu128 assert pass; record size/compile time.
3. Install (per-user default; once with `/DIR=C:\Temp\WarlockApp /SILENT`). Test shell with `$env:WARLOCK_HOME = "C:\Temp\wh-test"`: shortcut launches under pythonw, gate passes, fatal banners list the two missing-weights rows, first-run overlay shows correct GPU verdicts + ~23 GB + disk check.
4. Fetch pipeline with bundled python: download one **small** row (e.g. dinov2, 0.4 GB) end-to-end — proves worker spawn/verify/publish/banner refresh. Then SDXL and one reference generation. For TRELLIS: copy the existing `~/.warlock/models/trellis2-gguf` into the test home to prove the engine launches from `{app}\vendor` and reconstructs; separately start the engine download and cancel mid-flight to prove staging cleanup (avoids a second 16 GB pull).
5. Upgrade over-install (scratch version) — clean slate, data intact. Uninstall — `{app}` gone, `~/.warlock` intact.
6. Laptop.

## Risks / notes

- Inno single-exe vs DiskSpanning for the ~4 GB compressed payload: empirical, settle at first compile.
- pythonw null-stdio: shim runs before any pygame/imgui import; grep studio for bare `print(` as a belt-and-braces audit.
- Keep doctor row name `"TRELLIS GGUF weights"` verbatim; check no test asserts the old hint literal.
- Old hand-downloaded partial GGUF sets read "missing" under the named-10-file probe — correct by the partial-reads-absent invariant; documented in troubleshooting.
- Unsigned exe → SmartScreen "More info → Run anyway"; accepted and documented.
