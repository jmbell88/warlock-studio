# Warlock Studio — Public Release Readiness

**Version audited:** 0.0.28 (`b656335`, master level with origin)
**Date:** 2026-08-24
**Assumed release shape:** open-source the GitHub repository *and* ship the Windows installer
**Standard applied:** full quality bar — everything worth fixing before strangers see it

**Method.** Ten parallel audits over ten independent domains: test/CI health, licensing and
distribution, documentation, first-run and onboarding, the core pipeline and job system, the
application shell, the headless editor packages, model selection and output quality, security
and untrusted input, and performance and platform. Every finding marked ✔ was re-verified by
hand against the cited source before being written down; findings without ✔ are reported at
the auditing agent's own stated confidence. Several plausible agent findings were discarded
on verification and are not reproduced here.

---

## Resolution — 2026-08-24

**Everything in this report that could be closed by writing code has been, with
regression tests.** What remains needs a person, a card, a clean machine or a
credit card, and is recorded in `TODO.md` §8b rather than here.

The verdict below stands as written at audit time. This section is what changed
after it.

### Closed in code

| # | Finding | What was done |
|---|---|---|
| B1 | No licence | **GPL-3.0-or-later.** `LICENSE` (verbatim GPL text), `pyproject.toml` `license`/`license-files`/classifiers, `LicenseFile=` in the `.iss`, a README section explaining *why* GPL, and a `CONTRIBUTING.md` inbound term. |
| B2 | `examples/` published third-party art | Untracked and gitignored (20 files, ~26 MB), and `pyproject.toml` gained a `[tool.hatch.build.targets.sdist]` **allowlist** — an exclude only stops what somebody thought of, and the failure was a directory nobody thought of. sdist went **41.2 MB → 9.9 MB with zero `examples/` members**. `tests/test_release_hygiene.py` pins both. *The history purge is still owed.* |
| B3 | GPL `bpy` in the installer | Resolved by B1's choice: one GPL-3.0 product, so the combination is coherent. Documented in the README and the notices file. |
| B4 | Model licences undisclosed | `BaseModel` gained `license` / `commercial` / `license_note`, populated for all eleven. The 2D picker draws a WARN line, the models table draws a `non-commercial` marker on the row the Download button is on, `docs/MODELS.md` gained a full table, and the README Setup block warns before the fetch. |
| B6 | No third-party notice | `THIRD-PARTY-NOTICES.md`, staged by `build.ps1` into the install root **and beside `vendor/`** — a test asserts every binary in `runtime-manifest.json` appears in it. |
| 2.1 | CI never tested the 3.12 floor; no wheel smoke test | A `floor` job on 3.12 without `--extra rig`, and a wheel-install smoke test that installs into a clean venv and imports from outside the checkout, asserting both force-includes. |
| 2.3 | Changelog reads as an instability signal | `_news_popup` shows full text for the running release and **opening sentences** for the history (`changelog.lead`, already used by the What's New card). `CHANGELOG.md` gained a preface explaining who it is written for. The fix is a shorter surface, not less candour. |
| 2.3 | "39 chapters", index lists 38 | 38. |
| 2.3 / 2.10 | RAM and generation time unstated | Requirements rewritten: **32 GB RAM** with the host-commit reason, 16 GB VRAM, ~23 GB disk, a display floor, and "roughly two minutes of GPU per attempt". |
| 2.4 | README led with three `hf download` commands | Setup leads with the in-app downloader and the first-run panel; the CLI moved to *"The same downloads, from a terminal"*, keeping the `--include` repetition trap. |
| 2.4 | No-CUDA machines passed VRAM admission silently | `doctor._vram_check` is **fatal** when a probe actually finds no device, with "no CPU fallback" in the text. |
| 2.5 | **Cancelling a restyle deleted a previous successful one** | The discard branch deletes only the two staging names; the worker `commit()`s its cancel token once the sidecar lands. The regression test was **verified to fail against the old code** first. |
| 2.5 | `_unload_under_lease` self-deadlock | Fixed in `leases`: `use()` is covered by the *same thread's* `maintain()`, which is provably safe (the exclusive holder is the only operation running) and fixes the class rather than the instance. A second test asserts a foreign thread still waits. |
| 2.6 | Clay outliner drag bypassed `tab.saving` | Refuses outright, mirroring `inker_timeline._reorder` — `begin_disabled` does not stop a drag-drop source registering. |
| 2.6 | Tour scrim darkened real modals | `App._modal_open` extracted to `main.modal_open`; the tour suspends its draw (not its state) while any modal is up. One list, not two copies. |
| 2.6 | Persisted `window_size` unvalidated | `_window_size` validates shape, floors at `MIN_SIZE`, and clamps to `get_desktop_sizes()`. This also closes **2.10's** unclamped default: 1600×950 at 125% asked for 2000×1187 on a 1080p panel. An explicit override stays unclamped for the screenshot harness. |
| 2.6 | Esc chain omitted the tour | Fixed in `docs/manual/37` and the `Ctrl+/` sheet. |
| 2.6 | Layout-editor outline hairline at 200% | `sp()`-scaled thickness. |
| 2.7 | **PNG save-in-place truncated the user's only copy** | `_write_atomic` — stage, `os.replace`, unlink on failure — applied to the in-place save and both export writers. |
| 2.7 / 2.9 | **`.aseprite` unbounded `zlib.decompress` ×3** | An `_inflate` helper bounded by each chunk's own declared arithmetic (`tmx._decompress`'s idiom) plus an absolute ceiling, because a cel's rectangle is two u16s and an "honest" 65535² one still asks for 17 GiB. |
| 2.7 | `write_ora` left its `.tmp` on failure | `try/finally`. |
| 2.9 | **Zip "claimed size" ceiling bypassable** | Measured first: a member declaring **10 bytes** transiently allocated **1,070 MiB** on stock `zipfile` before the CRC caught it — a 510 KB archive defeating a 1 GiB ceiling. New `studio/zipguard.BoundedZip` — a subclass, so the bound is a property of the archive object rather than a rule 18 call sites must remember. The same bomb now peaks at **0.1 MiB**. All four doors converted, import pins updated, and a scan test forbids a plain reading `ZipFile`. |
| 2.9 | Infinite-map `<chunk>` dimensions uncapped | `_chunk_side` caps at the engine's own `MAX_DIMENSION`. |
| 2.9 | `.tmx` traversal comment overclaimed | Corrected to say plainly that a relative path is anchored but *not bounded*, why that is an accepted trade in an offline app, and that it must be revisited before any outbound request lands. |
| 2.9 | Windows reserved device names | `reserved_check` on the composed filename: `CON`/`AUX`/`COM1`…, plus trailing dots and spaces (a collision `require_distinct_names` cannot see). `sheet_CON` stays legal — the reservation is on the whole stem. |
| 2.10 | ~1.6 s torch import at every startup | `probe_slow` now reaches `_vram_check`, which was the row defeating the documented deferral. Measured: **1,511 ms → 47 ms**, and torch is no longer imported at startup. The stale docstring that justified the old behaviour is corrected. |
| 2.10 | `httpx` CLI import taxing every start | Measured 160 ms (65 ms of it `httpx._main`, dragging in `rich`/`click`/`pygments`) for one `isinstance`. Now a `sys.modules` lookup — an `httpx.TransportError` cannot exist unless httpx is already imported. |
| 2.2 | No community health files | `SECURITY.md` (with a real in-scope/out-of-scope split for an offline app), `CONTRIBUTING.md`, two issue templates, and a `config.yml` routing security reports privately. |
| 4 | App silent about Troupe's maturity | `modes.MATURITY` drives an **"Experimental" chip** on the rail item — verified by screenshot. |
| 4 | Rail gave a word and an icon | `modes.PURPOSE` — one sentence per mode, passed into the `tooltip` argument `_item` already accepted and no call site used. The collapsed rail prepends the label. |
| 4 | "Send to Troupe" on non-humanoid meshes | Both hints now name the humanoid requirement; the inspector button had no tooltip at all. |
| 2.8 | "Seamless" stated confidently | Now *"likely seamless … check the wrap"*, matching what `docs/measurements/2026-08-09-seam-threshold-cfg.md` actually found on a CFG base. |

### Deliberately not changed

- **`can_send_to_troupe` still does not read the rig template.** The report
  suggests greying the action; the predicate's docstring already argues the
  trade-off — the template lives only in `rig.json`, which is a disk read asked
  every frame, and recording it on the mesh row would mean a worker writing a
  fact onto its source's params that must then join `DERIVED_PARAMS`. The
  service refusal is immediate, pre-mint and well worded. The real gap was that
  the *hints* never mentioned the rule, and that is fixed.
- **Undo memory bounding, the Plotter allocation-on-distant-click, guided-tour
  coverage for the remaining six workspaces, and the perf lane's breadth** stay
  deferred to 1.1, as this report's own plan proposed.

### Still owed — a person, a card or a clean machine

Recorded in `TODO.md` §8b: B5 (the installer on real hardware), the git-history
purge of the Nintendo and ULPC blobs, code signing, the graded `sdxl_cfg` mesh
run, the model-roster decision, the tile-sheet art direction, and verifying the
newly added 3.12 CI leg.

### Verification after the changes

```
$ uv run pytest -q
12090 passed, 17 skipped, 45 warnings in 51.26s

$ uv run pytest -m gpu -n 0 -q
26 passed, 12126 deselected, 62 warnings in 111.65s

$ uv run ruff check .
All checks passed!

$ uv run python scripts/preflight.py --fast
preflight passed

$ uv build
warlock-0.0.28.tar.gz              9,889,919 bytes  (1058 members, 0 from examples/)
warlock-0.0.28-py3-none-any.whl    8,030,042 bytes  (473 members)
```

---

## Verdict

**Not yet — but what stands between this and a public release is roughly a week of decisions,
paperwork and one measurement run, not months of engineering.**

The engineering is not the problem. The suite is **12,041 passing, 16 skipped, zero failures
in 68 seconds**; the GPU lane is **26 passed in 91 seconds**; ruff is clean; the
version-lockstep preflight passes; the wheel builds. There is not a single `TODO`, `FIXME`,
`XXX` or `HACK` anywhere in `src/warlock`, and the only `NotImplementedError`s are
abstract-method contracts. Nearly every untrusted parser carries an explicit, commented
decompression-bomb guard. Nearly every document writer stages to a temp file and
`os.replace`s. The offline promise is real, and every network-capable call site was traced to
confirm it. Repeatedly, an audit went hunting a known bug class and found it already fixed,
with the incident narrated in a comment at the site and a regression test naming it. This is a
more disciplined codebase than most commercial products of its size.

What blocks the release is almost entirely **legal and evidentiary**:

1. **There is no licence of any kind** — not in the repo, not in `pyproject.toml`, not in the
   installer. Open-sourcing without one grants nobody the right to use or fork it.
2. **The repository contains, and the source distribution ships, Nintendo-derived game art and
   ULPC CC-BY-SA/GPL assets.** All 20 files in `examples/` are git-tracked. Making the repo
   public publishes them.
3. **The installer bundles GPL-3.0 `bpy`** into one executable alongside code that currently
   has no licence at all.
4. **Two shipped models restrict commercial use of their output and the app never says so** —
   in a tool whose entire purpose is producing assets people will sell.
5. **The installer — the only path a non-developer will take — has never been run end to end
   on real hardware.** The project's own manual says so.
6. **The headline feature has no positive quality evidence.** The only completed graded mesh
   run scored 0 usable out of 20, and no graded run has ever targeted the shipped default
   model.

None of these needs new architecture. Four are decisions, one is a build run, one is a GPU
afternoon. The code defects that did turn up — and there are real ones — are a day or two of
work between them, and none is structural.

The honest framing: **a strong tool with an unfinished release process.**

---

## 1. Release blockers

These must close before the repo goes public or the installer reaches anyone.

### B1 — No licence anywhere ✔

`find . -iname 'LICEN*'` returns nothing outside `.venv`. `pyproject.toml` has no `license`
field and no classifiers. `installer/README.md:9-10` states it outright: *"No project licence
is selected or embedded by this installer input."* `installer/warlock.iss` has no
`LicenseFile=` directive, so Inno Setup shows the user no terms at all. GitHub reports
`licenseInfo: null`.

Without a licence, default copyright applies: nobody may legally use, copy, modify or
distribute the work, and nobody can contribute to it. This is the cheapest blocker to close
and it gates the others — the `bpy` question cannot even be framed until there is a licence
for it to be compatible with.

**Fix:** choose a licence; add `LICENSE`, `pyproject.toml`'s `license` and `classifiers`, and
`LicenseFile=` in the `.iss`. The choice interacts with B3.

### B2 — `examples/` publishes Nintendo-derived art and CC-BY-SA/GPL assets ✔

All 20 files in `examples/` are **git-tracked** (`git ls-files examples/`), so open-sourcing
the repository distributes them. They are also in the **source distribution**: `uv build`
produces a 41.2 MB `warlock-0.0.28.tar.gz` containing all 20, about 26 MB of it. CI runs
`uv build` on every push (`.github/workflows/windows-ci.yml`).

Among them:

- `examples/light_world.png` — described in `TODO.md` as *"the LTTP light world map —
  4110×5136, indexed, 204 colours"*. A rip of *The Legend of Zelda: A Link to the Past*.
- `examples/zelda_1.jpg` — likewise Nintendo material.
- `examples/*_base.obj`, `*_base.blend`, `*_spritesheet.png` — ULPC-derived, which `CLAUDE.md`
  and `TODO.md` both record as **CC-BY-SA/GPL**, with no attribution file anywhere in the tree.

The wheel is clean: `[tool.hatch.build.targets.wheel] packages = ["src/warlock"]` scopes it,
and the installer stages only `src/warlock`, `docs/manual` and three `vendor/` directories
(`installer/build.ps1:99-109`). Both were verified. The gap is that **there is no
`[tool.hatch.build.targets.sdist]` section at all**, so hatchling's sdist default sweeps in
everything not gitignored — and `.gitignore` does not mention `examples/`.

This is a takedown risk on the Nintendo files and a licence violation on the ULPC files. The
project already treats the exclusion as non-negotiable in prose; it was simply never enforced
against git or the sdist.

**Fix:** remove the Nintendo-derived files from the repository *and its history* before going
public — `TODO.md` already notes both are colour-destroyed and useful only as look references.
For the ULPC files, either remove them or add an `examples/ATTRIBUTION.md` naming the sources
and licence. Add an sdist exclude, and a test pinning it the way `tests/test_ux_todo_fixes.py`
pins the deleted plan filenames.

### B3 — The installer bundles GPL-3.0 `bpy` ✔

`installer/build.ps1:80` runs `uv export ... --extra rig`, and line 83 `uv pip sync`s the
result into the staged Python that Inno Setup packs into one `.exe`.
`.venv/Lib/site-packages/bpy-5.2.0.dist-info/METADATA` reads `License: GPL-3.0`.

The subprocess boundary is genuinely maintained — `src/warlock/pipelines/blender_worker.py` is
the only `import bpy` anywhere under `src/`, exactly as the invariant claims. That addresses
the *derivative work* question. It does not by itself address the separate *distribution*
question: one installer, one product, containing GPL-3.0 code next to code that has no licence
at all.

**Fix — a decision, three options:** (a) licence the project GPL-3.0-compatibly; (b) drop
`--extra rig` from the public installer and offer rigging as a separate opt-in the user
installs themselves; (c) take actual legal advice. Option (b) is cheapest and costs the
installer its rigging feature until resolved.

### B4 — Non-commercial model licences are never disclosed ✔

`grep -i license` over `src/warlock/models.py` and `docs/MODELS.md` returns nothing. No licence
text, acknowledgement or warning appears at download time in `fetch.py`, `fetch_worker.py`, or
the Settings → Models UI.

- **SDXL-Turbo** (`models.py:554-579`, promoted in `README.md:8` as "the fast option") is
  governed by Stability's non-commercial research community licence; commercial use requires a
  paid membership.
- **Playground v2.5** permits commercial use only under a 1M-monthly-user cap and requires
  shipping its licence plus a specific attribution string — an obligation currently unmet.
- SDXL 1.0, DreamShaper XL and Juggernaut XL are OpenRAIL++/OpenRAIL-M — use-restriction
  clauses, but commercially permissive. TRELLIS.2-4B, BiRefNet and FLUX.2 klein-4B are clean
  (MIT / MIT / Apache-2.0).

The users of a game-asset generator will sell the output. Telling them nothing is the posture
most likely to hurt someone who trusted the tool.

**Fix:** carry a `license` field on every `BaseModel` in `models.py`; show it in the model
picker and at download confirmation; add a licence column to `docs/MODELS.md`. A one-line
"output may not be used commercially" badge on Turbo is the minimum.

### B5 — The installer has never been run on real hardware ✔

`docs/manual/01-before-you-begin.md:52`: *"has not yet been run end to end on real hardware, so
the checkout above is the path to trust."* `TODO.md:308`: *"The installer — built, unverified
on real hardware."*

Its mechanism is well built — a per-user `%LOCALAPPDATA%` install at
`PrivilegesRequired=lowest`, a pinned CPython 3.13 and CUDA 12.8 runtime checked by
`installer/verify_runtime.py`, a real uninstaller that tells the user their `~/.warlock` data
survives. None of it has been executed.

**Fix:** build it; install on a clean Windows VM with no Python and no CUDA toolkit; run one
generation; uninstall; reinstall over the top. One afternoon, not optional.

### B6 — No third-party notice for the vendored binaries ✔

No `NOTICE` or `THIRD-PARTY` file exists. `installer/runtime-manifest.json` pins eleven
vendored binaries — `trellis-server.exe`, `trellis-cli.exe`, four ggml DLLs, three NVIDIA CUDA
redistributable DLLs, `gltfpack.exe`, `warlockc.dll` — and `installer/build.ps1:103-106` copies
the binaries with no accompanying licence text. MIT requires the notice travel with the binary;
NVIDIA's redistributable EULA has its own terms.

The project already does this correctly elsewhere, which marks the gap as oversight rather than
position: `LICENSE-lucide.txt` and `LICENSE-inter.txt` ship beside the fonts, and the vendored
BiRefNet carries a pinned commit, SHA-256s, a documented diff and its own `ATTRIBUTION.md`.

**Fix:** one `THIRD-PARTY-NOTICES.md`, staged into the installer beside the binaries.

---

## 2. Findings by domain

Severity: **Blocker** · **Major** · **Minor** · **Polish**.
✔ = independently verified against source during this audit.

### 2.1 Tests, lint and CI — *ship*

The strongest domain. Full numbers in Appendix A.

| Finding | Sev | Notes |
|---|---|---|
| CI never tests the stated Python 3.12 floor ✔ | Major | `pyproject.toml:5` declares `requires-python = ">=3.12"`, but `.github/workflows/windows-ci.yml:19-22` installs 3.13 only and always syncs `--extra rig` (which needs 3.13). The documented floor, and the "rig unavailable" degradation path, are never exercised. **Fix:** add a 3.12 leg without `--extra rig`. |
| CI has no gpu lane, no perf lane, no wheel-install smoke test, no installer build ✔ | Major | The workflow is four steps: preflight, `pytest -q`, `uv build`. The wheel that "builds" is never proven to import or run, and the GPU-critical paths CLAUDE.md itself names (model loading, VRAM accounting, conditioning) are gated only by developer discipline. |
| `assets/44593039ccee/` fixture is absent, so real-rigged-GLB tests silently skip | Minor | `tests/test_gltf_loader.py:25-31`, `tests/test_viewer_gl.py:238-243`. Worth deciding whether that asset should be committed; otherwise this coverage is permanently dark. |
| 16 skips, no hidden holes of consequence | — | ~15 `needs_dll` (the DLL is present here, so they run), 2 Windows-only, a few `needs_real`. |

The xdist configuration is measured rather than guessed — `pyproject.toml:171-199` documents
why workers are pinned at 8 (`-n auto`'s 24 workers OOM'd twice in twelve runs) and why
`--dist loadfile` is required (one imgui context per file).

### 2.2 Licensing and distribution — *do not ship*

Covered in full as B1–B4 and B6 above. One further item:

| Finding | Sev | Notes |
|---|---|---|
| Unsigned installer ✔ | Major | No `SignTool=` in `installer/warlock.iss`, no signing step in `build.ps1`. Every public install trips SmartScreen's "unrecognized app" wall — for a free indie tool distributed outside a store, this is the single largest install-abandonment cause. An OV code-signing certificate is a few hundred dollars a year; SmartScreen reputation then accrues over time. |
| No `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, or issue templates ✔ | Minor | `.github/` holds only the CI workflow. These become genuinely needed the day the repo goes public — `SECURITY.md` especially, since there is currently no stated route to report a vulnerability privately. |

### 2.3 Documentation — *ship with fixes*

Unusually rigorous. Nearly every load-bearing README claim was checked against code and held:
eleven modes (`studio/modes.py:35-77`), nineteen blend modes (`inker/composite.py:40-68`),
seven skeleton templates, `DEFAULT_BASE_MODEL = "sdxl_cfg"` at 30 steps / CFG 7.0
(`models.py:39`, `646-661`), `YAW_CHOICES = (4, 8, 16)`, `JOURNAL_SECONDS = 120.0`,
`TILE_SIZES = (16, 32, 48, 64)`. Chapter numbering, the shortcut table and help-target coverage
are all test-gated in both directions. Manual and root screenshots were all regenerated for
0.0.28.

| Finding | Sev | Notes |
|---|---|---|
| `CHANGELOG.md` and `TODO.md` read as instability signals to a stranger | Major | `CHANGELOG.md:9-20` is written for the author — internal review scores, crash tracebacks, `TODO.md:389-403` reproduces a raw `RuntimeError: host memory is 92% committed`. `panes/landing.py:704-717` shows that text **verbatim** to any user who clicks "All release notes…". The candour is an asset for contributors and a liability on the front page. **Fix:** a short user-facing preface, or truncate `_news_popup` to leads for all releases. |
| GitHub repo description is stale ✔ | Minor | Still reads "SDXL-Turbo → TRELLIS.2"; the default has been full-CFG SDXL for some time. First thing a visitor reads. |
| README says "39 chapters", index lists 38 | Minor | `README.md:190` vs `docs/manual/00-index.md`. |
| Manual is essentially unillustrated | Minor | 4 of 38 chapters embed a screenshot; Clay, Poser, Plotter, Packwright, Troupe, Review and Settings are text-only. Thin for a visual tool. |
| No freshness gate on screenshots | Minor | `scripts/screenshot_modes.py` regenerates on request; nothing asserts the images match the current version. They are current today; nothing stops the next release drifting. |
| Guided tour covers 2 of 11 modes | Minor | `studio/tour/scripts.py:1-13` documents this as deliberate scope (`FIRST_HOUR`, `INKER_BASICS`, both chosen to need no GPU). Still an onboarding gap for six workspaces. |
| Generation time is never stated before the download | Polish | "Roughly two minutes of GPU per attempt" appears in `docs/manual/02` and `23`, but nowhere the user sees before committing to ~23 GB. |

### 2.4 First run and onboarding — *ship with fixes*

Better than expected. The first-run modal (`studio/panes/first_run.py:33-218`) shows GPU name
and VRAM, three readiness verdicts, the exact required downloads with combined size, and a
disk-space refusal — before the user touches anything. The download path is genuinely robust:
`fetch_worker.py` sets `HF_HUB_OFFLINE=0` only in its own child environment, stages into a
sibling `.part` directory, verifies hub-recorded digests before publishing, and `present()`
treats a partial directory as absent, so a killed download cannot masquerade as installed.

| Finding | Sev | Notes |
|---|---|---|
| README leads with three manual `hf download` commands | Major | `README.md:115-141`. A stranger following Setup top-to-bottom pastes terminal commands — getting `--include`/`--exclude` ordering exactly right, per `models.py:295-300` — before reaching "can also be fetched from inside the app" at line 141. **Fix:** lead with the in-app path; keep the CLI as the headless fallback it already is. |
| No-CUDA machines pass VRAM admission silently | Minor | `vram.py:571-578` returns `ok=True, fatal=False` with "no CUDA device detected; VRAM admission control is off", so an AMD/Intel box reads amber at worst in doctor. The real refusal arrives later as `RuntimeError("trellis-server exited during startup (code N)")` (`trellis.py:273-275`) — not layperson wording. Mitigated by the first-run modal saying plainly there is no CPU fallback. **Fix:** treat "no CUDA device" as fatal for the 3D path in doctor. |
| `pythonw.exe` entry point hides pre-logging startup crashes | Minor | `installer/warlock.iss:43`. No console; a crash before `_setup_logging` attaches leaves a silently-closed window. The population most likely to hit an unusual driver state is exactly the first-run one. |
| ~23 GB before the first asset | — | TRELLIS.2 GGUF 16.1 GB (`models.py:498-524`) + SDXL 1.0 fp16 7.0 GB (`models.py:544-550`). Correctly surfaced up front; noted here because it is the single biggest adoption filter and no amount of polish changes it. |

### 2.5 Core pipeline and job system — *ship with fixes*

Exceptionally hardened. VRAM handoff, subprocess lifecycle, sqlite locking and the t2i
stdin-reader invariant are all correctly implemented and match their documentation. One real
defect:

| Finding | Sev | Notes |
|---|---|---|
| Cancelling a pixel-sheet restyle deletes a *previous, successful* restyle's output ✔ | Major | `rigging.sheet_pixel_path` / `sheet_pixel_png_path` are pure functions of `(job_dir, sheet_id)` — `<id>.pixel.json` / `<id>.pixel.png`. `service/sheets.py:283` deliberately allows restyling the same sheet repeatedly with the same `sheet_id`, so every restyle targets the same two files. The cancel branch at `_q_jobs.py:450-458` deletes that pair unconditionally. Restyle a sheet, keep the result, start a second restyle with a different seed, cancel it — the first result is destroyed. Its sibling branches all avoid exactly this and say why in comments (`rig`: "a cancelled re-rig must not destroy the rig it corrects"; `retexture` writes to a temp; `sprite_synthesis` mints a fresh `draft_id`). `pixel_sheet` does neither. Additionally `_cancel.commit()` appears only at `_q_rig.py:167`, so even a first restyle has a publish-then-cancel window. `tests/test_sheet.py:1138` only cancels the *first* restyle, so it cannot see this. Loss is regenerable, hence Major not Blocker. **Fix:** stage through a temp name as `retexture` does, delete only the temp, and `commit()` after the successful replace. |
| `_unload_under_lease` self-deadlocks at shutdown under `WARLOCK_T2I_IN_PROCESS=1` | Minor | `queue.py:853-871` holds `leases.MODELS.maintain()` then calls `pipe.unload()`, which under the debug flag takes `leases.MODELS.use()` on the same thread; `use()` has no timeout. Debug-only path, not reachable in the shipped default. Worth a note on the flag. |

Two plausible-looking findings were investigated and **refuted**: `service.troupe.check_troupe`
appears to skip a weights check, but the generic door-level `check_weights` runs afterward with
`base_model` already pinned (`_jobs_create.py:486`, `502`); and `service.downloads.download`'s
un-merged `stderr` pipe is drained by a dedicated reader thread from the start
(`downloads.py:880-895`).

### 2.6 Application shell and UI — *ship with fixes*

| Finding | Sev | Notes |
|---|---|---|
| Clay's outliner drag-reorder bypasses the `tab.saving` guard ✔ | Major | `panes/clay_outliner.py:137` gates `_reorder` only on `filtered`; `panes/inker_timeline.py:1222` refuses outright when the tab is busy, with a comment stating exactly why: *"begin_disabled does not stop a drag-drop source from registering."* `_body()` wrapping the outliner in `begin_disabled(tab.saving)` therefore does not protect it. A drag completed during an autosave or export encode mutates `doc.objects` mid-write. Same bug class the codebase already paid to fix once, in the one place the fix was not copied. **Fix:** mirror Inker's early return. |
| The tour's scrim darkens a real modal opened on top of it | Major | `panes/tour.py:206-258` draws the veil on `get_foreground_draw_list()`, which composites above every window unconditionally — that is why the tour card needs its own hole. `App._modal_open` deliberately excludes the tour so the app stays live, but nothing stops a confirm or prompt opening during a tour, and the ring may then circle a control the modal covers. **Fix:** give confirm/prompt/matte rects the same hole treatment, or suspend the tour draw while a modal is open. |
| Persisted `window_size` is trusted with no validation ✔ | Major | `main.py:749-756` feeds `settings.get("window_size")` straight to `pygame.display.set_mode` with no shape check and no clamp against `MIN_SIZE` — that clamp exists only on the live resize path, after the window already exists. `Settings.load()` discards the *whole* file on a version mismatch, but a single malformed key in an otherwise-valid file sails through. If `set_mode` raises, `run()`'s handler reports the crash gracefully but never rewrites the key, so it recurs every launch with no in-app recovery for a non-developer. `_ui_scale()` at `main.py:142-151` already states the rule this setting skips: *"A junk value must not brick the window."* |
| Esc-chain help text omits the tour step | Polish | `docs/manual/37-shortcuts.md` and the `Ctrl+/` sheet describe manual → profile sheet → mode; the real order (`main.py:2429-2471`) is manual → running tour → profile sheet → mode. Correct behaviour, undersold documentation. |
| Layout-editor selection outline does not scale thickness with DPI | Polish | `layout_edit.py:127-129` passes only `rounding`; every other highlight rect in the codebase passes an `sp()`-scaled thickness. A 1px hairline at 200% scale. |

### 2.7 Headless editor packages — *ship with fixes*

These hold the user's actual creative work, so data loss dominates.

| Finding | Sev | Notes |
|---|---|---|
| PNG save-in-place is not atomic ✔ | Major | `inker_mode.py:2241` is `path.write_bytes(doc.png_bytes())`. `WRITABLE_SUFFIXES = (".ora", ".png")` at line 82 deliberately allows saving back over an opened `.png`. Every other writer in the app stages to `.tmp` and `os.replace`s — `ora.py:924-967`, `aseout.py:1396-1399`, `clay_mode.py:346-348`, `plotter_io.py:355-359`, `packwright_io.py:167-171`. A crash or full disk mid-write truncates the user's only copy. **Fix:** route this branch through the same helper. |
| `.aseprite` import has three unbounded `zlib.decompress` ✔ | Major | `inker/asein.py:554`, `648`, `670`. Every sibling parser bounds the same operation deliberately: `ora.py:154` sums declared sizes against `MAX_DECOMPRESSED_BYTES = 1 << 30`, `tmx.py:253` uses `decompressobj`. The size check at `asein.py:676` runs *after* decompression completes. A few-MB crafted or corrupt `.aseprite` — the kind of file people download from asset sites — can inflate to gigabytes and take the process down. Also a security finding (2.9). **Fix:** `decompressobj().decompress(raw, bound)` with the bound derived from the already-known declared dimensions. |
| Export writers are non-atomic | Minor | `inker_mode.py:1184` (flattened PNG) and `4084-4085` (tileset `.tsx` + `.png`) share the shape above. Narrower risk — the live document is untouched — but re-exporting over an existing file can still truncate it. |
| `write_ora` leaves its `.tmp` behind on a failed encode | Polish | `ora.py:924-967` has no `try/finally`, where `plotter_io`, `packwright_io` and `journal` all unlink theirs. Not data loss — `replace` only runs on success — just a stray dotfile. |

Undo is uid-addressed throughout, as the invariant requires; the few index-based edits (palette
slots, add-only tileset lists) are index-stable by construction and documented as such.

### 2.8 Model selection and output quality — *ship with fixes*

This is where the product's credibility is decided, and it is the weakest evidentiary area.

| Finding | Sev | Notes |
|---|---|---|
| No positive mesh-quality evidence for the shipped default | Major | The only completed graded run is **0 of 20 usable** (`docs/measurements/2026-08-13-tier-qualification.md`, "Every mesh was rejected") — on deliberately hard subjects, using `playground` rather than the shipped `sdxl_cfg`. Root-caused to subject difficulty plus a tightened instrument, not a regression (`2026-08-13-mesh-regression-check.md`). The superseded binary-era number was 19/41 (46%) accept (`2026-08-09-rebaseline.md`). No graded run has ever targeted the default. **Fix:** one GPU afternoon on a representative corpus at `sdxl_cfg`. Until then the tool ships with no evidence for its headline claim. |
| Tile-sheet art direction ships knowingly unresolved | Major | `docs/measurements/2026-08-18-tile-sheet-grid.md` says so itself: the mechanism works, the output is "one continuous brick wall" or "near-identical grey mush". **Fix:** mark experimental, or ship the "N materials, one grid" option the document identifies as the answer. |
| Two models carry zero measurement and sit as peers of the default | Major | `juggernaut` and `dreamshaper` have no hits anywhere in `docs/measurements/`; the picker (`settings_2d.py:957`) offers all eleven flatly, 6.9 GB each. |
| `sdxl_cfg_pag` is offered as an equal despite losing its own bench | Minor | `2026-08-17-reference-source-bench.md`: the control won 55 of 80 paired units, and PAG cost +34% sampling time. |
| "Seamless" verdicts are advisory on the shipped checkpoint | Minor | `2026-08-09-seam-threshold-cfg.md`: "the edge-energy ratio does not separate seamless tiles from seamed ones on a CFG base." `inspector.py:812` states the verdict confidently without the caveat. |

**Honest numbers, per path:** reference image is the healthiest stage at **93.75% composition-gate
pass** (`2026-08-17-reference-source-bench.md`). Mesh hole rate: 15 of 37 baseline meshes
exceeded the 0.07 retry threshold (`2026-08-04-hole-rate-baseline.md`). Tile sheets: mechanism
proven, art direction not. Troupe: never quality-tested end to end.

**Recommended shipping roster:** keep `sdxl_cfg` (default), `turbo` (labelled *draft*, and see
B4), `sdxl`, `pixel`. Move `sdxl_cfg_pag`, `lightning`, `playground` and both `flux_klein`
variants behind an Advanced toggle. Drop `juggernaut` and `dreamshaper` from the default picker
until either is measured.

### 2.9 Security, privacy and untrusted input — *ship with fixes*

**The offline claim is true.** `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` are set at
`src/warlock/__init__.py:9-10` before anything imports. A full-tree sweep for `httpx`,
`requests`, `urllib`, `socket`, `webbrowser`, telemetry and crash reporting found exactly three
call sites: the trellis client hardcoded to `http://127.0.0.1:{port}` (`trellis.py:143,156`),
the `fetch_worker` child that flips the flag in its own environment only, and `errors.py`
importing `httpx` solely to catch a type. No update checker, no analytics, no crash reporter,
no hardcoded external domain. `trust_remote_code` is **not** used — BiRefNet is vendored and
loaded from readable local code (`matting.py:237-252`).

| Finding | Sev | Notes |
|---|---|---|
| Unbounded decompression in `.aseprite` import ✔ | Major | Same as 2.7. The only parser in the codebase whose siblings all guard this and it does not. |
| Zip "claimed size" ceiling is bypassable | Major | `ora.py:1731`, `clay/serialize.py:459`, `packwright/wpack.py:166`, `plotter/wmap.py:1145` all sum attacker-controlled `info.file_size` before reading, but `zipfile`'s internal read can allocate up to ~2 GB transiently before truncating to the declared size. Real but bounded per call. **Fix:** read via `zf.open()` with your own byte counter. |
| Infinite-map `<chunk>` dimensions are uncapped | Major | `plotter/tmx.py:744-751` reads chunk `width`/`height` with no ceiling and feeds them into the decode bound, defeating the otherwise-correct per-layer limit. Fixed-size maps are properly capped at `MAX_DIMENSION = 4096`. |
| `.tmx`/`.tsx` external refs allow unlimited `..` traversal | Minor | `plotter_io.py:91-115` deliberately permits `../` to match Tiled's real folder layout, refusing absolute and UNC paths. The comment's claim that traversal stays rooted at the opened file is not strictly true. Impact is low — same-user read only, no exfiltration channel in an offline app. Documented tradeoff. |
| Windows reserved device names not screened in export templates | Polish | `inker/sheetout.py:103-110`. A tag named `CON` produces a filename Windows refuses. |

No pickle, marshal, eval, exec or unsafe YAML anywhere in the untrusted-parsing path;
`np.load(..., allow_pickle=False)` is explicit. `db.py` is fully parameterised. All subprocesses
use list-argv `Popen`, resolve via absolute paths rather than PATH search, and call
`winjob.assign` immediately after spawn. Nothing personal is baked into exported assets —
`provenance.py` records library versions and model fingerprints only.

### 2.10 Performance and platform — *ship with fixes*

| Finding | Sev | Notes |
|---|---|---|
| Startup always pays a ~1.6 s torch import | Major | `doctor._cuda_check` correctly defers torch via `probe_slow=False`, but `_vram_check` in the same `run_checks()` calls `vram.probe()` (`vram.py:502-512`), which imports it regardless — measured at 1.57 s. Defeats the documented fix on every cold start with the recommended install. |
| Default window is never clamped to the desktop | Major | `main.py:63` `DEFAULT_SIZE = (1600, 950)`, scaled by DPI but never checked against `pygame.display.get_desktop_sizes()`. Does not fit 1366×768 at all; at 125% scaling — Windows' recommendation for many 1080p laptops — the requested 2000×1187 does not fit a 1920×1080 panel either. |
| Undo memory cliff on large Inker documents | Major | Per-stroke undo is dirty-rect patches, but rotate/scale/crop/canvas-resize push a full-layer snapshot: ~1.9 GiB for one step at 4096² × 30 layers. `_evict` only enforces the 192 MiB budget once depth exceeds `UNDO_MIN_DEPTH = 8`, so ~15 GiB can accumulate before eviction engages. |
| One distant Plotter click allocates the whole window | Major | `_map_geometry.py:349-383` grows the dense uint32 window to reach the painted cell, capped at 4096 per side — 64 MiB per tile layer, instantly, for one click. Bounded, but surprising. |
| RAM requirement is unstated and understated | Major | README gives OS, GPU and disk but no RAM figure. The project's own invariant record documents admission control refusing jobs at 96% host commit on a **63.5 GB** machine with 24 GB physically free, because WDDM charges trellis's ~16 GiB device allocation against host commit. **32 GB should be stated.** |
| Perf lane covers a narrow slice | Minor | Only 5 files carry `@pytest.mark.perf`. No wall-clock budget exists for Inker compositing or flatten, Plotter render, Packwright packing, Library refresh, Troupe's 256-cell render, or journal writes. |
| `httpx`'s CLI import taxes every start | Polish | `errors.py:14`'s bare `import httpx` pulls `typer`/`rich`/`click`/`pygments` transitively — ~110 ms for a CLI the app never uses. |

The platform story is otherwise honest: Windows + NVIDIA only, stated plainly in the first
paragraph of both README and manual, with no false cross-platform promise and no CPU-fallback
trap. The frame loop throttles to 12 fps when idle, so it does not pin a core on the Home
screen.

---

## 3. What is genuinely good

Worth stating plainly, because the list above is long and the balance matters.

- **The test suite is real.** 12,041 tests, 68 seconds, zero failures, and the GPU lane green
  in 91 seconds. Not smoke tests — `tests/` is 158k lines against 185k lines of source.
- **The invariants are enforced, not aspirational.** Import pins walk the full AST so a lazy
  import cannot sneak past. A scan test enforces the `winjob` rule. Chapter numbering is gated
  in both directions. The compat matrix is parsed as data.
- **Atomicity discipline is near-universal.** Five of six document writers stage and replace;
  the journal writes its sidecar last as a completion gate; `publish.py` and `instance.py`
  survive a hard kill mid-operation.
- **Untrusted input is taken seriously.** DTD refusal by name, `MAX_IMAGE_PIXELS`, `MAX_NODES`,
  `MAX_TRIANGLES`, `MAX_TEXTURE_PIXELS`, `allow_pickle=False`, parameterised SQL throughout,
  and a hand-rolled GLB loader with explicit ceilings rather than trusting trimesh.
- **The offline promise holds** under a full-tree audit, with the one documented exception
  behaving exactly as documented.
- **The download path is better than most commercial installers'** — staged `.part` directory,
  digest verification before publish, partial directories treated as absent.
- **The code explains itself.** Comments narrate the incident that motivated the guard. Several
  audits went looking for a classic bug and found the fix already in place with the failure
  mode spelled out.
- **The project is honest with itself.** `docs/measurements/` records results that make the
  product look bad. `docs/manual/11` separates "Proven / Untested / Provisional / Not built"
  for its own flagship feature. That is rare and it is worth preserving — the fix for the
  changelog tone is a user-facing summary, not less candour.

---

## 4. Recommended v1.0 scope

| Mode | Maturity | v1.0 |
|---|---|---|
| Home | Mature | Ship |
| Library | Mature | Ship |
| Create | Mature | Ship |
| Inker | Mature — 113 test files, the deepest-tested subsystem here | Ship |
| Review | Mature | Ship |
| Settings | Mature | Ship |
| Plotter | Usable — 30 test files; Tiled interop is self-certified only | Ship |
| Packwright | Usable — 17 test files, capped and perf-checked | Ship |
| Clay | Usable — thinner coverage relative to surface | Ship |
| Poser | Usable — inherits Troupe's caveat for clip authoring | Ship |
| **Troupe** | **Experimental** | **Ship marked experimental** |

**Nothing needs cutting.** No dead buttons, no reachable `NotImplementedError`, no placeholder
panes were found. The one scoping decision is Troupe: it is code-complete and a user really can
get a rendered sheet, but three of its own phases are unstarted, its 22 keyframes are
provisional, and its palette claim rests on a textured base mesh that does not exist. The
manual is candid about all of this; **the app is not**. `rail.py:363` passes no tooltip, and
`_item` suppresses the tooltip entirely once the label is legible (`rail.py:213-214`) — so
hovering "Troupe" in the expanded rail says nothing at all, and says only "Troupe" collapsed.

Two cheap product fixes worth doing regardless:

- **An "Experimental" chip on the Troupe rail item**, reusing the manual's own wording.
- **A one-line purpose tooltip on every rail item.** "Inker", "Clay", "Poser", "Troupe",
  "Plotter", "Packwright" are invented names and the rail is the primary navigation; a new user
  currently gets a word and an icon. `rail.py:213` already accepts a `tooltip` argument that no
  call site passes — this is a small table beside `modes.MODES`, not a feature.

Also: Troupe is humanoid-only (`service/troupe.py:74-78`) while Poser advertises seven
skeletons. Grey the "Send to Troupe" action on non-humanoid meshes rather than letting the
runtime refusal be the first the user hears of it.

---

## 5. Sequenced plan

**Wave 1 — the legal gate (days, mostly decisions).** Nothing ships until these close.

1. Choose and add a licence (B1) — decide B3 at the same time, since they interact.
2. Purge the Nintendo-derived files from `examples/` and from git history; attribute or remove
   the ULPC assets; add an sdist exclude and a test pinning it (B2).
3. Decide `bpy`: GPL-compatible licence, or drop `--extra rig` from the public installer (B3).
4. Add `THIRD-PARTY-NOTICES.md` and stage it into the installer (B6).
5. Add per-model `license` metadata, surface it in the picker and at download (B4).
6. Add `SECURITY.md`, `CONTRIBUTING.md`, issue templates. Fix the stale GitHub description.

**Wave 2 — prove the shipping path (one day).**

7. Build the installer; install on a clean Windows VM; generate one asset; uninstall; reinstall
   over the top (B5).
8. Buy and wire a code-signing certificate.
9. Add a Python 3.12 CI leg without `--extra rig`, and a wheel-install smoke test.

**Wave 3 — the correctness fixes (one to two days).**

10. Pixel-sheet cancel: stage through a temp, delete only the temp, `commit()` after replace.
11. Bound the three `zlib.decompress` calls in `asein.py`; cap infinite-map chunk dimensions;
    replace the claimed-size zip ceiling with a counted read.
12. Make PNG save-in-place atomic; do the same for the two export writers.
13. Add the `tab.saving` guard to Clay's outliner reorder.
14. Validate and clamp persisted `window_size`; clamp the default window to the desktop.

**Wave 4 — the first-hour experience (one day).**

15. Rewrite README Setup to lead with the in-app downloader; state the RAM requirement (32 GB)
    and expected generation time up front.
16. Add the Troupe experimental marker and the rail tooltips.
17. Give the changelog a user-facing preface, or truncate `_news_popup` to leads.
18. Make doctor treat "no CUDA device" as fatal for the 3D path.

**Wave 5 — the evidence (one GPU afternoon, plus judgement).**

19. Run one graded mesh corpus at the shipped `sdxl_cfg` default. This is the number the
    product will be judged on and it currently does not exist.
20. Decide the model roster: hide the unmeasured two, label Turbo as draft.
21. Decide whether tile sheets ship, ship experimental, or wait for the "N materials" fix.

**Deferred to 1.1:** Troupe phases 6/7/8, the real-Aseprite and real-Tiled fixture passes,
authored keyframes, a textured base mesh, the `plotter-wave-2` branch, undo-memory bounding,
the lazy torch import, guided-tour coverage for the remaining six workspaces.

---

## Appendix A — Verification output (2026-08-24)

```
$ uv run pytest -q
12041 passed, 16 skipped, 45 warnings in 67.66s (0:01:07)

$ uv run pytest -m gpu -n 0 -q
26 passed, 12076 deselected, 62 warnings in 90.85s (0:01:30)

$ uv run ruff check .
All checks passed!

$ uv run python scripts/preflight.py --fast
ok    version lockstep  (0.0.28)
ok    ruff
skip  test suite (--fast)
preflight passed

$ uv build
Successfully built dist\warlock-0.0.28.tar.gz          41,202,463 bytes  (1165 members)
Successfully built dist\warlock-0.0.28-py3-none-any.whl  7,995,225 bytes  (471 members)
```

Wheel contents: `warlock/studio` 275 · `warlock/manual` 43 · `warlock/` 42 · `warlock/pipelines`
35 · `warlock/service` 28 · `warlock/bench` 22 · `warlock/templates` 18 · `warlock/assets` 4.
`examples/` is **absent from the wheel** and **present in the sdist** (20 files, ~26 MB) — see B2.

The only warnings are third-party deprecations (`torch.jit.script`, a diffusers `__array__`
copy-keyword notice). `dist/` is gitignored; nothing else in the tree was modified.

## Appendix B — Licence inventory

| Component | Source | Licence | Wheel | Installer | Attribution |
|---|---|---|---|---|---|
| Warlock Studio | this repo | **none declared** | — | — | — |
| trellis-server.exe, ggml DLLs | pwilkin/trellis.cpp | MIT | No | Yes | **No** |
| cublas / cudart DLLs | NVIDIA | CUDA EULA | No | Yes | **No** |
| gltfpack.exe | meshoptimizer | MIT | No | Yes | **No** |
| lucide.ttf | lucide-icons | ISC | Yes | Yes | Yes |
| Inter (PUA-stripped) | rsms/inter | OFL 1.1 | Yes | Yes | Yes |
| birefnet modelling code | ZhengPeng7/BiRefNet | MIT | Yes | Yes | Yes |
| `bpy` 5.2.0 | Blender Foundation | **GPL-3.0** | No | **Yes** | **No** |
| `pygame-ce` | pygame community | LGPL-2.1 | Yes | Yes | standard |
| manifold3d / opencv / imgui-bundle / trimesh / zstandard | — | Apache-2.0 / Apache-2.0 / MIT / MIT / BSD-3 | Yes | Yes | standard |
| SDXL 1.0 weights | Stability AI | OpenRAIL++-M | user fetch | — | **No** |
| SDXL-Turbo weights | Stability AI | **non-commercial** | user fetch | — | **No** |
| Playground v2.5 | Playground | Community (1M MAU) | user fetch | — | **No** |
| Juggernaut XL v9 / DreamShaper XL | RunDiffusion / Lykon | OpenRAIL-M / OpenRAIL++ | user fetch | — | **No** |
| FLUX.2 klein(-base) 4B | Black Forest Labs | Apache-2.0 | user fetch | — | n/a |
| TRELLIS.2-4B / BiRefNet weights | Microsoft / ZhengPeng7 | MIT | user fetch | — | n/a |
| `examples/` (ULPC + Nintendo-derived) | LPC/OpenGameArt, Nintendo | CC-BY-SA / GPL / **unlicensed** | No | No | **No — and git-tracked** |

## Appendix C — Format round-trip

| Format | Read | Write | Known losses | Test-enforced? |
|---|---|---|---|---|
| `.ora` | yes | yes (atomic) | group `composite-op`, ICC profiles, foreign per-layer attrs — inbound only | prose only |
| `.aseprite` | yes | yes (atomic) | cel opacity/z-index, user data, per-frame palettes, colour profile, group opacity — all documented | fixed-point corpus; **never opened in real Aseprite** |
| `.tmx` / `.tsx` | yes | yes (atomic) | legacy constructs refused by name; no silent drops | yes — `tests/plotter/test_compat_matrix.py` parses `COMPAT.md` as data |
| `.wmap` | yes | yes (atomic) | none claimed; forward versions refused | yes |
| `.wblk` | yes | yes (atomic) | none claimed; forward versions refused | yes |
| `.png` (document) | yes | **yes, non-atomic** | flattened by definition | correctness yes, atomicity no |

## Appendix D — Scale limits and the honest system requirement

| Subsystem | Where it stops being viable |
|---|---|
| Inker 4096² × 30 layers | strokes fine; one rotate/crop/resize pins ~1.9 GiB of undo |
| Plotter infinite map | capped at 4096²/layer (64 MiB), but one distant click allocates it at once |
| Packwright | fine — `MAX_SPRITES = 4096`, packing prune already O(n²) |
| Clay | vectorised and perf-tested to ~100k faces |
| Library 5,000 jobs | paged sqlite stays cheap; the periodic refresh runs on the frame thread |
| Startup | ~1.6 s torch import always paid |

**What the download page should say**, derived from the code rather than from the README:

- **OS:** Windows 10/11, 64-bit. No macOS, no Linux.
- **GPU:** NVIDIA with CUDA. **16 GB VRAM** for 3D reconstruction (`vram.py:55`
  `TRELLIS_GIB = 16.0`). No CPU fallback exists. Recommended: RTX 4080/5080-class or better.
- **RAM: 32 GB.** Currently unstated. Windows charges the GPU allocation against host commit,
  and the project's own logs show job refusals on a 63.5 GB machine.
- **Disk:** ~23 GB of model weights before the first asset, plus 35–50 MB per generated 3D job
  with no automatic age-out (manual prune only).
- **Display:** 1920×1080 or larger at 100% scaling — the default 1600×950 window is not clamped
  to the desktop and does not fit smaller or heavily-scaled panels.
