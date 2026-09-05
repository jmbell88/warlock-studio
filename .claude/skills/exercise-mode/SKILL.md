---
name: exercise-mode
description: Drive every control in one Warlock mode through the app's real input path, photograph each press, and report which controls are dead, crashing, greyed with no reason, or visibly doing the wrong thing. Use when asked to exercise, audit, or sanity-check a mode's controls (/exercise-mode inker).
---

# Exercise one mode's controls

`tests/test_studio_smoke.py` asserts that every pane *builds*. It asserts
nothing about whether a control is wired to anything. `scripts/screenshot_modes.py`
photographs a mode *at rest*. Nothing in the repo presses a button.

So the defect nobody catches is a control that draws correctly and does
nothing: clipped past its content region, disabled with no reason, wired to a
handler that was renamed, or reaching one that raises into the frame. This
skill runs the driver that presses them and judges the pictures it produces.

**This pass reports. It does not fix.** Fixing is a separate ask.

## Before you start

- **It needs a real window.** The driver draws on screen and reads the
  framebuffer back. It cannot run headless, and it cannot run over SSH.
- **It must not run while `pytest` is running.** Several tests read module
  source, and `src/` must not move underneath them.
- It takes minutes, not seconds: every control gets several frames of settle
  and a PNG.

## 1. Resolve the mode

Read `warlock.studio.modes.KEYS` and check the argument against it. Refuse an
unknown one and print the list — do not guess at a near-match.

```
uv run python -c "from warlock.studio import modes; print(' '.join(modes.KEYS))"
```

## 2. Run it against a throwaway home

All three variables, every time. `WARLOCK_DATA_DIR` alone does **not** move the
sqlite store, and a run that seeds jobs into the user's real library is a
library the user has to clean up.

```powershell
$scratch = "<scratchpad>/exercise-<mode>"
$home_   = "$scratch/home"
New-Item -ItemType Directory -Force $home_ | Out-Null
$env:WARLOCK_HOME     = $home_
$env:WARLOCK_DATA_DIR = "$home_/data"
$env:WARLOCK_DB       = "$home_/warlock.db"
$env:WARLOCK_UI_PROBE = "1"
uv run python scripts/exercise_mode.py --mode <mode> --out "$scratch/out"
```

`WARLOCK_UI_PROBE=1` is not optional: without it the census is empty and the
driver refuses to start rather than reporting a clean run over nothing.

Two things the run does to itself, both worth stating in the report:

- The **doctor banner is dismissed** before the baseline. It is a property of
  the throwaway home (no weights downloaded into it), not of the mode, and it
  is a full-width strip that would make the baseline image disagree with every
  later one by its own height. Its two buttons go uncovered.
- A control that **raises wedges imgui**: the exception unwinds past
  `imgui.render`, so the frame is never ended and the id stack is unbalanced,
  and nothing after it can draw. The driver writes the manifest naming the
  culprit and stops, printing `ABORTED at <key>`. Re-run with
  `--skip "<key prefix>"` to cover the rest of the mode while the defect is
  open, and report **both** the crash and what the skip left out.

If the run hangs instead, a control opened an OS file dialog that `REFUSED` in
`scripts/exercise_mode.py` does not name. Kill it, add the label, say so in the
report.

## 3. Read the manifest, then the pictures

`<out>/manifest.json` has one record per control, plus a `palette` list and a
`raw_imgui_controls` count. Every record carries a machine-assigned `verdict`.

Look at the PNG for **every** record whose verdict is in `always_look`
(`raised`, `inert`, `toast-error`, `disabled-no-reason`, `hard-reset`), plus
`clipped`, plus a sample of the rest — and always at `00-baseline.png` first,
because every judgement is a comparison against it.

| verdict | means | look? |
| --- | --- | --- |
| `raised` | an exception escaped the frame | always |
| `inert` | no task, no toast, no state delta, no pixel delta | always — the prime suspect |
| `toast-error` | the press produced an error-level notice | always |
| `disabled-no-reason` | drawn greyed with an empty `why`/`reason` | always |
| `hard-reset` | could not be undone back to the baseline | always |
| `clipped` | imgui clipped it away; nobody can click it | always |
| `disabled` | greyed, with an explanation | skim |
| `submitted` | reached a handler through `TaskRunner.submit` | sample |
| `state-changed` | the digest moved | sample |
| `pixels-changed` | only the frame differs | sample |
| `skipped` | named on `--skip` | no — but say what was skipped |
| `refused` | on the do-not-press list | no |

## 4. Judge each picture against the control's own label

This is the half a script cannot do. For each shot you open, ask: does the
screen now show what a control called **"Bucket"**, **"Add layer"**,
**"Export sheet"** should have done? A press that changed pixels is not
thereby correct — a button that opens the wrong panel is `state-changed` and
still a defect.

Note especially:
- a control whose effect landed in a *different* pane than its label implies;
- a toast whose wording does not match the button pressed;
- an `inert` control that a look at the picture shows *did* work (the digest
  missed it — say so, it is a gap in the harness, not in the app). The digest
  covers mode, stage, selection, tool, undo serial, open documents, toast count,
  dirty, and the five overlays; anything outside that reads as `inert`.

## 5. Write the report

Four buckets and a coverage line:

- **Crashed** — verdict `raised`, with the top frame of each traceback.
- **Visibly wrong** — the screen disagrees with the label.
- **Suspected dead** — `inert` and `clipped`, each with what you expected to see.
- **Confirmed working** — a count, plus anything notable.

Then the coverage line, stating all three numbers plainly:

> Reached N of M controls the census saw; K left behind a frontier the walk did
> not open (round bound hit / not hit); R controls call imgui directly and are
> not probe-visible at all.

`frontier_left` and `raw_imgui_controls` in the manifest are those last two. A
report that omits them reads as full coverage and is not.

If everything the pass reached responded correctly, say exactly that — with
the coverage line. A clean result stated plainly is a finding.
