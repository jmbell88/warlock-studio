# Contributing

Thanks for looking. A few things about this codebase will save you time.

## Getting set up

```powershell
uv sync --extra studio --extra text2image --extra rig --extra music
uv run pytest
uv run ruff check .
```

**Pass all four extras** (`studio`, `text2image`, `rig`, `music`). A bare `uv sync` *prunes* them
and breaks about ten test files at collection. `rig` needs a Python 3.13 environment or it silently
installs nothing.

The suite is 16,392 tests in about two minutes, parallel by default. Three lanes
are excluded from the default run and each is opt-in:

| Lane | Command | When |
|---|---|---|
| GPU | `uv run pytest -m gpu -n 0` | Before changing model loading, VRAM accounting or conditioning. Serial is enforced -- N workers means N simultaneous 7 GB loads onto one card. |
| Performance | `uv run pytest -m perf -n 0` | Wall-clock budgets; meaningless under contention. |
| One test | `uv run pytest tests/x.py::y -n 0` | Quicker than paying for worker startup. |

**Never edit `src/` while the suite is running.** Several tests read module
source, and you will get failures that have nothing to do with your change.

## Before you write anything

Read **`docs/INVARIANTS.md`** for the subsystem you are touching. It is the
authoritative record of this codebase's hard constraints *and the measured
reasoning behind each one* -- most of them exist because something specific went
wrong, and the file says what. `CLAUDE.md` has the one-line summaries.

The ones that most often surprise people:

- **The app is fully offline.** `HF_HUB_OFFLINE=1` is set before anything
  imports. Nothing downloads at runtime except the user-initiated fetch worker,
  in its own environment.
- **Three threads.** The pygame frame loop never blocks; the asyncio worker
  lives on `warlock-loop`; everything blocking goes through `TaskRunner`. One GL
  context.
- **`service/` is the only business-logic layer.** Panes and tests both call it.
  Refusals raise `service.errors` exceptions carrying a `field`.
- **`bpy` never runs in the app process**, and every subprocess goes in the
  `winjob` kill-on-close job. A scan test enforces both.
- **The headless editor packages** (`studio/inker/`, `clay/`, `plotter/`,
  `packwright/`) import no imgui, moderngl, pygame or `service`. Import-pinning
  tests enforce the exact outward set, so adding an import means updating the
  pin -- deliberately.
- **Document writers stage to a temp and `os.replace`.** Never write over a
  user's file in place.
- **Untrusted parsers bound their allocations** before making them, not after.

## Style

Match the surrounding code. The distinctive thing about this codebase is that
comments explain *why*, usually by naming the incident that motivated the guard
-- if you fix a bug, say in a comment what the bug was, and add a test whose
name is the claim. A regression test that passes against the unfixed code is not
a regression test; check that it fails first.

## Commits and pull requests

- Run `uv run python scripts/preflight.py` before opening a PR. To run the whole
  of Windows CI locally first -- and the installer after it -- use
  `pwsh scripts\rebuild.ps1`.
- One logical change per commit.
- If you change something a stored measurement depends on (`trellis_band`,
  `SEAM_MAX`, the grade scale), a `docs/measurements/` document comes first.
- If you change behaviour the manual describes, update the manual in the same
  commit. Chapter numbering is test-gated in both directions.

## Licence

Contributions are accepted under **GPL-3.0-or-later**, matching the project. By
opening a pull request you agree your contribution ships under those terms.
