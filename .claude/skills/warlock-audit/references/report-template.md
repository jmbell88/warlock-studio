# Report template — `docs/audit-YYYY-MM-DD.md`

Copy this skeleton. Every section stays, even when its body is "none". The column layout
is the 2026-09-04 beta audit's, so the two files read alike.

---

# Warlock Studio audit — YYYY-MM-DD

**Recommendation:** one sentence: what to do before what (e.g. "close the two Highs
before the next installer build").

**N findings: C Critical, H High, M Medium, L Low; K human-only.**

Audited: version **X.Y.Z**, working tree at **`<short hash>`**. Dirty at start:
`<git status --short output, or "clean">`. Scope: `<scope arg>` → slices
`<list>`. `uv run ruff check .`: `<result>`. The suite was not run by this audit; findings
are from reading and from the offline probes below.

**Vocabulary.** *Critical*: crash, data loss, or corrupts a document. *High*: a wrong
result the user will hit, or a promise in the docs the code breaks. *Medium*: a defect
with a workaround, or a hard-invariant violation not yet reachable. *Low*: hardening,
consistency, naming. *Reproduced*: an executable example confirmed it (scripts under
`<probes path>`, run with `uv run --no-sync python <script>`). *Inspected*: follows from
the current code path. *Evidence gap*: release proof is missing from the repository, not
a demonstrated failure.

**Second-look.** Every Critical and High below was re-read by the orchestrator against
the working tree. Dropped or downgraded during merge: `<ids and one reason each, or
"none">`.

## Critical

| ID / where | Finding and evidence | Recommended fix |
|---|---|---|
| **slice-01 — Mode: control** | **Claim.** [file.py:123](src/...). *Reproduced/Inspected.* Why it matters. | Suggested fix. Regression: `test_name`. |

## High

(same table)

## Medium

(same table)

## Low

(same table)

## Human-only — owed, not open

Findings that need a card, a clean machine, real Aseprite or Tiled, or ears. These move
to `TODO.md` in the fix phase and are not counted as open here.

| ID | What has to happen | Who / what it needs |
|---|---|---|

## Coverage

- Slices run: `<list>`; slices skipped: `<list and why>`.
- Explorers that failed or timed out, and the files that left uncovered: `<or none>`.
- Checklist items not evaluable in this scope: `<or none>`.
- Not done by this audit: no interactive pass, no fresh install, no GPU inference, no
  third-party editor validation, no dependency-advisory scan. A passing suite does not
  close those.

## How this file is closed

A finding is struck the day it is built: `~~claim~~ Built YYYY-MM-DD: what was done, and
the test.` A human-only finding is moved to `TODO.md`. When nothing is left, this file is
deleted rather than ticked; `git log --diff-filter=D -- docs/audit-*.md` finds it.
Nothing under `src/` or `scripts/` may cite this filename.
