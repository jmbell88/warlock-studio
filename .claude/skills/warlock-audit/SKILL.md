---
name: warlock-audit
description: Audit one slice of Warlock Studio (a mode, the service layer, the pipelines, the shell, the docs, a defect theme, or all of it) by fanning Sonnet explorers out over the tree and merging their findings into a dated docs/audit-YYYY-MM-DD.md, then, only when asked, close the buildable findings with file-owned Sonnet fixers and one suite run. Use when asked to audit, review the code base, do a doc check-up, or find what is wrong in a mode (/warlock-audit sirens, /warlock-audit docs, /warlock-audit all).
argument-hint: <scope>
arguments: [scope]
---

# Audit a slice of Warlock Studio

You are the orchestrator. Your job is to *run this procedure*, not to redesign it.
Every long text you need is in `references/` next to this file; read a reference the
first time a step names it and copy from it rather than paraphrasing. When the
procedure and your judgement disagree, the procedure wins unless the user says so.

Three audits precede this skill: the 2026-08-24 release audit, the 2026-09-02 review
(`FINDINGS.md`, seven per-subsystem passes, deleted 2026-09-04) and the 2026-09-04 beta
audit (`docs/beta-audit-2026-09-04.md`, closed 2026-09-05 in commit c7a78fb2). Each
found real defects, and each also *re-listed work that was already done*. The
2026-09-02 review names that as a defect in the audit, and so does this skill: a
finding that is already fixed in the tree, or already refused by a test, is your mistake.

**This pass reports. It does not fix.** The fix phase (step 8) runs only when the user
asks for it after reading the report, in this session or a later one.

## Rules that hold in every step

`references/constraints.md` is pasted, verbatim, into every subagent prompt. Read it
now; it is short. Its two load-bearing lines: no subagent ever runs `git stash`,
`checkout`, `reset` or `restore` (a fixer did on 2026-09-04 and reverted nine other
agents' edits plus the user's uncommitted work), and no subagent ever runs the full
test suite. A subagent runs only the test files its brief names, with `-n 0`.

The orchestrator has one extra rule: the default suite runs **once**, by you, in step 8,
after every fixer has returned. Not before, and never while anything else may be editing.

## 1. Resolve the scope

`$scope` is one of:

- `all` — every row of `references/subsystems.md`.
- a mode key — one of the thirteen in `warlock.studio.modes.KEYS`. Check it:
  ```
  uv run --no-sync python -c "from warlock.studio import modes; print(' '.join(modes.KEYS))"
  ```
  `home`, `library`, `review` and `settings` resolve to the `shell` row; `create` to
  `create`; the eight workspaces to their own rows.
- `shell`, `create`, `service`, `pipelines`, `docs`, `tour` — rows of `subsystems.md`.
- a theme key from `references/defect-classes.md` (for example `undo`, `frame-thread`,
  `staged-writes`). A theme runs *one* checklist item across *every* row.

Refuse anything else and print the three lists. Do not guess at a near-match.

## 2. Freeze the baseline

Record, and put in the report verbatim:

```
git rev-parse --short HEAD
git status --short
grep -m1 '^version' pyproject.toml
uv run ruff check .
```

Findings are against the **working tree**, not HEAD, and the report says which files were
dirty when the audit started. Do not run pytest here.

Make one scratchpad directory for the run, `<scratchpad>/audit-<date>/`, with a
`probes/` subdirectory. Explorers write reproduction scripts there.

## 3. Fan out explorers

Read `references/brief-explorer.md` once. For each slice in scope, launch one `Agent`
call in a **single message**, with:

- `subagent_type: general-purpose`
- `model: sonnet`
- `description`: `audit:<slice key>`
- `prompt`: the explorer brief, with the constraints block, the slice's row from
  `subsystems.md`, the probes directory path, the HEAD hash from step 2, and (for a
  theme scope) the one checklist item to walk.

Launch at most eight at once; when a batch returns, launch the next. Do not read the
code yourself while they run; your context is for the merge.

## 4. Merge

Collect every record. Then:

1. **Dedupe** on (`where`, `claim`). Two explorers reporting one defect from two files is
   one finding with two `where` entries.
2. **Drop what is guarded.** A record whose `existing_guard` names a real test that would
   fail on the claim is not a finding. Check the test exists.
3. **Second-look every Critical and High yourself.** Open the cited lines. Confirm the
   claim is true of the working tree. If the record says *Reproduced*, run its probe from
   `probes/`. Downgrade or drop what does not survive; note in the report that you did.
4. **Assign ids** `<slice>-NN` in severity order and keep them stable from here on.
5. **Split out `human_only: yes`** into their own table; they are owed, not open.

Severity and evidence words come from `references/report-template.md` and nowhere else.

## 5. Write the audit file

`docs/audit-YYYY-MM-DD.md`, from `references/report-template.md`. Every section of the
template is present even when empty ("Critical — none"). The coverage section is not
optional: which slices ran, which explorers failed or timed out and what they left
uncovered, and what could not be verified on this machine.

If a file for today already exists, append a dated second run beneath a `---`; do not
overwrite.

## 6. Report in chat, then stop

Counts by severity on one line. Each High in one line with its id. The file path. Which
slices were skipped. Then stop. Do not begin fixing, and do not offer to.

## 7. Doc check-up

When the scope is `docs` or `all`, the `docs` explorer walks `references/docs-checkup.md`
instead of the code checklist. Its findings use the same record shape. A `TODO.md` entry
that has become buildable is a finding of severity Medium with `suggested_fix: build it`.

## 8. Fix phase — only when the user asks

The work order is the audit file. Read it, not your memory of it.

1. **Partition by file ownership.** List the files each open, non-human finding will
   touch (source, its tests, its manual chapter, `docs/INVARIANTS.md` if a constraint
   moves). Findings that share a file go to the same fixer. No two concurrent fixers own
   one file. `docs/INVARIANTS.md`, `CHANGELOG.md` and `docs/manual/*` are shared: fixers
   *return* the paragraph they want added and you write it in step 4.
2. **Launch fixers** from `references/brief-fixer.md`, `model: sonnet`, one message,
   at most eight at once, `description`: `fix:<ids>`. Each brief carries the constraints
   block, its findings, its owned files, the test files it may run, and the scratchpad.
3. **Read every return.** A fixer that says "needs a file I do not own" gets the file
   in a second round after the fixer that owns it has returned. A fixer that could not
   make its regression fail before the fix has not proven the finding; re-open it.
4. **Write the shared files** from what fixers returned: INVARIANTS paragraphs, manual
   lines, a CHANGELOG entry under the unreleased heading in that file's voice.
5. **Run the gate, once, in this order, nothing else editing:**
   ```
   uv run ruff check .
   uv run pytest
   uv run python scripts/preflight.py --fast
   ```
   A failure inside a fixer's files goes back to *that* fixer with the traceback. A
   failure outside every fixer's files is a finding: record it, dispatch one fixer, rerun.
   Do not patch test failures yourself from the orchestrator seat.
6. **Update the audit file.** Strike each closed finding as the 2026-09-02 review did:
   `~~claim~~ Built YYYY-MM-DD: what was actually done, and the test.` Move each
   `human_only` finding into `TODO.md` as a `## P<next>.` entry in that file's format
   (*Why it is yours* / *Do* / *Expected outcome*). If no finding is left open, delete the
   audit file; git keeps it.
7. **Commit once**, only the paths this pass touched:
   ```
   git add <each path>       # never -A, never .
   git commit -m "<title>" -m "<body>" -m "Co-Authored-By: ..." -m "Claude-Session: ..."
   ```
   Titles are sentences about what changed for the user, not "fix audit findings"; read
   `git log --format=%s -20` for the register. If `git status` shows files you did not
   touch, they are the user's: leave them unstaged and say so.
8. **Report**: what closed (ids), what moved to `TODO.md`, the gate output's last lines,
   the commit hash.
