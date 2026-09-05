# Constraints — paste this block verbatim into every subagent prompt

These rules are not advice. Each one is a recorded incident in this repository.

- **Git is read-only for you.** Never run `git stash`, `git checkout`, `git reset`,
  `git restore`, `git clean`, `git add` or `git commit`. On 2026-09-04 a fixer ran
  `git stash` to compare pre-fix behaviour and reverted nine other agents' edits and the
  user's uncommitted work; recovery took `git fsck`. To see the committed version of a
  file, read `git show HEAD:<path>`. To see what has changed, read `git diff -- <path>`.
- **Never run the full test suite.** Run only the test files your brief names, and only
  as `uv run pytest <files> -n 0`. Never a bare `uv run pytest`, never `-n auto`, never
  `--dist load`, never `-m gpu`, never `-m perf`. Several tests read module source while
  they run, so a suite run while another agent edits `src/` fails both of you. The
  orchestrator runs the suite once, after everyone has returned.
- **Scratch files go in the scratchpad path your brief gives you**, never in the tree.
- **Never cite `TODO.md`, a plan file, or the audit file's name from `src/` or
  `scripts/`.** `tests/test_ux_todo_fixes.py` refuses it. Cite the programme instead:
  "the 2026-09-05 audit, finding sirens-03".
- **Uncommitted changes you did not make are the user's.** Do not revert them, do not
  tidy them, do not report them as defects unless they are one.
- **`uv run --no-sync`** for any Python you run; a bare `uv sync` prunes the extras and
  breaks collection for ten test files.
- **Windows.** Paths may have spaces; quote them. Line endings in the working copy are
  CRLF and in blobs LF; do not "fix" line endings.
