# Fixer brief

Fill the `{...}` slots, prepend the constraints block, and send as the prompt. One
fixer owns a disjoint set of files; if two findings share a file they go to one fixer.

---

You are closing findings from the `{date}` audit of Warlock Studio, a fully offline
desktop app for making game assets. Read `CLAUDE.md` first; it is short and every rule in
it applies to you.

**Your findings** (the records are the specification; do not widen them):

```
{finding records, pasted whole}
```

**Files you own** (edit these and nothing else):

```
{owned paths}
```

**Test files you may run**, always as `uv run pytest <files> -n 0`:

```
{the regression file(s) you will add}
{the existing test modules that cover your owned files}
{gate files, only if your fix touches a repo-wide table — see below}
```

**Scratchpad:** `{scratch path}`

## The standard a fix meets here

1. **A regression test whose name is the claim**, in the test module beside the code (or
   the file named above), and it must **fail against the unfixed code**. Prove it: write
   the test first, run it, paste the failing assertion into your return; then fix; then
   run it again. A test that passed before the fix proves nothing and is re-opened.
2. **Comments explain why**, naming the incident: "the 2026-09-05 audit found X because
   Y". Never cite `TODO.md` or the audit file's name from `src/` or `scripts/`.
3. **When a hard constraint moves**, `docs/INVARIANTS.md` gets a paragraph in the same
   change. You do not own that file: return the paragraph, in that file's voice (bold
   lead-in sentence, then the rule and the incident), and the orchestrator writes it.
4. **When behaviour the manual describes moves**, the manual line moves in the same
   change. Same rule: return the chapter and the sentence.
5. **When a corpus-keyed constant changes** (`trellis_band`, `SEAM_MAX`, the grade scale,
   anything a `docs/measurements/` document is keyed on), stop and return "needs a
   measurement document first" instead of changing it.
6. **Refusals** raise `service.errors` exceptions carrying a `field`. **Writes** onto a
   served name are staged to a temp sibling and `os.replace`d. **Undo** is addressed by
   uid. **Headless packages** (`studio/inker`, `clay`, `plotter`, `packwright`, `sirens`,
   `troupe`, `muse`, `tour`) import no imgui, moderngl, pygame or `service`; if your fix
   needs one, it is in the wrong layer. **Every subprocess** goes inside the `winjob`
   wrapper. **A new job kind** is a sweep of every stage-keyed table.
7. **Pre-fix behaviour** is read from `git show HEAD:<path>`, never by checking anything
   out. If you need to compare outputs, write both to the scratchpad.

## Tests you may run, and the one rule about them

Only the files listed above, only with `-n 0`. If your fix touched a repo-wide table
(`modes.KEYS`, a `_MOVED` table, the stage-keyed job tables, a manual chapter number, an
import pin), run the specific gate test file for that table, named above, and no other.
Never run the suite. The orchestrator runs it once after every fixer has returned; a
failure it finds in your files comes back to you.

## If you cannot finish

Stop and return rather than improvising: you need a file you do not own; the finding is
already fixed in the tree (say by whom, `git log -1 -- <path>`); the finding is wrong
(say why, with lines); or the fix wants a design decision. Partial edits are fine to leave
in place if every file you touched still imports and your named tests pass.

## Return

- Files changed, one line each.
- For each finding: the regression test name, its failing output before the fix (pasted),
  its passing run after.
- Paragraphs for shared files (INVARIANTS, manual, CHANGELOG), each marked with its
  destination.
- Anything left open, and why.
- Nothing else.
