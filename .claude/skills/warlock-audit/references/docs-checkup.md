# Doc check-up — what the `docs` slice walks

Paste this whole file into the `docs` explorer's brief in place of the defect-class
checklist. Every row says whether a test already gates it. **Gated** rows are not
reported unless the explorer can show the gate is not actually covering the case;
**not gated** rows are reported as findings whose `suggested_fix` includes the gate.

The rule of this repository: a document that disagrees with the tree is worse than no
document. Stale counts, dead filenames and promises the code does not keep are each a
finding, Medium by default, High when a user following the document would lose work or
reach a dead end.

## A. Structure (mostly gated)

| Check | Gate |
|---|---|
| Version lockstep across `pyproject.toml`, `__init__.py`, `CHANGELOG.md`, `INSTALL.md` | `scripts/preflight.py`, `tests/test_changelog.py` — gated |
| Every release has a CHANGELOG entry; no entry for an unreleased version claims a date | `tests/test_changelog.py` — gated |
| Manual chapters: number decides order and part; 01–19 tutorial; index lists every chapter; every section a loader can find | `tests/manual/test_sections.py`, `test_loader.py`, `test_coverage.py` — gated |
| Shortcut sheet (chapter 38) matches `shortcuts.py` | `tests/manual/test_shortcuts.py` — gated |
| Every relative link and image in `docs/` resolves | `tests/test_external_doc_links.py` — gated; confirm it covers root `.md` files too |
| No `src/` or `scripts/` file cites `TODO.md` or a deleted plan filename | `tests/test_ux_todo_fixes.py` — gated |
| Workspace-count words ("eight workspaces", "thirteen modes") | `tests/test_findings_followups.py` — gated; check its word list still matches `modes.KEYS` |
| COMPAT.md Tiled rows are executable | `tests/plotter/test_compat_matrix.py` — gated |

## B. Inventories against their source of truth (not gated unless noted)

| Document | Compare against | Notes |
|---|---|---|
| `README.md` mode list and numbering | `modes.KEYS` | L04 found Muse missing and Sirens twice |
| `docs/manual/20-overview.md` mode list | `modes.KEYS` | same incident |
| `CLAUDE.md` mode sentence, command list, file names, "N test files" claims | `modes.KEYS`, `pyproject.toml` scripts, the tree | rewritten 2026-09-04; drifts fast |
| `CONTRIBUTING.md` and `docs/manual/39-installation.md` extras tables | `[project.optional-dependencies]` in `pyproject.toml` | L05: "three extras" under a four-extra command |
| `INSTALL.md` sizes, steps, pack workflow | `installer/build.ps1`, `packs.py`, Settings → Packs pane | M16: guide described the pre-pack runtime |
| `SECURITY.md` file formats and network exceptions | `studio/filetypes.py`, `fetch_worker`, `pack_worker` | L07 |
| `THIRD-PARTY-NOTICES.md` model table | `docs/MODELS.md` registry, `models.py` | L06: Hybrid Demucs absent |
| `docs/MODELS.md` recipe registry and licences | `models.py`, `guidance.py`, `packs.py` | also check fenced blocks close (L05 found a heading swallowed) |
| `docs/COMPAT.md` Aseprite ledger | `studio/inker/asein.py`, `aseout.py`, INVARIANTS 261 | not executable; read both ways |
| Test counts anywhere ("~16k tests", "roughly 12,000") | `uv run --no-sync pytest --co -q 2>nul | tail -1` if cheap, else flag as volatile and propose deleting the number | |

## C. INVARIANTS.md against the code

For each of the 239 bold lead-in paragraphs in the slice's range (or all, for `docs`):

- every backticked symbol still exists (`grep -rn` the name under `src/`); a paragraph
  citing a renamed or deleted function is a finding (Low), and one whose *rule* the code
  no longer follows is a finding at the rule's severity;
- every cited test file exists;
- every "(YYYY-MM-DD, finding X)" citation points at a real audit or measurement.

## D. `docs/measurements/` against the constants they key

For each document that names a constant (`SEAM_MAX`, `trellis_band`, the grade scale,
`MAX_TOTAL_BYTES`, the VRAM figures, the perceptual-hash floor), find the constant in
`src/` and confirm the value matches the latest dated document. A constant that moved
without a document is a finding (High: the stored corpus is keyed on it).

## E. `TODO.md` against the tree

`TODO.md` may hold only: work a human must do, fully specified deliberately unstarted
work, and open audit findings. For each entry:

- **Could a Sonnet fixer build it today with no hardware, art or decision?** Then it is
  a finding: severity Medium, `claim: "TODO P<n> is buildable and unbuilt"`,
  `suggested_fix: build it and strike it`.
- **Does the entry describe the tree as it was?** ("the check is fatal" when it is now
  nonfatal — the 2026-09-04 audit found this). Finding, Low.
- **Is a closed record's claim still true?** A struck item whose fix regressed is High.

## F. The manual against the panes

For each chapter in the slice: every control the chapter names exists with that label in
the pane source (grep the label string under `studio/panes/`); every behaviour sentence
("Delete asks", "Export never overwrites") matches the handler. Label drift is Low;
behaviour drift is High (defect class 5).

## G. Comments and docstrings

`CLAUDE.md`'s rule: comments explain *why*, naming the incident. A docstring that
describes behaviour the function no longer has is a finding (Low), the same as a manual
sentence. Do not report missing docstrings.
