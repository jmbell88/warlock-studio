# Warlock Studio Manual — Design

Date: 2026-08-04
Status: approved

## Goal

A complete documentation/manual for Warlock Studio that is both browsable inside
the app and readable as plain markdown files in the repo/on GitHub. One source
of truth: the same `.md` files serve both.

Audience: users (how to use the app), operators (setup, models, env vars,
troubleshooting), and developers (architecture, pipelines, extending).

## Decisions

- **Audience**: users + operators + developers, all in one manual.
- **In-app form**: a browsable Manual window rendering the markdown, plus
  contextual `(?)` entry points on panes that jump to the relevant chapter
  anchor. Developer chapters render in-app too, under their own part.
- **Media**: text-only. No screenshots, no image files — prose, tables and
  shortcut lists. Avoids the screenshot-staleness trap and keeps the in-app
  renderer image-free.
- **Renderer**: in-house markdown-subset renderer (not `imgui_md`, which is
  wired to hello_imgui's asset/font system the app deliberately does not use;
  not docs-as-Python-data, which inverts authorship).

## Content: chapters

`docs/manual/`, numbered files, one chapter each. First `# H1` in the file is
the chapter title; filename ordering is chapter ordering. `00-index.md` is the
table of contents.

**Part I — Using Warlock Studio**

1. `01-overview.md` — what the app is, the 2D→3D pipeline idea, the three
   modes, tour of the window (settings / viewport / inspector).
2. `02-generating-references.md` — 2D mode: prompt + guidance fields, base
   models & style LoRAs, candidate fan-out, seeds, approving a reference.
3. `03-generating-meshes.md` — 3D mode: promote-to-model, image upload, mesh
   params, triangle budget/retarget, mesh audit vs mesh report (what each
   badge means), exports (GLB/STL/OBJ/FBX/collision/textures).
4. `04-rigging-and-posing.md` — skeleton templates, auto-rig, envelope
   fallback, pose editor & gizmos, posed GLB downloads.
5. `05-sprite-sheets.md` — sheet panel, grid logic (poses × 8 yaws), preview
   vs final render, JSON sidecar format.
6. `06-paint.md` — tools, layers, blend modes, selections, transform, ORA/PNG
   saving, the three pipeline bridges (Open in Paint / Save as reference /
   Send to 3D).
7. `07-library-and-jobs.md` — job lifecycle, rerun/promotion (what carries
   over, what's stripped), cancel, prune, provenance.
8. `08-shortcuts.md` — every keybinding in one table, by mode.

**Part II — Setup & operations**

9. `09-installation.md` — uv sync extras, trellis binary, weights downloads,
   doctor.
10. `10-configuration.md` — every `WARLOCK_*` env var with default and effect,
    VRAM modes, idle eviction.
11. `11-troubleshooting.md` — OOM, missing weights, bpy absent, non-manifold
    rig failures, mesh holes (band sweep result), where logs/assets live.

**Part III — Architecture**

12. `12-architecture.md` — threads, queue, service layer, single-connection
    SQLite, offline invariant.
13. `13-pipelines.md` — source.glb vs model.glb, normalize/optimize ordering,
    prompt chunking, Blender-out-of-process, pose contract, sheet planning.
14. `14-extending.md` — adding a base model (registry entry), a skeleton
    template (JSON file), a LoRA; the derived-params rule; the pure-module
    boundaries (paint, sheet, manual itself).

Part III is written *from* the invariants in CLAUDE.md but *for a reader* —
CLAUDE.md stays untouched and remains the authoritative agent-instruction file.

## File layout, packaging, loading

- Canonical files: `docs/manual/*.md` — visible in the repo, readable on
  GitHub.
- Wheel: hatchling `force-include` maps `docs/manual` → `warlock/manual`.
  Exactly one copy in the repo; no build-step sync.
- Runtime loader (`src/warlock/studio/manual/loader.py`): resolves chapters via `importlib.resources` (installed) with a dev-checkout
  fallback to `docs/manual/` — same pattern as `src/warlock/templates/`.
- Chapter discovery: filename ordering; title from first H1; no frontmatter.
- Cross-references: ordinary relative markdown links
  (`[Rigging](04-rigging-and-posing.md#templates)`) so they resolve on GitHub
  and in-app alike. The in-app renderer treats a `.md` link target as chapter
  navigation and a `#fragment` as scroll-to-anchor.

## In-app renderer

**Pure half — `src/warlock/studio/manual/parser.py`.** No imgui / moderngl /
pygame imports (the paint-engine rule). Parses a deliberate markdown subset
into a flat list of typed blocks:

- `Heading(level, text, anchor)` — anchors are GitHub-style slugs so the same
  `#fragment` works in both worlds.
- `Paragraph(spans)` — spans carry bold / italic / inline-code / link runs.
- `CodeBlock(text, lang)`
- `ListItem(depth, ordered, spans)`
- `Table(rows)` (spans per cell)

The parser is **strict**: constructs outside the subset (images, inline HTML,
nested blockquotes, footnotes) raise, so a docs-integrity test fails rather
than the app rendering something wrong. This is the mechanism that keeps the
files inside the supported subset.

**imgui half — `src/warlock/studio/manual/render.py`.** Draws the block list
with plain imgui widgets through the existing moderngl backend: wrapped text,
theme-styled colors for code/bold, `begin_table` for tables, selectables for
links. Link behavior: `.md` target → navigate chapter; `#fragment` → scroll to
anchor; `http(s)` → copy URL to clipboard (offline app — never launch a
browser).

## Manual window

- Opened from a **Help** button in the top bar and by `F1`.
- Left: TOC tree grouped by the three parts. Right: chapter view with scroll
  and a prev/next footer.
- Search box filters TOC by chapter title + heading text. No full-text index.
- State (`open`, `chapter`, `pending_anchor`) lives on `studio/state.py` like
  every other pane's state.

## Context help

`HELP_TARGETS`: a map of pane key → `(chapter, anchor)`. Each pane header gets
a `(?)` icon button calling `state.manual.open_at(chapter, anchor)`. Panes know
one string; the manual knows the map. That is the entire coupling.

## Testing

- **Parser unit tests**: subset accepted, violations rejected, anchor slugs
  stable, span runs correct.
- **Docs integrity test**: every chapter parses; every cross-link and anchor
  resolves; `00-index.md` links every chapter; every `HELP_TARGETS` entry
  points at a real chapter + anchor.
- **Render half**: skip-without-GL, like the existing viewer tests.
- **Content**: a consistency pass of chapter text against README / CLAUDE.md
  facts before merge.

## Out of scope

- Screenshots or image assets of any kind.
- Full-text search.
- Localisation.
- Rendering markdown from anywhere other than the packaged manual.
