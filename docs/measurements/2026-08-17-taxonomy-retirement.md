# 2026-08-17 — Taxonomy retirement (PROMPT_VERSION 4 → 5)

## What changed

Twelve of the thirteen taxonomy option tables were removed from
`guidance.py`, and with them their keys from `vectors.VECTOR_PARAMS`:

`category`, `silhouette`, `material`, `condition`, `rarity`, `emissive`,
`setting`, `genre`, `mood`, `art_style`, `palette`, `framing`.

`platform` survives, but only as a **geometry-resolution** selection (512 for
2D, 1024 for 3D): its 2D prompt-fragment role and the "detail brief" combo
built on it are retired with the rest. The shipped `PRESETS`, the vector-preset
save mechanism, `colour_conflicts`, `TILE_FIELDS` and `framing_clause`/
`view_clause` all die with the tables they were written against.

## Why this needs a measurements document

Per the repo rule, a constant the stored corpus is keyed on gets its document
before it changes. `vectors.config_vector` / `vector_key` key every verdict,
observation and findings bucket on `VECTOR_PARAMS`, so removing twelve keys
re-keys the corpus:

- **Same-settings vectors re-key.** A job whose params carry any retired key
  hashed to a different `vector_key` than the same settings will hash to now.
  Evidence accumulation restarts per configuration — exactly the accepted cost
  `_LEGACY_ALIASES` already recorded for a rename ("evidence gathered under the
  old names simply stops accumulating"). A retirement is the same cost applied
  once, deliberately, rather than per-rename.
- **Old rows are tolerated, not migrated.** `config_vector` skips params keys
  not in `VECTOR_PARAMS`, `normalize` iterates its own tables so stale keys in
  stored params are simply ignored, and `compose_prompt` already skipped
  unknowns. Reroll and promotion of a pre-retirement job succeed with the
  fragments simply gone.
- **`size_m` loses its category default.** With `CATEGORIES` gone, an empty
  `size_m` always falls back to `DEFAULT_SIZE_M` (1.0 m) instead of the chosen
  category's typical size. Explicit sizes are unchanged.

## PROMPT_VERSION 4 → 5

The `{view}` slot in `PROMPT_TEMPLATE` is re-inlined as the literal
"3/4 perspective view" it was extracted from, so the **empty-params default
composition is byte-identical** to version 4. The bump is for the non-default
population: any stored job whose params carry taxonomy fragments (nearly all
of them) now composes a shorter prompt, so a recipe recorded under 4 that
named taxonomy no longer reproduces byte-for-byte. Bench suites are re-minted
(`core-v2`, `pixel-v2`) rather than edited, because suites are fingerprinted.

## Background: why retire rather than keep

No taxonomy axis ever measured a win:

- The one framing measurement (`2026-08-09-framing-axis.md`) was null and
  directionally *against* the alternative; the option was already `hidden`.
- The 2026-08-07 rogue sweep's only signal was `bg_removal`, not any taxonomy
  field; the 17 refusals were a prompt-template property.
- The tier qualification (`2026-08-13-tier-qualification.md`) and re-baseline
  work never attributed a quality difference to a taxonomy selection.

Meanwhile the taxonomy carried real ongoing cost: a 13-table vocabulary in the
UI ("More options" with three guidance groups plus a detail brief), the alias
table, the colour-conflict advisor, preset plumbing, sweep axes and ~35 test
files coupled to the tables. The user's brief is the prompt; the surviving
fields are the ones that select *machinery* (platform resolution, base model,
style LoRA, conditioning), not adjectives.
