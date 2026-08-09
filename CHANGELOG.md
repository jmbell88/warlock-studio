# Changelog

Hand-written, newest first. Nothing here is derived from git: the commit
subjects are `Warlock vN.N.N` and carry no detail, so this file is the only
record of what a version actually changed. The top heading's version must
match `pyproject.toml` — a test asserts it, so a release bump cannot leave this
file behind.

## 0.0.16 — 2026-08-09

- **A mesh verdict is a grade, not a bit.** Review files −5..+5 instead of
  accept/reject, and the five rejection reasons became optional tags — legal at
  any grade, with five good ones added beside them — because a solid slab and a
  mesh a modeller would fix in five minutes were the same row. Image labels stay
  one keypress; they feed binary probes and a grade would be thresholded straight
  back to a bit.
- The re-baseline campaign ran. `bg_removal=birefnet` is the baseline —
  `auto` took 0 accepts in 90 units — and `hole_worst` reads the right way round
  again now the solid-slab failure mode is gone. Written up under
  `docs/measurements/`.
- **Home is a two-column screen.** What's new and the machine's status on one
  side, everything you were working on down the other. The tile grid is gone:
  seven of its nine tiles were the mode switch again, one click either way.
- **Library and Profiles are modes**, in the switch beside everything else,
  rather than sub-views of Home with nowhere else to live.
- **One Resume list across every document mode.** Inker, Clay, Plotter and
  Packwright each kept their own recent-files list; they are now one list,
  newest first, with your assets folded in and each row badged with the mode
  that opens it.
- Positional Alt+digit mode switching is gone. Twelve modes and ten digits was
  never going to fit; the palette (Ctrl+K) is the keyboard route.

## 0.0.14 — 2026-08-09

- **Plotter**: a tilemap editor, with Tiled `.tmx` import and export.
- **Packwright**: a deterministic sprite-atlas packer with a TexturePacker
  sidecar and a `.tsx` tileset for Plotter.
- Inker learned animation: a sparse track x frame grid, onion skinning,
  playback and a sprite-sheet export.
- A light palette, an accessibility reduce-motion switch, and three sidebar
  widths.

## 0.0.13 — 2026-08-08

- **Clay**: block a model out from primitives by hand and export it as an
  ordinary asset.
- Native kernels for the compositor, the contour tracer and the mesh audit,
  built on demand and always beside the numpy reference they are checked
  against.
- A download button for model weights, in a subprocess so the app itself stays
  offline.
- The manual moved into the app as its own mode.
