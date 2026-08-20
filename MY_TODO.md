# MY_TODO.md — what's left, and who has to do it

Written 2026-08-20, after a pass that closed every open item in the repo that
code alone could close. Everything below is here for one of three reasons: it
**needs you** (a human with an app, a card, or an opinion about art), it **needs
a decision** before it can be built, or it **is a program** rather than a task
and should start with a design conversation.

Nothing here is blocked on me finding time. Where I could build it, I did.

The suite is green: **10,872 passing**, ruff clean. (`tests/test_inker_textures.py`
and `tests/test_dialogs_prompt.py` still flake under xdist and pass serially —
that is the known scheduling flake, not a regression.)

---

## 1. Art direction — blocks the whole Troupe pipeline

**This is the single highest-value thing on the list.** Everything downstream of
it is built and waiting.

- [ ] **Two or three pixel-art references** whose look you want. `LPC_ALT.md`
      Phase 1 says plainly that without them it "has no bar to clear" — every
      palette, outline and shading decision is judged against these, and
      "too generic" was one of the four complaints the whole program exists to
      escape.
- [ ] **Palette ramps**, or approval of ones derived from those references, as
      `.hex`/`.gpl` through `service/palettes.py`. Median-cut quantisation is
      *why* 3D-derived sheets come out muddy; authored ramps are the fix.
- [ ] **A textured base mesh** (or a run through the Phase 4 reference chain).
      Both `examples/*_base.obj` carry no texture, so every frame currently
      quantises into the pale end of the palette — **the palette is unproven on
      anything but greys.**

## 2. Author the 22 keyframes — the editor is now built for exactly this

**New this pass:** Poser has a clip editor. Open Poser → **Clips** in the left
sidebar. Pick a key, pose the skeleton with the normal gizmos, **Update key from
pose**. Onion skin ghosts the keys either side; **Play** scrubs the real
interpolation. **Save clips** writes to your own data folder and never touches
what the build ships, so **Revert** is always available and an update can't
overwrite you. Manual: *Poser → Editing clips*.

- [ ] **Author the keys.** The shipped 22 are provisional. `LPC_ALT.md` calls
      this "the most important art task in the program", and it is the one thing
      that cannot be automated away.
- [ ] Decide **who does it** — you, or an animator. The plan flags this as a
      scheduling question, and it still is.

Two things I found while building it that you'll want to know before you start:

- **Easing does nothing at the current segment lengths.** It reshapes *where
  inside a step* frames land, so it needs a step of ≥3 frames. Every shipped
  step is 1 or 2, and `ease` is a smoothstep whose value at the only interior
  sample is exactly 0.5 — so `idle`'s `ease` renders identically to `linear`
  right now. `ease_in`/`ease_out` do differ. The panel says so; I mention it
  because otherwise you'd change it, see nothing, and assume it's ignored.
- **The arms hang slightly forward** on the shipped keys (noted at the 0d
  spike, still true). Worth an art pass while you're in there.

## 3. Decisions I need from you

- [ ] **Where does the "judge clips as pixels" preview live?** Phase 2 asks for
      a live low-res *sprite* preview in Poser. **It cannot go there**: Poser's
      preview is a *meshless armature*, so there is nothing to pixelise — a
      sprite preview would show reduced bone lines. Either Poser learns to load
      a rigged asset for preview, or the pixel verdict stays in Troupe where the
      mesh is. I shipped the scrubber (real interpolation, real timing) as the
      fast loop instead, which is right regardless. **Your call which way.**
- [ ] **Troupe Phase 6 — the cleanup workflow.** Not built, deliberately. It is
      three features, and the plan's own risk list says the hard one is *"worth
      designing deliberately rather than discovering on contact"*:
      propagate-a-correction across frames/directions, mirror-assisted cleanup
      (the measured W/E mirror property), and **re-render one animation without
      discarding hand edits**. That last one is a genuine design problem, not an
      implementation task. Worth a conversation before anyone writes code.
- [ ] **Troupe Phases 7 and 8** stay deferred, as the plan has them — layered
      equipment and AI restyle both gate on whole-character generation working,
      which gates on item 1 above.

## 4. Manual passes — need an app I don't have

Both follow the repo's standing rule: *the claim only strengthens once a human
with the app installed has looked.*

- [ ] **Open a Warlock-written `.aseprite` in real Aseprite.**
      `tests/inker/fixtures/aseprite/FIXTURES.md` names the four fixtures worth
      authoring first. **Start with the tilemap ones** —
      `tilemap-rgb`, `tilemap-indexed`, `spare-tileset`. Their chunk field order
      was written by inverting the *reader*, and a round trip through our own
      two halves cannot catch an order both halves get wrong together. That's
      the highest-value five minutes on this whole list.
      Also worth a glance in the same sitting: **new this pass**, every RGB and
      grayscale file now carries a palette chunk derived from the art's own
      colours (divergence #23) — including a 1-entry transparent palette on a
      blank document. Check Aseprite is happy with that.
- [ ] **Author a `.tmx`/`.tsx` fixture in real Tiled**, and re-check a grid
      pack's `.tsx` geometry (pow2 rounding is off by default now, so the
      standing verification is stale). `tsx.TILED_VERSION` is pinned at
      **1.10.2** against a 1.12.2 target and only moves when a human with Tiled
      has looked — it has been wrongly bumped once already.

## 5. GPU runs — need your card

- [ ] **`uv run pytest -m gpu -n 0`.** I have not run the GPU lane at all this
      pass. Nothing I changed touches model loading or VRAM, but the lane is the
      only thing that sees real weights.
- [ ] **Run a `charsheet` job end to end against real Blender.** Phase 4's job
      has *never* been run on a card. The pieces either side of it have (Phase
      0d), and the render call is `rigging.sheet_spec` + `run_worker` exactly as
      `_sheet` makes it — but the end-to-end run is owed, and it is how you'd
      find out that the clip edits from item 2 actually reach a rendered sheet.
- [ ] **Troupe Phase 0e** — humanoid reconstruction from a single image is
      untested. It only matters for the *generated-character* path; the
      supplied-base-mesh path works without it.

## 6. Small, unscheduled, genuinely optional

The Aseprite P1 backlog stays unscheduled *by design* — `ASEPRITE_PARITY.md`
says items are pulled into sessions individually, never waved. I did not build
any of it. Three are cheap and sit on machinery that already exists, if you want
them:

- [ ] Shift-to-line-from-last-point (pure `inker_canvas` press logic + `line_pixels`)
- [ ] Onion-skin "current layer only" (filter the `frame_stack` fold to one track)
- [ ] Color Range selection (the global `similar` exists; it needs a UI door)

## 7. Not on this list on purpose

- **`EXE_PLAN.md`** — you excluded it. It is still fully specified and unstarted,
  and it is the largest single piece of unbuilt work in the repo.
- **Scale and crop of a tilemap layer** stay refused, permanently. They
  *resample*, and a tileset cannot follow a resample — there is no permutation
  to teach, only a re-cut, which is a different operation. This is a decision,
  not a gap.
- **Pen/tablet pressure, ICC colour, per-frame palettes, per-cel opacity** and
  the rest of `ASEPRITE_PARITY.md`'s named non-goals.

---

## What changed this pass, for context

Twelve open items closed, all of Aseprite Waves 3–5's leftovers plus the
measurement-gated one:

| | |
|---|---|
| **Flip/rotate a tilemap layer** | The eight-symmetry flag algebra moved out of the Plotter into the shared `tilegrid.gid` leaf (divergence #24) so both editors share one table; whole-canvas *and* timeline-range flip, rotate and tile-aligned shift now carry a tilemap's refs. The bar was the pixels, not a flag table. |
| **Stroke invalidation** | Measured first (`docs/measurements/2026-08-20-stroke-invalidation.md`). The known `symmetry="xy"` cliff was real and was the *smaller* half — a plain stroke was recompositing 33× more area per dab at its end than its start. **6.6× faster** on ordinary strokes, 3.9× on `xy`, 64–107× less area. |
| **`.aseprite` palette chunk** | Every file now carries one; derived from the art's own colours rather than reciting Aseprite's default from memory. Eight fixtures regenerated, the three indexed ones deliberately untouched. |
| **Export refusals** | Every early refusal in `_submit_export` now has a mode-level test; `slices_conflict` is recorded in the sidecar instead of dropped silently; a visible group whose layers are all hidden no longer writes a transparent sheet. |
| **Troupe Phase 2** | The clip editor above. |

One trap worth remembering if you ever regenerate the Aseprite corpus: **run it
under `uv run python`, never a bare `python`.** The system interpreter has a
different zlib, and its compressed output silently rewrote three fixtures the
gate considered perfectly correct.
