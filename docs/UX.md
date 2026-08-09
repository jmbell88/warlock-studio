# Warlock Studio — the Apple-feel design language and roadmap

*Written 2026-08-09. This is the design document behind `LEFTOVERS.md` §17. It
has two halves: a review of what the UI is today (Part I), and the phased
roadmap that closes the gap (Part III), with the principles that connect them
(Part II) and how each phase proves itself (Part IV). Every claim about the
current code carries a `file:line` anchor into `src/warlock/studio/` so a later
session can execute a phase without re-auditing.*

**The brief, as decided:** all four Apple qualities matter — visual refinement,
motion and fluidity, simplicity and progressive disclosure, and polish in the
small moments. Foundation first, so every mode improves at once; full technical
depth, including the custom-GPU tier, phased so each rung is independently
shippable; and *sensibility only* — the mode-switch shell, the panes and the
command palette stay. Warlock should look, move and feel Apple-grade without
wearing macOS chrome on Windows.

---

## Part I — Design review

### The thesis

The judgment layer is already frequently Apple-grade; the substrate is not.
Nearly every interaction decision in this codebase has a written rationale
naming the specific user failure it fixes — toast dwell as reading time
(`state.py:492-497`), refusals that teach ("Clay opens .wblk documents and
.glb meshes", `main.py:1512-1593`), verb-labeled confirms with the destructive
action red and the escape neutral (`dialogs.py:198-199`), greyed-not-absent
controls as policy (`widgets.py:601-606`), empty states everywhere
(`widgets.py:963`, `overlay.py:348-372`). That is the hard half of Apple
design and it is largely done.

What is missing is the *substrate* those judgments sit on: the type ramp tops
out at 16 px, spacing at 16 px, radii at 6 px; exactly one widget has a shadow;
exactly three widgets animate; every mode switch, popup, modal and hover is a
hard cut; and there is no translucency anywhere. The gap to "Apple" is scale,
depth, motion and disclosure — in that order of cheapness.

### What is already Apple-adjacent (keep, and build on)

- **A genuinely centralized system.** One token module (`tokens.py`), one
  style module (`theme.py`), one widget kit (`widgets.py`), live palette
  resolution through `theme.__getattr__` so a theme switch repaints hand-drawn
  rects too. A redesign lands in three files and radiates.
- **Elevation instead of outlines, stated as policy** (`theme.py:86-87`;
  `window_border_size=0`, `frame_border_size=0`). An 11-role palette, one
  indigo accent, near-black neutrals — "the Final Cut register"
  (`tokens.py:85`). The light palette preserves *roles*, not inverted values
  (`tokens.py:95-101`).
- **`segmented_control` is the single most convincing element in the app**
  (`widgets.py:673-727`): one rounded track, an animated sliding pill,
  per-segment text-alpha crossfade. Its docstring names its reference: "the
  macOS mode switch." `toggle` (`widgets.py:730-773`) is its sibling.
- **The toast system** (`widgets.py:1011-1103`): dwell per level chosen as
  reading time, hover pauses the clock by pushing `born` forward, eased rise
  on birth, fade on death, overflow counted ("+N more") rather than dropped.
- **The command palette** (`panes/palette.py`): anchored so the input never
  jumps, stable ranking, disabled commands listed-and-greyed rather than
  hidden — "a user searching for 'wireframe' from Home learns nothing from an
  empty result" (`palette.py:19-22`).
- **One icon family** (Lucide 0.525.0, `icons.py`), one type family (Inter in
  three weights), icon baseline metrics solved properly with a written
  derivation (`fonts.py:25-33`).

### The substrate gaps

**1. The vocabulary is thin.** Three type sizes — `TEXT_SMALL 11`,
`TEXT_BODY 13`, `TEXT_TITLE 16` (`tokens.py:73-75`) — so the Home hero
"Warlock Studio" renders at 16 px SemiBold (`panes/landing.py:211-212`), the
largest type in the entire app. Four spacing steps topping out at 16
(`tokens.py:60-63`); every larger gap is a literal (`sp(20)`, `sp(40)`,
`sp(48)` in `landing.py:159,202,219`). Two radii topping out at 6
(`tokens.py:67-68`). `PANE_PADDING = 5.0` (`layout.py:54`) — the most
utilitarian number in the codebase. ALL-CAPS micro-labels on every field and
section (`widgets.py:159-164, 843-846`) are a DAW idiom, not typography.
ASCII status glyphs — `"..."`, `">>"`, `"OK"`, `"!"`, `"x"`
(`theme.py:53-59`) — and a literal `"(?)"` help marker (`widgets.py:652-654`).
The comment at `tokens.py:56-59` records that `SP_6`, `SP_10` and `RADIUS_L`
were deleted *for having no readers* — the correct discipline, which is why
every addition below ships with its readers named.

**2. Depth is one recipe on one widget.** `card` (`widgets.py:918-960`) draws
two hard-edged translucent black rects offset 1 px and 3 px downward — no
blur, no spread. Its fill is a hard threshold mid-animation
(`ELEV_1 if lift < 0.5 else ELEV_2`, `widgets.py:940`), so the background pops
while the shadow fades — a visible artifact. Nothing else in the app has any
shadow; there is zero blur or translucency anywhere (the only alpha is flat
constants: popup 0.98, toast 0.96).

**3. Motion exists on exactly three widgets** — the segment pill, the toggle,
and card hover-lift — plus the hand-rolled toast rise and `drop_flash`.
`motion.py` (45 lines) is one frame-rate-independent exponential approach; no
easing curves, no springs. Everything else hard-cuts: mode transitions
(`_set_mode` swaps the whole screen in one frame — the pill slides, the
content pops), popups, modals, the palette, tooltips, all imgui hover states
(instantaneous full-saturation color swaps: `button_hovered = ACCENT @ 0.75`,
`theme.py:123-128`), sidebar width changes, splitter hover, collapsing
headers, library selection, the splash (3 s hold, then a cut,
`splash.py:38`). There is no reduce-motion setting.

**4. Structure fights simplicity in a few places.** Quit is the eleventh
segment of the navigation control (`main.py:2763-2770`) — mitigated by an
unconditional confirm, but a destructive action living inside navigation is a
mitigation, not a fix. The mode switch is eleven segments flat, with Manual
and Settings at the same visual weight as the five creative workspaces. The
2D generate pane is "twelve collapsible sections tall" by its own comment
(`panes/settings_2d.py:55-60`), ~15 combos plus a taxonomy grid; disclosure is
limited to `default_open=False` headers. And 2D's "platform detail" versus
3D's "Detail" is a naming collision papered over with tooltips
(`settings_2d.py:194-197`, `settings_3d.py:65-68`).

**5. The feedback substrate is half-wired.** `Invalid.field` is designed,
documented as "the UI highlights it" (`service/errors.py:19-24`), threaded
through `invalid_from` — and read by nothing in `studio/`. A field-level
refusal arrives as an undifferentiated red toast in the bottom-right corner,
far from the control that caused it. The partial compensation is the best
error moment in the app: `settings_2d.validate` renders each refusal as red
text directly above a disabled Generate button (`settings_2d.py:752-801`) —
summary-level, not field-level. There is no keyboard focus model: imgui nav is
deliberately off (`dialogs.py:208-210` records why — a focused button drew as
focused and did nothing), so focus rings exist only on three hand-built
surfaces (Home tiles, the confirm modal, the palette). There is no undo
outside Inker/Clay — the toast action vocabulary is `log`/`show`/`review`
(`widgets.py:986`) and never grew "Undo", so the library leans on soft-delete
plus confirms instead.

**6. Discovery is documentation-shaped.** The library filter's six-prefix
query syntax lives in a four-line tooltip (`panes/library.py:286-300`);
keyboard bindings live in a ~60-row unsearchable popup (`main.py:2888-3032`);
help is a manual mode. All good documentation — none of it learned in flow.

---

## Part II — Design principles

Six principles, each executed by named phases below. They are the Apple
sensibility restated as rules this codebase can enforce.

1. **Restraint.** One accent, and spacing does the work of lines. A separator
   is a failure of rhythm; the elevation ramp — not outlines, not rules — says
   what belongs together. (Phases 0, 2.)
2. **Hierarchy.** A real type ramp, used consistently: display type exists so
   a screen can have exactly one loud thing. Today nothing in the app can be
   louder than a section heading. (Phases 0, 2.)
3. **Depth is light.** Shadow is one physical story told everywhere — the
   same ramp for cards, popups, modals, toasts — and eventually real
   materials: what floats blurs what it floats over. (Phases 2, 5.)
4. **Motion is continuity.** Nothing pops. Everything that changes state
   moves through the change, briefly and with an ease — and a reduce-motion
   switch turns all of it off in one place. (Phases 0, 1, 5.)
5. **Forgiveness over interrogation.** Undo beats confirm wherever the
   machinery allows it; the soft-delete trash already exists, so most confirms
   are one "Undo" toast action away from being unnecessary. (Phase 3.)
6. **The common path is short.** The controls a first job needs are the ones
   on screen; power is one honest reveal away, never hidden. This app already
   believes greyed-beats-absent — disclosure is the same principle applied to
   density. (Phase 3.)

And the standing constraint over all six: **sensibility, not chrome.** No fake
macOS toolbar, no sheets pretending to be NSPanel, no SF Symbols lookalikes.
The existing shell — mode switch, three-column workspaces, command palette —
is the product's own shape; the phases refine how it looks, moves and
discloses, not what it is.

---

## Part III — The phased roadmap

Phases are ordered so each consumes the one before it. 0–4 are imgui-native
and cheap-to-moderate; 5 is the GPU tier, each item independently shippable.
Every phase ends green on the full suite and re-runs
`scripts/screenshot_modes.py` — this repo's definition of "somebody looked at
it" — at UI scale 1.0 *and* 1.5, both palettes (the 1.0-only trap is recorded
in `LEFTOVERS.md`'s appendix).

### Phase 0 — Widen the vocabulary — **shipped 2026-08-09**

*Tokens, fonts, motion primitives. Foundation for everything; nothing visible
changes except the first readers.*

**What shipped, and the one place it departed from this list.** `TEXT_HEADING`
20 / `TEXT_DISPLAY` 28 with `fonts.heading()`/`fonts.display()` (SemiBold at
28 — Inter Bold was not vendored, decided against the screenshot as this
section required); `SP_5`/`SP_6`; `DUR_SLOW`; `ease_out_cubic` /
`ease_in_out_cubic`; reduce-motion as an app-Settings toggle honoured centrally
in `motion.py`; the card fill threshold lerped.

**`RADIUS_L` and `SP_8` were deliberately *not* added.** Neither acquired a
reader: the radius's call sites are Phase 2's by this document's own Phase 2
bullet, and the two literals `SP_8` was to own (`landing.py`'s `sp(48)` icon
column, `empty_state`'s `sp(40)`) are *positions*, not gaps, and this section
already permits them to stay literal. Adding them anyway would have put two
names in `tokens.py` with nothing reading them, which is the exact state the
comment at `tokens.py:56-59` records deleting them from — and which
`test_studio_wiring.test_the_spacing_scale_carries_only_the_steps_in_use`
fails on. That test was rewritten in passing: it named `SP_6`/`SP_10`/
`RADIUS_L` and asserted their absence, freezing one afternoon's answer, and
now scans `studio/` for a reader of every `SP_*`/`RADIUS_*`/`TEXT_*`/`DUR_*`
token — the rule instead of its 2026 output. **They arrive in Phase 2.**

The Manual's chapter titles took `TEXT_HEADING`/`TEXT_TITLE` (replacing the
private literals 22 and 17) rather than display size; "the Manual gets a real
title size" stays Phase 2's, where the type pass is. Four primitives the phase
turned out to need are new in `motion.py` and are described in Phase 1's note.

- **Type ramp**: add `TEXT_HEADING = 20.0` and `TEXT_DISPLAY = 28.0` to
  `tokens.py`, with `fonts.heading()` and `fonts.display()` beside
  `small()`/`label()`/`title()` (`fonts.py:101-111`). Display weight is a
  taste call at implementation time: SemiBold at 28 px may carry it, or Inter
  Bold joins the three vendored faces (`resources/fonts/`) — decide against a
  screenshot, not in advance. First readers: the Home hero
  (`landing.py:211-212`) and the Manual chapter titles.
- **Spacing**: add `SP_5 = 20`, `SP_6 = 24`, `SP_8 = 32`. First readers: the
  literal `sp(20)`/`sp(40)`/`sp(48)` call sites in `landing.py:159,202,219`
  (40 and 48 become `SP_8`-derived or stay literal with a comment — the point
  is that *panes* stop inventing gaps, not that every number is a token).
  The rule from `tokens.py:56-59` applies: a token with no reader gets
  deleted, so each lands in the same commit as its readers.
- **Radius**: re-add `RADIUS_L = 10.0` — it died for having no readers; its
  readers arrive in Phase 2 (cards, modals, the palette window).
- **Motion**: extend `motion.py` with a small easing vocabulary —
  `ease_out_cubic` and `ease_in_out_cubic` alongside the exponential
  approach — plus `DUR_SLOW = 0.30` in `tokens.py` for mode-scale
  transitions. Add **reduce-motion** as an app-Settings toggle
  (`panes/app_settings.py`), honored *centrally* in `motion.py` (durations
  collapse to ~0) so every consumer inherits it, including the hand-rolled
  toast rise and `drop_flash` once they route through it.
- **Fix in passing**: the card fill threshold artifact — lerp
  `ELEV_1 → ELEV_2` by `lift` instead of branching at 0.5
  (`widgets.py:940`).

*Verification*: suite green; screenshot pass is pixel-identical everywhere
except the named first readers.

### Phase 1 — Motion everywhere — **shipped 2026-08-09**

*The imgui-native motion tier. The single highest feel-per-line phase: it is
what makes the app stop popping.*

**What shipped.** All six bullets: the mode crossfade, popover enter on the
confirm modal / text prompt / palette / shortcuts / diagnostics, hover
interpolation on `_glyph_button`, `primary_button` and `destructive_button`,
the sidebar width / splitter hover / library selection / splash, and the idle
clamp. Four notes a later session needs.

**`motion.py` grew four primitives, and each answers a defect rather than a
preference.** `peek` reads a value without advancing it — a button's colour is
needed to *draw* the button and its hover is known only afterwards, and the
"call `value` twice" alternative steps the easing twice a frame and towards the
wrong target first, which is what made `card`'s lift take visibly longer to
arrive than to leave. `animating()` is the idle clamp's wake condition and
counts only keys the **last frame touched**: without that stamp a card hovered
and then navigated away from keeps a target nothing will ask for again, and the
app never idles again either. `ease` is the one-shot counterpart for a move
with a known length (the veil, a popover), and under reduce-motion it reads
1.0 on its first frame — *arrived*, not *never appears*, which is the failure
a naive `duration = 0` has. `seed` states where an animated value starts, for
`layout.SIDEBAR_W`, which lives outside the module and changes outside a frame.

**The mode transition is a one-sided veil, not a two-buffer crossfade.** imgui
has one framebuffer; keeping the previous frame's is Phase 5's offscreen copy.
A full-viewport quad in the window background colour, easing `1 → 0`, on the
foreground draw list so it also covers the modals — and painted only, so a
transition can never eat a click. Measured: 13 frames to converge at
`DUR_BASE`, and mean frame time 2.35 ms while switching against 2.48 ms
steady, i.e. no cost the meter can see.

**Chips were on this list and have no hover state to interpolate** —
`widgets._chip` is a painted rect with no item behind it. Nothing was done to
them and nothing should be until they become interactive.

**`scripts/screenshot_modes.py` had to learn about motion, or the verification
bar in Part IV would have stopped working in the phase that most needed it.**
Its three warm-up frames are 50 ms of a 200 ms transition, so every capture
after this phase was a picture of a half-cleared veil; it now waits (bounded,
never a bare `while`) for `motion.animating()` to go false. It also gained
`--scale`, because Part IV asks for 1.0 *and* 1.5 and the script could only
capture whatever the monitor was.

- **Mode transitions.** A short content crossfade on `_set_mode`: the frame
  after a switch draws a full-viewport overlay on the foreground draw list
  whose alpha eases `1 → 0` over `DUR_BASE` (the pill already slides; now the
  content follows instead of teleporting). One implementation in `main.py`,
  zero per-pane work. Honors reduce-motion.
- **Popover enter.** Modals, the confirm dialog, the command palette and the
  diagnostics popup fade + rise ~6 px on their appearing frames (imgui alpha
  push + a cursor offset driven by `motion.value`). The palette and confirm
  already track their own "appearing" state; the pattern generalizes.
- **Hover interpolation.** Kill the instantaneous saturated color swaps: the
  hand-drawn widgets (`icon_button`, `primary_button`, `_glyph_button`,
  chips, cards) route hover color through `motion.value` keyed on the item id
  — the mechanism `segmented_control` already uses for text alpha
  (`widgets.py:711-715`). Stock-imgui controls (combos, sliders, headers)
  keep instant hover for now; they are replaced or wrapped in Phase 2, and
  soft hover arrives with the wrap. Chasing them via per-frame style pushes
  is not worth the frame-loop cost.
- **The rest of the hard cuts**: sidebar width change (`layout.set_sidebar`)
  eases over `DUR_BASE`; splitter hover fades in; the library selection bar
  fades/slides between cards; the splash fades out into the app instead of
  cutting (`splash.py`).
- **Frame-loop discipline.** All of this is per-frame arithmetic on floats in
  the existing `motion` dict — no allocations, no threads, nothing the
  "must never block" invariant would notice. The 12 FPS idle clamp
  (`main.py:53-57`) needs one amendment: an active animation counts as
  "something can change", so `motion.py` exposes whether any value is still
  approaching its target and the frame loop stays at full rate until none is.

*Verification*: F10 FPS meter shows no regression during transitions; suite
green; a screen recording of mode switches, palette open and hover is the
review artifact (screenshots cannot see motion — record, then look).

### Phase 2 — Visual refinement

*Consumes Phase 0's vocabulary. This is the pass a screenshot can see.*

- **Whitespace.** `PANE_PADDING` 5 → 10 (`layout.py:54`), `window_padding`
  (12,12) → (16,16), `item_spacing` vertical 7 → 8 (`theme.py:92-94`) —
  exact values settled against screenshots, but the direction is fixed:
  Apple-scale gutters, and room around every form. The known cost: denser
  panes (Inker toolbox, library sidebar) must be re-checked at 1.5 scale,
  where clipped-control bugs have shipped before.
- **Sections breathe instead of ruling.** `widgets.section`
  (`widgets.py:159-164`) drops its `separator()` — hierarchy comes from
  `SP_6` of space above and the label itself. Decide the label register once
  and apply everywhere: field labels stay small-caps (they earn their keep in
  dense forms), but *section* headings become sentence-case Medium at
  `TEXT_BODY`+ — the current everything-is-caps flattens the hierarchy the
  headings exist to create.
- **Display type lands.** Home hero at `TEXT_DISPLAY`; pane/inspector
  headers at `TEXT_HEADING`; the Manual gets a real title size. One loud
  thing per screen.
- **Iconography.** Replace the ASCII status glyphs (`theme.py:53-59`) and
  the literal `"(?)"` (`widgets.py:652-654`) with Lucide glyphs from the
  already-vendored set, preserving the double-encoding rationale (glyph +
  color, never hue alone, `widgets.py:191-195`). Toast close `x` becomes a
  proper icon button. The screenshot-legibility argument for ASCII is kept
  honest by choosing distinct glyph *shapes* per status, not just colors.
- **One shadow, told everywhere.** Extract the card's layered-rect shadow
  into a `widgets.shadow(rect, radius, elevation)` helper, upgrade it from
  two layers to four (wider, lower-alpha steps read as blur even without
  one), and apply it at three elevations: resting (cards), raised (popups,
  palette, toasts), overlay (modals). `RADIUS_L` arrives with it: cards,
  modals and the palette go to 10; controls stay at 4–6. This is the
  pre-blur depth story; Phase 5 swaps the recipe's insides for a real
  blurred atlas without touching call sites.
- **The mode switch grows hierarchy, keeps its shape.** Quit leaves the
  segmented control (`main.py:2763-2770`) for the top-right strip beside
  the health dot and `?` — a destructive action stops living inside
  navigation, and the confirm stays. The remaining ten segments gain one
  visual grouping gap between *places* (Home, Manual, Settings) and
  *workspaces* (2D, 3D, Inker, Clay, Review, Plotter, Packwright) — a
  spacing change inside `segmented_control`'s layout, not a structural one;
  Alt+N positions are untouched.

*Verification*: full screenshot pass, both scales, both palettes, diffed
against pre-phase captures; the forms-and-layout floor tests
(`tests/test_forms_and_layout.py`) green; the manual's screenshots (if any
chapter embeds one) re-taken.

### Phase 3 — Simplicity and disclosure

*The structural half of "feels effortless". Larger items; each is its own
plan when executed.*

- **Field-level errors, end to end.** Panes catch `Invalid`, remember
  `(field, message)` for the frame, and the named control draws
  `widgets.ring` plus the message inline beside it — the promise
  `service/errors.py:19-24` already makes. The summary-above-Generate block
  (`settings_2d.py:752-801`) stays as the aggregate view; the ring is the
  pointer. One shared helper (`widgets.field_error`) so all panes say it the
  same way. This is the highest-value item in the phase: the plumbing exists
  on both sides and only the last inch is missing.
- **The 2D form gets a common path.** Today: twelve sections. Target: the
  first screen shows prompt, preset, output and Run; the three taxonomy
  groups (Subject/Style/Surface, `settings_2d.py:88-96`) and Advanced sit
  behind one honest "More options" reveal that remembers its state. Presets
  become the primary interface (they already fill visible fields and show
  "Custom" on divergence, `settings_2d.py:210-232`); the taxonomy becomes
  the refinement. Nothing is removed — this is disclosure, not deletion, and
  the findings-hints under fields survive wherever their field lands.
- **Naming.** "Platform detail" (2D, a prompt fragment) and "Detail" (3D,
  geometry resolution) get real, different names — e.g. "Era styling" and
  "Mesh resolution" — ending the two-tooltips-apologizing situation
  (`settings_2d.py:194-197`, `settings_3d.py:65-68`).
- **A scoped focus model.** Not imgui nav (recorded broken:
  `dialogs.py:208-210`) — the Home-tiles pattern generalized: a per-pane tab
  order over the hand-built widgets, `widgets.ring` as the single focus
  visual, Tab/Shift-Tab to traverse, Enter to activate. Scope it to the two
  generate panes and the confirm dialogs first; a full-app focus pass is its
  own later project.
- **Undo as forgiveness.** The toast action vocabulary grows `undo`:
  trashing an asset raises "Moved to trash — Undo" (restore already exists,
  `library.py:930-960`), and the card's Delete keeps its no-confirm behavior
  with a better safety net. Candidates after that: sweep-unit hiding,
  filter clears. Confirms stay only where the act is irreversible (prune,
  empty trash) — which is exactly the rule the codebase already states.

*Verification*: a deliberate bad submit shows the ring on the right control;
the 2D pane's common path fits without scrolling at default window size and
1.0/1.5 scale; keyboard-only job submission works in both generate panes;
Undo restores within the toast's dwell.

### Phase 4 — Small moments

- **First-run orientation.** Home already solves "models missing"
  (`landing.py:237-284`). Add the other half: a one-time, dismissible
  orientation card on Home ("A prompt becomes a reference image; a reference
  becomes a mesh — start with New 2D Image") that never returns once
  dismissed. And a "Continue" tile that appears when a recent document or
  selection exists — Home deliberately remembers no *mode*, but resuming
  *work* is a different promise and the library already knows the most
  recent asset.
- **In-flow discovery.** The library filter box gets clickable prefix chips
  (`tag:` `status:` `kind:` …) under focus instead of a four-line tooltip
  (`library.py:286-300`); the shortcuts popup gets a filter box (it is ~60
  rows, `main.py:2888-3032`) — the palette's subsequence matcher
  (`palette.py:60-100`) is right there to reuse.
- **The splash earns its three seconds.** Fade the logo in, fade the whole
  splash out into Home (Phase 1's machinery); replace "Starting Warlock
  Studio..." with the doctor's live progress line so the hold communicates
  instead of stalling (`splash.py:38-46` holds ≥3 s regardless — spend it).
- **Progress card polish.** It is "the one piece of UI the whole app is
  judged by" (`overlay.py:243-305`, its own docstring): give it the Phase 2
  shadow treatment, `RADIUS_L`, and eased appearance/disappearance rather
  than popping over the viewport.
- **Empty-state pass.** They are structurally everywhere
  (`overlay.py:348-372`, `library.py:150-166`); with display type and the
  new spacing they become designed moments instead of centered muted text.

*Verification*: fresh-profile launch shows the orientation once and never
again; screenshot pass; the manual's coverage test still passes (new panes or
controls acquired along the way need their `(?)` or exemption).

### Phase 5 — The GPU tier

*The custom-rendering ceiling. Each item is independently shippable, each gets
a cost estimate and a "worth it?" gate against a screenshot/recording before
it merges. Ordered by payoff. All of it lives behind `imgui_backend.py`'s
existing moderngl bridge — the app already owns its GL context and renderer
(`studio/imgui_backend.py`), which is what makes this tier possible at all.*

1. **Real soft shadows** (payoff: high; cost: low-moderate). Generate a
   9-slice Gaussian-blurred shadow atlas once at startup (one texture per
   radius/elevation pair, Pillow or numpy — no runtime blur), register it
   with the backend, and `widgets.shadow` draws nine textured quads instead
   of layered rects. Call sites unchanged from Phase 2. This alone closes
   most of the visible depth gap.
2. **Vibrancy for what floats** (payoff: high; cost: moderate). The frame
   already renders to the default framebuffer in one pass; add an offscreen
   copy of the composed frame *before* the overlay layer, downsample + blur
   it (two-pass separable Gaussian in moderngl, quarter-res), and the
   palette, modals and toasts sample it as their background with a tint —
   translucent materials, the real thing. Gate: measure frame time on the
   idle clamp; the copy+blur runs only on frames where a floating surface is
   visible.
3. **Spring motion** (payoff: medium; cost: low). A critically-damped spring
   in `motion.py` beside the exponential — velocity-carrying, so the segment
   pill and modal enter get the characteristic Apple overshoot-and-settle.
   Reduce-motion collapses it like everything else.
4. **Squircle corners** (payoff: low-medium; cost: moderate). A superellipse
   SDF fragment shader for card/modal backgrounds (textured-quad path
   through the backend, or a custom draw callback). Genuine continuous-
   curvature corners at `RADIUS_L`+; controls at 4–6 px stay circular —
   below ~8 px the difference is invisible and not worth the pipeline.

*Verification per item*: before/after recording; frame-time on a 4K window at
1.5 scale within budget (no regression to the idle clamp, no hitch during
mode transitions); suite green; each item toggleable off (a config flag per
item while it stabilizes, folded away once trusted).

---

## Part IV — Verification and measurement

- **`scripts/screenshot_modes.py` is the bar.** Every phase re-runs it —
  real `App`, real fonts, frames off the framebuffer — at UI scale 1.0
  **and** 1.5, both palettes, before/after. The 1.0-only trap has shipped
  three real defects before (`LEFTOVERS.md` appendix); it does not get a
  fourth.
- **Motion cannot be screenshotted.** Phases 1 and 5 add a short screen
  recording of the canonical moments (mode switch, palette open, hover,
  toast lifecycle) as their review artifact.
- **The frame loop stays innocent.** All animation is per-frame float
  arithmetic on existing state — no allocations in the loop, no threads, no
  GL work outside the frame thread. The idle-clamp amendment (Phase 1) is
  the one deliberate frame-loop change and it is a wake condition, not work.
- **Tokens ship with readers** (`tokens.py:56-59`'s own rule), and the suite
  stays green both ways (`WARLOCK_NATIVE=0` too).
- **Accessibility is not traded away.** Reduce-motion (Phase 0) covers
  vestibular sensitivity; the glyph+color double-encoding survives the
  iconography pass; the focus model (Phase 3) is a strict accessibility gain.
