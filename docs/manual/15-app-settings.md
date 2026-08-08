# App settings

**Settings** in the mode switch holds the handful of preferences that are the app's rather than a
job's. They are stored in `studio_settings.json` beside everything else the app remembers, and none
of them need a variable set before launch.

**Interface.** *UI scale* is a multiplier on top of whatever your monitor's own DPI scaling already
is, from 0.5× to 2×. On a display that is already heavily scaled the slider stops short of 2× and
says so, because the combined scale is capped — the control only offers zooms it can actually
apply. It takes effect as you drag it, but the font atlas is baked once at startup, so
text only becomes properly crisp at the new size after a restart — everything is drawn at the right
size immediately either way. *Show frame rate* is the same toggle as `F10`.

**Layout.** *Reset pane sizes* puts the split between the inspector and the library — both now on
the right sidebar — back to its default, undoing any dragging of that divider. The sidebars
themselves are a fixed 300 px and are not draggable. *Reset collapsed sections* re-opens every
section that has been collapsed anywhere in the app.

**Models.** Every model the app knows about — image models, style LoRAs, the conditioning adapters,
and the matting, pose and measurement models — with a tick beside the ones whose weights are on disk
and a **Download** button beside the ones that are missing, plus whether rigging is available. It is
the same information the startup diagnostics report, in a place you can look at without opening the
log. Tick several rows and *Download selected* fetches them together; four of the image models share
one set of SDXL 1.0 weights, and picking all four downloads them once.

Downloading does not make the app itself online. The button starts a separate process that fetches
one repository and exits, into a staging folder beside the destination that is only moved into place
if the fetch succeeded — so a download interrupted halfway leaves nothing behind rather than a model
directory that looks finished. Free disk is checked against the whole selection first, and the whole
selection is refused if it will not fit. Everything is still equally installable by hand — see
[Model weights](13-installation.md#model-weights) and
[Adding an image model](19-extending.md#adding-an-image-model).

Not everything the app remembers has a control in this pane. `studio_settings.json` also holds your
saved profiles and settings presets, the sidebar's internal split, and the pixel-art export
preferences — the
size and palette set in an asset's [Pixel art](03-generating-references.md#pixel-art) section, which
are the app's preferences rather than any one job's and so apply to whichever asset you look at
next.
