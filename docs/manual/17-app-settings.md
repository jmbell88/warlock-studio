# App settings

**Settings** in the mode switch holds the handful of preferences that are the app's rather than a
job's. They are stored in `studio_settings.json` beside everything else the app remembers, and none
of them need a variable set before launch.

**Interface.** *UI scale* is a multiplier on top of whatever your monitor's own DPI scaling already
is, from 0.5× to 2×. On a display that is already heavily scaled the slider stops short of 2× and
says so, because the combined scale is capped — the control only offers zooms it can actually
apply. It takes effect as you drag it, and the font atlas is re-baked when you let go — between
frames rather than during one, since a rebuild invalidates every font handle a half-drawn frame is
holding. Nothing needs a restart.

*Theme* switches the whole palette, including the viewport background, and takes effect at once. The
light palette keeps the same *roles* as the dark one rather than inverting its numbers: a panel is
still the surface a form sits on, and the elevation steps still read as raised — which on a light
ground means slightly darker rather than lighter. *Show frame rate* is the same toggle as `F10`.

**Layout.** *Sidebar width* offers narrow, default and wide (260, 300 and 360 px). Three named sizes
rather than a drag: a form has a width that reads well, and what a free drag bought was a way to make
the app look broken — but one number cannot suit a 1600-wide window and a 5120 one. *Reset pane
sizes* puts the split between the inspector and the library — both on the right sidebar — back to its
default, undoing any dragging of that divider. *Reset collapsed sections* re-opens every section that
has been collapsed anywhere in the app.

**Configuration.** *Effective configuration* lists every environment variable the app reads and what
this process resolved it to, with the ones actually set by the environment first and named by their
variable. It is the same table `warlock doctor` prints and the same one behind the health dot's
diagnostics popup, and *Copy as text* puts it on the clipboard for a bug report. It is read-only:
every entry is consumed at import time, so an editable version would have to say "restart to apply"
under every field.

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
[Model weights](15-installation.md#model-weights) and
[Adding an image model](21-extending.md#adding-an-image-model).

Not everything the app remembers has a control in this pane. `studio_settings.json` also holds your
saved profiles and settings presets, the sidebar's internal split, and the pixel-art export
preferences — the
size and palette set in an asset's [Pixel art](03-generating-references.md#pixel-art) section, which
are the app's preferences rather than any one job's and so apply to whichever asset you look at
next.
