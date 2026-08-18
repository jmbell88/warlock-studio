# App settings

**Settings**, in the rail's footer, holds the handful of preferences that are the app's rather than a
job's. They are stored in `studio_settings.json` beside everything else the app remembers, and none
of them need a variable set before launch.

The pane is one centred column with four categories across the top — **Appearance**, **Models**,
**Storage** and **Advanced**. The column keeps its width however wide the window is: everything here
is a short labelled row, and a form stretched across a 5K display leaves the label and the control it
belongs to at opposite ends of the desk. Which category you last had open is not remembered between
launches — it is where you were, not something you chose.

## Appearance

*UI scale* is a multiplier on top of whatever your monitor's own DPI scaling already
is, from 0.5× to 2×. On a display that is already heavily scaled the slider stops short of 2× and
says so, because the combined scale is capped — the control only offers zooms it can actually
apply. It takes effect as you drag it, and the font atlas is re-baked when you let go — between
frames rather than during one, since a rebuild invalidates every font handle a half-drawn frame is
holding. Nothing needs a restart.

*Theme* switches the whole palette and takes effect at once. The
light palette keeps the same *roles* as the dark one rather than inverting its numbers: a panel is
still the surface a form sits on, and the elevation steps still read as raised — which on a light
ground means slightly darker rather than lighter. The one thing it does **not** repaint is the 3D
viewport's background, which stays the dark `#0F1014` under either theme: that colour is a property
of the renderer rather than of the palette, and making it follow the theme means threading a colour
into the render-skip key so a theme switch triggers a redraw. It is a known gap, deliberately left
open. *Show frame rate* is the same toggle as `F10`.

*Reduce motion* turns off every animation in the app at once — the mode transition, hover, the
rail's sliding selection and its expand, a popover's rise. Nothing disappears and nothing behaves differently: things
arrive in place instead of travelling there, and the few effects that are a *fade* rather than a
move (a toast's opacity, the acknowledgement flash when a file is dropped) keep their timing and
lose their ramp.

There is no **Effects** section any more. Four switches used to sit here — soft shadows, translucent
panels, spring motion and continuous corners — while those rendering features were new. Each of them
already falls back on its own when the graphics side cannot provide it (concentric outlines, solid
fills, an eased stop, a circular arc), which is what the switches were really insuring against, so
they were removed rather than left as four settings that only ever meant "pretend the graphics card
failed". *Reduce motion* above is unaffected: it is an accessibility setting, not a rendering tier,
and it still turns off spring motion along with everything else that moves.

## Models

Every model the app knows about — image models, style LoRAs, the conditioning adapters,
and the matting, pose and measurement models — with a tick beside the ones whose weights are on disk
and a **Download** button beside the ones that are missing, plus whether rigging is available. It is
the same information the startup diagnostics report, in a place you can look at without opening the
log. Tick several rows and *Download selected* fetches them together; four of the image models share
one set of SDXL 1.0 weights, and picking all four downloads them once.

Beside each image model, once the app has measured your card, sits a fit note: nothing at all when
the model runs comfortably, **tight fit** when it will load but cannot stay resident beside the
reconstruction engine — so every 3D job pays a stop and restart for it — and **won't fit this GPU**
when the checkpoint alone is larger than the VRAM budget. A **Recommended for this GPU** line under
the list names the best base model your card can actually hold. All of it is skipped on a host with
no measurable GPU: an unknown budget is not a shortfall.

**Removing a model.** A downloaded row carries a **Remove** button and the amount it would actually
free. That figure is usually smaller than the download was, and deliberately: the four SDXL 1.0
recipes share one 7 GB checkpoint, so removing *SDXL 1.0 + Hyper-SD* deletes only its own 0.8 GB
adapter and leaves the weights the other three are still standing on. The checkpoint goes when the
last model using it does. A recipe with no files of its own — *SDXL 1.0 (full CFG)* is the plain
case — has nothing to remove and shows no button at all.

Removal asks first, and refuses in three cases rather than doing something surprising. It will not
run while any job is queued or running. It will not touch a directory `WARLOCK_T2I_DIR` points at:
that is a folder you pointed the app at, not one it downloaded. And it never deletes anything
outside the model root. Removing the *default* base model **is** allowed, and the consequence is
that generation refuses every job that does not name another model until you reinstall one or pick a
different base — the refusal says so and links back to this pane.

Downloading does not make the app itself online. The button starts a separate process that fetches
one repository and exits, into a staging folder beside the destination that is only moved into place
if the fetch succeeded — so a download interrupted halfway leaves nothing behind rather than a model
directory that looks finished. Free disk is checked against the whole selection first, and the whole
selection is refused if it will not fit. Everything is still equally installable by hand — see
[Model weights](17-installation.md#model-weights) and
[Adding an image model](23-extending.md#adding-an-image-model).

## Storage

Two figures and two buttons. The figures are how many job directories exist and what they occupy,
and what is sitting in the [trash](13-library-and-jobs.md#the-trash) waiting to be emptied. Both are
measured on a background thread and the last answer is drawn until a new one arrives, so neither
walks the disk while you are looking at something else. They are the same measurements the library
reports, not a second opinion about the same directories.

**Prune...** deletes everything but the newest N assets from disk, after a confirm that carries the
count — N is yours to choose and it starts at twenty every time it is asked. Running jobs are never
touched, and neither is anything you accepted or labelled.

**Clean library...** is the other end of the same scale: every asset, trashed or not, including the
accepted ones and the labelled images the quality judge and the tier checks are measured against.
The verdict rows survive; the pixels behind them do not. Your pose library, style profiles, Inker
autosaves and settings are kept, and it refuses outright while anything is queued or running.

Both used to sit at the foot of the library, under the list of assets, which is the one place where
"clean library" reads as an action on the assets you can see rather than on all of them.
[Library and jobs](13-library-and-jobs.md#storage-and-pruning) has the longer account of what
each one deletes and why prune removes from disk rather than trashing.

## Advanced

**Layout.** *Sidebar width* offers narrow, default and wide (260, 300 and 360 px). Three named sizes
rather than a drag: a form has a width that reads well, and what a free drag bought was a way to make
the app look broken — but one number cannot suit a 1600-wide window and a 5120 one. *Reset pane
sizes* puts the split between the inspector and the library — both on the right sidebar — back to its
default, undoing any dragging of that divider. *Reset collapsed sections* re-opens every section that
has been collapsed anywhere in the app.

The sidebars narrow on their own when there is not room for the width you picked — at a high UI scale
in a small window, three columns at their full size want more pixels than the window can be shrunk
to. They give way in that order: both sidebars narrow first, down to a width a form is still usable
at, and only then does the centre pane start to shrink. Nothing is pushed off the edge, so the
inspector stays reachable at any scale the slider offers.

Moving the window to a monitor with different scaling re-reads the new display and rebuilds the
interface at its size, fonts included — you do not have to restart. Your *UI scale* is applied fresh
against the new monitor rather than carried across as a number of pixels, so a zoom that had to be
capped on one display is offered in full again on a display with room for it.

**Configuration.** *Effective configuration* lists every environment variable the app reads and what
this process resolved it to, with the ones actually set by the environment first and named by their
variable. It is the same table `warlock doctor` prints and the same one behind the health dot's
Issues popup, and *Copy as text* puts it on the clipboard for a bug report. It is read-only:
every entry is consumed at import time, so an editable version would have to say "restart to apply"
under every field.

**If Warlock crashes**, it says so on the way out rather than simply vanishing: a small dialog
reports that something went wrong, tells you whether there is unsaved work waiting to be offered
back on the next launch, and asks whether to open the folder your log is in. Answering no costs
nothing — the log is written either way, and the recovery offer does not depend on it.

Not everything the app remembers has a control in this pane. `studio_settings.json` also holds your
saved profiles, the sidebar's internal split, and the pixel-art export
preferences — the
size and palette set in an asset's [Pixel art](03-generating-references.md#pixel-art) section, which
are the app's preferences rather than any one job's and so apply to whichever asset you look at
next.
