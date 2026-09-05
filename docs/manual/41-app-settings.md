# App settings

**Settings**, in the rail's footer, holds the handful of preferences that are the app's rather than a
job's. They are stored in `studio_settings.json` beside everything else the app remembers, and none
of them need a variable set before launch.

The pane is one centred column with six categories across the top — **Appearance**, **Models**,
**Packs**, **Storage**, **Health** and **Advanced**. The column keeps its width however wide the
window is: everything here is a short labelled row, and a form stretched across a 5K display leaves
the label and the control it belongs to at opposite ends of the desk. Which category you last had
open is not remembered between launches — it is where you were, not something you chose.

## Appearance

*UI scale* is a multiplier on top of whatever your monitor's own DPI scaling already
is, from 0.5× to 2×. On a display that is already heavily scaled the slider stops short of 2× and
says so, because the combined scale is capped — the control only offers zooms it can actually
apply. It takes effect as you drag it, and the font atlas is re-baked when you let go — between
frames rather than during one, since a rebuild invalidates every font handle a half-drawn frame is
holding. Nothing needs a restart.

*Theme* switches the whole palette and takes effect at once. There are three. *Dark* is the
default. *Light* keeps the same *roles* as the dark one rather than inverting its numbers: a panel is
still the surface a form sits on, and the elevation steps still read as raised — which on a light
ground means slightly darker rather than lighter. *Pixel* is a second dark palette in the register
the pixel workspaces belong to: warm neutral greys against a near-black canvas surround, with amber
where the other two spend indigo. It keeps dark's direction — a step away from the floor still reads
lighter — and differs from it in temperature, so the Inker, the Plotter and Packwright sit in
something closer to the tools their work comes from. It is not a copy of any one program's chrome:
those are typically grey on grey, and every colour here has to clear a measured contrast bar. The
one thing **no** theme repaints is the 3D
viewport's background, which stays the dark `#0F1014` under all three: that colour is a property
of the renderer rather than of the palette, and making it follow the theme means threading a colour
into the render-skip key so a theme switch triggers a redraw. It is a known gap, deliberately left
open. *Show frame rate* is the same toggle as `F10`.

*System resources* puts a live reading of VRAM, RAM and CPU at the right end of
the status line, in every mode. It is there because the app already forces the question on you: a
generation is refused at the door when there is not enough VRAM free, and re-checked when it is
about to run, and until now nothing on screen said what the number those refusals are about
actually was. The VRAM figure is read from the driver, so it counts every process on the card and
not just this one. It updates once a second, costs about a twentieth of a millisecond to take, and
is the first thing dropped when the window gets too narrow to hold both it and the workspace name.
A card the driver does not report on simply leaves VRAM out rather than showing a placeholder.

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
and the matting, pose and measurement models — as a table of four columns: **Model**, **Size**,
**Description** and **Actions**. A tick beside the name means the weights are on disk; a hollow mark
with a checkbox means they are not, and **Install** fetches them. It is the same information the
startup diagnostics report, in a place you can look at without opening the log. Tick several rows
and *Download selected* fetches them together; four of the image models share one set of SDXL 1.0
weights, and picking all four downloads them once.

The **Description** column is one sentence saying what the model is *for* — which is the question a
list of thirty names cannot answer, and the reason picking one used to mean reading `docs/MODELS.md`
beside the app. Hovering the name gives the longer version, along with the repository the weights
come from, what the model costs on the card, and, for a style LoRA, the trigger words it was trained
on — the words the app prepends to your prompt whenever that adapter is selected. A model this build
has no download recipe for stays in the table as an inert **Unavailable** rather than disappearing:
a row that says it cannot be fetched is more use than a row that is not there.

A
running download shows its rate and an estimate of the time left, and carries its own **Cancel**
beside the bar. Cancelling installs nothing: the fetch stages beside the destination and only moves
into place once it has finished, and the staging a cancelled fetch leaves is swept the next time
this pane is opened, which says so when it reclaims anything.

Under each image model's size, once the app has measured your card, sits a fit note: nothing at all
when the model runs comfortably, **tight fit** when it will load but cannot stay resident beside the
reconstruction engine — so every 3D job pays a stop and restart for it — and **won't fit** when the
checkpoint alone is larger than the VRAM budget. Both carry the measured figure on hover. A
**Recommended for this GPU** line under the list names the best base model your card can actually
hold. All of it is skipped on a host with no measurable GPU: an unknown budget is not a shortfall.

**Deleting a model.** A downloaded row carries a **Delete** button, and hovering it says how much
removing would actually free. That figure is usually smaller than the download was, and deliberately: the four SDXL 1.0
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
[Model weights](39-installation.md#model-weights) and
[Adding an image model](45-extending.md#adding-an-image-model).

### Your style LoRAs

Under the model tables is the one place a style comes from somewhere other than a download. Two
buttons, one list.

**Import a LoRA file...** takes any `.safetensors` adapter — one you trained elsewhere, one from a
model site — and asks for the four things the picker and the loader read: a name, the trigger
words it was trained with, its working weight, and which family it fits (SDXL or FLUX.2 klein; an
adapter never loads onto the other). Tick *Licensed for commercial use* only if you know that it
is; the picker shows the answer beside the style. The file is copied under `~/.warlock/models/
loras/` and appears in every LoRA picker immediately.

**Train from a folder...** trains one here, on your card, from your own art. Point it at a folder
of 3 to 100 images in the style — no captions needed — give the style a name and a trigger
phrase, and it queues as a job like any other. It runs on an undistilled SDXL checkpoint (the
Quality recipe's), takes the whole card for the duration (the reconstruction engine and the image
model are unloaded first and come back on the next job), and about 800 steps is half an hour on a
fast card. When it finishes the style is registered exactly as an imported one, at weight 1.0, with
your trigger words. Your images never leave the machine.

**Remove** beside an imported or trained style deletes its file and its entry. Built-in styles are
not listed here and cannot be removed.

## Packs

Models are weights; **packs** are the code that reads them. The installed build carries the app, its
renderer and the reconstruction engine — around a gigabyte — and the three heavy dependency sets
arrive here, chosen:

| Pack | What it turns on |
| --- | --- |
| **Image generation** | Create's reference stage, host-side matting, candidate ranking |
| **Rigging** | Poser and Troupe |
| **Music generation** | Muse, and stem separation |

Each row says what the pack is for, which workspaces it unlocks, and what it costs — in two figures,
because they land in two places: the download goes to a wheel cache under `~/.warlock/packs`, and
the unpacked packages go into the app's own runtime, which on a per-user install is often another
drive. Both are checked for free space before anything starts, and the whole pack is refused if
either will not hold it.

**Install** downloads and installs it, and the two phases are visible on the one bar. While it is
downloading, **Cancel** stops it and costs you nothing: every wheel is verified against the digest
this build pins and only then moved into place, so a stopped install leaves a part-file that the
next attempt resumes from. Once the packages start going in, the Cancel button goes away — that
half writes into the runtime the app is running out of, and interrupting it is the one thing that
could leave the installation half-made. It is a separate process throughout; this one stays offline
and never becomes able to download anything.

When it finishes, the app re-runs every startup check and the workspaces the pack unlocks come to
life without a restart. If something still cannot be imported the toast says so and asks for a
restart, rather than leaving you with a mode that is greyed out for no stated reason.

A pack cannot be removed from here. Uninstalling torch out from under a running application is not
the same act as deleting a folder of weights, and the way to get the disk back is to reinstall the
base app.

**A greyed workspace sends you here first when a pack is what it is waiting for.** Create, Poser,
Troupe and Muse each need two separate things — the pack, which is the code, and the model weights,
which are what the code reads — and the pack has to come first, because weights with nothing to load
them do nothing at all. So on a fresh install, clicking one of those modes opens this page and the
tooltip names the pack by its own size; once it is installed, clicking the mode again opens
**Models** instead, with the rows that mode needs already ticked.

On a source checkout there is nothing to install: no build ever generated a pack manifest, and each
row prints the `uv sync --extra ...` line that does the same job. See
[Installation](39-installation.md#if-you-installed-rather-than-cloned).

## Storage

Three figures and two buttons. The figures are how many job directories exist and what they occupy,
what is sitting in the [trash](36-library-and-jobs.md#the-trash) waiting to be emptied, and how much
disk the downloaded model weights are actually using. All three are measured on a background thread
and the last answer is drawn until a new one arrives, so none of them walks the disk while you are
looking at something else. The first two are the same measurements the library reports, not a second
opinion about the same directories; the third is a real measurement of the model store rather than
the sizes the Models list declares, which are approximations kept for the progress bar and the
free-disk check.

**Prune...** deletes everything but the newest N assets from disk, after a confirm that carries the
count — N is yours to choose and it starts at twenty every time it is asked. Running jobs are never
touched, and neither is anything you accepted or labelled.

**Clean library...** is the other end of the same scale: every asset, trashed or not, including the
accepted ones and the labelled images the quality judge and the tier checks are measured against.
The verdict rows survive; the pixels behind them do not. Your pose library, Inker
autosaves and settings are kept, and it refuses outright while anything is queued or running.

Both used to sit at the foot of the library, under the list of assets, which is the one place where
"clean library" reads as an action on the assets you can see rather than on all of them.
[Library and jobs](36-library-and-jobs.md#storage-and-pruning) has the longer account of what
each one deletes and why prune removes from disk rather than trashing.

## Health

Every check `warlock doctor` runs, as a list: a tick or a cross, the check's name, and the one line
of detail saying what it found. Green is passing, amber is a warning, red is fatal. The line above
the list counts the failures, which is the same number Home's health row shows — clicking that row
opens this page.

This is the only place in the app that names a *non-fatal* failure. A fatal one also raises the
error banner across the top of the window, but a style LoRA whose file has been moved, or Blender
missing so rigging is unavailable, is otherwise a count and a tooltip.

**Copy details** puts the whole list on the clipboard in the form `FAIL name: detail`, which is what
to paste into a bug report. **Run checks again** re-runs every probe including the slow ones the
startup path defers — worth pressing after installing something a row complained about, because the
static checks are otherwise only recomputed at launch. **Troubleshooting** opens
[chapter 12](42-troubleshooting.md), which is where a check that keeps failing after its remedy is
covered.

**Dismissed** appears only when you have dismissed something from the error banner. The banner's
Dismiss moves a message here rather than deleting it: a worker that died is reported through the
banner and through no check row at all, so this is the only copy.

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
variable. It is the same table `warlock doctor` prints and the same one behind the status bar's
Issues popup, and *Copy as text* puts it on the clipboard for a bug report. It is read-only:
every entry is consumed at import time, so an editable version would have to say "restart to apply"
under every field.

**If Warlock crashes**, it says so on the way out rather than simply vanishing: a small dialog
reports that something went wrong, tells you whether there is unsaved work waiting to be offered
back on the next launch, and asks whether to open the folder your log is in. Answering no costs
nothing — the log is written either way, and the recovery offer does not depend on it.

Not everything the app remembers has a control in this pane. `studio_settings.json` also holds your
the sidebar's internal split, and the pixel-art export
preferences — the
size and palette set in an asset's [Pixel art](22-generating-references.md#pixel-art) section, which
are the app's preferences rather than any one job's and so apply to whichever asset you look at
next.
