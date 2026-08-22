# Finding your work again

The last chapter made something. This one is about where it went, how to get back to it, and the
four different things the app means by "delete" — which is worth learning before you need it rather
than after.

Nothing here needs a GPU or any model weights. Every screen in this chapter works on a bare install.

## Everything is a job

Warlock has no save button for generated work, because there is nothing to save. The moment you
press Generate, a **job** exists: a row in the library, a directory on disk, and a status that
changes as the work runs. It is already yours before it has finished.

That is why the library is also the queue. There is no separate "queue" screen to switch to — a
queued job and a finished one are the same kind of row in the same list, one with a progress bar and
one with a picture. Filter the list by status and you are looking at the queue.

A job's life is short and has five possible ends:

| Status | Meaning |
| --- | --- |
| **queued** | Waiting its turn. The card shows its position in the queue. |
| **running** | Working. The card shows a live progress bar. |
| **done** | Finished, with files on disk. |
| **error** | Failed. The row says why. |
| **cancelled** | You stopped it. |

## Home

Home is the screen the app opens on, and it is a chooser rather than a dashboard. Three parts of it
are worth knowing.

**Resume** is a grid of thumbnails of your recent work — both documents you had open in the editors
and finished assets from the library, newest first. Up and Down walk it and Enter opens.

**Unsaved work** appears only if a previous session ended badly. Warlock autosaves an open document
after a couple of minutes of unsaved changes, and if the app did not shut down cleanly those copies
are offered here as one row per document, each with its own Recover button. It is deliberately not a
single all-or-nothing question — a session that crashed with one document worth keeping and nine
worth discarding should not force one answer for all ten. Declining keeps the files; nothing ages
out from under you.

**The status line** is one quiet row combining health, the queue and anything waiting to be reviewed.
Each part of it is a link to the screen that answers it.

## The library

**Library** in the rail opens the full-window version: a filter rail on the left, a grid of
thumbnails, and an inspector that appears when something is selected. The same list, narrower, is
the right-hand column of Create.

**Filtering** is one text box with a small vocabulary of prefixes — `tag:`, `status:`, `kind:`,
`stage:`, `id:`, `name:` — and clickable chips that insert them for you, so you can discover the
syntax by using it rather than by reading about it.

One thing about that box will bite you eventually, and the pane does say so: **the list is a window
onto your history, not all of it.** It holds the newest N of M jobs, and a filter searches what is
loaded. When a filter is active and there is more history behind it, the pane tells you, and
**Load older** widens the window.

**Sorting** offers newest, name, kind, duration, size on disk, and score. Under *newest* the list
also grows date headings — Today, Yesterday, This week, then by month.

**Opening** an asset is a click, or Enter on the selection. Where it lands depends on what it is: a
reference opens on the Reference stage, a mesh on Mesh, a drawing in Inker, a tile map in Plotter, a
character sheet in Troupe. You do not choose the destination and you should not have to.

## The four kinds of delete

This is the part to read twice. The app has four destructive actions and they make different
promises.

**Delete**, on a card or in the overflow menu, is a **trash** operation. It is reversible. The asset
moves to the trash and stays there until you say otherwise. There is no "are you sure?" because the
trash *is* the confirmation — asking twice for a reversible action trains people to click through
questions that matter.

**Delete permanently**, available only from inside the trash, is irreversible and asks first.

**Prune**, in Settings under Storage, keeps the newest N jobs and deletes the rest from disk — and
note *deleted*, not trashed, because a prune that moved two hundred jobs into the trash would free
no space at all while reporting that it had. Its dialogue makes a promise:

```text
Everything but the newest N jobs is deleted from disk. Running jobs are kept,
and so is anything you accepted or labelled. This cannot be undone.
```

**Clean library**, also under Storage, is the one that breaks that promise, and it says so plainly:

```text
Every asset is removed from disk, trashed or not -- including ones you accepted
and images you labelled, which is what the quality judge and the tier checks are
measured against. The verdict rows survive; the pixels behind them do not.
```

The asymmetry is the whole point. Prune protects your judged work; clean does not. If you have been
grading meshes or labelling references, that judged material is training data and evidence, and
clean is the one button that throws it away. Your pose library, style anchors and settings survive
both.

## Getting an asset somewhere else

Right-click any card for the overflow menu. Besides the obvious — copy the job id, open the folder,
rename — it is the way an asset crosses into another part of the app:

- **Open in Inker** to paint over a reference.
- **Open in Clay** to edit a mesh. It imports `model.glb`, never `source.glb`, for the reason
  [Your first asset](02-your-first-asset.md#what-comes-back) gives.
- **Add to Plotter as a tileset**, or **Add to a Packwright atlas**.
- **Copy settings to form** repopulates the Create forms from that job without submitting anything —
  the fastest way to make a variation on something that worked.
- **Reroll** and **Remesh**, which the last chapter covered.

## Rows that behave oddly

A few rows in the library are not assets in their own right. A rig, a sprite sheet, a character
sheet, a re-texture — each of these is a *product of another asset*. It has no thumbnail of its own
and no directory of its own; it writes into the directory of the job it was made from.

The app knows this and opens them onto the screen that actually shows them, rather than onto the
blank screen their own stage would imply. It is worth knowing they exist so that a row without a
picture does not look like a bug.

## What to read next

[Judging what you made](04-judging-what-you-made.md) — how to tell a good mesh from a bad one, what
the app's own measurements are worth, and which of them mean less than their names suggest.
