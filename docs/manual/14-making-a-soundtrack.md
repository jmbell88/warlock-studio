# Making a soundtrack

Sirens is the tracker: pattern grid, five NES-shaped voices, one `.wsng` document, WAV out. Like
Inker and Packwright it needs no GPU, no weights and nothing downloaded — but unlike them it needs a
sound card if you want to hear anything. Everything else in this chapter works without one.

This walks one path: a new song, a bassline, a lead, a hat, an envelope, a sound effect, an export.
Fifteen minutes, and at the end you have a looping track and a coin pickup on disk.

## A new song

Open **Sirens** from the rail and press `Ctrl+N`.

You get five channels — two pulses, a triangle, a noise and a sample slot — one 64-row pattern, one
instrument per voice kind, and an order list already pointing at that pattern. That last part matters:
a song whose order is empty plays nothing, and a brand-new document that started that way would make
your first typed note silent for a reason nothing on screen explains.

Set the transport's **Tempo** to something you can count to. 120 is a fine place to start. Under the
two sliders the pane prints milliseconds per row — that is the number you can check against a
metronome, which "speed 6" is not.

## A bassline on the triangle

Click into the **Triangle** channel's note column, third group across.

The keyboard is a piano. The lower row `zsxdcvgbhnjm` is the octave the **Octave** field names, and
`q2w3er5t6y7u` is the octave above it — the black keys are where they look on a keyboard. Set Octave
to **3**, because a bass line an octave higher stops being a bass line.

Set **Step** to **4**. Now each note you type drops the caret a whole beat, so you are writing one
note per beat without counting rows.

Type `z`, `z`, `v`, `x` — C, C, F, D. Four notes, four beats, one bar.

Two things happened that you did not ask for. The instrument column filled in with whatever
instrument is selected on the right, which is what makes the note audible rather than silent. And the
row numbers with a stripe behind them are every fourth row: one beat, so you can see the downbeat
without counting.

Press `Space`.

The first press may say *Still rendering your latest edits*. That is not a failure. Sirens renders
the whole song into a buffer and plays the buffer — it is not a live synthesiser — so an edit is
heard on the next render rather than instantly. The trade is that what you hear is bit-for-bit what
the exported WAV will contain. Press `Space` again a second later.

If Play is greyed out with a sentence beside it, this machine has no audio device. Keep going anyway:
everything in this chapter except hearing it works regardless.

## A lead on a pulse

Click into **Pulse 1**'s note column. Set **Octave** to **5** and **Step** back to **1**, so you can
write sixteenths.

Type a phrase — `q`, `w`, `e`, `r`, then a gap, then `t`, `y`, `u`. Anything. Press `Space` and hear
it over the bass.

Two keys worth having now:

- The backtick key writes `===`, a note-off. Put one where you want the lead to stop rather than
  ring into the next note. It **cuts**. `Shift+Backtick` writes `~~~` instead, which lets go of the
  note and plays the instrument's release tail — the gentler of the two, and the one to reach for at
  the end of a phrase.
- `Shift+1` and `Shift+2` transpose a selection down and up a semitone. Hold `Shift` with the arrow
  keys to make one first, and `Esc` to drop it.

## A hat on the noise channel

Click into the **Noise** channel. The noise voice has no pitch worth speaking of — it is an LFSR, and
the note mostly chooses how bright the hiss is.

Set **Step** to **2** and type `z` eight times. That is a hat on every other sixteenth.

It is probably too loud. There are two ways to fix that and they are worth telling apart. Press the
right arrow twice to put the caret in the **volume** column and type a single hex digit — `8` is
about half, `4` quieter still — and that row alone is quieter. Or shape the instrument itself, which
is the better fix for a hat because it fixes every note at once. Keep reading.

## An envelope

Select the noise instrument in the **Instruments** panel. The **Envelopes** pane under it is where a
chiptune instrument actually lives: four short lists of numbers stepped once per tick. The shape of
the list *is* the sound.

Drag across the **Volume** graph, starting high on the left and falling to nothing over five or six
columns. That is a hat: loud for an instant, then gone. Press `Space` and hear the difference —
the same notes, an entirely different instrument.

Some things worth knowing while you are in here:

- **A whole drag is one undo step.** Painting a decay across twenty columns is not twenty `Ctrl+Z`
  presses.
- **Painting past the end lengthens the sequence.** You do not set a length first.
- The graph has **two draggable markers**. The **loop** point is where a held note repeats from. The
  **release** point splits the sequence: everything before it is what a held note plays, everything
  from it is the tail after the note is let go, and **the tail never loops**. The editor draws the
  tail on its own ground in its own colour so you can see which half is which. You hear it wherever
  you write a `~~~` (`Shift+Backtick`), and at the *end of the song* whether you asked for it or
  not — every voice still sounding is released there.

For a plucked lead, try the same shape on the pulse instrument: full volume for one tick, then a fall
to about a third, then a loop point so it sustains there.

## Making it a song

One pattern is one bar of one idea. A song is several patterns in an order.

In the **Order** panel, press **+ Pattern** for a second one, write something different into it, then
press **+ To order** to append it. The two lists are deliberately separate: adding a pattern does not
put it in the order, and removing an order entry does not delete the pattern — which is what lets one
pattern appear in the order three times.

Tick **Loop the song**. That is the difference between a track and a soundtrack: the render now
carries loop points, and those loop points end up in the exported WAV's `smpl` chunk, so an engine
that reads one loops at the right sample instead of restarting. The slider under it picks *which*
entry the loop returns to — leave it at 00 for now, and set it to 01 the day the first pattern is an
intro you only want heard once.

Two things worth trying while it plays. Click a channel's name in the strip over the grid to mute
it, and the **S** beside it to hear that one alone: that is how you find out whether the hat is too
loud without deleting it. And press **From the caret** rather than **Play** — it starts at the row
the caret is on, which is what you want the twentieth time you check the same bar.

## A sound effect

Sound effects live in the same document as the music. In the **Sound effects** panel, press **+ Add**.

You get an effect, its own eight-row pattern, and the grid immediately pointing at it — because an
Add button whose only visible result is a new row in a list looks like a button that did not work.

Write a coin pickup: two or three quick ascending notes on a pulse channel, an octave or so apart. Set
the effect's **Tempo** field high — 300 or so — so the whole thing is over in a fraction of a second.
An effect keeps its own tempo on purpose: a coin pickup is forty milliseconds whatever the music is
doing.

Press the **play** button on that effect's row to audition it. That renders the effect alone and
plays it once; it does not touch the song's buffer, so auditioning does not cost you the render of a
three-minute track.

The panel tells you which of the two the grid is currently showing. Click a pattern in the **Order**
panel to go back to the song.

## Saving and exporting

`Ctrl+S` saves the document as `.wsng` — a zip of the song as JSON, its patterns as arrays and any
samples as WAVs. That file is the composition, and it is the only thing here you cannot regenerate.

**Export audio...** in the **Song file** panel asks for a *folder*, not a filename, because it writes
a family of files:

- `song.wav` — the whole mix, with your loop points in its `smpl` chunk.
- `stems/Pulse 1.wav`, `stems/Triangle.wav`, … — one per channel.
- `sfx/coin.wav`, … — one per sound effect, named after the effect.

Every WAV is a pure function of the `.wsng`. Export the same untouched document twice and you get
byte-identical files, which is what makes an exported track something a build script regenerates
rather than an artefact you have to keep and hope about.

The stems are sample-aligned with `song.wav` and carry its loop points, so you can drop them into a
mixer and they line up.

## Try it

1. `Ctrl+N`, tempo 120, and a four-note bass on the triangle at octave 3 with Step 4.
2. Play it. Note that the first press may tell you it is still rendering, and that this is normal.
3. Add a lead on Pulse 1 and a hat on Noise.
4. Paint a decay into the noise instrument's volume envelope and hear the hat become a hat.
5. Drag the **release** marker left, then play the whole song through: the last chord decays
   through the tail instead of stopping dead.
6. Second pattern, add it to the order, tick **Loop the song**, and move it above the first with
   the **▲** on its row -- the loop point follows it.
7. Add a sound effect, write three ascending notes, raise its tempo, audition it.
8. `Ctrl+S`, then **Export audio...** into an empty folder.
9. Export again into a second folder and diff the two. They are identical.

## What to read next

[Sirens](34-sirens.md) — the reference chapter: every effect letter, the sample instruments, what
`Fxx` does differently here from FamiTracker, and what the mode deliberately does not do.

That is the last tutorial. The reference chapters go deeper on all of it —
[Overview](20-overview.md) is the front door.
