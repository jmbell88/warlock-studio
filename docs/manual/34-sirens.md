# Sirens

Sirens is the chiptune tracker: a grid you type notes into, five voices in the shape of an NES sound
chip, and a `.wsng` document that exports as WAV. It is the one mode in this app whose output you
listen to rather than look at.

It exists because everything else here makes something you can see, and a game is not finished when
it is silent. The same argument that put a raster editor and a tile-map editor in an asset generator
puts a tracker in it: the music for a pixel-art game is a small, hand-authored thing, and the tools
for making one are a separate download in a separate idiom that nothing else you have made can reach.

It is a mode, not a takeover. Switching away leaves every open song where it was, several songs stay
open at once, and the layout follows the rest of the app: the transport and the order list on the
left, the pattern grid in the middle, the instruments, their envelopes, the sound effects and the
file panel on the right.

**No sound card is not an error.** Everything except playback works on a machine with no audio
device at all — writing, editing, saving, loading and exporting a WAV. The transport says so in
those words rather than greying a Play button with no explanation, because a disabled control with
no sentence beside it reads as a broken app rather than as a fact about the machine.

## Starting a song

With nothing open, the middle column offers **New song** and **Open a file...**, and lists what you
had open recently. `Ctrl+N` and `Ctrl+O` do the same from the keyboard. The document's own format is
`.wsng` — a zip holding the song as JSON, its patterns as numpy arrays and any imported samples as
WAVs.

A new song is not empty. It has five channels, one pattern of 64 rows, one instrument per voice kind
and an order that already points at the pattern — so a note typed into the first row makes a sound.
A document you have to configure before it can play anything is one where a beginner's first typed
note is silent for a reason nothing on screen explains.

A document that cannot be opened is refused with the reason rather than a generic failure, and a
file too big to be a song is refused before a byte of it is read.

## The five channels

| Channel | Voice | What it is for |
| --- | --- | --- |
| Pulse 1 | pulse | Melody. Four duty cycles, panned slightly left. |
| Pulse 2 | pulse | Harmony or a second melody, panned slightly right. |
| Triangle | triangle | Bass. The chip's 32-step staircase, not a smooth ramp — the steps are the sound. |
| Noise | noise | Percussion, from an LFSR. The note sets how fast it clocks — how bright the hiss is. |
| Sample | sample | A `.wav` you imported, pitched from the note. `C-4` plays it at its recorded speed. |

That is the 2A03's arrangement, in its order, and it is what a piece of music in this idiom was
written for. **This build gives you those five and no control to add a sixth** — the engine will
play up to 32 channels and a second triangle sounds exactly like the first, but nothing on screen
mints one, so a `.wsng` has the five it was created with. Say so plainly rather than leave it to be
looked for: the arrangement is the constraint the idiom is written under, and working inside it is
most of what makes a track sound like this.

Two departures from the hardware are deliberate and both are audible. Tuning is equal-tempered
rather than a period table, because a track written here has to sit under modern music without
beating against it. And the voices are synthesised at four times the sample rate and filtered back
down, because a naive square wave at 44.1 kHz folds its own harmonics down into the audible range,
where they do not read as brightness — they read as the instrument being out of tune.

## The pattern grid

The middle of the window is one pattern at a time. Each row is a sixteenth note; each channel has
five columns:

| Column | Width | What it holds |
| --- | --- | --- |
| Note | 3 | `C-4`, `D#5`, `===` for a note-off, `~~~` for a release. `...` is nothing at all. |
| Instrument | 2 | Which instrument the note is played on, by its number. |
| Volume | 2 | `00` to `0F`. Empty means "whatever it already was". |
| Effect | 1 | One letter. See [Effects](#effects). |
| Parameter | 2 | That effect's two hex digits. |

A run of dots is not decoration: it is how the eye finds the rows where something actually happens.
Every fourth row — one beat — carries a stripe behind it, so counting to the downbeat is not manual.

**Only the note column can be typed into in this build, and that is the mode's one real gap.** The
other four are drawn, the caret moves into them with the arrow keys, and no key writes anything
there. The instrument column is filled in for you when you type a note; the volume, effect and
parameter columns stay empty. Everything the [Effects](#effects) table describes is played by the
synthesiser and reaches an export, so a `.wsng` that carries an `F78` will change tempo — there is
simply nothing in this build that puts one there. It is why Sirens still wears an **Experimental**
chip on the rail. Volume is reachable another way in the meantime: put the shape you want into the
instrument's volume envelope, or make a second instrument that is quieter.

**Typing notes.** The keyboard is a piano in two rows, FamiTracker's layout: `zsxdcvgbhnjm` is the
octave the Octave field names, and `q2w3er5t6y7u` is the one above it. The piano only fires in the
note column, because `e` in the effect column is the letter of an effect rather than an E natural.
The backtick key writes a note-off (`===`), which **cuts** the voice dead. `Delete` clears the
selection.

The gentler `~~~` — let go of the note and play the instrument's release tail — has no key in this
build either. A `.wsng` can carry one and the synthesiser plays it, and every voice still sounding
at the end of a song is released automatically, so an envelope's release half is what you hear
there. Mid-song, a note either rings or is cut.

Those are *letter* positions rather than physical key positions, which is right for anyone arriving
from another tracker and wrong on an AZERTY keyboard. Serving one of the two means not serving the
other, and the tracker convention won.

A typed note also stamps the currently selected instrument into the instrument column. Every tracker
does this, and without it a typed note is silent for a reason that is invisible.

**The toolbar over the grid** carries the caret's own state rather than any setting: **Octave** is
which octave the lower piano row plays (`-` and `=` move it), **Step** is how far the caret drops
after each entry — set it to 4 and you are writing on the beat — and **Follow** scrolls the grid
with the playhead instead of with the caret.

**Selections and moving about.** Shift with the arrow keys extends a block over rows and channels;
`Esc` drops it. `Page Up` and `Page Down` move sixteen rows, which is four beats — one bar in every
time signature this idiom uses. `Shift+1` and `Shift+2` transpose the block down and up a semitone.

Clicking anywhere in the grid moves the caret there.

## Effects

One letter and two hex digits, in the last two columns. The letters are the tracker convention
rather than an invention — someone arriving from FamiTracker or DefleMask already knows all of them.

**This is what the synthesiser plays, not what you can type.** As the section above says, this build
has no way to enter a value into the effect column; the table is here because these are what a
`.wsng` means, what a stem preserves and what an export renders, and because it is the list the
column editor will answer to when it exists.

| Letter | What it does |
| --- | --- |
| `0xy` | Arpeggio: cycle the note, +x and +y semitones, one per tick |
| `1xx` | Slide the pitch up, xx cents per tick |
| `2xx` | Slide the pitch down, xx cents per tick |
| `3xx` | Slide to the note, xx cents per tick (portamento) |
| `4xy` | Vibrato: x is the speed, y the depth |
| `Axy` | Slide the volume, x up or y down per tick |
| `Bxx` | Jump to order position xx |
| `Cxx` | Stop the song here |
| `Dxx` | End this pattern, resume at row xx of the next |
| `Fxx` | Set the tempo to xx beats per minute |

**`Fxx` is a tempo here, and that is a departure.** In FamiTracker the same letter sets ticks per
row, and changes the tempo as a side effect. In this engine a row is a sixteenth note by definition
and **Speed** is how finely a row is subdivided, so a mid-song speed change would alter how fast
envelopes run and leave the music at exactly the tempo it was — a control that appears to do nothing.
What somebody reaching for that letter wants is the audible half of it, so that is what it does:
`F78` is 120 BPM. The letter is kept because it is the one every tracker uses for "the timing changes
here".

## Patterns and the order

The left column's lower half is two lists, and they are deliberately two.

**Patterns** is every pattern the document holds, with a row-count slider for the selected one (1 to
256 rows). **Order** is the sequence they play in. Adding a pattern does not add it to the order, and
removing an entry from the order does not delete the pattern — the whole point of an order list is
that a pattern can appear in it several times, or not at all.

**Loop at the end** is what makes a soundtrack a soundtrack. With it on, the render carries loop
points, and those loop points are written into the exported WAV's `smpl` chunk — so an engine that
reads one loops the track at the right sample rather than restarting it. A loop that lives only in a
sidecar the engine never reads is a track that does not loop.

An order with nothing in it plays nothing, and the transport says so rather than looking broken.

## Instruments

An instrument is not an ADSR with four knobs. It is four short lists of numbers stepped once per
tick, and the shape of the list *is* the sound: a volume of `15 14 12 8 4 2 1 0` is a pluck, and
`15 15 15 15` on a loop is an organ. So the instrument list is a list, and the four sequences are a
pane of their own underneath it.

The list gives each instrument a name, one of the four kinds — pulse, triangle, noise, sample — and
a number. **The number is what the grid's instrument column holds** and what you type there. It is a
per-document slot rather than a position in the list, so removing an instrument does not renumber the
ones after it and does not silently repoint the notes that named them. A song holds 128.

A new instrument is a plain sustained tone at full volume with a short decay on release: audible
immediately, and obviously a starting point.

### The envelope editor

Four bar graphs — **Volume**, **Arpeggio**, **Pitch** and **Duty** — for whichever instrument is
selected. Drag across one to paint a shape into it; painting past the end lengthens the sequence,
so a decay is one gesture rather than a length field and then a drag. A whole drag is **one** undo
step.

Each graph carries two markers you can drag:

- The **loop** point: playback returns here and repeats from it for as long as the note is held.
- The **release** point: everything before it is what a held note plays, and everything from it is
  the tail that plays once the note is let go. **The tail never loops.**

That split is the one rule about envelopes that is invisible in a list of numbers, so the editor
draws the tail on its own ground, in its own colour, with the loop's span underlined along the
bottom of the held half. You should be able to see which half is which without being told.

**Where you will actually hear the tail, in this build, is at the end of a song** — every voice
still sounding when the order runs out is released, so the last chord decays through its release
half rather than stopping dead. Mid-song it needs a `~~~` in the note column, which has no key yet
(see [The pattern grid](#the-pattern-grid)); the note-off you *can* type cuts instead. That does not
make the release half decorative — it decides how a track ends — but it is worth knowing before you
spend an afternoon on one.

**Duty** means different things to different voices, and that is worth knowing before you drag it.
On a pulse it is the square's width, one of four. On a **noise** instrument it is not a duty cycle at
all but the LFSR's tap — 0 is the long, hiss-like sequence and 1 the short, pitched rattle. On a
triangle and a sample it does nothing.

Volume runs 0–15 and duty 0–3, which are the engine's own bounds. Arpeggio and pitch have no bound
in the engine — they are added to a floating-point pitch — so what the editor offers, one octave and
one semitone per tick, is its *reach* rather than a rule. A sequence that already holds more widens
its own graph, so the editor never draws a file clipped and then paints over the clipping.

### Samples

An instrument of kind **sample** plays a recording instead of synthesising one, pitched from the note
you play it at: `C-4` is its recorded speed, an octave up is twice as fast. Import a `.wav` from the
instrument panel, or drop one on the window while Sirens is in front. A song holds 64.

Two files with the same name are two samples rather than one overwriting the other. Removing a sample
leaves the instruments that named it alone, and the instrument panel says, in as many words, that
this instrument names a sample the song no longer holds and is therefore silent — rather than
blanking the field and leaving you to work out why nothing plays.

## Playing it

Press **Play**, or `Space`. Playback is **render-then-play**: the whole song is synthesised into a
buffer and the buffer is handed to the sound device. It is not a live synthesiser feeding an audio
callback, which means two things worth knowing. Edits are heard on the next render rather than
instantly, and the transport says `Rendering...` while that is happening — a few seconds for a long
song. In exchange, what you hear is bit-for-bit what the exported WAV contains, on every machine,
with no chance of a dropout because something else on the computer was busy.

**Re-render** forces one; you rarely need it, because any edit re-arms the renderer by itself.

**Tempo** and **Speed** in the transport belong to the song rather than to the transport: they are
undoable and they re-arm the renderer. A tempo change you cannot take back is the one edit in a
tracker people make by accident, because the control is a slider next to a Play button. Under them
the pane prints milliseconds per row, which is the one number you can check against a metronome —
ticks per row and beats per minute do not combine into anything readable.

If this machine has no audio device, the Play button is disabled and the reason is printed beside it.
Everything else in the mode still works.

## Sound effects

A song document holds sound effects as well as music. Each one is a little pattern of its own with
its own tempo and speed, listed in the **Sound effects** panel.

An effect keeps its own timing on purpose: a coin pickup is forty milliseconds whatever the music is
doing, and tying it to the song's tempo would mean every effect in the document changed length the
moment you slowed the track down.

**Selecting an effect points the pattern grid at that effect's pattern**, so the grid is the effect
editor — the same five columns, the same piano rows, the same undo stack. There is no second, smaller
grid to learn. The panel says which of the two the grid is currently showing, and clicking a pattern
in the Order panel takes you back to the song.

The **play** button on each row auditions that effect: it renders it and plays it once, without
touching the song's own buffer, so auditioning during a long track does not cost you the render.

Removing an effect leaves its pattern in the document — a removal that also deleted a pattern you had
put in the song's order would be a removal that changed the music. Undo takes back either.

A song holds 64 effects.

## Exporting the audio

**Export audio...** in the Song file panel asks for a *folder* rather than a filename, and writes:

| File | What it is |
| --- | --- |
| `song.wav` | The whole mix, with the order list's loop points in its `smpl` chunk. |
| `stems/<channel>.wav` | One file per channel — five, named after the channels. |
| `sfx/<effect>.wav` | One file per sound effect, named after the effect. |

A folder rather than a filename because this is the one export in the app that writes a family of
files under names it chooses: a typed filename would land on `song.wav` and be ignored by the twelve
beside it.

**The `.wsng` is the composition and every WAV is derived from it.** Exporting a document nobody has
touched twice writes byte-identical files both times — there is no timestamp, no writer string and no
randomness anywhere in the path. That is what makes an exported track something a build script can
regenerate rather than an artefact you have to keep.

**A stem keeps the other channels' effect column, and that is deliberate.** `Bxx`, `Cxx`, `Dxx` and
`Fxx` belong to the player rather than to a voice, and any channel may carry them — so a stem
rendered from a grid with the other channels wiped clean would jump differently, halt somewhere else
and run at a different tempo than the mix it is supposed to line up with. Only the note, instrument
and volume columns are blanked. Stems are therefore sample-aligned with `song.wav` and carry its loop
points.

Nothing is written until every file has been encoded, so an export that is refused halfway leaves no
half-populated `stems/` behind to be mistaken for a finished one.

A channel or effect whose name cannot be a filename gets a positional fallback — `effect3` — rather
than taking the whole export down over one row of forty. Two channels with the same name become two
files, numbered, rather than one silently overwriting the other.

## Where the files go

| File | What it is |
| --- | --- |
| `<name>.wsng` | The document: channels, patterns, order, instruments, samples and effects. |
| `song.wav` | An exported mix. Derived; the `.wsng` is the source. |
| `stems/<channel>.wav` | One exported channel. |
| `sfx/<effect>.wav` | One exported sound effect. |

## What is deliberately not here

Sirens is a tracker for writing a game's music, not a digital audio workstation, and several things
you might reach for are absent on purpose rather than pending:

- **No mixing effects.** No reverb, no delay, no filters, no per-channel EQ. A channel has a volume
  and a pan and nothing else. Those belong to whatever you take the stems into.
- **No recording.** Nothing here captures live input; a sample arrives as a `.wav` file.
- **No MIDI**, in or out. The keyboard is the piano.
- **No import of other trackers' formats.** A `.wsng` is written here or nowhere. `.wav` is the only
  format that comes in, and only as a sample.

And two limits of the current build, stated plainly rather than left to be discovered:

- **Four of a cell's five columns cannot be typed into, and the note column takes only two things.**
  Notes and a hard note-off; volume, effect and parameter stay empty, the instrument column is
  filled in for you, and the release note `~~~` has no key. This is the unfinished part of the mode
  and the reason for the Experimental chip. See [The pattern grid](#the-pattern-grid).
- **Playback has no scrub and no start-from-here.** `Space` plays the song from the beginning and
  stops it. To hear one section, put it at the top of the order.

## When it goes wrong

**Nothing plays and Play is greyed out.** This machine has no audio device that the app could open.
The sentence beside the button says so. Writing, saving and exporting are unaffected — you can
compose here and listen elsewhere.

**A typed note is silent.** Three usual causes, in order of likelihood: there is no instrument
selected, so nothing was stamped into the instrument column; the instrument named is a `sample` one
whose sample has been removed; or the pattern you are typing into is not in the order list, so the
song never reaches it.

**Nothing happens when I type in the volume or effect column.** Nothing is meant to yet — those
columns take no keyboard input in this build. See [The pattern grid](#the-pattern-grid).

**My note stops dead instead of fading.** The backtick writes `===`, which cuts. The fade is what
the instrument's release half does, and reaching it mid-song needs a `~~~` that this build has no
key for. At the end of the song it happens by itself.

**The song plays nothing at all.** The order list is empty. Add a pattern to it from the Order panel.

**An edit does not change what I hear.** The render is a few seconds behind you on a long song, and
the transport says `Rendering...` while it catches up. If it says an error instead, that is a
refusal — most often a song past the ten-minute render ceiling.

**A sound effect changed nothing about the song.** It should not have. An effect is not in the order
list, so nothing about it can change `song.wav`. What it changes is the next audition and the next
export.

**Export is greyed out.** The document has neither an order list nor a sound effect, so an export
would be a folder of empty WAVs.

See [Keyboard shortcuts](38-shortcuts.md#sirens) for every binding, and
[Making a soundtrack](14-making-a-soundtrack.md) for one path through the mode from an empty document
to an exported WAV.
