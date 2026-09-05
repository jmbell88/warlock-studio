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
had open recently. `Ctrl+N` and `Ctrl+O` do the same from the keyboard, and once a song is open the
**Song file** panel carries the same **New**, **Open...**, **Save** and **Save As...** buttons every
workspace has, over the file's path and one line saying whether it is saved. The document's own format is
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

**Right-click a channel's name** — the buttons in the strip over the grid — to rename it, change
which voice it plays, and pan it between the speakers. The notes written on a channel stay exactly
where they are when its voice changes: the voice is how they sound, not what they are, so turning
the noise channel into a second triangle is one click and no retyping. Left-click that button mutes
the channel and the **S** beside it solos; both are about listening, so neither is saved into the
`.wsng` — handing somebody else a song with a part missing is not a thing a file should be able to
do. They belong to the song you set them on, so opening the same file in a second tab to compare
two versions gives you two independent sets of mutes.

If the pane is too narrow for every channel, the ones that do not fit are counted in the margins
(`<2` on the left, `3>` on the right) and the grid follows the caret sideways as you arrow into
them.

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
| Volume | 2 | `00` to `0F` — one hex digit typed. Empty means "whatever it already was". |
| Effect | 1 | One letter. See [Effects](#effects). |
| Parameter | 2 | That effect's two hex digits. |

A run of dots is not decoration: it is how the eye finds the rows where something actually happens.
Every fourth row — one beat — carries a stripe behind it, so counting to the downbeat is not manual.

**Which column the caret is in decides what a key means.** That is the one thing to know about
typing here, and it is why the same letters do different jobs in different columns: `c` is a note in
the first column and the hex digit twelve in the third, and `b` is a note in the first column and
the jump effect in the fourth. Move between columns with the left and right arrow keys.

**The line under the grid says what the keyboard does in the column the caret is in**, and changes
as you move across. It names that column's keyboard — the piano rows, the hex digits, the effect
letters — followed by the keys that mean the same thing everywhere, and it grows the block chords
once a selection exists. It is there so that the paragraph above is answered on screen rather than
only on this page.

**Typing notes.** The keyboard is a piano in two rows, FamiTracker's layout: `zsxdcvgbhnjm` is the
octave the Octave field names, and `q2w3er5t6y7u` is the one above it. The piano only fires in the
note column, because `e` in the effect column is the letter of an effect rather than an E natural.
Press a piano key anywhere else and the app names the column you are actually in and how many left
arrows walk back to Note — a piano key outside the note column is not stray typing, it is somebody
aiming at a note, and it is the one rejected key here worth a sentence.
The backtick key writes a note-off (`===`), which **cuts** the voice dead; `Shift+Backtick` writes a
release (`~~~`), which lets go of the note and plays the instrument's release tail instead. They are
one key with and without Shift because they are one gesture with two endings, and the tilde is the
character the cell itself draws.

Those are *letter* positions rather than physical key positions, which is right for anyone arriving
from another tracker and wrong on an AZERTY keyboard. Serving one of the two means not serving the
other, and the tracker convention won.

A typed note also stamps the currently selected instrument into the instrument column. Every tracker
does this, and without it a typed note is silent for a reason that is invisible.

**Typing everything else.** The instrument and parameter columns take **two hex digits**: the first
fills the left-hand digit and the second the right, and while an entry is half done the caret rings
the one character it is waiting for rather than the whole cell. The volume column takes **one** digit,
`0` to `F`. The effect column takes an effect's **letter**, from the table below.

A digit replaces one half of what is already there rather than starting a fresh value, so fixing the
right-hand digit of `4F` is one keystroke on the second character. Moving the caret — an arrow key, a
click, anything — ends a half-finished entry, so the second digit can never land in a cell you have
left. Two things are refused rather than written: a letter the engine has no effect for, and an
instrument number past `7F`, which is a slot no song can hold. In both cases nothing happens at all,
because a value the synthesiser cannot use is worse in the cell than an empty one — it would draw as
something the song does not play.

`Delete` and `Backspace` blank the column the caret is in, so a wrong instrument number is taken back
without losing the note beside it. With a block selected they blank the whole block, across every
column.

**The toolbar over the grid** carries the caret's own state rather than any setting: **Octave** is
which octave the lower piano row plays (`-` and `=` move it), **Step** is how far the caret drops
after each entry — set it to 4 and you are writing on the beat — and **Follow** puts the caret on
the row that is sounding, so what is under the highlight is what your next keystroke writes to.
Turn it off to type into bar 3 while bar 1 plays.

**The channel strip over the grid** names every channel and says whether the mix is playing it.
Click a name to mute it, the **S** beside it to hear that channel alone; solo wins over every mute,
so checking a bass line and going back does not mean undoing four mutes. Neither is part of the
song — a `.wsng` that remembered your mutes would hand somebody else a track with a part missing —
and both re-render, because what you hear is the render.

**Selections and moving about.** Shift with the arrow keys extends a block over rows and channels;
`Esc` drops it. `Page Up` and `Page Down` move sixteen rows, which is four beats — one bar in every
time signature this idiom uses. `Shift+1` and `Shift+2` transpose the block down and up a semitone.
`Ctrl+C` copies the block (or the cell under the caret), `Ctrl+X` cuts it, and `Ctrl+V` puts it down
with its top-left corner at the caret — a block that runs off the bottom or the last channel is
clipped, not refused. The clipboard belongs to the mode rather than the song, so a bar copied in one
tab pastes into another, and a cut is one undo step.

`Home` and `End` jump to the first and last row; hold Shift and they select to it. `Insert` opens a
blank row at the caret and pushes the rest of that channel down; `Shift+Delete` takes the caret's
row out and pulls the rest up. Neither changes how many rows the pattern has — the row pushed off
the bottom is gone — because a pattern's length is what the order list and every other pattern are
written against. With a block selected, both reach every channel the block covers rather than just
the one under the caret.

`Ctrl+G` **interpolates**: type a value at the top of a selected block and another at the bottom,
and the rows between are filled with the straight line between them. It ramps the note, the volume
and the effect parameter, and only where both ends actually hold a value — a fade needs two ends,
and inventing one from an empty cell writes notes nobody typed. Instrument numbers are never ramped:
they are a set with no order, so a "line" from `01` to `07` would name five slots you did not choose.

`Ctrl+Up` and `Ctrl+Down` step the selected instrument — the one a typed note is stamped with —
without leaving the keyboard.

**Notes play as you type them**, on the instrument they were stamped with and through the voice the
channel is, so writing a melody does not mean pressing Space after every note. Turn it off with
**Preview** in the toolbar. A preview never interrupts playback: while the song is playing, the song
is what you hear.

Clicking anywhere in the grid moves the caret there, into the column you clicked: click on an
effect's letter and the next key you press is an effect letter, not a note.

## Effects

One letter and two hex digits, in the last two columns. The letters are the tracker convention
rather than an invention — someone arriving from FamiTracker or DefleMask already knows all of them.

Put the caret in the effect column and press the letter; the two digits go in the column after it.
A letter that is not in this table writes nothing, so the table is also the whole list of what the
effect column will accept.

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

**An effect runs until you cancel it.** The six voice effects — `0xy` `1xx` `2xx` `3xx` `4xy` `Axy`
— keep doing what they were told on every row after the one you typed them on. An empty effect
column says nothing, not "stop". To stop one, type it again with a zero parameter:

| To stop | Type |
| --- | --- |
| An arpeggio | `000` |
| A pitch slide, up or down | `100` or `200` |
| A portamento (notes stop gliding and are struck again) | `300` |
| A vibrato | `4x0` — any speed, depth zero |
| A volume slide | `A00` |

That is FamiTracker's rule and every module tracker's. The three player effects — `Bxx`, `Cxx`,
`Dxx` — are events rather than states: they happen on the row they are on and nothing carries over.
`Fxx` sets the tempo, which then stays set until another `Fxx`.

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

Each order row carries its own three buttons: **▲**/**▼** move the entry, the middle one points it
at a different pattern, and the bin takes it out. A pattern's own row has **Duplicate** — a chorus
is usually the verse with two rows changed, and a copy is one undo step, named `verse 2` so the two
can be told apart in the list — a name field, and **Delete**, which asks first when the order list
plays that pattern and names how many entries go with it. All of it is one Ctrl+Z.

**Loop the song** is what makes a soundtrack a soundtrack. With it on, the render carries loop
points, and those loop points are written into the exported WAV's `smpl` chunk — so an engine that
reads one loops the track at the right sample rather than restarting it. A loop that lives only in a
sidecar the engine never reads is a track that does not loop.

**The loop point is any entry, not only the first.** Pick it with the slider under the checkbox; the
entry it names is marked in the list. That is how an intro that plays once works: entries 00–01 are
the intro, the loop starts at 02, and the WAV tells the engine so. Move an entry above the loop
point and the loop follows it rather than silently pointing at whatever landed there.

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

**Two things reach the tail.** `Shift+Backtick` writes a `~~~` under a note, which lets go of it and
plays the tail where you put it; and every voice still sounding when the order runs out is released
anyway, so the last chord of a song decays through its release half rather than stopping dead. The
plain backtick's `===` is the other choice, and it cuts. Which of the two you want is a musical
decision and the grid shows you which one is there.

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
instrument panel, or drop one on the window while Sirens is in front. A song holds 64, and one
sample runs to four minutes — long enough for a whole track from Muse, and past that the import is
refused rather than trimmed.

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

**From the caret** plays from the row the caret is on, in the song's own timing — writing bar 40 of
a three-minute track does not mean hearing the first two minutes to check it. It plays the *song*,
so a pattern the order list never reaches says so rather than starting from the top. When a pattern
appears more than once in the order list, it starts at the entry you selected there, not at the
first place that pattern is used — so writing the last chorus and pressing it plays the last chorus.
**This pattern** plays the pattern the grid is editing, once, whether or not the order reaches it;
it never touches the song's own buffer, so pressing Play afterwards plays the song. **Loop
playback** repeats what was rendered — from the caret it still repeats the *whole* song rather than
the part after the caret, so a loop is a loop of the piece however you started it. The loop *point*
in the order list is what an exported WAV tells a game engine, while this is for listening.

There is one sound device and one song on it. Switching to another tab stops what was playing rather
than leaving it sounding under a transport belonging to a different song, and Stop always means the
song you are looking at.

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

## When it goes wrong

**Nothing plays and Play is greyed out.** This machine has no audio device that the app could open.
The sentence beside the button says so. Writing, saving and exporting are unaffected — you can
compose here and listen elsewhere.

**A typed note is silent.** Four usual causes, in order of likelihood: there is no instrument
selected, so nothing was stamped into the instrument column; the instrument named is a `sample` one
whose sample has been removed; the row is on the **Sample** channel and the instrument named is not
a `sample` one, so there is no recording to play; or the pattern you are typing into is not in the
order list, so the song never reaches it.

The channel decides the voice and the instrument brings its envelopes, so an instrument of one kind
on a channel of another is not refused and is usually what you want — a pluck written on pulse works
on triangle unchanged. The sample channel is the exception above, because a recording is the one
thing an envelope cannot stand in for.

**Nothing happens when I type in the effect column.** That letter is not one the engine has an
effect for, and a letter it cannot play is refused rather than written. The [Effects](#effects) table
is the whole list. The same is true of an instrument number past `7F`: no song has that slot. The
exception is a letter that is also a piano key: that one is answered out loud, because it is far
more likely to be a note aimed at the wrong column than an effect that does not exist.

**A digit landed in the wrong cell.** The instrument and parameter columns take two digits, and the
caret rings the one it is waiting for. If you moved between them the entry was ended, and the digit
you typed next started a new one where the caret then was.

**My note stops dead instead of fading.** The backtick writes `===`, which cuts. The fade is the
instrument's release half, and reaching it needs `Shift+Backtick` — a `~~~` — instead. At the end of
the song it happens by itself either way.

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
