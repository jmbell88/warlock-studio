# Muse

Muse generates a finished piece of music from a description. It is the other half of the pair
[Sirens](34-sirens.md) opens: that mode is a tracker you author note by note, and this one is a
model you ask.

The two are not a pipeline and not rivals. They are the same relationship
[Create](22-generating-references.md) has to [Inker](28-inker.md) — generate a thing, or draw it —
applied to sound, and which one is right depends entirely on whether you want two minutes of ambient
dungeon music at four in the morning or a chiptune loop you control every note of.

The model is **ACE-Step v1**, a 3.5B text-to-music diffusion transformer that runs locally and
offline on the same card the image pipeline uses. It is the only model this mode has, and there is
no fallback: without its weights Muse refuses at the door and names the download.

## What it takes

Two inputs, and they are the model's own two rather than an interface layered over them.

**Style tags.** Comma-separated, not a sentence. The text encoder was trained on tag strings, and
prose describing a mood gets you a vaguer result than the same mood in tags. Instruments, genre,
tempo words, key, and production adjectives all work:

```
dark ambient, dungeon, low strings, slow, sparse percussion, minor key
```

**Lyrics.** A block marked up with `[verse]`, `[chorus]` and friends. Optional, and empty means an
instrumental — which is what most game music is. Non-English lyrics work; the model carries its own
language segmentation and pronunciation stack.

## The window

The brief runs across the top and the recipe column sits on the right, which is the same split
[Create](22-generating-references.md) keeps between its bar and its settings column: **the bar is
what to make, the column is how.** No control appears in both.

### The brief

| Control | What it does |
| --- | --- |
| **Style tags** | The description. Comma-separated tags. |
| **Lyrics** | The lyric block, or empty for an instrumental. |
| **Duration** | 30, 60, 120 or 240 seconds. The parameter that decides what the press costs. |
| **Count** | How many takes one press queues, each with its own seed. |
| **Generate** | Queues them. `Ctrl+Enter` does the same from anywhere in the mode. |

Duration is bounded rather than free because it is the one parameter whose cost is unbounded: it
sets the length of what the model samples, so it drives both the generation time and the VRAM
figure the app has to check *before* admitting the job. Four minutes is longer than any game loop
needs.

### The recipe

| Control | What it does |
| --- | --- |
| **Steps** | Sampling steps. 60 is the default; below about 30 the output falls apart, above about 80 you are buying time. |
| **Guidance** | How closely the model follows your tags. Low wanders, high obeys and can flatten. |
| **Scheduler** | Euler, Heun or ping-pong. Euler is the default and what the other defaults were chosen against. |
| **Guidance type** | APG, CFG or `cfg_star`. APG is the model's own default and holds up best at high guidance. |
| **Omega** | ACE-Step's granularity term. Leave it alone unless a take is muddy in a way guidance does not fix. |
| **Pin the seed** | Off by default: every take gets a fresh seed. On, the number applies to the first take of a press and the rest walk from it. |

Every one of these is named for the model's own parameter, deliberately — so a setting here, the
value stored in a finished take's recipe, and ACE-Step's own documentation are three views of one
word rather than three vocabularies.

## The takes

Each press queues *count* rows and the tray draws one card each, newest first. A card shows the
tags, the status, the length, the **seed** it was drawn with, and — for a take made from another —
what kind of derivation it was and which take it came from. The seed is on the card because that is
what tells two near-identical generations apart without going to the Library for it.

**Play / Stop** auditions the take. Playback shares the one audio device with Sirens, so starting a
take stops whatever the tracker was playing and vice versa — there is one sound card and this app
does not pretend otherwise. A machine with no device says so rather than offering a dead button.

**Open in Sirens** imports the track into the tracker as a sample instrument. See below.

The tray keeps every take until you delete it. Nothing is overwritten, and two takes of the same
brief are two files — which is the comparison the count control exists for: if every take is wrong
the same way it is your tags, and if one is right and one is not it is the seed.

## The player

Press **Play** on a card and the strip appears along the bottom of the window, full width. It holds
the take you are listening to.

The waveform is the reason the strip exists, and it is not decoration: it is the coordinate system
every other control on the strip is expressed in. The playhead is a position in it, a click on it is
a seek, and the loop region is two marks in it. (A card still shows no waveform, on purpose — a card
is a thing you press, and a picture there would tell you nothing that pressing it does not.)

| Control | What it does |
| --- | --- |
| **Play / Stop** | Starts from wherever the playhead is, rather than always from the beginning. |
| The waveform | Click to move the playhead. Drag either loop marker to move it. |
| **Volume** | The one sound channel's level. It is the *same* control Sirens' transport draws — this app has one sound card and does not pretend otherwise, so setting it here sets it there. |

Keyboard, while Muse is the mode:

| Keys | Action |
| --- | --- |
| Space | Play or stop the selected take |
| Left / Right | Nudge the playhead a second. Hold Shift for ten |
| Home | Back to the start |
| `[` / `]` | Set the loop's start or end **at the playhead** |
| L | Look for loop points |

The bracket keys are the reason the keyboard is a real alternative to dragging a marker rather than
a shortcut for the buttons: they place a marker exactly where you are listening, which is the thing
a mouse is bad at.

## Looping a take

Two halves, and they answer different halves of the problem. Neither is the other.

**Make it loop** (on a card's *Make more* menu) asks the model to rewrite the joint. Muse rolls the
take by half its length so that the seam between its end and its beginning sits in the *middle* of
the file, repaints across it — where the music on both sides is context the model can see — and
rolls it back. The join is then something composed rather than a cut. What it does not do is make
the first and last samples equal: ACE-Step has no cyclic objective and is not being asked for one.

Loop points belong to the take, not to the player: audition another take and come back, and your
region and crossfade are where you left them.

**Find loop points** (on the player) is the other half. It searches the take for the two positions
where the music most nearly repeats, judging each candidate on the moment itself, on the third of a
second leading *into* it — which is what distinguishes "these two places sound alike" from "the
music arriving here is the music arriving there" — on the level either side, and on how much of the
take the loop covers. It offers a handful of answers rather than one, because this is a heuristic
over material nobody composed to loop, and adopts the best immediately so you can hear it. Press a
numbered button to try another; drag either marker to adjust it by hand.

Both markers snap to a **rising** zero crossing, never to whichever crossing is nearest. Two
crossings of matching sign have matching slope; joining a rising edge to a falling one clicks
audibly even though both samples are zero.

**Crossfade** trades two things you can hear. At zero the seam is a butt join and keeps every
transient; longer is certainly smooth and audibly dips the music through the join. 40 ms is the
default.

A loop that is already seamless is left alone: Muse measures the join a repeat actually crosses
before it fades anything, and if no fade would make that join smaller, it applies none — so raising
the crossfade can never put a click into a loop that did not have one.

### The two exports, and why you cannot have both

**Export the loop** writes the crossfaded body as its own file, marked as looping end to end.

**Export the track with loop points** writes the whole take with the two positions in an `smpl`
chunk — the chunk Unity, Godot, FMOD and Wwise all read.

The second is greyed out whenever the crossfade is above zero, and the reason is not a limitation
being worked around. A crossfaded seam is made of samples that **do not exist in the take** — Muse
writes them. So loop points into the untouched file would be a loop that clicks, wearing a label
saying it does not. Set the crossfade to zero, or export the loop itself.

## Making more of a take

A take is never the end of the line. **Make more** on a card offers six ways to get another take out
of one you already have, each of which queues a new row with the first as its parent — so nothing is
overwritten and you can always go back.

| Choice | What it does |
| --- | --- |
| **Another like this** | Same brief, same underlying noise, nudged toward a fresh draw. **Variation** at 0 is this take again; at 1 it is a different piece to the same brief. This is the one to reach for first. |
| **Extend** | More music before or after. Neither end may be longer than the take itself — extend twice to go further. It is the only way to make a track longer. |
| **Repaint a section** | Regenerate one window and leave the rest alone. Good for a passage that wanders. |
| **Make it loop** | Rewrite the joint between the end and the beginning. See [Looping a take](#looping-a-take). |
| **Change the words or tags** | The same piece to a different brief. Leave a field empty to keep this take's. It has to change *something* — for another take of the same brief, use *Another like this*. |
| **Something like this** | A new piece using this take as a reference. **Closeness** is how near to stay; at the top of the range the model would take no sampling steps at all, which is refused. |

Each of these takes a **How many** the way the brief does, for the same reason: several cheap
candidates to choose between is the point.

**A derived take keeps its parent's seed.** That is what makes it a derivation of *that* take rather
than a different piece filed underneath it; what varies is a second seed the derivation owns. So
rerolling a variation gives you another variation, not the original back.

## Where a take lives

A take is an ordinary job row. It appears in the [library](36-library-and-jobs.md) with its prompt,
its seed and its recipe alongside every image and mesh you have made, it can be filtered, renamed,
tagged, trashed and pruned like any of them, and its file is `track.wav` in the job's own directory.

That is why Muse is a mode and not a stage of Create, despite its results being job rows: a Create
stage is a position on that mode's rail, and the rail computes over things a track does not have —
there is no next stage to advance to, no mesh to reconstruct from it, and no viewport to frame it
in. What Muse owns is a form and a tray, which is a workspace.

## The bridge to Sirens

Press **Open in Sirens** and the track lands in the open song's sample table as a sample instrument,
and the app switches modes once it is in. With no song open, one is started. The switch waits on the
import deliberately: if the file cannot be read, you are told so here, beside the take, rather than a
mode away from it.

It works because the formats already agreed: Muse writes a 44.1 kHz WAV and Sirens' sample
instruments read 44.1 kHz WAVs. Nothing is converted, and a generated sample is not a special kind
of sample — it arrives through exactly the door a WAV dragged in from your own disk arrives through,
which is also why Sirens needed no changes at all to accept one.

What it is for: a generated phrase as a one-shot under a chiptune arrangement, a pad chopped into a
sustained instrument, or a finished track as a reference to write against.

### Composing from a song

The bridge runs both ways. In Sirens, **Compose in Muse...** on the Song file panel renders the open
arrangement and hands it to the model as a reference: you write the shape of the piece in the
tracker — the chords, the form, where it turns — and ask the model to perform it with real
instruments.

Fill in the style tags in Muse first, because those are what the model is being asked *for*; the
song is only what it is being asked to sound like. How closely it follows is the **Closeness**
slider under the button: at the low end the song is a loose suggestion, at the high end the model
is performing your arrangement almost bar for bar. (The same control appears under *Make more →
Something like this* on a take, where it governs that take instead.)

Your loop points travel with the render, since they are written into the WAV the model reads. The
rates already agree in this direction too — the tracker renders at 44.1 kHz.

## Stems

**Split a take into stems** with the **Stems** button on its card. You get four files beside the
take — `drums.wav`, `bass.wav`, `other.wav` and `vocals.wav` — which is what a DAW wants and what
lets you keep the drums and rewrite everything else.

The model is Hybrid Demucs, and it is a separate ~320 MiB download. Muse works without it; what you
lose is those four files. The model *class* ships inside torchaudio, which this build already
installs, so the download is the trained weights and nothing more. Separation takes a minute or so
and runs in a short-lived process that exits when it is done.

**On the licence, plainly.** The Demucs code is MIT, but Meta has stated the trained weights are
provided for scientific purposes only, and the model here was trained the same way with no new
grant. So the download is marked non-commercial and warns you at the moment you agree to it. Stems
you make with it are not cleanly licensed for a commercial release. That is a decision this app
cannot make for you, so it labels it and leaves it to you. (Open-Unmix is not an escape: its code is
MIT and the dataset it was trained on is CC BY-NC-SA.)

Two unrelated things are called `stems/` in this app. Sirens exports one folder of that name — one
WAV per tracker channel, which is what *it* has to separate. Muse writes another, beside the take,
holding what the separation model pulled out of a finished mixdown. Same word, different things.

## Export formats

Beyond the WAV, a finished take can be downloaded as **FLAC**, **MP3** or **OGG**. They are produced
on first request and cached beside the track, the same way the mesh exports work — so every take
already on disk gains them without being regenerated.

No extra download is involved and there is no converter to install: the audio library this app
already uses encodes all four itself. This is worth saying because everyone assumes otherwise.

## What it does not do

**Compose a loop from nothing.** ACE-Step has no cyclic objective and never will have one: it writes
a *piece*, with a beginning and an end. What Muse adds is the two halves of making one loop anyway —
the model rewriting the joint, and the player finding, hearing and shipping the cut. See
[Looping a take](#looping-a-take). What is still not true is that a generated track ends on the
sample it began on; that is what the loop points are for.

**Separate what it never mixed.** The model writes a stereo mixdown, not channels — a diffusion
model has no multitrack to hand back. What [Stems](#stems) does is *estimate* four sources from that
mixdown, which is a different and much harder thing, and it shows: expect a little bleed between
them, most audibly where two instruments share a register.

**Editing, in a document sense.** A take is a file, not a document — there is no undo stack and
nothing to open. Every change is another take with a parent, which is how the rest of this app works
too: see [Making more of a take](#making-more-of-a-take). For editing that *is* document-shaped,
take it into Sirens.

**Write your lyrics.** You supply the words; nothing here composes them. That needs a language model
and this build deliberately ships none.

## What to read next

- [Generating a soundtrack](16-generating-a-soundtrack.md) — the tutorial: a description to a track to a sample.
- [Sirens](34-sirens.md) — the tracker, and what a sample instrument is once it lands there.
- [Installation](39-installation.md) — the ACE-Step download.
