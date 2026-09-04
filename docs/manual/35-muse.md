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
tags, the status, the length, and two buttons.

**Play / Stop** auditions the take. Playback shares the one audio device with Sirens, so starting a
take stops whatever the tracker was playing and vice versa — there is one sound card and this app
does not pretend otherwise. A machine with no device says so rather than offering a dead button.

**Open in Sirens** imports the track into the tracker as a sample instrument. See below.

The tray keeps every take until you delete it. Nothing is overwritten, and two takes of the same
brief are two files — which is the comparison the count control exists for: if every take is wrong
the same way it is your tags, and if one is right and one is not it is the seed.

## Where a take lives

A take is an ordinary job row. It appears in the [library](36-library-and-jobs.md) with its prompt,
its seed and its recipe alongside every image and mesh you have made, it can be filtered, renamed,
tagged, trashed and pruned like any of them, and its file is `track.wav` in the job's own directory.

That is why Muse is a mode and not a stage of Create, despite its results being job rows: a Create
stage is a position on that mode's rail, and the rail computes over things a track does not have —
there is no next stage to advance to, no mesh to reconstruct from it, and no viewport to frame it
in. What Muse owns is a form and a tray, which is a workspace.

## The bridge to Sirens

Press **Open in Sirens** and the app switches modes, and the track lands in the open song's sample
table as a sample instrument. With no song open, one is started.

It works because the formats already agreed: Muse writes a 44.1 kHz WAV and Sirens' sample
instruments read 44.1 kHz WAVs. Nothing is converted, and a generated sample is not a special kind
of sample — it arrives through exactly the door a WAV dragged in from your own disk arrives through,
which is also why Sirens needed no changes at all to accept one.

What it is for: a generated phrase as a one-shot under a chiptune arrangement, a pad chopped into a
sustained instrument, or a finished track as a reference to write against.

The bridge is one-way. Rendering a tracker arrangement and having the model perform it is the
interesting direction this pairing opens up, and it is deliberately not built yet — it needs the
tracker's render staged to a file and threaded through ACE-Step's audio-to-audio path, which is a
separate piece of work from the one-way import.

## What it does not do

**Loop points.** ACE-Step does not produce seamless loops, and this is a real gap for game music
rather than an oversight being glossed over. A generated track that ends where it began is luck. For
a loop, either fade it in your engine or use the material in Sirens, where a loop is something you
author.

**Stems.** One stereo mixdown per take. Sirens exports stems because it *has* channels; a diffusion
model's output has none to separate.

**Editing.** A take is a file, not a document — there is no undo stack and nothing to open. Change
the brief and generate again, which is cheap, or take it into Sirens, which is where editing lives.

## What to read next

- [Generating a soundtrack](16-generating-a-soundtrack.md) — the tutorial: a description to a track to a sample.
- [Sirens](34-sirens.md) — the tracker, and what a sample instrument is once it lands there.
- [Installation](39-installation.md) — the ACE-Step download.
