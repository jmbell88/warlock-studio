# Generating a soundtrack

[Making a soundtrack](14-making-a-soundtrack.md) is the other half of this pair, and the difference
is the verb. There you author a track note by note in a tracker. Here you describe one and a model
writes it — the same relationship [Create](22-generating-references.md) has to
[Inker](28-inker.md), applied to sound.

This walks one path: a description, a take, an audition, a second take with the recipe changed, and
the finished track opened in Sirens as a sample. Ten minutes of your attention and a few minutes of
your card.

You need the ACE-Step weights on disk — about 8.3 GB, and Muse refuses at the door without them
rather than generating something worse. [Installation](39-installation.md) has the download, and
Settings → Models has a button that does it for you.

## Describing what you want

Open **Muse** from the rail. Across the top is the brief: a tags field, a lyrics field, a duration,
a count, and Generate. Everything else — steps, guidance, the scheduler, the seed — is in the recipe
column on the right, and you can ignore all of it for now.

The tags field is not a sentence. ACE-Step was trained on **comma-separated style tags**, and a
paragraph describing a mood in prose gets you something noticeably vaguer than the same mood in
tags. Type this:

```
dark ambient, dungeon, low strings, slow, sparse percussion, minor key
```

Leave the lyrics field empty. An empty lyric block means an instrumental, which is what game music
almost always is, and it is the one field in this mode that is genuinely optional.

Set **Duration** to 60s and **Count** to 2, then press **Generate** (or `Ctrl+Enter`).

Two rows appear in the tray below, one per take, each with its own seed. They queue behind whatever
else the app is doing and run one at a time — a music job holds about the same amount of your card
as an image model does, so it does not overlap with a mesh reconstruction.

The progress bar is music-shaped: a load phase while 8.3 GB comes off disk, then a sampling phase
that counts steps. The load is the slow part on a cold cache and it happens once — a second take
against the same model reuses the loaded pipeline unless the queue has evicted it in between.

## Listening to what came out

When a take finishes, its card's **Play** button lights up. Press it. Press it again — it says
**Stop** now — to silence it, or press Play on the other take to switch straight to it.

Playback needs a sound card. Without one the button says so rather than doing nothing, exactly as
Sirens' transport does.

Compare the two takes. They are the same request at two seeds, which is the comparison the count
control exists for: if both are wrong in the same way it is the tags, and if one is right and one is
not it is the seed, and those two problems have completely different fixes.

## A second pass, with the recipe

Say the takes are the right mood but muddy. Open the recipe column and try one change at a time:

- **Steps** — 60 is the default. Below about 30 the output audibly falls apart; above about 80 you
  are paying for time rather than quality. Lower it while you are auditioning ideas.
- **Guidance** — how closely the model follows your tags. Low wanders and can surprise you; high
  obeys and can flatten. If your tags are being ignored, raise it before you rewrite them.
- **Pin the seed** — off by default, which means every take gets a fresh one. Turn it on once you
  have a take you like and want to change *one other thing* about it.

Change one, press Generate again, and listen to the new pair beside the old. Nothing is overwritten:
every take is its own row with its own file, and the tray keeps them all until you delete them.

## Opening it in Sirens

Press **Open in Sirens** on the take you kept.

The app switches to the tracker and the track lands in the open song's sample table as a sample
instrument. If no song was open, one is started for you.

This is the one bridge between the two audio modes, and it exists because the formats already
agreed: Muse writes a 44.1 kHz WAV and Sirens' sample instruments read 44.1 kHz WAVs. Nothing is
converted and nothing is special about a generated sample — it arrives exactly as a WAV you dragged
in from your own disk would.

What it is for: playing a generated phrase as a one-shot under a chiptune arrangement, chopping a
pad into a sustained instrument, or using a finished track as a reference to write against.

## Where the file is

Every take is a job row like any other, so it is in the [library](36-library-and-jobs.md) with its
prompt, its seed and its recipe, and the file itself is `track.wav` in that job's directory. Export
it the way you export anything else.

There are no loop points. ACE-Step does not produce seamless loops, and a generated track that ends
where it began is luck rather than a feature — so for looping game music, either fade it in your
engine or use it as material in Sirens rather than as a finished loop.

## What to read next

- [Muse](35-muse.md) — the reference chapter: every control, and what each recipe value does.
- [Sirens](34-sirens.md) — the tracker the bridge lands in.
- [Making a soundtrack](14-making-a-soundtrack.md) — the same goal, authored by hand.
