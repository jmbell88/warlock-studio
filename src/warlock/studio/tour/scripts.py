"""The tours themselves. Data, not code.

Four of them, and each is chosen rather than convenient.

``first-hour`` orients someone who has just opened the app, and **every step of
it runs on a machine with no GPU and no weights** -- the one step that could ask
for a generation asks politely and advances on Next instead of waiting, because
a point-and-wait step that waits for something the machine cannot do is a trap
rather than a lesson.

``inker-basics`` proves the same machinery against a real document editor, and
it was picked over Create for the same reason: drawing needs no card, no weights
and no subprocess, so every reader can finish it and so can the smoke test.

``sirens-basics`` and ``sirens-sfx`` are beside it and for that same reason --
a tracker needs no GPU, no weights and no subprocess either -- with one honest
difference: the mode
is about a *sound*, and a machine with no audio device cannot make one. So no
step of it waits on hearing anything. Every ``done`` condition here is a mode, a
document, or notes the reader has written -- never a sound -- which is what
lets somebody on a silent box finish the tour and still have written a bar.

Each tour ends by handing the reader to the chapter that goes deeper. The link
runs one way -- a step names a chapter, a chapter never names a step -- which is
what keeps the manual's parser strict and its pages free of buttons.
"""

from __future__ import annotations

from .steps import Condition, Step, Tour

FIRST_HOUR = Tour(
    key="first-hour",
    title="Around the app",
    blurb="Four minutes. What the parts are, and the one thing worth knowing early.",
    steps=(
        Step(
            id="welcome",
            title="This is Warlock Studio",
            body=(
                "It makes game art on this machine: 3D models, sprite sheets, tile maps, "
                "drawings and music. Nothing here talks to a server.\n\n"
                "This tour points at things and waits for you. It never clicks anything "
                "for you, and Esc ends it at any point."
            ),
            mode="home",
            chapter=("01-before-you-begin", None),
        ),
        Step(
            id="rail",
            title="The rail",
            body=(
                "Everything lives behind these. The top group is the pipeline that turns "
                "an idea into a finished asset; the group under it is the eight workspaces "
                "you edit in. Settings sits at the bottom.\n\n"
                "Ctrl+K opens the command palette, which reaches every one of them by "
                "name -- it is the keyboard's way through this tour and through the app."
            ),
            anchor="rail/home",
        ),
        Step(
            id="open-create",
            title="Open Create",
            body="Click it -- or Ctrl+K and type its name -- and I will carry on.",
            anchor="rail/create",
            done=Condition("mode_is", "create"),
        ),
        Step(
            id="stages",
            title="An asset's five stages",
            body=(
                "Reference, Mesh, Rig, Pose, Export. That is the whole path an asset "
                "takes, and it is a breadcrumb rather than a wizard -- you can step back "
                "to any of them."
            ),
            mode="create",
            anchor="create/stages",
        ),
        Step(
            id="prompt",
            title="Describe one thing",
            body=(
                "A subject, not a scene. Two objects in the picture become one fused "
                "mesh, so composition matters more than adjectives here.\n\n"
                "Try: a mossy stone well, weathered, fantasy game prop"
            ),
            mode="create",
            anchor="create/prompt",
            chapter=("02-your-first-asset", "stage-one-the-reference"),
        ),
        # Optional in the sense that matters here: it points at a *type* rather
        # than a control, and the reader can pass it in one press. It earns its
        # place in this tour on the same rule the others do -- ``first-hour``
        # runs on a machine with no GPU and no weights, and Character is the
        # one generation type that does too, so it is the only thing on this
        # screen a reader without a card can be told to try rather than told
        # about. It waits on ``mode_is create`` rather than Next: the reader is
        # already there, so it advances the moment they are ready, but a reader
        # who wandered off to look at the rail is not left with a card
        # describing a screen they cannot see.
        Step(
            id="character-type",
            title="One type needs no card",
            body=(
                "Set Generation type to Character and the column changes: no model, "
                "no seed, no reference. Warlock builds the body itself from a registry "
                "of species, rigs it and renders a sprite sheet, all on the CPU.\n\n"
                "Try: fire ogre, 3/4 top down sprite sheet -- and watch the column "
                "read your words back. If you would rather carry on here, Next."
            ),
            mode="create",
            done=Condition("mode_is", "create"),
            chapter=("22-generating-references", "characters"),
        ),
        Step(
            id="generate",
            title="Generate",
            body=(
                "Press it if the image model is installed. If it is not, press it "
                "anyway -- the refusal names what is missing and offers to fetch it, "
                "which is worth seeing once.\n\n"
                "Either way, press Next when you are ready."
            ),
            mode="create",
            anchor="create/generate",
        ),
        Step(
            id="the-gate",
            title="It stops at a picture, on purpose",
            body=(
                "A prompt does not produce a mesh. It produces a reference image and "
                "waits, because reconstruction costs about two minutes of GPU and the "
                "biggest factor in the result is the picture it started from.\n\n"
                "You look, then you press Make 3D. That pause is the single most "
                "important thing in the app."
            ),
            mode="create",
            chapter=("02-your-first-asset", "why-there-are-two-stages"),
        ),
        Step(
            id="library",
            title="The library is also the queue",
            body=(
                "There is no separate queue screen. A running job and a finished one are "
                "the same kind of row in this list -- one with a progress bar, one with a "
                "picture. Open it and have a look."
            ),
            anchor="rail/library",
            done=Condition("mode_is", "library"),
            chapter=("03-finding-your-work", None),
        ),
        Step(
            id="done",
            title="That is the shape of it",
            body=(
                "Two things worth remembering: F1 opens the manual about whatever you "
                "are looking at, and Ctrl+K is a command palette that reaches anywhere.\n\n"
                "The tutorial chapters go from here."
            ),
            chapter=("01-before-you-begin", "what-to-read-next"),
        ),
    ),
)

INKER_BASICS = Tour(
    key="inker-basics",
    title="Drawing in Inker",
    blurb="Five minutes. A canvas, a stroke, a layer and a frame. No GPU needed.",
    steps=(
        Step(
            id="open-inker",
            title="Open Inker",
            body=(
                "The raster editor: layers, a timeline, palettes and twenty-four tools. "
                "None of it needs a graphics card or a downloaded model."
            ),
            anchor="rail/inker",
            done=Condition("mode_is", "inker"),
        ),
        Step(
            id="new-doc",
            title="Start a canvas",
            body="Ctrl+N. Something small is easiest to see -- 32 by 32 will do.",
            mode="inker",
            done=Condition("doc_open", "inker"),
        ),
        Step(
            id="toolbox",
            title="The toolbox",
            body=(
                "Twenty-four tools in twelve slots. Each slot has a letter, and pressing "
                "that letter again cycles within the slot -- so B is the brush, and B "
                "again is the spray."
            ),
            mode="inker",
            anchor="inker/tools",
            chapter=("05-drawing", "tools"),
        ),
        Step(
            id="brush",
            title="Take the brush",
            body="Press B, or click it.",
            mode="inker",
            anchor="inker/tools",
            done=Condition("tool_is", "brush"),
        ),
        Step(
            id="draw",
            title="Draw something",
            body=(
                "Anything. If the edges come out soft and you wanted them hard, that is "
                "the nib rather than the brush size -- a pixel nib gives you one "
                "aliased pixel at a time."
            ),
            mode="inker",
        ),
        Step(
            id="colour",
            title="Colour",
            body=(
                "Hold Alt with any paint tool for a temporary eyedropper. The mode "
                "switch at the top of this panel is worth knowing about: Indexed stores "
                "a palette index per pixel, so nothing can drift off-palette."
            ),
            mode="inker",
            anchor="inker/colors",
            chapter=("05-drawing", "colour"),
        ),
        Step(
            id="picker",
            title="The picker",
            body=(
                "RGB, HSV, HSL and Gray over the same colour, with a hex field that "
                "takes what somebody pasted you.\n\n"
                "On an indexed drawing, with a colour picked out of the palette, these "
                "sliders edit that palette entry -- so every pixel painted in it "
                "changes at once, in one undo step."
            ),
            mode="inker",
            anchor="inker/picker",
            chapter=("28-inker", "the-colour-picker"),
        ),
        Step(
            id="layer",
            title="Add a layer",
            body=(
                "Ctrl+Shift+N, or the button on this panel.\n\n"
                "The layers list and the timeline are one panel here, which is why "
                "a layer row and a timeline row look alike -- they are the same row."
            ),
            mode="inker",
            anchor="inker/layers",
            done=Condition("layers_at_least", "2"),
        ),
        Step(
            id="blend",
            title="Layers do more than stack",
            body=(
                "Each one has an opacity and one of nineteen blend modes. Set this one "
                "to Multiply at about 60% and paint shadow on it.\n\n"
                "Undo here is addressed by layer identity rather than by position, so "
                "reordering the stack never sends a later undo to the wrong layer."
            ),
            mode="inker",
            anchor="inker/layers",
        ),
        Step(
            id="animate",
            title="Give it a timeline",
            body=(
                "Press Animate this drawing. Your layers become the first column of a "
                "grid, and nothing about them changes."
            ),
            mode="inker",
            anchor="inker/timeline",
            done=Condition("animated"),
        ),
        Step(
            id="frames",
            title="Copy a frame, or link one",
            body=(
                "Copying gives you independent pixels. Linking puts the *same* cel in "
                "two places, so painting either changes both.\n\n"
                "Getting that backwards is the classic first-hour mistake, and it is "
                "also how you hold a background still across thirty frames."
            ),
            mode="inker",
            anchor="inker/timeline",
            chapter=("06-animating", "copy-versus-link"),
        ),
        Step(
            id="generation",
            title="Getting it out again",
            body=(
                "Make 3D hands the drawing to the mesh pipeline; Save as reference "
                "puts it in the library; Add to Packwright sends it to the atlas "
                "packer.\n\n"
                "A verb you cannot use right now is greyed with the reason in its "
                "tooltip rather than hidden -- Revert to original needs a document "
                "that came out of the library in the first place."
            ),
            mode="inker",
            anchor="inker/generate",
            chapter=("28-inker", "pipeline-bridges"),
        ),
    ),
)

SIRENS_BASICS = Tour(
    key="sirens-basics",
    title="Writing a tune in Sirens",
    blurb="Five minutes. A pattern, an instrument, an envelope and a WAV. No GPU needed.",
    steps=(
        Step(
            id="open-sirens",
            title="Open Sirens",
            body=(
                "The chiptune tracker: five NES-shaped voices, a grid you type notes into, "
                "and a song that exports as WAV. No graphics card and no downloaded model.\n\n"
                "If this machine has no sound device you can still do every step of this "
                "tour -- you simply will not hear it, and the transport says so."
            ),
            anchor="rail/sirens",
            done=Condition("mode_is", "sirens"),
        ),
        Step(
            id="new-song",
            title="Start a song",
            body=(
                "Ctrl+N.\n\n"
                "It is not an empty document. Five channels, one 64-row pattern, one "
                "instrument per voice kind, and an order that already points at the "
                "pattern -- so the first note you type makes a sound."
            ),
            mode="sirens",
            done=Condition("doc_open", "sirens"),
        ),
        Step(
            id="the-grid",
            title="The grid",
            body=(
                "One pattern at a time, five columns per channel: note, instrument, "
                "volume, effect, parameter. A row is a sixteenth note, and every fourth "
                "row -- one beat -- has a stripe behind it.\n\n"
                "The dots are not decoration. A run of them is how the eye finds the rows "
                "where something happens.\n\n"
                "One thing to know before you try it: which column the caret is in decides "
                "what a key means. c is a note in the first column and the hex digit twelve "
                "in the third."
            ),
            mode="sirens",
            anchor="sirens/grid",
            chapter=("34-sirens", "the-pattern-grid"),
        ),
        Step(
            id="type-a-bar",
            title="Type a bassline",
            body=(
                "Click into the Triangle channel's note column. Set Octave to 3 and Step "
                "to 4 in the strip over the grid, then type z z v x.\n\n"
                "The keyboard is a piano: zsxdcvgbhnjm is the octave Octave names, and "
                "q2w3er5t6y7u is the one above it. It only fires in the note column -- "
                "e in the effect column is the letter of an effect."
            ),
            mode="sirens",
            anchor="sirens/grid",
            # The pivotal step, and the one that had no business advancing on
            # Next: it asks for four notes, so it waits for four notes. Counted
            # over the whole document rather than the bar the step names --
            # somebody who typed their four somewhere else has still learned
            # the thing this step teaches.
            done=Condition("notes_at_least", "4"),
            chapter=("14-making-a-soundtrack", "a-bassline-on-the-triangle"),
        ),
        Step(
            id="transport",
            title="Play it",
            body=(
                "Space, or the button here.\n\n"
                "The first press may say it is still rendering. That is the design rather "
                "than a delay to apologise for: the whole song is synthesised into a "
                "buffer and the buffer is played, so what you hear is bit-for-bit what "
                "the exported WAV will contain."
            ),
            mode="sirens",
            anchor="sirens/transport",
            chapter=("34-sirens", "playing-it"),
        ),
        Step(
            id="instruments",
            title="Instruments are numbered, and the number is in the cell",
            body=(
                "A typed note stamps whichever instrument is selected here into the "
                "grid's instrument column -- without that, a typed note is silent for a "
                "reason nothing on screen explains.\n\n"
                "The number on each row is what the cell holds. It is a slot rather than "
                "a position, so removing one does not renumber the notes that named the "
                "others."
            ),
            mode="sirens",
            anchor="sirens/instruments",
            chapter=("34-sirens", "instruments"),
        ),
        Step(
            id="envelopes",
            title="Drag a shape into one",
            body=(
                "A chiptune instrument is four short lists of numbers stepped once a "
                "tick, and the shape of the list is the sound. Drag across the Volume "
                "graph, high on the left falling to nothing.\n\n"
                "The whole drag is one undo step, and painting past the end lengthens "
                "the sequence. The release marker splits it: everything from there is "
                "the tail after the note ends, and the tail never loops."
            ),
            mode="sirens",
            anchor="sirens/envelopes",
            chapter=("34-sirens", "the-envelope-editor"),
        ),
        Step(
            id="order",
            title="Patterns and the order are two lists",
            body=(
                "Adding a pattern does not add it to the order, and removing an order "
                "entry does not delete the pattern. That is the point: one pattern can "
                "appear in the order three times.\n\n"
                "Tick Loop at the end. The render then carries loop points, and they go "
                "into the exported WAV's smpl chunk -- which is what makes a track a "
                "soundtrack rather than something that restarts."
            ),
            mode="sirens",
            anchor="sirens/orders",
            chapter=("34-sirens", "patterns-and-the-order"),
        ),
        Step(
            id="effects",
            title="Sound effects live in the same document",
            body=(
                "Press Add. You get an effect, a little pattern of its own, and the grid "
                "pointing straight at it -- so the grid is the effect editor and there is "
                "no second one to learn.\n\n"
                "An effect keeps its own tempo, because a coin pickup is forty "
                "milliseconds whatever the music is doing. The play button on the row "
                "auditions it without touching the song's own buffer."
            ),
            mode="sirens",
            anchor="sirens/effects",
            chapter=("34-sirens", "sound-effects"),
        ),
        Step(
            id="export",
            title="Getting the audio out",
            body=(
                "Export audio... asks for a folder rather than a filename, because it "
                "writes song.wav, one WAV per channel under stems/ and one per sound "
                "effect under sfx/.\n\n"
                "The .wsng is the composition and every WAV is derived from it: export "
                "an untouched document twice and the files are byte-identical, so a "
                "build script can regenerate them."
            ),
            mode="sirens",
            anchor="sirens/bridge",
            chapter=("34-sirens", "exporting-the-audio"),
        ),
    ),
)

SIRENS_SFX = Tour(
    key="sirens-sfx",
    title="Making a sound effect",
    blurb="Three minutes. A blip, a shape, an audition and a WAV. No GPU needed.",
    steps=(
        Step(
            id="open-sirens",
            title="Open Sirens",
            body=(
                "A sound effect lives in the same document a song does, and is made with "
                "the same grid. That is the one idea this tour is about: there is no "
                "separate effect editor to learn.\n\n"
                "If this machine has no sound device you can still do every step -- you "
                "simply will not hear the result, and the audition button says so."
            ),
            anchor="rail/sirens",
            done=Condition("mode_is", "sirens"),
        ),
        Step(
            id="new-song",
            title="Start a document",
            body=(
                "Ctrl+N.\n\n"
                "Even if you only want effects, you start from a song: the effects are "
                "kept inside it, and the instruments they play through are the ones it "
                "already has."
            ),
            mode="sirens",
            done=Condition("doc_open", "sirens"),
        ),
        Step(
            id="find-effects",
            title="Find the Sound effects panel",
            body=(
                "Right-hand column, below Instruments and Envelopes.\n\n"
                "An effect is a name, a tempo, a speed and a pattern of its own. Nothing "
                "else -- which is why the grid can edit it."
            ),
            mode="sirens",
            anchor="sirens/effects",
            chapter=("34-sirens", "sound-effects"),
        ),
        Step(
            id="add-one",
            title="Press Add",
            body=(
                "One press makes three things: the effect, an eight-row pattern belonging "
                "to it, and a caret move -- the grid in the middle of the window is now "
                "showing that pattern rather than the song's.\n\n"
                "It is a single undo step, so Ctrl+Z takes all of it back together."
            ),
            mode="sirens",
            anchor="sirens/effects",
            done=Condition("sfx_at_least", "1"),
        ),
        Step(
            id="the-grid-moved",
            title="The grid is now the effect editor",
            body=(
                "Look at the strip above the grid: it names the pattern you are editing "
                "and says whether it is a song pattern or a sound effect. That readout is "
                "the answer to 'what am I typing into', and it is worth checking whenever "
                "a keystroke does something you did not expect.\n\n"
                "Eight rows rather than sixty-four. An effect is short by construction."
            ),
            mode="sirens",
            chapter=("34-sirens", "the-pattern-grid"),
        ),
        Step(
            id="write-a-blip",
            title="Type two notes",
            body=(
                "Click the first row's note column and type a note -- z is C, s is C#, x "
                "is D, and the octave is on the strip above.\n\n"
                "Then go down two rows and type a note several steps higher. Two notes a "
                "few rows apart is already a blip; a rising pair reads as a pickup and a "
                "falling pair as a hit."
            ),
            mode="sirens",
            # "Type two notes", so: two. The count is the step's own text
            # rather than a number picked to be easy.
            done=Condition("notes_at_least", "2"),
            chapter=("14-making-a-soundtrack", "a-sound-effect"),
        ),
        Step(
            id="pick-an-instrument",
            title="Choose the voice it plays through",
            body=(
                "The second column is the instrument number, and the panel here is where "
                "those are defined.\n\n"
                "For an effect the noise channel's instrument is usually the interesting "
                "one -- it is what a hit, a step or an explosion is made of, where the "
                "pulse voices read as musical."
            ),
            mode="sirens",
            anchor="sirens/instruments",
            chapter=("34-sirens", "instruments"),
        ),
        Step(
            id="shape-it",
            title="Give it a shape",
            body=(
                "An envelope is what makes two notes a sound rather than a beep. Drag the "
                "volume envelope's decay in and the blip becomes percussive; leave it flat "
                "and it stays a tone.\n\n"
                "This is the control that most changes how an effect reads, and it is the "
                "one nothing on the grid can tell you about."
            ),
            mode="sirens",
            anchor="sirens/envelopes",
            chapter=("34-sirens", "the-envelope-editor"),
        ),
        Step(
            id="tempo-and-speed",
            title="Tempo and speed belong to the effect",
            body=(
                "Back in the Sound effects panel, the two sliders under the name are the "
                "effect's own -- not the song's.\n\n"
                "That is deliberate: a coin pickup is forty milliseconds whatever the "
                "music happens to be doing. Raise the tempo and the same eight rows play "
                "faster, which is usually how a blip becomes a *blip*."
            ),
            mode="sirens",
            anchor="sirens/effects",
            chapter=("34-sirens", "sound-effects"),
        ),
        Step(
            id="audition",
            title="Audition it",
            body=(
                "The play button on the effect's own row. It plays that effect and leaves "
                "the song's buffer alone, so pressing Space afterwards still plays the "
                "song rather than the effect.\n\n"
                "On a machine with no sound device the button is disabled and says why. "
                "Everything else on this tour still works."
            ),
            mode="sirens",
            anchor="sirens/effects",
            chapter=("34-sirens", "sound-effects"),
        ),
        Step(
            id="export",
            title="Getting it out",
            body=(
                "Export audio... asks for a folder, and every sound effect is written into "
                "sfx/ as its own WAV, named after the effect.\n\n"
                "They are a pure function of the document: export an untouched one twice "
                "and the files are byte-identical, so a build script can regenerate them "
                "rather than keeping them in the repository."
            ),
            mode="sirens",
            anchor="sirens/bridge",
            chapter=("34-sirens", "exporting-the-audio"),
        ),
    ),
)

MUSE_BASICS = Tour(
    key="muse-basics",
    title="Generating a track in Muse",
    blurb=(
        "Five minutes. Tags, a take, an audition, and the track opened in Sirens. "
        "Needs the ACE-Step weights."
    ),
    steps=(
        Step(
            id="open-muse",
            title="Open Muse",
            body=(
                "The other audio mode. Sirens is a tracker you write note by note; this "
                "one is a model you describe a piece of music to.\n\n"
                "It needs about 8.3 GB of weights on disk. Without them it refuses at the "
                "door and names the download rather than making something worse."
            ),
            anchor="rail/muse",
            done=Condition("mode_is", "muse"),
        ),
        Step(
            id="tags",
            title="Describe it in tags",
            body=(
                "Comma-separated style tags, not a sentence. The model was trained on tag "
                "strings, and a paragraph of prose gets you a vaguer result than the same "
                "mood in tags.\n\n"
                "Try: dark ambient, dungeon, low strings, slow, minor key."
            ),
            mode="muse",
            anchor="muse/tags",
            chapter=("35-muse", "what-it-takes"),
        ),
        Step(
            id="lyrics",
            title="Leave the lyrics empty",
            body=(
                "The second field takes a lyric block marked up with [verse] and [chorus]. "
                "Empty means an instrumental, which is what game music almost always is -- "
                "so this is the one field here you can ignore entirely."
            ),
            mode="muse",
            anchor="muse/lyrics",
            chapter=("35-muse", "what-it-takes"),
        ),
        Step(
            id="generate",
            title="Ask for two takes",
            body=(
                "Set the count to 2 and press Generate.\n\n"
                "Two is not twice the value of one: they are the same request at two "
                "seeds, and comparing them tells you which problem you have. Both wrong "
                "the same way is your tags. One right and one not is the seed."
            ),
            mode="muse",
            anchor="muse/generate",
            chapter=("35-muse", "the-window"),
        ),
        Step(
            id="audition",
            title="Listen to them",
            body=(
                "Each finished take gets a card with a Play button. Press it, then press "
                "the other one -- starting a second take stops the first, because there is "
                "one sound card and this app does not pretend otherwise.\n\n"
                "With no sound device the button says so rather than doing nothing."
            ),
            mode="muse",
            chapter=("35-muse", "the-takes"),
        ),
        Step(
            id="into-sirens",
            title="Take it into Sirens",
            body=(
                "Open in Sirens imports the track into the tracker as a sample "
                "instrument.\n\n"
                "It works because nothing had to be built for it: Muse writes a 44.1 kHz "
                "WAV and Sirens' sample instruments read 44.1 kHz WAVs, so a generated "
                "track arrives through exactly the door a file you dragged in arrives "
                "through."
            ),
            mode="muse",
            chapter=("35-muse", "the-bridge-to-sirens"),
        ),
    ),
)

#: Every tour, in offer order.
TOURS: tuple[Tour, ...] = (
    FIRST_HOUR,
    INKER_BASICS,
    MUSE_BASICS,
    SIRENS_BASICS,
    SIRENS_SFX,
)


def find(key: str) -> Tour | None:
    """The tour ``key`` names, or ``None``.

    ``None`` rather than raising: the key comes out of saved settings, and a
    tour renamed between releases must not take the frame that restored it.
    """
    return next((tour for tour in TOURS if tour.key == key), None)


__all__ = [
    "FIRST_HOUR",
    "INKER_BASICS",
    "MUSE_BASICS",
    "SIRENS_BASICS",
    "SIRENS_SFX",
    "TOURS",
    "find",
]
