"""The tours themselves. Data, not code.

Two of them in this wave, and the pair is chosen rather than convenient.

``first-hour`` orients someone who has just opened the app, and **every step of
it runs on a machine with no GPU and no weights** -- the one step that could ask
for a generation asks politely and advances on Next instead of waiting, because
a point-and-wait step that waits for something the machine cannot do is a trap
rather than a lesson.

``inker-basics`` proves the same machinery against a real document editor, and
it was picked over Create for the same reason: drawing needs no card, no weights
and no subprocess, so every reader can finish it and so can the smoke test.

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
                "It makes game art on this machine: 3D models, sprite sheets, tile maps "
                "and drawings. Nothing here talks to a server.\n\n"
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
                "an idea into a finished asset; the group under it is the six workspaces "
                "you edit in. Settings sits at the bottom."
            ),
            anchor="rail/home",
        ),
        Step(
            id="open-create",
            title="Open Create",
            body="Click it, and I will carry on.",
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
    ),
)

#: Every tour, in offer order.
TOURS: tuple[Tour, ...] = (FIRST_HOUR, INKER_BASICS)


def find(key: str) -> Tour | None:
    """The tour ``key`` names, or ``None``.

    ``None`` rather than raising: the key comes out of saved settings, and a
    tour renamed between releases must not take the frame that restored it.
    """
    return next((tour for tour in TOURS if tour.key == key), None)


__all__ = ["FIRST_HOUR", "INKER_BASICS", "TOURS", "find"]
