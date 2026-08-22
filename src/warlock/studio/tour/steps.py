"""What a tour step is, and the vocabulary of things it can wait for.

A step is data. It names a mode it is about, a control to point at, a sentence
to say, and **what would make it finished** -- and that last one is why the
tours can be point-and-wait rather than a slideshow with a Next button.

The waiting condition is a *named predicate plus an argument*, never a callable.
A closure would have to reach into live app state, which would drag imgui and
``service`` into this module and cost it the headless tests that are the reason
it is separate. Instead the drawing half builds one small snapshot per frame and
answers the name. ``tests/tour/test_tour_conditions.py`` asserts the two
vocabularies match in both directions, so a step waiting on a name nobody
evaluates fails the suite rather than hanging the tour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Every condition name the evaluator must answer. A step naming anything else
#: is a test failure, not a runtime surprise -- an unknown name would read as
#: "never satisfied", which on a point-and-wait step is indistinguishable from
#: the app being broken.
CONDITIONS: tuple[str, ...] = (
    # Advance when the reader presses Next. For steps that explain rather than
    # ask -- there is no way to detect having read a sentence.
    "manual",
    # ``arg`` is a mode key: satisfied once that mode is on screen.
    "mode_is",
    # ``arg`` is a workspace mode key: satisfied once it holds a document.
    "doc_open",
    # ``arg`` is an Inker tool key: satisfied while that tool is selected.
    "tool_is",
    # ``arg`` is a count: the active Inker document has at least that many
    # layers.
    "layers_at_least",
    # The active Inker document has a timeline.
    "animated",
)


@dataclass(frozen=True, slots=True)
class Condition:
    """One thing a step waits for. ``name`` must be in :data:`CONDITIONS`."""

    name: str
    arg: str | None = None


#: The step that advances on Next, spelled once so the tours do not each say it.
MANUAL = Condition("manual")


@dataclass(frozen=True, slots=True)
class Step:
    """One thing to point at and one thing to say about it.

    ``anchor`` is a key into ``studio.anchors``, recorded by the pane that draws
    the control. A step whose anchor did not draw this frame still shows its
    card -- it simply has nothing to ring, which is the right behaviour when a
    control is inside a collapsed section or behind a tab.
    """

    id: str
    title: str
    body: str
    mode: str | None = None
    anchor: str | None = None
    done: Condition = MANUAL
    #: ``(chapter_key, anchor_or_None)`` for a "read more" link into the manual.
    chapter: tuple[str, str | None] | None = None


@dataclass(frozen=True, slots=True)
class Tour:
    """A named sequence of steps, and the sentence that offers it."""

    key: str
    title: str
    blurb: str
    steps: tuple[Step, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.steps)

    def step(self, index: int) -> Step | None:
        """The step at ``index``, or ``None`` past the end.

        Clamped rather than raising, because the index lives in session state
        and a tour edited between sessions would otherwise crash the frame that
        restored it.
        """
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None
