"""The step vocabulary and the thing that answers it, checked against each other.

``studio/tour`` declares what a step may wait for and ``panes/tour`` answers it,
and the two are deliberately separate: the declaration has to stay headless, and
the answer has to read live app state. A split like that drifts unless something
compares the halves, and the failure it drifts into is silent -- an unanswered
name is "never satisfied", which on a point-and-wait step is a tour that hangs
on a screen where the reader has already done the thing.

So this asserts both directions, and it also exercises the evaluator against a
stub rather than trusting the names alone: a condition that is spelled in both
lists and reads the wrong attribute passes a name check and fails a person.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio.panes import tour as tour_pane
from warlock.studio.tour.steps import CONDITIONS


def _ctx(**state):
    base = {"mode": "home", "inker": None, "sirens": None}
    base.update(state)
    return SimpleNamespace(state=SimpleNamespace(**base))


def test_every_declared_condition_is_answered():
    missing = sorted(set(CONDITIONS) - tour_pane.HANDLED)
    assert not missing, (
        f"steps may wait for {missing}, and panes/tour answers none of them -- "
        "which reads as 'never satisfied' and hangs the tour"
    )


def test_the_evaluator_answers_nothing_it_was_not_asked_to():
    extra = sorted(tour_pane.HANDLED - set(CONDITIONS))
    assert not extra, (
        f"panes/tour answers {extra}, which no step may name. Either the "
        "vocabulary wants it or it is dead."
    )


def test_an_unknown_name_is_false_rather_than_an_error():
    """Belt and braces behind the two assertions above.

    They make an unknown name impossible; this makes it harmless if it happens
    anyway. A raise here would take the frame, and a tour that cannot be
    dismissed because drawing it crashes is worse than one that will not
    advance.
    """
    assert tour_pane.satisfied(_ctx(), "no-such-condition", None) is False


def test_manual_is_never_satisfied_by_itself():
    """It is the *absence* of a condition: the reader presses Next."""

    assert tour_pane.satisfied(_ctx(), "manual", None) is False


def test_mode_is_reads_the_current_mode():
    assert tour_pane.satisfied(_ctx(mode="inker"), "mode_is", "inker") is True
    assert tour_pane.satisfied(_ctx(mode="home"), "mode_is", "inker") is False


def test_doc_open_is_false_with_no_workspace_state():
    assert tour_pane.satisfied(_ctx(), "doc_open", "inker") is False
    assert tour_pane.satisfied(_ctx(plotter=None), "doc_open", "plotter") is False


def test_doc_open_sees_an_inker_document():
    inker = SimpleNamespace(active=SimpleNamespace(doc=object()), tool="brush")
    assert tour_pane.satisfied(_ctx(inker=inker), "doc_open", "inker") is True


def test_tool_is_reads_the_inker_tool():
    inker = SimpleNamespace(active=None, tool="eraser")
    assert tour_pane.satisfied(_ctx(inker=inker), "tool_is", "eraser") is True
    assert tour_pane.satisfied(_ctx(inker=inker), "tool_is", "brush") is False
    assert tour_pane.satisfied(_ctx(), "tool_is", "brush") is False


def test_layers_at_least_counts_the_stack():
    doc = SimpleNamespace(stack=[object(), object()], anim=None)
    inker = SimpleNamespace(active=SimpleNamespace(doc=doc), tool="brush")
    assert tour_pane.satisfied(_ctx(inker=inker), "layers_at_least", "2") is True
    assert tour_pane.satisfied(_ctx(inker=inker), "layers_at_least", "3") is False


def test_layers_at_least_survives_a_document_shaped_differently():
    """A missing stack is 'not yet', never a traceback in the frame loop."""

    inker = SimpleNamespace(active=SimpleNamespace(doc=object()), tool="brush")
    assert tour_pane.satisfied(_ctx(inker=inker), "layers_at_least", "2") is False
    doc = SimpleNamespace(stack=[object()], anim=None)
    live = SimpleNamespace(active=SimpleNamespace(doc=doc), tool="brush")
    assert tour_pane.satisfied(_ctx(inker=live), "layers_at_least", "not-a-number") is False


def test_sfx_at_least_counts_the_one_shots():
    doc = SimpleNamespace(oneshots=[object()])
    sirens = SimpleNamespace(active=SimpleNamespace(doc=doc))
    assert tour_pane.satisfied(_ctx(sirens=sirens), "sfx_at_least", "1") is True
    assert tour_pane.satisfied(_ctx(sirens=sirens), "sfx_at_least", "2") is False
    # No mode open at all, which is the state the step before this one is in.
    assert tour_pane.satisfied(_ctx(), "sfx_at_least", "1") is False


def test_sfx_at_least_survives_a_document_shaped_differently():
    """The ``layers_at_least`` rule: 'not yet', never a traceback in the loop."""

    sirens = SimpleNamespace(active=SimpleNamespace(doc=object()))
    assert tour_pane.satisfied(_ctx(sirens=sirens), "sfx_at_least", "1") is False
    live = SimpleNamespace(active=SimpleNamespace(doc=SimpleNamespace(oneshots=[])))
    assert tour_pane.satisfied(_ctx(sirens=live), "sfx_at_least", "not-a-number") is False
    # A song with no effects yet answers a zero threshold, which is what makes
    # the count a threshold rather than a "has any".
    assert tour_pane.satisfied(_ctx(sirens=live), "sfx_at_least", "0") is True


def test_animated_reads_the_timeline():
    still = SimpleNamespace(stack=[], anim=None)
    moving = SimpleNamespace(stack=[], anim=object())
    assert (
        tour_pane.satisfied(
            _ctx(inker=SimpleNamespace(active=SimpleNamespace(doc=still), tool="brush")),
            "animated",
            None,
        )
        is False
    )
    assert (
        tour_pane.satisfied(
            _ctx(inker=SimpleNamespace(active=SimpleNamespace(doc=moving), tool="brush")),
            "animated",
            None,
        )
        is True
    )
