"""What "everything", "the opposite", "delete this" and "copy this" mean.

Four operations that are entirely about the *document* -- which objects or
elements are selected, and what happens to them -- and that had come to live in
``studio/clay_mode.py``, the mode layer, because that is where the keyboard
shortcuts firing them are. The ops registry then had to import the mode layer to
reach them, and the mode layer imports the registry: a cycle between a
UI-shaped module and a headless one, held together by both sides importing the
other lazily inside function bodies.

They are here instead, in the headless package, where the rest of this
vocabulary already is. The mode layer keeps same-signature wrappers (its
privates are what the panes and the tests call), and ``clay_ops`` now calls
straight through to this module -- which is what makes ``clay_ops`` free of any
reference to ``clay_mode`` at all.

**Nothing here toasts.** A refusal is a string handed back to the caller, and
the caller decides what a refusal looks like -- the mode layer shows a toast,
and a test reads the list. That is the same boundary the rest of ``clay/``
keeps, and it is why this module can be tested without a ``ctx``.
"""

from __future__ import annotations

from typing import Any

from . import elements as el

__all__ = ["delete_selected", "duplicate_selected", "invert", "select_all"]


def select_all(doc: Any) -> None:
    """Everything *visible*, in the current mode's sense of everything.

    The object branch used to take every object and the element branch below it
    has always skipped the invisible ones, which is the asymmetry rather than a
    choice: ``visible=False`` is documented to mean an object does not render,
    does not export and **cannot be picked**, and Ctrl+A was picking them. It
    only showed once merge existed -- Ctrl+A then Ctrl+M pulled geometry the
    user could not see into the survivor -- but Delete and Duplicate were reading
    the same selection all along.
    """
    if doc.element_mode == "object":
        doc.select([obj.uid for obj in doc.objects if obj.visible])
        return
    for obj in doc.objects:
        if obj.visible:
            doc.set_element_sel(obj.uid, el.select_all(obj.mesh, doc.element_mode))


def invert(doc: Any) -> None:
    """Ctrl+Shift+I. In object mode it inverts which objects are selected."""
    if doc.element_mode == "object":
        # Visible only, for ``select_all``'s reason: inverting into a hidden
        # object selects something the user cannot see or click back off.
        doc.select([o.uid for o in doc.objects if o.visible and o.uid not in doc.selection])
        return
    for obj in doc.objects:
        if not obj.visible:
            continue
        doc.set_element_sel(
            obj.uid,
            el.invert(obj.mesh, doc.element_sel_of(obj.uid), doc.element_mode),
        )


def delete_selected(doc: Any) -> list[str]:
    """Delete what is selected, in whatever sense the current mode means it.

    -> the refusals, one sentence each, for the caller to show. A refusal on one
    object does not abandon the others, which is the ``run_mesh_op`` rule.

    In an element mode this **never** falls through to removing objects. A user
    in face mode pressing Delete means "get rid of these faces", and quietly
    deleting the whole object instead is the most destructive possible
    misreading of one keystroke -- the more so because the object selection in
    an element mode is derived from the element selection, so every object with
    anything selected inside it would go.
    """
    from . import ops_topo

    if doc.element_mode == "object":
        for uid in list(doc.selection):
            doc.remove_object(uid)
        return []
    refusals: list[str] = []
    for uid in list(doc.element_sel):
        obj = doc.by_uid(uid)
        faces = el.convert(obj.mesh, doc.element_sel_of(uid), "face")
        if el.is_empty(faces):
            continue
        try:
            mesh, sel = ops_topo.delete_faces(obj.mesh, faces)
        except el.OpError as error:
            refusals.append(str(error))
            continue
        doc.set_mesh(uid, mesh, select=sel)
    return refusals


def duplicate_selected(doc: Any) -> list[int]:
    """A copy of every selected object, selected in its place. -> the new uids.

    ``taken`` grows as it goes so two copies of one name in a single press do
    not both land on ``Box.001``.
    """
    from . import document as bd
    from . import ops

    taken = [obj.name for obj in doc.objects]
    fresh = []
    for uid in list(doc.selection):
        copy = ops.duplicate(doc.by_uid(uid), bd.new_uid(), taken=taken)
        taken.append(copy.name)
        doc.add_object(copy)
        fresh.append(copy.uid)
    if fresh:
        doc.select(fresh)
    return fresh
