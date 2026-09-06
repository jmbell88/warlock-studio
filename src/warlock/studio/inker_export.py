"""Inker's exports: every file the mode writes that is not a save.

One door per output -- a flattened PNG, the slices, a sheet, a GIF, a PNG
sequence, a range, a tag, a tag each, a layer each -- plus the stepper they all
share (_begin_export/pump_export/_submit_export), which walks the frames on the
frame thread and hands one payload to a task.

Lifted out of ``studio/inker_mode`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed, so the
move is code motion over tested behaviour rather than a rewrite.

``inker_mode`` is imported as a *module* and never ``from``-imported: every
attribute is resolved at call time, so this file and its parent may be
imported in either order. The parent serves these names back through a PEP
562 ``__getattr__``, which is what keeps ``inker_mode.export_png`` and the
rest working for every caller and every test.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pipelines import sheet as sheetlib
from . import atomic, dialogs, icons, inker_mode
from .inker_state import InkerDoc


def export_png(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """A flattened PNG. Not a save: it does not change what the tab points at,
    so the document stays dirty against its own file.

    ``repeat`` writes straight to the recorded destination and opens no dialog
    -- Repeat Last Export (6.9). The *destination* is what makes that safe to
    do silently: it is a path this user chose for this document, and the toast
    afterwards names it.
    """
    tab = tab or inker_mode.active(ctx)
    if tab is None or tab.saving:
        return
    inker_mode.stop_play(tab)  # settle the stack before capturing; see save()
    doc = tab.doc
    # Not a save, but the same rule about what is on the canvas: the composite
    # a floating buffer draws into is the pane's, not the document's, so an
    # export would otherwise be missing pixels the user is looking at.
    inker_mode._settle(ctx, tab)
    suggested = tab.path.stem if tab.path else "untitled"
    state = ctx.state.inker
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))

    recorded = tab.export_dest if repeat else None

    def run() -> dict[str, Any] | None:
        dest = recorded or dialogs.save_file(
            "Export flattened PNG", f"{suggested}.png", inker_mode.PNG_FILTER
        )
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        atomic.write_bytes(dest, doc.png_bytes(scale=scale))
        # ``dest`` and ``export_kind`` so the *next* repeat has something to
        # repeat -- ``on_task_done`` records both.
        return {"exported": dest, "dest": dest, "export_kind": "png"}

    inker_mode._start(ctx, tab, f"inker-export:{tab.uid}", run)


#: A filesystem-safe stem: the same character set ``plotter.tmx._stem`` allows,
#: because both are a slice's or a tileset's *name* headed for a filename and
#: there is no reason for the two rules to disagree.
_SLICE_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slice_filenames(entries: list[Any]) -> list[str]:
    """Sanitised, collision-free file stems, one per slice, in document order.

    S1 does not pin slice names unique -- two slices called "Hitbox" are a
    legitimate authoring choice -- so this is where the collision is caught: the
    first "Hitbox" keeps its name and the second becomes "Hitbox_2", the same
    shape ``tmx._stem`` uses for two tilesets that share a name, with a trailing
    counter rather than a leading index because a human picks these files off a
    folder listing instead of an engine matching them by position.

    Every candidate is checked against **every name already handed out**, not
    against other occurrences of the same sanitised base -- a counter kept per
    base independently of the others can mint the same bumped name twice (a
    third "Hitbox" landing on "Hitbox_2" a second slice already claimed, or a
    literal "a_2" colliding with what a repeated "a" bumps to), which is a
    silent overwrite in ``run()`` rather than a name a user ever sees.

    **This module holds two collision policies and the split is deliberate.**
    :func:`_split_stems` *refuses* where this bumps, and the deciding question
    is whether anything downstream addresses the file **by name**. Nothing
    addresses a slice PNG by name -- a human picks it off a folder listing, and
    "Hitbox_2.png" is a name they can read and live with. A tag or a layer *is*
    addressed by name by whatever consumes the sheet, so a second "walk"
    quietly becoming "walk_2.png" would be a file claiming to be a clip that
    does not exist. Bumping is friendly where a human disambiguates and
    dishonest where a machine does. A third naming helper answers that same
    question before it picks a side, and ``tests/inker/test_slice_export.py``
    pins both halves against each other so neither can drift onto the other's
    policy unnoticed.
    """
    taken: set[str] = set()
    out = []
    for entry in entries:
        base = _SLICE_SAFE.sub("-", entry.name).strip("-") or "slice"
        candidate = base
        counter = 2
        while candidate in taken:
            candidate = f"{base}_{counter}"
            counter += 1
        taken.add(candidate)
        out.append(candidate)
    return out


def export_slices(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """Every slice as its own PNG, cropped from the current frame's flatten.

    Each slice resolves ``at(current_frame_uid)`` -- so a keyed slice exports
    the rectangle the panel beside it is showing right now, on a still document
    exactly as on an animated one. A per-frame *matrix* of one crop per slice
    per frame is a different export and stays out of scope here; it is
    Packwright's job.

    Not spread through the stepper the animated exports use: this reads one
    flatten, not one per frame, so there is nothing to spend across app frames.
    The geometry -- names and bounds -- is resolved here, on the frame thread,
    for ``_submit_export``'s reason about ``slices_snapshot``: the tab is
    locked (``saving``) for the rest of the call, so "now" and "inside the
    task" would answer the same question, and every other read in this
    function already happens here.
    """
    tab = tab or inker_mode.active(ctx)
    if tab is None or tab.busy:
        return
    doc = tab.doc
    if not doc.slices:
        return
    inker_mode._settle(ctx, tab)
    suggested = tab.path.stem if tab.path else "untitled"
    state = ctx.state.inker
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))
    frame_uid = tab.frame_uid
    # Both halves of Repeat Last Export, which this runner had neither of: it
    # never recorded ``dest``/``export_kind``, so ``REPEATABLE``'s "slices" row
    # could not be reached, and it ignored ``repeat``, so reaching it would
    # have opened the dialog anyway (the review's theme T5).
    recorded = tab.export_dest if repeat else None
    names = _slice_filenames(doc.slices)
    crops = [
        (name, entry.at(frame_uid).bounds)
        for name, entry in zip(names, doc.slices, strict=True)
    ]

    def run() -> dict[str, Any] | None:
        from PIL import Image

        from .inker.transform import upscale

        dest = recorded or dialogs.save_file(
            "Export slices as PNGs",
            _suggested_dialog_name(tab, suggested, ".png"),
            inker_mode.PNG_FILTER,
        )
        if dest is None:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Read here, inside the task, for ``_write``'s reason: the encoders only
        # read, and the frame thread only ever appends to a layer's pixels in
        # place, so the worst this catches is a stroke that was mid-flight.
        flat = doc.flatten()
        first = None
        for name, (x0, y0, x1, y1) in crops:
            crop = upscale(flat[y0:y1, x0:x1], scale)
            out = dest.parent / f"{name}.png"
            atomic.save_image(out, Image.fromarray(crop, "RGBA"), "PNG")
            if first is None:
                first = out
        # ``dest`` and ``export_kind`` for the same reason ``export_png``
        # records them: a repeat writes where this wrote, and the crops are
        # named from the slices rather than from the path, so the path the user
        # picked is the whole of what has to be remembered.
        return {"exported": first, "dest": dest, "export_kind": "slices"}

    inker_mode._start(ctx, tab, f"inker-export:{tab.uid}", run)


def export_sheet(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """An animated document as a packed PNG plus its JSON sidecar.

    Mirrors ``export_png`` exactly -- gated, floating buffer committed first, one
    task under the same key so the two can never run at once, and
    ``{"exported": path}`` back so the existing completion branch toasts it
    unchanged. What differs is only what gets written.

    With one addition the other exports do not need: the frames are read off the
    document on the **frame thread**, because ``_write`` gets away with encoding
    the live document (the encoders only read) and flattening a clip does not --
    it fills and evicts the document's frame cache and copies track properties
    down onto cels, the same structures the onion-skin draw is walking sixty
    times a second.

    That read used to happen inline, in this call. A sixty-frame clip is sixty
    flattens on the frame the user clicked the button, which is a freeze; and it
    cannot move to a task thread for the reason above. So it is spread instead:
    the tab is locked (``saving``, which already refuses every mutation), a
    stepper is parked on the mode state, and :func:`pump_export` flattens one
    frame per app frame until the work list is done. Then, and only then, the
    encode is submitted. This is ``viewer/sheet.StripRender``'s answer to
    exactly the same problem -- sixteen GPU readbacks in one frame versus
    sixteen frames of one -- at a different layer.
    """
    _begin_export(ctx, tab, "sheet", repeat=repeat)


def export_gif(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """An animated document as a GIF anyone can open.

    ``export_sheet``'s shape exactly, down to sharing its task key -- the two
    read the same frames off the same document and must not run at once -- and
    the same frame-spread read, through the same stepper.

    A GIF loops forever rather than honouring a tag's loop flag, and that is the
    honest reading rather than a shortcut: a tag names a *span* of the timeline
    and the export is the whole timeline, so there is no one tag whose flag this
    could be. Exporting a single tag is a different feature and would need to say
    which one.
    """
    _begin_export(ctx, tab, "gif", repeat=repeat)


def export_pngs(ctx: Any, tab: InkerDoc | None = None, *, repeat: bool = False) -> None:
    """Every frame as its own numbered PNG, through the same stepper.

    The plainest export there is, and the one an engine with its own importer
    asks for: no atlas to slice, no sidecar to parse. The spread is untouched --
    the frames are read exactly as the sheet and the GIF read them, and only
    the write differs.
    """
    _begin_export(ctx, tab, "pngs", repeat=repeat)


@dataclass
class _Leg:
    """One *output file's* worth of an export: which frames, read how.

    An ordinary export has exactly one of these and its ``label`` is empty. A
    split has one per tag or per layer, and the label is what the filename is
    built from -- see :func:`_split_stems`.
    """

    uids: list[str]
    #: What this output is called after the stem, or "" for the single-file
    #: export that is named by the dialog alone.
    label: str = ""
    #: The inclusive frame range being exported, or None for the whole
    #: timeline. Sliced **at begin**, and ``timing`` is sliced at submit with
    #: this same pair -- safe because the tab has been locked (``saving``) for
    #: the whole spread, so the frame count cannot have moved between them.
    span: tuple[int, int] | None = None
    #: What a GIF's loop block should say: True forever, False once, or a
    #: repeat count. See ``gifout.loop_option``.
    loop: bool | int = True
    #: The tracks this leg composites, or None for the whole stack. A split by
    #: layer is the only caller that sets it, and it is what sends the flatten
    #: through ``sheetout.flatten_subset`` -- which stays out of the document's
    #: frame cache, because that cache is keyed on the frame uid alone.
    track_uids: tuple[int, ...] | None = None
    #: "" | "tag" | "layer" -- which of ``sheetout.filename_for``'s two keys
    #: ``label`` fills, so ``_split_stems`` knows whether to build each stem's
    #: default template from :data:`sheetout.DEFAULT_TAG_TEMPLATE` or
    #: :data:`sheetout.DEFAULT_LAYER_TEMPLATE`. Empty for the ordinary,
    #: unsplit leg, whose label is empty too.
    split_kind: str = ""
    frames: list[Any] = field(default_factory=list)
    #: One exact index plane per read frame, or None where the frame's flatten
    #: is not a cel's own materialisation. Parallel to ``frames`` and appended
    #: in the same step, so the two cannot come apart. Only a GIF reads it --
    #: see ``sheetout.index_plane_one``.
    planes: list[Any] = field(default_factory=list)
    #: The colour table in force on each read frame -- the document's, or that
    #: frame's own override. Parallel to ``planes`` and appended beside it, for
    #: the same reason: a table read at a different moment than the slots it
    #: resolves is a frame drawn with one frame's indices and another's colours.
    palettes: list[Any] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return len(self.frames) >= len(self.uids)


@dataclass
class _Export:
    """One export's frame-by-frame read of the document.

    Lives on ``InkerState`` rather than on the tab because it is not a property
    of the document -- it is one in-flight operation, and there is one at a time
    by construction (both exports share a task key, and the tab is locked while
    it runs).

    **A split is one export, not several.** One lock, one stepper, one flatten
    per pump, one task at the end -- the legs are read back to back inside the
    machinery that already guarantees all four. N exports racing each other for
    the same task key, with the tab locked N times over, is the shape this
    deliberately does not take.

    ``uids``/``frames``/``planes``/``span``/``loop`` read the leg being flattened
    now, so every caller written before splits existed still sees the export it
    always saw.
    """

    tab: InkerDoc
    kind: str  # "sheet" | "gif" | "pngs"
    suggested: str
    legs: list[_Leg]
    #: The destination a **repeat** is writing to, or None for the ordinary
    #: export that asks. Carried on the export rather than read off the tab at
    #: the dialog, because by then the tab is locked and the answer has to be
    #: the one the click was made with (6.9).
    recorded: Path | None = None
    #: Which leg the stepper is on. Never rewound: a finished leg's frames stay
    #: on it until the submit reads them.
    at: int = 0

    @property
    def leg(self) -> _Leg:
        return self.legs[self.at]

    @property
    def uids(self) -> list[str]:
        return self.leg.uids

    @property
    def frames(self) -> list[Any]:
        return self.leg.frames

    @property
    def planes(self) -> list[Any]:
        return self.leg.planes

    @property
    def palettes(self) -> list[Any]:
        return self.leg.palettes

    @property
    def span(self) -> tuple[int, int] | None:
        return self.leg.span

    @property
    def loop(self) -> bool | int:
        return self.leg.loop

    @property
    def read(self) -> int:
        """Frames flattened so far, across every leg. One per pump, always."""
        return sum(len(leg.frames) for leg in self.legs)


def export_range(
    ctx: Any,
    tab: InkerDoc | None,
    kind: str,
    span: tuple[int, int],
    *,
    loop: bool | int = True,
) -> None:
    """Export part of the timeline -- a tag, or a marquee'd range.

    The same three exports over fewer frames, so it is the same entry point
    with a span rather than a second pipeline: the frames are read by the same
    stepper, the durations and tags by the same ``timing``, and the sheet is
    written by the same ``sheet.sidecar``. What a span changes is only which
    frames go in, that the tags come back renumbered, and that a directional
    layout is dropped -- all of which ``sheetout`` decides, not this.
    """
    _begin_export(ctx, tab, kind, span=span, loop=loop)


def export_tag(ctx: Any, tab: InkerDoc | None, kind: str, index: int) -> None:
    """One tag, as a sheet or a GIF, with its own looping honoured.

    ``tag.repeat or tag.loop`` is the whole of the difference from a range
    export: a repeat count is the more specific answer to "how many times does
    this play", and 0 means the flag decides -- exactly the rule playback
    follows, spelled once here so the file and the editor cannot disagree.
    """
    tab = tab or inker_mode.active(ctx)
    anim = None if tab is None else tab.doc.anim
    if tab is None or anim is None or not 0 <= index < len(anim.tags):
        return
    from .inker import sheetout

    tag = anim.tags[index]
    _begin_export(
        ctx,
        tab,
        kind,
        span=sheetout.tag_span(anim, tag),
        loop=tag.repeat or tag.loop,
    )


def export_per_tag(ctx: Any, tab: InkerDoc | None = None, kind: str = "sheet") -> None:
    """One file per tag, in one export.

    Each output is exactly what :func:`export_tag` writes for that tag on its
    own -- same span through ``sheetout.tag_span``, same looping, same rebased
    tags in the sidecar -- so a batch and a one-at-a-time sweep produce the same
    files. That is the whole reason the span logic is shared rather than
    repeated here.
    """
    from .inker import sheetout

    tab = tab or inker_mode.active(ctx)
    anim = None if tab is None else tab.doc.anim
    if tab is None or anim is None:
        return
    if not anim.tags:
        # Reachable even though the menu item is disabled: the verb is engine
        # API, and a refusal that says why beats one that does nothing.
        ctx.toast("This document has no tags to split by.", "warn")
        return
    _begin_export(
        ctx,
        tab,
        kind,
        legs=[
            _Leg(
                uids=[],
                label=tag.name,
                span=sheetout.tag_span(anim, tag),
                loop=tag.repeat or tag.loop,
                split_kind="tag",
            )
            for tag in anim.tags
        ],
    )


def export_per_layer(ctx: Any, tab: InkerDoc | None = None, kind: str = "sheet") -> None:
    """One file per top-level layer row, in one export.

    ``sheetout.layer_splits`` decides what a "layer" is here -- a track, or a
    whole group as the one row the panel shows -- and each leg composites only
    its own tracks. The frames are the same frames; what differs is how much of
    the stack goes into each of them.
    """
    from .inker import sheetout

    tab = tab or inker_mode.active(ctx)
    if tab is None or tab.doc.anim is None:
        return
    splits = sheetout.layer_splits(tab.doc)
    if not splits:
        ctx.toast("Every layer is hidden; there is nothing to split.", "warn")
        return
    _begin_export(
        ctx,
        tab,
        kind,
        legs=[
            _Leg(uids=[], label=name, track_uids=uids, split_kind="layer")
            for name, uids in splits
        ],
    )


#: Which function repeats which recorded export. A table rather than a chain
#: of ifs, so a seventh export kind is one row and cannot be forgotten by the
#: repeat path alone -- which is exactly how a "repeat" command goes stale.
REPEATABLE: dict[str, str] = {
    "png": "export_png",
    "sheet": "export_sheet",
    "gif": "export_gif",
    "pngs": "export_pngs",
    "slices": "export_slices",
}


def repeat_export(ctx: Any, tab: InkerDoc | None = None) -> bool:
    """Ctrl+Shift+X: run the last export again, with no dialog.

    **The hot-path escape valve**: configure once, then one key forever. It is
    the whole reason the per-document destination memory exists, and until now
    that memory only seeded the *dialog* -- so the user still had to answer it.

    Refused out loud when this document has never been exported, because "the
    last export" is not a thing yet and a silent key is one the user cannot
    tell from a broken one.
    """
    state = inker_mode.ensure(ctx)
    tab = tab or state.active
    if tab is None:
        return False
    kind = getattr(tab, "export_kind", "")
    verb = REPEATABLE.get(kind)
    if not verb or tab.export_dest is None:
        state.say(
            "Nothing to repeat yet -- export once and this runs the same one "
            "again."
        )
        return False
    globals()[verb](ctx, tab, repeat=True)
    return True


def _begin_export(
    ctx: Any,
    tab: InkerDoc | None,
    kind: str,
    *,
    span: tuple[int, int] | None = None,
    loop: bool | int = True,
    legs: list[_Leg] | None = None,
    repeat: bool = False,
) -> None:
    """Lock the tab and park the stepper. The click-frame half of an export.

    ``legs`` is the split form: one entry per output file, each carrying its own
    span and its own tracks, and ``None`` is the ordinary single-file export
    (one unlabelled leg over ``span``). Everything after this point -- the
    stepper, the lock, the submit -- is the same code for both.
    """
    from .inker import sheetout

    tab = tab or inker_mode.active(ctx)
    state = ctx.state.inker
    if tab is None or state is None:
        return
    # **Say why.** All three of these refusals were silent, and each is a
    # normal thing to run into: pressing Export while the clip is playing, or
    # while a save is landing, or while another tab's export dialog is up. A
    # menu item that does nothing and says nothing reads as a broken build.
    if tab.doc.anim is None:
        ctx.toast("This drawing has no timeline to export frames from.", "warn")
        return
    if tab.playing:
        ctx.toast("Stop playback first; an export reads the frames as they are.", "warn")
        return
    if tab.busy:
        ctx.toast(inker_mode._no_document_reason(tab), "warn")
        return
    if state.export is not None:
        ctx.toast("An export is already being set up; finish that one first.", "warn")
        return
    # Seeded once per tab, not on every click: a fresh tab's ``export_options``
    # is empty and this is a no-op, but a tab exported once already restores
    # what *it* last used -- over whatever the previous tab exporting left on
    # the shared controls -- the first time it exports again. Guarded on
    # ``export_seed_uid`` so a user who tweaks a control and clicks Export a
    # second time for the *same* tab keeps that edit rather than having it
    # silently put back; only a switch away and back re-suggests.
    if state.export_seed_uid != tab.uid:
        state.apply_export_options(tab.export_options)
        state.export_seed_uid = tab.uid
    suggested = tab.path.stem if tab.path else "untitled"
    if legs is None:
        legs = [_Leg(uids=[], span=span, loop=loop)]
    leg_kind = legs[0].split_kind if legs else ""
    template = str(getattr(state, "export_template", "") or "").strip() or None
    try:
        # The work lists are filled *here* rather than by the callers, so a span
        # that holds no frames refuses before the lock -- and so "frame 3 of 60"
        # is read once, for every leg, on the frame the button was pressed.
        for leg in legs:
            leg.uids = sheetout.frame_uids(tab.doc, leg.span)
        # Checked here rather than in the runner, where the stem the user picked
        # is finally known: a collision is a property of the labels alone, so it
        # can be refused before the tab is locked and before somebody names a
        # batch that was never going to be written.
        _split_stems(
            suggested, [leg.label for leg in legs], kind=leg_kind, template=template
        )
    except ValueError as exc:
        ctx.toast(f"Cannot export: {exc}.", "warn")
        return
    # **After the refusals, not before them.** Settling commits the floating
    # buffer and cancels the filter and conversion previews, all of which change
    # the document -- so running it above the two checks meant an export that
    # went on to say "cannot export" had already folded a paste into the layers
    # on its way to refusing. Nothing either check reads is affected by it: they
    # ask which frames a span holds and whether the labels collide.
    inker_mode._settle(ctx, tab)
    # Locked before the first flatten, not at submit time: the whole point of
    # spreading the read is that frames go by between here and the encode, and
    # an edit landing in one of them would put half of two documents in the
    # sheet. ``saving`` is the flag ``busy`` already refuses mutation on.
    tab.saving = True
    state.export = _Export(
        tab=tab,
        kind=kind,
        suggested=suggested,
        legs=legs,
        recorded=tab.export_dest if repeat else None,
    )


def _suggested_dialog_name(tab: InkerDoc, suggested: str, ext: str) -> str:
    """The ``default_name`` an export's save dialog opens with.

    A bare filename ordinarily, exactly what every export always passed. Once
    this tab has exported before, ``export_dest``'s *folder* is folded in
    too -- a native picker reads a path's directory half as where to open --
    so picking this tab back up suggests the folder it was last written into
    rather than wherever the picker happened to be left by the tab exported in
    between.
    """
    dest = tab.export_dest
    if dest is None:
        return f"{suggested}{ext}"
    return str(dest.parent / f"{suggested}{ext}")


def _split_stems(
    stem: str, labels: Sequence[str], *, kind: str = "", template: str | None = None
) -> list[str]:
    """One filename stem per output, through :func:`sheetout.filename_for`.

    **The one place a split's filenames are decided**, which is what makes
    Task 5's filename templates a single edit rather than a sweep through
    three runners. An empty label is the unsplit export and keeps the stem
    the dialog was given, byte for byte -- no template involved, because
    there is nothing here for one to distinguish -- but that shortcut only
    applies when ``kind`` is also empty. A split leg's label can *itself* be
    empty (a loaded ``.ase``/ORA may carry a tag or a track with no name),
    and that empty label is a real, if badly named, split output -- not the
    unsplit sentinel. Collapsing it onto the bare stem would write a file
    indistinguishable from a whole-document export, so a falsy label under a
    non-empty ``kind`` falls back to the literal word ``"tag"``/``"layer"``
    instead, and still goes through the same template and collision check as
    every other label.

    ``kind`` is ``"tag"`` or ``"layer"``, and it picks both which of
    ``filename_for``'s two keys a non-empty label fills and which default
    template applies when ``template`` is None -- :data:`sheetout.
    DEFAULT_TAG_TEMPLATE` or :data:`sheetout.DEFAULT_LAYER_TEMPLATE`, the
    exact ``f"{stem}_{safe}"`` this always wrote before templates existed.

    A collision is **refused**, where ``_slice_filenames`` bumps: a slice is a
    rectangle a person picks off a folder listing, and "Hitbox_2.png" is a name
    they can live with -- but a tag and a layer are addressed *by name* by
    whatever consumes the sheet, and a second "walk" quietly becoming
    "walk_2.png" is a file claiming to be a clip called walk_2. Refusing is the
    only answer that cannot silently be believed. A template that renders two
    labels the same collides here for the identical reason.
    """
    from .inker import sheetout

    default = (
        sheetout.DEFAULT_LAYER_TEMPLATE
        if kind == "layer"
        else sheetout.DEFAULT_TAG_TEMPLATE
    )
    tmpl = template or default
    out: list[str] = []
    for label in labels:
        if not label:
            if not kind:
                out.append(stem)
                continue
            label = "layer" if kind == "layer" else "tag"
        out.append(
            sheetout.filename_for(
                tmpl,
                title=stem,
                tag=None if kind == "layer" else label,
                layer=label if kind == "layer" else None,
            )
        )
    sheetout.require_distinct_names(out)
    return out


def _frame_palette(doc: Any, uid: str) -> list | None:
    """One frame's own colour table, or None to mean "the document's".

    None rather than the effective table, so a document that has never used
    per-frame palettes carries no per-frame data through the export at all:
    ``has_frame_palettes`` is this feature's one-boolean gate everywhere else
    (``group_fold``'s rule) and this keeps the export's cost the same shape.

    Resolved by uid here because that is what the stepper holds -- the same
    ``leg.uids`` entry ``index_plane_one`` was just handed, which is what makes
    the table and the slots a matched pair.
    """
    anim = getattr(doc, "anim", None)
    if anim is None or not doc.has_frame_palettes:
        return None
    frame = next((entry for entry in anim.frames if str(entry.uid) == str(uid)), None)
    return None if frame is None else anim.frame_palette(frame.uid)


def pump_export(ctx: Any) -> None:
    """One frame of an in-flight export's read. Called once a frame by the app.

    Beside ``journal.pump`` and in every mode for the same reason: a user who
    started an export and switched to the library must still get their file.
    """
    state = getattr(ctx.state, "inker", None)
    export = None if state is None else state.export
    if export is None:
        return
    from .inker import sheetout

    tab = export.tab
    if tab not in state.docs or tab.doc.anim is None:
        # The tab was closed under the export. Nothing has been written and the
        # lock goes with the tab, so there is nothing to undo.
        state.export = None
        return
    leg = export.leg
    try:
        uid = leg.uids[len(leg.frames)]
        if leg.track_uids is None:
            plane = sheetout.flatten_one(tab.doc, uid)
            # Read here, beside the flatten it describes and on the same frame:
            # taken later it would describe a document the user has since
            # edited, and the two have to be a matched pair or the GIF is drawn
            # with one frame's slots and another frame's colours.
            leg.planes.append(sheetout.index_plane_one(tab.doc, uid))
            leg.palettes.append(_frame_palette(tab.doc, uid))
        else:
            plane = sheetout.flatten_subset(tab.doc, uid, leg.track_uids)
            # None rather than a subset index plane: ``index_plane_one`` decides
            # by comparing a candidate cel against the *whole* frame's flatten,
            # which a subset is not. A GIF of one layer therefore quantises from
            # the colours, as every GIF did before index planes existed --
            # correct, just not slot-stable.
            leg.planes.append(None)
            leg.palettes.append(None)
        leg.frames.append(plane)
    except Exception as exc:  # noqa: BLE001 - the lock must clear whatever failed
        # Any failure, not a list of three: a ``MemoryError`` on a big flatten
        # left ``tab.saving`` set and ``state.export`` armed forever, and every
        # later export on every tab silently returned at the top of this pump.
        state.export = None
        tab.saving = False
        ctx.toast(f"Export failed: a frame could not be flattened ({exc}).", "warn")
        return
    if not leg.done:
        return
    if export.at + 1 < len(export.legs):
        # Exactly one flatten has happened this pump, so the next leg starts on
        # the next one: a batch that ran the legs back to back here would be the
        # freeze the stepper exists to prevent, N times over.
        export.at += 1
        return
    state.export = None
    _submit_export(ctx, export)


@dataclass
class _Payload:
    """One output file, as the runners need it: pixels plus what describes them.

    Built on the frame thread by :func:`_submit_export` and read on the task
    thread, which is why it holds values rather than a document to ask.
    """

    label: str
    frames: list[Any]
    planes: list[Any]
    #: One colour table per frame, or None to mean "the document's". Only the
    #: GIF runner reads it; a sheet is RGBA and has no table to write.
    palettes: list[Any]
    durations: list[int]
    tags: list[Any]
    layout: Any
    slices: list[Any]
    loop: bool | int


def pump_undo_trim(ctx: Any) -> None:
    """Say so, once per event, when the history dropped steps to stay in memory.

    Beside ``pump_export`` and ``journal.pump``, and in every mode for their
    reason: the press that trimmed the stack is usually the last thing the user
    does before switching away, and the missing undo is discovered somewhere
    else entirely.

    The engine counts rather than calls back -- ``studio.undo`` imports nothing
    and two headless packages depend on that -- so the comparison lives here,
    against a per-tab mark.
    """
    state = getattr(ctx.state, "inker", None)
    if state is None:
        return
    for tab in state.docs:
        history = getattr(tab.doc, "history", None)
        trimmed = getattr(history, "trimmed", 0)
        if trimmed == tab.trim_seen:
            continue
        # The mark *is* the coalesce: this runs every frame in every mode, and
        # a tab left unmarked would raise the same toast sixty times a second
        # for as long as the count stayed different.
        tab.trim_seen = trimmed
        if trimmed:
            ctx.toast(
                f"Undo history trimmed on {tab.title}: that step was too large "
                "to keep alongside the others.",
                "warn",
            )


def _submit_export(ctx: Any, export: _Export) -> None:
    """The work list is read; hand it to a task. Frame thread."""
    from .inker import gifout, sheetout
    from .inker.transform import upscale

    tab, suggested = export.tab, export.suggested
    doc = tab.doc
    state = ctx.state.inker
    # Read here, on the frame thread, with the frames: an app-level setting the
    # user could change while the encode is in flight would otherwise decide
    # the file's size halfway through writing it.
    scale = max(1, int(getattr(state, "export_scale", 1) or 1))
    # Same reason as ``scale`` beside it: a setting the user could change
    # mid-encode must not decide, halfway through, how this file is packed.
    arrange = getattr(state, "export_arrange", None)
    wrap = max(1, int(getattr(state, "export_wrap", 1) or 1))
    merge = bool(getattr(state, "export_merge", False))
    skip_empty = bool(getattr(state, "export_skip_empty", False))
    trim = bool(getattr(state, "export_trim", False))
    padding = max(0, int(getattr(state, "export_padding", 0) or 0))
    extrude = max(0, int(getattr(state, "export_extrude", 0) or 0))
    template = str(getattr(state, "export_template", "") or "").strip() or None
    # Which of ``sheetout``'s two split templates applies -- see ``_Leg.
    # split_kind``. Every leg of one export shares it, since a batch is either
    # a tag split, a layer split or the ordinary unsplit export, never a mix.
    split_kind = export.legs[0].split_kind if export.legs else ""
    # The whole option set this export is about to run with, captured once
    # here rather than re-read per runner: what a completed export records
    # onto its tab (``on_task_done``) has to be the settings that actually
    # produced the file, not whatever the controls hold by the time the task
    # finishes.
    export_options = state.export_options_snapshot()
    if export.kind == "sheet" and padding < extrude * 2:
        # Refused here, before the file dialog, for the same reason the
        # arrange/layout and merge/layout conflicts below are: ``sheetout.build``
        # would raise this itself, but by then the user has already picked a
        # filename and the failure arrives as an opaque task error.
        tab.saving = False
        ctx.toast(
            f"Padding must be at least twice Extrude to give every sprite "
            f"room ({extrude} x 2 = {extrude * 2}, padding is {padding}).",
            "warn",
        )
        return
    # One payload per output file. A single-file export has exactly one and
    # every line below reads the same as it did before splits existed; a split
    # has one per tag or per layer, each carrying its own timing and its own
    # slices, because a sidecar has to be self-consistent with the file it is
    # beside rather than with the document the batch came from.
    loads: list[_Payload] = []
    for leg in export.legs:
        durations, tags, layout = sheetout.timing(doc, leg.span)
        # Read here, with the timing, and for its reason: it walks the document,
        # so it belongs on the frame thread beside the flatten rather than
        # inside the task. Cheap -- a handful of rectangles -- so there is
        # nothing to spread.
        #
        # Sliced by the *same* span as the frames and the timing, because
        # ``slices_block`` keys by cell index: a span export's third cell is the
        # third frame of the span, and a whole-timeline snapshot here would hang
        # frame 0's rectangles on it.
        # **The JSON meta switches** (6.9), read here with everything else the
        # sidecar is made of: a setting the user could change mid-encode must
        # not decide, halfway through, what the file says about itself.
        slices = (
            sheetout.slices_snapshot(doc, leg.span)
            if getattr(state, "export_meta_slices", True)
            else []
        )
        frames = leg.frames
        if export.kind == "sheet" and layout is not None:
            if len(frames) != layout.frame_count:
                # Refused on the frame thread, before the file dialog: the
                # engine raises the same ValueError as a backstop, but by then
                # the user has picked a filename and the failure arrives as a
                # task error with no obvious cause. A frame added to (or removed
                # from) a sprite sheet is an ordinary edit, so the fix is to say
                # which count is wrong.
                tab.saving = False
                ctx.toast(
                    f"This is a {layout.kind} sheet of {layout.frame_count} "
                    f"frames and the document has {len(frames)}.",
                    "warn",
                )
                return
            if arrange is not None:
                # The same early-refusal shape as the count mismatch above, for
                # the same reason: ``plan_frames`` would raise this itself, but
                # by then the user has already picked a filename and the failure
                # arrives as an opaque task error. A document with its own
                # directional grid keeps it -- Grid is the only Arrange choice
                # such a document has.
                tab.saving = False
                ctx.toast(
                    f"This is a {layout.kind} sheet, which keeps its own fixed "
                    "grid; set Arrange back to Grid to export it.",
                    "warn",
                )
                return
            if merge or skip_empty:
                # Same shape and same reason as the arrange/layout refusal
                # above: a directional grid's cells are poses by yaws, so there
                # is nothing for Merge or Skip empty to act on, and letting the
                # request through would have ``sheetout.compose`` raise it as an
                # opaque task error instead.
                tab.saving = False
                ctx.toast(
                    f"This is a {layout.kind} sheet, which keeps its own fixed "
                    "grid; turn Merge and Skip empty off to export it.",
                    "warn",
                )
                return
        loads.append(
            _Payload(
                label=leg.label,
                frames=frames,
                planes=leg.planes,
                palettes=leg.palettes,
                durations=durations,
                tags=tags if getattr(state, "export_meta_tags", True) else [],
                layout=layout,
                slices=slices,
                loop=leg.loop,
            )
        )

    def run_sheet() -> dict[str, Any] | None:
        """Every leg composed, then every file written. Task thread.

        **Two loops rather than one, and that is the whole point.** A split is
        one export producing N files, and ``compose`` can refuse a leg the
        others are fine with -- ``skip_empty`` over a tag with nothing drawn in
        it is the reachable case, and the atlas ceiling and the padding rule are
        two more. Written inside a single loop, a refusal on leg k left legs
        0..k-1 on disk under names the user has every reason to believe, with
        the rest missing and only a toast to say so.

        The seam is the *runner's* own start rather than a pre-dialog check on
        the frame thread, deliberately. The all-empty case alone could be
        checked from the flattens ``_submit_export`` already holds -- but only
        that one: the ceiling and the padding refusals need the plan, which
        needs the compose, so a frame-thread door would leave the same
        half-written batch reachable two other ways while carrying a *second*
        copy of the emptiness rule (a second opinion about what "empty" means is
        exactly the drift ``sheetout`` centralises to avoid). Composing first
        catches every refusal ``compose`` has, present and future, with no rule
        duplicated. The precedent is ``packwright_io._write(files: dict[Path,
        bytes])`` -- encode all, then write all -- for the same reason.

        The cost is honest and bounded: N atlases live at once instead of one.
        A split is per tag or per top-level layer, so N is single digits on any
        real document, and each atlas is the sheet that leg was going to write
        anyway.
        """
        import json

        dest = export.recorded or dialogs.save_file(
            "Export sprite sheet",
            _suggested_dialog_name(tab, suggested, ".png"),
            inker_mode.PNG_FILTER,
        )
        if dest is None:
            return None
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        stems = _split_stems(
            dest.stem, [load.label for load in loads], kind=split_kind, template=template
        )
        composed: list[tuple[str, Any, dict[str, Any]]] = []
        try:
            for stem, load in zip(stems, loads, strict=True):
                # Upscaled *before* ``compose``, so the plan is built on the
                # scaled frame size and the cells, the trims and the sidecar all
                # describe the atlas that is actually written. Scaling the
                # finished atlas instead would leave every rectangle in the
                # sidecar naming the wrong pixels. ``sheet.py`` stays the sole
                # writer of the format; none of this is new code in it.
                try:
                    image, plan, extra = sheetout.compose(
                        [upscale(plane, scale) for plane in load.frames],
                        load.durations,
                        load.tags,
                        load.layout,
                        # The slice geometry through the same magnification, or
                        # the sidecar describes a canvas that is not the atlas
                        # beside it.
                        sheetout.scale_slices(load.slices, scale),
                        name=suggested,
                        arrange=arrange,
                        wrap=wrap if arrange in ("rows", "columns") else None,
                        merge=merge,
                        skip_empty=skip_empty,
                        trim=trim,
                        padding=padding,
                        extrude=extrude,
                    )
                except ValueError as exc:
                    if not split_kind:
                        raise
                    # Which leg, by name. One file's refusal arriving as the
                    # batch's bare reason ("every frame is empty") says nothing
                    # about *which* tag or layer the user has to go and look at,
                    # and a split is exactly the export where that is the whole
                    # question.
                    raise ValueError(
                        f"{split_kind} {load.label or split_kind!r}: {exc}"
                    ) from exc
                composed.append(
                    (
                        stem,
                        image,
                        sheetlib.sidecar(
                            plan,
                            sheet_id=stem,
                            source_job=tab.job_id,
                            image=f"{stem}.png",
                            created=time.time(),
                            name=suggested,
                            trims=extra["trims"],
                            animation=extra["animation"],
                            pivots=extra["pivots"],
                            slices=extra["slices"],
                            slices_conflict=extra["slices_conflict"],
                        ),
                    )
                )
        except BaseException:
            for _stem, image, _meta in composed:
                image.close()
            raise

        dest.parent.mkdir(parents=True, exist_ok=True)
        first: Path | None = None
        for stem, image, meta in composed:
            out = dest.with_name(f"{stem}.png")
            try:
                atomic.save_image(out, image, "PNG")
            finally:
                image.close()
            # ``with_name`` rather than ``with_suffix``: it spells the sidecar's
            # filename directly from ``stem``, the same way ``out`` itself was
            # just built two lines up, rather than leaning on ``with_suffix`` to
            # rederive that same name by parsing it back out of ``out``'s own
            # name.
            atomic.write_text(
                out.with_name(f"{stem}.json"), json.dumps(meta, indent=2)
            )
            if first is None:
                first = out
        return {
            "exported": first,
            "dest": dest,
            "options": dict(export_options),
            "export_kind": export.kind,
        }

    # The document's own table when it has one, so an indexed clip exports the
    # colours that were authored rather than a per-frame quantise of them. Read
    # on the frame thread here, with the frames, not inside the task.
    #
    # The *fallback*, since 2026-09-03: a frame carrying its own table
    # (``animation.frame_palettes``) is written with that one instead. This was
    # read once, outside the frame loop, so every such frame had its slots
    # resolved through the document's table and came out the wrong colours --
    # the ORA and Aseprite writers both honoured the override and GIF was the
    # only exporter that did not.
    palette = list(doc.palette) if doc.palette else None

    def run_gif() -> dict[str, Any] | None:
        dest = export.recorded or dialogs.save_file(
            "Export animated GIF",
            _suggested_dialog_name(tab, suggested, ".gif"),
            inker_mode.GIF_FILTER,
        )
        if dest is None:
            return None
        if dest.suffix.lower() != ".gif":
            dest = dest.with_suffix(".gif")
        stems = _split_stems(
            dest.stem, [load.label for load in loads], kind=split_kind, template=template
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        first: Path | None = None
        for stem, load in zip(stems, loads, strict=True):
            out = dest.with_name(f"{stem}.gif")
            # Through ``atomic.staged`` rather than one of its ``write_*``
            # helpers, because the writer here takes a path rather than giving
            # back bytes -- and ``write_gif`` names its own format, so a
            # ``.tmp`` destination costs it nothing.
            with atomic.staged(out) as tmp:
                # Upscaled before the quantiser, not after: a GIF holds palette
                # indices, so there is no "after" -- magnifying the indexed
                # image would be magnifying a palette lookup rather than a
                # picture.
                gifout.write_gif(
                    tmp,
                    [upscale(plane, scale) for plane in load.frames],
                    load.durations,
                    loop=load.loop,
                    palette=palette,
                    palettes=load.palettes,
                    # Magnified by the same whole number as the pixels, which is
                    # exact on an index plane in a way it is on nothing else:
                    # ``upscale`` repeats each element, so a magnified slot is
                    # still that slot.
                    indices=[
                        None if slots is None else upscale(slots, scale)
                        for slots in load.planes
                    ],
                )
            if first is None:
                first = out
        return {
            "exported": first,
            "dest": dest,
            "options": dict(export_options),
            "export_kind": export.kind,
        }

    def run_pngs() -> dict[str, Any] | None:
        """One PNG per frame, numbered. The plainest thing an engine can eat.

        Numbered from the chosen filename's stem rather than asking for a
        directory: every tool that consumes a sequence wants ``name_0000.png``
        beside its siblings, and a save dialog is the one place a user is
        already picking both the folder and the name.

        The per-frame name goes through :func:`sheetout.filename_for` a
        second time, on top of the stem ``_split_stems`` already built: a
        split's own template (``{title}_{tag}``/``{title}_{layer}``, or a
        custom one) decides how the *outputs* of a batch differ from each
        other, and this decides how the *frames inside one output* differ --
        two questions, so a split PNG sequence always numbers its frames with
        :data:`sheetout.DEFAULT_FRAME_TEMPLATE` rather than reading the same
        custom template twice for two different things.
        """
        from PIL import Image

        # ``export.recorded`` like the sheet and the GIF beside it: this was the
        # one runner of the three that read the dialog unconditionally, so
        # Repeat Last Export -- documented as "asks nothing" -- reopened the
        # picker for a PNG sequence (the review's theme T5).
        dest = export.recorded or dialogs.save_file(
            "Export PNG sequence",
            _suggested_dialog_name(tab, suggested, ".png"),
            inker_mode.PNG_FILTER,
        )
        if dest is None:
            return None
        stems = _split_stems(
            dest.stem, [load.label for load in loads], kind=split_kind, template=template
        )
        frame_template = (
            sheetout.DEFAULT_FRAME_TEMPLATE
            if split_kind
            else (template or sheetout.DEFAULT_FRAME_TEMPLATE)
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        first = dest
        for stem, load in zip(stems, loads, strict=True):
            names = [
                sheetout.filename_for(frame_template, title=stem, frame=index)
                for index in range(len(load.frames))
            ]
            sheetout.require_distinct_names(names)
            for name, plane in zip(names, load.frames, strict=True):
                # Checked where the path is *built*, not only where the name was
                # composed: ``filename_for``'s ``containment_check`` decides what
                # a name may be and this decides where the join actually lands,
                # which are two different mistakes -- and this is the one runner
                # that joins onto ``dest.parent`` rather than going through
                # ``with_name``, which raises on a separator by itself.
                out = inker_mode._under(dest.parent, f"{name}.png")
                atomic.save_image(
                    out, Image.fromarray(upscale(plane, scale), "RGBA"), "PNG"
                )
                if first is dest:
                    first = out
        return {
            "exported": first,
            "dest": dest,
            "options": dict(export_options),
            "export_kind": export.kind,
        }

    runners = {"sheet": run_sheet, "gif": run_gif, "pngs": run_pngs}
    # ``start_save`` rather than a bare submit, so a refused key clears the lock
    # this function did not set -- the tab has been locked since the click.
    inker_mode._start(ctx, tab, f"inker-export:{tab.uid}", runners.get(export.kind, run_gif))


# --- the five doors, spelled once ------------------------------------------
#
# Until 2026-09-05 these five labels, tooltips and refusal sentences lived in
# ``panes/inker_timeline.py`` and nowhere else: a second toolbar row under the
# transport. At 1280x800 that row overflowed -- three of the five collapsed
# into a ``...`` menu, "Skip empty" was clipped mid-word and the onion row
# below it was cut off by the pane's bottom edge -- so the exports the row
# existed for were the part of it you could not see. They moved to Inker's
# bridge (``panes/inker_generate.py``) and to the File menu, and the labels
# came here so the two presentations cannot drift: a door is *one* record and
# both readers ask :func:`doors` for it.


@dataclass(frozen=True)
class Door:
    """One export the Inker offers, wherever it is presented."""

    key: str
    label: str
    icon: str
    tooltip: str
    #: Why this door is refused when its own precondition is unmet. Empty for
    #: a door that only ever waits on the document.
    refusal: str = ""


#: No drawing at all. Every door carries it, because a bridge that draws the
#: five whether or not a document is open has to say why they are grey.
NO_DOCUMENT_WHY = "No drawing is open."

#: Mid-write. The same sentence the rest of the app's document buttons give.
BUSY_WHY = "This document is being written; the buttons come back when it lands."

DOORS: tuple[Door, ...] = (
    Door(
        "sheet",
        "Export sheet...",
        icons.GRID,
        "Writes a packed PNG of every frame plus a JSON sidecar "
        "naming the cells, their durations and any tags.",
    ),
    Door(
        "gif",
        "Export GIF...",
        icons.FILM,
        "Writes the whole timeline as an animated GIF, looping. A "
        "GIF holds no partial transparency and times frames in hundredths "
        "of a second, so soft edges become hard ones and a duration is "
        "rounded to the nearest 10 ms.",
    ),
    Door(
        "pngs",
        "Export PNGs...",
        icons.IMAGE,
        "Writes one numbered PNG per frame beside the name you "
        "pick -- name_0000.png, name_0001.png and so on.",
    ),
    Door(
        "per-tag",
        "Export sheet per tag...",
        icons.FLAG,
        "Writes one sheet per tag, each exactly what exporting "
        "that tag on its own would write -- name_walk.png, "
        "name_idle.png, each with its own sidecar.",
        refusal="This document has no tags to split by.",
    ),
    Door(
        "per-layer",
        "Export sheet per layer...",
        icons.LAYERS,
        "Writes one sheet per layer, or per group as the panel "
        "shows it -- each holding only that layer's own pixels. Hidden "
        "layers are left out.",
        refusal=(
            "There is only one visible layer, so a split would write "
            "the sheet Export sheet already writes."
        ),
    ),
)


def doors() -> tuple[Door, ...]:
    return DOORS


def door_state(door: Door, tab: Any) -> tuple[bool, str]:
    """``(enabled, reason)`` for one door against the open document.

    Both readers -- the bridge's buttons and the File menu's rows -- call this,
    so a door is never live in one place and grey in the other, and a grey one
    always carries a sentence (the harness audits for exactly that).
    """
    if tab is None:
        return (False, NO_DOCUMENT_WHY)
    if getattr(tab, "busy", False):
        return (False, BUSY_WHY)
    if door.key == "per-tag":
        tags = getattr(getattr(tab.doc, "anim", None), "tags", None)
        if not tags:
            return (False, door.refusal)
    elif door.key == "per-layer":
        from .inker import sheetout

        if len(sheetout.layer_splits(tab.doc)) <= 1:
            return (False, door.refusal)
    return (True, "")


def open_door(ctx: Any, tab: Any, key: str) -> None:
    """Run one door by key. The single dispatch both readers share."""
    if key == "sheet":
        export_sheet(ctx, tab)
    elif key == "gif":
        export_gif(ctx, tab)
    elif key == "pngs":
        export_pngs(ctx, tab)
    elif key == "per-tag":
        export_per_tag(ctx, tab, "sheet")
    elif key == "per-layer":
        export_per_layer(ctx, tab, "sheet")


# --- the sheet knobs, which moved with the doors ----------------------------
#
# These used to be the *trailing* of the export toolbar row in the timeline.
# They came with the exports because every one of them is read by
# ``_submit_export`` and by nothing else: they describe the file that is
# written, not the clip that is played. (The Onion and Thumbs switches stayed
# behind for the mirror-image reason -- they change what the strip draws and
# never reach a file.)

#: The whole-number magnifications the export combo offers. Whole numbers only,
#: because the point of the setting is that nothing is resampled -- x1.5 would
#: have to invent a rule for which source pixel a destination one comes from,
#: which is exactly what ``transform.upscale`` exists not to do.
EXPORT_SCALES = (("1", "1x"), ("2", "2x"), ("3", "3x"), ("4", "4x"), ("8", "8x"))

#: The ``##inkertemplate`` field's placeholder -- shown, never typed, so an
#: empty box still says what "empty" means without a caption stealing width
#: from the row. ``sheetout.filename_for``'s own default for a plain PNG
#: sequence; the tooltip covers the split defaults, which this field cannot
#: show both of at once.
EXPORT_TEMPLATE_HINT = "{title}_{frame}"

#: How a sheet export packs its cells. "grid" is the combo's spelling of
#: ``InkerState.export_arrange is None`` -- the row-wrap this always did --
#: since ``widgets.combo`` needs a real key for every option and ``None``
#: cannot be one.
ARRANGE_OPTIONS = (
    ("grid", "Grid"),
    ("horizontal", "Horizontal strip"),
    ("vertical", "Vertical strip"),
    ("rows", "Rows..."),
    ("columns", "Columns..."),
)
