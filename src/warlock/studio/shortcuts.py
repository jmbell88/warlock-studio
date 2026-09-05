"""The Ctrl+/ sheet's contents, as data, and the filter over it.

**Module-level and imgui-free**, which is the whole reason it is a file rather
than a pair of methods: ``tests/manual/test_shortcuts.py`` compares it against
chapter 16, and that gate did not exist while the popup drifted into saying
"F1 -- Switch to the Manual" and "thirteen modes" against a tree with ten and no
such mode.

Lifted out of ``studio/main`` on 2026-09-04 (T7 of the 2026-09-02 review). Pure
code motion: the two functions are unchanged, and ``main`` re-exports both,
where every caller and every test already names them.
"""

from __future__ import annotations


def shortcut_sections() -> list[tuple[str, list[tuple[str, str]]]]:
    """Every binding the Ctrl+/ sheet lists, as data.

    Module-level and imgui-free so ``tests/manual/test_shortcuts.py`` can
    compare it against chapter 16 -- which is the gate that did not exist
    while the popup drifted into saying "F1 -- Switch to the Manual" and
    "thirteen modes" against a tree with ten and no such mode.
    Every group's rows are gathered before anything is drawn because the
    query decides which *groups* survive: a heading over nothing is a section
    that looks broken.
    """
    sections: list[tuple[str, list[tuple[str, str]]]] = []

    def table(title: str, rows: list[tuple[str, str]]) -> None:
        sections.append((title, rows))

    table(
        "Everywhere",
        [
            # No per-mode digit: ten modes against ten digits reads as a
            # promise of a stable mapping that the next mode breaks, and
            # the palette is the keyboard route to all of them.
            ("Ctrl+K", "Command palette -- switch mode, or open an asset"),
            ("Ctrl+/", "This list"),
            ("F1", "Open the manual over whatever is on screen"),
            (
                "Esc",
                "Close the topmost thing: the manual, then a running tour, "
                "then a mode you passed through",
            ),
            ("F10", "Toggle the frame-rate readout"),
        ],
    )
    # One heading, because there is one mode (the UI redesign, wave 5). The
    # rows that used to be split "2D" from "3D" are the same keys either
    # way -- what changed is which stage of Create you are standing on,
    # and the stage rail is a click rather than a shortcut, so there is
    # nothing here to key.
    table(
        "Create",
        [
            ("Ctrl+Enter", "Run the stage: Generate, or Make 3D"),
            ("Tab / Shift+Tab", "Move between the form's controls"),
            ("Enter", "Press the stage's button when it is the one focused"),
            ("Up / Down", "Previous / next asset in the library"),
            ("Right-click a card", "Its actions menu"),
            ("F", "Frame the model"),
            ("W", "Toggle wireframe"),
            ("S", "Toggle turntable"),
            ("Esc", "Exit comparison / pose edit"),
        ],
    )
    table(
        "Review",
        [
            ("1 - 5", "Grade the mesh +1 to +5 (+3 is usable)"),
            ("R then 1 - 5", "Grade it -1 to -5"),
            ("0", "Grade it 0 - no opinion either way"),
            ("Ctrl + 1-5", "Toggle a good tag for the next grade"),
            ("Shift + 1-5", "Toggle a bad tag for the next grade"),
            ("S", "Skip to the next unverdicted unit"),
            ("Left / Right", "Previous / next unit"),
            ("Esc", "Clear the pending sign and tags"),
        ],
    )
    table(
        "Review - a judging pass",
        [
            ("A", "Accept - files +3"),
            ("R", "Reject - files -3, rather than arming a negative"),
            ("S", "Skip, staying in the pass"),
            ("Esc", "End the pass and show its report"),
        ],
    )
    from . import inker_state
    from .clay_mode import TOOL_KEYS as CLAY_KEYS
    from .inker_mode import ALT_TOOL_CHORDS

    table(
        "Clay",
        [
            (
                " / ".join(k.upper() for k in CLAY_KEYS),
                # Capitalised here rather than in ``TOOL_KEYS``: those
                # values are the tool *ids* ``state.tool`` is compared
                # against and the saved documents carry, and this is the
                # one place they are read as English. Joined lowercase
                # they were the only row in the popup that did not start
                # with a capital, one line above "Vertex / edge / face".
                " / ".join(CLAY_KEYS.values()).capitalize(),
            ),
            ("1 / 2 / 3 / 4", "Vertex / edge / face / object mode"),
            ("E", "Extrude (with faces selected)"),
            # The keyboard's half of a drag. G and S rather than G, R and S:
            # R is the Scale tool's letter and E is Rotate's, both taken long
            # before this, so rotate is reached mid-drag instead.
            ("G / S", "Move / scale the selection -- no handle to grab"),
            ("L", "Select everything joined to the selection"),
            ("Ctrl+= / Ctrl+-", "Grow / shrink the selection by one ring"),
            ("Alt+click", "Select the edge loop; Ctrl+Alt+click takes the ring"),
            ("G / R / S", "Switch the transform while a drag is under way"),
            ("F", "Frame the selection"),
            ("Delete", "Delete -- faces in an element mode, objects otherwise"),
            ("Ctrl+J", "Duplicate (object mode)"),
            ("Ctrl+M / Ctrl+Shift+M", "Merge / union the selection (object mode)"),
            ("Ctrl+D", "Deselect"),
            ("Ctrl+A", "Select all, in the current mode"),
            ("Ctrl+Shift+I", "Invert the selection"),
            ("Right-click", "Context menu"),
            ("Alt+drag", "Orbit, in any mode"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            # "Ctrl+N / O / W", matching Inker's row below rather than
            # stopping at O: Clay closes a document with Ctrl+W like every
            # other document mode, and the popup simply never said so
            # (UX-13). The axis views were missing from both this popup and
            # the manual's "full list", which made six of them
            # undiscoverable.
            ("Ctrl+N / O / W", "New / open / close"),
            ("Ctrl+E", "Export to the library"),
            ("Ctrl+Tab / Ctrl+Shift+Tab", "Next / previous document"),
            ("Ctrl+1 / 3 / 7", "Look along front / right / top"),
            ("Ctrl+Shift+1 / 3 / 7", "The opposite view: back / left / bottom"),
            ("Ctrl+5", "Orthographic / perspective"),
        ],
    )
    # **The letters, named, six to a row.** This was one squashed row
    # reading "A, B, C, D, E, ..." with the note "hover a tool for its
    # letter" -- which is a shortcut sheet declining to be one, and the
    # only mode's table that did. Six per row keeps the two columns
    # readable, and the order is the toolbox's, so the pairs that sit
    # together there (brush/spray, line/curve, the two lassos) sit
    # together here.
    tool_rows = []
    band = list(inker_state.TOOLS)
    for start in range(0, len(band), 6):
        chunk = band[start : start + 6]
        tool_rows.append(
            (
                " / ".join(letter for _key, _label, letter in chunk),
                " / ".join(label for _key, label, _letter in chunk),
            )
        )
    # Aseprite files these two-to-a-slot and cycles with Shift; here they
    # are second bindings beside the plain letters, so they are listed
    # rather than left to the tooltips.
    alt = " / ".join(
        f"{chord} {inker_state.tool_label(tool)}" for tool, chord in ALT_TOOL_CHORDS.items()
    )
    table(
        "Inker",
        [
            *tool_rows,
            (alt, "The same tools on Aseprite's shifted letters"),
            ("X", "Swap colours"),
            ("1 - 0", "Brush opacity, 10% to 100%"),
            ("Alt+1 - 9", "Recall a numbered custom brush"),
            ("Alt+Shift+1 - 9", "Store the captured brush in that slot"),
            ("Ctrl+Shift+N", "New layer"),
            ("Ctrl+Alt+I / Ctrl+Alt+C", "Image size / canvas size"),
            ("Alt+S", "Solo the active layer, and again to bring the rest back"),
            ("Ctrl+Shift+Up / Down", "Move the layer up / down the stack"),
            ("[ / ]", "Brush size (Shift: hardness)"),
            ("Shift+click", "Paint a line from where the last stroke ended"),
            ("+ / -", "Zoom in / out, by whole scales"),
            ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
            ("Space / middle drag", "Pan"),
            ("Wheel", "Zoom in 5% steps -- the rule every canvas shares"),
            ("Shift+wheel", "Scroll sideways"),
            ("Ctrl+4 / Ctrl+5", "Rotate the view a quarter turn / flip it"),
            ("Arrows", "Nudge a pixel (Shift: eight)"),
            ("Delete", "Delete what is selected"),
            ("Esc", "Cancel -- a move, playback, a float, then the selection"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            ("Ctrl+E", "Save as a reference in the library"),
            ("Ctrl+Shift+E", "Export PNG"),
            ("Ctrl+Shift+X", "Repeat the last export -- same file, no dialog"),
            ("Ctrl+N / O / W", "New / open / close"),
            ("Ctrl+A / D", "Select all / deselect"),
            ("Ctrl+Shift+D", "Reselect what was last dismissed"),
            ("Ctrl+C / X / V", "Copy / cut / paste"),
            ("Ctrl+Shift+V", "Paste as a layer"),
            ("Ctrl+Shift+C", "Copy merged -- what is visible in the selection"),
            ("Ctrl+J / Ctrl+Shift+J", "Copy / move the selection to its own layer"),
            ("Ctrl+Shift+I", "Invert the selection"),
            ("Ctrl+T", "Free transform"),
            ("Ctrl+B", "Capture the selection as an image brush"),
            ("Ctrl+Tab / Ctrl+Shift+Tab", "Next / previous tab"),
            (", / .", "Previous / next frame (animated)"),
            ("Enter", "Play or pause (animated)"),
        ],
    )
    from .plotter_state import TOOLS as PLOTTER_TOOLS

    table(
        "Plotter",
        [
            ("1 - 9", "Recall a numbered stamp"),
            ("Ctrl+Shift+1 - 9", "Store the stamp in hand in that slot"),
            ("Right-drag", "Capture a block off the map, keeping the tool"),
            ("Right-click an object", "Duplicate, raise, lower or delete it"),
            ("H", "Highlight the current layer"),
            ("+ / -", "Zoom in / out, by whole scales"),
            ("Ctrl+Shift+I", "Invert the selection"),
            (
                " / ".join(letter for _k, _l, letter in PLOTTER_TOOLS),
                " / ".join(label for _k, label, _letter in PLOTTER_TOOLS),
            ),
            ("X / Y / Z", "Flip the brush across, down; turn it (Shift turns back)"),
            ("Shift+click", "Stamp a line from the last cell painted"),
            ("Pick drag", "Capture a block off the map as the brush"),
            ("Wand Ctrl+click", "Select every cell of that tile, map-wide"),
            ("Shift / Alt", "Add to / subtract from the selection (Wand and marquee)"),
            ("Ctrl+A / Ctrl+D", "Select all / deselect (Ctrl+Shift+A also)"),
            ("Ctrl+C / Ctrl+X / Ctrl+V", "Copy / cut / paste as the brush"),
            ("Ctrl+J", "Duplicate the selected object"),
            ("Ctrl+click / Alt+click", "Insert / remove a polygon vertex"),
            ("Delete", "Clear the selection, or remove the object"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            ("Ctrl+E", "Export to the library"),
            ("Ctrl+Shift+E", "Export a Tiled .tmx"),
            ("Ctrl+N / O / W", "New / open / close"),
            ("Ctrl+G", "Toggle the grid"),
            ("Ctrl+Tab / Ctrl+Shift+Tab", "Next / previous map"),
            ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
            ("Space / middle drag", "Pan (wheel zooms)"),
            ("Esc", "Cancel a drag, then the object, then the selection"),
        ],
    )
    table(
        "Poser",
        [
            # The mode is otherwise mouse-shaped -- joints are clicked and
            # gizmos are dragged. The view keys and the mouse rows are Clay's,
            # on the same chords: the 2026-09-05 pass found this table saying
            # "two rows are the whole group" a week after six more were bound.
            ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            ("Ctrl+1 / Ctrl+3 / Ctrl+7", "Look along front / right / top (Shift: the opposite)"),
            ("Ctrl+5", "Toggle orthographic and perspective"),
            ("F", "Frame the armature"),
            ("Alt+drag", "Orbit (middle drag pans)"),
            ("Esc", "Deselect the joint"),
        ],
    )
    table(
        "Packwright",
        [
            ("R", "Repack now"),
            ("Delete", "Remove the selected source"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            ("Ctrl+E", "Export to the library"),
            ("Ctrl+Shift+E", "Export the atlas and its JSON"),
            ("Ctrl+N / O / W", "New / open / close"),
            ("Ctrl+Tab / Ctrl+Shift+Tab", "Next / previous atlas"),
            ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
            # Middle drag alone, not "Space / middle drag" as Plotter's row
            # says: there is no space-pan in this mode to advertise.
            ("Middle drag", "Pan (wheel zooms)"),
        ],
    )
    table(
        "Troupe",
        [
            ("Space", "Play / pause the preview"),
            # Stepping pauses, which is why the two rows are not "step" alone:
            # the binding does two things and a sheet that named one of them
            # would be describing a different control.
            ("Left / Right", "Step one frame, and pause"),
            ("Up / Down", "Turn the character one direction, holding the frame"),
            ("PageUp / PageDown", "Previous / next animation"),
            ("Home / End", "First / last frame of the run, and pause"),
        ],
    )
    table(
        "Muse",
        [
            ("Space", "Play or stop the selected take"),
            ("Up / Down", "Move the tray's selection"),
            ("Ctrl+Enter", "Generate, from wherever the caret is"),
            ("Left / Right", "Nudge the playhead a second (Shift: ten)"),
            ("Home", "Playhead back to the start"),
            ("[ / ]", "Set the loop's start / end at the playhead"),
            ("L", "Look for loop points"),
        ],
    )
    table(
        "Sirens",
        [
            ("Space", "Play the song, or stop it if it is sounding"),
            # The two piano rows as one row rather than twenty-four, and spelled
            # exactly as the chapter spells them: they are a *layout* rather
            # than twenty-four bindings, and a sheet that listed each letter
            # would be a sheet nobody could read past.
            ("zsxdcvgbhnjm / q2w3er5t6y7u", "The two piano rows, in the note column"),
            ("Backtick / Shift+Backtick", "Note-off (cuts) / release (plays the tail)"),
            # The three key cells below are deliberately prose. Which hex digit
            # or which letter is not a binding -- the *column the caret is in*
            # is what decides what the key means, and a sheet listing sixteen
            # digits twice would be a sheet nobody reads past.
            ("Hex digits", "Instrument and parameter: two digits. Volume: one"),
            ("An effect's letter", "The effect column -- only letters the engine has"),
            ("- / =", "Octave down / up"),
            ("Shift+1 / Shift+2", "Transpose the selection down / up a semitone"),
            ("Page Up / Page Down", "Move sixteen rows -- four beats"),
            ("Delete", "Clear the column under the caret, or the whole selection"),
            ("Ctrl+C / Ctrl+X / Ctrl+V", "Copy / cut / paste a block at the caret"),
            ("Esc", "Drop the selection"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            ("Ctrl+N / O / W", "New song / open a file / close the tab"),
            ("Ctrl+Tab / Ctrl+Shift+Tab", "Next / previous song"),
            ("Ctrl+Shift+E", "Export WAV + stems"),
        ],
    )
    return sections


def filter_shortcuts(
    sections: list[tuple[str, list[tuple[str, str]]]], query: str
) -> list[tuple[str, list[tuple[str, str]]]]:
    """The shortcut list narrowed by ``query`` (UX.md Phase 4).

    Pure, and through ``palette.match`` rather than a substring test, because
    the popup and the command palette are two lists of the same kind of thing
    and a second matcher is a second answer to "does 'ctz' find Ctrl+Z".

    A row matches on its keys, its description **or its group's name**, which is
    the rule that makes "clay" list Clay's fifteen bindings rather than the two
    whose text happens to say the word. A group whose heading matched keeps all
    of its rows for the same reason; a group with no surviving row is dropped
    entirely, because a heading over nothing reads as a section that broke.

    Rows are deliberately *not* re-ordered by score: within a group they are in
    a hand-chosen order, and a filter that also reshuffles is two changes to
    read at once.
    """
    from . import palette

    if not query.strip():
        return list(sections)
    out = []
    for title, rows in sections:
        if palette.match(query, title) is not None:
            out.append((title, list(rows)))
            continue
        kept = [row for row in rows if palette.match(query, f"{row[0]} {row[1]}") is not None]
        if kept:
            out.append((title, kept))
    return out
