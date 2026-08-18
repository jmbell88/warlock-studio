# Keyboard shortcuts

A shorter version is in the app: press the **?** button in the top-right of the top bar. That popup
is a condensed subset — the tables below are the full list. It has a filter box of its own at the
top, matched the way the command palette matches (so `ctz` finds `Ctrl+Z`) against a binding's keys,
its description, or the name of the group it is in — typing `clay` lists all of Clay's rather than
the two whose wording happens to say the word.

One rule explains an apparent overlap between the tables below. Inker, Clay and Review each take
every key while they are on screen — so in Inker `W` picks the wand rather than toggling wireframe,
in Clay `W` is the move tool, and in Review `S` skips a unit. Create's viewport bindings would
otherwise act on a viewport that is not on screen.

## Everywhere

These five work in every mode, including Inker, Clay and Review — they are checked before any mode
sees the key, which is the one exception to the rule above.

| Keys | Action |
| --- | --- |
| Ctrl+K | Open the command palette — switch mode, or open an asset |
| Ctrl+/ | Open this list, as a searchable popup |
| F1 | Open this manual over whatever is on screen, or put it away again |
| Esc | Close the manual, then the profile sheet, then leave a mode you passed through |
| F10 | Toggle the frame-rate readout |

**There is no per-mode digit.** There was, while there were ten modes and ten digits: `Alt+N` for
the nth segment, so the binding was the picture on screen rather than a second table to keep in
agreement with it. That argument stopped holding as soon as there were twelve segments — either two
modes have no key, or something has to say which two, and that something is exactly the second table
the positional scheme existed to avoid. Switching modes is a mouse action and a palette action, and
the digits go back to the workspace modes that were already reaching for them.

**Esc goes through what is on top first.** The manual can be raised over the profile sheet, and the
sheet over a mode, so Esc closes the topmost of those before it means anything to the mode
underneath. Closing the profile sheet over a half-typed profile still asks before discarding it.
After that: in a mode with something to drop — a comparison, a pose edit, a floating selection —
Esc drops that and stays put, and only leaves a mode that has nothing of its own to cancel. Home is
where it stops: the app opens there, so there is usually nothing behind it.

## Moving around without the mouse

**Tab moves to the next control, Shift+Tab to the previous, Space or Enter operates the one you
land on.** That works in every pane — forms, the app's Settings, the profile sheet, the library, the
inspector, the navigation rail. The control you are on is drawn with an accent-coloured ring around it,
and a button that shows only an icon puts its name in a tooltip as you arrive, so you never have to
recognise a glyph to know what pressing it would do.

**The arrow keys belong to whatever is on screen.** Home and the library move their selection with
Up and Down, Review steps between units with Left and Right, and Inker and Plotter pan while Space
is held. In those five the arrows do that and nothing else — they do not also step the ring, which
would be two things answering one key. Tab is never taken over in this way, which is what keeps
traversal available everywhere.

Inside a text field the arrows go back to being cursor movement, whichever mode you are in, and Tab
leaves the field rather than typing into it.

Typed text comes from the operating system rather than from the raw key, so a compose key, an accent
and an input method all produce what they should — including characters outside the Basic
Multilingual Plane, which are no longer dropped on the way in. An input method's candidate window is
placed against the field you are typing into.

## The command palette

`Ctrl+K` opens a search box over whatever is on screen. Type a few letters — initials work, so
`gtc` finds *Go to Clay* — then Up/Down to move and Enter to run. Esc closes it.

It lists every mode, the generate action, the viewport toggles and the actions for the selected
asset. A command that cannot run right now is still listed, greyed: an empty result would not tell
you the command exists or which mode owns it.

Typing also searches your assets by name, prompt or job id; picking one selects it and opens it in
the pane that made it.

## Home and the Library

Both screens are lists, and both take the same keys — the Resume list on Home, the asset cards in
the Library.

| Keys | Action |
| --- | --- |
| Up / Down | Move through the rows / the cards |
| Enter | Open the highlighted row — a Library asset opens at the stage that made it |

In the Library the arrows move through the cards exactly as they do in Create's sidebar: the
same filtered, sorted list the cards are drawn from, clamped at the ends rather than wrapping.

## Create

| Keys | Action |
| --- | --- |
| Ctrl+Enter | Run the stage: Generate, or Make 3D |
| Tab / Shift+Tab | Move between the form's controls |
| Enter | Press the stage's button when it is the one focused |
| Up / Down | Previous / next asset in the library |
| Right-click a card | The same actions menu the `...` button opens |
| F | Frame the model |
| W | Toggle wireframe |
| S | Toggle turntable |
| Esc | Exit comparison / pose edit |
| Ctrl+Z / Ctrl+Y | Undo / redo a pose edit (Ctrl+Shift+Z also redoes) |

The undo row applies **while pose edit is open**, and only there. It is the pose editor's own
history — one step per gizmo drag, per preset, per mirror and per reset — and it is the same
history Poser uses, because it is the same editor. It is dropped when you leave pose mode: a step
holds rotations by bone name, and replaying one onto a different skeleton would find whichever
bones happened to share a name.

## Review

| Keys | Action |
| --- | --- |
| 1 – 5 | Grade the mesh +1 to +5 (+3 is usable) |
| R then 1 – 5 | Grade it −1 to −5 |
| 0 | Grade it 0 — no opinion either way |
| Ctrl + 1 – 5 | Toggle a good tag for the next grade |
| Shift + 1 – 5 | Toggle a bad tag for the next grade |
| S | Skip to the next unverdicted unit |
| Left / Right | Previous / next unit |
| Esc | Clear the pending sign and tags |

## Review — a judging pass

While a judging pass is open, four keys mean something else. Everything not listed here — the
digits, the tag modifiers, the arrows — works exactly as it does above, and a grade pressed during a
pass files and advances the pass like an accept does.

| Keys | Action |
| --- | --- |
| A | Accept — files +3 |
| R | Reject — files −3, rather than arming a negative sign |
| S | Skip to the next unjudged unit, staying in the pass |
| Esc | End the pass and show its report |

## Review — labelling images

While a labelling pass is open it owns these keys, and the verdict bindings above do not apply:
one loop at a time, so a keypress about a picture can never be filed as a verdict about a mesh.

| Keys | Action |
| --- | --- |
| A | Good |
| R | Bad — no reason step |
| S | Skip to the next unanswered image |
| Left / Right | Previous / next image |
| Esc | Close the pass |

## Clay

| Keys | Action |
| --- | --- |
| Q / W / E / R | Select / move / rotate / scale |
| 1 / 2 / 3 / 4 | Vertex / edge / face / object mode |
| E | Extrude, in any element mode |
| F | Frame the selection |
| X / Y / Z | Lock a drag already under way to that axis; the same key again clears it |
| digits, `.`, `-` | Type the drag's value outright; `Backspace` takes a character back |
| Enter | Commit the drag |
| Delete | Delete the selection — faces in an element mode, objects in object mode |
| Ctrl+D | Duplicate the selection (object mode only) |
| Ctrl+J | Merge the selected objects into one (object mode only) |
| Ctrl+Shift+J | Union the selected objects — as a merge, but cutting away what is inside the overlap |
| Ctrl+A | Select everything, in the current mode's sense |
| Ctrl+Shift+I | Invert the selection |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+N / Ctrl+O | New / open a document |
| Ctrl+E | Export to the library |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous document |
| Ctrl+W | Close the document |
| Ctrl+1 / Ctrl+3 / Ctrl+7 | Look along front / right / top |
| Ctrl+Shift+1 / +3 / +7 | The opposite view: back / left / bottom |
| Ctrl+5 | Toggle orthographic and perspective |
| Esc | Cancel a drag; otherwise step back: element selection, then element mode, then object selection |

**The mouse.** Left-drag in empty space orbits, and `Alt`+left-drag always orbits whatever mode you
are in. Middle-drag pans and the wheel dollies. Right-click opens the context menu — right-drag does
nothing, so grabbing the wrong button mid-orbit costs you nothing.

In an element mode, left-click selects an element, `Shift`+click adds and `Ctrl`+click removes;
left-drag in empty space with `Q` selected sweeps a marquee.

## Inker

| Keys | Action |
| --- | --- |
| `B` | Brush |
| `A` | Spray |
| `E` | Eraser |
| `G` | Fill |
| `K` or `Shift+G` | Gradient |
| `R` | Blur |
| `N` | Smudge |
| `H` | Shading |
| `L` | Line |
| `F` | Curve (Enter or a double-click finishes it, Esc abandons it) |
| `U` | Rect |
| `J` or `Shift+U` | Ellipse |
| `P` | Polyline (finished the same way as the curve) |
| `O` | Polygon (also closes if you click back on its first point) |
| `M` | Marquee select |
| `S` or `Shift+M` | Ellipse select |
| `Q` | Lasso |
| `D` | Poly lasso (Enter or a double-click closes it, Esc abandons it) |
| `W` | Wand |
| `V` | Move |
| `I` | Pick a colour from the canvas |
| `T` | Text |
| `C` | Slice |
| `X` | Swap the two colours |
| `[` / `]` | Brush size (with Shift, hardness) |
| `+` / `-` | Zoom in / out, by whole scales |
| Shift+click | With a paint tool, draw a line from where the last stroke ended |
| Arrows | Nudge by a pixel — the floating selection, or the layer under the Move tool (Shift, 8 px) |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+E | Save the drawing into the library as a reference |
| Ctrl+Shift+E | Export a flattened PNG |
| Ctrl+N | New document |
| Ctrl+O | Open a file |
| Ctrl+W | Close the current tab |
| Ctrl+A / Ctrl+D | Select all / deselect |
| Ctrl+Shift+D | Reselect what Ctrl+D dismissed |
| Ctrl+C / Ctrl+X / Ctrl+V | Copy / cut / paste |
| Ctrl+Shift+V | Paste as a new layer |
| Ctrl+Shift+I | Invert the selection |
| Ctrl+J / Ctrl+Shift+J | Layer from selection — copy it up / cut it up |
| Ctrl+T | Free transform (Enter applies, Esc cancels) |
| Ctrl+B | Capture the selection as the brush tip |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous tab |
| Ctrl+0 / Ctrl+1 | Fit to the pane / 100% |
| Ctrl+4 / Ctrl+Shift+4 | Turn the view a quarter clockwise / anticlockwise |
| Ctrl+5 | Mirror the view left to right |
| Space drag, middle drag | Pan (the wheel zooms) |
| Delete | Clear the selected pixels |
| Esc | Cancel a floating selection, then deselect |

Shift and Alt are modifiers rather than shortcuts: holding **Shift** while dragging a selection adds
to the current one, and **Alt** subtracts. Both are described in
[Selections and transform](08-inker.md#selections-and-transform). With a *paint* tool Shift means
something else — it draws a straight line from wherever the last stroke finished, so click,
Shift+click, Shift+click walks a chain of segments in the brush already in your hand.

Three tools answer to a shifted letter as well as their own, because Aseprite files them
two-to-a-slot and a hand trained there reaches for the shifted one: `Shift+G` is the gradient,
`Shift+U` the ellipse, `Shift+M` the elliptical marquee. The plain letters still work.

`+` and `-` step through whole zoom scales — 25%, 50%, 100%, 200%, 300% and up — rather than the
wheel's 5% notches. Pixel art wants the whole ones: at 135% a source pixel is 1.35 screen pixels, so
some are drawn one wide and some two, and a dither comes out as bands. The wheel is still the fine
control, and Ctrl+0 and Ctrl+1 still fit and reset.

### Inker — an animated document

These are live only once a document has been animated; on a still drawing the keys do nothing.

| Keys | Action |
| --- | --- |
| `,` / `.` | Step back / forward a frame |
| Enter | Play, and Play again to stop |
| Esc | Stop playback |

Enter reaches playback only after the transform branch has had it: with a free transform open, Enter
applies the transform and does not also start the clip. See
[Playback](09-inker-animation.md#playback).

## Poser

| Keys | Action |
| --- | --- |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Esc | Deselect the joint |

The mode is otherwise mouse-shaped: joints are clicked and gizmos are dragged. The undo binding is
the same one the asset pose editor answers to, because they are
[one editor with two doors](06-poser.md#undo-and-redo).

## Plotter

| Keys | Action |
| --- | --- |
| `B` | Stamp |
| `E` | Erase |
| `F` | Fill |
| `T` | Terrain |
| `P` | Shape (rectangle or ellipse) |
| `R` | Rectangular select |
| `I` | Pick the tile under the cursor |
| `S` | Objects |
| `X` / `Y` | Flip the brush across / down |
| `Z` | Turn the brush a quarter clockwise (Shift+Z turns it back) |
| Shift+click | Stamp a line from the last cell painted |
| Ctrl+A / Ctrl+D | Select all / deselect (Ctrl+Shift+A also deselects) |
| Ctrl+C / Ctrl+X | Copy / cut the selected cells |
| Ctrl+V | Load the copy into the brush and switch to Stamp |
| Delete | Clear the selected cells, or remove the selected object |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+E | Export to the library |
| Ctrl+Shift+E | Export a Tiled `.tmx` |
| Ctrl+N | New map |
| Ctrl+O | Open a file |
| Ctrl+W | Close the current tab |
| Ctrl+G | Toggle the grid |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous map |
| Ctrl+0 / Ctrl+1 | Fit to the pane / 100% |
| Space drag, middle drag | Pan (the wheel zooms) |
| Esc | Cancel a drag, then the object, then the selection |

The tool letters are [Tiled](https://www.mapeditor.org/)'s. This editor reads and writes Tiled's
files, so the editor you are most likely arriving from is that one — and its letters differ from
[Inker](08-inker.md)'s in two places, so following both was never possible.

Esc is **staged**: one press undoes one thing, outermost first. Cancelling a drag does not also
throw away the selection you spent a gesture placing.

## Packwright

| Keys | Action |
| --- | --- |
| `R` | Repack now |
| Delete | Remove the selected source |
| Ctrl+Z / Ctrl+Y | Undo / redo |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+E | Export to the library |
| Ctrl+Shift+E | Export the atlas and its JSON |
| Ctrl+N | New atlas |
| Ctrl+O | Open a file |
| Ctrl+W | Close the current tab |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous atlas |
| Ctrl+0 / Ctrl+1 | Fit to the pane / 100% |
| Middle drag | Pan (the wheel zooms) |
| Esc | Deselect |

Packing is automatic — `R` is only there for when you want it now rather than on the next change.
