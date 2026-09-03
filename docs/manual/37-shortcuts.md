# Keyboard shortcuts

A shorter version is in the app: press **Ctrl+/**, or the **Shortcuts** button in the navigation
rail's footer. That popup
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
| Esc | Close the manual, then a running tour, then leave a mode you passed through |
| F10 | Toggle the frame-rate readout |

**There is no per-mode digit.** There was, while there were ten modes and ten digits: `Alt+N` for
the nth segment, so the binding was the picture on screen rather than a second table to keep in
agreement with it. That argument stopped holding as soon as there were twelve segments — either two
modes have no key, or something has to say which two, and that something is exactly the second table
the positional scheme existed to avoid. Switching modes is a mouse action and a palette action, and
the digits go back to the workspace modes that were already reaching for them.

**Esc goes through what is on top first.** The manual can be raised over a running tour (a step's
*Read more* does exactly that), and a tour runs over a mode —
so Esc closes the topmost of those before it means anything to the mode underneath.
After that: in a mode with something to drop — a comparison, a pose edit, a floating selection —
Esc drops that and stays put, and only leaves a mode that has nothing of its own to cancel. Home is
where it stops: the app opens there, so there is usually nothing behind it.

## Moving around without the mouse

**Tab moves to the next control, Shift+Tab to the previous, Space or Enter operates the one you
land on.** That works in every pane — forms, the app's Settings, the library, the
inspector, the navigation rail. The control you are on is drawn with an accent-coloured ring around it,
and a button that shows only an icon puts its name in a tooltip as you arrive, so you never have to
recognise a glyph to know what pressing it would do.

**The arrow keys belong to whatever is on screen.** Home moves its selection with Up and Down and
the library moves through its grid with all four, Review steps between units with Left and Right,
Troupe steps a clip a frame at a time with Left and Right, and Inker and Plotter pan while Space
is held. In those six the arrows do that and nothing else — they do not also step the ring, which
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

It lists every mode, the generate action, the viewport toggles, the actions for the selected asset,
and each guided tour as **Take the tour: …** — see [New here?](21-home.md#new-here). A command that
cannot run right now is still listed, greyed: an empty result would not tell you the command exists
or which mode owns it.

Everything in the palette is also in the menu bar, and the reverse: both are drawn from the same
registry, so neither can offer something the other does not.

Typing also searches your assets by name, prompt or job id; picking one selects it and opens it in
the pane that made it.

## Home and the Library

Both screens are selections you walk with the arrows and open with Enter — the Resume list on Home,
the asset cards in the Library — but the two are shaped differently, because one is a short ring and
the other is a grid.

| Keys | Action |
| --- | --- |
| Up / Down | Move through the Resume rows / up and down a row of cards |
| Left / Right | Move one card (the Library only) |
| Enter | Open the highlighted row — a Library asset opens at the stage that made it |

Home's Resume list is one column and **wraps** at both ends: it is a short ring rather than a list,
so pressing Up at the top is not a dead key. The Library is a grid, so Up and Down move by a whole
row and Left and Right by one card, over the same filtered, sorted list the cards are drawn from —
and it is **clamped** rather than wrapped, because that list is the newest N of many and one press
at the top landing on the oldest asset the window happens to hold is a jump the other arrow will not
undo. Create's sidebar walks the same list with Up and Down alone, its rows being one card wide.

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
| G / S | Move / scale the selection with no handle to grab — the drag follows the pointer |
| L | Select everything joined to what is selected — two shapes welded into one mesh come apart |
| Ctrl+= / Ctrl+- | Grow / shrink the selection by one ring |
| Alt+click | Select the edge loop under the pointer; `Ctrl`+`Alt`+click takes the ring instead |
| G / R / S | Switch which transform a running drag is doing; the objects go back first |
| F | Frame the selection |
| X / Y / Z | Lock a drag already under way to that axis; the same key again clears it |
| digits, `.`, `-` | Type the drag's value outright; `Backspace` takes a character back |
| Enter | Commit the drag |
| Left-click / right-click | Commit / cancel a keyboard drag — it holds no button, so a press is how it ends |
| Delete | Delete the selection — faces in an element mode, objects in object mode |
| Ctrl+J | Duplicate the selection (object mode only) |
| Ctrl+M | Merge the selected objects into one (object mode only) |
| Ctrl+Shift+M | Union the selected objects — as a merge, but cutting away what is inside the overlap |
| Ctrl+D | Deselect — the same key Inker and Plotter use |
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
| Alt+drag | Orbit, whatever mode you are in |
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
| `A` or `Shift+B` | Spray |
| `E` | Eraser |
| `G` | Fill |
| `K` or `Shift+G` | Gradient |
| `R` | Blur |
| `N` | Smudge |
| `H` | Shading |
| `L` | Line |
| `F` or `Shift+L` | Curve (Enter or a double-click finishes it, Esc abandons it) |
| `U` | Rect |
| `J` or `Shift+U` | Ellipse |
| `P` | Polyline (finished the same way as the curve) |
| `O` or `Shift+D` | Polygon (also closes if you click back on its first point) |
| `M` | Marquee select |
| `S` or `Shift+M` | Ellipse select |
| `Q` | Lasso |
| `D` | Poly lasso (Enter or a double-click closes it, Esc abandons it) |
| `W` | Wand |
| `V` | Move |
| `I` | Pick a colour from the canvas |
| `T` | Text |
| `C` or `Shift+C` | Slice |
| `Y` | Tile stamp |
| `X` | Swap the two colours |
| `1` - `0` | Brush opacity, 10% to 100% |
| Alt+`1` - `9` | Recall a numbered custom brush |
| Alt+Shift+1 - 9 | Store the captured brush in that slot |
| Alt+N / Alt+D | New frame / duplicate the current frame |
| Home / End | First / last frame |
| F3 | Onion skin on or off |
| Ctrl+U / Ctrl+I | Hue / saturation, and Invert colours -- both open the filter popup |
| Ctrl+Alt+I | Image size (resamples the picture) |
| Ctrl+Alt+C | Canvas size (changes the room around it) |
| Ctrl+Shift+N | New layer |
| Ctrl+Shift+L | Duplicate the layer |
| Ctrl+Shift+M | Merge the layer down into the one below |
| Shift+H / Shift+V | Flip the whole sprite across / down |
| Alt+S | Solo the active layer, and again to bring the rest back |
| Ctrl+Shift+Up / Down | Move the layer up / down the stack |
| `[` / `]` | Brush size (with Shift, hardness) |
| `+` / `-` | Zoom in / out, by whole scales |
| Shift+click | With a paint tool, draw a line from where the last stroke ended |
| Arrows | Nudge by a pixel — the floating selection, or the layer under the Move tool (Shift, 8 px) |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+E | Save the drawing into the library as a reference |
| Ctrl+Shift+E | Export a flattened PNG |
| Ctrl+Shift+X | Repeat the last export -- same file, no dialog |
| Ctrl+N | New document |
| Ctrl+O | Open a file |
| Ctrl+W | Close the current tab |
| Ctrl+A / Ctrl+D | Select all / deselect |
| Ctrl+Shift+D | Reselect what Ctrl+D dismissed |
| Ctrl+C / Ctrl+X / Ctrl+V | Copy / cut / paste |
| Ctrl+Shift+V | Paste as a new layer |
| Ctrl+Shift+C | Copy what is visible inside the selection (merged) |
| Ctrl+Shift+I | Invert the selection |
| Ctrl+J / Ctrl+Shift+J | Layer from selection — copy it up / cut it up |
| Ctrl+T | Free transform (Enter applies, Esc cancels) |
| Ctrl+B | Capture the selection as the brush tip |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous tab |
| Ctrl+0 / Ctrl+1 | Fit to the pane / 100% |
| Ctrl+4 / Ctrl+Shift+4 | Turn the view a quarter clockwise / anticlockwise |
| Ctrl+5 | Mirror the view left to right |
| Space drag, middle drag | Pan |
| Shift+wheel | Scroll the canvas sideways (the wheel alone scrolls it up and down) |
| Ctrl+wheel | Zoom in 5% steps, and by whole scales past 800% |
| Delete | Clear the selected pixels |
| Esc | Cancel a floating selection, then deselect |

Selection modifiers follow Aseprite: **Shift** adds, **Alt+Shift** subtracts, and
**Ctrl+Shift** intersects. They are described in
[Selections and transform](28-inker.md#selections-and-transform). With a *paint* tool Shift means
something else — it draws a straight line from wherever the last stroke finished, so click,
Shift+click, Shift+click walks a chain of segments in the brush already in your hand.

The Aseprite bindings are primary: `Shift+B` spray, `Shift+G` gradient, `Shift+L` curve,
`Shift+U` ellipse, `Shift+D` polygon, `Shift+M` elliptical marquee, and `Shift+C` slice.
The earlier Inker letters remain as compatibility aliases.

**Edit > Keyboard Shortcuts** (`Ctrl+Alt+Shift+K`) searches commands, tools and contextual action
modifiers. A target can have multiple semicolon-separated bindings. **Default** restores one target;
**Reset all** restores the complete Aseprite-compatible table. **Copy JSON** and **Paste JSON** export
and import the persisted overrides without replacing defaults that were never changed. Holding
`Alt` temporarily selects the eyedropper and holding `Ctrl` temporarily selects Move; releasing the
key restores the tool you were using.

`+` and `-` step through whole zoom scales — 25%, 50%, 100%, 200%, 300% and up — rather than the
wheel's 5% notches. Pixel art wants the whole ones: at 135% a source pixel is 1.35 screen pixels, so
some are drawn one wide and some two, and a dither comes out as bands. Ctrl+wheel is still the fine
control, and Ctrl+0 and Ctrl+1 still fit and reset. The wheel on its own scrolls rather than zooming,
which is Aseprite's default.

**The ceiling is 6400%**, the same as Aseprite's: a 16 px sprite at the old 1000% was 160 screen
pixels, which on a large display is a stamp in the middle of an empty canvas. Past 800% a Ctrl+wheel
notch is a whole scale rather than 5%, because 5% of 6400% is a twentieth of a pixel and the top
would otherwise be a thousand notches away.

### Inker — an animated document

These are live only once a document has been animated; on a still drawing the keys do nothing.

| Keys | Action |
| --- | --- |
| `,` / `.` | Step back / forward a frame |
| Enter | Play, and Play again to stop |
| Esc | Stop playback |

Enter reaches playback only after the transform branch has had it: with a free transform open, Enter
applies the transform and does not also start the clip. See
[Playback](29-inker-animation.md#playback).

## Poser

| Keys | Action |
| --- | --- |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Ctrl+S / Ctrl+Shift+S | Save the pose / save it under a new name |
| Esc | Deselect the joint |

The mode is otherwise mouse-shaped: joints are clicked and gizmos are dragged. The undo binding is
the same one the asset pose editor answers to, because they are
[one editor with two doors](26-poser.md#undo-and-redo).

## Plotter

| Keys | Action |
| --- | --- |
| `B` | Stamp |
| `E` | Erase |
| `F` | Fill |
| `T` | Terrain |
| `P` | Shape (rectangle or ellipse) |
| `R` | Rectangular select |
| `W` | Wand |
| `I` | Pick the tile under the cursor |
| `S` | Objects (on an object layer, `R` `I` `E` `P` `L` `T` `X` insert a shape) |
| `1` - `9` | Recall a numbered stamp |
| Ctrl+Shift+1 - 9 | Store the stamp in hand in that slot |
| Right-drag | Capture a block off the map, without leaving the tool |
| Right-click an object | Duplicate, raise, lower or delete it |
| `H` | Highlight the current layer |
| `+` / `-` | Zoom in / out, by whole scales |
| Ctrl+Shift+I | Invert the selection |
| `X` / `Y` | Flip the brush across / down |
| `Z` | Turn the brush a quarter clockwise (Shift+Z turns it back) |
| Shift+click | Stamp a line from the last cell painted |
| Pick drag | Capture a block off the map as the brush |
| Wand Ctrl+click | Select every cell using that tile, map-wide |
| Shift / Alt | Add to / subtract from the selection (Wand and marquee) |
| Ctrl+J | Duplicate the selected object |
| Ctrl+click / Alt+click | Insert / remove a polygon vertex |
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
[Inker](28-inker.md)'s in two places, so following both was never possible.

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

## Troupe

| Keys | Action |
| --- | --- |
| Space | Play / pause the preview |
| Left / Right | Step one frame, and pause |
| Up / Down | Turn the character one direction, holding the frame |
| PageUp / PageDown | Previous / next animation |
| Home / End | First / last frame of the run, and pause |

Turning holds the frame on purpose, so you can see the same moment of a stride from another side.
Changing animation starts the new one from its first frame.

There is nothing else. Troupe holds no document — the sheet it plays was published by a job and
lives in that job's directory — so there is no save, no undo and no tab to close.

## Sirens

| Keys | Action |
| --- | --- |
| Space | Play the song, or stop it if it is sounding |
| `zsxdcvgbhnjm` | The lower piano row — the octave the Octave field names |
| `q2w3er5t6y7u` | The upper piano row — one octave above it |
| Backtick | Write a note-off (`===`) — cuts the voice dead |
| Shift+Backtick | Write a release (`~~~`) — plays the instrument's release tail |
| `0`–`9`, `A`–`F` | Hex digits, in the instrument, volume and parameter columns |
| An effect's letter | In the effect column — only the letters the engine has |
| `-` / `=` | Octave down / up |
| Up / Down | Move the caret a row |
| Left / Right | Move the caret a column |
| Shift+Up / Shift+Down | Extend the selection by a row |
| Shift+Left / Shift+Right | Extend the selection by a channel |
| Page Up / Page Down | Move sixteen rows — four beats |
| Shift+1 / Shift+2 | Transpose the selection down / up a semitone |
| Delete / Backspace | Clear the column under the caret, or the whole selection |
| Ctrl+C / Ctrl+X | Copy / cut the selection (or the caret's cell) as a block |
| Ctrl+V | Paste the block at the caret |
| Esc | Drop the selection |
| Ctrl+Z / Ctrl+Y | Undo / redo |
| Ctrl+Shift+Z | Redo as well |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+N | New song |
| Ctrl+O | Open a file |
| Ctrl+W | Close the current tab |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous song |

**Which column the caret is in decides what a key means**, which is why the same letters appear
twice above. The piano rows fire in the **note** column only: `e` in the effect column is the letter
of an effect rather than an E natural, and `c` in the volume column is the hex digit twelve. The
instrument and parameter columns take two digits — the first fills the high nibble, the second the
low, and the caret rings whichever one is next — while volume takes one. See
[Sirens](34-sirens.md#the-pattern-grid).

Those are *letter* positions rather than physical key positions, so the layout is right for anyone
arriving from another tracker and wrong on an AZERTY keyboard.

There is no binding for the export — it opens a folder picker, and it is the **Export audio...**
button on the Song file panel.
