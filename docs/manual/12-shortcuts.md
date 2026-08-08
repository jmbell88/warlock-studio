# Keyboard shortcuts

A shorter version is in the app: press the **?** button in the top-right of the top bar. That popup
is a condensed subset — the tables below are the full list.

One rule explains an apparent overlap between the tables below. Inker, Clay and Review each take
every key while they are on screen — so in Inker `W` picks the wand rather than toggling wireframe,
in Clay `W` is the move tool, and in Review `S` skips a unit. The 2D / 3D viewport bindings would
otherwise act on a viewport that is not on screen.

## Everywhere

These four work in every mode, including Inker, Clay and Review — they are checked before any mode
sees the key, which is the one exception to the rule above.

| Keys | Action |
| --- | --- |
| Alt+1 … Alt+8 | Switch mode, in the order the switch draws them |
| Ctrl+K | Open the command palette |
| F1 | Switch to the Manual |
| Esc | Leave Home, the Manual or app Settings, back to the mode you came from |
| F10 | Toggle the frame-rate readout |

**Why Alt and not Ctrl.** Inker already uses Ctrl+0 and Ctrl+1 for its zoom, and a binding checked
above the modes takes whatever it names away from them for good.

**Esc.** In a mode with something to drop — a comparison, a pose edit, a floating selection — Esc
drops that and stays put. It only leaves a mode that has nothing of its own to cancel. Home is
where it stops: the app opens there, so there is usually nothing behind it.

## The command palette

`Ctrl+K` opens a search box over whatever is on screen. Type a few letters — initials work, so
`gtc` finds *Go to Clay* — then Up/Down to move and Enter to run. Esc closes it.

It lists every mode, the generate action, the viewport toggles and the actions for the selected
asset. A command that cannot run right now is still listed, greyed: an empty result would not tell
you the command exists or which mode owns it.

Typing also searches your assets by name, prompt or job id; picking one selects it and opens it in
the pane that made it.

## 2D and 3D

| Keys | Action |
| --- | --- |
| Ctrl+Enter | Generate / Make 3D |
| Up / Down | Previous / next asset in the library |
| Right-click a card | The same actions menu the `...` button opens |
| F | Frame the model |
| W | Toggle wireframe |
| S | Toggle turntable |
| Esc | Exit comparison / pose edit |

## Review

| Keys | Action |
| --- | --- |
| A | Accept the unit on screen |
| R | Reject — then `1`–`5` picks the reason |
| S | Skip to the next unverdicted unit |
| Left / Right | Previous / next unit |
| Esc | Cancel a pending reject |

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
| E | Extrude, with faces selected |
| F | Frame the selection |
| Delete | Delete the selection — faces in an element mode, objects in object mode |
| Ctrl+D | Duplicate the selection (object mode only) |
| Ctrl+J | Merge the selected objects into one (object mode only) |
| Ctrl+A | Select everything, in the current mode's sense |
| Ctrl+Shift+I | Invert the selection |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+N / Ctrl+O | New / open a document |
| Ctrl+E | Export to the library |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous document |
| Esc | Step back: element selection, then element mode, then object selection |

**The mouse.** Left-drag in empty space orbits, and `Alt`+left-drag always orbits whatever mode you
are in. Middle-drag pans and the wheel dollies. Right-click opens the context menu — right-drag does
nothing, so grabbing the wrong button mid-orbit costs you nothing.

In an element mode, left-click selects an element, `Shift`+click adds and `Ctrl`+click removes;
left-drag in empty space with `Q` selected sweeps a marquee.

## Inker

| Keys | Action |
| --- | --- |
| `B` | Brush |
| `E` | Eraser |
| `G` | Fill |
| `U` | Gradient |
| `R` | Blur |
| `N` | Smudge |
| `P` | Line |
| `K` | Rect |
| `J` | Ellipse |
| `M` | Marquee select |
| `S` | Ellipse select |
| `Q` | Lasso |
| `W` | Wand |
| `V` | Move |
| `I` | Pick a colour from the canvas |
| `X` | Swap the two colours |
| `[` / `]` | Brush size (with Shift, hardness) |
| Ctrl+Z / Ctrl+Y | Undo / redo (Ctrl+Shift+Z also redoes) |
| Ctrl+S / Ctrl+Shift+S | Save / save as |
| Ctrl+Shift+E | Export a flattened PNG |
| Ctrl+N | New document |
| Ctrl+O | Open a file |
| Ctrl+W | Close the current tab |
| Ctrl+A / Ctrl+D | Select all / deselect |
| Ctrl+C / Ctrl+X / Ctrl+V | Copy / cut / paste |
| Ctrl+Shift+V | Paste as a new layer |
| Ctrl+Shift+I | Invert the selection |
| Ctrl+T | Free transform (Enter applies, Esc cancels) |
| Ctrl+Tab / Ctrl+Shift+Tab | Next / previous tab |
| Ctrl+0 / Ctrl+1 | Fit to the pane / 100% |
| Space drag, middle drag | Pan (the wheel zooms) |
| Delete | Clear the selected pixels |
| Esc | Cancel a floating selection, then deselect |

Shift and Alt are modifiers rather than shortcuts: holding **Shift** while dragging a selection adds
to the current one, and **Alt** subtracts. Both are described in
[Selections and transform](07-inker.md#selections-and-transform).
