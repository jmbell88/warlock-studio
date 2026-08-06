# Keyboard shortcuts

A shorter version is in the app: press the **?** button in the top-right of the top bar. That popup
is a condensed subset — the tables below are the full list.

One rule explains an apparent overlap between the tables below. While an Inker or Clay document is
open, that mode takes every key — so in Inker `W` picks the wand rather than toggling wireframe, and
in Clay `W` is the move tool. Those viewport bindings would otherwise act on a viewport that is not
on screen.

## Everywhere

| Keys | Action |
| --- | --- |
| F1 | Switch to the Manual |
| F10 | Toggle the frame-rate readout |
| Ctrl+Enter | Generate / Make 3D |
| F | Frame the model |
| W | Toggle wireframe |
| S | Toggle turntable |
| Esc | Exit comparison / pose edit |

## Clay

| Keys | Action |
| --- | --- |
| Q / W / E / R | Select / move / rotate / scale |
| 1 / 2 / 3 / 4 | Vertex / edge / face / object mode |
| E | Extrude, with faces selected |
| F | Frame the selection |
| Delete | Delete the selection — faces in an element mode, objects in object mode |
| Ctrl+D | Duplicate the selection (object mode only) |
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
[Selections and transform](06-inker.md#selections-and-transform).
