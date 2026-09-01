# Drawing

Inker is Warlock's raster editor: layers, a timeline, palettes, selections and twenty-four tools. It
is a real pixel-art and painting program, not a touch-up panel bolted to a generator, and it needs
no GPU and no model weights — everything in this chapter works on a bare install.

It has two jobs in this app. One is drawing from nothing. The other is fixing what the generators
produced, which is why references, sprite sheets and character sheets all have an **Open in Inker**
button on them.

This chapter covers still images. The timeline gets [its own chapter](06-animating.md) next.

## Getting a canvas

`Ctrl+N` starts a document, `Ctrl+O` opens one. Documents are tabs, and `Ctrl+Tab` cycles them.

The window is three columns and a strip: colour on the left, canvas in the middle, the toolbox and
the pipeline bridges on the right, and the timeline — layers down, frames across — always along the
bottom. Above the canvas is one row, the context bar: what the tool in your hand is set to, and the
symmetry toggles.

Move around the canvas with the wheel, which scrolls; hold `Ctrl` and roll to zoom, and space-drag
or middle-drag to pan. There are scrollbars along the right and bottom edges too. `Ctrl+0` fits
the canvas to the window and `Ctrl+1` snaps back to 100%. Zoom and pan are *not* tools here — they
do not take a slot in the toolbox, because a toolbox slot spent on navigation is a slot not spent on
drawing.

The chequerboard behind your artwork is transparency, not a colour.

## Tools

Twenty-four tools in twelve groups. Each group has a letter, and **pressing that letter again cycles
within the group** — so `B` is the brush, `B` again is the spray, and the first press always lands
on what that letter usually means.

| Key | Tools |
| --- | --- |
| `B` | Brush, Spray |
| `E` | Eraser |
| `G` | Fill, Gradient (`K`) |
| `R` | Blur, Smudge (`N`), Shading (`H`) |
| `L` | Line, Curve (`F`) |
| `U` | Rect, Ellipse (`J`) |
| `P` | Polyline, Polygon (`O`) |
| `M` | Marquee, Ellipse select (`S`) |
| `Q` | Lasso, Poly lasso (`D`) |
| `W` | Wand |
| `V` | Move, Pick (`I`) |
| `T` | Text, Slice (`C`), Tile stamp (`Y`) |

If you know Aseprite, most of those letters are its letters — `B`, `E`, `G`, `L`, `U`, `M`, `W`, `I`
are all where you expect. A few could not be: the gradient is on `K` because `U` is Aseprite's
rectangle and a hand trained there pressing `U` should get a rectangle.

A tool's options appear in a bar above the canvas, and they belong to *that tool*. The wand's
tolerance and the fill's tolerance are the same setting family applied to two different tools, and
each remembers its own value.

The one that matters most for pixel art is the **nib**. A pixel nib gives you hard, aliased,
one-pixel-at-a-time edges; a soft nib gives you a brush with falloff. Hardness only appears for soft
nibs, because a pixel nib has none to adjust.

## Inks

Every painting tool has an **ink**, which is what the tool does with the colour rather than where it
puts it. There are five, and they replace the per-tool ink menus you may be used to:

| Ink | What it does |
| --- | --- |
| **Alpha** | Ordinary composite-over. The default, and what you want most of the time. |
| **Simple** | Writes colour and alpha exactly as given. |
| **Copy** | Exact colour, ignoring stroke opacity *and* the dab's antialiasing. This is the pixel-art ink. |
| **Lock alpha** | Paints only where there are already pixels. Recolouring without changing the silhouette. |
| **Shading** | Walks the active ramp instead of painting a colour — one step lighter or darker per stroke. |

Copy is the one to know about. If you are drawing pixel art and finding half-transparent fringes on
your strokes, you want Copy, not a smaller brush.

## Colour

A document is in one of three colour modes, and the switch is at the top of the colour panel.

**RGB** is what you expect. **Grayscale** is the same thing constrained to greys. **Indexed** is
genuinely different: the document stores a palette index per pixel rather than a colour, so
reordering the palette recolours the whole image and nothing can drift off-palette. If you are
making art for a system with a fixed palette, that is the mode that enforces it.

There is a fourth state that is not a mode: **palette-constrained RGB** — full RGBA storage with a
palette applied as a constraint on what you can write. It is the gentler option when you want
discipline without committing to index storage.

Picking colours: hold `Alt` with any paint tool for a temporary eyedropper, or press `I` for the
tool. `Alt` with the right mouse button picks into the background colour.

Hold `Ctrl` and click to **select the layer under the cursor** — the fastest way to find which of
twenty layers a particular pixel belongs to.

## Layers

The layers panel and the timeline's rows are one thing drawn once, which is why a layer row here
looks like a track row there.

Each layer has visibility, opacity, a blend mode — nineteen of them — and two locks. The **content
lock** stops edits to the pixels; **alpha lock** confines them to pixels that already exist. Alpha
lock is both a layer flag and an ink, and either one turns it on.

Dragging across the eye icons toggles a run of layers in one gesture, and that whole gesture is one
undo step rather than one per layer.

Layers can be grouped, and a group is a span over the flat stack rather than a real tree — the stack
stays authoritative, which is what keeps ordering unambiguous.

Two layer *types* are worth knowing. A **background** layer is opaque and sits at the bottom. A
**reference** layer is one you can see and cannot paint on — for tracing over something.

## Selections

Five tools make selections: rectangular marquee, elliptical, freehand lasso, polygonal lasso, and
the magic wand. Beyond them the Select menu offers the things you actually reach for — Select All
(`Ctrl+A`), Deselect (`Ctrl+D`), Reselect (`Ctrl+Shift+D`), Inverse (`Ctrl+Shift+I`), this layer's
pixels, a colour range, and Grow, Shrink, Feather and Border for adjusting one you have.

`Ctrl+J` copies the selection to a new layer; `Ctrl+Shift+J` moves it to one.

Moving a selection lifts the pixels into a **floating** state, which you then commit or cancel. That
float outlives a lock toggle, so locking a layer mid-move does not strand your pixels.

One small kindness: an action that changes nothing does not push an undo step. Select All twice,
feather by zero, or redraw the same marquee, and your undo history stays where it was.

## Undo

`Ctrl+Z` and `Ctrl+Y`, and one property worth stating because it is unusual: **undo is addressed by
layer identity, not by position in the stack.** Reorder your layers and every earlier undo step
still lands on the layer it was about. That is not something most editors promise.

## Symmetry

Four toggles at the right-hand end of the context bar: **H** (left/right), **V** (top/bottom), and
`\` and `/`, the two 45-degree mirrors. They **compose** — switch on any combination and each one
reflects everything the others already produced — and **Reset** beside them switches the lot off.
**Radial** and the axis the mirrors reflect about are in the canvas popover off the toolbox, because
each needs a number. Strokes mirror as you draw them, and every paint tool inherits it.

## Saving and exporting

The native format is **ORA** — an open, zipped, layered format that other editors read. `Ctrl+S`
saves it.

Beyond that: PNG (`Ctrl+Shift+E`) flattens; there is an `.aseprite` writer; and sheets and GIFs come
from the timeline, which is the next chapter.

If you use Aseprite, read [the divergence list](28-inker.md) before assuming a habit transfers.
Inker is deliberately close but not identical, and the differences are documented individually
rather than discovered.

## Try it

1. `Ctrl+N` for a 32×32 document.
2. Switch the colour mode to Indexed and build a small palette — six or eight colours.
3. Set the ink to **Copy** and draw a simple object with the brush on a pixel nib.
4. Add a layer, set it to **Multiply** at about 60% opacity, and paint shadow on it.
5. Toggle the new layer's visibility, then reorder the two layers, then press `Ctrl+Z` a few times —
   and watch the undos land on the right layers despite the reorder.
6. `Ctrl+S` to save the ORA.

## What to read next

[Animating](06-animating.md) — frames, tags, onion skinning, and the difference between copying a
frame and linking one.
