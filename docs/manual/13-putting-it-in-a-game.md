# Putting it in a game

The last tutorial chapter, and the one about leaving. Everything Warlock makes is meant to be
imported somewhere else, and this covers what the formats are, what has already been done to make
them import cleanly, and — honestly — which of the interoperability claims have actually been
checked.

## Exports are derived, not stored

Almost nothing in a job's directory is generated up front. Ask for an FBX and it is produced then and
cached; never ask and it never costs you anything. So the list below is what is *available*, not what
is sitting on your disk.

**From a mesh:**

| File | What it is |
| --- | --- |
| `model.glb` | The one to use. Optimised, centred and grounded. |
| `source.glb` | The raw reconstruction, kept as evidence. |
| STL | Geometry only. For printing rather than for engines. |
| OBJ | A zip, since OBJ is never one file. |
| FBX | For pipelines that insist on it. |
| `collision.glb` | A convex hull. |
| `textures.zip` | The maps on their own. |
| `rig.glb` | Once rigged. |
| A baked GLB per saved pose | Posed geometry, ready to use. |

**From a reference:** a 512 px transparent `icon.png`; a trimmed `sprite.png` with its pivot
recorded; `pixel_32/64/128.png` reduced to a palette; and a `manifest.json` carrying every size,
trim box, pivot and the recipe that produced them. Tiles additionally get an estimated PBR material
set.

**From the workspaces:** sprite sheets as PNG plus a JSON sidecar; Inker's ORA, PNG, GIF and sheets;
Plotter's TMX and TMJ; Packwright's atlases and sidecars.

**In bulk:** zip a named artifact across many jobs at once, or set `WARLOCK_EXPORT_DIR` and have
exports mirrored into it — a game project's `assets/` folder, for instance.

## Two things already done for you

**Grounding.** Every mesh has its pivot put on the ground and its X and Z centred, always, whether or
not you asked for a particular size. A model whose origin sits in the middle of its bounding volume
is a manual fix-up on every single import, forever, so the app does it once.

**`model.glb`, not `source.glb`.** Everything downstream uses the derived file. Take that one. The
raw reconstruction is kept so that every derived file can be rebuilt from it and so that you can see
what the engine actually produced — it is not the one to ship.

## 3D engines

glTF is the target format and every major engine reads it.

**Godot** imports `.glb` directly — drop it in the project and it is an importable scene. This is the
smoothest path, and `WARLOCK_EXPORT_DIR` pointed at a Godot project makes it smoother still.

**Unity** and **Unreal** both read glTF, Unity via a package and Unreal natively. FBX is there if
your pipeline is built around it.

Rigged exports carry their skeleton and skin weights. A pose bakes to its own GLB rather than
travelling as an animation track — for a still pose that is what you want, and it sidesteps a class
of exporter problem where a posed model arrives at rest with the pose demoted to an animation nobody
plays.

## 2D and sprite sheets

A sheet is a PNG plus a JSON sidecar, and the sidecar is deliberately engine-neutral: cell
rectangles, tags, durations and pivots, in plain JSON, for you to read with whatever you already have.

Packwright's sidecar is TexturePacker's format instead, which a great many 2D toolchains already
understand.

## Tiled and Aseprite: read this before relying on it

Warlock reads and writes Tiled's `.tmx`/`.tmj`/`.tsx` and Aseprite's `.aseprite`, and both are
modelled carefully — the divergences are enumerated individually in the reference chapters rather
than discovered by accident.

There is one caveat, and it is important enough to state plainly rather than bury.

**Every test fixture for both formats was written by Warlock itself.** A green test proves that
Warlock's reader and Warlock's writer agree with each other. It does not prove that either real
application agrees with them.

For Tiled that has now been checked once, by hand, in both directions: on 2026-08-29 a Plotter map
export was opened and worked on in Tiled 1.12.x, and a map Tiled 1.12.2 itself wrote was read back
in Plotter. Both were plain orthogonal maps, so what is confirmed is the map header, the external
tileset reference and the layer data — not flipped tiles, objects or properties. For Aseprite,
nothing has been checked at all.

So: the exports are believed correct, and past plain tile layers that belief has not been checked
against the applications themselves. Try one before building a workflow on it, and expect it to
work — but check.

One specific thing to know if the file is going to Tiled: several constructs are Warlock's own
extensions rather than Tiled features — oblique projection, per-layer blend modes, the capsule shape,
per-object opacity, list properties. Round trips fine through Warlock; invisible to Tiled.

For Aseprite, the notable losses on write are per-frame palettes and colour profiles. Cel opacity,
a cel's z-index and the colours and notes on layers, cels and tags are written and read back; what is
still dropped from *user data* is a note on a slice, on a tileset or on an individual tile, and
Aseprite's custom properties tree. Each loss is reported with a warning rather than dropped
silently.

## A whole pipeline

To make the shape of it concrete, here is one path end to end. Every step has its own chapter:

1. Prompt a reference in Create, approve it, reconstruct a mesh.
2. Judge it in Review. Remesh if the picture was fine and the mesh was not.
3. Rig it, and pose it in Poser.
4. Render a character sheet in Troupe.
5. Clean the sheet up by hand in Inker.
6. Pack it with other sprites in Packwright.
7. Build the level it lives in with Plotter.
8. Export into your engine.

Not every asset needs all eight, and most need two or three. But every one of those steps hands its
output to the next as an ordinary library asset, which is the thing that makes the app one app rather
than seven.

## What to read next

One tutorial left, and it is the only one that makes a sound:
[Making a soundtrack](14-making-a-soundtrack.md) — Sirens, the tracker.

After that the reference chapters go deeper on everything touched here —
[Overview](20-overview.md) is the front door to them, and each workspace has its own.

If something is not behaving, [Troubleshooting](41-troubleshooting.md) is organised by symptom, and
`uv run warlock doctor` answers the same questions from a terminal.
