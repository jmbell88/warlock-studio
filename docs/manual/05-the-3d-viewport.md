# The 3D viewport

The middle column of Create is the viewport: a live, interactive view of whatever is selected. It is
the same view at every stage — Mesh, Rig, Pose and Export all look at one scene — and it is the same
view Review judges in. Nothing in it changes your asset. It is a camera and a set of ways to look.

Before there is a mesh it shows the reference image instead, full size, and the camera controls do
not apply. That is the one case where the toolbar changes shape rather than the view.

## Moving the camera

The camera orbits a point, and every control is relative to that point rather than to the world:

| Input | What it does |
| --- | --- |
| Left-drag | Orbit around the framed point |
| Middle-drag | Pan — the framed point moves with the pointer |
| Wheel | Dolly in and out |
| `F` | Frame the model again, undoing a pan and a dolly at once |

Panning moves the point the orbit is about, which is why `F` is the reset rather than a zoom
control: after a pan the camera is circling somewhere that is no longer the asset, and no amount of
orbiting brings it back.

The dolly has limits that scale with the asset. A 3 cm gem and a 100 m building are both framed to
fill the view, so a fixed near and far plane would clip one or the other; the limits are computed
from the model's own size when it is framed.

## The toolbar

Along the top of the viewport, over the view rather than beside it:

- **Frame** (`F`) — put the camera back where framing put it.
- **Wireframe** (`W`) — draw the triangles over the shaded surface. This is the fastest read on
  whether a triangle budget did what you asked, and it is the view the
  [mesh report](04-generating-meshes.md#mesh-audit-and-mesh-report) numbers describe.
- **Turntable** (`S`) — rotate the asset slowly and continuously. It runs at one orbit per thirty
  seconds, which is slow enough to look at and fast enough not to need waiting for.
- **Screenshot** — save exactly what is on screen as a PNG, at the viewport's own resolution. It
  asks where to put it. Disabled when there is no model.
- **Zoom in / out** — the wheel already does this; the buttons exist so the control is findable.
- **Exit comparison** — only while two assets are being compared.
- **Clear** — only when there is something in the view. It empties the viewport; reselecting the
  asset brings it back, so nothing is lost.

The toolbar **wraps** rather than running off the edge. It carries up to ten controls over a column
whose width is whatever the side panes have left, so widening the inspector moves the last few
controls onto a second row instead of clipping them away.

## What else is drawn

The scene is not only the mesh. Depending on what is selected and which stage you are on it also
carries:

- **The ground plane and grid**, which is what makes the asset's scale legible at all — a mesh
  floating in an empty view has no size.
- **The skeleton**, once an asset is rigged: bone lines through the mesh, and a marker at each joint
  you can click. See [Rigging and posing](06-rigging-and-posing.md).
- **The comparison split**, when two assets are being compared from the library.

The lighting is a generated sky/horizon/ground gradient rather than an imported HDRI, because
nothing is fetched at runtime and no HDR file is vendored. Its job is not to be looked at — the
background is a flat colour — but to light the surface from more than one direction so that a
normal map reads.

## When the view is empty

An empty viewport says what to do rather than sitting blank: it names the thing to select, or the
shortcut that would make one. That placeholder is the only place in the app where `Ctrl+N` and
`Ctrl+O` appear on screen.

## Performance

The viewport draws every frame the app draws, so it is the first thing to feel a slow machine.
[Troubleshooting](22-troubleshooting.md#the-window-feels-sluggish) covers what to do about a stuttering
turntable; the short version is that the frame loop never waits for a job, so a stutter is the
renderer and not the queue.
