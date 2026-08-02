# Make Warlock Studio Produce Trustworthy Godot and Pygame-CE Assets

## Summary

Warlock Studio will support two explicit delivery contracts:

1. **Godot 4:** validated, optimized GLB static props.
2. **Pygame-CE:** transparent PNG icons or directional sprite sheets rendered from the final approved model, accompanied by JSON frame metadata.

Pygame-CE will not be treated as a GLB runtime. Its outputs will load as alpha-preserving `Surface` objects and can be divided into frames with `subsurface()`, matching the official [Pygame-CE image](https://pyga.me/docs/ref/image.html) and [Surface](https://pyga.me/docs/ref/surface.html) APIs.

The previously verified corrections remain:

- The measured desktop result was 290,590 triangles and 22.4 MB.
- Current GLBs contain vertex normals, base color, and combined metallic/roughness—not a normal texture.
- Scale is applied, but the exported pivot is not grounded.
- The visible-hole audit does not prove watertightness.
- The shared SQLite connection is not serialized by `asyncio.to_thread`.

## Implementation Changes

### 1. Persistence and API correctness

- Protect all `JobStore` connection operations with an internal `threading.RLock`.
- Remove documentation claiming `to_thread` serializes DB access.
- Add migrations for:
  - `stage TEXT NOT NULL DEFAULT 'model'`
  - `parent_id TEXT NULL`
- Existing jobs migrate to `stage='model'`.
- Split generation randomness into `reference_seed` and `mesh_seed`; continue reading legacy `seed`.
- Expose structured `metrics`, `quality_status`, `quality_reasons`, artifacts, and lineage in API responses.

### 2. Approve-reference-first workflow

- Text submissions from the UI stop after producing `input.png`.
- A completed reference offers:
  - **Generate 3D**
  - **Try another reference**
  - **Edit prompt/settings**
- `POST /api/jobs/{id}/model` creates a child model job from the approved reference.
- Extend `POST /api/jobs` with `output=reference|model`:
  - Preserve `model` as the backward-compatible API default.
  - Have the text UI explicitly request `reference`.
- Preserve existing reroll/remesh routes as compatibility aliases.
- Add a server-owned prompt-preview endpoint so the UI shows the actual composed prompt.
- Preview uploads and warn about cropping, multiple subjects, small dimensions, occlusion, and complex backgrounds.
- Forward TRELLIS's `bg_removal` option and expose Auto, BiRefNet, and Threshold.

### 3. Qualify and integrate mesh optimization

- Vendor a pinned native `gltfpack.exe`, its MIT license, version/checksum metadata, and `WARLOCK_GLTFPACK` override.
- Qualify simplification on at least three representative TRELLIS props:
  - Box-like prop such as a chest
  - Thin hard-surface prop such as a sword
  - Rounded or irregular prop such as a potion or rock
- Test candidate targets of 20k, 50k, and 100k triangles.
- Each qualified output must retain:
  - Positions, vertex normals, and UVs
  - Base-color texture
  - Combined metallic/roughness texture
  - Embedded PNG data and material assignment
  - A directly loadable core GLB without required compression extensions
- Use:

  `gltfpack -i source.glb -o candidate.glb -si <target/source> -noq -ke -km`

- Do not expose a named tier until its visual and material-preservation checks pass.
- Candidate profiles:
  - Draft: 20,000 triangles
  - Standard: 50,000 triangles
  - Detailed: 100,000 triangles
  - Raw reconstruction
  - Custom: 5,000–200,000
- Preserve the TRELLIS response as `source.glb`.
- Produce `model.glb` through a validated temporary file and atomic replacement.
- Add `POST /api/jobs/{id}/optimize` to rebuild from `source.glb` without rerunning TRELLIS.
- Display requested and achieved triangle counts instead of claiming exact enforcement.

#### Qualification status

The plumbing is built (`pipelines/optimize.py`, the `source.glb`/`model.glb`
split, `POST /api/jobs/{id}/optimize`, `WARLOCK_GLTFPACK`, a non-fatal doctor
check). What is *not* done is the qualification itself, and the UI reflects
that: `#g-profile` offers only "Raw reconstruction".

| Tier | Triangles | Qualified | Notes |
|---|---|---|---|
| Draft | 20,000 | no | not yet run |
| Standard | 50,000 | no | not yet run |
| Detailed | 100,000 | no | not yet run |
| Custom | 5,000–200,000 | no | gated behind the named tiers |
| Raw | — | n/a | always available, needs no binary |

To qualify a tier: vendor `vendor/gltfpack/gltfpack.exe` (plus `LICENSE` and a
`VERSION` file carrying the release tag and the exe's SHA-256), run a chest, a
sword and a rock through it at that budget, confirm every item in the retention
list above, then add the tier as an `<option>` in `static/index.html` and fill
in its row here. The API already accepts every tier name — only the UI is
gated, so a tier can be exercised with a direct POST before it is exposed.

### 4. Normalize scale and pivot

- Optimize before applying transforms.
- Verify the expected TRELLIS structure before rewriting GLB JSON:
  - One active scene
  - One root hierarchy
  - One static mesh
  - No skin or animation
- Insert transforms below the scene root so GLB, OBJ, and STL consumers retain them.
- Apply:
  - Requested longest-axis size in metres
  - X/Z centering
  - Minimum Y at zero
- If the structure differs, preserve the model and mark it `review` instead of rewriting it unsafely.
- Store transform, size, profile, and lineage metadata in `asset.extras.warlock`.

### 5. Replace the quality badge with a readiness report

Audit the final optimized model for:

- Parse success and finite attributes
- Triangles and vertices
- Degenerate triangles
- Components
- Boundary and non-manifold edges
- True watertightness
- Visible silhouette holes
- Requested and achieved dimensions
- Bottom-center pivot
- Vertex normals and UVs
- Base-color and metallic/roughness textures
- Optional normal texture
- glTF extensions
- Source and final file sizes

Classify results:

- `ready`: valid, within budget/scale/pivot tolerances, and required material data present.
- `review`: downloadable with explicit topology, material, budget, or visual warnings.
- `invalid`: corrupt or non-finite.

Rename the existing “watertight” silhouette badge to “No visible holes.” Only topology analysis may use “watertight.”

## Pygame-CE Sprite Export

### Export behavior

- Render sprite outputs client-side using the already-vendored three.js viewer and the final `model.glb`.
- Use a dedicated transparent WebGL renderer:
  - Alpha-enabled canvas
  - Transparent clear color
  - Straight-alpha PNG output
  - sRGB output
  - No grid, ground plane, or baked drop shadow
- Use an orthographic camera and one shared framing calculation across every direction so the sprite does not resize or jitter between frames.
- Derive framing from the model's full bounding sphere and grounded pivot, with consistent padding.

### Presets

- **Pygame icon**
  - One 512×512 transparent PNG
  - 45° azimuth
  - 35.264° elevation
- **Four directions**
  - Four 256×256 frames
  - Azimuths 0°, 90°, 180°, and 270°
  - One 1024×256 sheet
- **Eight-direction isometric**
  - Default Pygame export
  - Eight 256×256 frames
  - Azimuths every 45°
  - 35.264° elevation
  - Four columns × two rows, producing a 1024×512 sheet
- **Custom turntable**
  - 4, 8, or 16 directions
  - 128, 256, or 512 pixel square frames
  - Configurable elevation

### Metadata contract

Download a PNG plus a JSON file with this stable structure:

```json
{
  "schema": "warlock-pygame-sprites/v1",
  "image": "asset_sprites.png",
  "frame_width": 256,
  "frame_height": 256,
  "columns": 4,
  "rows": 2,
  "alpha": "straight",
  "frames": [
    {
      "index": 0,
      "x": 0,
      "y": 0,
      "width": 256,
      "height": 256,
      "azimuth_degrees": 0,
      "elevation_degrees": 35.264,
      "pivot_x": 128,
      "pivot_y": 230
    }
  ]
}
```

- `pivot_x` and `pivot_y` identify the projected bottom-center model origin for consistent placement in Pygame.
- Frames are stored left-to-right, top-to-bottom in increasing azimuth order.
- Transparent pixels remain RGBA rather than using a color key.
- Provide a documentation example using:

  `pygame.image.load(path).convert_alpha()`

  followed by `Surface.subsurface()` with the metadata rectangles.

### Pygame validation

- Confirm exported PNGs load with per-pixel alpha in the supported Pygame-CE version.
- Verify every metadata rectangle lies inside the sheet.
- Verify all directions use identical frame dimensions and camera scale.
- Verify the grounded pivot projects to a stable pixel coordinate across directions.
- Check for opaque backgrounds, clipped geometry, empty frames, and edge contact.
- Treat sprite-export warnings separately from GLB readiness; a good GLB can still require reframing before becoming a good sprite sheet.

#### Measured, 2026-08-02

The sidecar now carries `pivot_x`/`pivot_y` and a per-cell `trim` (the alpha
bounding box), both additive at `SHEET_VERSION = 1`. Three of the checks above
are covered by
`tests/test_sheet.py::test_the_reported_pivot_sits_at_the_subjects_feet_in_every_direction`,
which renders a real lopsided subject at eight directions and asserts against
each cell's own alpha bbox — the same rectangle a `subsurface()` blit honours:

- **Stable across directions: yes.** The pivot is projected once from yaw 0 and
  is byte-identical in all eight cells, because the ortho camera is framed once
  from the rest bbox and only spins.
- **Grounded and centred: yes.** At elevation 0 the pivot lands within 2 px of
  the bottom of every cell's silhouette and within 1.5 px of the cell's
  horizontal centre.
- **Rectangles inside the sheet: yes**, by construction — `measure_trim` reads
  `Image.getbbox()` on the already-decoded frame during `pack`.

Still unmeasured: loading the PNG through Pygame-CE itself. Pygame is not a
dependency of this project and adding one to assert `convert_alpha()` would be
testing Pygame, not Warlock — the pixels and the rectangles are verified above
with Pillow. Run it by hand once against a real character sheet before calling
the Pygame path done.

## Additional Improvements

- Correct README claims about generated PBR data.
- Distinguish TRELLIS's 512 volumetric texture decode from its measured 2048×2048 exported atlases.
- Hide characters and broad environments from the primary static-prop workflow; retain old values for compatibility.
- Replace arbitrary text-to-image directory swapping with explicit SDXL-Turbo and optional Flux Schnell adapters.
- Add capability-based health reporting for:
  - Reference generation
  - TRELLIS reconstruction
  - BiRefNet
  - GLB optimization
  - Browser sprite export
- Limit uploads to 20 MB and 16 megapixels.
- Reject decompression bombs and out-of-range seeds; limit prompts to 1,000 characters.
- Clean up input directories when DB insertion fails.
- Keep GLB primary for Godot; label OBJ and textureless STL as secondary interchange formats.

## Test and Acceptance Plan

- Preserve the current 217-test and clean-lint baseline.
- Add backend tests for:
  - Concurrent DB access
  - Migration and legacy parameters
  - Reference promotion and independent seeds
  - Background-removal forwarding
  - Optimizer command, ratio, timeout, validation, and atomic replacement
  - Material/UV preservation
  - Transform invariants and safe fallback
  - Topology versus silhouette metrics
  - Upload and input limits
  - Backward-compatible endpoints
- Add browser-level tests for:
  - Deterministic camera directions
  - Orthographic framing
  - Transparent canvas output
  - Sprite-sheet packing
  - Metadata coordinates and pivots
  - Icon, four-direction, eight-direction, and custom presets
- Godot acceptance:
  - Import qualified GLBs without unsupported required extensions.
  - Longest dimension within 1% of target.
  - Ground pivot within 1 mm or 0.1% of height.
  - Achieved triangles no more than 5% above target unless reported.
  - Base-color and metallic/roughness render correctly.
- Pygame-CE acceptance:
  - Load PNG with `convert_alpha()`.
  - Extract every frame using the emitted metadata.
  - Blit frames without background artifacts.
  - Preserve consistent placement while switching directions.
- Product benchmark:
  - At least 12 deterministic static-prop prompts.
  - At least 10/12 reach an approved reference within three attempts.
  - At least 9/12 produce usable GLBs with minor or no cleanup.
  - Each usable model must also produce an unclipped eight-direction Pygame sheet.

## Assumptions

- Godot consumes optimized GLB assets.
- Pygame-CE consumes rendered PNG sprites and JSON metadata, not GLB at runtime.
- Outputs are static props; sprite sheets represent viewpoints, not skeletal animation frames.
- Rigging, animation, collision generation, and automatic LOD packages remain out of scope.
- Current uncommitted changes remain user-owned baseline work.
- Named triangle tiers remain hidden until qualification passes.
