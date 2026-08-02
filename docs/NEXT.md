# What is not built yet

Forward-planning only: everything below is *unbuilt* work, kept because the
code or CLAUDE.md refers to it. History (the original delivery-contract spec
and its executed plans) lives in git, not here.

## 1. Qualify and expose the gltfpack triangle tiers

The plumbing is done and dormant: `pipelines/optimize.py`, the
`source.glb`/`model.glb` split, `POST /api/jobs/{id}/optimize` (which reports
the rig artifacts it makes stale), `WARLOCK_GLTFPACK`, and a non-fatal doctor
check. What is missing is the binary and the qualification, and the UI
reflects that: `#g-profile` offers only "Raw reconstruction".

| Tier | Triangles | Qualified | Notes |
|---|---|---|---|
| Draft | 20,000 | no | not yet run |
| Standard | 50,000 | no | not yet run |
| Detailed | 100,000 | no | not yet run |
| Custom | 5,000–200,000 | no | gated behind the named tiers |
| Raw | — | n/a | always available, needs no binary |

To qualify a tier:

1. Vendor `vendor/gltfpack/gltfpack.exe` plus its MIT `LICENSE` and a
   `VERSION` file carrying the release tag and the exe's SHA-256.
2. Run a chest, a sword and a rock through it at that budget
   (`gltfpack -i source.glb -o out.glb -si <ratio> -noq -ke -km` — the flags
   `optimize.run` already uses).
3. Confirm the output retains positions, vertex normals, UVs, the base-color
   and metallic/roughness textures, embedded PNG data, material assignment,
   and loads as a core GLB with no required compression extensions.
4. Add the tier as an `<option>` in `static/index.html` and fill in its row
   here. The API already accepts every tier name — only the UI is gated, so a
   tier can be exercised with a direct POST before it is exposed.

There is also no UI for `POST /api/jobs/{id}/optimize` itself yet — retarget
is reachable only by hand until a control lands next to the downloads row.

## 2. Smaller open items

- **GLB structure verification + provenance**: `normalize_glb` rewrites the
  JSON chunk without verifying the one-scene/one-root/no-skin shape it
  assumes, and records its transform only in job params — not in an
  `asset.extras.warlock` block inside the file.
- **Capability-based health**: `/api/health` lists raw checks; it could also
  say which *features* (reference generation, reconstruction, BiRefNet,
  optimization, rigging) are currently available as one flag each.
- **Bulk export file selection**: `/api/export` accepts a `files` list but
  the UI always exports `model.glb` only.
- **Process-tree kill**: trellis-server shutdown is plain terminate/kill. If
  the exe ever spawns children they would leak; that is the point at which a
  psutil-based tree kill earns its dependency back.
- **Pygame-CE hand check**: the sheet sidecar's pivot and trim rectangles are
  verified with Pillow (`tests/test_sheet.py`); actually loading a sheet
  through `pygame.image.load(...).convert_alpha()` + `subsurface()` has not
  been done once against a real character sheet. Pygame is deliberately not a
  dependency — run it by hand.
