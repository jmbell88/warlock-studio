# Plan C — Character Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Warlock from "it can rig a biped" into a creature-and-animation pipeline: more skeletons, shipped pose libraries, mirrored posing, correctable joints, and sprite sheets that carry animation frames and pivots.

**Architecture:** Everything here leans on decisions already made. A skeleton is a JSON file, so new creatures cost no code. A pose is a bone→local-quaternion map, which makes a pose portable across any rig from the same template — that portability is what makes shipped pose libraries and mirroring possible without a new format. `sheet.plan()` is pure host-side code with a `yaws` kwarg no caller passes and a `Cell.frame` field documented as the animation seam; both are opened up here rather than replaced.

**Tech Stack:** JSON templates, `pipelines/sheet.py` (pure), Blender out-of-process, three.js r170 (vendored), plain-JS frontend.

## Global Constraints

See `2026-08-02-warlock-review-index.md` § Global Constraints. Every task's
requirements implicitly include that section. Load-bearing here in particular:

- `rigging.py` must stay importable with **no bpy anywhere**; only
  `pipelines/blender_worker.py` may `import bpy`.
- The pose contract: **glTF node-local quaternions, stored XYZW** (three.js
  `Quaternion.toArray()` order); the worker converts to WXYZ. `bone_heuristic="BLENDER"`
  on import and `export_rest_position_armature=False` on export must not change.
- The sheet grid is decided in `pipelines/sheet.py`, never in Blender. The camera is
  framed **once** from the rest bbox. Cells arrive **grouped by row** so the worker
  re-poses once per pose. Column 0 is the front view (yaw 0 looks along +Y; templates
  put forward at −Y).
- `tests/test_rigging.py::test_a_posed_glb_carries_back_exactly_the_rotations_it_was_given`
  pins the pose identity by reading the exported GLB's JSON chunk directly. It must
  stay green.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/warlock/templates/*.json` | normalized landmark skeletons | **Create** 5 new |
| `src/warlock/templates/poses/<template>.json` | shipped pose libraries | **Create** |
| `src/warlock/rigging.py` | template registry, pose storage, worker specs | Modify |
| `src/warlock/doctor.py` | `_probe_blender` hardcodes two template names | Modify |
| `src/warlock/pipelines/sheet.py` | grid, packing, sidecar | Modify: yaws, frames, pivot, trim |
| `src/warlock/pipelines/blender_worker.py` | rig/pose/sheet ops | Modify: skin-only op, pivot projection |
| `src/warlock/app.py` | rig/pose/sheet routes | Modify |
| `src/warlock/static/app.js` | pose editor, sheet setup | Modify |

---

### Task 1: More skeleton templates (review item #14)

Exactly two templates exist (`humanoid.json`, `quadruped.json`, 19 bones each).
A skeleton is a JSON file by design, so this is the highest coverage-per-line item
in the whole review. `doctor._probe_blender` hardcodes the current two names and
must stop doing so.

**Files:**
- Create: `src/warlock/templates/bird.json`, `fish.json`, `insect.json`,
  `serpent.json`, `biped_tail.json`
- Modify: `src/warlock/doctor.py:66-73`
- Test: `tests/test_rigging.py`, `tests/test_doctor.py`

**Interfaces:**
- Consumes: the existing `rigging._parse_template` contract — each file needs
  `key`, `label`, `root`, `bones` (each with `name`, `parent`, `head`, `tail`), and
  optional `mirror_pairs`.
- Produces: five new keys in `rigging.templates()`, automatically surfaced by
  `/api/rig/templates` and the UI select with no further change.
- Consumed by: Task 2 (pose libraries are keyed by template).

**Coordinate contract** (from `humanoid.json`'s own comment, repeated here because
the implementer may read this task in isolation): normalized positions in a unit
bounding box, **Blender axes** — +X is the subject's left, −Y is forward, +Z is up;
x and y span −0.5..0.5 about the bbox centre, z spans 0 (floor) to 1 (top).
`rigging.fit_template` scales these onto the measured mesh bbox.

- [ ] **Step 1: Write the failing test**

In `tests/test_rigging.py`:

```python
EXPECTED_TEMPLATES = {
    "humanoid", "quadruped", "bird", "fish", "insect", "serpent", "biped_tail"
}


def test_every_shipped_template_parses():
    from warlock import rigging

    assert set(rigging.templates()) == EXPECTED_TEMPLATES


@pytest.mark.parametrize("key", sorted(EXPECTED_TEMPLATES))
def test_template_is_well_formed(key):
    """Parsing already rejects unknown parents and bad roots; this pins the
    things _parse_template does not check and that a hand-authored file gets
    wrong: bones inside the unit box, and mirror pairs naming real bones."""
    from warlock import rigging

    template = rigging.get_template(key)
    names = {b["name"] for b in template.bones}
    for bone in template.bones:
        for end in ("head", "tail"):
            x, y, z = bone[end]
            assert -0.5 <= x <= 0.5, f"{key}/{bone['name']} {end} x out of box"
            assert -0.5 <= y <= 0.5, f"{key}/{bone['name']} {end} y out of box"
            assert 0.0 <= z <= 1.0, f"{key}/{bone['name']} {end} z out of box"
    for a, b in template.mirror_pairs:
        assert a in names and b in names, f"{key} mirrors a bone it does not have"


@pytest.mark.parametrize("key", sorted(EXPECTED_TEMPLATES))
def test_fitting_produces_no_zero_length_bones(key):
    from warlock import rigging

    fitted = rigging.fit_template(rigging.get_template(key), [-1, -1, 0], [1, 1, 2])
    for bone in fitted:
        assert rigging._distance(bone["head"], bone["tail"]) > 0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_rigging.py -k template -v`
Expected: FAIL — the set comparison shows five missing keys

- [ ] **Step 3: Write `bird.json`**

Create `src/warlock/templates/bird.json`. Forward is −Y, so the beak is at
negative Y and the tail at positive Y; wings spread along ±X.

```json
{
  "key": "bird",
  "label": "Bird (winged biped)",
  "root": "hips",
  "comment": "Normalized landmark positions in a unit bounding box, Blender axes: +X is the subject's left, -Y is forward, +Z is up. x/y span -0.5..0.5, z spans 0 (feet) to 1 (top). rigging.fit_template scales these onto the mesh bbox.",
  "bones": [
    {"name": "hips",         "parent": null,          "head": [0.00,  0.05, 0.48], "tail": [0.00, -0.02, 0.55]},
    {"name": "spine",        "parent": "hips",        "head": [0.00, -0.02, 0.55], "tail": [0.00, -0.10, 0.64]},
    {"name": "chest",        "parent": "spine",       "head": [0.00, -0.10, 0.64], "tail": [0.00, -0.17, 0.72]},
    {"name": "neck",         "parent": "chest",       "head": [0.00, -0.17, 0.72], "tail": [0.00, -0.24, 0.84]},
    {"name": "head",         "parent": "neck",        "head": [0.00, -0.24, 0.84], "tail": [0.00, -0.34, 0.92]},
    {"name": "beak",         "parent": "head",        "head": [0.00, -0.34, 0.92], "tail": [0.00, -0.46, 0.90]},

    {"name": "tail",         "parent": "hips",        "head": [0.00,  0.05, 0.48], "tail": [0.00,  0.30, 0.44]},
    {"name": "tail_tip",     "parent": "tail",        "head": [0.00,  0.30, 0.44], "tail": [0.00,  0.48, 0.40]},

    {"name": "wing_base.L",  "parent": "chest",       "head": [0.04, -0.08, 0.68], "tail": [0.18, -0.04, 0.70]},
    {"name": "wing_mid.L",   "parent": "wing_base.L", "head": [0.18, -0.04, 0.70], "tail": [0.34,  0.00, 0.68]},
    {"name": "wing_tip.L",   "parent": "wing_mid.L",  "head": [0.34,  0.00, 0.68], "tail": [0.48,  0.06, 0.64]},

    {"name": "wing_base.R",  "parent": "chest",       "head": [-0.04, -0.08, 0.68], "tail": [-0.18, -0.04, 0.70]},
    {"name": "wing_mid.R",   "parent": "wing_base.R", "head": [-0.18, -0.04, 0.70], "tail": [-0.34,  0.00, 0.68]},
    {"name": "wing_tip.R",   "parent": "wing_mid.R",  "head": [-0.34,  0.00, 0.68], "tail": [-0.48,  0.06, 0.64]},

    {"name": "thigh.L",      "parent": "hips",        "head": [0.07,  0.02, 0.46], "tail": [0.08,  0.00, 0.28]},
    {"name": "shin.L",       "parent": "thigh.L",     "head": [0.08,  0.00, 0.28], "tail": [0.08, -0.02, 0.10]},
    {"name": "foot.L",       "parent": "shin.L",      "head": [0.08, -0.02, 0.10], "tail": [0.08, -0.16, 0.00]},

    {"name": "thigh.R",      "parent": "hips",        "head": [-0.07,  0.02, 0.46], "tail": [-0.08,  0.00, 0.28]},
    {"name": "shin.R",       "parent": "thigh.R",     "head": [-0.08,  0.00, 0.28], "tail": [-0.08, -0.02, 0.10]},
    {"name": "foot.R",       "parent": "shin.R",      "head": [-0.08, -0.02, 0.10], "tail": [-0.08, -0.16, 0.00]}
  ],
  "mirror_pairs": [
    ["wing_base.L", "wing_base.R"],
    ["wing_mid.L", "wing_mid.R"],
    ["wing_tip.L", "wing_tip.R"],
    ["thigh.L", "thigh.R"],
    ["shin.L", "shin.R"],
    ["foot.L", "foot.R"]
  ]
}
```

- [ ] **Step 4: Write `fish.json`**

A fish's bbox is long in Y and shallow in Z, so its spine runs along Y at
mid-height and `fit_template`'s bbox-proportional scaling does the right thing.

```json
{
  "key": "fish",
  "label": "Fish (swimmer)",
  "root": "spine_01",
  "comment": "Normalized landmark positions in a unit bounding box, Blender axes: +X is the subject's left, -Y is forward, +Z is up. x/y span -0.5..0.5, z spans 0 to 1.",
  "bones": [
    {"name": "spine_01",  "parent": null,        "head": [0.00, -0.30, 0.50], "tail": [0.00, -0.14, 0.50]},
    {"name": "spine_02",  "parent": "spine_01",  "head": [0.00, -0.14, 0.50], "tail": [0.00,  0.02, 0.50]},
    {"name": "spine_03",  "parent": "spine_02",  "head": [0.00,  0.02, 0.50], "tail": [0.00,  0.18, 0.50]},
    {"name": "spine_04",  "parent": "spine_03",  "head": [0.00,  0.18, 0.50], "tail": [0.00,  0.34, 0.50]},
    {"name": "tail_fin",  "parent": "spine_04",  "head": [0.00,  0.34, 0.50], "tail": [0.00,  0.48, 0.50]},

    {"name": "head",      "parent": "spine_01",  "head": [0.00, -0.30, 0.50], "tail": [0.00, -0.46, 0.50]},
    {"name": "jaw",       "parent": "head",      "head": [0.00, -0.42, 0.46], "tail": [0.00, -0.48, 0.42]},

    {"name": "dorsal",    "parent": "spine_02",  "head": [0.00,  0.00, 0.62], "tail": [0.00,  0.04, 0.84]},
    {"name": "ventral",   "parent": "spine_03",  "head": [0.00,  0.10, 0.38], "tail": [0.00,  0.14, 0.18]},

    {"name": "pectoral.L", "parent": "spine_01", "head": [0.05, -0.16, 0.44], "tail": [0.26, -0.06, 0.38]},
    {"name": "pelvic.L",   "parent": "spine_02", "head": [0.04,  0.02, 0.40], "tail": [0.20,  0.10, 0.32]},

    {"name": "pectoral.R", "parent": "spine_01", "head": [-0.05, -0.16, 0.44], "tail": [-0.26, -0.06, 0.38]},
    {"name": "pelvic.R",   "parent": "spine_02", "head": [-0.04,  0.02, 0.40], "tail": [-0.20,  0.10, 0.32]}
  ],
  "mirror_pairs": [
    ["pectoral.L", "pectoral.R"],
    ["pelvic.L", "pelvic.R"]
  ]
}
```

- [ ] **Step 5: Write `insect.json`**

Six legs in three pairs, plus a three-segment body. Legs splay outward and down.

```json
{
  "key": "insect",
  "label": "Insect / spider (six-legged)",
  "root": "thorax",
  "comment": "Normalized landmark positions in a unit bounding box, Blender axes: +X is the subject's left, -Y is forward, +Z is up. x/y span -0.5..0.5, z spans 0 to 1.",
  "bones": [
    {"name": "thorax",   "parent": null,      "head": [0.00,  0.00, 0.60], "tail": [0.00, -0.16, 0.62]},
    {"name": "head",     "parent": "thorax",  "head": [0.00, -0.16, 0.62], "tail": [0.00, -0.34, 0.62]},
    {"name": "abdomen",  "parent": "thorax",  "head": [0.00,  0.00, 0.60], "tail": [0.00,  0.30, 0.58]},

    {"name": "mandible.L", "parent": "head",  "head": [0.04, -0.32, 0.60], "tail": [0.10, -0.46, 0.56]},
    {"name": "mandible.R", "parent": "head",  "head": [-0.04, -0.32, 0.60], "tail": [-0.10, -0.46, 0.56]},

    {"name": "leg_a_upper.L", "parent": "thorax", "head": [0.06, -0.10, 0.58], "tail": [0.26, -0.22, 0.42]},
    {"name": "leg_a_lower.L", "parent": "leg_a_upper.L", "head": [0.26, -0.22, 0.42], "tail": [0.40, -0.28, 0.00]},
    {"name": "leg_b_upper.L", "parent": "thorax", "head": [0.06,  0.00, 0.58], "tail": [0.28,  0.00, 0.42]},
    {"name": "leg_b_lower.L", "parent": "leg_b_upper.L", "head": [0.28,  0.00, 0.42], "tail": [0.44,  0.02, 0.00]},
    {"name": "leg_c_upper.L", "parent": "thorax", "head": [0.06,  0.10, 0.58], "tail": [0.26,  0.22, 0.42]},
    {"name": "leg_c_lower.L", "parent": "leg_c_upper.L", "head": [0.26,  0.22, 0.42], "tail": [0.40,  0.30, 0.00]},

    {"name": "leg_a_upper.R", "parent": "thorax", "head": [-0.06, -0.10, 0.58], "tail": [-0.26, -0.22, 0.42]},
    {"name": "leg_a_lower.R", "parent": "leg_a_upper.R", "head": [-0.26, -0.22, 0.42], "tail": [-0.40, -0.28, 0.00]},
    {"name": "leg_b_upper.R", "parent": "thorax", "head": [-0.06,  0.00, 0.58], "tail": [-0.28,  0.00, 0.42]},
    {"name": "leg_b_lower.R", "parent": "leg_b_upper.R", "head": [-0.28,  0.00, 0.42], "tail": [-0.44,  0.02, 0.00]},
    {"name": "leg_c_upper.R", "parent": "thorax", "head": [-0.06,  0.10, 0.58], "tail": [-0.26,  0.22, 0.42]},
    {"name": "leg_c_lower.R", "parent": "leg_c_upper.R", "head": [-0.26,  0.22, 0.42], "tail": [-0.40,  0.30, 0.00]}
  ],
  "mirror_pairs": [
    ["mandible.L", "mandible.R"],
    ["leg_a_upper.L", "leg_a_upper.R"],
    ["leg_a_lower.L", "leg_a_lower.R"],
    ["leg_b_upper.L", "leg_b_upper.R"],
    ["leg_b_lower.L", "leg_b_lower.R"],
    ["leg_c_upper.L", "leg_c_upper.R"],
    ["leg_c_lower.L", "leg_c_lower.R"]
  ]
}
```

- [ ] **Step 6: Write `serpent.json`**

A pure chain, no limbs, no mirror pairs — the simplest possible template and the
one that proves the format does not assume bilateral symmetry.

```json
{
  "key": "serpent",
  "label": "Serpent (limbless chain)",
  "root": "spine_01",
  "comment": "Normalized landmark positions in a unit bounding box, Blender axes: +X is the subject's left, -Y is forward, +Z is up. x/y span -0.5..0.5, z spans 0 to 1.",
  "bones": [
    {"name": "spine_01", "parent": null,       "head": [0.00, -0.34, 0.30], "tail": [0.00, -0.24, 0.30]},
    {"name": "spine_02", "parent": "spine_01", "head": [0.00, -0.24, 0.30], "tail": [0.00, -0.14, 0.30]},
    {"name": "spine_03", "parent": "spine_02", "head": [0.00, -0.14, 0.30], "tail": [0.00, -0.04, 0.30]},
    {"name": "spine_04", "parent": "spine_03", "head": [0.00, -0.04, 0.30], "tail": [0.00,  0.06, 0.30]},
    {"name": "spine_05", "parent": "spine_04", "head": [0.00,  0.06, 0.30], "tail": [0.00,  0.16, 0.30]},
    {"name": "spine_06", "parent": "spine_05", "head": [0.00,  0.16, 0.30], "tail": [0.00,  0.26, 0.30]},
    {"name": "spine_07", "parent": "spine_06", "head": [0.00,  0.26, 0.30], "tail": [0.00,  0.36, 0.30]},
    {"name": "tail_tip", "parent": "spine_07", "head": [0.00,  0.36, 0.30], "tail": [0.00,  0.48, 0.30]},

    {"name": "neck",     "parent": "spine_01", "head": [0.00, -0.34, 0.30], "tail": [0.00, -0.42, 0.36]},
    {"name": "head",     "parent": "neck",     "head": [0.00, -0.42, 0.36], "tail": [0.00, -0.50, 0.40]}
  ],
  "mirror_pairs": []
}
```

- [ ] **Step 7: Write `biped_tail.json`**

The humanoid with a five-segment tail — the most common request for a creature
that is otherwise a biped. Copy `humanoid.json` verbatim, change `key` to
`biped_tail`, `label` to `"Biped with tail"`, and append these bones before the
closing `]`, plus the same `mirror_pairs` list unchanged:

```json
    {"name": "tail_01", "parent": "hips",    "head": [0.00,  0.03, 0.53], "tail": [0.00,  0.12, 0.48]},
    {"name": "tail_02", "parent": "tail_01", "head": [0.00,  0.12, 0.48], "tail": [0.00,  0.21, 0.42]},
    {"name": "tail_03", "parent": "tail_02", "head": [0.00,  0.21, 0.42], "tail": [0.00,  0.30, 0.35]},
    {"name": "tail_04", "parent": "tail_03", "head": [0.00,  0.30, 0.35], "tail": [0.00,  0.39, 0.28]},
    {"name": "tail_05", "parent": "tail_04", "head": [0.00,  0.39, 0.28], "tail": [0.00,  0.48, 0.22]}
```

- [ ] **Step 8: Stop hardcoding two template names in doctor**

In `src/warlock/doctor.py:_probe_blender`, replace the first three lines:

```python
def _probe_blender() -> Check:
    # Any template at all, not a hardcoded pair: templates are files, adding one
    # is the supported way to add a skeleton, and naming two of them here made
    # renaming or removing either a silent rigging outage.
    if not rigging.templates():
        return Check(
            "Blender (rigging)", False,
            f"no skeleton templates found in {rigging.TEMPLATE_DIR}",
            fatal=False,
        )
```

- [ ] **Step 9: Run the tests**

Run: `uv run pytest tests/test_rigging.py tests/test_doctor.py -v`
Expected: PASS. If a bone fails the unit-box assertion, fix the JSON — the box is
the contract `fit_template` scales against and a bone outside it lands outside the
mesh.

- [ ] **Step 10: Run everything and commit**

The UI needs no change: `/api/rig/templates` returns `rigging.catalog()` and
`app.js:loadRig` builds the select from it.

```bash
uv run pytest -q && uv run ruff check .
git add src/warlock/templates src/warlock/doctor.py tests/
git commit -m "Warlock v0.0.1

Add bird, fish, insect, serpent and biped_tail skeletons."
```

---

### Task 2: Shipped pose presets per template (review item #15)

Poses are per-job files, so every new character starts from a T-pose. A pose is a
bone-name→local-quaternion map, which is already portable across any rig fitted
from the same template — so a library is just JSON.

**Files:**
- Create: `src/warlock/templates/poses/humanoid.json` (and one per template you
  author poses for)
- Modify: `src/warlock/rigging.py` (preset registry), `src/warlock/app.py` (route)
- Modify: `src/warlock/static/app.js`, `index.html`
- Test: `tests/test_rigging.py`, `tests/test_poses_api.py`

**Interfaces:**
- Consumes: `rigging.get_template` (Task 1), `rigging.validate_pose` (existing).
- Produces: `rigging.preset_poses(template_key: str) -> list[dict]`, each
  `{"name": str, "bones": {bone: [x, y, z, w]}}`;
  `GET /api/rig/templates/{key}/poses -> {"poses": [...]}`.
- Consumed by: Task 5 (animation clips interpolate between two poses, and a shipped
  walk-contact pair is what makes that immediately useful).

**Quaternion contract:** XYZW, three.js order, a **local** rotation on the joint
node — identical to what the browser saves. Preset files are hand-authored in the
same numbers the editor produces, so a preset and a user pose are indistinguishable
by the time they reach the worker.

- [ ] **Step 1: Write the failing test**

In `tests/test_rigging.py`:

```python
def test_preset_poses_validate_against_their_template():
    from warlock import rigging

    for key in rigging.templates():
        bone_names = [b["name"] for b in rigging.get_template(key).bones]
        for preset in rigging.preset_poses(key):
            # The same validation a browser-saved pose goes through: unknown
            # bones and non-unit quaternions are rejected identically, so a
            # shipped preset can never be a thing the API would refuse.
            rigging.validate_pose(preset, bone_names)


def test_a_template_with_no_preset_file_returns_an_empty_list():
    from warlock import rigging

    assert rigging.preset_poses("serpent") == [] or all(
        "name" in p for p in rigging.preset_poses("serpent")
    )


def test_unknown_template_presets_raise():
    from warlock import rigging

    with pytest.raises(ValueError):
        rigging.preset_poses("not-a-template")
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_rigging.py -k preset -v`
Expected: FAIL — `AttributeError: preset_poses`

- [ ] **Step 3: Implement the registry**

In `src/warlock/rigging.py`, after the template registry section:

```python
# --- shipped pose libraries -------------------------------------------------
#
# A pose is a bone-name -> local-quaternion map, and fit_template puts a given
# template's bones in the same place on every mesh. So a pose authored against
# one humanoid rig applies to every other humanoid rig -- which is what makes a
# shipped library possible at all, and why these live next to the templates
# rather than being seeded into each job's poses/ directory.

PRESET_DIR = TEMPLATE_DIR / "poses"

_presets: dict[str, list[dict[str, Any]]] | None = None


def _load_presets() -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            found[path.stem] = [
                {"name": str(p["name"]), "bones": p["bones"]} for p in raw["poses"]
            ]
        except Exception:
            # A malformed library costs you that library, not the app -- the
            # same rule _load_templates follows.
            log.exception("skipping unusable pose library %s", path)
    return found


def preset_poses(template_key: str) -> list[dict[str, Any]]:
    """The shipped poses for a template. Raises ValueError on an unknown key."""
    global _presets
    get_template(template_key)  # validates the key against the registry
    if _presets is None:
        _presets = _load_presets()
    return [dict(p) for p in _presets.get(template_key, [])]
```

- [ ] **Step 4: Author the humanoid library**

Create `src/warlock/templates/poses/humanoid.json`. Quaternions are XYZW and
local; identity is `[0, 0, 0, 1]`. A rotation of θ about the local X axis is
`[sin(θ/2), 0, 0, cos(θ/2)]`. Only the bones a pose actually moves need listing —
`_reset_pose` in the worker returns everything else to rest.

```json
{
  "template": "humanoid",
  "comment": "Local joint rotations, XYZW, exactly what the browser's pose editor saves. Only moved bones are listed; the worker resets the rest.",
  "poses": [
    {
      "name": "idle",
      "bones": {
        "upper_arm.L": [0.0, 0.0, -0.1305, 0.9914],
        "upper_arm.R": [0.0, 0.0, 0.1305, 0.9914],
        "forearm.L": [0.0, 0.0, -0.0872, 0.9962],
        "forearm.R": [0.0, 0.0, 0.0872, 0.9962]
      }
    },
    {
      "name": "walk contact A",
      "bones": {
        "thigh.L": [0.2588, 0.0, 0.0, 0.9659],
        "shin.L": [-0.0872, 0.0, 0.0, 0.9962],
        "thigh.R": [-0.2588, 0.0, 0.0, 0.9659],
        "shin.R": [0.1736, 0.0, 0.0, 0.9848],
        "upper_arm.L": [-0.1736, 0.0, -0.1305, 0.9762],
        "upper_arm.R": [0.1736, 0.0, 0.1305, 0.9762]
      }
    },
    {
      "name": "walk contact B",
      "bones": {
        "thigh.L": [-0.2588, 0.0, 0.0, 0.9659],
        "shin.L": [0.1736, 0.0, 0.0, 0.9848],
        "thigh.R": [0.2588, 0.0, 0.0, 0.9659],
        "shin.R": [-0.0872, 0.0, 0.0, 0.9962],
        "upper_arm.L": [0.1736, 0.0, -0.1305, 0.9762],
        "upper_arm.R": [-0.1736, 0.0, 0.1305, 0.9762]
      }
    },
    {
      "name": "run",
      "bones": {
        "spine": [0.1305, 0.0, 0.0, 0.9914],
        "thigh.L": [0.4226, 0.0, 0.0, 0.9063],
        "shin.L": [-0.3420, 0.0, 0.0, 0.9397],
        "thigh.R": [-0.3420, 0.0, 0.0, 0.9397],
        "shin.R": [0.4226, 0.0, 0.0, 0.9063],
        "upper_arm.L": [-0.4226, 0.0, -0.1736, 0.8897],
        "forearm.L": [-0.5000, 0.0, 0.0, 0.8660],
        "upper_arm.R": [0.4226, 0.0, 0.1736, 0.8897],
        "forearm.R": [-0.5000, 0.0, 0.0, 0.8660]
      }
    },
    {
      "name": "attack",
      "bones": {
        "spine": [0.0, 0.0, -0.1736, 0.9848],
        "chest": [0.0, 0.0, -0.1305, 0.9914],
        "upper_arm.R": [-0.7071, 0.0, 0.0, 0.7071],
        "forearm.R": [-0.2588, 0.0, 0.0, 0.9659],
        "upper_arm.L": [0.2588, 0.0, -0.2588, 0.9306]
      }
    },
    {
      "name": "hit",
      "bones": {
        "spine": [-0.2588, 0.0, 0.0, 0.9659],
        "chest": [-0.1736, 0.0, 0.0, 0.9848],
        "head": [-0.2588, 0.0, 0.0, 0.9659],
        "upper_arm.L": [-0.3420, 0.0, -0.3420, 0.8754],
        "upper_arm.R": [-0.3420, 0.0, 0.3420, 0.8754]
      }
    },
    {
      "name": "death",
      "bones": {
        "hips": [-0.7071, 0.0, 0.0, 0.7071],
        "spine": [0.1736, 0.0, 0.0, 0.9848],
        "head": [0.2588, 0.0, 0.0, 0.9659],
        "thigh.L": [0.3420, 0.0, 0.0, 0.9397],
        "shin.L": [-0.5000, 0.0, 0.0, 0.8660],
        "thigh.R": [0.2588, 0.0, 0.0, 0.9659],
        "shin.R": [-0.4226, 0.0, 0.0, 0.9063],
        "upper_arm.L": [0.0, 0.0, -0.5000, 0.8660],
        "upper_arm.R": [0.0, 0.0, 0.5000, 0.8660]
      }
    }
  ]
}
```

Author the same seven-pose set for `quadruped` and `biped_tail` if time allows;
the test above passes with only `humanoid.json` present, and a template with no
library simply offers no presets.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_rigging.py -k preset -v`
Expected: PASS

- [ ] **Step 6: Write the failing API test**

In `tests/test_poses_api.py`:

```python
def test_template_presets_route(client):
    r = client.get("/api/rig/templates/humanoid/poses")
    assert r.status_code == 200
    names = [p["name"] for p in r.json()["poses"]]
    assert "idle" in names


def test_unknown_template_presets_is_a_400(client):
    assert client.get("/api/rig/templates/nope/poses").status_code == 400
```

- [ ] **Step 7: Add the route**

In `src/warlock/app.py`, after `rig_templates`:

```python
    @app.get("/api/rig/templates/{key}/poses")
    async def template_presets(key: str) -> dict[str, Any]:
        """The shipped pose library for a skeleton.

        Read-only and job-independent: applying one saves an ordinary pose
        through the existing POST route, so a preset and a hand-made pose are
        the same thing by the time anything else sees them.
        """
        try:
            poses = await asyncio.to_thread(rigging.preset_poses, key)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"poses": poses}
```

- [ ] **Step 8: Run it**

Run: `uv run pytest tests/test_poses_api.py -v`
Expected: PASS

- [ ] **Step 9: Offer presets in the pose editor**

In `static/index.html`, inside the pose editor block beside `pose-save`:

```html
        <select id="pose-preset"><option value="">preset…</option></select>
```

In `static/app.js`, load them when a rigged job is selected (inside
`setPoseJob`, after the rig is fetched) and apply on change:

```js
async function loadPresets(templateKey) {
  const select = document.getElementById("pose-preset");
  select.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  setText(blank, "preset…");
  select.append(blank);
  if (!templateKey) return;
  try {
    const { poses } = await (await fetch(`/api/rig/templates/${templateKey}/poses`)).json();
    poseState.presets = poses ?? [];
    for (const p of poseState.presets) {
      const opt = document.createElement("option");
      opt.value = p.name;
      setText(opt, p.name);
      select.append(opt);
    }
  } catch (e) {
    console.error("preset library failed", e);   // posing still works by hand
  }
}

document.getElementById("pose-preset").addEventListener("change", (e) => {
  const preset = (poseState.presets ?? []).find((p) => p.name === e.target.value);
  if (!preset) return;
  // A preset is applied to the live rig exactly like a saved pose: same bone
  // map, same code path, so it can be adjusted before it is saved.
  resetPose();
  applyPose(preset);
  e.target.value = "";
});
```

`applyPose` already takes a `{bones}` record, and the rig's template key is in
`rig.json`, which `GET /api/jobs/{id}/rig` returns as `template`.

- [ ] **Step 10: Verify by hand and commit**

Rig a mesh, open the pose editor, pick "run" from the preset select, confirm the
skeleton moves and the pose can be saved and baked.

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Ship a pose library per skeleton template."
```

---

### Task 3: Mirror posing (review item #16)

`mirror_pairs` is written into `rig.json` (`blender_worker.py:451`) and used by
nothing. Pose one arm, reflect it to the other.

**Files:**
- Modify: `src/warlock/static/app.js` (pose editor), `index.html`
- Test: `tests/test_rigging.py` (the reflection maths, host-side)

**Interfaces:**
- Produces: `rigging.mirror_quaternion(q: Sequence[float]) -> list[float]` (XYZW in,
  XYZW out) — implemented host-side and unit-tested there, then mirrored in JS.
- Produces: `mirrorPose(bones, pairs)` in `app.js`.
- Consumes: `rig.json`'s `mirror_pairs`, already present.

**The maths, stated once:** the templates are symmetric about the YZ plane
(mirroring negates X). Reflecting a rotation across that plane maps a quaternion
`(x, y, z, w)` to `(x, −y, −z, w)`. Do **not** negate `x`: reflection conjugates the
rotation, which flips the sign of the components perpendicular to the mirror
normal, and the mirror normal here is X.

- [ ] **Step 1: Write the failing test**

In `tests/test_rigging.py`:

```python
import math


def test_mirror_quaternion_reflects_across_the_yz_plane():
    from warlock import rigging

    # A rotation about Z becomes the opposite rotation about Z under an X mirror.
    half = math.radians(30) / 2
    q = [0.0, 0.0, math.sin(half), math.cos(half)]
    assert rigging.mirror_quaternion(q) == pytest.approx(
        [0.0, 0.0, -math.sin(half), math.cos(half)]
    )


def test_mirroring_twice_is_the_identity():
    from warlock import rigging

    q = [0.1830, 0.2588, 0.3536, 0.8810]
    twice = rigging.mirror_quaternion(rigging.mirror_quaternion(q))
    assert twice == pytest.approx(q)


def test_a_rotation_about_x_survives_mirroring():
    """A limb swinging forward/back mirrors to the same swing, not its opposite."""
    from warlock import rigging

    half = math.radians(40) / 2
    q = [math.sin(half), 0.0, 0.0, math.cos(half)]
    assert rigging.mirror_quaternion(q) == pytest.approx(q)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_rigging.py -k mirror -v`
Expected: FAIL — `AttributeError: mirror_quaternion`

- [ ] **Step 3: Implement it**

In `src/warlock/rigging.py`, in the pose-payloads section:

```python
def mirror_quaternion(q: Sequence[float]) -> list[float]:
    """Reflect a local joint rotation across the subject's YZ plane.

    Every template is symmetric about X (the mirror normal), and reflecting a
    rotation conjugates it: the components perpendicular to the normal flip
    sign and the one along it does not. So (x, y, z, w) -> (x, -y, -z, w).

    Lives here rather than only in the browser because it is the kind of sign
    convention that is wrong in a way you cannot see -- a mirrored arm that
    rotates the wrong way about one axis still looks plausible in a static
    pose. The JS copy in app.js must stay identical to this.
    """
    x, y, z, w = (float(v) for v in q)
    return [x, -y, -z, w]


def mirror_pose(
    bones: dict[str, Any], pairs: Sequence[Sequence[str]]
) -> dict[str, list[float]]:
    """Copy every posed bone onto its mirror partner, reflected.

    Bones with no partner (a spine, a tail) are left exactly as they are: they
    sit on the mirror plane, so reflecting them would rotate a centred limb off
    centre.
    """
    partner: dict[str, str] = {}
    for a, b in pairs:
        partner[a] = b
        partner[b] = a
    out = {name: [float(v) for v in q] for name, q in bones.items()}
    for name, quat in bones.items():
        other = partner.get(name)
        if other is not None:
            out[other] = mirror_quaternion(quat)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_rigging.py -k mirror -v`
Expected: PASS

- [ ] **Step 5: Add a test for `mirror_pose`**

```python
def test_mirror_pose_fills_in_the_other_side_and_leaves_centre_bones_alone():
    from warlock import rigging

    pairs = [["upper_arm.L", "upper_arm.R"]]
    posed = {"upper_arm.L": [0.0, 0.5, 0.0, 0.8660], "spine": [0.1, 0.0, 0.0, 0.9950]}
    out = rigging.mirror_pose(posed, pairs)
    assert out["upper_arm.R"] == pytest.approx([0.0, -0.5, 0.0, 0.8660])
    assert out["spine"] == pytest.approx([0.1, 0.0, 0.0, 0.9950])
```

Run: `uv run pytest tests/test_rigging.py -k mirror -v` → PASS

- [ ] **Step 6: Add the editor button**

In `static/index.html`, beside `pose-reset-all`:

```html
        <button type="button" id="pose-mirror">mirror L↔R</button>
```

In `static/app.js`, keep the rig's `mirror_pairs` when the rig is fetched (store it
on `poseState`, e.g. `poseState.mirrorPairs = rig.mirror_pairs ?? []`), and add:

```js
// Must stay identical to rigging.mirror_quaternion -- the templates are
// symmetric about X, and reflecting a rotation flips the components
// perpendicular to the mirror normal.
function mirrorQuaternion([x, y, z, w]) {
  return [x, -y, -z, w];
}

document.getElementById("pose-mirror").addEventListener("click", () => {
  const pairs = poseState.mirrorPairs ?? [];
  if (!pairs.length) return;
  const partner = new Map();
  for (const [a, b] of pairs) { partner.set(a, b); partner.set(b, a); }
  const bones = currentPoseBones();
  const mirrored = { ...bones };
  for (const [name, quat] of Object.entries(bones)) {
    const other = partner.get(name);
    if (other) mirrored[other] = mirrorQuaternion(quat);
  }
  applyPose({ bones: mirrored });
});
```

Hide the button when `poseState.mirrorPairs` is empty (serpent, fish tails) —
set `document.getElementById("pose-mirror").hidden = !(poseState.mirrorPairs ?? []).length;`
wherever the editor is opened.

- [ ] **Step 7: Verify by hand and commit**

Rig a humanoid, rotate the left arm, press "mirror L↔R", confirm the right arm
takes the reflected rotation and the spine does not move.

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Use rig.json's mirror pairs to mirror a pose L to R."
```

---

### Task 4: Joint-adjust pass (review item #17)

`rig.json` records fitted joint positions explicitly so a later adjust pass can
correct them without re-rigging. That pass does not exist. Fitting is
bbox-proportional and misses knees and shoulders on unusual silhouettes.

**Files:**
- Modify: `src/warlock/rigging.py` (`adjust_spec`, validation)
- Modify: `src/warlock/pipelines/blender_worker.py` (`op_rig` gains an override)
- Modify: `src/warlock/app.py` (route), `static/app.js` (drag markers)
- Test: `tests/test_rigging.py`, `tests/test_rig_api.py`

**Interfaces:**
- Produces: `rigging.validate_joints(payload: dict, template: Template) -> list[dict]`,
  returning the same `[{name, parent, head, tail}]` shape `fit_template` produces.
- Produces: `rigging.rig_spec(job_dir, template_key, bones=None)` — passing `bones`
  skips fitting and uses them verbatim.
- Produces: `POST /api/jobs/{id}/rig/joints` with a JSON body
  `{"bones": [{"name": str, "head": [x,y,z], "tail": [x,y,z]}, ...]}`, queueing a
  re-rig with the corrected skeleton.
- Consumes: `rig.json`'s existing `bones` and `bounds` fields.

**What this does and does not do:** it re-runs the *skinning* with corrected joints,
which means a new `rig.glb`. Saved poses survive because they are keyed by bone
name and the bone names do not change — `_apply_pose` already skips names a rig does
not have, so even a template change degrades rather than fails.

- [ ] **Step 1: Write the failing validation test**

In `tests/test_rigging.py`:

```python
def test_validate_joints_accepts_a_full_corrected_skeleton():
    from warlock import rigging

    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    payload = {"bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]}
                         for b in fitted]}
    out = rigging.validate_joints(payload, template)
    assert [b["name"] for b in out] == [b["name"] for b in template.bones]
    assert out[0]["parent"] == template.bones[0]["parent"]


def test_validate_joints_rejects_a_missing_bone():
    from warlock import rigging

    template = rigging.get_template("humanoid")
    with pytest.raises(ValueError):
        rigging.validate_joints({"bones": [{"name": "hips", "head": [0, 0, 0],
                                            "tail": [0, 0, 1]}]}, template)


def test_validate_joints_rejects_a_zero_length_bone():
    from warlock import rigging

    template = rigging.get_template("humanoid")
    payload = {"bones": [{"name": b["name"], "head": [0.0, 0.0, 0.0],
                          "tail": [0.0, 0.0, 0.0]} for b in template.bones]}
    with pytest.raises(ValueError):
        rigging.validate_joints(payload, template)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_rigging.py -k validate_joints -v`
Expected: FAIL — `AttributeError: validate_joints`

- [ ] **Step 3: Implement validation and the spec override**

In `src/warlock/rigging.py`, after `fit_template`:

```python
def validate_joints(payload: dict[str, Any], template: Template) -> list[dict[str, Any]]:
    """Normalize a corrected skeleton, or raise ValueError.

    The whole skeleton, not a patch: a partial correction would leave the caller
    and the worker disagreeing about which joints came from the fit and which
    from the user, and the fitted positions are already in rig.json for the
    editor to start from. Parentage comes from the template and is never
    caller-supplied -- a client cannot restructure the hierarchy, only move it.

    A zero-length bone is rejected rather than nudged: Blender silently deletes
    one on leaving edit mode and takes its children with it, so accepting it
    would produce a rig missing limbs with nothing to explain why.
    """
    raw = payload.get("bones")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joints payload requires a non-empty 'bones' list")
    by_name = {}
    for entry in raw:
        name = str(entry.get("name") or "")
        for end in ("head", "tail"):
            point = entry.get(end)
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError(f"bone {name!r} {end} must be a 3-vector")
            try:
                entry[end] = [float(v) for v in point]
            except (TypeError, ValueError):
                raise ValueError(f"bone {name!r} {end} is not numeric") from None
        by_name[name] = entry

    expected = [b["name"] for b in template.bones]
    missing = [n for n in expected if n not in by_name]
    if missing:
        raise ValueError(f"joints payload is missing bone(s): {missing}")
    unknown = [n for n in by_name if n not in expected]
    if unknown:
        raise ValueError(f"joints payload names unknown bone(s): {unknown}")

    span = max(
        _distance(by_name[a]["head"], by_name[b]["head"])
        for a in expected for b in expected
    ) or 1.0
    out = []
    for bone in template.bones:
        entry = by_name[bone["name"]]
        if _distance(entry["head"], entry["tail"]) < span * MIN_BONE_FRACTION:
            raise ValueError(f"bone {bone['name']!r} would be zero-length")
        out.append(
            {
                "name": bone["name"],
                "parent": bone["parent"],
                "head": entry["head"],
                "tail": entry["tail"],
            }
        )
    return out
```

Change `rig_spec` to accept the override:

```python
def rig_spec(
    job_dir: Path, template_key: str, bones: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The worker spec for rigging a finished job's mesh.

    ``bones`` overrides the bbox-proportional fit with joints the user moved.
    The fit is deliberately approximate -- rig.json records what it chose so a
    correction can be made without re-running anything else.
    """
    get_template(template_key)  # fail here, not three seconds into a subprocess
    spec = {
        "op": "rig",
        "source_glb": str(job_dir / "model.glb"),
        "out_glb": str(job_dir / "rig.glb"),
        "out_json": str(job_dir / "rig.json"),
        "result_path": str(job_dir / ".blender_result.json"),
        "template": template_key,
    }
    if bones is not None:
        spec["bones"] = bones
    return spec
```

- [ ] **Step 4: Honour the override in the worker**

In `src/warlock/pipelines/blender_worker.py:op_rig`, replace the fitting lines:

```python
    progress(0.25, "Fitting skeleton")
    lo, hi = _world_bounds(mesh)
    # Caller-supplied joints win over the fit. They are already validated
    # against the template host-side (rigging.validate_joints), so this is a
    # straight substitution rather than a second, disagreeing check.
    bones = spec.get("bones") or rigging.fit_template(template, lo, hi)
    arm_obj = _build_armature(bpy, bones)
```

and record the provenance in `rig_meta`:

```python
        "adjusted": bool(spec.get("bones")),
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_rigging.py -k "validate_joints or rig_spec" -v`
Expected: PASS

- [ ] **Step 6: Write the failing API test**

In `tests/test_rig_api.py`:

```python
def test_adjust_joints_queues_a_rerig(client, tmp_path):
    import json

    import warlock.config as config_mod
    from warlock import rigging

    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "a knight"}).json()["id"]
    job_dir = config_mod.get_config().job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid", "bones": fitted}), encoding="utf-8"
    )
    (job_dir / "rig.glb").write_bytes(b"glb")

    r = client.post(
        f"/api/jobs/{job_id}/rig/joints",
        json={"bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]}
                        for b in fitted]},
    )
    assert r.status_code == 200
    rig_job = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert rig_job["kind"] == "rig"
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["bones"]


def test_adjust_joints_on_an_unrigged_job_is_a_400(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/rig/joints", json={"bones": []}).status_code == 400
```

The `client` fixture here is whatever `tests/test_rig_api.py` already uses — do not
introduce a new one.

- [ ] **Step 7: Add the route and honour it in the queue**

In `src/warlock/app.py`, after `create_rig`:

```python
    @app.post("/api/jobs/{job_id}/rig/joints")
    async def adjust_joints(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Re-rig a mesh with joints the user moved.

        JSON rather than a form, for the same reason poses are: the body is a
        list of 3-vectors. A queued rig job rather than an inline call, for the
        same reason the original rig is: skinning is minutes of CPU and must
        never overlap a trellis run.
        """
        _check_job_id(job_id)
        source = await asyncio.to_thread(store().get, job_id)
        if source is None:
            raise HTTPException(404, "no such job")
        job_dir = config.job_dir(job_id)
        rig = await asyncio.to_thread(rigging.read_rig, job_dir)
        if rig is None or not (job_dir / "model.glb").exists():
            raise HTTPException(400, "job is not rigged")
        try:
            template = rigging.get_template(str(rig.get("template") or config.rig_template))
            bones = rigging.validate_joints(payload, template)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        params = {
            "source_job": job_id,
            "template": template.key,
            "bones": bones,
            "adjusted": True,
        }
        new_id = await asyncio.to_thread(
            store().create, "rig", source["prompt"], params, uuid.uuid4().hex[:12]
        )
        return {"id": new_id, "source_job": job_id}
```

In `src/warlock/queue.py:_rig`, pass the override through:

```python
        spec = rigging.rig_spec(source_dir, template, params.get("bones"))
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 9: Add drag-to-move markers in the UI**

The pose editor already builds clickable joint spheres (`syncJointMarkers`,
`bindRig`) and already attaches a `TransformControls` gizmo. Reuse both in a
"joints" mode:

- Add a toggle in `index.html`: `<button type="button" id="joint-toggle">adjust joints</button>`
- In `app.js`, add a `poseState.mode` of `"pose" | "joints"`. In joints mode, set
  the gizmo to `translate` (`gizmo.setMode("translate")`) instead of `rotate`, and
  drag the marker itself rather than the bone.
- Track the moved head positions in a `Map` keyed by bone name. A bone's `tail`
  follows its first child's `head` where one exists, and otherwise moves rigidly
  with its own head — state that in a comment, because it is the rule that keeps a
  dragged chain connected.
- On "apply", POST the full bone list (fitted positions from `rig.json` with the
  moved heads/tails substituted) to `/api/jobs/{id}/rig/joints`, then `poll(true)`.
- Restore `gizmo.setMode("rotate")` when leaving joints mode.

The marker positions in `rig.json` are in **Blender world space** for the source
mesh; the viewer shows the **glTF Y-up** export. Convert with the same single swap
the exporter applies at the root: `(x, y, z)_blender -> (x, z, -y)_gltf`, and invert
it on the way back. Put that conversion in one named function used in both
directions so the two can never drift.

- [ ] **Step 10: Verify by hand and commit**

Rig a humanoid whose knees sit wrong, drag them into place, apply, and confirm a
new rig job runs and the resulting `rig.glb` has the corrected joints.

```bash
git add -A
git commit -m "Warlock v0.0.1

Add a joint-adjust pass that re-skins without re-fitting."
```

---

### Task 5: Animation clips in sprite sheets (review item #18)

`Cell.frame` is documented as "always 0 today; the seam animated clips arrive on",
and cells arrive grouped by row precisely so this can work. Interpolating between
two saved poses turns Warlock into a complete 2D character pipeline.

**Files:**
- Modify: `src/warlock/pipelines/sheet.py`
- Modify: `src/warlock/app.py:510-575`, `src/warlock/queue.py:509-621`
- Modify: `src/warlock/static/app.js`, `index.html`
- Test: `tests/test_sheet.py`

**Interfaces:**
- Produces: `sheet.slerp(a: Sequence[float], b: Sequence[float], t: float) -> list[float]`
  (XYZW in and out, shortest-arc);
  `sheet.interpolate(pose_a: Mapping, pose_b: Mapping, frames: int) -> list[dict]`
  returning `frames` pose records named `"<a> -> <b> #n"` with `frame` indices;
  `sheet.plan(..., yaws: int = DEFAULT_YAWS)` already exists and is now passed through.
- Consumes: `rigging.read_pose` (existing) and the pose contract.
- The sidecar format does **not** change version: an animated clip is more cells with
  `frame > 0`, which is exactly what the flat `cells` list was built for.

- [ ] **Step 1: Write the failing test**

In `tests/test_sheet.py`:

```python
import math


def test_slerp_endpoints_and_midpoint():
    from warlock.pipelines import sheet

    a = [0.0, 0.0, 0.0, 1.0]
    half = math.radians(90) / 2
    b = [0.0, 0.0, math.sin(half), math.cos(half)]
    assert sheet.slerp(a, b, 0.0) == pytest.approx(a)
    assert sheet.slerp(a, b, 1.0) == pytest.approx(b)
    quarter = math.radians(45) / 2
    assert sheet.slerp(a, b, 0.5) == pytest.approx(
        [0.0, 0.0, math.sin(quarter), math.cos(quarter)], abs=1e-6
    )


def test_slerp_takes_the_short_way_round():
    """Negating a quaternion is the same rotation; without the sign fix the
    interpolation spins the long way and the sprite counter-rotates."""
    from warlock.pipelines import sheet

    a = [0.0, 0.0, 0.0, 1.0]
    b = [0.0, 0.0, 0.0, -1.0]
    mid = sheet.slerp(a, b, 0.5)
    assert abs(mid[3]) == pytest.approx(1.0, abs=1e-6)


def test_interpolate_produces_numbered_frames_between_two_poses():
    from warlock.pipelines import sheet

    a = {"id": "a" * 12, "name": "contact A", "bones": {"thigh.L": [0.0, 0.0, 0.0, 1.0]}}
    half = math.radians(60) / 2
    b = {"id": "b" * 12, "name": "contact B",
         "bones": {"thigh.L": [math.sin(half), 0.0, 0.0, math.cos(half)]}}

    frames = sheet.interpolate(a, b, 4)

    assert len(frames) == 4
    assert [f["frame"] for f in frames] == [0, 1, 2, 3]
    assert frames[0]["bones"]["thigh.L"] == pytest.approx(a["bones"]["thigh.L"])
    # The last frame stops short of B, so a looping clip does not hold a
    # duplicate frame at the seam.
    assert frames[-1]["bones"]["thigh.L"] != pytest.approx(b["bones"]["thigh.L"])


def test_a_bone_posed_in_only_one_end_interpolates_from_rest():
    from warlock.pipelines import sheet

    a = {"id": "a" * 12, "name": "A", "bones": {}}
    b = {"id": "b" * 12, "name": "B", "bones": {"head": [0.0, 0.0, 0.7071, 0.7071]}}
    frames = sheet.interpolate(a, b, 2)
    assert frames[0]["bones"]["head"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_plan_honours_the_yaw_count():
    from warlock.pipelines import sheet

    layout = sheet.plan([], yaws=4)
    assert layout.columns == 4
    assert len(layout.cells) == 4
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_sheet.py -k "slerp or interpolate" -v`
Expected: FAIL — `AttributeError: slerp`

- [ ] **Step 3: Implement slerp and interpolate**

In `src/warlock/pipelines/sheet.py`, after `yaw_angles`:

```python
# --- animation ---------------------------------------------------------------
#
# An animated clip is not a new format: it is more cells whose ``frame`` is above
# zero, which is exactly what the flat cells list in the sidecar was built for.
# Interpolation happens here, on the host, for the same reason the grid does --
# it is pure arithmetic, it must be testable without a GPU, and the browser
# preview and the Blender renderer must agree about what frame 3 of a clip is.

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)   # XYZW, three.js order

# Below this the two rotations are parallel and the slerp denominator collapses;
# a straight lerp is then both correct and stable.
SLERP_LINEAR_THRESHOLD = 0.9995

MAX_CLIP_FRAMES = 32


def slerp(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    """Shortest-arc spherical interpolation between two XYZW quaternions."""
    import math

    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:
        # q and -q are the same rotation but interpolate the long way round.
        # Without this a walk cycle counter-rotates through its own midpoint.
        bx, by, bz, bw, dot = -bx, -by, -bz, -bw, -dot
    if dot > SLERP_LINEAR_THRESHOLD:
        out = [
            ax + (bx - ax) * t,
            ay + (by - ay) * t,
            az + (bz - az) * t,
            aw + (bw - aw) * t,
        ]
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        wa = math.sin((1.0 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta
        out = [ax * wa + bx * wb, ay * wa + by * wb, az * wa + bz * wb, aw * wa + bw * wb]
    norm = sum(v * v for v in out) ** 0.5 or 1.0
    return [v / norm for v in out]


def interpolate(
    pose_a: Mapping[str, Any], pose_b: Mapping[str, Any], frames: int
) -> list[dict[str, Any]]:
    """``frames`` pose records stepping from A toward B.

    The last frame stops *short* of B rather than landing on it, so a clip that
    loops back to A does not hold a duplicate frame at the seam. A bone posed in
    only one of the two ends interpolates from rest, which is what the worker's
    _reset_pose already means by an omitted bone.
    """
    if not 1 <= frames <= MAX_CLIP_FRAMES:
        raise ValueError(f"a clip must be 1-{MAX_CLIP_FRAMES} frames")
    bones_a = pose_a.get("bones") or {}
    bones_b = pose_b.get("bones") or {}
    names = sorted(set(bones_a) | set(bones_b))
    name = f"{pose_a.get('name', 'A')} -> {pose_b.get('name', 'B')}"
    out: list[dict[str, Any]] = []
    for i in range(frames):
        t = i / frames
        out.append(
            {
                # The row's identity is the clip, not the frame: the worker
                # re-poses per cell within a clip row, which is the one place
                # the group-by-row optimisation does not apply.
                "id": f"{pose_a.get('id') or ''}:{pose_b.get('id') or ''}",
                "name": f"{name} #{i}",
                "frame": i,
                "bones": {
                    bone: slerp(
                        bones_a.get(bone, IDENTITY_QUAT),
                        bones_b.get(bone, IDENTITY_QUAT),
                        t,
                    )
                    for bone in names
                },
            }
        )
    return out
```

- [ ] **Step 4: Carry `frame` through `plan`**

In `plan`, replace `frame=0,` in the `Cell(...)` construction with:

```python
                    frame=int(pose.get("frame", 0)),
```

A record produced by `interpolate` carries its own `frame`; an ordinary saved pose
has none and defaults to 0, which is exactly the previous behaviour.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_sheet.py -v`
Expected: PASS

- [ ] **Step 6: Accept a clip on the sheet route**

In `src/warlock/app.py:create_sheet`, add form fields:

```python
        clip_from: Annotated[str | None, Form()] = None,
        clip_to: Annotated[str | None, Form()] = None,
        clip_frames: Annotated[int, Form()] = 8,
        yaws: Annotated[int | None, Form()] = None,
```

and, after the existing per-pose record loop, expand a clip into rows:

```python
        if clip_from or clip_to:
            if not (clip_from and clip_to):
                raise HTTPException(400, "a clip needs both clip_from and clip_to")
            for pose_id in (clip_from, clip_to):
                _check_pose_id(pose_id)
            ends = [
                await asyncio.to_thread(rigging.read_pose, job_dir, pid)
                for pid in (clip_from, clip_to)
            ]
            if any(e is None for e in ends):
                raise HTTPException(404, "no such pose")
            try:
                records = sheetlib.interpolate(ends[0], ends[1], clip_frames)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            if not (job_dir / "rig.glb").exists():
                raise HTTPException(400, "an animated clip needs a rigged mesh")
```

Pass `yaws` into both the validation `plan()` call and `params`:

```python
            sheetlib.plan(
                records,
                frame_size=frame_size or sheetlib.DEFAULT_FRAME_SIZE,
                elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
                lighting=lighting or "flat",
                yaws=yaws or sheetlib.DEFAULT_YAWS,
            )
```

and add to `params`:

```python
            "yaws": yaws or sheetlib.DEFAULT_YAWS,
            "clip": (
                {"from": clip_from, "to": clip_to, "frames": clip_frames}
                if clip_from
                else None
            ),
```

- [ ] **Step 7: Rebuild the same rows in the worker**

In `src/warlock/queue.py:_sheet`, after the existing pose-reading loop, insert:

```python
        clip = params.get("clip")
        if clip:
            # Rebuilt from the same two poses rather than shipped in params: the
            # host is the single place a grid or a clip is decided (see
            # pipelines/sheet.py), and storing the expanded frames would be a
            # second copy that could disagree with it.
            ends = [
                await asyncio.to_thread(rigging.read_pose, source_dir, str(clip[k]))
                for k in ("from", "to")
            ]
            if any(e is None for e in ends):
                raise RuntimeError("a pose in this clip no longer exists")
            records = sheetlib.interpolate(ends[0], ends[1], int(clip["frames"]))
```

and pass the yaw count into `plan`:

```python
            yaws=int(params.get("yaws", sheetlib.DEFAULT_YAWS)),
```

The `bones` lookup built from `records` is keyed by `r["id"]`, which is not unique
across clip frames — replace it with a per-row list so each cell gets its own frame's
rotations:

```python
        # Keyed by (pose id, frame) rather than pose id: every frame of a clip
        # shares an id by construction, and keying on the id alone would render
        # frame 0 in every row of the clip.
        bones = {(r.get("id"), r.get("frame", 0)): r["bones"] for r in records}
        cells = [
            {
                "index": c.index,
                "yaw": c.yaw,
                "pose": c.pose,
                "frame": c.frame,
                "bones": bones.get((c.pose, c.frame)) or {},
            }
            for c in layout.cells
        ]
```

- [ ] **Step 8: Re-pose per frame in the Blender worker**

In `src/warlock/pipelines/blender_worker.py:op_sheet`, the `posed` cache key must
include the frame or every row of a clip renders identically:

```python
    posed: Any = "__rest__"
    rendered = []
    for i, cell in enumerate(cells):
        # Cells arrive grouped by row, so this re-poses once per row rather than
        # once per frame. A clip's rows differ only by frame, which is why the
        # cache key carries it -- without that every frame of a clip would
        # render the first one.
        key = (cell.get("pose"), cell.get("frame", 0))
        if armature is not None and key != posed:
            _reset_pose(armature)
            _apply_pose(armature, cell.get("bones") or {})
            posed = key
```

- [ ] **Step 9: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 10: Add the UI**

In `static/index.html`, inside the sheet setup block:

```html
      <label for="sheet-yaws">Directions</label>
      <select id="sheet-yaws"></select>
      <label class="check"><input type="checkbox" id="sheet-clip"> Animated clip</label>
      <div id="sheet-clip-setup" hidden>
        <select id="sheet-clip-from"></select>
        <select id="sheet-clip-to"></select>
        <label for="sheet-clip-frames">Frames</label>
        <input type="number" id="sheet-clip-frames" min="2" max="32" value="8">
      </div>
```

In `app.js`: populate `sheet-yaws` with 4/8/16 (defaulting to 8), populate the two
clip selects from the job's saved poses, toggle `sheet-clip-setup` on the checkbox,
and add the fields to the render POST. Update `syncSheetRows`/`renderSheetPreview`
so the preview shows `frames` rows of `yaws` columns when a clip is selected —
the preview must agree with `plan()` because that is the whole point of the split.

- [ ] **Step 11: Verify by hand and commit**

Rig a humanoid, save the two shipped walk-contact presets as poses, render an
8-frame clip at 8 directions, and confirm the sheet is 8×8 and the sidecar's cells
carry `frame` 0–7.

```bash
git add -A
git commit -m "Warlock v0.0.1

Render animated sprite-sheet clips by interpolating between two poses."
```

---

### Task 6: Sheet direction count, pivot metadata and trim data (review item #19)

Three pieces: `yaws` (done in Task 5 Step 6 — this task is the remaining two),
per-cell projected pivot so engines can place sprites without guessing, and
alpha-bbox trim data for tight packing.

**Files:**
- Modify: `src/warlock/pipelines/sheet.py` (`Cell`, `pack`, `sidecar`)
- Modify: `src/warlock/pipelines/blender_worker.py` (`op_sheet` reports the pivot)
- Modify: `src/warlock/queue.py:_sheet`
- Test: `tests/test_sheet.py`

**Interfaces:**
- Consumes: `yaws` plumbing from Task 5.
- Produces: each sidecar cell gains `pivot_x`, `pivot_y` (pixels, the projected
  ground origin) and `trim` (`{"x", "y", "w", "h"}`, the alpha bounding box within
  the cell, or `null` when the cell is empty).
- Produces: `sheet.measure_trim(image, cell_size) -> dict | None`.
- The sidecar stays `SHEET_VERSION = 1`: these are additive keys, and an importer
  that ignores them reads the sheet exactly as before.

**Pivot definition** (matching `docs/NEXT.md`'s Pygame contract): the model's
**grounded, horizontally centred origin** projected into the cell. With an
orthographic camera framed once from the rest bbox, that projection is the same
for every cell in a column-independent way *except* for the vertical offset, which
depends only on elevation — so it is computed once in Blender and reported back,
not guessed per frame.

- [ ] **Step 1: Write the failing test**

In `tests/test_sheet.py`:

```python
def test_sidecar_carries_a_pivot_per_cell():
    from warlock.pipelines import sheet

    layout = sheet.plan([], frame_size=128, yaws=4)
    meta = sheet.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png",
        created=1.0, pivot=(64.0, 118.0),
    )
    assert all(c["pivot_x"] == 64.0 and c["pivot_y"] == 118.0 for c in meta["cells"])


def test_pivot_defaults_to_the_cell_centre_bottom_when_unmeasured():
    from warlock.pipelines import sheet

    layout = sheet.plan([], frame_size=128, yaws=4)
    meta = sheet.sidecar(
        layout, sheet_id="a" * 12, source_job="b" * 12, image="s.png", created=1.0
    )
    assert meta["cells"][0]["pivot_x"] == 64.0
    assert meta["cells"][0]["pivot_y"] == 128.0


def test_trim_measures_the_alpha_bounding_box(tmp_path):
    from PIL import Image

    from warlock.pipelines import sheet

    frame = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    frame.paste((255, 0, 0, 255), (10, 20, 30, 50))
    assert sheet.measure_trim(frame) == {"x": 10, "y": 20, "w": 20, "h": 30}


def test_trim_of_an_empty_frame_is_none():
    from PIL import Image

    from warlock.pipelines import sheet

    assert sheet.measure_trim(Image.new("RGBA", (64, 64), (0, 0, 0, 0))) is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_sheet.py -k "pivot or trim" -v`
Expected: FAIL — `sidecar() got an unexpected keyword argument 'pivot'`

- [ ] **Step 3: Implement trim and pivot in `sheet.py`**

Add to `Cell.as_dict` a trim/pivot-aware form. Simplest is to keep `Cell` pure and
merge the extras in `sidecar`:

```python
def measure_trim(image: Any) -> dict[str, int] | None:
    """The alpha bounding box within one rendered frame, or None if it is empty.

    Handed to importers that pack tightly: a 128px cell whose subject occupies
    40x90 of it wastes most of its texture, and the trim rectangle is what lets
    a packer reclaim that without re-rendering. Measured rather than computed
    from the bbox because the silhouette, not the bounding volume, is what a
    packer cares about.
    """
    alpha = image.convert("RGBA").getchannel("A")
    box = alpha.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
```

Change `pack` to collect trims while it composites — it already opens every frame,
so measuring there costs one extra pass over an image already in memory:

```python
def pack(sheet: Plan, frames: Mapping[int, Path], out_png: Path) -> dict[int, dict[str, int] | None]:
    """Composite the rendered frames into one RGBA atlas.

    Returns each cell's alpha bounding box, measured here because every frame is
    already open and decoded at this point.

    A missing or wrong-sized frame raises rather than silently leaving a hole: a
    sheet with an invisible gap in it looks like a modelling problem, and the
    user would go looking in the wrong place.
    """
    from PIL import Image

    size = sheet.frame_size
    atlas = Image.new("RGBA", (sheet.width, sheet.height), (0, 0, 0, 0))
    trims: dict[int, dict[str, int] | None] = {}
    try:
        for cell in sheet.cells:
            path = frames.get(cell.index)
            if path is None or not path.exists():
                raise ValueError(f"no rendered frame for cell {cell.index}")
            with Image.open(path) as frame:
                frame = frame.convert("RGBA")
                if frame.size != (size, size):
                    frame = frame.resize((size, size), Image.LANCZOS)
                trims[cell.index] = measure_trim(frame)
                atlas.paste(frame, (cell.x, cell.y))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(out_png, "PNG")
    finally:
        atlas.close()
    return trims
```

Extend `sidecar`:

```python
def sidecar(
    sheet: Plan,
    *,
    sheet_id: str,
    source_job: str,
    image: str,
    created: float,
    name: str = "",
    pivot: tuple[float, float] | None = None,
    trims: Mapping[int, dict[str, int] | None] | None = None,
) -> dict[str, Any]:
```

and build the cells with the extras:

```python
    # The projected ground origin, in pixels within a cell. Identical for every
    # cell by construction: the camera is framed once from the rest bbox and
    # only spins, so the subject's origin lands in the same place in every
    # direction. That stability is the property an engine needs to place a
    # sprite without it drifting as the character turns.
    px, py = pivot if pivot is not None else (sheet.frame_size / 2.0, float(sheet.frame_size))
    cells = []
    for c in sheet.cells:
        entry = c.as_dict(sheet.frame_size)
        entry["pivot_x"] = px
        entry["pivot_y"] = py
        entry["trim"] = (trims or {}).get(c.index)
        cells.append(entry)
```

and use `"cells": cells,` in the returned dict.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sheet.py -v`
Expected: PASS

- [ ] **Step 5: Report the pivot from Blender**

In `blender_worker.op_sheet`, after the camera is set up, project the grounded
origin into pixels and return it:

```python
    # The subject's ground origin: horizontally centred, sitting on the bbox
    # floor. Projected once, because the ortho camera is framed once and only
    # spins -- so this pixel is the same in every direction, which is exactly
    # what makes it usable as a sprite pivot.
    origin = (centre[0], centre[1], lo[2])
    pivot = _project(bpy, cam, origin, size)
```

and add the helper beside `_aim_camera`:

```python
def _project(bpy: Any, cam: Any, point: Sequence[float], size: int) -> tuple[float, float]:
    """World point -> pixel coordinates within one square frame.

    bpy_extras' own camera projection rather than reimplementing the ortho
    matrix: it already accounts for ortho_scale, the sensor fit and the aspect,
    and a hand-rolled version that disagreed would put every sprite's feet in
    the wrong place with nothing to indicate why.
    """
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    ndc = world_to_camera_view(bpy.context.scene, cam, Vector(point))
    # world_to_camera_view returns 0..1 with y up; image pixels are y down.
    return (float(ndc.x) * size, (1.0 - float(ndc.y)) * size)
```

Add `pivot` to `op_sheet`'s return: `return {"ok": True, "frames": rendered, "bounds": {...}, "pivot": list(pivot)}`.
Import `Sequence` from `collections.abc` at the top of `blender_worker.py`.

The camera is aimed per cell, so compute the pivot **after** the first
`_aim_camera` call — or, more simply, aim it once at yaw 0 before the render loop
and project then. Do the latter; it is one line and keeps the projection out of
the loop.

- [ ] **Step 6: Thread it through the queue**

In `src/warlock/queue.py:_sheet`, capture the worker result and pass both extras
into `sidecar`:

```python
            result = await asyncio.to_thread(
                functools.partial(
                    rigging.run_worker,
                    spec,
                    on_progress=on_progress,
                    on_start=on_start,
                    timeout=self.config.sheet_timeout,
                )
            )
            ...
            trims = await asyncio.to_thread(sheetlib.pack, layout, frames, png)

        pivot = result.get("pivot")
        meta = sheetlib.sidecar(
            layout,
            sheet_id=sheet_id,
            source_job=source_id,
            image=png.name,
            created=time.time(),
            name=str(params.get("name") or ""),
            pivot=tuple(pivot) if pivot else None,
            trims=trims,
        )
```

Note `pack` now returns the trims rather than the path — check no other caller
relies on its old return value (`grep -rn "sheetlib.pack\|sheet.pack" src tests`).

- [ ] **Step 7: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 8: Verify against Pygame-CE**

Per `docs/NEXT.md`'s acceptance checks: load the PNG with
`pygame.image.load(path).convert_alpha()`, slice every cell with
`Surface.subsurface()` using the sidecar rectangles, and confirm the pivot places
the sprite consistently as the direction changes. Record the result in
`docs/NEXT.md`.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "Warlock v0.0.1

Add sheet direction count, per-cell pivot and alpha trim metadata."
```

---

## Plan C self-review notes

- Items #14–#19 each map to a task: #14→T1, #15→T2, #16→T3, #17→T4, #18→T5, #19→T5(a)+T6(b,c).
- The `yaws` kwarg (#19a) is plumbed in Task 5 Step 6 because the clip work touches
  the same `plan()` call sites; Task 6 covers the pivot and trim halves.
- Three places where a cache key had to grow a `frame` component are called out
  explicitly (queue's `bones` dict, the worker's `posed` key, `Cell.frame` in `plan`)
  — getting any one wrong renders every frame of a clip identically, which looks
  like a posing bug rather than a keying bug.
- `mirror_quaternion` exists in both Python and JS on purpose; the Python copy is
  the tested one and the comment in each says so.
