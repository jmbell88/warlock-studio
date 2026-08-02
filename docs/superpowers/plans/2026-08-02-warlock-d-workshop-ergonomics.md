# Plan D — Workshop Ergonomics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a workshop with 100+ assets navigable: name and find things, see what you made without clicking, compare variants side by side, inspect a mesh, and stop losing form state.

**Architecture:** Mostly frontend, with two small server additions — searchable metadata on a job (name, tags, favorite) and a thumbnail upload endpoint. Nothing here touches the queue, the VRAM handoff or the Blender boundary. Filtering is deliberately client-side: `/api/jobs` already returns the whole list a workshop holds, and a server-side search would be a second, disagreeing definition of "matches".

**Tech Stack:** sqlite3, FastAPI, three.js r170 (vendored), plain JS, `localStorage`.

## Global Constraints

See `2026-08-02-warlock-review-index.md` § Global Constraints. Every task's
requirements implicitly include that section. Load-bearing here:

- **No frontend build step.** No npm, no bundler, no `package.json`. `BufferGeometryUtils`
  and any other three.js addon must come from the already-vendored r170 under
  `static/vendor/three/`; adding one is a one-time manual download, not a fix.
- **`JobStore._lock` on every method that touches `self._conn`.**
- `db.MIGRATIONS` is **append-only** — never edit a shipped entry.

## Cross-plan dependency

Task 1 adds a `MIGRATIONS` entry. **Plan A Task 4 adds the first one.** If Plan A
has already run, this becomes entry index 1; if not, this becomes entry index 0 and
Plan A appends after it. Check `len(db.MIGRATIONS)` before writing and append —
never insert.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/warlock/db.py` | job store | Modify: `name`/`tags`/`favorite` + migration |
| `src/warlock/app.py` | routes | Modify: `PATCH /api/jobs/{id}`, `POST .../thumb.png` |
| `src/warlock/static/index.html` | markup + styles | Modify: filter bar, toolbar, compare pane, toasts |
| `src/warlock/static/app.js` | all UI behaviour | Modify throughout |

---

### Task 1: Job naming, tags, favourites and search (review item #20)

A job's identity is forever its raw prompt. There is no rename, tag, favourite,
filter or search. At 100+ assets the list is unusable.

**Files:**
- Modify: `src/warlock/db.py`
- Modify: `src/warlock/app.py`
- Modify: `src/warlock/static/index.html`, `src/warlock/static/app.js`
- Test: `tests/test_db.py`, `tests/test_migrate.py`, `tests/test_api.py`

**Interfaces:**
- Produces: columns `jobs.name TEXT NOT NULL DEFAULT ''`,
  `jobs.tags TEXT NOT NULL DEFAULT ''` (comma-separated, normalized lowercase),
  `jobs.favorite INTEGER NOT NULL DEFAULT 0`.
- Produces: `JobStore.set_meta(job_id: str, *, name: str | None = None, tags: str | None = None, favorite: bool | None = None) -> bool`
  — returns False when the job does not exist.
- Produces: `PATCH /api/jobs/{id}` taking a JSON body
  `{"name"?: str, "tags"?: list[str] | str, "favorite"?: bool}` and returning the
  updated job.
- Consumed by: Task 6 (presets reuse the tag normalizer).

**Why columns rather than the params blob:** these are the fields the list is
sorted and filtered by. `params` is a JSON string in sqlite, so filtering on it
means parsing every row; a column is what makes "show me favourites tagged
`weapon`" a query rather than a scan. The review offers either; this is the choice
and the reason.

- [ ] **Step 1: Write the failing tests**

In `tests/test_db.py`:

```python
def test_set_meta_updates_name_tags_and_favorite(store):
    job_id = store.create("text", "a barrel", {})
    assert store.set_meta(job_id, name="Oak barrel", tags="prop,fantasy", favorite=True)
    row = store.get(job_id)
    assert row["name"] == "Oak barrel"
    assert row["tags"] == "prop,fantasy"
    assert row["favorite"] == 1


def test_set_meta_leaves_unspecified_fields_alone(store):
    job_id = store.create("text", "a barrel", {})
    store.set_meta(job_id, name="Oak barrel")
    store.set_meta(job_id, favorite=True)
    row = store.get(job_id)
    assert row["name"] == "Oak barrel"
    assert row["favorite"] == 1


def test_set_meta_on_a_missing_job_is_false(store):
    assert store.set_meta("0" * 12, name="x") is False


def test_new_jobs_default_to_empty_metadata(store):
    row = store.get(store.create("text", "a barrel", {}))
    assert row["name"] == ""
    assert row["tags"] == ""
    assert row["favorite"] == 0
```

In `tests/test_migrate.py`:

```python
def test_a_db_without_metadata_columns_gains_them(tmp_path):
    import sqlite3

    from warlock.db import JobStore

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            prompt TEXT, params TEXT NOT NULL DEFAULT '{}', error TEXT,
            created_at REAL NOT NULL, started_at REAL, finished_at REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO jobs (id, kind, status, params, created_at)"
        " VALUES ('aaaaaaaaaaaa', 'text', 'done', '{}', 1.0)"
    )
    conn.commit()
    conn.close()

    store = JobStore(path)
    try:
        row = store.list(10)[0]
        assert row["name"] == ""
        assert row["tags"] == ""
        assert row["favorite"] == 0
    finally:
        store.close()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_db.py tests/test_migrate.py -v`
Expected: FAIL — `AttributeError: set_meta`

- [ ] **Step 3: Add the columns and the migration**

In `src/warlock/db.py`, extend `_SCHEMA`'s `CREATE TABLE` with:

```sql
    name        TEXT NOT NULL DEFAULT '',   -- user-given title; the prompt is the fallback
    tags        TEXT NOT NULL DEFAULT '',   -- comma-separated, normalized lowercase
    favorite    INTEGER NOT NULL DEFAULT 0
);
```

and **append** to `MIGRATIONS` (do not insert — check `len(MIGRATIONS)` first, see
the cross-plan note at the top):

```python
    # A job's identity used to be its raw prompt forever. Columns rather than
    # params keys because these are what the list filters and sorts on, and
    # params is a JSON string sqlite cannot index into.
    [
        "ALTER TABLE jobs ADD COLUMN name TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN tags TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_jobs_favorite ON jobs(favorite)",
    ],
```

If Plan A has not run, `_migrate` still needs its skip-existing-column guard — copy
it from Plan A Task 4 Step 3 verbatim, because a fresh DB gets these columns from
`_SCHEMA` and would otherwise fail replaying the `ALTER`.

- [ ] **Step 4: Add `set_meta`**

```python
    def set_meta(
        self,
        job_id: str,
        *,
        name: str | None = None,
        tags: str | None = None,
        favorite: bool | None = None,
    ) -> bool:
        """Update the user-facing metadata. Only the fields given are written.

        Partial by design: the UI's star button and its rename field are
        separate actions on the same row, and a full-row write from either
        would silently clobber whatever the other just did.
        """
        sets: list[str] = []
        args: list[Any] = []
        for column, value in (("name", name), ("tags", tags)):
            if value is not None:
                sets.append(f"{column} = ?")
                args.append(value)
        if favorite is not None:
            sets.append("favorite = ?")
            args.append(1 if favorite else 0)
        if not sets:
            return self.get(job_id) is not None
        args.append(job_id)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", args
            )
            self._conn.commit()
            return cur.rowcount > 0
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_db.py tests/test_migrate.py -v`
Expected: PASS

- [ ] **Step 6: Write the failing API test**

In `tests/test_api.py`:

```python
def test_patch_job_metadata(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"}).json()["id"]
    r = client.patch(
        f"/api/jobs/{job_id}",
        json={"name": "Oak barrel", "tags": ["Prop", " Fantasy "], "favorite": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Oak barrel"
    assert body["tags"] == "fantasy,prop"      # normalized and sorted
    assert body["favorite"] == 1


def test_patch_rejects_an_overlong_name(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    assert client.patch(f"/api/jobs/{job_id}", json={"name": "x" * 200}).status_code == 400


def test_patch_a_missing_job_is_404(client):
    assert client.patch(f"/api/jobs/{'0' * 12}", json={"name": "x"}).status_code == 404
```

- [ ] **Step 7: Add the route**

In `src/warlock/app.py`, near the other job routes:

```python
    MAX_JOB_NAME = 120
    MAX_TAGS = 20
    MAX_TAG_LEN = 32

    def _normalize_tags(raw: Any) -> str:
        """A list or a comma string -> a sorted, deduped, lowercase csv.

        Normalized on the way in rather than at every read: a filter that has to
        case-fold and trim on each keystroke over a thousand rows is the kind of
        thing that makes a UI feel broken, and 'Prop' and 'prop ' being two tags
        is the kind of thing that makes a workshop unsearchable.
        """
        if raw is None:
            return ""
        items = raw if isinstance(raw, list) else str(raw).split(",")
        tags = sorted({t.strip().lower() for t in (str(i) for i in items) if t.strip()})
        if len(tags) > MAX_TAGS:
            raise HTTPException(400, f"at most {MAX_TAGS} tags")
        if any(len(t) > MAX_TAG_LEN for t in tags):
            raise HTTPException(400, f"a tag may be at most {MAX_TAG_LEN} characters")
        return ",".join(tags)

    @app.patch("/api/jobs/{job_id}")
    async def update_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Rename, retag or (un)favourite a job.

        JSON rather than a form like most of this API, for the same reason poses
        are: tags are a list, which multipart can only carry as a re-encoded
        string.
        """
        _check_job_id(job_id)
        name = payload.get("name")
        if name is not None:
            name = str(name).strip()
            if len(name) > MAX_JOB_NAME:
                raise HTTPException(400, f"name must be at most {MAX_JOB_NAME} characters")
        tags = _normalize_tags(payload["tags"]) if "tags" in payload else None
        favorite = bool(payload["favorite"]) if "favorite" in payload else None

        updated = await asyncio.to_thread(
            functools.partial(
                store().set_meta, job_id, name=name, tags=tags, favorite=favorite
            )
        )
        if not updated:
            raise HTTPException(404, "no such job")
        job = await asyncio.to_thread(store().get, job_id)
        _attach_files(job, config.job_dir(job_id))
        _attach_progress(job, app.state.worker)
        return job
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 9: Add the filter bar and card controls**

In `static/index.html`, above `<ul id="jobs">`:

```html
  <div id="filter-bar">
    <input type="search" id="filter-text" placeholder="search name, prompt or tag">
    <select id="filter-status">
      <option value="">all</option>
      <option value="done">done</option>
      <option value="running">running</option>
      <option value="error">error</option>
    </select>
    <select id="filter-kind">
      <option value="">any kind</option>
      <option value="text">text</option>
      <option value="image">image</option>
      <option value="rig">rig</option>
      <option value="sheet">sheet</option>
    </select>
    <label class="check"><input type="checkbox" id="filter-fav"> ★ only</label>
  </div>
```

In `static/app.js`, add the filter and apply it in `renderJobs`:

```js
const filters = {
  text: document.getElementById("filter-text"),
  status: document.getElementById("filter-status"),
  kind: document.getElementById("filter-kind"),
  fav: document.getElementById("filter-fav"),
};
for (const el of Object.values(filters)) {
  el.addEventListener("input", () => renderJobs([...jobsById.values()]));
}

// Client-side on purpose: /api/jobs already returns the whole workshop, and a
// server-side search would be a second definition of "matches" that could
// disagree with what the list is showing.
function jobMatches(job) {
  if (filters.status.value && job.status !== filters.status.value) return false;
  if (filters.kind.value && job.kind !== filters.kind.value) return false;
  if (filters.fav.checked && !job.favorite) return false;
  const q = filters.text.value.trim().toLowerCase();
  if (!q) return true;
  return [job.name, job.prompt, job.tags, job.id]
    .filter(Boolean)
    .some((field) => String(field).toLowerCase().includes(q));
}
```

In `renderJobs`, filter before the loop and hide non-matching nodes rather than
removing them (removal would fight the existing reuse-by-id logic):

```js
function renderJobs(jobs) {
  jobs.forEach((job, i) => {
    jobsById.set(job.id, job);
    const n = nodes.get(job.id) ?? createNode(job.id);
    updateNode(n, job);
    n.li.hidden = !jobMatches(job);
    if (ul.children[i] !== n.li) ul.insertBefore(n.li, ul.children[i] ?? null);
  });
  ...
}
```

Add a star button and an inline-rename to `createNode`:

```js
  const star = document.createElement("button");
  star.type = "button";
  star.className = "star";
  star.title = "Favourite";
  star.addEventListener("click", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    await patchJob(id, { favorite: !job?.favorite });
    poll(true);
  });
  title.addEventListener("dblclick", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    const next = prompt("Name this asset", job?.name || job?.prompt || "");
    if (next === null) return;
    await patchJob(id, { name: next });
    poll(true);
  });
```

and the shared helper:

```js
async function patchJob(id, body) {
  const r = await fetch(`/api/jobs/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    alert(detail.detail ?? "could not update the job");   // replaced by a toast in Task 5
  }
  return r.ok;
}
```

In `updateNode`, prefer the name over the prompt in the title and reflect the star:

```js
  setText(n.title, job.name || (job.kind === "rig"
    ? `rig · ${job.prompt ?? job.params?.source_job ?? job.id}`
    : job.prompt ?? `image job ${job.id}`));
  setText(n.star, job.favorite ? "★" : "☆");
  n.star.classList.toggle("on", Boolean(job.favorite));
```

- [ ] **Step 10: Verify by hand and commit**

Rename a job by double-clicking its title, star it, and confirm the filter box
finds it by name and by tag.

```bash
git add -A
git commit -m "Warlock v0.0.1

Add job names, tags, favourites and a client-side filter."
```

---

### Task 2: Rendered-model thumbnails (review item #21)

The list shows only the 44 px `input.png`; the actual mesh has no preview until you
click. Snapshot the three.js canvas on first load and POST it.

**Files:**
- Modify: `src/warlock/app.py` (`_MEDIA`, upload route)
- Modify: `src/warlock/static/app.js` (`showModel`, `updateNode`)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `POST /api/jobs/{id}/thumb.png` accepting a raw PNG body (max 512 KB)
  and writing `<job_dir>/thumb.png`; `"thumb.png"` added to `_MEDIA` and listed by
  `_attach_files` when present.
- Produces: `captureThumbnail(jobId)` in `app.js`.

**Why client-side:** the alternative is a server-side render, which needs Blender
(the `rig` extra) and a place on the serial GPU queue for something purely
cosmetic. The viewer already has the mesh loaded and framed at the moment the user
first opens it; a `toBlob` off that canvas is free.

- [ ] **Step 1: Write the failing test**

In `tests/test_api.py`:

```python
def test_thumbnail_upload_and_download(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    png = _png_bytes()
    r = client.post(
        f"/api/jobs/{job_id}/thumb.png", content=png, headers={"Content-Type": "image/png"}
    )
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}/files/thumb.png").status_code == 200
    assert "thumb.png" in client.get(f"/api/jobs/{job_id}").json()["files"]


def test_thumbnail_rejects_a_non_png(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    r = client.post(f"/api/jobs/{job_id}/thumb.png", content=b"not a png")
    assert r.status_code == 400


def test_thumbnail_rejects_an_oversized_body(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    r = client.post(f"/api/jobs/{job_id}/thumb.png", content=b"\x89PNG\r\n\x1a\n" + b"0" * 600_000)
    assert r.status_code == 413
```

`_png_bytes` already exists in `tests/test_api.py`.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_api.py -k thumb -v`
Expected: FAIL — 404 on the upload route

- [ ] **Step 3: Add the route**

In `src/warlock/app.py`:

```python
    # A canvas snapshot at list size. Bounded because it arrives as a raw body
    # from the browser and the only thing between it and the disk is this number.
    MAX_THUMB_BYTES = 512 * 1024
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

    @app.post("/api/jobs/{job_id}/thumb.png")
    async def put_thumbnail(job_id: str, request: Request) -> dict[str, Any]:
        """Store a client-rendered preview of the mesh.

        Rendered in the browser rather than on the server: the viewer already has
        the model loaded and framed when the user first opens it, so the snapshot
        is free -- while a server-side render would need Blender and a place on
        the serial GPU queue for something purely cosmetic.

        The magic-byte check is the whole validation: this is written under a
        fixed filename inside a job directory that already exists, so the only
        thing worth refusing is a body that is not the image it claims to be.
        """
        _check_job_id(job_id)
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        data = await request.body()
        if len(data) > MAX_THUMB_BYTES:
            raise HTTPException(413, "thumbnail too large")
        if not data.startswith(PNG_MAGIC):
            raise HTTPException(400, "thumbnail must be a PNG")
        job_dir = config.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread((job_dir / "thumb.png").write_bytes, data)
        return {"ok": True}
```

Add `Request` to the fastapi import line. Add `"thumb.png": "image/png",` to
`_MEDIA`, and in `_attach_files`:

```python
    if (job_dir / "thumb.png").exists():
        files.append("thumb.png")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_api.py -k thumb -v`
Expected: PASS

- [ ] **Step 5: Capture it in the viewer**

In `static/app.js`, `showModel(url, onReady)` already takes a ready callback. Add:

```js
// One snapshot per job, taken the first time its mesh is framed in the viewer.
// Deliberately after a render rather than on load: the canvas is only correct
// once the model has been drawn at least once with the camera framed.
const thumbedJobs = new Set();

function captureThumbnail(jobId) {
  if (!jobId || thumbedJobs.has(jobId)) return;
  const job = jobsById.get(jobId);
  if (job?.files?.includes("thumb.png")) { thumbedJobs.add(jobId); return; }
  thumbedJobs.add(jobId);
  requestAnimationFrame(() => {
    renderer.render(scene, camera);   // guarantee the buffer holds this model
    canvas.toBlob((blob) => {
      if (!blob) return;
      fetch(`/api/jobs/${jobId}/thumb.png`, {
        method: "POST",
        headers: { "Content-Type": "image/png" },
        body: blob,
      }).catch((e) => console.error("thumbnail upload failed", e));
    }, "image/png");
  });
}
```

The WebGL context is created at `app.js:9` without `preserveDrawingBuffer`, so
`toBlob` would return an empty image on most drivers. Change that one line:

```js
// preserveDrawingBuffer is what makes canvas.toBlob return the frame that was
// just drawn rather than a cleared buffer. It costs a little fill rate and is
// the only way to snapshot a thumbnail without a second offscreen renderer.
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
```

In `showSelected`, pass the capture as the ready callback:

```js
  showModel(`/api/jobs/${job.id}/files/model.glb`, () => captureThumbnail(job.id));
```

- [ ] **Step 6: Show it on the card**

In `updateNode`, prefer the thumbnail over `input.png`:

```js
  // The mesh, not the reference image, is what the user is looking for in a
  // list of a hundred assets -- the reference only stands in until one exists.
  const preferred = job.files.includes("thumb.png") ? "thumb.png" : "input.png";
  const hasImage = job.files.includes(preferred);
  const src = `/api/jobs/${job.id}/files/${preferred}`;
  if (hasImage && n.img.getAttribute("src") !== src) n.img.src = src;
  n.img.hidden = !hasImage;
```

- [ ] **Step 7: Verify by hand and commit**

Open a finished mesh, go back to the list, and confirm the card now shows a render
of the model rather than the reference image.

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Capture and serve a rendered thumbnail per mesh."
```

---

### Task 3: Compare view for variants (review item #22)

`params.rerun_of` exists and is never surfaced. Picking the best of N seeds is the
core loop of AI asset generation and currently means clicking back and forth.

**Files:**
- Modify: `src/warlock/static/index.html`, `src/warlock/static/app.js`
- Test: manual (frontend only)

**Interfaces:**
- Consumes: `params.rerun_of` (existing), `params.parent_id` / `parent_id` column
  (Plan A Task 4) when present.
- Produces: a second `THREE.WebGLRenderer` over a second canvas, with a shared
  `OrbitControls` target so both orbit together.

**Why a second renderer and not a split viewport:** the existing viewer owns a
single scene, camera and controls that the pose editor and sheet preview both
reach into. A second, independent renderer over its own canvas leaves all of that
untouched; a viewport split would mean every one of those consumers learning about
which half it is drawing into.

- [ ] **Step 1: Add the second canvas**

In `static/index.html`, beside `<canvas id="viewer">`:

```html
  <canvas id="viewer-b" hidden></canvas>
```

with CSS that puts the two side by side when `#stage` has a `comparing` class:

```css
    #stage.comparing #viewer,
    #stage.comparing #viewer-b { width: 50%; }
    #stage.comparing #viewer-b { display: block; }
```

Match the existing layout container's id — check `index.html` for what wraps the
viewer and use that, rather than introducing `#stage` if something else is there.

- [ ] **Step 2: Build the comparison renderer**

In `static/app.js`, after the main renderer setup:

```js
// --- compare view -----------------------------------------------------------
//
// A second renderer over its own canvas rather than a split viewport: the main
// scene, camera and controls are reached into by the pose editor and the sheet
// preview, and a viewport split would mean each of them learning which half it
// is drawing. This way none of them change.

const canvasB = document.getElementById("viewer-b");
let rendererB = null;
let sceneB = null;
let cameraB = null;
let modelB = null;
let comparing = null;   // the job id shown on the right

function ensureCompare() {
  if (rendererB) return;
  rendererB = new THREE.WebGLRenderer({ canvas: canvasB, antialias: true });
  rendererB.outputColorSpace = renderer.outputColorSpace;
  sceneB = new THREE.Scene();
  sceneB.environment = scene.environment;
  sceneB.background = scene.background;
  cameraB = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  const keyB = new THREE.DirectionalLight(0xffffff, 1.5);
  keyB.position.copy(key.position);
  sceneB.add(keyB);
  sceneB.add(new THREE.GridHelper(4, 16, 0x2c2f3a, 0x232530));
}

async function compareWith(jobId) {
  ensureCompare();
  comparing = jobId;
  document.getElementById("stage").classList.add("comparing");
  canvasB.hidden = false;
  if (modelB) disposeModel(modelB);
  const gltf = await loader.loadAsync(`/api/jobs/${jobId}/files/model.glb`);
  modelB = gltf.scene;
  sceneB.add(modelB);
  resize();
}

function stopComparing() {
  comparing = null;
  document.getElementById("stage").classList.remove("comparing");
  canvasB.hidden = true;
  if (modelB) { disposeModel(modelB); modelB = null; }
  resize();
}
```

- [ ] **Step 3: Sync the cameras and render both**

In the existing animation loop, after the main render:

```js
  if (comparing && rendererB) {
    // One camera state, two renders: the whole point of the compare view is
    // that the two meshes are seen from the identical angle, so the second
    // camera copies the first rather than having controls of its own.
    cameraB.position.copy(camera.position);
    cameraB.quaternion.copy(camera.quaternion);
    cameraB.updateProjectionMatrix();
    rendererB.render(sceneB, cameraB);
  }
```

In `resize()`, size the second canvas the same way the first is sized, halving both
widths when `comparing` is set, and set `cameraB.aspect` to match.

- [ ] **Step 4: Add the entry points**

In `createNode`, add a `compare` button (hidden unless the job has a `model.glb`
and is not the selected one):

```js
  const compare = document.createElement("button");
  setText(compare, "compare");
  compare.title = "Show this beside the selected model";
  compare.hidden = true;
  compare.addEventListener("click", (e) => {
    e.stopPropagation();
    if (comparing === id) stopComparing();
    else compareWith(id).catch((err) => console.error("compare failed", err));
  });
```

In `updateNode`:

```js
  n.compare.hidden = !done || !job.files.includes("model.glb") || job.id === selected;
  setText(n.compare, comparing === job.id ? "stop compare" : "compare");
```

- [ ] **Step 5: Surface the variant lineage**

In `updateNode`, mark siblings so the loop is discoverable:

```js
  // rerun_of has been stored since re-roll shipped and shown nowhere. A variant
  // that does not say what it is a variant of is just another row.
  const from = job.params?.rerun_of || job.parent_id;
  setText(n.lineage, from ? `variant of ${from.slice(0, 6)}` : "");
```

Create the `lineage` div in `createNode` alongside `stage`, append it into `info`,
and add it to the returned node object.

- [ ] **Step 6: Verify by hand and commit**

Generate a mesh, re-roll it, select one and press "compare" on the other; orbit and
confirm both move together.

```bash
uv run ruff check .
git add src/warlock/static
git commit -m "Warlock v0.0.1

Add a side-by-side compare view with a synced camera."
```

---

### Task 4: Viewer inspection tools (review item #23)

The viewer has no wireframe, no tri/vert count, no dimensions, no turntable.

**Files:**
- Modify: `src/warlock/static/index.html`, `src/warlock/static/app.js`
- Test: manual (frontend only)

**Interfaces:**
- Consumes: the loaded `model` root and `params.mesh_report` (Plan B Task 1) when
  present.
- Produces: `modelStats(root) -> {triangles, vertices, size: {x, y, z}}` in `app.js`.

- [ ] **Step 1: Add the toolbar**

In `static/index.html`, inside the viewer container:

```html
  <div id="viewer-tools">
    <button type="button" id="tool-wire" title="Wireframe">wire</button>
    <button type="button" id="tool-spin" title="Turntable">spin</button>
    <button type="button" id="tool-frame" title="Reframe (F)">frame</button>
    <span id="tool-stats"></span>
  </div>
```

```css
    #viewer-tools { position: absolute; left: 8px; bottom: 8px; display: flex;
                    gap: 6px; align-items: center; font-size: 11px; color: #8b8fa3; }
    #viewer-tools button.on { color: #7c6cf0; }
```

- [ ] **Step 2: Implement the stats**

In `static/app.js`:

```js
// Counted off the loaded scene rather than read from mesh_report: the report
// describes model.glb, and the viewer may be showing rig.glb or a baked pose.
// The number under the model should describe the model under it.
function modelStats(root) {
  let triangles = 0;
  let vertices = 0;
  root.traverse((o) => {
    const g = o.isMesh ? o.geometry : null;
    if (!g) return;
    const position = g.getAttribute("position");
    vertices += position ? position.count : 0;
    triangles += g.index ? g.index.count / 3 : (position ? position.count / 3 : 0);
  });
  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  return { triangles: Math.round(triangles), vertices, size };
}

function updateStats(root) {
  const el = document.getElementById("tool-stats");
  if (!root) { setText(el, ""); return; }
  const s = modelStats(root);
  setText(
    el,
    `${s.triangles.toLocaleString()} tris · ${s.vertices.toLocaleString()} verts · ` +
      `${s.size.x.toFixed(2)} × ${s.size.y.toFixed(2)} × ${s.size.z.toFixed(2)} m`
  );
}
```

Call `updateStats(model)` at the end of `showModel`'s ready path, and
`updateStats(null)` in `disposeModel`'s caller when the viewer is cleared.

- [ ] **Step 3: Wireframe and turntable**

```js
let wireframe = false;
document.getElementById("tool-wire").addEventListener("click", (e) => {
  wireframe = !wireframe;
  e.currentTarget.classList.toggle("on", wireframe);
  if (!model) return;
  model.traverse((o) => {
    if (!o.isMesh) return;
    // An array covers multi-material meshes, which a joined trellis export can
    // still be after an OBJ round-trip.
    for (const m of [].concat(o.material)) m.wireframe = wireframe;
  });
});

let spinning = false;
document.getElementById("tool-spin").addEventListener("click", (e) => {
  spinning = !spinning;
  e.currentTarget.classList.toggle("on", spinning);
  // OrbitControls' own autoRotate, not a manual rotation of the model: rotating
  // the model would move the thing the pose gizmo and the joint markers are
  // positioned against.
  controls.autoRotate = spinning;
});
```

`controls.autoRotate` needs `controls.update()` called every frame — check the
existing animation loop already calls it (it does, for damping) and add
`controls.autoRotate = false` to `enterPoseEditing` so a spinning turntable cannot
fight a gizmo drag.

- [ ] **Step 4: Reframe**

Extract whatever framing `showModel` already does into a named `frameModel()` and
bind it to both the button and the `F` key (see Task 5's shortcut table).

- [ ] **Step 5: Verify by hand and commit**

Load a mesh, toggle wireframe, start the turntable, confirm the stats line reads
plausible triangle counts and metre dimensions.

```bash
uv run ruff check .
git add src/warlock/static
git commit -m "Warlock v0.0.1

Add wireframe, turntable, reframe and a stats readout to the viewer."
```

---

### Task 5: Input and feedback QoL (review item #24)

Drag-drop and paste for image jobs, keyboard shortcuts, toasts replacing the eight
raw `alert()` calls, a browser notification when a long job finishes, and form
state that survives a reload.

**Files:**
- Modify: `src/warlock/static/index.html`, `src/warlock/static/app.js`
- Test: manual (frontend only)

**Interfaces:**
- Produces: `toast(message, kind = "error")` in `app.js`, replacing every `alert()`.
- Produces: `saveFormState()` / `restoreFormState()` backed by `localStorage`.
- Consumes: nothing.

- [ ] **Step 1: Add the toast host and replace every alert**

In `static/index.html`, before `</body>`:

```html
  <div id="toasts"></div>
```

```css
    #toasts { position: fixed; right: 12px; bottom: 12px; display: flex;
              flex-direction: column; gap: 6px; z-index: 50; }
    .toast { padding: 8px 12px; border-radius: 6px; background: #23252e;
             border-left: 3px solid #7c6cf0; font-size: 12px; max-width: 320px; }
    .toast.error { border-left-color: #e0574a; }
    .toast.ok { border-left-color: #4cc38a; }
```

In `static/app.js`:

```js
// alert() blocks the event loop, which in a page that polls every 600 ms means
// the progress bar freezes behind the dialog the user has to dismiss to see it.
function toast(message, kind = "error") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  setText(el, message);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 8000 : 4000);
}
```

Replace all eight `alert(...)` call sites (grep: `grep -n "alert(" src/warlock/static/app.js`)
with `toast(...)`. They are at roughly lines 393, 506, 529, 963, 1196 and the
remainder in the pose/sheet handlers.

- [ ] **Step 2: Drag-drop and paste for image jobs**

```js
// The image tab was a bare <input type="file">. Dropping a reference or pasting
// one from a screenshot tool is how people actually get an image into this.
const imageInput = document.getElementById("image");
const dropZone = document.getElementById("image-input");

function acceptImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  imageInput.files = dt.files;
  kind = "image";
  tabs.image.click();
  toast(`Using ${file.name || "pasted image"}`, "ok");
}

for (const type of ["dragover", "dragenter"]) {
  dropZone.addEventListener(type, (e) => { e.preventDefault(); dropZone.classList.add("drop"); });
}
for (const type of ["dragleave", "drop"]) {
  dropZone.addEventListener(type, () => dropZone.classList.remove("drop"));
}
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  acceptImage(e.dataTransfer?.files?.[0]);
});
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
  if (item) acceptImage(item.getAsFile());
});
```

- [ ] **Step 3: Keyboard shortcuts**

```js
// Deliberately few, and none that fire while typing: a prompt textarea that
// eats Escape or F would be worse than having no shortcuts at all.
window.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName);
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    document.getElementById("form").requestSubmit();
    return;
  }
  if (typing) return;
  if (e.key === "Escape" && poseState.editing) exitPoseEditing();
  if (e.key === "f" || e.key === "F") frameModel();
});
```

- [ ] **Step 4: Notify when a long job finishes**

In the poll handler, where a job's status transitions to `done` or `error`:

```js
// Only for jobs that ran long enough that the user plausibly went elsewhere --
// a notification for a 4-second reference render is noise.
const NOTIFY_AFTER_SECONDS = 45;

function maybeNotify(job) {
  const ran = (job.finished_at ?? 0) - (job.started_at ?? 0);
  if (ran < NOTIFY_AFTER_SECONDS) return;
  const title = job.name || job.prompt || job.id;
  if (Notification?.permission === "granted") {
    new Notification(job.status === "done" ? "Model ready" : "Job failed", { body: title });
  }
  toast(`${job.status === "done" ? "Finished" : "Failed"}: ${title}`,
        job.status === "done" ? "ok" : "error");
}
```

Track the previous status per id in a `Map` and call `maybeNotify` only on the
transition. Ask for permission once, on the first successful submit, not on page
load — an unprompted permission dialog on arrival is the thing everyone dismisses:

```js
  if (Notification?.permission === "default") Notification.requestPermission();
```

- [ ] **Step 5: Persist form state**

```js
// Every setting except the seed, which is deliberately rerolled per submit.
const FORM_STATE_KEY = "warlock.form.v1";

function saveFormState() {
  const state = { prompt: document.getElementById("prompt").value };
  for (const field of GUIDANCE_FIELDS) state[field] = guidanceSelects[field]?.value ?? "";
  state.size_m = sizeInput.value;
  state.lora_weight = loraWeight.value;
  state.negative_prompt = document.getElementById("negative-prompt")?.value ?? "";
  localStorage.setItem(FORM_STATE_KEY, JSON.stringify(state));
}

function restoreFormState() {
  let state;
  try {
    state = JSON.parse(localStorage.getItem(FORM_STATE_KEY) || "null");
  } catch {
    return;   // a corrupt blob costs the restore, not the page
  }
  if (!state) return;
  if (state.prompt) document.getElementById("prompt").value = state.prompt;
  for (const field of GUIDANCE_FIELDS) {
    const select = guidanceSelects[field];
    // Only if the option still exists -- a stored model key can outlive a
    // registry entry, and setting a select to a missing value blanks it.
    if (select && state[field] && [...select.options].some((o) => o.value === state[field])) {
      select.value = state[field];
    }
  }
  if (state.size_m) { sizeInput.value = state.size_m; sizeEdited = true; }
  if (state.lora_weight) { loraWeight.value = state.lora_weight; syncLoraWeight(); }
  const negative = document.getElementById("negative-prompt");
  if (negative && state.negative_prompt) negative.value = state.negative_prompt;
  syncPlatformHint();
}
```

Call `restoreFormState()` at the end of `loadGuidance()` (the selects must be
populated first) and `saveFormState()` on every `change` of the form and on submit.

- [ ] **Step 6: Verify by hand and commit**

Paste an image from the clipboard, submit with Ctrl+Enter, trigger an error and
confirm it appears as a toast rather than a dialog, reload and confirm the form
comes back.

```bash
uv run ruff check .
git add src/warlock/static
git commit -m "Warlock v0.0.1

Toasts, drag-drop and paste, shortcuts, notifications and persisted form state."
```

---

### Task 6: Prompt and recipe presets (review item #25)

No prompt history or saved recipes exist. A small shipped preset library plus
auto-saved history makes a style reproducible across an asset set.

**Files:**
- Modify: `src/warlock/guidance.py` (shipped presets in the catalog)
- Modify: `src/warlock/static/index.html`, `src/warlock/static/app.js`
- Test: `tests/test_guidance.py`

**Interfaces:**
- Consumes: `copySettingsToForm` (Plan A Task 3) — the same function fills the form
  from a preset as from a past job.
- Produces: `guidance.PRESETS: tuple[dict[str, Any], ...]`, surfaced at
  `catalog()["presets"]`. Each entry is `{"key", "label", "prompt", "fields": {...}}`
  where `fields` holds guidance keys only.
- Produces: `localStorage` history under `warlock.history.v1`, capped at 20 entries.

**Why the presets ship server-side:** they name guidance keys and model keys, and
`guidance.normalize` is what decides whether those are valid. A preset defined in
JS could name a LoRA that was removed from the registry and would fail at submit
with a 400 the user cannot interpret; defined here, the same test suite that covers
the taxonomy covers the presets.

- [ ] **Step 1: Write the failing test**

In `tests/test_guidance.py`:

```python
def test_every_shipped_preset_normalizes():
    from warlock import guidance

    assert guidance.PRESETS
    for preset in guidance.PRESETS:
        # If a preset names a taxonomy or model key that has been renamed or
        # removed, this is where it fails -- not in the user's browser as an
        # uninterpretable 400 at submit time.
        guidance.normalize(dict(preset["fields"]))
        assert preset["prompt"]
        assert preset["label"]


def test_presets_appear_in_the_catalog():
    from warlock import guidance

    keys = {p["key"] for p in guidance.catalog()["presets"]}
    assert "handpainted_prop" in keys
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_guidance.py -k preset -v`
Expected: FAIL — `AttributeError: PRESETS`

- [ ] **Step 3: Define the presets**

In `src/warlock/guidance.py`, after the option tables:

```python
# Whole recipes: a prompt skeleton plus every guidance field that makes the
# style land. Defined here rather than in the browser because the fields name
# taxonomy and model keys, and normalize() is what decides whether those are
# still valid -- a preset that names a removed LoRA should fail this module's
# tests, not a user's submit.
PRESETS: tuple[dict[str, Any], ...] = (
    {
        "key": "handpainted_prop",
        "label": "Hand-painted fantasy prop",
        "prompt": "a weathered wooden crate bound with iron",
        "fields": {
            "category": "prop",
            "genre": "fantasy",
            "art_style": "handpainted",
            "platform": "desktop",
            "base_model": "sdxl",
            "style_lora": "render3d",
        },
    },
    {
        "key": "ps1_character",
        "label": "PS1 low-poly character",
        "prompt": "a hooded adventurer standing in a neutral pose",
        "fields": {
            "category": "character",
            "genre": "fantasy",
            "art_style": "lowpoly",
            "platform": "mobile",
            "base_model": "sdxl",
            "style_lora": "ps1",
        },
    },
    {
        "key": "scifi_hero_weapon",
        "label": "Sci-fi hero weapon",
        "prompt": "a compact energy rifle with panel seams and glowing vents",
        "fields": {
            "category": "weapon",
            "genre": "scifi",
            "art_style": "realistic",
            "platform": "hero",
            "base_model": "playground",
        },
    },
    {
        "key": "modern_pickup",
        "label": "Modern consumable pickup",
        "prompt": "a small first-aid kit",
        "fields": {
            "category": "consumable",
            "genre": "modern",
            "art_style": "stylized",
            "platform": "mobile",
            "base_model": "turbo",
        },
    },
)
```

and add to `catalog()`'s returned dict:

```python
        "presets": [dict(p) for p in PRESETS],
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_guidance.py -v`
Expected: PASS

- [ ] **Step 5: Add the preset select and history**

In `static/index.html`, above the prompt textarea:

```html
      <div id="preset-row">
        <select id="preset"><option value="">preset…</option></select>
        <select id="history"><option value="">recent…</option></select>
      </div>
```

In `static/app.js`, inside `loadGuidance()` after the selects are populated:

```js
  const presetSelect = document.getElementById("preset");
  for (const preset of data.presets ?? []) {
    const opt = document.createElement("option");
    opt.value = preset.key;
    setText(opt, preset.label);
    presetSelect.append(opt);
  }
  presets = data.presets ?? [];
```

and:

```js
let presets = [];

document.getElementById("preset").addEventListener("change", (e) => {
  const preset = presets.find((p) => p.key === e.target.value);
  if (!preset) return;
  // Reuses the same filler a finished job's "copy settings" uses, so a preset
  // and a past recipe land in the form by exactly one code path.
  copySettingsToForm({ prompt: preset.prompt, params: preset.fields });
  e.target.value = "";
  saveFormState();
});

// --- submission history -----------------------------------------------------

const HISTORY_KEY = "warlock.history.v1";
const HISTORY_MAX = 20;

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function recordSubmission(prompt, fields) {
  if (!prompt) return;
  const history = readHistory().filter((h) => h.prompt !== prompt);
  history.unshift({ prompt, fields, at: Date.now() });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, HISTORY_MAX)));
  renderHistory();
}

function renderHistory() {
  const select = document.getElementById("history");
  select.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  setText(blank, "recent…");
  select.append(blank);
  readHistory().forEach((entry, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    setText(opt, entry.prompt.slice(0, 60));
    select.append(opt);
  });
}

document.getElementById("history").addEventListener("change", (e) => {
  const entry = readHistory()[Number(e.target.value)];
  if (!entry) return;
  copySettingsToForm({ prompt: entry.prompt, params: entry.fields });
  e.target.value = "";
  saveFormState();
});
```

Call `recordSubmission(prompt, fields)` in the submit handler's success branch,
where `fields` is the same object the form built from `GUIDANCE_FIELDS`, and
`renderHistory()` once at startup.

- [ ] **Step 6: Verify by hand and commit**

Pick "PS1 low-poly character", confirm the prompt and every select fill in, submit,
then confirm it appears under "recent…" and refills correctly after a reload.

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Ship prompt/recipe presets and a local submission history."
```

---

## Plan D self-review notes

- Items #20–#25 each map to a task: #20→T1, #21→T2, #22→T3, #23→T4, #24→T5, #25→T6.
- Task 6 depends on `copySettingsToForm` from **Plan A Task 3**. If Plan D runs first,
  implement that function as part of Task 6 Step 5 — the body is given in Plan A
  Task 3 Step 3 and can be lifted verbatim.
- Task 5's `saveFormState` references `#negative-prompt` (Plan A Task 2). The
  optional-chaining guards handle its absence, so the order does not matter.
- Task 2 changes one line of the shared renderer construction
  (`preserveDrawingBuffer: true`); that is the only global side effect in this plan.
- Task 1's migration must be **appended** to `MIGRATIONS`, never inserted — check
  `len(db.MIGRATIONS)` first.
