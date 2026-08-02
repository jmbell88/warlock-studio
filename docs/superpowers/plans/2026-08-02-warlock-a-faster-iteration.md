# Plan A — Faster Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop paying a multi-minute trellis run to find out the SDXL reference image was wrong.

**Architecture:** Three unrelated cheap wins first (background-removal forwarding, negative prompt, showing the settings a job actually ran with), then the structural change: a text job can stop after `input.png`, and a separate promote call runs the 3D stage from an approved reference. The queue stays a single serial worker and a reference job is just a job whose `stage` is `reference`; nothing about cancellation, progress or VRAM changes.

**Tech Stack:** FastAPI, sqlite3, diffusers, plain-JS frontend (no build step).

## Global Constraints

See `2026-08-02-warlock-review-index.md` § Global Constraints. Every task's
requirements implicitly include that section. In particular: `uv run pytest -q`
and `uv run ruff check .` green before every commit; `JobStore._lock` on every
method touching `self._conn`; VRAM stop-before-load preserved under
`vram_exclusive`.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/warlock/pipelines/trellis.py` | trellis-server subprocess + `/generate` client | Modify: forward `bg_removal` |
| `src/warlock/guidance.py` | validated job params, prompt composition | Modify: `bg_removal`, `negative_prompt` |
| `src/warlock/pipelines/text2image.py` | diffusers pipeline | Modify: negative prompt |
| `src/warlock/db.py` | job store | Modify: first `MIGRATIONS` entry, `stage`/`parent_id` |
| `src/warlock/app.py` | HTTP routes | Modify: `output=`, `POST /{id}/model`, `count=` |
| `src/warlock/queue.py` | serial GPU worker | Modify: stop at reference, split seeds, batch |
| `src/warlock/static/index.html` + `app.js` | UI | Modify: new controls, settings panel, approve bar |

---

### Task 1: Forward `bg_removal` to trellis-server (review item #6)

`trellis-server.exe`'s `/generate` accepts a `bg_removal` form field
(`auto|threshold|birefnet`) that `trellis.py:220` never sends. A bad cutout is a
top cause of garbage meshes and `doctor._birefnet_check` already reports whether
`birefnet.gguf` is present.

**Files:**
- Modify: `src/warlock/pipelines/trellis.py:204-228`
- Modify: `src/warlock/guidance.py` (add `BG_REMOVAL`, validate in `normalize`)
- Modify: `src/warlock/app.py:136-207` (form field), `src/warlock/queue.py:436-449`
- Modify: `src/warlock/static/index.html`, `src/warlock/static/app.js`
- Test: `tests/test_trellis.py`, `tests/test_guidance.py`, `tests/test_queue.py`

**Interfaces:**
- Produces: `TrellisServer.generate(image_path, output_path, *, seed=42, resolution=1024, bg_removal: str | None = None) -> Path`
- Produces: `guidance.BG_REMOVAL: tuple[str, ...] = ("auto", "birefnet", "threshold")`;
  `normalize()` output carries `"bg_removal": str` always (default `"auto"`).
- Consumes (Task 7 and Plan B): the same `params["bg_removal"]` key.

- [ ] **Step 1: Write the failing test for the client**

In `tests/test_trellis.py`, add:

```python
def test_generate_forwards_bg_removal(tmp_path, monkeypatch):
    """The exe accepts a bg_removal form field; we never used to send it."""
    import warlock.pipelines.trellis as trellis_mod

    sent = {}

    class FakeResponse:
        status_code = 200
        content = b"glb"
        text = ""

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, files=None, data=None):
            sent.update(data)
            return FakeResponse()

    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", FakeClient)
    server = trellis_mod.TrellisServer(tmp_path / "x.exe", tmp_path, 1234)
    monkeypatch.setattr(server, "ensure_started", _noop_async)
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    asyncio.run(
        server.generate(image, tmp_path / "out.glb", seed=7, bg_removal="birefnet")
    )
    assert sent["bg_removal"] == "birefnet"
    assert sent["seed"] == "7"


async def _noop_async(*_a, **_k):
    return None
```

Add `import asyncio` at the top of the file if it is not already imported.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_trellis.py::test_generate_forwards_bg_removal -v`
Expected: FAIL — `generate() got an unexpected keyword argument 'bg_removal'`

- [ ] **Step 3: Add the parameter**

In `src/warlock/pipelines/trellis.py`, change `generate`:

```python
    async def generate(
        self,
        image_path: Path,
        output_path: Path,
        *,
        seed: int = 42,
        resolution: int = 1024,
        bg_removal: str | None = None,
    ) -> Path:
        """Run image -> 3D and write the returned GLB to output_path.

        ``bg_removal`` picks how the server mattes the input: birefnet is the
        learned matte (needs birefnet.gguf, see doctor), threshold is the cheap
        cutout, auto lets the server decide. Omitted entirely when None so the
        exe applies its own default rather than being handed a keyword it may
        not know.
        """
        await self.ensure_started()
        self.last_used = time.monotonic()
        data = {"seed": str(seed), "resolution": str(resolution)}
        if bg_removal is not None:
            data["bg_removal"] = bg_removal
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            with image_path.open("rb") as fh:
                r = await client.post(
                    f"{self.base_url}/generate",
                    files={"image": (image_path.name, fh)},
                    data=data,
                )
```

Leave the rest of the method unchanged.

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_trellis.py::test_generate_forwards_bg_removal -v`
Expected: PASS

- [ ] **Step 5: Write the failing guidance test**

In `tests/test_guidance.py`:

```python
def test_bg_removal_defaults_to_auto_and_rejects_unknown():
    from warlock import guidance

    assert guidance.normalize({})["bg_removal"] == "auto"
    assert guidance.normalize({"bg_removal": "birefnet"})["bg_removal"] == "birefnet"
    with pytest.raises(ValueError):
        guidance.normalize({"bg_removal": "magic"})
```

- [ ] **Step 6: Run it and watch it fail**

Run: `uv run pytest tests/test_guidance.py::test_bg_removal_defaults_to_auto_and_rejects_unknown -v`
Expected: FAIL — `KeyError: 'bg_removal'`

- [ ] **Step 7: Validate it in guidance**

In `src/warlock/guidance.py`, after `DEFAULT_PLATFORM`:

```python
# How trellis-server mattes the input image. Not an Option table: these are
# server capabilities, not prompt fragments, so they never reach compose_prompt.
BG_REMOVAL = ("auto", "birefnet", "threshold")
DEFAULT_BG_REMOVAL = "auto"
```

In `normalize`, before building `out`:

```python
    bg_removal = raw.get("bg_removal")
    if bg_removal in (None, ""):
        bg_removal = DEFAULT_BG_REMOVAL
    elif str(bg_removal) not in BG_REMOVAL:
        raise ValueError(f"bg_removal must be one of {list(BG_REMOVAL)}")
```

and add `"bg_removal": str(bg_removal),` to the `out` dict literal.

In `catalog()`, add to the returned dict (beside `"size_range_m"`):

```python
        "bg_removal": list(BG_REMOVAL),
```

and add `"bg_removal": DEFAULT_BG_REMOVAL,` to the `"defaults"` dict.

- [ ] **Step 8: Run it**

Run: `uv run pytest tests/test_guidance.py -v`
Expected: PASS

- [ ] **Step 9: Write the failing worker test**

In `tests/test_queue.py`:

```python
@pytest.mark.asyncio
async def test_worker_forwards_bg_removal_to_trellis(worker_env):
    worker, store = worker_env
    job_id = store.create("image", None, {"seed": 1, "bg_removal": "birefnet"})
    (worker.config.job_dir(job_id)).mkdir(parents=True, exist_ok=True)
    (worker.config.job_dir(job_id) / "input.png").write_bytes(b"png")
    await _run_until_done(worker, store, job_id)
    assert worker.trellis.generate_calls[-1]["bg_removal"] == "birefnet"
```

Reuse whatever fixture/helper `tests/test_queue.py` already uses to drive one job
to completion; if the names differ, match the existing file rather than
introducing `worker_env`/`_run_until_done`. Also extend `FakeTrellisServer.generate`
in `tests/conftest.py` to accept and record it:

```python
    async def generate(
        self,
        image_path: Path,
        output_path: Path,
        *,
        seed: int = 42,
        resolution: int = 1024,
        bg_removal: str | None = None,
    ) -> Path:
        self.generate_calls.append(
            {
                "image_path": image_path,
                "seed": seed,
                "resolution": resolution,
                "bg_removal": bg_removal,
            }
        )
```

`tests/test_fakes_match_real_signatures.py` exists precisely to catch a fake
drifting from the real class — it will fail until both sides match.

- [ ] **Step 10: Run it and watch it fail**

Run: `uv run pytest tests/test_queue.py -k bg_removal -v`
Expected: FAIL — `bg_removal` is `None`

- [ ] **Step 11: Pass it from the worker**

In `src/warlock/queue.py:447`:

```python
        await self.trellis.generate(
            image_path,
            glb_path,
            seed=seed,
            resolution=resolution,
            bg_removal=str(params.get("bg_removal") or "auto"),
        )
```

- [ ] **Step 12: Run the suite**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 13: Add the UI control**

In `src/warlock/static/index.html`, after the `g-platform-row` div:

```html
      <div id="g-bg_removal-row">
        <label for="g-bg_removal">Background removal</label>
        <select id="g-bg_removal" data-guidance="bg_removal"></select>
      </div>
```

In `src/warlock/static/app.js`, add `"bg_removal"` to `GUIDANCE_FIELDS` (line 205)
and, inside `loadGuidance()`, populate it from the new catalog key. The catalog
returns a bare list of strings here rather than `{key,label}` objects, so build the
options explicitly:

```js
  const bgSelect = document.getElementById("g-bg_removal");
  for (const key of data.bg_removal ?? []) {
    const opt = document.createElement("option");
    opt.value = key;
    setText(opt, key === "birefnet" ? "BiRefNet (best)" : key === "threshold" ? "Threshold (fast)" : "Auto");
    bgSelect.append(opt);
  }
  bgSelect.value = data.defaults?.bg_removal ?? "auto";
  guidanceSelects.bg_removal = bgSelect;
```

The existing submit handler loops over `GUIDANCE_FIELDS` and sends any select with
a value, so no submit change is needed. Unlike genre/art_style this row must stay
visible for image jobs — do **not** add it to the hidden-for-image list at
`app.js:309`.

- [ ] **Step 14: Commit**

```bash
git add src/warlock/pipelines/trellis.py src/warlock/guidance.py src/warlock/queue.py src/warlock/app.py src/warlock/static tests/
git commit -m "Warlock v0.0.1

Forward bg_removal to trellis-server and expose it in the UI."
```

---

### Task 2: Negative prompt (review item #4)

No negative-prompt field exists anywhere. For game assets a default of
"blurry, multiple objects, cropped, watermark" cuts bad references substantially.

**Files:**
- Modify: `src/warlock/pipelines/text2image.py:30-35, 196-249`
- Modify: `src/warlock/guidance.py` (validate + store), `src/warlock/queue.py:410-424`
- Modify: `src/warlock/app.py` (form field), `static/index.html`, `static/app.js`
- Test: `tests/test_guidance.py`, `tests/test_queue.py`

**Interfaces:**
- Consumes: `guidance.normalize` from Task 1.
- Produces: `params["negative_prompt"]: str` (may be empty);
  `Text2Image.generate(..., negative_prompt: str | None = None)`;
  `guidance.DEFAULT_NEGATIVE_PROMPT: str`.

- [ ] **Step 1: Write the failing test**

In `tests/test_guidance.py`:

```python
def test_negative_prompt_defaults_and_is_length_capped():
    from warlock import guidance

    assert guidance.normalize({})["negative_prompt"] == guidance.DEFAULT_NEGATIVE_PROMPT
    assert guidance.normalize({"negative_prompt": " smooth "})["negative_prompt"] == "smooth"
    with pytest.raises(ValueError):
        guidance.normalize({"negative_prompt": "x" * 1001})
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_guidance.py -k negative -v`
Expected: FAIL — `AttributeError: DEFAULT_NEGATIVE_PROMPT`

- [ ] **Step 3: Implement it in guidance**

In `src/warlock/guidance.py`, beside `BG_REMOVAL`:

```python
# What a TRELLIS reference image must not be. A second subject or a cropped one
# is the single most common cause of a mesh that reconstructs into nonsense, so
# this is a default rather than an empty field the user has to discover.
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, multiple objects, cropped, cut off, "
    "text, watermark, signature, busy background, human hands"
)
MAX_NEGATIVE_PROMPT = 1000
```

In `normalize`, before building `out`:

```python
    negative = raw.get("negative_prompt")
    if negative is None:
        negative = DEFAULT_NEGATIVE_PROMPT
    negative = str(negative).strip()
    if len(negative) > MAX_NEGATIVE_PROMPT:
        raise ValueError(f"negative_prompt must be at most {MAX_NEGATIVE_PROMPT} characters")
```

Add `"negative_prompt": negative,` to `out`. Note the asymmetry with
`bg_removal`: an explicitly empty string here means "no negative prompt at all"
and is honoured, while a missing key means "use the default". Add
`"negative_prompt": DEFAULT_NEGATIVE_PROMPT,` to `catalog()["defaults"]`.

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_guidance.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing pipeline test**

In `tests/test_queue.py`:

```python
@pytest.mark.asyncio
async def test_worker_passes_negative_prompt(worker_env):
    worker, store = worker_env
    job_id = store.create(
        "text", "a barrel", {"seed": 1, "negative_prompt": "blurry, two objects"}
    )
    await _run_until_done(worker, store, job_id)
    assert worker._text2image.negatives[-1] == "blurry, two objects"
```

Add to `FakeText2Image` in `tests/conftest.py`: `self.negatives: list[str | None] = []`
in `__init__`, a `negative_prompt=None` keyword on `generate`, and
`self.negatives.append(negative_prompt)` in the body.

- [ ] **Step 6: Run it and watch it fail**

Run: `uv run pytest tests/test_queue.py -k negative_prompt -v`
Expected: FAIL — unexpected keyword argument

- [ ] **Step 7: Implement in text2image and the worker**

In `src/warlock/pipelines/text2image.py`, add the parameter to `generate`:

```python
        negative_prompt: str | None = None,
```

and pass it to the pipe call:

```python
        image = self._pipe(
            text,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
```

A distilled model at `guidance_scale=0.0` ignores the negative prompt entirely —
that is a property of CFG, not a bug here. Note it in the docstring:

```python
        """...

        ``negative_prompt`` only bites when the checkpoint runs with CFG: a
        4-step distilled base at guidance_scale 0 discards it, which is why the
        UI notes it applies to the CFG bases (playground) rather than silently
        doing nothing.
        """
```

In `src/warlock/queue.py`, inside the `functools.partial(t2i.generate, ...)` call
at line 411, add:

```python
                        negative_prompt=str(params.get("negative_prompt") or ""),
```

- [ ] **Step 8: Run it**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 9: Add the API form field and UI**

In `src/warlock/app.py:136`, add to `create_job`'s signature:

```python
        negative_prompt: Annotated[str | None, Form()] = None,
```

and add `"negative_prompt": negative_prompt,` to the dict passed to
`guidance.normalize`.

In `static/index.html`, under the prompt textarea:

```html
      <div id="negative-row">
        <label for="negative-prompt">Negative prompt</label>
        <textarea id="negative-prompt" rows="2"></textarea>
        <p class="hint">Only applies to CFG models (Playground); distilled 4-step bases ignore it.</p>
      </div>
```

In `app.js`, populate its default from the guidance catalog inside `loadGuidance()`:

```js
  const negative = document.getElementById("negative-prompt");
  if (!negative.value) negative.value = data.defaults?.negative_prompt ?? "";
```

and in the submit handler, alongside the prompt:

```js
    fd.set("negative_prompt", document.getElementById("negative-prompt").value);
```

Hide `negative-row` for image jobs the same way `g-genre-row` is hidden (add
`"negative"` handling in the tab click handler at `app.js:309` by also toggling
`document.getElementById("negative-row").hidden = k !== "text"`).

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "Warlock v0.0.1

Add a negative prompt with a game-asset default."
```

---

### Task 3: Show the composed prompt and the settings a job ran with (review item #5)

`params.composed_prompt`, `seed`, `base_model`, `style_lora`, `resolution` and
`size_m` are already stored and already returned by `/api/jobs`. The job cards
render only the raw prompt. This task is frontend-only.

**Files:**
- Modify: `src/warlock/static/app.js:456-627` (`createNode`, `updateNode`)
- Modify: `src/warlock/static/index.html` (styles only, if needed)
- Test: manual — there is no browser test harness in this repo and adding one is
  out of scope (see the no-build-step invariant).

**Interfaces:**
- Consumes: the existing `/api/jobs` response — no API change.
- Produces: a `copySettingsToForm(job)` function in `app.js`, reused by Plan D
  Task 6 (prompt presets).

- [ ] **Step 1: Add the settings panel elements in `createNode`**

In `src/warlock/static/app.js`, inside `createNode(id)`, after the `quality` div is
created and before `info.append(...)`:

```js
  const settings = document.createElement("div");
  settings.className = "job-settings";
  settings.hidden = true;
  const settingsToggle = document.createElement("button");
  settingsToggle.type = "button";
  settingsToggle.className = "link";
  setText(settingsToggle, "settings");
  settingsToggle.hidden = true;
```

Change the `info.append` line to:

```js
  info.append(title, status, stage, err, quality, settingsToggle, settings, bar);
```

Add the toggle handler beside the other `addEventListener` calls in `createNode`:

```js
  settingsToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    settings.hidden = !settings.hidden;
    setText(settingsToggle, settings.hidden ? "settings" : "hide settings");
  });
```

And extend the returned node object at the end of `createNode`:

```js
  const n = { li, img, title, status, stage, err, quality, settings, settingsToggle,
              bar, fill, act, reroll, remesh, rigBtn };
```

- [ ] **Step 2: Render the settings in `updateNode`**

Add this before the `const active = ...` line in `updateNode`:

```js
  // Everything below is already in the API response and was never shown: without
  // it a good result is not reproducible, because the card only ever said what
  // the user typed, not what was actually sent to the model.
  const p = job.params ?? {};
  const rows = [
    ["seed", p.seed],
    ["reference seed", p.reference_seed],
    ["mesh seed", p.mesh_seed],
    ["model", p.base_model],
    ["style", p.style_lora && `${p.style_lora} @ ${p.lora_weight ?? "?"}`],
    ["resolution", p.resolution],
    ["size", p.size_m && `${p.size_m} m`],
    ["background", p.bg_removal],
    ["prompt sent", p.composed_prompt],
    ["negative", p.negative_prompt],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");
  n.settingsToggle.hidden = rows.length === 0;
  n.settings.replaceChildren();
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "settings-row";
    const k = document.createElement("span");
    setText(k, label);
    const v = document.createElement("span");
    setText(v, String(value));
    row.append(k, v);
    n.settings.append(row);
  }
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "link";
  setText(copy, "copy settings to form");
  copy.addEventListener("click", (e) => {
    e.stopPropagation();
    copySettingsToForm(job);
  });
  if (rows.length) n.settings.append(copy);
```

`reference_seed`/`mesh_seed` are rendered here before Task 5 introduces them; the
filter drops them until they exist, so this needs no revisit.

- [ ] **Step 3: Implement `copySettingsToForm`**

Add near `loadGuidance` in `app.js`:

```js
// Refill the form from a finished job, so a recipe that worked is one click from
// being reused with a tweak. Only fields the form actually owns -- derived
// values (composed_prompt, scale_factor, mesh_audit) describe that run, not this
// one, exactly as the /rerun route already reasons.
function copySettingsToForm(job) {
  const p = job.params ?? {};
  if (job.prompt) document.getElementById("prompt").value = job.prompt;
  for (const field of GUIDANCE_FIELDS) {
    const select = guidanceSelects[field];
    if (select && p[field]) select.value = p[field];
  }
  if (p.size_m) {
    sizeInput.value = p.size_m;
    sizeEdited = true;
  }
  if (p.lora_weight !== undefined) {
    loraWeight.value = p.lora_weight;
    syncLoraWeight();
  }
  if (p.negative_prompt !== undefined) {
    document.getElementById("negative-prompt").value = p.negative_prompt;
  }
  if (p.seed !== undefined) seedInput.value = p.seed;
  syncPlatformHint();
}
```

- [ ] **Step 4: Add the styles**

In `static/index.html`'s `<style>` block:

```css
    .job-settings { font-size: 11px; color: #8b8fa3; margin-top: 4px; }
    .settings-row { display: flex; justify-content: space-between; gap: 8px; }
    .settings-row span:last-child { color: #c9cbd6; text-align: right; word-break: break-word; }
    button.link { background: none; border: 0; color: #7c6cf0; cursor: pointer; padding: 0; font-size: 11px; }
```

- [ ] **Step 5: Verify by hand**

Run: `uv run warlock serve` (see `src/warlock/cli.py` for the exact command name),
open the UI, submit a text job, and confirm the card shows a "settings" toggle
whose panel lists the composed prompt and that "copy settings to form" refills the
form. Note in the commit message that this was verified manually.

- [ ] **Step 6: Commit**

```bash
git add src/warlock/static
git commit -m "Warlock v0.0.1

Show each job's composed prompt and settings, with copy-to-form."
```

---

### Task 4: Migration machinery, `stage` and `parent_id` (review item #1, part 1)

`db.MIGRATIONS` is an empty list with a documented contract and no entries. This
task writes the first one. It is a prerequisite for Tasks 6 and 7 and for Plan D
Task 1.

**Files:**
- Modify: `src/warlock/db.py:13-41, 67-90`
- Test: `tests/test_migrate.py`, `tests/test_db.py`

**Interfaces:**
- Produces: `jobs.stage TEXT NOT NULL DEFAULT 'model'`, `jobs.parent_id TEXT NULL`;
  `JobStore.create(kind, prompt, params, job_id=None, *, stage="model", parent_id=None) -> str`;
  `JobStore.children(parent_id: str) -> list[dict]`;
  `JobStore.set_stage(job_id: str, stage: str) -> None`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing migration test**

In `tests/test_migrate.py`:

```python
def test_pre_migration_db_gains_stage_and_parent(tmp_path):
    """A DB created before the columns existed must converge on the same shape
    as a fresh one, with existing rows defaulted to stage='model'."""
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
        " VALUES ('a'||substr('000000000000',1,11), 'text', 'done', '{}', 1.0)"
    )
    conn.commit()
    conn.close()

    store = JobStore(path)
    try:
        row = store.list(10)[0]
        assert row["stage"] == "model"
        assert row["parent_id"] is None
    finally:
        store.close()


def test_create_records_stage_and_parent(store):
    parent = store.create("text", "a barrel", {}, stage="reference")
    child = store.create("text", "a barrel", {}, parent_id=parent)
    assert store.get(parent)["stage"] == "reference"
    assert [c["id"] for c in store.children(parent)] == [child]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_migrate.py -v`
Expected: FAIL — `KeyError: 'stage'`

- [ ] **Step 3: Add the migration and the columns**

In `src/warlock/db.py`, add the columns to `_SCHEMA`'s `CREATE TABLE` so a fresh DB
has them from the start:

```sql
    finished_at REAL,
    stage       TEXT NOT NULL DEFAULT 'model',  -- 'reference' | 'model'
    parent_id   TEXT                            -- the reference job this was promoted from
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_id);
```

and append the first migration entry, which brings a *pre-existing* DB to the same
shape. `ADD COLUMN` on a table that already has it raises, so each statement is
guarded by the `user_version` counter alone — that is exactly what the
append-only contract buys, so no `IF NOT EXISTS` dance is needed:

```python
MIGRATIONS: list[list[str]] = [
    # 1 -- approve-reference-first. A row that predates the split was a
    # single-stage generate, which is what stage='model' means.
    [
        "ALTER TABLE jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'model'",
        "ALTER TABLE jobs ADD COLUMN parent_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_id)",
    ],
]
```

A fresh DB runs `_SCHEMA` (which already has the columns) and then replays this
entry, which would fail. Guard `_migrate` by checking the live column set:

```python
def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    columns = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    for i in range(version, len(MIGRATIONS)):
        for stmt in MIGRATIONS[i]:
            # A fresh DB got these columns from _SCHEMA; replaying the ALTER
            # would fail on it. Skipping the statement rather than the whole
            # entry keeps fresh and migrated DBs converging, which is the
            # property the append-only contract exists to protect.
            if stmt.startswith("ALTER TABLE jobs ADD COLUMN"):
                name = stmt.split("ADD COLUMN", 1)[1].split()[0]
                if name in columns:
                    continue
            conn.execute(stmt)
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()
```

- [ ] **Step 4: Extend `create`, and add `children` / `set_stage`**

```python
    def create(
        self,
        kind: str,
        prompt: str | None,
        params: dict[str, Any],
        job_id: str | None = None,
        *,
        stage: str = "model",
        parent_id: str | None = None,
    ) -> str:
        job_id = job_id or uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, prompt, params, created_at,"
                " stage, parent_id) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)",
                (job_id, kind, prompt, json.dumps(params), time.time(), stage, parent_id),
            )
            self._conn.commit()
        return job_id

    def set_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE jobs SET stage = ? WHERE id = ?", (stage, job_id))
            self._conn.commit()

    def children(self, parent_id: str) -> list[dict[str, Any]]:
        """Every job promoted from this one, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE parent_id = ? ORDER BY created_at", (parent_id,)
            ).fetchall()
        return [self._to_dict(r) for r in rows]
```

Both new methods take `self._lock` before touching `self._conn`, per the invariant.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_migrate.py tests/test_db.py -v`
Expected: PASS

- [ ] **Step 6: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 7: Commit**

```bash
git add src/warlock/db.py tests/
git commit -m "Warlock v0.0.1

Add the first schema migration: jobs.stage and jobs.parent_id."
```

---

### Task 5: Split `reference_seed` and `mesh_seed` (review item #3)

One `seed` feeds both stages (`app.py:182`, `queue.py:381`), so you cannot keep a
good reference and reroll only the mesh.

**Files:**
- Modify: `src/warlock/app.py:136-207, 288-343`
- Modify: `src/warlock/queue.py:371-451`
- Test: `tests/test_api.py`, `tests/test_queue.py`

**Interfaces:**
- Consumes: `JobStore.create(..., stage=, parent_id=)` from Task 4.
- Produces: `params["reference_seed"]: int` and `params["mesh_seed"]: int`, both
  always present on new jobs. Legacy `params["seed"]` is still read as the
  fallback for both and still written, so old rows and the existing
  `/rerun` route keep working unchanged.

- [ ] **Step 1: Write the failing test**

In `tests/test_api.py`:

```python
def test_seeds_split_and_legacy_seed_still_read(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "a barrel", "reference_seed": 11, "mesh_seed": 22},
    )
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["reference_seed"] == 11
    assert params["mesh_seed"] == 22

    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel", "seed": 5})
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["reference_seed"] == 5
    assert params["mesh_seed"] == 5
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_api.py -k seeds_split -v`
Expected: FAIL — `KeyError: 'reference_seed'`

- [ ] **Step 3: Accept both on the route**

In `src/warlock/app.py`, add to `create_job`'s signature after `seed`:

```python
        reference_seed: Annotated[int | None, Form()] = None,
        mesh_seed: Annotated[int | None, Form()] = None,
```

and replace `params["seed"] = seed` with:

```python
        # One seed used to drive both stages, so "keep this reference, try
        # another mesh" was impossible without also redrawing the image. seed
        # remains the fallback for both so old rows and old clients are
        # unchanged.
        params["seed"] = seed
        params["reference_seed"] = seed if reference_seed is None else reference_seed
        params["mesh_seed"] = seed if mesh_seed is None else mesh_seed
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_api.py -k seeds_split -v`
Expected: PASS

- [ ] **Step 5: Write the failing worker test**

In `tests/test_queue.py`:

```python
@pytest.mark.asyncio
async def test_worker_uses_each_stage_seed(worker_env):
    worker, store = worker_env
    job_id = store.create(
        "text", "a barrel", {"seed": 1, "reference_seed": 11, "mesh_seed": 22}
    )
    await _run_until_done(worker, store, job_id)
    assert worker.trellis.generate_calls[-1]["seed"] == 22
```

Assert the reference seed the same way `tests/test_queue.py` already inspects
`FakeText2Image` calls; if it does not record seeds yet, add
`self.seeds: list[int] = []` and `self.seeds.append(seed)` to the fake and assert
`worker._text2image.seeds[-1] == 11`.

- [ ] **Step 6: Run it and watch it fail**

Run: `uv run pytest tests/test_queue.py -k each_stage_seed -v`
Expected: FAIL — trellis got seed 1

- [ ] **Step 7: Read both in the worker**

In `src/warlock/queue.py:_generate`, replace `seed = int(params.get("seed", 42))` with:

```python
        # Two seeds, one per stage, falling back to the single legacy seed so a
        # job row written before the split still reproduces exactly.
        legacy_seed = int(params.get("seed", 42))
        reference_seed = int(params.get("reference_seed", legacy_seed))
        mesh_seed = int(params.get("mesh_seed", legacy_seed))
```

Use `seed=reference_seed` in the `t2i.generate` partial and `seed=mesh_seed` in the
`self.trellis.generate` call.

- [ ] **Step 8: Give the rerun route the split**

In `app.py:rerun_job`, replace `params["seed"] = seed if seed is not None else _random_seed()`
with:

```python
        fresh = seed if seed is not None else _random_seed()
        params["seed"] = fresh
        if mode == "remesh":
            # The reference is being reused verbatim; only the 3D stage rerolls.
            params["reference_seed"] = source["params"].get(
                "reference_seed", source["params"].get("seed", fresh)
            )
            params["mesh_seed"] = fresh
        else:
            params["reference_seed"] = fresh
            params["mesh_seed"] = fresh
```

- [ ] **Step 9: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "Warlock v0.0.1

Split reference_seed and mesh_seed, keeping legacy seed as the fallback."
```

---

### Task 6: Approve-reference-first (review item #1, part 2)

A text job stops after `input.png`; `POST /api/jobs/{id}/model` promotes an
approved reference into a mesh job.

**Files:**
- Modify: `src/warlock/app.py:136-207` (`output=`), add `promote_to_model` route
- Modify: `src/warlock/queue.py:371-451` (`_generate` returns early), `:708-731` (`_attach_files`)
- Modify: `src/warlock/static/app.js`, `static/index.html`
- Test: `tests/test_api.py`, `tests/test_queue.py`

**Interfaces:**
- Consumes: `JobStore.create(..., stage=, parent_id=)` and `JobStore.children` (Task 4);
  the split seeds (Task 5).
- Produces: `POST /api/jobs` accepts `output: "reference" | "model"` (default
  `"model"`, so every existing client is unchanged);
  `POST /api/jobs/{id}/model` returns `{"id": str, "parent": str, "mesh_seed": int}`.

- [ ] **Step 1: Write the failing test**

In `tests/test_api.py`:

```python
def test_reference_output_stops_before_the_mesh(client):
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a barrel", "output": "reference"}
    )
    ref_id = r.json()["id"]
    assert client.get(f"/api/jobs/{ref_id}").json()["stage"] == "reference"


def test_promote_creates_a_child_model_job(client, tmp_path):
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a barrel", "output": "reference"}
    )
    ref_id = r.json()["id"]
    # The worker is not running a real pipeline in this fixture; stand the
    # reference in by hand, which is exactly the state promotion requires.
    import warlock.config as config_mod

    job_dir = config_mod.get_config().job_dir(ref_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"png")

    r = client.post(f"/api/jobs/{ref_id}/model")
    assert r.status_code == 200
    child = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert child["parent_id"] == ref_id
    assert child["kind"] == "image"
    assert child["stage"] == "model"


def test_promote_without_a_reference_is_a_400(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"})
    assert client.post(f"/api/jobs/{r.json()['id']}/model").status_code == 400
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_api.py -k "reference_output or promote" -v`
Expected: FAIL — 404 on the promote route, `stage` is `model`

- [ ] **Step 3: Accept `output` on create**

In `app.py:create_job`, add the form field:

```python
        output: Annotated[str, Form()] = "model",
```

and validate it beside the `kind` check:

```python
        if output not in ("reference", "model"):
            raise HTTPException(400, "output must be 'reference' or 'model'")
        if output == "reference" and kind != "text":
            # An image job's reference is the upload; there is nothing to approve.
            raise HTTPException(400, "only text jobs can stop at a reference")
```

Pass it through to the store at the end of the route:

```python
        await asyncio.to_thread(
            functools.partial(store().create, kind, prompt, params, job_id, stage=output)
        )
```

- [ ] **Step 4: Add the promote route**

Insert after `rerun_job` in `app.py`:

```python
    @app.post("/api/jobs/{job_id}/model")
    async def promote_to_model(
        job_id: str, mesh_seed: Annotated[int | None, Form()] = None
    ) -> dict[str, Any]:
        """Run the 3D stage from a reference the user approved.

        The child is an ordinary image job whose input.png is the parent's, which
        is the same reduction /rerun?mode=remesh already makes -- so the queue,
        the progress model and cancellation need no special case. What is new is
        parent_id, which is what lets the history show a reference and its
        attempts as one lineage instead of unrelated rows.
        """
        _check_job_id(job_id)
        source = await asyncio.to_thread(store().get, job_id)
        if source is None:
            raise HTTPException(404, "no such job")
        if source["stage"] != "reference":
            raise HTTPException(400, "this job is not a reference")
        if source["status"] != "done":
            raise HTTPException(400, f"reference is {source['status']}")
        src_png = config.job_dir(job_id) / "input.png"
        if not src_png.exists():
            raise HTTPException(400, "reference has no image")

        params = {
            k: v
            for k, v in source["params"].items()
            if k not in ("composed_prompt", "scale_factor", "mesh_audit")
        }
        params["mesh_seed"] = mesh_seed if mesh_seed is not None else _random_seed()
        params["seed"] = params["mesh_seed"]

        new_id = uuid.uuid4().hex[:12]
        new_dir = config.job_dir(new_id)
        new_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, src_png, new_dir / "input.png")
        await asyncio.to_thread(
            functools.partial(
                store().create,
                "image",
                source["prompt"],
                params,
                new_id,
                stage="model",
                parent_id=job_id,
            )
        )
        return {"id": new_id, "parent": job_id, "mesh_seed": params["mesh_seed"]}
```

- [ ] **Step 5: Run the API tests**

Run: `uv run pytest tests/test_api.py -k "reference_output or promote" -v`
Expected: PASS (the third still fails until Step 6 if `stage` defaults leak — it
should already pass, since a plain create is `stage='model'`)

- [ ] **Step 6: Write the failing worker test**

In `tests/test_queue.py`:

```python
@pytest.mark.asyncio
async def test_a_reference_job_never_reaches_trellis(worker_env):
    worker, store = worker_env
    job_id = store.create("text", "a barrel", {"seed": 1}, stage="reference")
    await _run_until_done(worker, store, job_id)
    assert store.get(job_id)["status"] == "done"
    assert worker.trellis.generate_calls == []
    assert (worker.config.job_dir(job_id) / "input.png").exists()
```

- [ ] **Step 7: Stop the worker after the reference**

In `src/warlock/queue.py:_generate`, immediately after the text branch's `finally`
block (i.e. after the SDXL stage and before the existing
`if self._cancel.event.is_set(): return`), insert:

```python
            if job.get("stage") == "reference":
                # The whole point of the split: the user judges the image before
                # anything pays for a trellis run. The job is finished here --
                # promotion creates a separate child job (app.promote_to_model).
                _log_vram("after reference-only job")
                return
```

`job` is the row dict from `JobStore._to_dict`, so `stage` is present.

- [ ] **Step 8: Make the reference downloadable and the lineage visible**

In `app.py:_attach_files`, no change is needed — `input.png` is already listed on
existence. In `list_jobs` and `get_job`, the `stage`/`parent_id` columns already
ride along in the row dict, so the API surface is complete.

- [ ] **Step 9: Run the tests**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 10: Wire the UI approve bar**

In `static/index.html`, add a checkbox beside the seed controls:

```html
      <label class="check"><input type="checkbox" id="approve-first" checked>
        Approve reference before 3D</label>
```

In `app.js`'s submit handler, before the fetch:

```js
  // Text jobs only: an image job's reference is the upload itself.
  if (kind === "text" && document.getElementById("approve-first").checked) {
    fd.set("output", "reference");
  }
```

In `createNode`, add two buttons beside `reroll`/`remesh`:

```js
  const make3d = document.createElement("button");
  setText(make3d, "generate 3D");
  make3d.title = "Run the 3D stage from this approved reference";
  make3d.hidden = true;
  const another = document.createElement("button");
  setText(another, "try another");
  another.title = "Same prompt and settings, new reference seed";
  another.hidden = true;
```

Append them into `actions` (`actions.append(make3d, another, reroll, remesh, rigBtn, act)`),
add them to the returned `n` object, and bind:

```js
  make3d.addEventListener("click", async (e) => {
    e.stopPropagation();
    make3d.disabled = true;
    try {
      const r = await fetch(`/api/jobs/${id}/model`, { method: "POST" });
      const body = await r.json();
      if (!r.ok) { alert(body.detail ?? "could not start the 3D stage"); return; }
      selected = body.id;
      shownModelFor = null;
      pending = body.id;
      downloads.style.display = "none";
      showOverlay();
    } finally {
      make3d.disabled = false;
    }
    poll(true);
  });
  another.addEventListener("click", (e) => {
    e.stopPropagation();
    // A fresh reference is exactly a re-roll of the reference job, which the
    // existing route already does correctly now that seeds are split.
    nodes.get(id).reroll.click();
  });
```

In `updateNode`, gate them:

```js
  const isReference = job.stage === "reference";
  n.make3d.hidden = !(isReference && done && job.files.includes("input.png"));
  n.another.hidden = !(isReference && done);
  // A reference has no mesh, so the mesh-only actions stay hidden for it.
  n.remesh.hidden = n.remesh.hidden || isReference;
  n.rigBtn.hidden = n.rigBtn.hidden || isReference;
```

- [ ] **Step 11: Verify by hand and commit**

Start the app, submit a text job with "Approve reference before 3D" ticked, confirm
the job finishes at the image, and that "generate 3D" produces a child job that
renders a mesh.

```bash
git add -A
git commit -m "Warlock v0.0.1

Approve-reference-first: output=reference and POST /api/jobs/{id}/model."
```

---

### Task 7: Batch reference candidates (review item #2)

SDXL-Turbo is 4 steps (~1 s/image). Generating N references per submit and picking
one is nearly free, and composes with Task 6: each candidate is its own reference
job, so promotion, cancellation and history all work unchanged.

**Files:**
- Modify: `src/warlock/app.py:136-207`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `output="reference"` (Task 6), split seeds (Task 5).
- Produces: `POST /api/jobs` accepts `count: int = 1` (1–8) and returns
  `{"id": str, "ids": list[str]}` — `id` stays the first, so every existing
  client that reads `body.id` is unchanged.

- [ ] **Step 1: Write the failing test**

In `tests/test_api.py`:

```python
def test_count_creates_n_reference_jobs_with_distinct_seeds(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "a barrel", "output": "reference", "count": 4},
    )
    body = r.json()
    assert len(body["ids"]) == 4
    assert body["id"] == body["ids"][0]
    seeds = {
        client.get(f"/api/jobs/{i}").json()["params"]["reference_seed"]
        for i in body["ids"]
    }
    assert len(seeds) == 4


def test_count_above_one_requires_reference_output(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "x", "count": 3})
    assert r.status_code == 400


def test_count_is_bounded(client):
    r = client.post(
        "/api/jobs",
        data={"kind": "text", "prompt": "x", "output": "reference", "count": 99},
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_api.py -k count -v`
Expected: FAIL — `KeyError: 'ids'`

- [ ] **Step 3: Implement it**

Add the constant near `ALLOWED_RESOLUTIONS` in `app.py`:

```python
# A submit may ask for several reference candidates at once. Bounded because each
# is a real queued job holding a place in the serial worker.
MAX_REFERENCE_COUNT = 8
```

Add the form field to `create_job`:

```python
        count: Annotated[int, Form()] = 1,
```

Validate beside `output`:

```python
        if not 1 <= count <= MAX_REFERENCE_COUNT:
            raise HTTPException(400, f"count must be between 1 and {MAX_REFERENCE_COUNT}")
        if count > 1 and output != "reference":
            # N meshes per submit is minutes of GPU each; only the cheap 4-step
            # reference stage is worth batching.
            raise HTTPException(400, "count > 1 requires output=reference")
```

Replace the final create block with a loop:

```python
        ids: list[str] = []
        for i in range(count):
            candidate = dict(params)
            if i > 0:
                # Candidate 0 keeps the requested seed so a pinned seed still
                # reproduces; the rest fan out from it.
                candidate["reference_seed"] = _random_seed()
                candidate["seed"] = candidate["reference_seed"]
            job_id = uuid.uuid4().hex[:12]
            if normalized is not None:
                job_dir = config.job_dir(job_id)
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "input.png").write_bytes(normalized)
            await asyncio.to_thread(
                functools.partial(
                    store().create, kind, prompt, candidate, job_id, stage=output
                )
            )
            ids.append(job_id)
        return {"id": ids[0], "ids": ids}
```

Delete the old single-create block it replaces (the `job_id = uuid4...` through
`return {"id": job_id}` lines).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_api.py -k count -v`
Expected: PASS

- [ ] **Step 5: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 6: Add the UI control**

In `static/index.html`, beside the approve-first checkbox:

```html
      <label for="ref-count">Candidates</label>
      <select id="ref-count">
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="4" selected>4</option>
        <option value="8">8</option>
      </select>
```

In `app.js`'s submit handler, beside the `output` line:

```js
  if (kind === "text" && document.getElementById("approve-first").checked) {
    fd.set("output", "reference");
    fd.set("count", document.getElementById("ref-count").value);
  }
```

The response's `body.id` is already what the handler follows, so the rest of the
handler is unchanged.

- [ ] **Step 7: Verify by hand and commit**

Submit with 4 candidates; confirm four reference cards appear and any one can be
promoted with "generate 3D".

```bash
git add -A
git commit -m "Warlock v0.0.1

Generate N reference candidates per submit."
```

---

## Plan A self-review notes

- Items #1–#6 each map to a task: #6→T1, #4→T2, #5→T3, #1→T4+T6, #3→T5, #2→T7.
- `params["seed"]` is still written by every path, so nothing that reads it breaks.
- `output` defaults to `"model"` and `count` to `1`, so the API is backward
  compatible and the existing `/rerun` routes are untouched (NEXT.md §2 asks for
  exactly that).
