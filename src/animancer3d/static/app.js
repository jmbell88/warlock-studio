import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// --- viewer -----------------------------------------------------------------

const canvas = document.getElementById("viewer");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14151a);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.6, 1.2, 1.6);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x223, 1.2));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(3, 4, 2);
scene.add(key);
const grid = new THREE.GridHelper(4, 16, 0x2c2f3a, 0x232530);
scene.add(grid);

let model = null;
const loader = new GLTFLoader();

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}
new ResizeObserver(resize).observe(canvas);

renderer.setAnimationLoop(() => {
  resize();
  controls.update();
  renderer.render(scene, camera);
});

function showModel(url) {
  loader.load(url, (gltf) => {
    if (model) scene.remove(model);
    model = gltf.scene;
    // Center on origin and rest on the grid.
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.set(-center.x, -box.min.y, -center.z);
    scene.add(model);
    const size = box.getSize(new THREE.Vector3()).length();
    camera.position.set(size, size * 0.75, size);
    controls.target.set(0, 0, 0);
    hideOverlay();
  }, undefined, () => {
    showOverlayError("Could not load the generated model.");
  });
}

// --- viewer overlay ---------------------------------------------------------

const placeholder = document.getElementById("placeholder");
const phIdle = document.getElementById("ph-idle");
const phProgress = document.getElementById("ph-progress");
const phLabel = document.getElementById("ph-label");
const phDetail = document.getElementById("ph-detail");
const phTime = document.getElementById("ph-time");
const phBar = phProgress.querySelector(".bar");
const phFill = phBar.querySelector("i");

function setText(el, s) { if (el.textContent !== s) el.textContent = s; }

// showModel() used to hide the placeholder permanently; the overlay lives inside
// it, so visibility has to be restored whenever a job starts.
function showOverlay() {
  placeholder.style.display = "grid";
  phIdle.hidden = true;
  phProgress.hidden = false;
}

function hideOverlay() {
  phProgress.hidden = true;
  phProgress.classList.remove("failed");
  phIdle.hidden = false;
  if (model) placeholder.style.display = "none";
}

function showOverlayError(message) {
  showOverlay();
  phProgress.classList.add("failed");
  setText(phLabel, "Generation failed");
  setText(phDetail, message || "");
  setText(phTime, "");
  phBar.classList.remove("indet");
  phFill.style.width = "100%";
}

function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  const s = Math.floor(seconds % 60);
  const m = Math.floor(seconds / 60);
  return m > 0 ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}

// --- job submission ---------------------------------------------------------

let kind = "text";
const tabs = { text: document.getElementById("tab-text"), image: document.getElementById("tab-image") };
for (const [k, btn] of Object.entries(tabs)) {
  btn.addEventListener("click", () => {
    kind = k;
    tabs.text.classList.toggle("active", k === "text");
    tabs.image.classList.toggle("active", k === "image");
    document.getElementById("text-input").hidden = k !== "text";
    document.getElementById("image-input").hidden = k !== "image";
  });
}

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.set("kind", kind);
  fd.set("seed", document.getElementById("seed").value || "42");
  fd.set("resolution", document.getElementById("resolution").value);
  if (kind === "text") {
    const prompt = document.getElementById("prompt").value.trim();
    if (!prompt) return;
    fd.set("prompt", prompt);
  } else {
    const file = document.getElementById("image").files[0];
    if (!file) return;
    fd.set("image", file);
  }
  const btn = document.getElementById("generate");
  btn.disabled = true;
  try {
    const r = await fetch("/api/jobs", { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) {
      alert(body.detail ?? "request failed");
    } else {
      // Follow the new job straight away instead of making the user find it.
      selected = body.id;
      shownModelFor = null;
      // Keep polling fast until the worker picks this up, otherwise the bar
      // would not start moving until the next idle tick.
      pending = body.id;
      downloads.style.display = "none";
      phProgress.classList.remove("failed");
      setText(phLabel, "Queued");
      setText(phDetail, "");
      setText(phTime, "");
      phBar.classList.add("indet");
      phFill.style.width = "0%";
      showOverlay();
    }
  } finally {
    btn.disabled = false;
  }
  poll(true);
});

// --- job list ---------------------------------------------------------------

const ul = document.getElementById("jobs");
const downloads = document.getElementById("downloads");
const nodes = new Map();   // job id -> element refs, reused across polls
const jobsById = new Map();
let selected = null;
let shownModelFor = null;

// --- banner --------------------------------------------------------------
// A single dismissible slot; the last shown "kind" wins so a doctor warning
// isn't clobbered by a transient disconnect and vice versa.

const banner = document.getElementById("banner");
const bannerText = document.getElementById("banner-text");
const bannerClose = document.getElementById("banner-close");
const dismissedBanners = new Set();
let activeBannerKind = null;

function showBanner(kind, text) {
  if (dismissedBanners.has(kind)) return;
  activeBannerKind = kind;
  setText(bannerText, text);
  banner.hidden = false;
}

function hideBanner(kind) {
  if (activeBannerKind !== kind) return;
  banner.hidden = true;
  activeBannerKind = null;
}

bannerClose.addEventListener("click", () => {
  if (activeBannerKind) dismissedBanners.add(activeBannerKind);
  banner.hidden = true;
  activeBannerKind = null;
});

function createNode(id) {
  const li = document.createElement("li");
  const img = document.createElement("img");
  img.hidden = true;
  const info = document.createElement("div");
  const title = document.createElement("div");
  title.className = "job-title";
  const status = document.createElement("div");
  const stage = document.createElement("div");
  stage.className = "job-stage";
  const err = document.createElement("div");
  err.className = "job-error";
  const bar = document.createElement("div");
  bar.className = "bar mini";
  bar.hidden = true;
  const fill = document.createElement("i");
  bar.append(fill);
  info.append(title, status, stage, err, bar);
  const actions = document.createElement("div");
  actions.className = "job-actions";
  const act = document.createElement("button");
  actions.append(act);
  li.append(img, info, actions);

  // Bound once, so a poll never rebinds stale closures.
  li.addEventListener("click", () => select(id));
  act.addEventListener("click", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    if (!job) return;
    const active = job.status === "queued" || job.status === "running";
    if (active) {
      act.disabled = true;
      setText(act, "cancelling…");
    }
    try {
      await fetch(`/api/jobs/${id}${active ? "/cancel" : ""}`, {
        method: active ? "POST" : "DELETE",
      });
    } finally {
      act.disabled = false;
    }
    if (!active && selected === id) {
      selected = null;
      hideOverlay();
      downloads.style.display = "none";
    }
    poll(true);
  });

  const n = { li, img, title, status, stage, err, bar, fill, act };
  nodes.set(id, n);
  return n;
}

function updateNode(n, job) {
  n.li.classList.toggle("selected", job.id === selected);

  const src = `/api/jobs/${job.id}/files/input.png`;
  const hasImage = job.files.includes("input.png");
  if (hasImage && n.img.getAttribute("src") !== src) n.img.src = src;
  n.img.hidden = !hasImage;

  setText(n.title, job.prompt ?? `image job ${job.id}`);

  // A cancelled job keeps running until its current GPU stage ends.
  const cancelling = job.status === "cancelled" && job.progress;
  const label = cancelling ? "cancelling" : job.status;
  n.status.className = `status ${cancelling ? "cancelling" : job.status}`;
  setText(n.status, label);

  const p = job.progress;
  if (p && !cancelling) {
    setText(n.stage, `${p.label}${p.detail ? ` · ${p.detail}` : ""}`);
    n.bar.hidden = false;
    n.fill.style.width = `${p.percent}%`;
  } else {
    setText(n.stage, "");
    n.bar.hidden = true;
  }

  setText(n.err, job.status === "error" && job.error ? job.error : "");

  const active = job.status === "queued" || job.status === "running";
  setText(n.act, active ? "cancel" : "delete");
}

function renderJobs(jobs) {
  jobs.forEach((job, i) => {
    jobsById.set(job.id, job);
    const n = nodes.get(job.id) ?? createNode(job.id);
    updateNode(n, job);
    if (ul.children[i] !== n.li) ul.insertBefore(n.li, ul.children[i] ?? null);
  });
  const live = new Set(jobs.map((j) => j.id));
  for (const [id, n] of nodes) {
    if (!live.has(id)) {
      n.li.remove();
      nodes.delete(id);
      jobsById.delete(id);
    }
  }
}

function select(id) {
  const job = jobsById.get(id);
  if (!job) return;
  selected = id;
  for (const [nid, n] of nodes) n.li.classList.toggle("selected", nid === id);
  if (job.files.includes("model.glb")) {
    showSelected(job);
  } else {
    shownModelFor = null;
    downloads.style.display = "none";
    if (job.status === "error") showOverlayError(job.error || "");
    else if (job.progress) showOverlay();
    else hideOverlay();
  }
}

function showSelected(job) {
  if (shownModelFor === job.id) return;
  shownModelFor = job.id;
  showModel(`/api/jobs/${job.id}/files/model.glb`);
  document.getElementById("dl-glb").href = `/api/jobs/${job.id}/files/model.glb`;
  document.getElementById("dl-obj").href = `/api/jobs/${job.id}/files/model_obj.zip`;
  document.getElementById("dl-stl").href = `/api/jobs/${job.id}/files/model.stl`;
  downloads.style.display = "flex";
}

// --- polling ----------------------------------------------------------------
//
// Two cadences: /api/progress is cheap (no DB, no disk) and drives the bar while
// a job runs; the full job list is fetched far less often.

const FAST_MS = 600, IDLE_MS = 2500, LIST_EVERY = 4;

let skew = 0;             // server clock - browser clock, seconds
let live = null;          // latest progress payload
let liveJobId = null;
let inFlight = false;
let wantList = false;      // survives a dropped poll, unlike a bare argument
let pending = null;        // submitted but not yet seen running
let tick = 0;
let etaEma = null;
const abort = new AbortController();
window.addEventListener("beforeunload", () => abort.abort());

let pollFailures = 0;

async function poll(forceList = false) {
  if (forceList) wantList = true;
  if (inFlight) return;
  inFlight = true;
  try {
    const r = await fetch("/api/progress", { signal: abort.signal });
    const data = await r.json();
    pollFailures = 0;
    hideBanner("disconnected");
    skew = data.server_time - Date.now() / 1000;

    const changed = data.job_id !== liveJobId;
    if (changed) etaEma = null;
    liveJobId = data.job_id;
    live = data.progress;

    if (wantList || changed || tick % LIST_EVERY === 0) {
      wantList = false;
      await refreshList();
    }
    tick++;

    // Stop chasing a submitted job once it is running or already finished.
    if (pending) {
      const p = jobsById.get(pending);
      if (pending === liveJobId || (p && p.status !== "queued")) pending = null;
    }
    renderOverlay();
  } catch (e) {
    if (e.name !== "AbortError") {
      console.error("poll failed", e);
      pollFailures++;
      if (pollFailures >= 3) showBanner("disconnected", "Lost connection to the server — retrying…");
    }
  } finally {
    inFlight = false;
  }
}

async function refreshList() {
  const jobs = await (await fetch("/api/jobs", { signal: abort.signal })).json();
  renderJobs(jobs);
  const current = jobs.find((j) => j.id === selected);
  if (current?.files.includes("model.glb")) {
    showSelected(current);
  } else if (current?.status === "error" && selected !== shownModelFor) {
    showOverlayError(current.error || "");
  }
}

function renderOverlay() {
  // Only narrate the job the user is actually looking at.
  if (!live || selected !== liveJobId) {
    if (phProgress.hidden === false && !phProgress.classList.contains("failed")) {
      const sel = jobsById.get(selected);
      if (!sel || sel.files?.includes("model.glb") || sel.status === "done") hideOverlay();
    }
    return;
  }
  showOverlay();
  phProgress.classList.toggle("failed", Boolean(live.error));
  setText(phLabel, live.error ? "Generation failed" : live.label);

  const indeterminate = live.percent <= 0;
  phBar.classList.toggle("indet", indeterminate);
  if (!indeterminate) phFill.style.width = `${live.percent}%`;

  const bits = [];
  if (live.error) bits.push(live.error);
  else {
    if (live.detail) bits.push(live.detail);
    if (live.stage_eta) bits.push(`~${Math.round(live.stage_eta)}s`);
    if (live.stage_index) bits.push(`stage ${live.stage_index}/${live.stage_total}`);
  }
  setText(phDetail, bits.join(" · "));
  renderClock();
}

// Ticks locally between polls so elapsed time reads smoothly rather than
// jumping once per request.
function renderClock() {
  if (!live || live.error || selected !== liveJobId) return;
  const elapsed = Date.now() / 1000 + skew - live.started_at;
  let out = `${fmtDuration(elapsed)} elapsed`;

  // Extrapolated ETA. Suppressed early (the estimate is noise) and on a cold
  // server, where the first stage silently absorbs an ~8 GB model load.
  if (!live.cold && live.percent >= 10 && live.percent < 100 && elapsed > 5) {
    const remaining = (elapsed / live.percent) * (100 - live.percent);
    etaEma = etaEma === null ? remaining : etaEma * 0.7 + remaining * 0.3;
    out += ` · ~${fmtDuration(etaEma)} left`;
  }
  setText(phTime, out);
}
setInterval(renderClock, 200);

async function checkDoctor() {
  try {
    const health = await (await fetch("/api/health")).json();
    const bad = (health.checks || []).filter((c) => c.fatal && !c.ok);
    if (bad.length) {
      showBanner("doctor", `Setup problem: ${bad.map((c) => c.detail).join("; ")}`);
    }
  } catch (e) {
    console.error("doctor check failed", e);
  }
}
checkDoctor();

(function loop() {
  poll().finally(() => setTimeout(loop, liveJobId || pending ? FAST_MS : IDLE_MS));
})();
