import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

// --- viewer -----------------------------------------------------------------

const canvas = document.getElementById("viewer");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.toneMapping = THREE.ACESFilmicToneMapping;
// Set explicitly rather than relying on the default: ACES darkens noticeably,
// and the exposure is what the environment intensity below is balanced against.
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14151a);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(1.6, 1.2, 1.6);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

// A generated sky/horizon/ground gradient rather than an imported studio HDRI:
// the vendored three build ships no RoomEnvironment addon, and the frontend has
// no build step and fetches nothing at runtime. PMREMGenerator blurs this into
// an irradiance probe, so 32x16 is ample.
//
// Without it, a downward-facing surface received only the hemisphere light's
// ground colour -- which was 0x000223, i.e. black -- so roughly a quarter of the
// model's surface rendered darker than the background and read as holes.
function gradientEnvironment(target) {
  const width = 32, height = 16;
  const sky = [0.42, 0.48, 0.60];
  const horizon = [0.34, 0.34, 0.36];
  const ground = [0.16, 0.155, 0.15];
  const data = new Uint16Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    // Row 0 is v=0, which equirectangular mapping puts at -Y.
    const v = (y + 0.5) / height;
    const t = Math.abs(v - 0.5) * 2;
    const edge = v > 0.5 ? sky : ground;
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) {
        data[i + c] = THREE.DataUtils.toHalfFloat(horizon[c] + (edge[c] - horizon[c]) * t);
      }
      data[i + 3] = THREE.DataUtils.toHalfFloat(1.0);
    }
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  // DataTexture defaults to NearestFilter, which would step the gradient into
  // visible bands before PMREM ever blurs it.
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  const pmrem = new THREE.PMREMGenerator(target);
  const environment = pmrem.fromEquirectangular(texture).texture;
  pmrem.dispose();
  texture.dispose();
  return environment;
}

// Built once at startup -- fromEquirectangular allocates a render target.
scene.environment = gradientEnvironment(renderer);
scene.environmentIntensity = 1.0;

// The environment now supplies the ambient fill, so the hemisphere light only
// tints it and the key light is purely for shaping.
scene.add(new THREE.HemisphereLight(0xffffff, 0x223, 0.3));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(3, 4, 2);
scene.add(key);
let grid = new THREE.GridHelper(4, 16, 0x2c2f3a, 0x232530);
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
  syncJointMarkers();
  renderer.render(scene, camera);
});

function disposeModel(root) {
  root.traverse((obj) => {
    if (!obj.isMesh) return;
    obj.geometry?.dispose();
    for (const material of [].concat(obj.material ?? [])) {
      for (const value of Object.values(material)) {
        if (value && value.isTexture) value.dispose();
      }
      material.dispose();
    }
  });
}

// onReady, when given, receives the placed root and the model's bounding
// radius once it is in the scene -- the pose editor needs both to bind its
// joint markers at a size that suits the mesh.
function showModel(url, onReady) {
  loader.load(url, (gltf) => {
    if (model) {
      scene.remove(model);
      disposeModel(model);
    }
    model = gltf.scene;
    // Center on origin and rest on the grid.
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.set(-center.x, -box.min.y, -center.z);
    scene.add(model);

    // Everything below is derived from the bounding box rather than fixed:
    // guidance.py sizes a model anywhere from 0.01 m to 100 m, so the old
    // hardcoded near/far of 0.01/100 and 4 m grid only ever framed a ~1 m prop.
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() * 0.5, 1e-4);
    camera.near = radius / 1000;
    camera.far = radius * 100;
    camera.updateProjectionMatrix();

    const distance = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov * 0.5))) * 1.25;
    camera.position.set(distance * 0.62, distance * 0.47, distance * 0.62);
    controls.target.set(0, size.y * 0.5, 0);
    controls.minDistance = radius * 0.5;
    controls.maxDistance = radius * 20;
    controls.update();

    // A power-of-ten span that comfortably contains the footprint.
    const span = Math.pow(10, Math.ceil(Math.log10(Math.max(size.x, size.z) * 2.5)));
    scene.remove(grid);
    grid.dispose();
    grid = new THREE.GridHelper(span, 16, 0x2c2f3a, 0x232530);
    scene.add(grid);

    onReady?.(model, radius);
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

// --- design guidance --------------------------------------------------------
// The taxonomy lives in guidance.py and arrives via /api/guidance, so the
// selects can never drift out of sync with what the API will accept.

const GUIDANCE_FIELDS = [
  "category", "genre", "art_style", "base_model", "style_lora", "platform", "bg_removal",
];
const guidanceSelects = {};
const sizeInput = document.getElementById("g-size");
const platformHint = document.getElementById("platform-hint");
const loraWeight = document.getElementById("g-lora-weight");
const loraWeightOut = document.getElementById("g-lora-weight-out");
let categorySizes = {};
let platformRes = {};
let loraDefaults = {};
// Once the user types their own size, changing category must not clobber it.
let sizeEdited = false;
sizeInput.addEventListener("input", () => { sizeEdited = true; });

function syncSizeFromCategory() {
  if (sizeEdited) return;
  const def = categorySizes[guidanceSelects.category.value];
  if (def !== undefined) sizeInput.value = def;
}

function syncPlatformHint() {
  const res = platformRes[guidanceSelects.platform.value];
  platformHint.textContent = res ? `Geometry resolution ${res}` : "";
}

// The slider is only meaningful once a style is picked, and each LoRA carries
// its own tuned strength, so selecting one seeds the slider rather than
// leaving the user to guess a number.
function syncLoraWeight() {
  const key = guidanceSelects.style_lora.value;
  document.getElementById("g-lora-weight").parentElement.hidden = !key;
  if (key && loraDefaults[key] !== undefined) loraWeight.value = loraDefaults[key];
  loraWeightOut.textContent = Number(loraWeight.value).toFixed(2);
}

async function loadGuidance() {
  const catalog = await (await fetch("/api/guidance")).json();
  // bg_removal is a server capability, not an Option table -- catalog.fields
  // has no entry for it, so it gets its own pass below rather than joining
  // this loop over {key,label} objects.
  for (const field of GUIDANCE_FIELDS) {
    if (field === "bg_removal") continue;
    const select = document.querySelector(`[data-guidance="${field}"]`);
    guidanceSelects[field] = select;
    // Genre, art style, category and style LoRA are all optional. Platform
    // always resolves to a geometry resolution and base model always resolves
    // to a checkpoint, so neither has a blank choice.
    if (field !== "platform" && field !== "base_model") {
      select.append(new Option(field === "style_lora" ? "— none —" : "— any —", ""));
    }
    for (const opt of catalog.fields[field]) {
      select.append(new Option(opt.label, opt.key));
      if (field === "category" && opt.default_size_m) categorySizes[opt.key] = opt.default_size_m;
      if (field === "platform" && opt.resolution) platformRes[opt.key] = opt.resolution;
      if (field === "style_lora" && opt.default_weight) loraDefaults[opt.key] = opt.default_weight;
    }
  }
  const bgSelect = document.getElementById("g-bg_removal");
  for (const key of catalog.bg_removal ?? []) {
    const opt = document.createElement("option");
    opt.value = key;
    setText(
      opt,
      key === "birefnet" ? "BiRefNet (best)" : key === "threshold" ? "Threshold (fast)" : "Auto",
    );
    bgSelect.append(opt);
  }
  bgSelect.value = catalog.defaults?.bg_removal ?? "auto";
  guidanceSelects.bg_removal = bgSelect;
  // Only seed it, never clobber -- a value the user already typed must survive
  // a second catalog load (e.g. after loadGuidance is re-triggered).
  const negative = document.getElementById("negative-prompt");
  if (!negative.value) negative.value = catalog.defaults?.negative_prompt ?? "";
  guidanceSelects.platform.value = catalog.defaults.platform;
  guidanceSelects.base_model.value = catalog.defaults.base_model;
  sizeInput.value = catalog.defaults.size_m;
  loraWeight.min = catalog.lora_weight_range[0];
  loraWeight.max = catalog.lora_weight_range[1];
  loraWeight.value = catalog.defaults.lora_weight;
  guidanceSelects.category.addEventListener("change", syncSizeFromCategory);
  guidanceSelects.platform.addEventListener("change", syncPlatformHint);
  guidanceSelects.style_lora.addEventListener("change", syncLoraWeight);
  loraWeight.addEventListener("input", () => {
    loraWeightOut.textContent = Number(loraWeight.value).toFixed(2);
  });
  syncPlatformHint();
  syncLoraWeight();
}

loadGuidance().catch((e) => console.error("could not load guidance options", e));

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

// --- seed -------------------------------------------------------------------
// Generation is deterministic in the seed, so a fixed default meant hitting
// Generate twice on an unchanged form produced byte-identical output. Random
// by default makes "give me another" the natural action; the lock is there for
// when you actually want to reproduce a result.

const seedInput = document.getElementById("seed");
const seedLock = document.getElementById("seed-lock");

// 31-bit, matching the server's _random_seed: fits a JS integer and an sqlite
// INTEGER without either end rounding it.
const newSeed = () => Math.floor(Math.random() * 2 ** 31);

function rollSeed() {
  seedInput.value = newSeed();
}

document.getElementById("seed-reroll").addEventListener("click", rollSeed);
rollSeed();

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
    // Genre, art style and the image-model selects only affect the SDXL stage,
    // and image jobs never run SDXL -- hide them rather than offer controls
    // that do nothing.
    for (const field of ["genre", "art_style", "base_model", "style_lora"]) {
      document.getElementById(`g-${field}-row`).hidden = k !== "text";
    }
    // Image jobs never run SDXL, so a negative prompt has nothing to act on --
    // unlike bg_removal, which stays visible because trellis-server matting
    // applies to both job kinds.
    document.getElementById("negative-row").hidden = k !== "text";
  });
}

// --- rigging ----------------------------------------------------------------
// Rigging needs bpy, which is an optional extra. /api/rig/templates answers
// both "which skeletons" and "is it installed at all"; when it isn't, every
// rig control stays hidden rather than becoming a button that only fails.

const rig = { available: false, templates: [] };
const rigOptions = document.getElementById("rig-options");
const rigEnable = document.getElementById("rig-enable");
const rigTemplate = document.getElementById("rig-template");

async function loadRig() {
  let info;
  try {
    info = await (await fetch("/api/rig/templates")).json();
  } catch (e) {
    console.error("rig catalog failed", e);   // stay hidden; generation is unaffected
    return;
  }
  rig.available = Boolean(info.available);
  rig.templates = info.templates ?? [];
  for (const t of rig.templates) {
    const opt = document.createElement("option");
    opt.value = t.key;
    setText(opt, t.label);
    rigTemplate.append(opt);
  }
  if (info.default) rigTemplate.value = info.default;
  rigOptions.hidden = !rig.available;
  // Cards rendered before this resolved have no rig button yet.
  for (const [id, n] of nodes) {
    const job = jobsById.get(id);
    if (job) updateNode(n, job);
  }
}

// A job is rigged once rig.glb is listed, which the API gates on rig.json --
// see _attach_files. Anything else means "no rig yet".
function isRigged(job) {
  return Boolean(job.files?.includes("rig.glb"));
}

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.set("kind", kind);
  fd.set("seed", seedInput.value || String(newSeed()));
  // No explicit resolution: the platform preset supplies it server-side.
  for (const field of GUIDANCE_FIELDS) {
    // Genre and art style are hidden for image jobs, so don't send them either.
    const row = document.getElementById(`g-${field}-row`);
    if (row && row.hidden) continue;
    const value = guidanceSelects[field]?.value;
    if (value) fd.set(field, value);
  }
  if (sizeInput.value) fd.set("size_m", sizeInput.value);
  if (rig.available && rigEnable.checked) {
    fd.set("rig", "true");
    fd.set("rig_template", rigTemplate.value);
  }
  // Only meaningful alongside a style LoRA, which is text-jobs-only.
  if (kind === "text" && guidanceSelects.style_lora?.value) {
    fd.set("lora_weight", loraWeight.value);
  }
  if (kind === "text") {
    const prompt = document.getElementById("prompt").value.trim();
    if (!prompt) return;
    fd.set("prompt", prompt);
    fd.set("negative_prompt", document.getElementById("negative-prompt").value);
    // Text jobs only: an image job's reference is the upload itself.
    if (document.getElementById("approve-first").checked) {
      fd.set("output", "reference");
      fd.set("count", document.getElementById("ref-count").value);
    }
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
      // Next submit gets a different mesh unless the user asked to pin it.
      if (!seedLock.checked) rollSeed();
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
  const quality = document.createElement("div");
  quality.className = "job-quality";
  const settings = document.createElement("div");
  settings.className = "job-settings";
  settings.hidden = true;
  const settingsToggle = document.createElement("button");
  settingsToggle.type = "button";
  settingsToggle.className = "link";
  setText(settingsToggle, "settings");
  settingsToggle.hidden = true;
  const bar = document.createElement("div");
  bar.className = "bar mini";
  bar.hidden = true;
  const fill = document.createElement("i");
  bar.append(fill);
  info.append(title, status, stage, err, quality, settingsToggle, settings, bar);
  const actions = document.createElement("div");
  actions.className = "job-actions";
  const reroll = document.createElement("button");
  setText(reroll, "re-roll");
  reroll.title = "Same prompt and settings, new seed";
  reroll.hidden = true;
  const remesh = document.createElement("button");
  setText(remesh, "re-3D");
  remesh.title = "Reuse this reference image, rerun only the 3D stage";
  remesh.hidden = true;
  const rigBtn = document.createElement("button");
  setText(rigBtn, "rig");
  rigBtn.title = "Fit a skeleton to this mesh";
  rigBtn.hidden = true;
  const make3d = document.createElement("button");
  setText(make3d, "generate 3D");
  make3d.title = "Run the 3D stage from this approved reference";
  make3d.hidden = true;
  const another = document.createElement("button");
  setText(another, "try another");
  another.title = "Same prompt and settings, new reference seed";
  another.hidden = true;
  const act = document.createElement("button");
  actions.append(make3d, another, reroll, remesh, rigBtn, act);
  li.append(img, info, actions);

  // Bound once, so a poll never rebinds stale closures. Shared by reroll and
  // "try another" (mode is always "reroll" for the latter -- a fresh
  // reference is exactly a re-roll of the reference job, and rerun_job keeps
  // it at stage="reference" now that seeds are split).
  li.addEventListener("click", () => select(id));
  async function runRerun(mode) {
    const fd = new FormData();
    fd.set("mode", mode);
    const r = await fetch(`/api/jobs/${id}/rerun`, { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) {
      alert(body.detail ?? "rerun failed");
      return;
    }
    // Follow the new job the same way a fresh submit does.
    selected = body.id;
    shownModelFor = null;
    pending = body.id;
    downloads.style.display = "none";
    showOverlay();
  }
  for (const [btn, mode] of [[reroll, "reroll"], [remesh, "remesh"]]) {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      btn.disabled = true;
      try {
        await runRerun(mode);
      } finally {
        btn.disabled = false;
      }
      poll(true);
    });
  }
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
  another.addEventListener("click", async (e) => {
    e.stopPropagation();
    another.disabled = true;
    try {
      await runRerun("reroll");
    } finally {
      another.disabled = false;
    }
    poll(true);
  });
  rigBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    rigBtn.disabled = true;
    try {
      const fd = new FormData();
      fd.set("template", rigTemplate.value);
      const r = await fetch(`/api/jobs/${id}/rig`, { method: "POST", body: fd });
      const body = await r.json();
      if (!r.ok) alert(body.detail ?? "could not queue the rig");
    } finally {
      rigBtn.disabled = false;
    }
    poll(true);
  });
  settingsToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    settings.hidden = !settings.hidden;
    setText(settingsToggle, settings.hidden ? "settings" : "hide settings");
  });
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
      setPoseJob(null);
      downloads.style.display = "none";
    }
    poll(true);
  });

  // Bound once here (not in updateNode, which runs every ~600ms poll) so a
  // poll never rebinds a stale closure; it looks the job up fresh at click
  // time instead of capturing it.
  const copySettings = document.createElement("button");
  copySettings.type = "button";
  copySettings.className = "link";
  setText(copySettings, "copy settings to form");
  copySettings.addEventListener("click", (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    if (job) copySettingsToForm(job);
  });

  const n = { li, img, title, status, stage, err, quality, settings, settingsToggle,
              copySettings, bar, fill, act, reroll, remesh, rigBtn, make3d, another };
  nodes.set(id, n);
  return n;
}

// meshaudit measures the fraction of the worst-case silhouette you can see
// straight through. ~0 is solid; the perforated crust trellis-server's default
// band produces measures 0.07-0.15. The thresholds mirror those bands.
function qualityBadge(audit) {
  const worst = audit?.worst;
  if (typeof worst !== "number") return null;
  const pct = (worst * 100).toFixed(worst < 0.1 ? 1 : 0);
  if (worst < 0.02) return { cls: "good", text: "watertight" };
  if (worst < 0.08) return { cls: "warn", text: `${pct}% see-through` };
  return { cls: "bad", text: `${pct}% see-through — try another seed` };
}

function updateNode(n, job) {
  n.li.classList.toggle("selected", job.id === selected);

  const src = `/api/jobs/${job.id}/files/input.png`;
  const hasImage = job.files.includes("input.png");
  if (hasImage && n.img.getAttribute("src") !== src) n.img.src = src;
  n.img.hidden = !hasImage;

  // A rig job carries its source's prompt, so it needs the prefix or the
  // history reads as the same model having been generated twice.
  setText(n.title, job.kind === "rig"
    ? `rig · ${job.prompt ?? job.params?.source_job ?? job.id}`
    : job.prompt ?? `image job ${job.id}`);

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

  const badge = job.status === "done" ? qualityBadge(job.params?.mesh_audit) : null;
  n.quality.className = `job-quality ${badge ? badge.cls : ""}`;
  setText(n.quality, badge ? badge.text : "");

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
  // The button itself is created once in createNode (see the comment there);
  // rebuilding the rows every poll must not touch its listener.
  if (rows.length) n.settings.append(n.copySettings);

  const active = job.status === "queued" || job.status === "running";
  setText(n.act, active ? "cancel" : "delete");

  // Only offered on a finished job: re-rolling a queued one just races it, and
  // re-3D needs a reference image that survived the run.
  const done = job.status === "done";
  const generated = job.kind !== "rig";
  n.reroll.hidden = !done || !generated;
  n.remesh.hidden = !done || !generated || !job.files.includes("input.png");
  // Offered once, on a finished mesh that isn't rigged yet. A rig job has no
  // mesh of its own, and re-rigging would silently overwrite the skeleton
  // whatever poses were saved against it.
  n.rigBtn.hidden =
    !rig.available || !done || !generated || !job.files.includes("model.glb") || isRigged(job);

  const isReference = job.stage === "reference";
  n.make3d.hidden = !(isReference && done && job.files.includes("input.png"));
  n.another.hidden = !(isReference && done);
  // A reference has no mesh, so the mesh-only actions stay hidden for it.
  n.remesh.hidden = n.remesh.hidden || isReference;
  n.rigBtn.hidden = n.rigBtn.hidden || isReference;
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
    setPoseJob(job);
    downloads.style.display = "none";
    if (job.status === "error") showOverlayError(job.error || "");
    else if (job.progress) showOverlay();
    else hideOverlay();
  }
}

function showSelected(job) {
  // The rig can land while the same job stays selected, so the download row
  // is refreshed even on the early-out path that skips reloading the mesh.
  const dlRig = document.getElementById("dl-rig");
  dlRig.href = `/api/jobs/${job.id}/files/rig.glb`;
  dlRig.hidden = !isRigged(job);
  const dlPng = document.getElementById("dl-png");
  dlPng.href = `/api/jobs/${job.id}/files/input.png`;
  dlPng.hidden = !job.files.includes("input.png");
  setPoseJob(job);
  // The pose editor has the rig loaded on purpose; a routine list refresh must
  // not yank it out from under an in-progress edit.
  if (poseState.editing && poseState.job === job.id) return;
  if (shownModelFor === job.id) return;
  shownModelFor = job.id;
  showModel(`/api/jobs/${job.id}/files/model.glb`);
  document.getElementById("dl-glb").href = `/api/jobs/${job.id}/files/model.glb`;
  document.getElementById("dl-obj").href = `/api/jobs/${job.id}/files/model_obj.zip`;
  document.getElementById("dl-stl").href = `/api/jobs/${job.id}/files/model.stl`;
  downloads.style.display = "flex";
}

// --- pose editor ------------------------------------------------------------
//
// Forward kinematics only: a pose is each joint's local rotation, nothing else.
// That is deliberate -- it is exactly what a glTF node carries, so the browser
// and the Blender worker describe a pose with the same numbers and no
// conversion sits between them (see blender_worker._rest_local_rotation).
//
// Editing swaps the viewer from model.glb to rig.glb, because only the rigged
// export has the skeleton to drag.

const posePanel = document.getElementById("pose-panel");
const poseToggle = document.getElementById("pose-toggle");
const poseEditor = document.getElementById("pose-editor");
const poseSelection = document.getElementById("pose-selection");
const poseList = document.getElementById("pose-list");

const poseState = {
  job: null,        // job id the panel is bound to
  editing: false,
  bones: new Map(), // bone name -> THREE.Bone in the live rig
  rest: new Map(),  // bone name -> its rest quaternion, for the reset buttons
  selected: null,
  current: null,    // id of the saved pose being edited, if any
  saved: [],
};

let markers = null;   // THREE.Group of clickable joint spheres
let gizmo = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const MARKER_IDLE = 0x7c6cf0, MARKER_ACTIVE = 0x4cc38a;

function ensureGizmo() {
  if (gizmo) return gizmo;
  gizmo = new TransformControls(camera, canvas);
  gizmo.setMode("rotate");
  // Local space: a rotation gizmo aligned to the joint's own axes is what
  // makes "bend the elbow" a single-ring drag.
  gizmo.setSpace("local");
  gizmo.size = 0.6;
  // r170's TransformControls has no dragging-changed event; mouseDown/mouseUp
  // are the equivalent pair, and orbiting mid-drag would fight the gizmo.
  gizmo.addEventListener("mouseDown", () => { controls.enabled = false; });
  gizmo.addEventListener("mouseUp", () => { controls.enabled = true; });
  scene.add(gizmo.getHelper());
  return gizmo;
}

// Markers live in their own group rather than as children of the bones, so a
// joint stays the same size on screen whatever the armature does to its scale.
function syncJointMarkers() {
  if (!markers) return;
  scene.updateMatrixWorld();
  for (const marker of markers.children) {
    marker.position.setFromMatrixPosition(marker.userData.bone.matrixWorld);
  }
}

function clearRig() {
  gizmo?.detach();
  if (markers) {
    scene.remove(markers);
    for (const m of markers.children) {
      m.geometry.dispose();
      m.material.dispose();
    }
    markers = null;
  }
  poseState.bones.clear();
  poseState.rest.clear();
  poseState.selected = null;
}

function bindRig(root, radius) {
  clearRig();
  const bones = [];
  root.traverse((obj) => {
    if (obj.isBone) bones.push(obj);
  });
  if (!bones.length) return;

  markers = new THREE.Group();
  // Not raycast against, not lit, always drawn on top: these are handles, and
  // a handle buried inside the mesh is a handle you cannot click.
  const material = new THREE.MeshBasicMaterial({ color: MARKER_IDLE, depthTest: false });
  const geometry = new THREE.SphereGeometry(radius * 0.022, 12, 8);
  for (const bone of bones) {
    poseState.bones.set(bone.name, bone);
    poseState.rest.set(bone.name, bone.quaternion.clone());
    const marker = new THREE.Mesh(geometry, material.clone());
    marker.renderOrder = 999;
    marker.userData.bone = bone;
    markers.add(marker);
  }
  material.dispose();
  scene.add(markers);
  syncJointMarkers();
}

function selectJoint(bone) {
  poseState.selected = bone;
  for (const marker of markers?.children ?? []) {
    marker.material.color.setHex(marker.userData.bone === bone ? MARKER_ACTIVE : MARKER_IDLE);
  }
  if (bone) ensureGizmo().attach(bone);
  else gizmo?.detach();
  setText(poseSelection, bone ? bone.name : "Click a joint to rotate it.");
}

canvas.addEventListener("pointerdown", (e) => {
  // gizmo.axis is set while a handle is hovered, which is the case for the
  // click that starts a drag -- that click belongs to the gizmo, not to us.
  if (!poseState.editing || !markers || gizmo?.dragging || gizmo?.axis) return;
  const rect = canvas.getBoundingClientRect();
  pointer.set(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(markers.children, false)[0];
  if (hit) selectJoint(hit.object.userData.bone);
});

function currentPoseBones() {
  const bones = {};
  for (const [name, bone] of poseState.bones) bones[name] = bone.quaternion.toArray();
  return bones;
}

function applyPose(record) {
  for (const [name, quat] of Object.entries(record.bones ?? {})) {
    poseState.bones.get(name)?.quaternion.fromArray(quat);
  }
  poseState.current = record.id ?? null;
}

function resetPose() {
  for (const [name, quat] of poseState.rest) poseState.bones.get(name)?.quaternion.copy(quat);
  poseState.current = null;
}

// Called on every selection and every list refresh, so it has to be a no-op
// for the job already bound -- otherwise an open editor would be rebuilt from
// scratch twice a second.
//
// The panel follows the mesh, not the rig: an unrigged prop still gets a
// sprite sheet (a turnaround of its rest pose), it just has nothing to pose.
async function setPoseJob(job) {
  const hasMesh = Boolean(job) && Boolean(job.files?.includes("model.glb"));
  const rigged = hasMesh && isRigged(job);
  posePanel.hidden = !hasMesh;
  document.getElementById("pose-head").hidden = !rigged;
  if (!rigged) exitPoseEditing(false);
  const id = hasMesh ? job.id : null;
  if (poseState.job === id) return;
  // No restore: the caller is already showing a different job's model.
  exitPoseEditing(false);
  poseState.job = id;
  poseState.current = null;
  poseState.saved = [];
  sheetsSeen = "";
  renderPoses();
  if (!id) {
    sheetList.replaceChildren();
    return;
  }
  if (rigged) await refreshPoses();
  await refreshSheets();
}

async function refreshPoses() {
  if (!poseState.job) return;
  try {
    const body = await (await fetch(`/api/jobs/${poseState.job}/poses`)).json();
    poseState.saved = body.poses ?? [];
  } catch (e) {
    console.error("could not load poses", e);
    poseState.saved = [];
  }
  renderPoses();
  syncSheetRows();
}

function renderPoses() {
  poseList.replaceChildren();
  for (const record of poseState.saved) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    setText(name, record.name);
    name.title = "Show this pose";
    name.addEventListener("click", async () => {
      if (!poseState.editing) await enterPoseEditing();
      applyPose(record);
    });
    const glb = document.createElement("a");
    setText(glb, "glb");
    glb.download = "";
    glb.href = `/api/jobs/${poseState.job}/poses/${record.id}/model.glb`;
    bindBusyDownload(glb);
    const del = document.createElement("button");
    setText(del, "×");
    del.title = "Delete this pose";
    del.addEventListener("click", async () => {
      if (!confirm(`Delete the pose “${record.name}”?`)) return;
      del.disabled = true;
      try {
        await fetch(`/api/jobs/${poseState.job}/poses/${record.id}`, { method: "DELETE" });
      } finally {
        del.disabled = false;
      }
      if (poseState.current === record.id) poseState.current = null;
      await refreshPoses();
    });
    li.append(name, glb, del);
    poseList.append(li);
  }
}

async function enterPoseEditing() {
  if (poseState.editing || !poseState.job) return;
  poseState.editing = true;
  poseToggle.classList.add("active");
  setText(poseToggle, "done");
  poseEditor.hidden = false;
  // The rig is a different file, so the viewer has to reload; clearing
  // shownModelFor is what lets the poll's showSelected put the plain mesh back
  // when editing ends.
  shownModelFor = null;
  await new Promise((resolve) => {
    showModel(`/api/jobs/${poseState.job}/files/rig.glb`, (root, radius) => {
      bindRig(root, radius);
      resolve();
    });
  });
  selectJoint(null);
}

function exitPoseEditing(restore = true) {
  if (!poseState.editing) return;
  poseState.editing = false;
  poseToggle.classList.remove("active");
  setText(poseToggle, "edit");
  poseEditor.hidden = true;
  clearRig();
  shownModelFor = null;
  const job = restore ? jobsById.get(poseState.job) : null;
  if (job) showSelected(job);
}

poseToggle.addEventListener("click", () => {
  if (poseState.editing) exitPoseEditing();
  else enterPoseEditing();
});

document.getElementById("pose-reset-bone").addEventListener("click", () => {
  const bone = poseState.selected;
  if (bone) bone.quaternion.copy(poseState.rest.get(bone.name));
});

document.getElementById("pose-reset-all").addEventListener("click", resetPose);

document.getElementById("pose-save").addEventListener("click", async () => {
  if (!poseState.bones.size) return;
  const existing = poseState.saved.find((p) => p.id === poseState.current);
  const name = prompt("Name this pose", existing?.name ?? "")?.trim();
  if (!name) return;
  // Saving under an existing name replaces it, rather than leaving two poses
  // called "idle" that differ by one shoulder.
  const clash = poseState.saved.find((p) => p.name === name);
  const body = { name, bones: currentPoseBones() };
  if (clash) body.id = clash.id;
  const r = await fetch(`/api/jobs/${poseState.job}/poses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const record = await r.json();
  if (!r.ok) {
    alert(record.detail ?? "could not save the pose");
    return;
  }
  poseState.current = record.id;
  await refreshPoses();
});

// --- sprite sheets ----------------------------------------------------------
//
// Two renderers of the same thing. The strip in the panel is three.js drawing
// the live scene through an orthographic camera at the eight yaws the server
// will use, so framing, elevation and the flat/lit choice can be judged before
// committing to a Blender render. The sheet that ships is the Blender one --
// this is a preview, never the product.

const sheetToggle = document.getElementById("sheet-toggle");
const sheetSetup = document.getElementById("sheet-setup");
const sheetPreview = document.getElementById("sheet-preview");
const sheetSummary = document.getElementById("sheet-summary");
const sheetRows = document.getElementById("sheet-rows");
const sheetLighting = document.getElementById("sheet-lighting");
const sheetSize = document.getElementById("sheet-size");
const sheetElevation = document.getElementById("sheet-elevation");
const sheetElevationOut = document.getElementById("sheet-elevation-out");
const sheetList = document.getElementById("sheet-list");

const PREVIEW_CELL = 64;
const sheetOptions = { yaws: [], frame_sizes: [], lighting: [], defaults: {} };
let sheetsSeen = "";

async function loadSheetOptions() {
  try {
    const body = await (await fetch("/api/sheets/options")).json();
    Object.assign(sheetOptions, body);
  } catch (e) {
    console.error("could not load sheet options", e);
    return;
  }
  for (const value of sheetOptions.lighting) {
    sheetLighting.append(new Option(value === "flat" ? "flat (unlit)" : "lit", value));
  }
  for (const value of sheetOptions.frame_sizes) {
    sheetSize.append(new Option(`${value} px`, value));
  }
  sheetLighting.value = sheetOptions.defaults.lighting;
  sheetSize.value = sheetOptions.defaults.frame_size;
  sheetElevation.value = sheetOptions.defaults.elevation;
  sheetElevationOut.textContent = `${sheetElevation.value}°`;
}

// Rendered to a target rather than to the visible canvas: reading the pixels
// back is the only way to composite eight views without the viewer flickering
// through all of them, and a render target has real alpha whether or not the
// page canvas does.
let previewTarget = null;

function renderSheetPreview() {
  if (sheetSetup.hidden || !model) return;
  const yaws = sheetOptions.yaws.length ? sheetOptions.yaws : [0, 45, 90, 135, 180, 225, 270, 315];
  const flat = sheetLighting.value === "flat";
  const elevation = THREE.MathUtils.degToRad(Number(sheetElevation.value));

  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  // The widest the subject can look from any yaw, matching op_sheet.
  const extent = Math.max(Math.hypot(size.x, size.z), size.y, 1e-6) * 1.12;
  const distance = extent * 2;
  const half = extent / 2;
  const cam = new THREE.OrthographicCamera(-half, half, half, -half, 0.001, distance * 4);

  previewTarget ??= new THREE.WebGLRenderTarget(PREVIEW_CELL, PREVIEW_CELL);
  const restore = beginPreviewScene(flat);
  const ctx = sheetPreview.getContext("2d");
  sheetPreview.width = PREVIEW_CELL * yaws.length;
  sheetPreview.height = PREVIEW_CELL;
  ctx.clearRect(0, 0, sheetPreview.width, sheetPreview.height);
  const pixels = new Uint8Array(PREVIEW_CELL * PREVIEW_CELL * 4);
  const cell = ctx.createImageData(PREVIEW_CELL, PREVIEW_CELL);

  try {
    yaws.forEach((yaw, i) => {
      // Blender's -Y front becomes +Z once the GLB is exported Y-up, so yaw 0
      // sits on +Z here and turns the same way the server's does.
      const a = THREE.MathUtils.degToRad(yaw);
      cam.position.set(
        centre.x + distance * Math.sin(a) * Math.cos(elevation),
        centre.y + distance * Math.sin(elevation),
        centre.z + distance * Math.cos(a) * Math.cos(elevation),
      );
      cam.lookAt(centre);
      renderer.setRenderTarget(previewTarget);
      renderer.setClearAlpha(0);
      renderer.clear();
      renderer.render(scene, cam);
      renderer.readRenderTargetPixels(previewTarget, 0, 0, PREVIEW_CELL, PREVIEW_CELL, pixels);
      // GL reads bottom-up; the canvas wants top-down.
      for (let y = 0; y < PREVIEW_CELL; y++) {
        const src = (PREVIEW_CELL - 1 - y) * PREVIEW_CELL * 4;
        cell.data.set(pixels.subarray(src, src + PREVIEW_CELL * 4), y * PREVIEW_CELL * 4);
      }
      ctx.putImageData(cell, i * PREVIEW_CELL, 0);
    });
  } finally {
    renderer.setRenderTarget(null);
    renderer.setClearAlpha(1);
    restore();
  }

  const rows = Math.max(sheetRows.selectedOptions.length, 1);
  const px = Number(sheetSize.value);
  setText(
    sheetSummary,
    `${rows} × ${yaws.length} cells · ${px * yaws.length}×${px * rows} px`,
  );
}

// Everything the sheet must not contain: the grid, the joint handles, the
// gizmo, and the viewer's background colour.
function beginPreviewScene(flat) {
  const undo = [];
  const hidden = [grid, markers, gizmo?.getHelper()].filter(Boolean);
  for (const obj of hidden) {
    const was = obj.visible;
    obj.visible = false;
    undo.push(() => { obj.visible = was; });
  }
  const background = scene.background;
  scene.background = null;
  undo.push(() => { scene.background = background; });
  if (flat) {
    model.traverse((obj) => {
      if (!obj.isMesh || Array.isArray(obj.material)) return;
      const original = obj.material;
      // Unlit but still textured -- the same thing _make_flat does in Blender
      // by driving an Emission node from whatever fed Base Color.
      obj.material = new THREE.MeshBasicMaterial({
        map: original.map ?? null,
        color: original.color ?? 0xffffff,
        // No `skinning: true` -- it has been implicit since three r151 and
        // passing it now only earns a console warning per material.
        transparent: original.transparent,
        side: original.side,
      });
      undo.push(() => {
        obj.material.dispose();
        obj.material = original;
      });
    });
  }
  return () => { for (const fn of undo.reverse()) fn(); };
}

async function refreshSheets() {
  if (!poseState.job) return;
  let sheets = [];
  try {
    sheets = (await (await fetch(`/api/jobs/${poseState.job}/sheets`)).json()).sheets ?? [];
  } catch (e) {
    console.error("could not load sheets", e);
  }
  sheetList.replaceChildren();
  for (const sheet of sheets) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    setText(label, `${sheet.name || sheet.id} · ${sheet.rows}×${sheet.columns} @ ${sheet.frame_size}`);
    label.title = `${sheet.lighting}, elevation ${sheet.elevation}°`;
    const png = document.createElement("a");
    setText(png, "png");
    png.download = "";
    png.href = `/api/jobs/${poseState.job}/sheets/${sheet.id}/sheet.png`;
    const json = document.createElement("a");
    setText(json, "json");
    json.download = `${sheet.id}.json`;
    json.href = `/api/jobs/${poseState.job}/sheets/${sheet.id}`;
    const del = document.createElement("button");
    setText(del, "×");
    del.title = "Delete this sheet";
    del.addEventListener("click", async () => {
      if (!confirm("Delete this sprite sheet?")) return;
      del.disabled = true;
      try {
        await fetch(`/api/jobs/${poseState.job}/sheets/${sheet.id}`, { method: "DELETE" });
      } finally {
        del.disabled = false;
      }
      await refreshSheets();
    });
    li.append(label, png, json, del);
    sheetList.append(li);
  }
}

// The row picker lists the poses already loaded for this job, so it can never
// offer a pose the server would reject.
function syncSheetRows() {
  const chosen = new Set([...sheetRows.selectedOptions].map((o) => o.value));
  sheetRows.replaceChildren();
  for (const record of poseState.saved) {
    const option = new Option(record.name, record.id);
    option.selected = chosen.has(record.id);
    sheetRows.append(option);
  }
}

sheetToggle.addEventListener("click", () => {
  sheetSetup.hidden = !sheetSetup.hidden;
  sheetToggle.classList.toggle("active", !sheetSetup.hidden);
  if (!sheetSetup.hidden) {
    syncSheetRows();
    renderSheetPreview();
  }
});

for (const el of [sheetLighting, sheetSize, sheetRows]) {
  el.addEventListener("change", renderSheetPreview);
}
sheetElevation.addEventListener("input", () => {
  sheetElevationOut.textContent = `${sheetElevation.value}°`;
  renderSheetPreview();
});

document.getElementById("sheet-render").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const fd = new FormData();
    for (const option of sheetRows.selectedOptions) fd.append("poses", option.value);
    fd.set("elevation", sheetElevation.value);
    fd.set("frame_size", sheetSize.value);
    fd.set("lighting", sheetLighting.value);
    const r = await fetch(`/api/jobs/${poseState.job}/sheets`, { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) alert(body.detail ?? "could not queue the sheet");
  } finally {
    btn.disabled = false;
  }
  poll(true);
});

// --- downloads --------------------------------------------------------------
// GLB is already on disk, but OBJ and STL are converted by the server on first
// request. A plain <a download> gives no sign that anything is happening, so
// fetch them by hand and show the wait. The href stays set so middle-click and
// "save link as" still work.

// A posed GLB is the same story: the server runs Blender to bake it.
function bindBusyDownload(anchor) {
  anchor.addEventListener("click", async (e) => {
    e.preventDefault();
    if (anchor.classList.contains("busy")) return;
    anchor.classList.add("busy");
    try {
      const r = await fetch(anchor.href);
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        alert(body.detail ?? `could not prepare ${anchor.textContent}`);
        return;
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const tmp = document.createElement("a");
      tmp.href = url;
      // The server sets the real name in Content-Disposition, but a blob URL
      // drops it, so rebuild it from the request path.
      tmp.download = anchor.href
        .split("/")
        .slice(-3)
        .filter((s) => s !== "files" && s !== "poses")
        .join("_");
      tmp.click();
      // Revoking synchronously can cancel a download that hasn't started yet.
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (err) {
      alert(`download failed: ${err}`);
    } finally {
      anchor.classList.remove("busy");
    }
  });
}

for (const id of ["dl-obj", "dl-stl"]) bindBusyDownload(document.getElementById(id));

// --- storage ----------------------------------------------------------------

const storageRow = document.getElementById("storage");
const storageText = document.getElementById("storage-text");

function formatBytes(n) {
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${(n / 1024 ** 3).toFixed(1)} GB`;
}

async function refreshStorage() {
  try {
    const s = await (await fetch("/api/storage")).json();
    if (!s.job_dirs) {
      storageRow.hidden = true;
      return;
    }
    setText(storageText, `${s.job_dirs} job${s.job_dirs === 1 ? "" : "s"} · ${formatBytes(s.bytes)} on disk`);
    storageRow.hidden = false;
  } catch {
    storageRow.hidden = true;   // never let a failed readout break the page
  }
}

document.getElementById("prune").addEventListener("click", async () => {
  const keep = 20;
  if (!confirm(`Delete all but the newest ${keep} jobs, including their files?`)) return;
  const btn = document.getElementById("prune");
  btn.disabled = true;
  try {
    const fd = new FormData();
    fd.set("keep", String(keep));
    const r = await fetch("/api/jobs/prune", { method: "POST", body: fd });
    const body = await r.json();
    if (!r.ok) alert(body.detail ?? "prune failed");
  } finally {
    btn.disabled = false;
  }
  await refreshStorage();
  poll(true);
});

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

// The storage readout walks every job directory, so it rides a much slower
// cadence than the list it sits under -- roughly once a minute at idle.
const STORAGE_EVERY = 10;
let listTick = 0;

async function refreshList() {
  const jobs = await (await fetch("/api/jobs", { signal: abort.signal })).json();
  renderJobs(jobs);
  if (listTick++ % STORAGE_EVERY === 0) refreshStorage();
  // A finished sheet job means a new file in the source job's directory, and
  // nothing else would tell the panel about it.
  if (poseState.job) {
    const signature = jobs
      .filter((j) => j.kind === "sheet" && j.params?.source_job === poseState.job)
      .map((j) => `${j.id}:${j.status}`)
      .join(",");
    if (signature !== sheetsSeen) {
      sheetsSeen = signature;
      refreshSheets();
    }
  }
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
loadRig();
loadSheetOptions();

(function loop() {
  poll().finally(() => setTimeout(loop, liveJobId || pending ? FAST_MS : IDLE_MS));
})();
