import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

// --- viewer -----------------------------------------------------------------

const canvas = document.getElementById("viewer");
// preserveDrawingBuffer is what makes canvas.toBlob return the frame that was
// just drawn rather than a cleared buffer. It costs a little fill rate and is
// the only way to snapshot a thumbnail without a second offscreen renderer.
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
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
  // The compare canvas is sized by CSS (50% each), so it is measured the same
  // way rather than by halving anything here -- one definition of the split.
  if (comparing && rendererB) {
    const bw = canvasB.clientWidth, bh = canvasB.clientHeight;
    if (bw > 0 && bh > 0 && (canvasB.width !== bw || canvasB.height !== bh)) {
      rendererB.setSize(bw, bh, false);
      cameraB.aspect = bw / bh;
      cameraB.updateProjectionMatrix();
    }
  }
}
new ResizeObserver(resize).observe(canvas);

renderer.setAnimationLoop(() => {
  resize();
  controls.update();
  syncJointMarkers();
  renderer.render(scene, camera);
  if (comparing && rendererB) {
    // One camera state, two renders: the whole point of the compare view is
    // that the two meshes are seen from the identical angle, so the second
    // camera copies the first rather than having controls of its own.
    cameraB.position.copy(camera.position);
    cameraB.quaternion.copy(camera.quaternion);
    cameraB.near = camera.near;
    cameraB.far = camera.far;
    cameraB.updateProjectionMatrix();
    rendererB.render(sceneB, cameraB);
  }
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

// Camera framing, split out of showModel so the "frame" button and the F key
// can put a lost camera back without reloading the mesh. Returns the bounding
// radius, which is what the pose editor sizes its joint markers against.
//
// Everything here is derived from the bounding box rather than fixed: guidance.py
// sizes a model anywhere from 0.01 m to 100 m, so the old hardcoded near/far of
// 0.01/100 and 4 m grid only ever framed a ~1 m prop.
function frameModel(root = model) {
  if (!root) return 0;
  const box = new THREE.Box3().setFromObject(root);
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
  return radius;
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

    const size = box.getSize(new THREE.Vector3());
    const radius = frameModel(model);

    // A power-of-ten span that comfortably contains the footprint.
    const span = Math.pow(10, Math.ceil(Math.log10(Math.max(size.x, size.z) * 2.5)));
    scene.remove(grid);
    grid.dispose();
    grid = new THREE.GridHelper(span, 16, 0x2c2f3a, 0x232530);
    scene.add(grid);

    // Wireframe is a viewer mode, not a property of one mesh: a model loaded
    // while it is on must come in wireframed too.
    applyWireframe();
    updateStats(model);
    onReady?.(model, radius);
    hideOverlay();
  }, undefined, () => {
    showOverlayError("Could not load the generated model.");
  });
}

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
  rendererB.toneMapping = renderer.toneMapping;
  rendererB.toneMappingExposure = renderer.toneMappingExposure;
  rendererB.outputColorSpace = renderer.outputColorSpace;
  sceneB = new THREE.Scene();
  // The environment probe is a render target built once at startup; sharing it
  // rather than generating a second one keeps the two halves lit identically,
  // which is the only way a side-by-side comparison means anything.
  sceneB.environment = scene.environment;
  sceneB.environmentIntensity = scene.environmentIntensity;
  sceneB.background = scene.background;
  cameraB = new THREE.PerspectiveCamera(camera.fov, 1, camera.near, camera.far);
  sceneB.add(new THREE.HemisphereLight(0xffffff, 0x223, 0.3));
  const keyB = new THREE.DirectionalLight(0xffffff, 1.5);
  keyB.position.copy(key.position);
  sceneB.add(keyB);
  sceneB.add(new THREE.GridHelper(4, 16, 0x2c2f3a, 0x232530));
}

async function compareWith(jobId) {
  if (poseState.editing && !confirmDiscardEdits("start a comparison")) return;
  if (poseState.editing) exitPoseEditing();
  ensureCompare();
  comparing = jobId;
  document.querySelector("main").classList.add("comparing");
  if (modelB) { sceneB.remove(modelB); disposeModel(modelB); modelB = null; }
  const gltf = await loader.loadAsync(`/api/jobs/${jobId}/files/model.glb`);
  modelB = gltf.scene;
  // Placed the same way showModel places the left-hand mesh: centred on the
  // origin and resting on the grid, or the two would not line up.
  const box = new THREE.Box3().setFromObject(modelB);
  const centre = box.getCenter(new THREE.Vector3());
  modelB.position.set(-centre.x, -box.min.y, -centre.z);
  sceneB.add(modelB);
  const selectedJob = jobsById.get(selected);
  const comparedJob = jobsById.get(jobId);
  setText(document.getElementById("compare-label-a"), `A · ${assetName(selectedJob)}`);
  setText(document.getElementById("compare-label-b"), `B · ${assetName(comparedJob)}`);
  document.getElementById("compare-label-a").hidden = false;
  document.getElementById("compare-label-b").hidden = false;
  document.getElementById("compare-exit").hidden = false;
  resize();
  refreshCompareButtons();
}

function stopComparing() {
  comparing = null;
  document.querySelector("main").classList.remove("comparing");
  document.getElementById("compare-label-a").hidden = true;
  document.getElementById("compare-label-b").hidden = true;
  document.getElementById("compare-exit").hidden = true;
  if (modelB) { sceneB.remove(modelB); disposeModel(modelB); modelB = null; }
  resize();
  refreshCompareButtons();
}

document.getElementById("compare-exit").addEventListener("click", stopComparing);

// Toggling compare must relabel the buttons now, not on the next poll: a
// button that still says "compare" after the pane opened reads as a no-op.
function refreshCompareButtons() {
  for (const [id, n] of nodes) {
    const job = jobsById.get(id);
    if (!job) continue;
    n.compare.hidden =
      job.status !== "done" || !job.files.includes("model.glb") || id === selected;
    setText(n.compare, comparing === id ? "Exit comparison" : "Compare with selected");
  }
}

// --- viewer tools -----------------------------------------------------------

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
      `${s.size.x.toFixed(2)} × ${s.size.y.toFixed(2)} × ${s.size.z.toFixed(2)} m`,
  );
}

let wireframe = false;

function applyWireframe() {
  if (!model) return;
  model.traverse((o) => {
    if (!o.isMesh) return;
    // An array covers multi-material meshes, which a joined trellis export can
    // still be after an OBJ round-trip.
    for (const m of [].concat(o.material)) m.wireframe = wireframe;
  });
}

document.getElementById("tool-wire").addEventListener("click", (e) => {
  wireframe = !wireframe;
  const button = e.currentTarget;
  button.classList.toggle("on", wireframe);
  button.setAttribute("aria-pressed", String(wireframe));
  setText(button, `Wireframe: ${wireframe ? "on" : "off"}`);
  applyWireframe();
});

const spinButton = document.getElementById("tool-spin");
spinButton.addEventListener("click", (e) => {
  // OrbitControls' own autoRotate, not a manual rotation of the model: rotating
  // the model would move the thing the pose gizmo and the joint markers are
  // positioned against.
  controls.autoRotate = !controls.autoRotate;
  const button = e.currentTarget;
  button.classList.toggle("on", controls.autoRotate);
  button.setAttribute("aria-pressed", String(controls.autoRotate));
  setText(button, `Turntable: ${controls.autoRotate ? "on" : "off"}`);
});

function stopSpinning() {
  controls.autoRotate = false;
  spinButton.classList.remove("on");
  spinButton.setAttribute("aria-pressed", "false");
  setText(spinButton, "Turntable: off");
}

document.getElementById("tool-frame").addEventListener("click", () => frameModel());

// The same canvas.toBlob the thumbnail upload already uses (the renderer is
// built with preserveDrawingBuffer for exactly this), pointed at the user
// instead of at the server.
document.getElementById("tool-shot").addEventListener("click", () => {
  if (!model) { toast("Open a model first."); return; }
  canvas.toBlob((blob) => {
    if (!blob) { toast("Could not capture the view."); return; }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `warlock_${selected ?? "view"}.png`;
    a.click();
    // Revoked on the next frame, not immediately: the click is dispatched
    // synchronously but the browser reads the blob after this turn.
    requestAnimationFrame(() => URL.revokeObjectURL(url));
  }, "image/png");
});

// --- viewer overlay ---------------------------------------------------------

const placeholder = document.getElementById("placeholder");
const phIdle = document.getElementById("ph-idle");
const phProgress = document.getElementById("ph-progress");
const phLabel = document.getElementById("ph-label");
const phDetail = document.getElementById("ph-detail");
const phTime = document.getElementById("ph-time");
const phBar = phProgress.querySelector(".bar");
const phFill = phBar.querySelector("i");
const phCancel = document.getElementById("ph-cancel");

// No confirm(): this button says exactly what it does, sits on the thing it
// acts on, and a blocking dialog would freeze the very progress bar behind it
// (which is why the card's confirm is worth avoiding here, not copying).
phCancel.addEventListener("click", async () => {
  const id = liveJobId;
  if (!id) return;
  phCancel.disabled = true;
  setText(phCancel, "Cancelling…");
  try {
    const r = await fetch(`/api/jobs/${id}/cancel`, { method: "POST" });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      toast(body.detail ?? "Could not cancel the job.");
    }
  } finally {
    phCancel.disabled = false;
  }
  poll(true);
});

function setText(el, s) { if (el.textContent !== s) el.textContent = s; }

// showModel() used to hide the placeholder permanently; the overlay lives inside
// it, so visibility has to be restored whenever a job starts.
function showOverlay() {
  // The progress view owns the pane; a reference image left showing under it
  // would narrate one job over another's picture.
  hideReference();
  placeholder.style.display = "grid";
  phIdle.hidden = true;
  phProgress.hidden = false;
}

function hideOverlay() {
  phProgress.hidden = true;
  phProgress.classList.remove("failed");
  phCancel.hidden = true;
  phIdle.hidden = false;
  // A visible reference preview owns the pane as much as a loaded mesh does;
  // dropping the idle text over either would read as a regression.
  if (model || !refPreview.hidden) placeholder.style.display = "none";
}

function showOverlayError(message) {
  showOverlay();
  phProgress.classList.add("failed");
  phCancel.hidden = true;
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
  "material", "condition", "setting", "palette", "emissive", "rarity", "silhouette", "mood",
];
const guidanceSelects = {};
const sizeInput = document.getElementById("g-size");
const platformHint = document.getElementById("platform-hint");
// The 3D pane's own copy of the platform field. Deliberately a second select
// rather than a shared one: in the 2D pane the field is a prompt fragment
// ("low-poly", "hero detail") and in the 3D pane it is the geometry resolution
// trellis runs at. One control cannot be owned by two panes, and the mesh form
// has to be overridable without editing the reference's recipe.
const meshPlatform = document.getElementById("m-platform");
const meshPlatformHint = document.getElementById("m-platform-hint");
const loraWeight = document.getElementById("g-lora-weight");
const loraWeightOut = document.getElementById("g-lora-weight-out");
let categorySizes = {};
let platformRes = {};
let loraDefaults = {};
// The catalog's own defaults, kept so the 3D pane can fall back to them for a
// source that never recorded a value (an upload has no params at all).
let guidanceDefaults = {};
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

function syncMeshPlatformHint() {
  const res = platformRes[meshPlatform.value];
  meshPlatformHint.textContent = res ? `${res} voxels` : "";
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
      // The 3D pane's platform select is filled from the same catalog pass, so
      // the two can never offer different tiers.
      if (field === "platform") meshPlatform.append(new Option(opt.label, opt.key));
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
  guidanceDefaults = catalog.defaults ?? {};
  guidanceSelects.platform.value = catalog.defaults.platform;
  meshPlatform.value = catalog.defaults.platform;
  guidanceSelects.base_model.value = catalog.defaults.base_model;
  sizeInput.value = catalog.defaults.size_m;
  loraWeight.min = catalog.lora_weight_range[0];
  loraWeight.max = catalog.lora_weight_range[1];
  loraWeight.value = catalog.defaults.lora_weight;
  guidanceSelects.category.addEventListener("change", syncSizeFromCategory);
  guidanceSelects.platform.addEventListener("change", syncPlatformHint);
  meshPlatform.addEventListener("change", syncMeshPlatformHint);
  guidanceSelects.style_lora.addEventListener("change", syncLoraWeight);
  loraWeight.addEventListener("input", () => {
    loraWeightOut.textContent = Number(loraWeight.value).toFixed(2);
  });
  const presetSelect = document.getElementById("preset");
  for (const preset of catalog.presets ?? []) {
    const opt = document.createElement("option");
    opt.value = preset.key;
    setText(opt, preset.label);
    presetSelect.append(opt);
  }
  presets = catalog.presets ?? [];
  syncPlatformHint();
  syncMeshPlatformHint();
  // platformRes only exists once this resolves, and the cost line reads it.
  syncSubmitCost();
  syncLoraWeight();
  // Last: the selects must exist and be populated before a stored value can be
  // checked against their options.
  restoreFormState();
}

loadGuidance().catch((e) => console.error("could not load guidance options", e));

// Refill the form from a finished job, so a recipe that worked is one click from
// being reused with a tweak. Only fields the form actually owns -- derived
// values (composed_prompt, scale_factor, mesh_audit) describe that run, not this
// one, exactly as the /rerun route already reasons.
// platform, base_model and bg_removal have no blank "-- any --"/"-- none --"
// option (see loadGuidance's option-building loop), so assigning "" to them
// would silently fall back to the browser's first-option default rather than
// showing blank -- they're only overwritten when the incoming params supply
// one. Every other field IS blanked when absent, or a value left over from a
// previously applied preset/job would silently survive into this one.
const NO_BLANK_OPTION_FIELDS = new Set(["platform", "base_model", "bg_removal"]);

function copySettingsToForm(job) {
  const p = job.params ?? {};
  if (job.prompt) document.getElementById("prompt").value = job.prompt;
  for (const field of GUIDANCE_FIELDS) {
    const select = guidanceSelects[field];
    if (!select) continue;
    if (NO_BLANK_OPTION_FIELDS.has(field)) {
      if (p[field]) select.value = p[field];
    } else {
      select.value = p[field] ?? "";
    }
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

// --- prompt preview ----------------------------------------------------
// A live token/chunk count so twelve guidance selects don't feel like a
// black box. Best-effort: /api/prompt-preview returns null tokens/chunks
// when the tokenizer isn't available, and this just shows the prompt text
// alone in that case.

let previewTimer = null;

async function refreshPromptPreview() {
  const el = document.getElementById("prompt-preview");
  if (mode !== "2d") { setText(el, ""); return; }
  const text = document.getElementById("prompt").value.trim();
  if (!text) { setText(el, ""); return; }

  const params = new URLSearchParams({ prompt: text });
  for (const field of GUIDANCE_FIELDS) {
    const value = guidanceSelects[field]?.value;
    if (value) params.set(field, value);
  }
  if (sizeInput.value) params.set("size_m", sizeInput.value);
  if (loraWeight.value) params.set("lora_weight", loraWeight.value);
  const negative = document.getElementById("negative-prompt")?.value;
  if (negative) params.set("negative_prompt", negative);

  let body;
  try {
    const r = await fetch(`/api/prompt-preview?${params}`);
    if (!r.ok) return;
    body = await r.json();
  } catch (e) {
    console.error("prompt preview failed", e);
    return;
  }
  const cost = body.tokens != null
    ? `${body.tokens} tokens / ${body.chunks} chunk${body.chunks === 1 ? "" : "s"}`
    : "token count unavailable";
  setText(el, `${body.prompt} — ${cost}`);
}

function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPromptPreview, 400);
}

document.getElementById("form").addEventListener("change", schedulePreview);
document.getElementById("prompt").addEventListener("input", schedulePreview);

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

// --- mode -------------------------------------------------------------------
//
// The app models one pipeline (text -> SDXL reference -> trellis -> mesh -> rig
// -> pose -> sheet) and used to model it as one form. It is really two stages
// with an approval between them, and the machinery for that split already
// existed: output=reference stops at the image and POST /api/jobs/{id}/model
// promotes it. The mode switch is that split made visible -- 2D owns every
// prompt control, 3D owns every mesh control, and neither shows the other's.

const MODE_KEY = "warlock.mode.v1";
const WORKSPACE_KEY = "warlock.workspace.v1";
const modeButtons = { "2d": document.getElementById("mode-2d"), "3d": document.getElementById("mode-3d") };
let mode = "2d";
// The finished 2D asset a "Make 3D" will promote. null means the 3D pane is
// working from an upload instead.
let source3d = null;

function setWorkspace(next, { remember = true } = {}) {
  if (!new Set(["settings", "preview", "inspector"]).has(next)) return;
  for (const name of ["settings", "preview", "inspector"]) {
    document.body.classList.toggle(`workspace-${name}`, name === next);
  }
  for (const button of document.querySelectorAll("[data-workspace]")) {
    button.setAttribute("aria-selected", String(button.dataset.workspace === next));
  }
  if (remember) localStorage.setItem(WORKSPACE_KEY, next);
  if (next === "preview") requestAnimationFrame(() => { resize(); frameModel(); });
}

for (const button of document.querySelectorAll("[data-workspace]")) {
  button.addEventListener("click", () => setWorkspace(button.dataset.workspace));
}

const inspectorToggle = document.getElementById("inspector-toggle");
function setInspectorOpen(open) {
  document.body.classList.toggle("inspector-open", open);
  inspectorToggle.setAttribute("aria-expanded", String(open));
  localStorage.setItem("warlock.inspector.v1", open ? "open" : "closed");
  if (open) document.getElementById("inspector").focus({ preventScroll: true });
}
inspectorToggle.addEventListener("click", () => setInspectorOpen(!document.body.classList.contains("inspector-open")));
document.getElementById("inspector-close").addEventListener("click", () => {
  setInspectorOpen(false);
  inspectorToggle.focus();
});

// What pressing Generate is about to cost. The queue is serial, so "8
// candidates" is eight runs waiting on each other and a higher geometry
// resolution is a longer reconstruction -- neither of which the form said
// anywhere. The sheet panel already does exactly this arithmetic for its own
// cell count; this is the same idea on the submit dock.
function syncSubmitCost() {
  const el = document.getElementById("submit-cost");
  if (mode === "2d") {
    const count = Number(document.getElementById("ref-count").value) || 1;
    setText(el, count === 1
      ? "One reference run."
      : `${count} candidates queue as ${count} separate runs, one at a time.`);
    return;
  }
  const res = platformRes[meshPlatform.value];
  setText(el, res
    ? `One reconstruction at ${res} voxels${res >= 1536 ? " — roughly twice the time of 1024" : ""}.`
    : "One reconstruction.");
}

for (const el of ["ref-count", "m-platform"]) {
  document.getElementById(el).addEventListener("change", syncSubmitCost);
}

function syncStageContext() {
  syncSubmitCost();
  const kicker = document.getElementById("stage-context-kicker");
  const detail = document.getElementById("stage-context");
  if (mode === "2d") {
    setText(kicker, "Stage 1 · Reference");
    const job = jobsById.get(selected);
    setText(detail, job?.stage === "reference"
      ? `Selected reference: ${assetName(job)}`
      : "Create reference candidates from your prompt.");
    return;
  }
  setText(kicker, "Stage 2 · Model");
  const source = source3d ? jobsById.get(source3d) : null;
  const file = document.getElementById("image")?.files?.[0];
  setText(detail, source
    ? `Selected reference: ${assetName(source)}`
    : file ? `Uploaded reference: ${file.name}` : "Choose a reference or upload an image.");
}

function setMode(next, { remember = true, userInitiated = false } = {}) {
  if (next !== "2d" && next !== "3d") return;
  if (userInitiated && next !== mode && poseState.editing) {
    if (!confirmDiscardEdits("change stages")) return;
    exitPoseEditing();
  }
  const changed = mode !== next;
  mode = next;
  document.body.classList.toggle("mode-2d", next === "2d");
  document.body.classList.toggle("mode-3d", next === "3d");
  modeButtons["2d"].classList.toggle("active", next === "2d");
  modeButtons["3d"].classList.toggle("active", next === "3d");
  modeButtons["2d"].setAttribute("aria-selected", String(next === "2d"));
  modeButtons["3d"].setAttribute("aria-selected", String(next === "3d"));
  modeButtons["2d"].tabIndex = next === "2d" ? 0 : -1;
  modeButtons["3d"].tabIndex = next === "3d" ? 0 : -1;
  setText(document.getElementById("generate"), next === "2d" ? "Generate references" : "Build 3D model");
  setText(
    document.getElementById("stage-description"),
    next === "2d"
      ? "Create reference candidates and choose the strongest image before building a mesh."
      : "Build a mesh from the selected reference or an uploaded image.",
  );
  syncStageContext();
  renderJobs([...jobsById.values()]);
  if (remember) localStorage.setItem(MODE_KEY, next);
  if (next === "3d" && changed) {
    // The canvas was visibility:hidden, not unmounted, so it kept its size --
    // but the pane widths can have changed under it, and a camera framed
    // against the old aspect reads as a collapsed or stretched viewport.
    requestAnimationFrame(() => { resize(); frameModel(); });
  }
  if (next === "2d") refreshPromptPreview();
}

for (const [key, btn] of Object.entries(modeButtons)) {
  btn.addEventListener("click", () => setMode(key, { userInitiated: true }));
  btn.addEventListener("keydown", (e) => {
    if (!new Set(["ArrowLeft", "ArrowRight"]).has(e.key)) return;
    e.preventDefault();
    const next = key === "2d" ? "3d" : "2d";
    setMode(next, { userInitiated: true });
    modeButtons[next].focus();
  });
}

// The source card in the 3D pane, and the mesh form seeded from it.
const sourceCard = document.getElementById("source-card");
const sourceThumb = document.getElementById("source-thumb");
const sourceName = document.getElementById("source-name");
const uploadZone = document.getElementById("image-input");

function setSource3d(job) {
  source3d = job?.id ?? null;
  sourceCard.hidden = !job;
  uploadZone.hidden = Boolean(job);
  if (job) {
    const src = `/api/jobs/${job.id}/files/input.png`;
    if (sourceThumb.getAttribute("src") !== src) sourceThumb.src = src;
    setText(sourceName, job.name || job.prompt || job.id);
    sourceThumb.alt = `Reference for ${assetName(job)}`;
  }
  // Reseeded either way: clearing the source means the next mesh comes from an
  // upload, which has no params to inherit and so wants the catalog defaults.
  seedMeshForm(job);
  syncStageContext();
}

// 3D mode has no prompt controls, so the mesh form seeds itself from whatever
// the source recorded and falls back to the catalog otherwise. An upload has no
// params at all, which is exactly the all-fallbacks case.
function seedMeshForm(job) {
  const p = job?.params ?? {};
  const platform = p.platform ?? guidanceDefaults.platform;
  if (platform && [...meshPlatform.options].some((o) => o.value === platform)) {
    meshPlatform.value = platform;
  }
  syncMeshPlatformHint();
  const size = p.size_m ?? categorySizes[p.category] ?? guidanceDefaults.size_m;
  if (size !== undefined) sizeInput.value = size;
  const bg = p.bg_removal ?? guidanceDefaults.bg_removal;
  const bgSelect = guidanceSelects.bg_removal;
  if (bg && bgSelect && [...bgSelect.options].some((o) => o.value === bg)) bgSelect.value = bg;
  // The budget is a property of this run, not of the reference: raw is the only
  // qualified tier, and a retarget is one POST away on a finished mesh.
  document.getElementById("g-profile").value = "raw";
  rigEnable.checked = false;
}

// The one meaning of "make this reference into a mesh": hand it to the 3D pane
// with its settings pre-filled, and let the user decide before paying for it.
function makeThreeD(job) {
  if (poseState.editing && !confirmDiscardEdits("choose a different reference")) return;
  if (poseState.editing) exitPoseEditing();
  setSource3d(job);
  setMode("3d");
  setWorkspace("settings");
}

document.getElementById("source-clear").addEventListener("click", () => setSource3d(null));

// --- job submission ---------------------------------------------------------

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

// Follow a job the page just created instead of making the user find it.
function followJob(id) {
  selected = id;
  shownModelFor = null;
  // Keep polling fast until the worker picks this up, otherwise the bar
  // would not start moving until the next idle tick.
  pending = id;
  downloads.style.display = "none";
  phProgress.classList.remove("failed");
  setText(phLabel, "Queued");
  setText(phDetail, "");
  setText(phTime, "");
  phBar.classList.add("indet");
  phFill.style.width = "0%";
  showOverlay();
  // Asked here, not on page load: an unprompted permission dialog on
  // arrival is the one everybody dismisses without reading.
  if (window.Notification?.permission === "default") Notification.requestPermission();
}

// The mesh-side settings, sent by both 3D paths: they are Form() overrides on
// the promote route and ordinary fields on an image job, and the names match
// so one builder serves both.
function meshFields(fd) {
  fd.set("platform", meshPlatform.value);
  if (sizeInput.value) fd.set("size_m", sizeInput.value);
  if (guidanceSelects.bg_removal?.value) fd.set("bg_removal", guidanceSelects.bg_removal.value);
  fd.set("profile", document.getElementById("g-profile").value);
  if (rig.available) {
    // Always sent as an explicit boolean: the promote route treats an absent
    // rig as "inherit", so unchecking the box must send "false" to clear an
    // inherited rig request rather than silently re-queueing it.
    fd.set("rig", String(rigEnable.checked));
    if (rigEnable.checked) fd.set("rig_template", rigTemplate.value);
  }
}

const formErrors = document.getElementById("form-errors");

function clearValidation() {
  formErrors.hidden = true;
  setText(formErrors, "");
  for (const field of document.querySelectorAll("#form [aria-invalid='true']")) {
    field.removeAttribute("aria-invalid");
    field.removeAttribute("aria-describedby");
  }
  for (const error of document.querySelectorAll("#form .field-error")) {
    error.hidden = true;
    setText(error, "");
  }
}

// A field the user cannot see cannot be fixed. #seed and #negative-prompt live
// inside the collapsed <details id="advanced-settings">, and the settings pane
// itself is off-screen on a narrow layout -- so "Fix this field" used to point
// at nothing, and the hidden-check below missed it because a closed <details>
// is not [hidden]. Open every ancestor disclosure and bring the pane forward
// before focusing anything.
function revealField(el) {
  if (!el) return;
  for (let node = el.parentElement; node; node = node.parentElement) {
    if (node.tagName === "DETAILS") node.open = true;
  }
  if (el.closest("#settings")) setWorkspace("settings");
}

function reportValidation(errors, { focus = true } = {}) {
  clearValidation();
  if (!errors.length) return true;
  for (const { field, message } of errors) {
    const input = document.getElementById(field);
    const output = document.querySelector(`[data-error-for="${field}"]`);
    // Every flagged field, not only the focused one: the message beside a
    // field inside a closed disclosure is as invisible as the field.
    revealField(input);
    if (input) {
      input.setAttribute("aria-invalid", "true");
      if (output) {
        output.id ||= `${field}-error`;
        input.setAttribute("aria-describedby", output.id);
      }
    }
    if (output) { setText(output, message); output.hidden = false; }
  }
  setText(formErrors, `Fix ${errors.length === 1 ? "this field" : `${errors.length} fields`} before continuing.`);
  formErrors.hidden = false;
  if (focus) {
    const first = document.getElementById(errors[0].field);
    // revealField has already opened whatever was merely collapsed; anything
    // still [hidden] belongs to the other mode's pane and cannot be shown.
    if (!first || first.hidden || first.closest("[hidden]")) formErrors.focus();
    else first.focus();
  }
  return false;
}

function reportRequestError(message) {
  clearValidation();
  setText(formErrors, message);
  formErrors.hidden = false;
  formErrors.focus();
}

function validateSeed(errors) {
  if (!seedInput.value) return;
  const seed = Number(seedInput.value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 2 ** 31 - 1) {
    errors.push({ field: "seed", message: "Use a whole-number seed from 0 to 2,147,483,647." });
  }
}

function validate2d() {
  const errors = [];
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) errors.push({ field: "prompt", message: "Describe the reference you want to create." });
  else if (prompt.length > 1000) {
    errors.push({ field: "prompt", message: `Shorten the prompt by ${prompt.length - 1000} characters.` });
  }
  validateSeed(errors);
  return reportValidation(errors);
}

function validate3d() {
  const errors = [];
  const file = document.getElementById("image").files[0];
  const source = source3d ? jobsById.get(source3d) : null;
  if (!source && !file) {
    errors.push({ field: "image", message: "Choose a finished reference asset or upload an image." });
  } else if (source && (source.status !== "done" || !source.files?.includes("input.png"))) {
    errors.push({ field: "image", message: "The selected reference is not ready. Choose a finished reference." });
  } else if (file && !file.type.startsWith("image/")) {
    errors.push({ field: "image", message: "Choose a PNG, JPEG, WebP, or another supported image file." });
  } else if (file && file.size > 20 * 1024 * 1024) {
    errors.push({ field: "image", message: "The source image must be 20 MB or smaller." });
  }
  const size = Number(sizeInput.value);
  if (!sizeInput.value || !Number.isFinite(size) || size < 0.01 || size > 100) {
    errors.push({ field: "g-size", message: "Enter a physical size from 0.01 to 100 metres." });
  }
  if (!source3d) validateSeed(errors);
  return reportValidation(errors);
}

// 2D always stops at a reference: promotion is the 3D pane's job, and it is the
// only thing that decides how the mesh is built.
async function submit2d() {
  const prompt = document.getElementById("prompt").value.trim();
  if (!validate2d()) return false;
  const fd = new FormData();
  // The same field set the form sends, kept aside so a successful submit can be
  // replayed from the history list by exactly the recipe that produced it.
  const submittedFields = {};
  fd.set("kind", "text");
  fd.set("seed", seedInput.value || String(newSeed()));
  // No explicit resolution: the platform preset supplies it server-side.
  for (const field of GUIDANCE_FIELDS) {
    // bg_removal's select lives in the 3D pane; a 2D submit must not bake in
    // state the 2D pane can't see. Same for size_m below it.
    if (field === "bg_removal") continue;
    const value = guidanceSelects[field]?.value;
    if (!value) continue;
    fd.set(field, value);
    submittedFields[field] = value;
  }
  if (guidanceSelects.style_lora?.value) fd.set("lora_weight", loraWeight.value);
  fd.set("prompt", prompt);
  fd.set("negative_prompt", document.getElementById("negative-prompt").value);
  fd.set("output", "reference");
  fd.set("count", document.getElementById("ref-count").value);
  const r = await fetch("/api/jobs", { method: "POST", body: fd });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    reportRequestError(body.detail ?? "Could not start reference generation.");
    return false;
  }
  followJob(body.id);
  // Next submit gets a different reference unless the user asked to pin it.
  if (!seedLock.checked) rollSeed();
  recordSubmission(prompt, submittedFields);
  saveFormState();
  return true;
}

// 3D either promotes the selected 2D asset (the normal path) or starts an
// ordinary image job from an upload. Both carry the same mesh overrides.
async function submit3d() {
  if (!validate3d()) return false;
  const fd = new FormData();
  meshFields(fd);
  let url;
  if (source3d) {
    url = `/api/jobs/${source3d}/model`;
  } else {
    const file = document.getElementById("image").files[0];
    fd.set("kind", "image");
    fd.set("seed", seedInput.value || String(newSeed()));
    fd.set("image", file);
    url = "/api/jobs";
  }
  const r = await fetch(url, { method: "POST", body: fd });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    reportRequestError(body.detail ?? "Could not start the 3D stage.");
    return false;
  }
  followJob(body.id);
  // The upload path consumed a seed like any 2D submit; the promote path
  // didn't (the mesh seed is inherited or overridden server-side).
  if (!source3d && !seedLock.checked) rollSeed();
  return true;
}

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("generate");
  btn.disabled = true;
  document.getElementById("form").setAttribute("aria-busy", "true");
  try {
    if (mode === "2d") await submit2d();
    else await submit3d();
  } finally {
    btn.disabled = false;
    document.getElementById("form").removeAttribute("aria-busy");
  }
  poll(true);
});

for (const field of ["prompt", "seed", "g-size", "image"]) {
  document.getElementById(field).addEventListener("input", () => {
    const input = document.getElementById(field);
    const output = document.querySelector(`[data-error-for="${field}"]`);
    input.removeAttribute("aria-invalid");
    input.removeAttribute("aria-describedby");
    if (output) output.hidden = true;
    formErrors.hidden = true;
    if (field === "image") syncStageContext();
  });
}

// --- reference input: drop and paste ----------------------------------------

// The upload zone was a bare <input type="file">. Dropping a reference or
// pasting one from a screenshot tool is how people actually get an image into
// this.
const imageInput = document.getElementById("image");
const dropZone = document.getElementById("image-input");

function acceptImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  // A DataTransfer is the only way to write to input.files, which is what the
  // submit handler reads -- so a dropped file and a chosen one are one path.
  const dt = new DataTransfer();
  dt.items.add(file);
  imageInput.files = dt.files;
  // An upload is a 3D source and it replaces any picked 2D asset.
  setSource3d(null);
  setMode("3d");
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

// --- keyboard shortcuts -----------------------------------------------------

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
  if (e.key === "Escape" && comparing) { stopComparing(); return; }
  if (e.key === "Escape" && poseState.editing) {
    if (confirmDiscardEdits("leave editing")) exitPoseEditing();
    return;
  }
  if (e.key === "f" || e.key === "F") frameModel();
  if (e.key === "w" || e.key === "W") document.getElementById("tool-wire").click();
  if (e.key === "s" || e.key === "S") document.getElementById("tool-spin").click();
});

// --- finish notifications ---------------------------------------------------

// Only for jobs that ran long enough that the user plausibly went elsewhere --
// a notification for a 4-second reference render is noise.
const NOTIFY_AFTER_SECONDS = 45;
const lastStatus = new Map();

function maybeNotify(job) {
  const ran = (job.finished_at ?? 0) - (job.started_at ?? 0);
  if (ran < NOTIFY_AFTER_SECONDS) return;
  const title = job.name || job.prompt || job.id;
  if (window.Notification?.permission === "granted") {
    new Notification(job.status === "done" ? "Model ready" : "Job failed", { body: title });
  }
  toast(`${job.status === "done" ? "Finished" : "Failed"}: ${title}`,
        job.status === "done" ? "ok" : "error");
}

// Only the transition is interesting: the list is re-rendered every few
// seconds, and a job stays 'done' forever after it lands.
function notifyTransitions(jobs) {
  for (const job of jobs) {
    const before = lastStatus.get(job.id);
    lastStatus.set(job.id, job.status);
    if (before === undefined || before === job.status) continue;
    if (job.status === "done" || job.status === "error") maybeNotify(job);
  }
}

// --- form state -------------------------------------------------------------

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

document.getElementById("form").addEventListener("change", saveFormState);

// --- presets and history ----------------------------------------------------

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

const HISTORY_KEY = "warlock.history.v1";
const HISTORY_MAX = 20;

function readHistory() {
  try {
    const entries = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(entries) ? entries : [];
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

renderHistory();

// --- job list ---------------------------------------------------------------

const ul = document.getElementById("jobs");
const downloads = document.getElementById("downloads");
const nodes = new Map();   // job id -> element refs, reused across polls
const jobsById = new Map();
let selected = null;
let shownModelFor = null;

function assetName(job) {
  if (!job) return "Unknown asset";
  return job.name || job.prompt || `${job.kind || "asset"} ${job.id}`;
}

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

// --- toasts -----------------------------------------------------------------

// alert() blocks the event loop, which in a page that polls every 600 ms means
// the progress bar freezes behind the dialog the user has to dismiss to see it.
function toast(message, kind = "error") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.role = kind === "error" ? "alert" : "status";
  setText(el, message);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 8000 : 4000);
}

// --- filtering --------------------------------------------------------------

const filters = {
  text: document.getElementById("filter-text"),
  stage: document.getElementById("filter-stage"),
  status: document.getElementById("filter-status"),
  kind: document.getElementById("filter-kind"),
  fav: document.getElementById("filter-fav"),
};
const FILTER_KEY = "warlock.filters.v2";

function saveFilters() {
  localStorage.setItem(FILTER_KEY, JSON.stringify({
    text: filters.text.value,
    stage: filters.stage.value,
    status: filters.status.value,
    kind: filters.kind.value,
    fav: filters.fav.checked,
  }));
}

function restoreFilters() {
  try {
    const saved = JSON.parse(localStorage.getItem(FILTER_KEY) || "null");
    if (!saved) return;
    for (const name of ["text", "stage", "status", "kind"]) {
      if (typeof saved[name] === "string") filters[name].value = saved[name];
    }
    filters.fav.checked = Boolean(saved.fav);
  } catch {
    // Corrupt preferences should never keep the library from rendering.
  }
}

function clearFilters() {
  filters.text.value = "";
  filters.stage.value = "";
  filters.status.value = "";
  filters.kind.value = "";
  filters.fav.checked = false;
  saveFilters();
  renderJobs([...jobsById.values()]);
  filters.text.focus();
}

restoreFilters();
for (const el of Object.values(filters)) {
  el.addEventListener("input", () => {
    saveFilters();
    renderJobs([...jobsById.values()]);
  });
}
document.getElementById("clear-filters").addEventListener("click", clearFilters);
document.getElementById("empty-clear-filters").addEventListener("click", clearFilters);

// Client-side on purpose: /api/jobs already returns the whole workshop, and a
// server-side search would be a second definition of "matches" that could
// disagree with what the list is showing.
function jobMatches(job) {
  // "model" is everything that is not a reference: a rig and a sheet job are
  // stored at the default stage, and they belong with the mesh they describe.
  if (filters.stage.value === "reference" && job.stage !== "reference") return false;
  if (filters.stage.value === "model" && job.stage === "reference") return false;
  if (filters.status.value && job.status !== filters.status.value) return false;
  if (filters.kind.value && job.kind !== filters.kind.value) return false;
  if (filters.fav.checked && !job.favorite) return false;
  const q = filters.text.value.trim().toLowerCase();
  if (!q) return true;
  return [job.name, job.prompt, job.tags, job.id]
    .filter(Boolean)
    .some((field) => String(field).toLowerCase().includes(q));
}

async function patchJob(id, body) {
  const r = await fetch(`/api/jobs/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    toast(detail.detail ?? "could not update the job");
  }
  return r.ok;
}

// A re-roll or a re-mesh, from a card or from the inspector. Module-level so
// both reach it by one path: "try another" on a card and in the inspector must
// not be able to mean two different requests.
// `seed` distinguishes the two things "run this again" can mean, which the API
// has always accepted and nothing ever sent: omitted is a re-roll (a fresh
// seed, a different result), and a given seed is a retry (the same recipe,
// which is what you want after a transient failure rather than a new gamble).
async function rerunJob(id, how, { seed = null } = {}) {
  const job = jobsById.get(id);
  const label = assetName(job);
  const retry = seed !== null && seed !== undefined;
  const message = retry
    ? `Run “${label}” again with seed ${seed}? This consumes another generation run.`
    : how === "remesh"
      ? `Build a new 3D mesh for “${label}”? This reuses the reference but consumes another full 3D run.`
      : `Generate another result for “${label}”? This consumes another generation run.`;
  if (!window.confirm(message)) return false;
  const fd = new FormData();
  fd.set("mode", how);
  if (retry) fd.set("seed", String(seed));
  const r = await fetch(`/api/jobs/${id}/rerun`, { method: "POST", body: fd });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    toast(body.detail ?? "rerun failed");
    return false;
  }
  followJob(body.id);
  poll(true);
  return true;
}

// Which number "the same seed" means. The API rewrites all three from one
// value on a reroll, and `seed` is the legacy fallback both stages resolve
// against, so it is the one that reproduces the original run.
function retrySeed(job) {
  const params = job?.params ?? {};
  const seed = params.seed ?? params.reference_seed ?? params.mesh_seed;
  return Number.isInteger(seed) ? seed : null;
}

function createNode(id) {
  const li = document.createElement("li");
  li.tabIndex = 0;
  li.role = "option";
  li.dataset.jobId = id;
  li.setAttribute("aria-selected", "false");
  const img = document.createElement("img");
  img.hidden = true;
  const info = document.createElement("div");
  const title = document.createElement("div");
  title.className = "job-title";
  const summaryLine = document.createElement("div");
  summaryLine.className = "job-summary";
  const status = document.createElement("div");
  const stage = document.createElement("div");
  stage.className = "job-stage";
  summaryLine.append(status, stage);
  const err = document.createElement("div");
  err.className = "job-error";
  const quality = document.createElement("div");
  quality.className = "job-quality";
  // rerun_of has been stored since re-roll shipped and shown nowhere. A variant
  // that does not say what it is a variant of is just another row.
  const lineage = document.createElement("div");
  lineage.className = "job-lineage";
  const settings = document.createElement("div");
  settings.className = "job-settings";
  settings.hidden = true;
  const settingsToggle = document.createElement("button");
  settingsToggle.type = "button";
  settingsToggle.className = "link";
  setText(settingsToggle, "Show generation settings");
  settingsToggle.setAttribute("aria-expanded", "false");
  settingsToggle.hidden = true;
  const bar = document.createElement("div");
  bar.className = "bar mini";
  bar.hidden = true;
  const fill = document.createElement("i");
  bar.append(fill);
  bar.role = "progressbar";
  bar.setAttribute("aria-label", "Job progress");
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  info.append(title, summaryLine, err, quality, lineage, settings, bar);
  const actions = document.createElement("div");
  actions.className = "job-actions";
  const primary = document.createElement("button");
  primary.type = "button";
  primary.className = "job-primary";
  const menu = document.createElement("details");
  menu.className = "job-menu";
  const menuSummary = document.createElement("summary");
  setText(menuSummary, "Actions");
  menuSummary.setAttribute("aria-label", "Asset actions");
  const menuPanel = document.createElement("div");
  menuPanel.className = "job-menu-panel";
  menu.append(menuSummary, menuPanel);
  const closeMenu = () => {
    menu.open = false;
    menuSummary.focus();
  };
  const reroll = document.createElement("button");
  reroll.type = "button";
  setText(reroll, "Generate another result");
  reroll.title = "Same prompt and settings, new seed";
  reroll.hidden = true;
  const remesh = document.createElement("button");
  remesh.type = "button";
  setText(remesh, "Rebuild 3D mesh");
  remesh.title = "Reuse this reference image, rerun only the 3D stage";
  remesh.hidden = true;
  const retry = document.createElement("button");
  retry.type = "button";
  setText(retry, "Retry with the same seed");
  retry.title = "Run this exact recipe again — the same prompt, settings and seed";
  retry.hidden = true;
  retry.addEventListener("click", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    if (!job) return;
    retry.disabled = true;
    try {
      await rerunJob(id, "reroll", { seed: retrySeed(job) });
    } finally {
      retry.disabled = false;
    }
    closeMenu();
    poll(true);
  });
  const rigBtn = document.createElement("button");
  rigBtn.type = "button";
  setText(rigBtn, "Fit a skeleton");
  rigBtn.title = "Fit a skeleton to this mesh";
  rigBtn.hidden = true;
  const another = document.createElement("button");
  another.type = "button";
  setText(another, "Generate another reference");
  another.title = "Same prompt and settings, new reference seed";
  another.hidden = true;
  const compare = document.createElement("button");
  compare.type = "button";
  setText(compare, "Compare with selected");
  compare.title = "Show this beside the selected model";
  compare.hidden = true;
  compare.addEventListener("click", (e) => {
    e.stopPropagation();
    if (comparing === id) stopComparing();
    else compareWith(id).catch((err) => toast(`could not load that model to compare: ${err}`));
    closeMenu();
  });
  const star = document.createElement("button");
  star.type = "button";
  star.className = "star";
  star.title = "Favorite";
  setText(star, "☆");
  star.addEventListener("click", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    await patchJob(id, { favorite: !job?.favorite });
    poll(true);
  });
  const rename = document.createElement("button");
  rename.type = "button";
  setText(rename, "Rename asset");
  rename.addEventListener("click", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    const next = window.prompt("Name this asset", job?.name || job?.prompt || "");
    if (next === null) return;
    await patchJob(id, { name: next });
    closeMenu();
    poll(true);
  });
  const act = document.createElement("button");
  act.type = "button";
  act.className = "danger";
  // Bulk-export selection. Independent of `selected` (which is the one job the
  // viewer is showing), so ticking a card must not also open it.
  const pick = document.createElement("input");
  pick.type = "checkbox";
  pick.className = "job-pick";
  pick.title = "Select for bulk export";
  pick.addEventListener("click", (e) => e.stopPropagation());
  pick.addEventListener("change", updateBulkBar);
  menuPanel.append(rename, another, reroll, retry, remesh, compare, rigBtn, settingsToggle, act);
  actions.append(star, pick, primary, menu);
  li.append(img, info, actions);

  li.addEventListener("click", () => select(id));
  li.addEventListener("keydown", (e) => {
    if (e.target !== li || !new Set(["Enter", " "]).has(e.key)) return;
    e.preventDefault();
    select(id);
  });
  primary.addEventListener("click", (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    if (!job) return;
    if (job.stage === "reference" && job.status === "done" && job.files.includes("input.png")) {
      makeThreeD(job);
    } else {
      select(id);
      if (window.matchMedia("(max-width: 900px)").matches) setWorkspace("preview");
    }
  });
  for (const [btn, how] of [[reroll, "reroll"], [remesh, "remesh"]]) {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      btn.disabled = true;
      try {
        await rerunJob(id, how);
      } finally {
        btn.disabled = false;
      }
      closeMenu();
      poll(true);
    });
  }
  another.addEventListener("click", async (e) => {
    e.stopPropagation();
    another.disabled = true;
    try {
      await rerunJob(id, "reroll");
    } finally {
      another.disabled = false;
    }
    closeMenu();
    poll(true);
  });
  rigBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    rigBtn.disabled = true;
    try {
      const fd = new FormData();
      fd.set("template", rigTemplate.value);
      const r = await fetch(`/api/jobs/${id}/rig`, { method: "POST", body: fd });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) toast(body.detail ?? "could not queue the rig");
    } finally {
      rigBtn.disabled = false;
    }
    closeMenu();
    poll(true);
  });
  settingsToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    settings.hidden = !settings.hidden;
    settingsToggle.setAttribute("aria-expanded", String(!settings.hidden));
    setText(settingsToggle, settings.hidden ? "Show generation settings" : "Hide generation settings");
    closeMenu();
  });
  act.addEventListener("click", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    if (!job) return;
    if (job.status === "cancelled" && job.progress) return;
    const active = job.status === "queued" || job.status === "running";
    const label = assetName(job);
    const question = active
      ? `Cancel the active job “${label}”? The current compute stage may take a moment to stop.`
      : `Permanently delete “${label}” and all of its generated files?`;
    if (!window.confirm(question)) return;
    if (active) {
      act.disabled = true;
      setText(act, "cancelling…");
    }
    try {
      const response = await fetch(`/api/jobs/${id}${active ? "/cancel" : ""}`, {
        method: active ? "POST" : "DELETE",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        toast(body.detail ?? (active ? "Could not cancel the job." : "Could not delete the asset."));
        return;
      }
    } finally {
      act.disabled = false;
    }
    if (!active && selected === id) {
      selected = null;
      hideOverlay();
      hideReference();
      setPoseJob(null);
      renderInspector(null);
      downloads.style.display = "none";
    }
    closeMenu();
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

  const n = { li, img, title, status, stage, err, quality, lineage, settings, settingsToggle,
              copySettings, bar, fill, act, reroll, retry, remesh, rigBtn, another, pick,
              star, compare, primary, menu, menuSummary };
  nodes.set(id, n);
  return n;
}

// --- bulk export ------------------------------------------------------------
//
// A selection of job ids POSTed to /api/export. Deliberately not fetch(): the
// browser handles the download itself, including the filename, instead of us
// buffering a 200 MB blob in the page.

const bulkBar = document.getElementById("bulk-bar");
const bulkCount = document.getElementById("bulk-count");
const bulkFolder = document.getElementById("bulk-folder");

function selectedIds() {
  return [...nodes].filter(([, n]) => n.pick.checked).map(([id]) => id);
}

function selectedExportFiles() {
  return [...document.querySelectorAll("input[name='bulk-file']:checked")].map((input) => input.value);
}

function updateBulkBar() {
  const n = selectedIds().length;
  bulkBar.hidden = n === 0;
  setText(bulkCount, `${n} selected`);
}

document.getElementById("bulk-zip").addEventListener("click", () => {
  const ids = selectedIds();
  if (!ids.length) return;
  const files = selectedExportFiles();
  if (!files.length) { toast("Choose at least one file type to export."); return; }
  const form = document.createElement("form");
  form.method = "POST";
  form.action = "/api/export";
  for (const id of ids) {
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "ids";
    input.value = id;
    form.append(input);
  }
  for (const file of files) {
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "files";
    input.value = file;
    form.append(input);
  }
  document.body.append(form);
  form.submit();
  form.remove();
});

bulkFolder.addEventListener("click", async () => {
  const ids = selectedIds();
  if (!ids.length) return;
  const files = selectedExportFiles();
  if (!files.length) { toast("Choose at least one file type to export."); return; }
  bulkFolder.disabled = true;
  try {
    const fd = new FormData();
    for (const id of ids) fd.append("ids", id);
    for (const file of files) fd.append("files", file);
    const r = await fetch("/api/export/folder", { method: "POST", body: fd });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) toast(body.detail ?? "export failed");
    else toast(`copied ${body.copied} file(s) into ${body.dir}`, "ok");
  } finally {
    bulkFolder.disabled = false;
  }
});

// meshaudit measures the fraction of the worst-case silhouette you can see
// straight through. ~0 is solid; the perforated crust trellis-server's default
// band produces measures 0.07-0.15. The thresholds mirror those bands.
// Two different measurements, deliberately not merged: meshaudit says how much
// of the silhouette you can see through (what a player notices) and meshreport
// says whether an engine will accept it (topology, materials, budget, pivot).
// Only the second may use the word watertight.
function qualityBadge(job) {
  const report = job.params?.mesh_report;
  if (report) {
    if (report.status === "invalid") {
      return { cls: "bad", text: "× Invalid mesh", reasons: report.reasons ?? [] };
    }
    if (report.status === "ready") {
      return { cls: "good", text: `✓ Ready · ${Number(report.triangles).toLocaleString()} tris` };
    }
    return {
      cls: "warn",
      text: `! Review · ${report.reasons.length} issue(s)`,
      reasons: report.reasons,
    };
  }
  const worst = job.params?.mesh_audit?.worst;
  if (typeof worst !== "number") return null;
  const pct = (worst * 100).toFixed(worst < 0.1 ? 1 : 0);
  if (worst < 0.02) return { cls: "good", text: "✓ No visible holes" };
  if (worst < 0.08) return { cls: "warn", text: `! ${pct}% see-through` };
  return { cls: "bad", text: `× ${pct}% see-through — try another seed` };
}

// What a job actually ran with. Already in the API response and never shown
// before: without it a good result is not reproducible, because the card only
// ever said what the user typed, not what was sent to the model. Shared by the
// card's collapsed readout and the inspector's.
function settingsRows(params) {
  return [
    ["seed", params.seed],
    ["reference seed", params.reference_seed],
    ["mesh seed", params.mesh_seed],
    ["model", params.base_model],
    ["style", params.style_lora && `${params.style_lora} @ ${params.lora_weight ?? "?"}`],
    ["resolution", params.resolution],
    ["size", params.size_m && `${params.size_m} m`],
    ["background", params.bg_removal],
    ["negative", params.negative_prompt],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");
}

function renderSettingsRows(target, rows) {
  target.replaceChildren();
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "settings-row";
    const k = document.createElement("span");
    setText(k, label);
    const v = document.createElement("span");
    setText(v, String(value));
    row.append(k, v);
    target.append(row);
  }
}

function updateNode(n, job) {
  n.li.classList.toggle("selected", job.id === selected);
  n.li.setAttribute("aria-selected", String(job.id === selected));
  n.li.setAttribute("aria-label", `${assetName(job)}, ${job.stage === "reference" ? "reference" : "model"}, ${job.status}`);

  // The mesh, not the reference image, is what the user is looking for in a
  // list of a hundred assets -- the reference only stands in until one exists.
  const preferred = job.files.includes("thumb.png") ? "thumb.png" : "input.png";
  const hasImage = job.files.includes(preferred);
  const src = `/api/jobs/${job.id}/files/${preferred}`;
  if (hasImage && n.img.getAttribute("src") !== src) n.img.src = src;
  n.img.hidden = !hasImage;
  n.img.alt = hasImage ? `Thumbnail for ${assetName(job)}` : "";

  // A rig job carries its source's prompt, so it needs the prefix or the
  // history reads as the same model having been generated twice. A user-given
  // name outranks all of it: naming a thing is how it stops being a prompt.
  setText(n.title, job.name || (job.kind === "rig"
    ? `rig · ${job.prompt ?? job.params?.source_job ?? job.id}`
    : job.prompt ?? `image job ${job.id}`));
  setText(n.star, job.favorite ? "★" : "☆");
  n.star.classList.toggle("on", Boolean(job.favorite));
  n.star.setAttribute("aria-pressed", String(Boolean(job.favorite)));
  n.star.setAttribute("aria-label", `${job.favorite ? "Remove" : "Add"} ${assetName(job)} ${job.favorite ? "from" : "to"} favorites`);
  n.pick.setAttribute("aria-label", `Select ${assetName(job)} for bulk export`);
  n.menuSummary.setAttribute("aria-label", `Actions for ${assetName(job)}`);

  const from = job.params?.rerun_of || job.parent_id;
  setText(n.lineage, from ? `variant of ${from.slice(0, 6)}` : "");

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
    n.bar.setAttribute("aria-valuenow", String(Math.max(0, Math.min(100, p.percent))));
  } else {
    const stageName = job.kind === "rig" ? "Rig job" : job.kind === "sheet" ? "Sprite sheet" :
      job.stage === "reference" ? "Stage 1 · Reference" : "Stage 2 · Model";
    setText(n.stage, stageName);
    n.bar.hidden = true;
    n.bar.removeAttribute("aria-valuenow");
  }

  // The line is one ellipsised row wide, so the full sentence goes in the
  // tooltip; the traceback behind it is in the inspector.
  const failed = job.status === "error";
  setText(n.err, failed && job.error ? job.error : "");
  if (failed && job.error) n.err.title = job.error;
  else n.err.removeAttribute("title");

  const badge = job.status === "done" ? qualityBadge(job) : null;
  n.quality.className = `job-quality ${badge ? badge.cls : ""}`;
  setText(n.quality, badge ? badge.text : "");
  n.quality.removeAttribute("title");

  const rows = settingsRows(job.params ?? {});
  n.settingsToggle.hidden = rows.length === 0;
  renderSettingsRows(n.settings, rows);
  // The button itself is created once in createNode (see the comment there);
  // rebuilding the rows every poll must not touch its listener.
  if (rows.length) n.settings.append(n.copySettings);

  const active = job.status === "queued" || job.status === "running";
  setText(n.act, cancelling ? "Cancellation pending" : active ? "Cancel active job" : "Delete permanently");
  n.act.disabled = cancelling;

  // Only offered on a settled job: re-rolling a queued one just races it, and
  // re-3D needs a reference image that survived the run. A *failed* job is
  // settled too, and is the case that most wants a re-run -- it had no retry
  // affordance at all, which left the API's own rerun route unreachable from
  // the one card that needed it.
  const done = job.status === "done";
  const settled = done || failed;
  const generated = job.kind !== "rig";
  const isReference = job.stage === "reference";
  n.reroll.hidden = !settled || !generated || isReference;
  n.retry.hidden = !settled || !generated || retrySeed(job) === null;
  n.remesh.hidden = !settled || !generated || !job.files.includes("input.png");
  // Offered once, on a finished mesh that isn't rigged yet. A rig job has no
  // mesh of its own, and re-rigging would silently overwrite the skeleton
  // whatever poses were saved against it.
  n.rigBtn.hidden =
    !rig.available || !done || !generated || !job.files.includes("model.glb") || isRigged(job);

  n.compare.hidden = !done || !job.files.includes("model.glb") || job.id === selected;
  setText(n.compare, comparing === job.id ? "Exit comparison" : "Compare with selected");

  n.another.hidden = !(isReference && settled);
  // A reference has no mesh, so the mesh-only actions stay hidden for it.
  n.remesh.hidden = n.remesh.hidden || isReference;
  n.rigBtn.hidden = n.rigBtn.hidden || isReference;

  if (active || cancelling) setText(n.primary, "View progress");
  else if (isReference && done && job.files.includes("input.png")) setText(n.primary, "Build model");
  else if (done && job.files.includes("model.glb")) setText(n.primary, "Open model");
  else if (job.status === "error") setText(n.primary, "Review error");
  else setText(n.primary, "Open asset");
  n.primary.setAttribute("aria-label", `${n.primary.textContent}: ${assetName(job)}`);
}

function renderJobs(jobs) {
  let visible = 0;
  jobs.forEach((job, i) => {
    jobsById.set(job.id, job);
    const n = nodes.get(job.id) ?? createNode(job.id);
    updateNode(n, job);
    // Hidden rather than removed: removal would fight the reuse-by-id logic
    // below, which treats "not in the list" as "this job is gone".
    n.li.hidden = !jobMatches(job);
    if (!n.li.hidden) visible++;
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
  // A deleted job cannot stay in the export selection, and the count on the
  // bar would otherwise keep counting cards that no longer exist.
  updateBulkBar();
  setText(document.getElementById("filter-count"), `${visible} of ${jobs.length} shown`);
  document.getElementById("jobs-empty").hidden = visible !== 0;
}

// --- inspector ---------------------------------------------------------------
//
// The right-hand pane: whatever the selected asset is, this is what it was made
// from and what can be done with it. In 2D that is the reference's recipe and
// its actions; in 3D it is the quality verdict, and the pose, sheet and file
// blocks below it, which manage their own visibility.
//
// Known limitation: candidates from one count>1 submit share no server-side
// grouping (create_job gives them no parent_id), so what is shown here is
// lineage -- parent_id and rerun_of -- rather than a strip of siblings.

const inspDetail = document.getElementById("insp-detail");
const inspEmpty = document.getElementById("insp-empty");
const inspQuality = document.getElementById("insp-quality");

function inspAction(label, title, onClick, primary = false) {
  const btn = document.createElement("button");
  btn.type = "button";
  setText(btn, label);
  btn.title = title;
  if (primary) btn.className = "primary";
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      await onClick();
    } finally {
      btn.disabled = false;
    }
  });
  return btn;
}

// The message the DB holds is one friendly sentence; the traceback that
// produced it has always been written to the job's error.log and had no route
// out of the server, so "Review error" showed the same ellipsised line the card
// already did. The log is fetched lazily -- opening the disclosure is the ask.
function renderErrorDetail(job) {
  const block = document.getElementById("insp-error");
  const text = document.getElementById("insp-error-text");
  const log = document.getElementById("insp-error-log");
  const logText = document.getElementById("insp-error-log-text");
  const failed = job?.status === "error";
  block.hidden = !failed;
  setText(text, failed ? (job.error || "The job failed without a message.") : "");
  const hasLog = Boolean(failed && job.files?.includes("error.log"));
  log.hidden = !hasLog;
  log.open = false;
  setText(logText, "Loading…");
  if (!hasLog) return;
  log.ontoggle = async () => {
    if (!log.open || logText.dataset.jobId === job.id) return;
    try {
      const r = await fetch(`/api/jobs/${job.id}/files/error.log`);
      setText(logText, r.ok ? await r.text() : "Could not read the error log.");
      if (r.ok) logText.dataset.jobId = job.id;
    } catch (e) {
      setText(logText, `Could not read the error log: ${e}`);
    }
  };
}

// Rebuilt only when the job it describes actually changed: showSelected runs on
// every list refresh, and re-creating the action buttons twice a second would
// disable one out from under the click that is running it.
let inspectedFor = null;

function renderInspector(job) {
  const signature = job
    ? `${job.id}|${job.status}|${job.files?.join(",")}|${job.params?.composed_prompt ?? ""}|${job.name ?? ""}|${job.tags ?? ""}|${job.favorite ? 1 : 0}|${job.error ?? ""}`
    : "";
  if (signature === inspectedFor) return;
  inspectedFor = signature;

  const badge = job && job.status === "done" ? qualityBadge(job) : null;
  inspQuality.className = badge ? badge.cls : "";
  setText(inspQuality, badge ? badge.text : "");
  inspQuality.removeAttribute("title");

  const qualityDetails = document.getElementById("insp-quality-details");
  const qualityReasons = document.getElementById("insp-quality-reasons");
  qualityReasons.replaceChildren();
  for (const reason of badge?.reasons ?? []) {
    const item = document.createElement("li");
    setText(item, reason);
    qualityReasons.append(item);
  }
  qualityDetails.hidden = !badge?.reasons?.length;

  const params = job?.params ?? {};
  inspDetail.hidden = !job;
  inspEmpty.hidden = Boolean(job);
  document.getElementById("export-section").hidden = !job || !job.files?.includes("model.glb");
  document.getElementById("insp-reference-detail").hidden = !job || job.stage !== "reference";
  document.getElementById("insp-model-detail").hidden = !job || job.stage === "reference";
  if (!job) {
    document.getElementById("insp-2d").setAttribute("aria-hidden", "true");
    document.getElementById("insp-3d").setAttribute("aria-hidden", "true");
    renderErrorDetail(null);
    return;
  }
  document.getElementById("insp-2d").setAttribute("aria-hidden", String(job.stage !== "reference"));
  document.getElementById("insp-3d").setAttribute("aria-hidden", String(job.stage === "reference"));

  const nameInput = document.getElementById("insp-name-input");
  const tagsInput = document.getElementById("insp-tags");
  const favoriteInput = document.getElementById("insp-favorite");
  if (![nameInput, tagsInput].includes(document.activeElement)) {
    nameInput.value = job.name || "";
    tagsInput.value = job.tags || "";
  }
  favoriteInput.checked = Boolean(job.favorite);
  favoriteInput.setAttribute("aria-label", `${job.favorite ? "Remove" : "Add"} ${assetName(job)} ${job.favorite ? "from" : "to"} favorites`);
  const status = document.getElementById("insp-status");
  status.className = `status ${job.status}`;
  setText(status, `${job.stage === "reference" ? "Reference" : "Model"} · ${job.status}`);
  const from = job.params?.rerun_of || job.parent_id;
  setText(document.getElementById("insp-lineage"), from ? `Variant of ${from}` : "Original asset");
  setText(document.getElementById("insp-meta"), `${job.kind} · ID ${job.id}`);

  const thumb = document.getElementById("insp-thumb");
  const preferred = job.files?.includes("thumb.png") ? "thumb.png" : "input.png";
  const hasThumb = job.files?.includes(preferred);
  const hasPng = job.files?.includes("input.png");
  if (hasThumb) {
    const src = `/api/jobs/${job.id}/files/${preferred}`;
    if (thumb.getAttribute("src") !== src) thumb.src = src;
  }
  thumb.hidden = !hasThumb;
  thumb.alt = hasThumb ? `Preview of ${assetName(job)}` : "";

  setText(
    document.getElementById("insp-prompt"),
    params.composed_prompt || job.prompt || "—",
  );
  const rows = settingsRows(params);
  renderSettingsRows(document.getElementById("insp-settings"), rows);
  document.getElementById("insp-generation").hidden = rows.length === 0;

  renderErrorDetail(job);

  const actions = document.getElementById("insp-actions");
  actions.replaceChildren();
  if (job.status === "error") {
    // A failed job used to offer nothing at all, so the only way back was to
    // retype the form. Retry first: after a transient failure the recipe was
    // fine and a new seed would be a different gamble, not a second attempt.
    const seed = retrySeed(job);
    if (seed !== null) {
      actions.append(
        inspAction("Retry with the same seed", `Run this again with seed ${seed}`,
                   () => rerunJob(job.id, "reroll", { seed }), true),
      );
    }
    if (job.kind !== "rig") {
      actions.append(
        inspAction("Generate another result", "Same prompt and settings, new seed",
                   () => rerunJob(job.id, "reroll")),
      );
    }
    if (job.files?.includes("input.png")) {
      actions.append(
        inspAction("Rebuild 3D mesh", "Reuse this reference image, rerun only the 3D stage",
                   () => rerunJob(job.id, "remesh")),
      );
    }
    return;
  }
  if (job.status !== "done") return;
  if (job.kind === "rig") return;   // a rig has no recipe of its own to re-run
  if (job.stage === "reference") {
    if (hasPng) {
      actions.append(
        inspAction("Build 3D model", "Open this reference in the Model stage",
                   () => makeThreeD(job), true),
      );
    }
    actions.append(
      inspAction("Generate another reference", "Same prompt and settings, new reference seed",
                 () => rerunJob(job.id, "reroll")),
    );
    if (hasPng) {
      const png = document.createElement("a");
      setText(png, "Download reference PNG");
      png.download = "";
      png.href = `/api/jobs/${job.id}/files/input.png`;
      actions.append(png);
    }
    return;
  }
  if (job.files.includes("model.glb")) {
    const glb = document.createElement("a");
    glb.className = "primary";
    setText(glb, "Download model GLB");
    glb.download = "";
    glb.href = `/api/jobs/${job.id}/files/model.glb`;
    actions.append(glb);
  }
  actions.append(
    inspAction("Generate another result", "Same prompt and settings, new seed",
               () => rerunJob(job.id, "reroll")),
  );
  if (hasPng) {
    actions.append(
      inspAction("Rebuild 3D mesh", "Reuse this reference image, rerun only the 3D stage",
                 () => rerunJob(job.id, "remesh")),
    );
  }
}

document.getElementById("insp-save-metadata").addEventListener("click", async () => {
  const job = jobsById.get(selected);
  if (!job) return;
  const button = document.getElementById("insp-save-metadata");
  button.disabled = true;
  try {
    const ok = await patchJob(job.id, {
      name: document.getElementById("insp-name-input").value,
      tags: document.getElementById("insp-tags").value,
      favorite: document.getElementById("insp-favorite").checked,
    });
    if (ok) {
      toast(`Saved metadata for ${assetName(job)}.`, "ok");
      inspectedFor = null;
      await poll(true);
    }
  } finally {
    button.disabled = false;
  }
});

// --- reference preview -------------------------------------------------------
// A finished reference-stage job has no mesh, and approving one from the 44px
// list thumbnail defeats the point of the approve-first split -- so a job with
// an input.png and no model.glb shows that image large in the main pane.

const refPreview = document.getElementById("ref-preview");
// The links that describe a mesh; a reference has none, so they hide with it.
const MESH_DL_IDS = ["dl-glb", "dl-obj", "dl-stl", "dl-fbx", "dl-collision", "dl-textures"];

function showReference(job) {
  const src = `/api/jobs/${job.id}/files/input.png`;
  if (refPreview.getAttribute("src") !== src) refPreview.src = src;
  refPreview.hidden = false;
  placeholder.style.display = "none";
  for (const id of MESH_DL_IDS) document.getElementById(id).hidden = true;
  const dlPng = document.getElementById("dl-png");
  dlPng.href = src;
  dlPng.hidden = false;
  document.getElementById("dl-rig").hidden = true;
  downloads.style.display = "grid";
}

function hideReference() {
  refPreview.hidden = true;
  // Restore the mesh links showSelected relies on being visible; dl-fbx keeps
  // its own gate (Blender does the conversion).
  for (const id of MESH_DL_IDS) {
    document.getElementById(id).hidden = id === "dl-fbx" && !rig.available;
  }
}

function select(id) {
  const job = jobsById.get(id);
  if (!job) return;
  if (selected !== id && poseState.editing) {
    if (!confirmDiscardEdits(`open “${assetName(job)}”`)) return;
    exitPoseEditing(false);
  }
  if (selected !== id && comparing) stopComparing();
  selected = id;
  // The job decides the mode, not the other way round: opening a reference in
  // 3D mode would show mesh tools for something that has no mesh.
  setMode(job.stage === "reference" ? "2d" : "3d");
  for (const [nid, n] of nodes) {
    n.li.classList.toggle("selected", nid === id);
    n.li.setAttribute("aria-selected", String(nid === id));
  }
  renderInspector(job);
  syncStageContext();
  if (job.files.includes("model.glb")) {
    showSelected(job);
  } else {
    shownModelFor = null;
    setPoseJob(job);
    downloads.style.display = "none";
    if (job.status === "error") {
      hideReference();
      showOverlayError(job.error || "");
    } else if (job.progress) {
      hideReference();
      showOverlay();
    } else if (job.files.includes("input.png")) {
      showReference(job);
    } else {
      hideReference();
      hideOverlay();
    }
  }
}

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

function showSelected(job) {
  // A mesh is about to occupy the pane; the reference preview must not sit
  // on top of it.
  hideReference();
  renderInspector(job);
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
  showModel(`/api/jobs/${job.id}/files/model.glb`, () => captureThumbnail(job.id));
  document.getElementById("dl-glb").href = `/api/jobs/${job.id}/files/model.glb`;
  document.getElementById("dl-obj").href = `/api/jobs/${job.id}/files/model_obj.zip`;
  document.getElementById("dl-stl").href = `/api/jobs/${job.id}/files/model.stl`;
  document.getElementById("dl-collision").href = `/api/jobs/${job.id}/files/collision.glb`;
  document.getElementById("dl-textures").href = `/api/jobs/${job.id}/files/textures.zip`;
  const dlFbx = document.getElementById("dl-fbx");
  dlFbx.href = `/api/jobs/${job.id}/files/model.fbx`;
  // Blender does the conversion, so hide it when rigging isn't installed.
  dlFbx.hidden = !rig.available;
  downloads.style.display = "grid";
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
  template: null,      // the rig's skeleton key, for the shipped preset library
  mirrorPairs: [],     // from rig.json; empty for a serpent or a fish
  presets: [],
  mode: "pose",        // "pose" rotates a joint; "joints" moves where it sits
  fitted: [],          // rig.json's bones: the fit this rig was built from
  moved: new Map(),    // bone name -> [dx, dy, dz] in Blender space
  dirty: false,        // local pose rotations not yet saved
};

function hasUnsavedEdits() {
  return Boolean(poseState.editing && (poseState.dirty || poseState.moved.size));
}

function confirmDiscardEdits(action) {
  if (!hasUnsavedEdits()) return true;
  return window.confirm(`Discard unsaved ${poseState.mode === "joints" ? "joint" : "pose"} changes and ${action}?`);
}

function syncEditModeBanner() {
  const banner = document.getElementById("edit-mode-banner");
  const apply = document.getElementById("edit-mode-apply");
  banner.hidden = !poseState.editing;
  if (!poseState.editing) return;
  const joints = poseState.mode === "joints";
  setText(
    document.getElementById("edit-mode-label"),
    `${joints ? "Joint placement" : "Pose editing"}${hasUnsavedEdits() ? " · unsaved changes" : ""}`,
  );
  setText(apply, joints ? "Apply joint positions" : "Save pose…");
  apply.disabled = joints && !poseState.moved.size;
  setText(document.getElementById("edit-mode-cancel"), joints ? "Revert and cancel" : "Cancel edits");
}

document.getElementById("edit-mode-apply").addEventListener("click", () => {
  document.getElementById(poseState.mode === "joints" ? "joint-apply" : "pose-save").click();
});
document.getElementById("edit-mode-cancel").addEventListener("click", () => {
  resetPose({ dirty: false });
  poseState.moved.clear();
  exitPoseEditing();
});

window.addEventListener("beforeunload", (e) => {
  if (!hasUnsavedEdits()) return;
  e.preventDefault();
  e.returnValue = "";
});

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
  gizmo.addEventListener("mouseUp", () => {
    controls.enabled = true;
    if (poseState.mode === "joints") recordJointDrag();
    else {
      poseState.dirty = true;
      syncEditModeBanner();
    }
  });
  scene.add(gizmo.getHelper());
  return gizmo;
}

// Markers live in their own group rather than as children of the bones, so a
// joint stays the same size on screen whatever the armature does to its scale.
function syncJointMarkers() {
  // In joints mode a marker is the thing being dragged, not a readout of where
  // its bone ended up -- snapping it back to matrixWorld every frame would undo
  // the drag as fast as the user makes it.
  if (!markers || poseState.mode === "joints") return;
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
  let handle = null;
  for (const marker of markers?.children ?? []) {
    const mine = marker.userData.bone === bone;
    marker.material.color.setHex(mine ? MARKER_ACTIVE : MARKER_IDLE);
    if (mine) handle = marker;
  }
  // Rotate the bone, but *translate the marker*: a joint's position is a
  // property of the armature's rest pose, which the viewer's copy of the rig
  // cannot express -- so the marker is the handle and the server re-skins.
  const target = poseState.mode === "joints" ? handle : bone;
  if (target) ensureGizmo().attach(target);
  else gizmo?.detach();
  setText(
    poseSelection,
    bone
      ? bone.name
      : poseState.mode === "joints"
        ? "Click a joint to move it."
        : "Click a joint to rotate it.",
  );
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

function applyPose(record, { dirty = true } = {}) {
  for (const [name, quat] of Object.entries(record.bones ?? {})) {
    poseState.bones.get(name)?.quaternion.fromArray(quat);
  }
  poseState.current = record.id ?? null;
  poseState.dirty = dirty;
  syncEditModeBanner();
}

function resetPose({ dirty = true } = {}) {
  for (const [name, quat] of poseState.rest) poseState.bones.get(name)?.quaternion.copy(quat);
  poseState.current = null;
  poseState.dirty = dirty;
  syncEditModeBanner();
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
  poseState.template = null;
  poseState.mirrorPairs = [];
  poseState.presets = [];
  poseState.fitted = [];
  poseState.moved.clear();
  sheetsSeen = "";
  renderPoses();
  if (!id) {
    sheetList.replaceChildren();
    return;
  }
  if (rigged) {
    await loadRigMeta(id);
    await refreshPoses();
  }
  await refreshSheets();
}

// rig.json carries the skeleton key and its mirror pairs; both are properties
// of the rig rather than of the job, which is why they are read once here and
// not on every pose refresh.
async function loadRigMeta(jobId) {
  try {
    const rig = await (await fetch(`/api/jobs/${jobId}/rig`)).json();
    poseState.template = rig.template ?? null;
    poseState.mirrorPairs = rig.mirror_pairs ?? [];
    poseState.fitted = rig.bones ?? [];
  } catch (e) {
    console.error("could not read the rig", e);   // posing still works by hand
  }
  syncPoseTools();
  await loadPresets(poseState.template);
}

// A skeleton with no mirror pairs (serpent, fish) has nothing to mirror, and a
// button that can only be a no-op is worse than no button.
function syncPoseTools() {
  document.getElementById("pose-mirror").hidden = !(poseState.mirrorPairs ?? []).length;
  // An older rig.json with no recorded fit has nothing for the editor to start
  // from, and a correction has to be the whole skeleton or nothing.
  document.getElementById("joint-toggle").hidden = !poseState.fitted.length;
}

async function loadPresets(templateKey) {
  const select = document.getElementById("pose-preset");
  select.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  setText(blank, "preset…");
  select.append(blank);
  poseState.presets = [];
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
  select.hidden = !poseState.presets.length;
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
      if (hasUnsavedEdits() && !confirmDiscardEdits(`show the saved pose “${record.name}”`)) return;
      if (poseState.mode === "joints") exitJointsMode();
      applyPose(record, { dirty: false });
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
  // A turntable and a gizmo drag fight each other: the joint follows the
  // pointer while the whole scene rotates out from under it.
  stopSpinning();
  poseState.editing = true;
  poseState.dirty = false;
  poseToggle.classList.add("active");
  poseToggle.setAttribute("aria-pressed", "true");
  setText(poseToggle, "Finish editing");
  poseEditor.hidden = false;
  syncEditModeBanner();
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
  poseState.mode = "pose";
  poseState.moved.clear();
  poseState.dirty = false;
  gizmo?.setMode("rotate");
  syncJointButtons();
  poseToggle.classList.remove("active");
  poseToggle.setAttribute("aria-pressed", "false");
  setText(poseToggle, "Edit pose");
  poseEditor.hidden = true;
  syncEditModeBanner();
  clearRig();
  shownModelFor = null;
  const job = restore ? jobsById.get(poseState.job) : null;
  if (job) showSelected(job);
}

poseToggle.addEventListener("click", () => {
  if (poseState.editing) {
    if (!confirmDiscardEdits("finish editing")) return;
    exitPoseEditing();
  }
  else enterPoseEditing();
});

document.getElementById("pose-reset-bone").addEventListener("click", () => {
  const bone = poseState.selected;
  if (bone) {
    bone.quaternion.copy(poseState.rest.get(bone.name));
    poseState.dirty = true;
    syncEditModeBanner();
  }
});

document.getElementById("pose-reset-all").addEventListener("click", () => resetPose());

// --- joint adjustment --------------------------------------------------------
//
// rig.json's joint positions are in *Blender* world space for the source mesh;
// the viewer shows the glTF export, which is Y-up. The exporter applies one
// axis swap at the root, so blender (x, y, z) is glTF (x, z, -y). Both
// directions live here, in one pair of functions, so they can never drift --
// getting the sign wrong tilts a dragged knee into the mesh rather than
// failing, which is not something you notice until the re-skin comes back.
//
// Only *displacements* are converted, never absolute positions: the viewer
// translates the whole model to sit it on the grid, and a delta is immune to
// that offset in a way an absolute coordinate is not.
function gltfDeltaToBlender([x, y, z]) {
  return [x, -z, y];
}

// The joint markers are the drag handles; each remembers where it started so a
// drag is a displacement rather than an absolute position.
function armJointHandles() {
  for (const marker of markers?.children ?? []) {
    marker.userData.home = marker.position.clone();
  }
}

function recordJointDrag() {
  const bone = poseState.selected;
  if (!bone || poseState.mode !== "joints") return;
  const marker = (markers?.children ?? []).find((m) => m.userData.bone === bone);
  if (!marker?.userData.home) return;
  const delta = marker.position.clone().sub(marker.userData.home);
  if (delta.lengthSq() === 0) poseState.moved.delete(bone.name);
  else poseState.moved.set(bone.name, gltfDeltaToBlender(delta.toArray()));
  syncJointButtons();
  syncEditModeBanner();
}

function syncJointButtons() {
  const on = poseState.mode === "joints";
  const toggle = document.getElementById("joint-toggle");
  toggle.classList.toggle("active", on);
  toggle.setAttribute("aria-pressed", String(on));
  setText(toggle, on ? "Finish joint placement" : "Adjust joints");
  document.getElementById("joint-apply").hidden = !on || !poseState.moved.size;
  document.getElementById("joint-revert").hidden = !on || !poseState.moved.size;
  syncEditModeBanner();
}

// The corrected skeleton the API wants: the whole thing, fitted positions with
// the dragged joints substituted.
//
// A bone's tail follows its first child's head where one exists, and otherwise
// moves rigidly with its own head. That is the rule that keeps a dragged chain
// connected -- moving a knee has to shorten the thigh and lengthen the shin,
// not leave a gap where the thigh used to end.
function correctedBones() {
  const shift = (point, name) => {
    const d = poseState.moved.get(name);
    return d ? [point[0] + d[0], point[1] + d[1], point[2] + d[2]] : [...point];
  };
  const heads = new Map();
  for (const bone of poseState.fitted) heads.set(bone.name, shift(bone.head, bone.name));
  const firstChild = new Map();
  for (const bone of poseState.fitted) {
    if (bone.parent && !firstChild.has(bone.parent)) firstChild.set(bone.parent, bone.name);
  }
  return poseState.fitted.map((bone) => {
    const child = firstChild.get(bone.name);
    return {
      name: bone.name,
      head: heads.get(bone.name),
      tail: child ? heads.get(child) : shift(bone.tail, bone.name),
    };
  });
}

function enterJointsMode() {
  poseState.mode = "joints";
  poseState.moved.clear();
  // A rotation left over from posing would make the markers show joints where
  // the *posed* bones are, and the correction is against the rest skeleton.
  resetPose({ dirty: false });
  scene.updateMatrixWorld();
  for (const marker of markers?.children ?? []) {
    marker.position.setFromMatrixPosition(marker.userData.bone.matrixWorld);
  }
  armJointHandles();
  ensureGizmo().setMode("translate");
  selectJoint(null);
  syncJointButtons();
}

function exitJointsMode() {
  poseState.mode = "pose";
  poseState.moved.clear();
  ensureGizmo().setMode("rotate");
  selectJoint(null);
  syncJointButtons();
}

document.getElementById("joint-toggle").addEventListener("click", () => {
  if (!poseState.editing || !poseState.fitted.length) return;
  if (poseState.mode === "joints") {
    if (!confirmDiscardEdits("return to pose editing")) return;
    exitJointsMode();
  }
  else enterJointsMode();
});

document.getElementById("joint-revert").addEventListener("click", () => {
  poseState.moved.clear();
  for (const marker of markers?.children ?? []) {
    if (marker.userData.home) marker.position.copy(marker.userData.home);
  }
  syncJointButtons();
  syncEditModeBanner();
});

document.getElementById("joint-apply").addEventListener("click", async () => {
  if (!poseState.moved.size) return;
  const apply = document.getElementById("joint-apply");
  apply.disabled = true;
  try {
    const r = await fetch(`/api/jobs/${poseState.job}/rig/joints`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bones: correctedBones() }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      toast(body.detail ?? "could not adjust the joints");
      return;
    }
    exitJointsMode();
    exitPoseEditing();
    poll(true);
  } finally {
    apply.disabled = false;
  }
});

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
  // Both sides are read before either is written, so a pair that is posed on
  // both sides swaps rather than one side overwriting the other mid-loop.
  for (const [name, quat] of Object.entries(bones)) {
    const other = partner.get(name);
    if (other) mirrored[other] = mirrorQuaternion(quat);
  }
  applyPose({ bones: mirrored });
});

document.getElementById("pose-preset").addEventListener("change", (e) => {
  const preset = (poseState.presets ?? []).find((p) => p.name === e.target.value);
  if (!preset) return;
  // A preset is applied to the live rig exactly like a saved pose: same bone
  // map, same code path, so it can be adjusted before it is saved. Reset first
  // because a preset lists only the bones it moves.
  resetPose();
  applyPose(preset);
  e.target.value = "";
});

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
  const record = await r.json().catch(() => ({}));
  if (!r.ok) {
    toast(record.detail ?? "could not save the pose");
    return;
  }
  poseState.current = record.id;
  poseState.dirty = false;
  syncEditModeBanner();
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
const sheetYaws = document.getElementById("sheet-yaws");
const sheetClip = document.getElementById("sheet-clip");
const sheetClipSetup = document.getElementById("sheet-clip-setup");
const sheetClipFrom = document.getElementById("sheet-clip-from");
const sheetClipTo = document.getElementById("sheet-clip-to");
const sheetClipFrames = document.getElementById("sheet-clip-frames");

// 4/8/16: the direction counts every 2D engine's facing conventions already
// use. Not free-form, because a count that does not divide 360 into the
// directions an engine indexes by is a sheet nothing can address.
const YAW_CHOICES = [4, 8, 16];

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
  for (const value of YAW_CHOICES) {
    sheetYaws.append(new Option(`${value} directions`, value));
  }
  sheetYaws.value = sheetOptions.yaws.length || 8;
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
  // Computed the same way sheet.yaw_angles does, from the count the user
  // picked -- the server's default list is only a default.
  const count = Number(sheetYaws.value) || sheetOptions.yaws.length || 8;
  const yaws = Array.from({ length: count }, (_, i) => (i * 360) / count);
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

  // The strip above is a direction preview -- it cannot pose the mesh, so
  // drawing one row per pose would draw the same row N times. The grid the
  // server will actually produce is stated here instead, and this line is the
  // part that has to agree with plan().
  const clipping = !sheetClip.hidden && sheetClip.checked;
  const rows = clipping
    ? Math.max(Number(sheetClipFrames.value) || 1, 1)
    : Math.max(sheetRows.selectedOptions.length, 1);
  const px = Number(sheetSize.value);
  setText(
    sheetSummary,
    `${rows} × ${yaws.length} = ${rows * yaws.length} render cells · ` +
      `${px * yaws.length}×${px * rows} px output`
      + (clipping ? " · animated clip" : ""),
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
      if (!confirm(`Delete the sprite sheet “${sheet.name || sheet.id}”?`)) return;
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
  syncClipEnds();
}

// A clip interpolates between two *saved* poses, so it needs two of them --
// and offering the control with nothing to put in it would only produce a 400.
function syncClipEnds() {
  const enough = poseState.saved.length >= 2;
  sheetClip.parentElement.hidden = !enough;
  if (!enough) sheetClip.checked = false;
  for (const [select, fallback] of [[sheetClipFrom, 0], [sheetClipTo, 1]]) {
    const previous = select.value;
    select.replaceChildren();
    for (const record of poseState.saved) select.append(new Option(record.name, record.id));
    const keep = poseState.saved.some((p) => p.id === previous);
    select.value = keep ? previous : (poseState.saved[fallback]?.id ?? "");
  }
  sheetClipSetup.hidden = !(enough && sheetClip.checked);
}

sheetToggle.addEventListener("click", () => {
  sheetSetup.hidden = !sheetSetup.hidden;
  sheetToggle.classList.toggle("active", !sheetSetup.hidden);
  sheetToggle.setAttribute("aria-expanded", String(!sheetSetup.hidden));
  setText(sheetToggle, sheetSetup.hidden ? "Set up" : "Close setup");
  if (!sheetSetup.hidden) {
    syncSheetRows();
    renderSheetPreview();
  }
});

for (const el of [sheetLighting, sheetSize, sheetRows, sheetYaws, sheetClipFrom, sheetClipTo]) {
  el.addEventListener("change", renderSheetPreview);
}
sheetClipFrames.addEventListener("input", renderSheetPreview);
sheetClip.addEventListener("change", () => {
  sheetClipSetup.hidden = !sheetClip.checked;
  // A clip replaces the pose rows; leaving both live would suggest the sheet
  // carries the static poses too, which the route refuses to do.
  sheetRows.disabled = sheetClip.checked;
  renderSheetPreview();
});
sheetElevation.addEventListener("input", () => {
  sheetElevationOut.textContent = `${sheetElevation.value}°`;
  renderSheetPreview();
});

document.getElementById("sheet-render").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const fd = new FormData();
    const clipping = sheetClip.checked && sheetClipFrom.value && sheetClipTo.value;
    if (clipping) {
      fd.set("clip_from", sheetClipFrom.value);
      fd.set("clip_to", sheetClipTo.value);
      fd.set("clip_frames", sheetClipFrames.value);
    } else {
      for (const option of sheetRows.selectedOptions) fd.append("poses", option.value);
    }
    fd.set("elevation", sheetElevation.value);
    fd.set("frame_size", sheetSize.value);
    fd.set("lighting", sheetLighting.value);
    fd.set("yaws", sheetYaws.value);
    const r = await fetch(`/api/jobs/${poseState.job}/sheets`, { method: "POST", body: fd });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) toast(body.detail ?? "could not queue the sheet");
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
        toast(body.detail ?? `could not prepare ${anchor.textContent}`);
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
      toast(`download failed: ${err}`);
    } finally {
      anchor.classList.remove("busy");
    }
  });
}

// Every artifact derived on demand: the first click pays for the conversion,
// so each needs the busy state rather than looking like a dead link.
for (const id of ["dl-obj", "dl-stl", "dl-collision", "dl-fbx", "dl-textures"]) {
  bindBusyDownload(document.getElementById(id));
}

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
    const body = await r.json().catch(() => ({}));
    if (!r.ok) toast(body.detail ?? "prune failed");
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
    syncTitle();
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
  syncStageContext();
  notifyTransitions(jobs);
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
  // Cheap when nothing changed (see the signature guard), and it is the only
  // thing that turns a running job's inspector into a finished one's.
  if (current) renderInspector(current);
  if (current?.files.includes("model.glb")) {
    showSelected(current);
  } else if (current?.status === "error" && selected !== shownModelFor) {
    showOverlayError(current.error || "");
  } else if (
    current?.status === "done" &&
    current.files.includes("input.png") &&
    refPreview.hidden
  ) {
    // A reference job the user was watching just finished: swap the pane
    // from the progress overlay to the image awaiting approval.
    phProgress.hidden = true;
    showReference(current);
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
  // Offered while the job is actually running: a finished or failed one has
  // nothing left to stop.
  phCancel.hidden = Boolean(live.error);
  if (!phCancel.hidden && !phCancel.disabled) setText(phCancel, "Cancel this job");

  const indeterminate = live.percent <= 0;
  phBar.classList.toggle("indet", indeterminate);
  if (!indeterminate) {
    phFill.style.width = `${live.percent}%`;
    phBar.setAttribute("aria-valuenow", String(Math.max(0, Math.min(100, live.percent))));
  } else {
    phBar.removeAttribute("aria-valuenow");
  }

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

// A backgrounded tab showed nothing at all until the 45-second completion
// notification, so the only way to know whether a two-minute run had started
// was to come back and look. The title is the one surface a hidden tab still
// has. Unlike the overlay this narrates whatever is running, not only the job
// the user happens to have selected -- a background tab has no selection.
const BASE_TITLE = document.title;

function syncTitle() {
  const next = live && liveJobId && !live.error
    ? `${Math.round(Math.max(0, Math.min(100, live.percent)))}% · ${live.label} — ${BASE_TITLE}`
    : BASE_TITLE;
  if (document.title !== next) document.title = next;
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
  } else if (live.cold && live.percent < 100) {
    // The case the suppression above is aimed at is also the slowest and most
    // confusing one, and it used to read as a stalled bar. Naming the wait is
    // not an estimate of the job -- it is an explanation of the silence.
    out += " · warming up (loading the model, usually ~15s)";
  }
  setText(phTime, out);
}
setInterval(renderClock, 200);

async function checkDoctor() {
  try {
    const health = await (await fetch("/api/health")).json();
    // Writing outside the data dir is opt-in, so the button only exists when
    // WARLOCK_EXPORT_DIR is configured.
    bulkFolder.hidden = !health.export_dir;
    if (health.export_dir) bulkFolder.title = `Copy into ${health.export_dir}`;
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
const savedWorkspace = localStorage.getItem(WORKSPACE_KEY);
setWorkspace(new Set(["settings", "preview", "inspector"]).has(savedWorkspace) ? savedWorkspace : "preview", { remember: false });
if (localStorage.getItem("warlock.inspector.v1") === "open") {
  document.body.classList.add("inspector-open");
  inspectorToggle.setAttribute("aria-expanded", "true");
}
// Last: setMode re-renders the list, so everything it touches has to exist first.
setMode(localStorage.getItem(MODE_KEY) === "3d" ? "3d" : "2d", { remember: false });

(function loop() {
  poll().finally(() => setTimeout(loop, liveJobId || pending ? FAST_MS : IDLE_MS));
})();
