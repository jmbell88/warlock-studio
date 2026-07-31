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
    document.getElementById("placeholder").style.display = "none";
  });
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
    if (!r.ok) alert((await r.json()).detail ?? "request failed");
  } finally {
    btn.disabled = false;
  }
  refresh();
});

// --- job list ---------------------------------------------------------------

let selected = null;

async function refresh() {
  const jobs = await (await fetch("/api/jobs")).json();
  const ul = document.getElementById("jobs");
  ul.replaceChildren(...jobs.map((job) => {
    const li = document.createElement("li");
    li.classList.toggle("selected", job.id === selected);
    if (job.files.includes("input.png")) {
      const img = document.createElement("img");
      img.src = `/api/jobs/${job.id}/files/input.png`;
      li.append(img);
    }
    const info = document.createElement("div");
    const title = document.createElement("div");
    title.className = "job-title";
    title.textContent = job.prompt ?? `image job ${job.id}`;
    const status = document.createElement("div");
    status.className = `status ${job.status}`;
    status.textContent = job.status;
    info.append(title, status);
    li.append(info);

    const actions = document.createElement("div");
    actions.className = "job-actions";
    const act = document.createElement("button");
    if (job.status === "queued" || job.status === "running") {
      act.textContent = "cancel";
      act.onclick = (e) => { e.stopPropagation(); fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" }).then(refresh); };
    } else {
      act.textContent = "delete";
      act.onclick = (e) => { e.stopPropagation(); fetch(`/api/jobs/${job.id}`, { method: "DELETE" }).then(refresh); };
    }
    actions.append(act);
    li.append(actions);

    li.addEventListener("click", () => select(job));
    return li;
  }));
  const current = jobs.find((j) => j.id === selected);
  if (current?.files.includes("model.glb")) showSelected(current);
}

function select(job) {
  selected = job.id;
  refresh();
  if (job.files.includes("model.glb")) showSelected(job);
}

let shownModelFor = null;
function showSelected(job) {
  if (shownModelFor === job.id) return;
  shownModelFor = job.id;
  showModel(`/api/jobs/${job.id}/files/model.glb`);
  document.getElementById("dl-glb").href = `/api/jobs/${job.id}/files/model.glb`;
  document.getElementById("dl-obj").href = `/api/jobs/${job.id}/files/model_obj.zip`;
  document.getElementById("dl-stl").href = `/api/jobs/${job.id}/files/model.stl`;
  document.getElementById("downloads").style.display = "flex";
}

refresh();
setInterval(refresh, 2500);
