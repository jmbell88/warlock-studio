# Third-party notices

Warlock Studio is licensed under the GNU General Public License v3.0 or later
(see [`LICENSE`](LICENSE)). It ships, bundles or downloads the components below,
each under its own terms. Nothing here overrides those terms.

This file travels with the Windows installer: `installer/build.ps1` stages it
beside the binaries it describes, because MIT and the NVIDIA redistributable
EULA both require the notice to accompany the binary rather than merely exist in
a repository somewhere.

---

## Bundled native binaries

Pinned by SHA-256 in `installer/runtime-manifest.json` and verified at build
time. All are redistributed unmodified.

| Component | Files | Upstream | Licence |
|---|---|---|---|
| trellis.cpp | `trellis-server.exe`, `trellis-cli.exe` | <https://github.com/pwilkin/trellis.cpp> | MIT |
| ggml | `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `ggml-cuda.dll` | <https://github.com/ggml-org/ggml> | MIT |
| NVIDIA CUDA runtime | `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` | NVIDIA CUDA Toolkit 12.8 redistributables | NVIDIA CUDA Toolkit EULA — redistribution permitted under the "Attachment A" redistributable list |
| meshoptimizer | `gltfpack.exe` | <https://github.com/zeux/meshoptimizer> | MIT |
| warlockc | `warlockc.dll` | this repository (`native/`) | GPL-3.0-or-later, as part of this program |

`warlockc.dll` is optional: every kernel in it has a NumPy fallback and the
application runs without the DLL present.

## Bundled Python runtime

The installer packs a CPython 3.13 runtime from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
(PSF-2.0, plus the licences of the C libraries it embeds — OpenSSL, SQLite,
zlib, libffi, and others; see that project's own `LICENSE` files, which travel
inside the runtime tree the installer copies).

## Bundled fonts

| Font | Upstream | Licence | Notice shipped as |
|---|---|---|---|
| Inter (PUA-stripped) | <https://github.com/rsms/inter> | SIL Open Font License 1.1 | `src/warlock/studio/resources/fonts/LICENSE-inter.txt` |
| Lucide icons | <https://github.com/lucide-icons/lucide> | ISC | `src/warlock/studio/resources/fonts/LICENSE-lucide.txt` |

## Vendored source

| Component | Where | Upstream | Licence |
|---|---|---|---|
| BiRefNet modelling code | `src/warlock/pipelines/birefnet/` | <https://github.com/ZhengPeng7/BiRefNet> | MIT |

Vendored rather than downloaded so that the application never executes Python it
fetched at runtime. The pinned commit, the SHA-256 of every original file and a
documented diff are in that directory's own
[`ATTRIBUTION.md`](src/warlock/pipelines/birefnet/ATTRIBUTION.md).

## Test fixtures

Not part of the application and not staged by the installer, but `/tests` is in
the source distribution's allowlist (`pyproject.toml`), so a source release
redistributes them — and CC-BY's attribution requirement travels with the file.

| Asset | Where | Upstream | Licence |
|---|---|---|---|
| CesiumMan | `tests/fixtures/humanoid/cesium_man.glb` | Cesium, via the [Khronos glTF sample models](https://github.com/KhronosGroup/glTF-Sample-Models) | CC-BY 4.0 |

**Anything published that was rendered from CesiumMan credits Cesium.** That
file's own [`ATTRIBUTION.md`](tests/fixtures/humanoid/ATTRIBUTION.md) carries the
full terms and travels beside it; this table exists because a reader looking for
what this project redistributes looks here first.

## Python dependencies

Installed from PyPI by `uv`, and packed into the installer's runtime. The
complete, exact set with resolved versions is `uv.lock`. The ones whose terms
are worth calling out:

| Package | Licence | Note |
|---|---|---|
| `bpy` (Blender as a Python module) | **GPL-3.0** | The reason this project is GPL-3.0. Only `src/warlock/pipelines/blender_worker.py` imports it, and only in a subprocess — but the installer distributes it inside one executable alongside this program, so the combined work is GPL-3.0. Installed by the `rig` extra. |
| `pygame-ce` | LGPL-2.1 | Used unmodified as a library. |
| `PyOpenGL`, `moderngl`, `imgui-bundle`, `trimesh`, `zstandard`, `pillow` | MIT / MIT / MIT / MIT / BSD-3 / MIT-CMU | |
| `numpy`, `scipy`, `opencv-python-headless` | BSD-3 / BSD-3 / Apache-2.0 | |
| `torch`, `diffusers`, `transformers`, `huggingface-hub`, `manifold3d` | BSD-3 / Apache-2.0 / Apache-2.0 / Apache-2.0 / Apache-2.0 | |

## Model weights

**Not bundled.** Every checkpoint is downloaded by the user, on request, from
Hugging Face — the installer ships none of them and this project redistributes
none of them. They are licensed by their publishers, and two of them restrict
commercial use of what you generate.

| Model | Publisher | Licence | Commercial use of output |
|---|---|---|---|
| SDXL 1.0 | Stability AI | OpenRAIL++-M | Permitted, subject to the use restrictions |
| SDXL-Turbo | Stability AI | Stability AI Non-Commercial Research Community License | **No** — commercial use requires a paid Stability membership |
| Playground v2.5 | Playground | Playground v2.5 Community License | Permitted below 1M monthly active users; requires shipping the licence and its attribution string |
| Juggernaut XL v9 | RunDiffusion | OpenRAIL-M | Permitted, subject to the use restrictions |
| DreamShaper XL | Lykon | OpenRAIL++-M | Permitted, subject to the use restrictions |
| FLUX.2 klein / klein-base 4B | Black Forest Labs | Apache-2.0 | Permitted |
| TRELLIS.2-4B | Microsoft | MIT | Permitted |
| BiRefNet weights | ZhengPeng7 | MIT | Permitted |

The application surfaces this per model: `warlock.models` carries a `license`
field on every entry, the model picker and the download confirmation show it,
and [`docs/MODELS.md`](docs/MODELS.md) lists it in full. If you intend to sell
what you generate, read the row for the model you generated it with.
