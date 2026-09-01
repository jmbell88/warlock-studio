"""GLSL sources and the program cache.

Shaders live here as Python strings rather than as ``.glsl`` files. The
programs are built from a common set of chunks plus a per-mesh ``#define``
list, so the interesting artefact is the *combination*, not any one file --
and a build step that concatenates files at import time would be the same
thing with an extra place to look.

Everything below is a reimplementation of three.js r170's stock physical
material, not an invention: this app's frontend never wrote a custom shader, so
"parity" means matching what three does. The pieces that matter most for
looking the same are the ACES fit's ``/ 0.6`` pre-scale, the exact piecewise
sRGB transfer, and Lazarov/Karis ``DFGApprox`` -- which three itself uses, so
the environment term matches by construction rather than by tuning.
"""

from __future__ import annotations

from typing import Any

MAX_JOINTS = 64

# --- shared chunks ----------------------------------------------------------

COLOR_CHUNK = """
const float PI = 3.141592653589793;
const float RECIPROCAL_PI = 0.3183098861837907;
const float EPSILON = 1e-6;

// three.js ACESFilmicToneMapping. The 1/0.6 pre-scale is not optional: drop it
// and every render comes out visibly dark, which is the classic "my ACES port
// looks wrong" bug.
vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 acesFilmic(vec3 color) {
    const mat3 ACESInputMat = mat3(
        vec3(0.59719, 0.07600, 0.02840),
        vec3(0.35458, 0.90834, 0.13383),
        vec3(0.04823, 0.01566, 0.83777)
    );
    const mat3 ACESOutputMat = mat3(
        vec3( 1.60475, -0.10208, -0.00327),
        vec3(-0.53108,  1.10813, -0.07276),
        vec3(-0.07367, -0.00605,  1.07602)
    );
    color *= u_exposure / 0.6;
    color = ACESInputMat * color;
    color = RRTAndODTFit(color);
    color = ACESOutputMat * color;
    return clamp(color, 0.0, 1.0);
}

// The exact piecewise sRGB OETF, not an approximate pow(1/2.2): the linear
// segment near black is where the difference is visible, and the background
// colour is computed with the same curve on the CPU.
vec3 srgbOETF(vec3 c) {
    return mix(pow(c, vec3(0.41666)) * 1.055 - vec3(0.055), c * 12.92,
               vec3(lessThanEqual(c, vec3(0.0031308))));
}

vec3 srgbToLinear(vec3 c) {
    return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92,
               vec3(lessThanEqual(c, vec3(0.04045))));
}

vec4 outputColor(vec3 linear, float alpha) {
    return vec4(srgbOETF(acesFilmic(linear)), alpha);
}
"""

BRDF_CHUNK = """
vec3 F_Schlick(vec3 f0, float f90, float dotVH) {
    float fresnel = exp2((-5.55473 * dotVH - 6.98316) * dotVH);
    return f0 * (1.0 - fresnel) + (f90 * fresnel);
}

float V_GGX_SmithCorrelated(float alpha, float dotNL, float dotNV) {
    float a2 = alpha * alpha;
    float gv = dotNL * sqrt(a2 + (1.0 - a2) * dotNV * dotNV);
    float gl = dotNV * sqrt(a2 + (1.0 - a2) * dotNL * dotNL);
    return 0.5 / max(gv + gl, EPSILON);
}

float D_GGX(float alpha, float dotNH) {
    float a2 = alpha * alpha;
    float denom = dotNH * dotNH * (a2 - 1.0) + 1.0;
    return RECIPROCAL_PI * a2 / (denom * denom);
}

vec3 BRDF_GGX(vec3 lightDir, vec3 viewDir, vec3 normal, vec3 f0, float roughness) {
    float alpha = roughness * roughness;
    vec3 halfDir = normalize(lightDir + viewDir);
    float dotNL = clamp(dot(normal, lightDir), 0.0, 1.0);
    float dotNV = clamp(dot(normal, viewDir), 0.0, 1.0);
    float dotNH = clamp(dot(normal, halfDir), 0.0, 1.0);
    float dotVH = clamp(dot(viewDir, halfDir), 0.0, 1.0);
    return F_Schlick(f0, 1.0, dotVH) * (V_GGX_SmithCorrelated(alpha, dotNL, dotNV)
                                        * D_GGX(alpha, dotNH));
}

// Lazarov's analytic fit to the split-sum environment BRTF, as used by three
// r170 itself -- so the specular IBL term matches by construction.
vec2 DFGApprox(vec3 normal, vec3 viewDir, float roughness) {
    float dotNV = clamp(dot(normal, viewDir), 0.0, 1.0);
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * dotNV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// The equirectangular mapping the specular probe is stored in. Must match
// env.sample_equirect exactly, or the reflection slides as the camera orbits.
vec2 equirectUv(vec3 dir) {
    return vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
                asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5);
}

// Ramamoorthi & Hanrahan's 9-coefficient irradiance evaluation. Every constant
// (the basis normalisation and the per-band convolution factor) is folded into
// the coefficients on the CPU, so this is the whole of it.
vec3 shIrradiance(vec3 sh[9], vec3 n) {
    return sh[0]
         + sh[1] * n.y + sh[2] * n.z + sh[3] * n.x
         + sh[4] * (n.x * n.y) + sh[5] * (n.y * n.z)
         + sh[6] * (3.0 * n.z * n.z - 1.0)
         + sh[7] * (n.x * n.z) + sh[8] * (n.x * n.x - n.y * n.y);
}
"""

# Not an f-string: GLSL is all braces, and doubling every one of them to
# interpolate a single number would make the shader unreadable to anyone
# reading it as GLSL.
SKIN_CHUNK = """
#ifdef SKINNED
uniform mat4 u_joints[__MAX_JOINTS__];
mat4 skinMatrix() {
    return u_joints[a_joints.x] * a_weights.x
         + u_joints[a_joints.y] * a_weights.y
         + u_joints[a_joints.z] * a_weights.z
         + u_joints[a_joints.w] * a_weights.w;
}
#endif
""".replace("__MAX_JOINTS__", str(MAX_JOINTS))

PBR_VERT = """
#version 330 core
in vec3 a_position;
in vec3 a_normal;
in vec2 a_uv;
#ifdef SKINNED
in ivec4 a_joints;
in vec4 a_weights;
#endif

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_matrix;

out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;

__SKIN__

void main() {
    vec4 local = vec4(a_position, 1.0);
    vec3 normal = a_normal;
#ifdef SKINNED
    mat4 skin = skinMatrix();
    local = skin * local;
    normal = mat3(skin) * normal;
#endif
    vec4 world = u_model * local;
    v_world = world.xyz;
    v_normal = u_normal_matrix * normal;
    v_uv = a_uv;
    gl_Position = u_proj * u_view * world;
}
"""

PBR_FRAG = """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

uniform vec3 u_camera_pos;
uniform float u_exposure;
// Multiplied into the output alpha, and 1.0 for every draw but one: Clay's
// X-ray, which renders the surface see-through so the elements behind it can
// be picked. A uniform rather than a factor folded into
// ``u_base_color_factor`` because that one is the *material's* opacity and
// belongs to the document -- writing a view setting into it would make an
// export come out translucent.
uniform float u_alpha;

uniform vec4 u_base_color_factor;
uniform float u_metallic;
uniform float u_roughness;
uniform vec3 u_emissive_factor;
uniform float u_alpha_cutoff;
uniform int u_alpha_mask;

uniform vec3 u_sh[9];
uniform sampler2D u_env;
uniform float u_env_max_mip;

uniform vec3 u_hemi_sky;
uniform vec3 u_hemi_ground;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;

#ifdef HAS_BASE_COLOR_MAP
uniform sampler2D u_base_color_map;
#endif
#ifdef HAS_MR_MAP
uniform sampler2D u_mr_map;
#endif
#ifdef HAS_NORMAL_MAP
uniform sampler2D u_normal_map;
#endif
#ifdef HAS_EMISSIVE_MAP
uniform sampler2D u_emissive_map;
#endif
#ifdef HAS_AO_MAP
uniform sampler2D u_ao_map;
#endif

__COLOR__
__BRDF__

#ifdef HAS_NORMAL_MAP
// Screen-space derivative TBN. The GLBs this app produces carry no TANGENT
// attribute, and a normal map without one has to derive its frame somehow;
// this is the standard Mikkelsen construction three falls back to as well.
vec3 perturbNormal(vec3 n, vec3 viewDir) {
    vec3 dp1 = dFdx(v_world);
    vec3 dp2 = dFdy(v_world);
    vec2 duv1 = dFdx(v_uv);
    vec2 duv2 = dFdy(v_uv);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(dot(t, t), dot(b, b)));
    vec3 mapped = texture(u_normal_map, v_uv).xyz * 2.0 - 1.0;
    return normalize(mat3(t * invmax, b * invmax, n) * mapped);
}
#endif

void main() {
    vec4 base = u_base_color_factor;
#ifdef HAS_BASE_COLOR_MAP
    vec4 sampled = texture(u_base_color_map, v_uv);
    // baseColorTexture is sRGB-encoded; everything below this line is linear.
    base *= vec4(srgbToLinear(sampled.rgb), sampled.a);
#endif
    if (u_alpha_mask == 1 && base.a < u_alpha_cutoff) discard;
    // OPAQUE means the alpha channel is ignored, not that it happens to be 1:
    // trellis writes a base-colour texture whose alpha varies, and honouring
    // it would render the mesh semi-transparent against the background.
    if (u_alpha_mask == 0) base.a = 1.0;

    float metallic = u_metallic;
    float roughness = u_roughness;
#ifdef HAS_MR_MAP
    // Linear, and channel-packed: G is roughness, B is metalness.
    vec3 mr = texture(u_mr_map, v_uv).rgb;
    roughness *= mr.g;
    metallic *= mr.b;
#endif
    roughness = clamp(roughness, 0.0525, 1.0);

    vec3 normal = normalize(v_normal);
    if (!gl_FrontFacing) normal = -normal;
    vec3 viewDir = normalize(u_camera_pos - v_world);
#ifdef HAS_NORMAL_MAP
    normal = perturbNormal(normal, viewDir);
#endif

    vec3 diffuseColor = base.rgb * (1.0 - metallic);
    vec3 f0 = mix(vec3(0.04), base.rgb, metallic);

    // -- direct ------------------------------------------------------------
    vec3 lightDir = normalize(u_light_dir);
    float dotNL = clamp(dot(normal, lightDir), 0.0, 1.0);
    vec3 irradiance = dotNL * u_light_color;
    vec3 lit = irradiance * RECIPROCAL_PI * diffuseColor;
    lit += irradiance * BRDF_GGX(lightDir, viewDir, normal, f0, roughness);

    // -- indirect diffuse --------------------------------------------------
    // The hemisphere light and the environment probe both contribute, exactly
    // as three adds them: the probe is what stops a downward-facing surface
    // going black and reading as a hole in the mesh.
    float hemiWeight = 0.5 * dot(normal, vec3(0.0, 1.0, 0.0)) + 0.5;
    vec3 ambient = mix(u_hemi_ground, u_hemi_sky, hemiWeight);
    ambient += max(shIrradiance(u_sh, normal), vec3(0.0));
    lit += ambient * RECIPROCAL_PI * diffuseColor;

    // -- indirect specular -------------------------------------------------
    vec3 reflected = reflect(-viewDir, normal);
    float mip = roughness * u_env_max_mip;
    vec3 radiance = textureLod(u_env, equirectUv(reflected), mip).rgb;
    vec2 fab = DFGApprox(normal, viewDir, roughness);
    lit += radiance * (f0 * fab.x + fab.y);

    vec3 emissive = u_emissive_factor;
#ifdef HAS_EMISSIVE_MAP
    emissive *= srgbToLinear(texture(u_emissive_map, v_uv).rgb);
#endif
    lit += emissive;

#ifdef HAS_AO_MAP
    // Occlusion applies to the indirect terms only in three; applying it to
    // everything is the cheap approximation, and these meshes ship no AO map
    // at all -- this branch exists so a hand-supplied GLB does not lose it.
    lit *= texture(u_ao_map, v_uv).r;
#endif

    f_color = outputColor(lit, base.a * u_alpha);
}
"""

UNLIT_VERT = PBR_VERT

UNLIT_FRAG = """
#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;

uniform float u_exposure;
uniform float u_alpha;
uniform vec4 u_base_color_factor;
#ifdef HAS_BASE_COLOR_MAP
uniform sampler2D u_base_color_map;
#endif

__COLOR__

void main() {
    vec4 base = u_base_color_factor;
#ifdef HAS_BASE_COLOR_MAP
    vec4 sampled = texture(u_base_color_map, v_uv);
    base *= vec4(srgbToLinear(sampled.rgb), sampled.a);
#endif
    // Tone-mapped like everything else: three's MeshBasicMaterial defaults to
    // toneMapped = true, so a flat sheet frame and a lit one agree about what
    // a given albedo looks like.
    f_color = outputColor(base.rgb, base.a * u_alpha);
}
"""

SOLID_VERT = """
#version 330 core
in vec3 a_position;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
void main() {
    gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
}
"""

SOLID_FRAG = """
#version 330 core
out vec4 f_color;
uniform vec4 u_color;
uniform float u_exposure;
__COLOR__
void main() {
    // Markers and gizmos are authored as sRGB swatches, so they go back out
    // through the same transform rather than bypassing it -- otherwise the
    // accent purple on a handle would not match the accent purple on a button.
    f_color = vec4(srgbOETF(acesFilmic(srgbToLinear(u_color.rgb))), u_color.a);
}
"""

LINES_VERT = """
#version 330 core
in vec3 a_position;
in vec3 a_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_proj * u_view * vec4(a_position, 1.0);
}
"""

LINES_FRAG = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
uniform float u_exposure;
uniform float u_alpha;
__COLOR__
void main() {
    f_color = vec4(srgbOETF(acesFilmic(srgbToLinear(v_color))), u_alpha);
}
"""

def _expand(source: str, defines: tuple[str, ...]) -> str:
    """Splice the shared chunks in and put the #defines after the #version.

    ``#version`` must be the first non-comment line of a GLSL source, so the
    defines cannot simply be prepended -- which is the one thing that makes
    this more than a string concatenation.
    """
    source = (
        source.replace("__COLOR__", COLOR_CHUNK)
        .replace("__BRDF__", BRDF_CHUNK)
        .replace("__SKIN__", SKIN_CHUNK)
    )
    lines = source.strip().splitlines()
    head, rest = lines[0], lines[1:]
    block = "\n".join(f"#define {d} 1" for d in defines)
    return "\n".join([head, block, *rest])


class ProgramCache:
    """One program per (name, defines) pair, built on first use.

    A mesh with a metallic-roughness map and one without are different programs
    in the same sense they are in three: the branch is compiled out, not taken
    per fragment.
    """

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def get(self, name: str, defines: tuple[str, ...] = ()) -> Any:
        # ``defines`` must already be canonical (sorted): GpuPrimitive sorts
        # its tuple once at build time (B16), so re-sorting here per draw per
        # primitive would pay the canonicalisation 60 times a second for
        # tuples that never change.
        key = (name, defines)
        program = self._cache.get(key)
        if program is None:
            vert, frag = SOURCES[name]
            program = self.ctx.program(
                vertex_shader=_expand(vert, key[1]),
                fragment_shader=_expand(frag, key[1]),
            )
            self._cache[key] = program
        return program

    def release(self) -> None:
        for program in self._cache.values():
            program.release()
        self._cache.clear()


# No "blit" entry, deliberately: the MSAA resolve is ``ctx.copy_framebuffer``
# (see ``glctx.Viewport.resolve``), so a fullscreen-copy program never had a
# caller and carrying its sources only claimed one existed.
SOURCES: dict[str, tuple[str, str]] = {
    "pbr": (PBR_VERT, PBR_FRAG),
    "unlit": (UNLIT_VERT, UNLIT_FRAG),
    "solid": (SOLID_VERT, SOLID_FRAG),
    "lines": (LINES_VERT, LINES_FRAG),
}
