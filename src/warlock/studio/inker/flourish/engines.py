"""Engine snippets: the few lines that load a baked sheet, per engine.

``pipelines.sheet.sidecar`` is engine-neutral on purpose (its docstring says
why), so what an engine needs is *derived from* the sidecar here and never
written into it. Each renderer takes the same small mapping -- what the
export wrote -- and returns text the user pastes. Pure string work; a test
executes the Pygame one against a stub and string-checks the rest.
"""

from __future__ import annotations

from typing import Any

ENGINES = ("pygame-ce", "godot", "unity", "phaser")


def describe(
    *,
    name: str,
    image: str,
    frame_width: int,
    frame_height: int,
    frames: int,
    fps: int,
    loop: bool,
    origin: tuple[int, int],
) -> dict[str, Any]:
    """The mapping every snippet reads. One shape, so a caller cannot hand
    Godot a different frame count than Pygame."""
    return {
        "name": str(name),
        "image": str(image),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "frames": int(frames),
        "fps": int(fps),
        "loop": bool(loop),
        "origin": [int(origin[0]), int(origin[1])],
    }


def snippet(engine: str, info: dict[str, Any]) -> str:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {list(ENGINES)}")
    return _RENDERERS[engine](info)


def _pygame(i: dict[str, Any]) -> str:
    ident = _ident(i["name"])
    head = f'{i["frames"]} frames of {i["frame_width"]}x{i["frame_height"]} at {i["fps"]} fps'
    return f'''# {i["name"]}: {head}
import pygame

class Animation:
    def __init__(self, spritesheet, frame_size, frame_count, fps, loop, origin):
        sheet = pygame.image.load(spritesheet).convert_alpha()
        w, h = frame_size
        self.frames = [sheet.subsurface((n * w, 0, w, h)) for n in range(frame_count)]
        self.fps, self.loop, self.origin = fps, loop, origin
        self.time = 0.0

    def update(self, dt):
        self.time += dt

    @property
    def done(self):
        return not self.loop and self.time * self.fps >= len(self.frames)

    def draw(self, surface, pos):
        index = int(self.time * self.fps)
        index = index % len(self.frames) if self.loop else min(index, len(self.frames) - 1)
        ox, oy = self.origin
        surface.blit(self.frames[index], (pos[0] - ox, pos[1] - oy))

{ident} = Animation(
    spritesheet="{i["image"]}",
    frame_size=({i["frame_width"]}, {i["frame_height"]}),
    frame_count={i["frames"]},
    fps={i["fps"]},
    loop={i["loop"]},
    origin=({i["origin"][0]}, {i["origin"][1]}),
)
'''


def _godot(i: dict[str, Any]) -> str:
    ident = _ident(i["name"])
    fw, fh = i["frame_width"], i["frame_height"]
    ox, oy = fw / 2 - i["origin"][0], fh / 2 - i["origin"][1]
    return f'''# {i["name"]}: build SpriteFrames from the sheet at runtime (Godot 4)
var {ident} := SpriteFrames.new()

func _ready() -> void:
    var sheet: Texture2D = load("res://{i["image"]}")
    {ident}.add_animation("{i["name"]}")
    {ident}.set_animation_speed("{i["name"]}", {i["fps"]})
    {ident}.set_animation_loop("{i["name"]}", {"true" if i["loop"] else "false"})
    for n in {i["frames"]}:
        var atlas := AtlasTexture.new()
        atlas.atlas = sheet
        atlas.region = Rect2(n * {fw}, 0, {fw}, {fh})
        {ident}.add_frame("{i["name"]}", atlas)
    $AnimatedSprite2D.sprite_frames = {ident}
    $AnimatedSprite2D.offset = Vector2({ox}, {oy})
    $AnimatedSprite2D.play("{i["name"]}")
'''


def _unity(i: dict[str, Any]) -> str:
    grid = f'{i["frame_width"]}x{i["frame_height"]}'
    pivot = f'({i["origin"][0]}, {i["origin"][1]})'
    return f'''// {i["name"]}: slice {i["image"]} in the Sprite Editor as a {grid} grid
// ({i["frames"]} cells, pivot at {pivot} px), then drive it from a script:
using UnityEngine;

public class {_ident(i["name"], pascal=True)}Player : MonoBehaviour
{{
    public Sprite[] frames;          // the {i["frames"]} sliced sprites, in order
    public float fps = {i["fps"]}f;
    public bool loop = {"true" if i["loop"] else "false"};
    SpriteRenderer sr; float t;

    void Awake() {{ sr = GetComponent<SpriteRenderer>(); }}

    void Update()
    {{
        t += Time.deltaTime;
        int index = (int)(t * fps);
        if (loop) index %= frames.Length;
        else if (index >= frames.Length) {{ Destroy(gameObject); return; }}
        sr.sprite = frames[index];
    }}
}}
'''


def _phaser(i: dict[str, Any]) -> str:
    ident = _ident(i["name"])
    fw, fh = i["frame_width"], i["frame_height"]
    return f'''// {i["name"]}: Phaser 3
preload() {{
    this.load.spritesheet("{ident}", "{i["image"]}", {{ frameWidth: {fw}, frameHeight: {fh} }});
}}

create() {{
    this.anims.create({{
        key: "{ident}",
        frames: this.anims.generateFrameNumbers("{ident}", {{ start: 0, end: {i["frames"] - 1} }}),
        frameRate: {i["fps"]},
        repeat: {-1 if i["loop"] else 0},
    }});
    const sprite = this.add.sprite(x, y, "{ident}");
    sprite.setOrigin({i["origin"][0] / fw:.4f}, {i["origin"][1] / fh:.4f});
    sprite.play("{ident}");
}}
'''


_RENDERERS = {"pygame-ce": _pygame, "godot": _godot, "unity": _unity, "phaser": _phaser}


def _ident(name: str, *, pascal: bool = False) -> str:
    parts = [p for p in "".join(c if c.isalnum() else " " for c in name).split() if p]
    if not parts:
        parts = ["effect"]
    if pascal:
        return "".join(p[:1].upper() + p[1:] for p in parts)
    ident = "_".join(p.lower() for p in parts)
    return ident if not ident[0].isdigit() else f"fx_{ident}"
