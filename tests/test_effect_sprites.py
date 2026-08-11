"""The Phase 5 sprite caches' resource discipline, pinned headlessly.

``tests/test_ux_phase45.py`` pins what the effects draw; this file pins what
they *hold* -- because every defect here is invisible in a screenshot and in
the GL smoke suite alike. A texture released without ``forget_texture`` leaves
the backend holding a dead object under a GL name the driver will reuse
(docs/INVARIANTS.md's registration rule), a pipeline that outlives its switch is three
framebuffers nobody can see, and a cache keyed on physical pixels strands one
UI scale's sprites the moment the user picks another. All of it is observable
from a stub renderer that records the order of register / forget / release,
which is what every test below drives.

The import pins at the bottom are ``tests/clay/test_clay_imports.py``'s idiom:
the modules' own docstrings claim ``vram.py``-style purity (numpy and stdlib,
GL only behind lazy in-function imports), and a claim nothing asserts is worth
exactly nothing.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from warlock.studio import (
    effects,
    imgui_backend,
    ninepatch,
    shadows,
    surfaces,
    tokens,
    vibrancy,
    widgets,
)

# --- a renderer and a GL context that only take notes ------------------------


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def index(self, kind: str, obj: object) -> int:
        return next(
            i for i, (k, o) in enumerate(self.events) if k == kind and o is obj
        )

    def has(self, kind: str, obj: object) -> bool:
        return any(k == kind and o is obj for k, o in self.events)


class _Tex:
    """Accepts the attributes ``_Pipeline`` sets, and logs its own death."""

    def __init__(self, log: _Log) -> None:
        self.log = log
        self.glo = 1 + len(log.events)
        self.filter = None
        self.repeat_x = self.repeat_y = True

    def use(self, unit: int = 0) -> None:
        pass

    def release(self) -> None:
        self.log.events.append(("release", self))


class _Releasable:
    def __init__(self, log: _Log) -> None:
        self.log = log

    def use(self) -> None:
        pass

    def release(self) -> None:
        self.log.events.append(("release", self))


class _Gl:
    """The five constructors ``_Pipeline`` calls, plus the capture blit."""

    def __init__(self, log: _Log) -> None:
        self.log = log
        self.screen = _Releasable(log)

    def texture(self, size, components):
        return _Tex(self.log)

    def framebuffer(self, color_attachments):
        return _Releasable(self.log)

    def program(self, vertex_shader, fragment_shader):
        return _Releasable(self.log)

    def buffer(self, data):
        return _Releasable(self.log)

    def vertex_array(self, program, spec):
        return _Releasable(self.log)

    def copy_framebuffer(self, dst, src) -> None:
        pass


class _Renderer:
    def __init__(self, log: _Log) -> None:
        self.log = log

    def register_texture(self, texture) -> int:
        self.log.events.append(("register", texture))
        return getattr(texture, "glo", 0)

    def forget_texture(self, texture) -> None:
        self.log.events.append(("forget", texture))


def _fresh_vibrancy(monkeypatch, renderer) -> None:
    monkeypatch.setattr(imgui_backend, "current", lambda: renderer)
    monkeypatch.setattr(effects, "VIBRANCY", True)
    monkeypatch.setattr(vibrancy, "_PIPELINE", None)
    monkeypatch.setattr(vibrancy, "_BROKEN", False)
    monkeypatch.setattr(vibrancy, "_STALE", True)
    monkeypatch.setattr(vibrancy, "_USED", False)
    monkeypatch.setattr(vibrancy, "_LAST_CAPTURE", 0.0)


# --- vibrancy: the pipeline follows the registration rule --------------------


def test_a_window_resize_forgets_the_old_pong_before_releasing_it(monkeypatch):
    """The resize path used to call ``_PIPELINE.release()`` directly, skipping
    the ``forget_texture`` half of the pair -- docs/INVARIANTS.md's rule: a texture
    released while still registered leaves the backend holding a dead object
    under a GL name the driver will reuse."""
    log = _Log()
    _fresh_vibrancy(monkeypatch, _Renderer(log))
    gl = _Gl(log)
    vibrancy.capture(gl, (800.0, 600.0))
    old = vibrancy._PIPELINE
    assert old is not None and old.size == (200, 150)
    vibrancy._LAST_CAPTURE = 0.0  # past the throttle, so the resize is seen
    vibrancy.capture(gl, (1024.0, 768.0))
    assert vibrancy._PIPELINE is not None and vibrancy._PIPELINE is not old
    assert log.has("forget", old.pong)
    assert log.index("forget", old.pong) < log.index("release", old.pong)


def test_switching_the_effect_off_takes_the_pipeline_with_it(monkeypatch):
    """Three textures and three framebuffers serving an effect nothing draws
    must not sit resident for the session on the strength of having once been
    on. The flag going off releases them from ``capture``'s own early-out --
    the one call the frame loop already makes every frame -- and does so
    exactly once."""
    log = _Log()
    _fresh_vibrancy(monkeypatch, _Renderer(log))
    gl = _Gl(log)
    vibrancy.capture(gl, (800.0, 600.0))
    old = vibrancy._PIPELINE
    assert old is not None
    monkeypatch.setattr(effects, "VIBRANCY", False)
    vibrancy._LAST_CAPTURE = 0.0
    vibrancy.capture(gl, (800.0, 600.0))
    assert vibrancy._PIPELINE is None
    assert log.index("forget", old.pong) < log.index("release", old.pong)
    # Idempotent: the next frame with the flag still off releases nothing more.
    seen = len(log.events)
    vibrancy.capture(gl, (800.0, 600.0))
    assert len(log.events) == seen
    assert vibrancy.backdrop() is None


# --- the sprite caches: a UI scale change does not strand a working set ------


def _fake_ninepatch(monkeypatch):
    """``build`` without GL, ``release`` that takes attendance."""
    released: list[ninepatch.Sprite] = []
    monkeypatch.setattr(ninepatch, "have_renderer", lambda: True)
    monkeypatch.setattr(
        ninepatch,
        "build",
        lambda alpha, corner: ninepatch.Sprite(texture=object(), ref=object(), corner=corner),
    )
    monkeypatch.setattr(ninepatch, "release", lambda sprites: released.extend(sprites))
    return released


def test_a_ui_scale_change_sweeps_the_shadow_sprites_it_re_keyed(monkeypatch):
    """The cache is keyed on physical pixels, so a scale change re-keys every
    request and the old scale's entries would sit resident until shutdown with
    nothing ever asking for them again."""
    released = _fake_ninepatch(monkeypatch)
    monkeypatch.setattr(effects, "SOFT_SHADOWS", True)
    monkeypatch.setattr(shadows, "_BROKEN", False)
    monkeypatch.setattr(shadows, "_SPRITES", {})
    monkeypatch.setattr(tokens, "SCALE", 1.0)
    monkeypatch.setattr(shadows, "_SCALE", 1.0, raising=False)
    first = shadows.sprite(10.0, 9.0)
    assert first is not None and released == []
    monkeypatch.setattr(tokens, "SCALE", 1.5)
    second = shadows.sprite(15.0, 13.5)
    assert second is not None
    assert first in released
    assert list(shadows._SPRITES.values()) == [second]


def test_a_ui_scale_change_sweeps_the_squircle_sprites_too(monkeypatch):
    released = _fake_ninepatch(monkeypatch)
    monkeypatch.setattr(effects, "SQUIRCLES", True)
    monkeypatch.setattr(surfaces, "_BROKEN", False)
    monkeypatch.setattr(surfaces, "_SPRITES", {})
    monkeypatch.setattr(tokens, "SCALE", 1.0)
    monkeypatch.setattr(surfaces, "_SCALE", 1.0, raising=False)
    first = surfaces.sprite(12.0)
    assert first is not None and released == []
    monkeypatch.setattr(tokens, "SCALE", 1.5)
    second = surfaces.sprite(18.0)
    assert second is not None
    assert first in released
    assert list(surfaces._SPRITES.values()) == [second]


# --- the size guard: declined before the mask is rasterised, and only once ---


def test_an_oversize_shadow_is_declined_before_it_rasterises(monkeypatch):
    """``ninepatch.build`` refuses past ``MAX_SIZE``, but it used to refuse
    *after* its argument had rasterised the whole mask -- and the refusal was
    not cached, so an oversize request paid the rasterise on every frame."""
    _fake_ninepatch(monkeypatch)
    calls: list[tuple[float, float]] = []
    monkeypatch.setattr(
        shadows,
        "pixels",
        lambda radius, spread: calls.append((radius, spread)) or np.zeros((3, 3), np.uint8),
    )
    monkeypatch.setattr(effects, "SOFT_SHADOWS", True)
    monkeypatch.setattr(shadows, "_BROKEN", False)
    monkeypatch.setattr(shadows, "_SPRITES", {})
    monkeypatch.setattr(shadows, "_SCALE", tokens.SCALE, raising=False)
    assert 2 * shadows._corner(260.0, 20.0) + 1 > ninepatch.MAX_SIZE
    assert shadows.sprite(260.0, 20.0) is None
    assert shadows.sprite(260.0, 20.0) is None
    assert calls == []
    # Remembered, not re-derived: the second call was one cache lookup.
    assert shadows._SPRITES[(260, 20)] is None


def test_an_oversize_squircle_is_declined_before_it_rasterises(monkeypatch):
    _fake_ninepatch(monkeypatch)
    calls: list[float] = []
    monkeypatch.setattr(
        surfaces,
        "pixels",
        lambda radius, exponent=surfaces.EXPONENT: calls.append(radius)
        or np.zeros((3, 3), np.uint8),
    )
    monkeypatch.setattr(effects, "SQUIRCLES", True)
    monkeypatch.setattr(surfaces, "_BROKEN", False)
    monkeypatch.setattr(surfaces, "_SPRITES", {})
    monkeypatch.setattr(surfaces, "_SCALE", tokens.SCALE, raising=False)
    assert 2 * int(math.ceil(300.0)) + 1 > ninepatch.MAX_SIZE
    assert surfaces.sprite(300.0) is None
    assert surfaces.sprite(300.0) is None
    assert calls == []
    assert surfaces._SPRITES[300] is None


def test_the_size_guard_has_one_owner():
    """The decline in the caches and the refusal in ``build`` are the same
    predicate, so they cannot disagree about the threshold."""
    assert ninepatch.fits((ninepatch.MAX_SIZE - 1) // 2)
    assert not ninepatch.fits((ninepatch.MAX_SIZE - 1) // 2 + 1)
    assert "fits(" in Path(ninepatch.__file__).read_text(encoding="utf-8")


# --- surface_fill: both paints decided before either draws -------------------


class _Draw:
    def __init__(self) -> None:
        self.quads: list[tuple] = []

    def add_image(self, ref, p_min, p_max, uv_min, uv_max, colour) -> None:
        self.quads.append((p_min, p_max))


def test_surface_fill_paints_neither_half_when_the_inset_fill_is_refused(monkeypatch):
    """There is a ~two-border-pixel window of sizes where the outer rect meets
    ``paint``'s floor and the rect inset by the border does not. Painting the
    border first meant returning False with it already on screen -- under the
    fallback fill the caller then draws on that False."""
    sprite = ninepatch.Sprite(texture=None, ref=object(), corner=10)
    monkeypatch.setattr(surfaces, "sprite", lambda radius: sprite)
    monkeypatch.setattr(tokens, "SCALE", 1.0)
    monkeypatch.setattr(widgets, "imgui", SimpleNamespace(get_color_u32=lambda c: 0))
    draw = _Draw()
    # 21 px: exactly 2 * corner + 1, so the outer rect fits and the rect inset
    # by the 1 px border (19 px) does not.
    assert widgets.surface_fill(draw, (0.0, 0.0), (21.0, 21.0), 12.0, (1, 1, 1, 1),
                                border=(0, 0, 0, 1)) is False
    assert draw.quads == []
    # Two border-pixels wider, both rects fit, and both halves paint.
    draw = _Draw()
    assert widgets.surface_fill(draw, (0.0, 0.0), (23.0, 23.0), 12.0, (1, 1, 1, 1),
                                border=(0, 0, 0, 1)) is True
    assert len(draw.quads) == 18


# --- the import pins ---------------------------------------------------------

STUDIO = Path(effects.__file__).parent

#: ``(absolute top-level roots, sibling modules imported at module scope)``.
#: The GL side -- moderngl, imgui_bundle, the backend -- is allowed only
#: behind lazy in-function imports, which the exact sets below refuse at the
#: top of the file without having to name them.
ALLOWED_TOP_LEVEL: dict[str, tuple[set[str], set[str]]] = {
    "effects.py": ({"__future__", "typing"}, set()),
    "ninepatch.py": ({"__future__", "logging", "dataclasses", "typing"}, set()),
    "shadows.py": ({"__future__", "logging", "math", "numpy"}, {"ninepatch", "tokens"}),
    "surfaces.py": ({"__future__", "logging", "math", "numpy"}, {"ninepatch", "tokens"}),
    "vibrancy.py": ({"__future__", "logging", "time", "typing"}, set()),
}

GL_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}


def _top_level(path: Path) -> tuple[set[str], set[str]]:
    """Only the imports at the top of the file, not the ones inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    absolute: set[str] = set()
    siblings: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                absolute.add((node.module or "").split(".")[0])
            elif node.module is None:
                siblings.update(alias.name for alias in node.names)
            else:
                siblings.add(node.module.split(".")[0])
    return absolute, siblings


def test_the_effect_modules_import_exactly_what_they_claim_to():
    """The docstrings claim ``vram.py``-style purity; this is the claim held
    to, in both directions -- a new outward import is a failing test and a
    decision rather than something found in a review later."""
    for name, expected in ALLOWED_TOP_LEVEL.items():
        assert _top_level(STUDIO / name) == expected, name


def test_no_effect_module_touches_gl_at_module_scope():
    """Named as well as implied by the exact sets above, because this is the
    half that makes the modules importable headlessly at all."""
    for name in ALLOWED_TOP_LEVEL:
        absolute, _siblings = _top_level(STUDIO / name)
        assert not (absolute & GL_ROOTS), f"{name} imports {absolute & GL_ROOTS} eagerly"


def test_no_effect_module_reaches_the_service_layer_or_the_queue():
    """At any depth, lazy or not: a sprite cache with a job queue behind it
    would put the whole service import graph under every pane that draws a
    shadow."""
    for name in ALLOWED_TOP_LEVEL:
        tree = ast.parse((STUDIO / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found = {node.module or ""}
            else:
                continue
            for module in found:
                assert not module.startswith("warlock.service"), f"{name} imports {module}"
                assert not module.startswith("warlock.queue"), f"{name} imports {module}"
                # ``warlock._q_*`` too: the queue's worker halves are the same
                # dependency wearing a different name, and the pin named only
                # the front door.
                assert not module.startswith("warlock._q"), f"{name} imports {module}"
