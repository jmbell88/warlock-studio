"""Fingerprints and the locked recipe."""

from __future__ import annotations

import json

from warlock import provenance
from warlock.config import Config


def test_a_missing_path_says_so_rather_than_raising(tmp_path):
    assert provenance.file_fingerprint(tmp_path / "nope.safetensors") == "missing"


def test_a_fingerprint_is_stable_and_moves_with_content(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"a" * 4096)
    first = provenance.file_fingerprint(f)
    assert provenance.file_fingerprint(f) == first

    # Cleared explicitly: the cache is keyed on (path, size, mtime_ns), and a
    # rewrite this fast lands inside the filesystem's timestamp resolution.
    f.write_bytes(b"b" * 4096)
    provenance._cache.clear()
    assert provenance.file_fingerprint(f) != first


def test_size_is_part_of_the_fingerprint(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"x" * 200)
    assert provenance.file_fingerprint(a) != provenance.file_fingerprint(b)


def test_only_the_ends_of_a_large_file_are_hashed(tmp_path):
    # The point of head+tail: a 7 GB checkpoint must not cost 30 s of disk per
    # job. A middle-byte edit is therefore knowingly invisible.
    big = tmp_path / "big.bin"
    size = 3 * provenance.HASH_BYTES
    big.write_bytes(b"\0" * size)
    first = provenance.file_fingerprint(big)

    provenance._cache.clear()
    data = bytearray(big.read_bytes())
    data[size // 2] = 1
    big.write_bytes(bytes(data))
    provenance._cache.clear()
    assert provenance.file_fingerprint(big) == first


def test_a_directory_fingerprint_covers_its_files(tmp_path):
    d = tmp_path / "model"
    (d / "unet").mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "unet" / "w.safetensors").write_bytes(b"weights")
    first = provenance.file_fingerprint(d)

    (d / "config.json").write_text('{"x": 1}')
    provenance._cache.clear()
    assert provenance.file_fingerprint(d) != first


def test_model_fingerprints_skips_the_absent_entries(tmp_path):
    out = provenance.model_fingerprints({"a": tmp_path, "b": None})
    assert set(out) == {"a"}


def test_versions_never_imports_anything(monkeypatch):
    import builtins

    real = builtins.__import__

    def refusing(name, *args, **kwargs):
        if name in ("torch", "diffusers"):
            raise AssertionError("versions() must only read sys.modules")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing)
    out = provenance.versions()
    assert out["recipe"] == str(provenance.RECIPE_VERSION)
    assert "python" in out


def test_versions_records_the_prompt_compiler(monkeypatch):
    from warlock.pipelines import prompt  # noqa: F401  (ensures it is imported)

    assert provenance.versions()["prompt"] == str(prompt.PROMPT_VERSION)


def test_trellis_recipe_is_serialisable():
    recipe = provenance.trellis_recipe(Config(), {"platform": "pc"}, mesh_seed=42)
    json.dumps(recipe)
    assert recipe["seed"] == 42
    assert recipe["version"] == provenance.RECIPE_VERSION


def test_a_pinned_server_config_is_what_the_recipe_records():
    """A job that restarted trellis-server with its own band/tex-res must not
    have the config's values recorded against it -- the recipe would describe a
    server that never ran."""
    config = Config()
    pinned = provenance.trellis_recipe(
        config, {"trellis_band": 8, "trellis_tex_res": 2048}, mesh_seed=42
    )
    assert (pinned["band"], pinned["tex_res"]) == (8, 2048)

    plain = provenance.trellis_recipe(config, {}, mesh_seed=42)
    assert (plain["band"], plain["tex_res"]) == (
        config.trellis_band,
        config.trellis_tex_res,
    )
