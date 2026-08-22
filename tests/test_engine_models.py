from __future__ import annotations

from warlock import fetch, models
from warlock.config import Config
from warlock.service import downloads


def _config(tmp_path):
    return Config(
        home=tmp_path / "home",
        data_dir=tmp_path / "home" / "assets",
        db_path=tmp_path / "home" / "jobs.sqlite",
        t2i_model_root=tmp_path / "image-models",
        trellis_models_dir=tmp_path / "engine-models",
    )


def test_engine_entry_is_first_and_uses_the_engine_root(tmp_path) -> None:
    config = _config(tmp_path)
    entry = fetch.entries()[0]
    assert entry.row_key == "engine:trellis_gguf"
    assert fetch.destination(config, entry, entry.fetch[0]) == config.trellis_models_dir


def test_engine_presence_requires_the_exact_pipeline(tmp_path) -> None:
    config = _config(tmp_path)
    spec = models.ENGINE_MODELS["trellis_gguf"]
    config.trellis_models_dir.mkdir(parents=True)
    for name in spec.probe[:-1]:
        (config.trellis_models_dir / name).write_bytes(b"weights")
    assert fetch.present(config, "engine", spec) is False
    (config.trellis_models_dir / spec.probe[-1]).write_bytes(b"weights")
    assert fetch.present(config, "engine", spec) is True


def test_engine_uninstall_stages_on_the_engine_volume(svc, monkeypatch) -> None:
    """WARLOCK_TRELLIS_MODELS may point at a drive unlike image models."""
    spec = models.ENGINE_MODELS["trellis_gguf"]
    svc.config.trellis_models_dir.mkdir(parents=True, exist_ok=True)
    for name in spec.probe:
        (svc.config.trellis_models_dir / name).write_bytes(b"weights")

    real_rename = downloads.os.rename

    def same_parent(src, dst):
        assert src.parent == dst.parent
        return real_rename(src, dst)

    monkeypatch.setattr(downloads.os, "rename", same_parent)
    result = downloads.uninstall(svc, ["engine:trellis_gguf"])

    assert result["removed"] == [str(svc.config.trellis_models_dir)]
    assert not svc.config.trellis_models_dir.exists()


def test_engine_download_command_uses_literal_powershell_quoting(tmp_path) -> None:
    config = _config(tmp_path / "a $literal directory")
    text = fetch.download_text(config, "engine", models.ENGINE_MODELS["trellis_gguf"])
    assert fetch.quote_for_shell(config.trellis_models_dir) in text
    assert f'"{config.trellis_models_dir}"' not in text
