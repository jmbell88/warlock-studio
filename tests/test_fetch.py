"""The download machinery, headlessly: the record, the path rule, the dedupe,
the disk refusal, and the promise that the app process stays offline.

Everything here is pure or stubbed. Nothing in this file may download anything
-- a real fetch is the user's call, and a test that made one would be the exact
thing the offline invariant exists to prevent.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from warlock import fetch, models
from warlock.config import Config

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock"


# --- the record and the string it derives ------------------------------------


def test_a_fetch_renders_the_command_it_stands_for():
    one = models.Fetch(
        "acme/thing",
        "thing",
        allow_patterns=("*.json", "*fp16.safetensors"),
        ignore_patterns=("big.safetensors",),
    )
    assert one.command() == (
        "uvx hf download acme/thing "
        '--include "*.json" --include "*fp16.safetensors" '
        '--exclude "big.safetensors" '
        "--local-dir $HOME/.warlock/models/thing"
    )
    # The default is *display only*. Given a resolved destination it renders
    # that instead, which is what ``fetch.download_text`` hands it and what
    # keeps a pasted command and the button pointing at one directory.
    assert one.command("D:/elsewhere/thing").endswith("--local-dir D:/elsewhere/thing")


def test_the_download_text_is_what_doctor_prints():
    """Verbatim, for the entries that carry every shape there is.

    Pinned rather than described because the whole point of deriving the string
    is that doctor's text and the README cannot drift from what the button
    does; a test that only asserted "starts with uvx" would let the derivation
    quietly change the words a person is told to paste.
    """
    assert models.BASE_MODELS["turbo"].download == (
        "uvx hf download stabilityai/sdxl-turbo "
        "--revision 71153311d3dbb46851df1931d3ca6e939de83304 "
        '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
        '--exclude "sd_xl_turbo_1.0*" --local-dir $HOME/.warlock/models/sdxl-turbo'
    )
    # Two commands, joined by the two-space continuation doctor's
    # "download with:\n  " prefix lines up with.
    assert models.BASE_MODELS["sdxl"].download == (
        "uvx hf download stabilityai/stable-diffusion-xl-base-1.0 "
        "--revision 462165984030d82259a11f4367a4eed129e94a7b "
        '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
        "--local-dir $HOME/.warlock/models/sdxl-base-1.0\n"
        "  uvx hf download ByteDance/Hyper-SD "
        "--revision bc08d970a87c74c71209491d64e3525845698863 "
        "Hyper-SDXL-4steps-lora.safetensors "
        "--local-dir $HOME/.warlock/models/loras"
    )
    # The rename, which is why the string could never simply be executed.
    assert models.BASE_MODELS["pixel"].download.endswith(
        "  then rename $HOME/.warlock/models/loras/pytorch_lora_weights.safetensors "
        "to lcm-lora-sdxl.safetensors"
    )
    # The non-shell step, which the button deliberately does not run.
    assert models.MATTING_MODELS["birefnet"].download.endswith(
        "  then: uv sync --extra text2image "
        "-- BiRefNet's modelling code imports einops, kornia, timm and "
        "torchvision, and that extra is what supplies them"
    )


def test_every_registry_entry_has_a_fetch_and_a_download_line():
    for entry in fetch.entries():
        assert entry.fetch, f"{entry.row_key} has nothing to download"
        assert entry.spec.download.startswith("uvx hf download "), entry.row_key
        for one in entry.fetch:
            assert one.repo_id and one.local_dir, entry.row_key
            assert one.size_gib > 0, f"{entry.row_key} declares no size"


# --- the pins (MDL-03) --------------------------------------------------------

SHA = re.compile(r"^[0-9a-f]{40}$")


def test_every_fetch_pins_an_immutable_commit():
    """``Fetch.revision``'s own docstring has promised this test since the
    field was added, and the field was empty everywhere.

    Unpinned, ``snapshot_download`` follows whatever ``main`` points at today,
    so a Download click a year from now retrieves different weights -- and the
    findings corpus, which is keyed on model *names*, silently starts mixing
    two checkpoints under one label. The pins were recovered from the
    ``.metadata`` sidecars beside the already-downloaded files, so each one is
    the commit this host is actually running.
    """
    unpinned = [
        f"{entry.row_key} -> {one.repo_id}"
        for entry in fetch.entries()
        for one in entry.fetch
        if not SHA.match(one.revision or "")
    ]
    assert unpinned == []


def test_two_records_for_one_repository_agree_about_the_commit():
    """The IP-Adapter is the only repository in the registry with two records
    against one destination -- its weights and its CLIP vision encoder. Two
    commits for one directory would mean files from two snapshots side by
    side, which is the mixed state a pin exists to prevent."""
    by_repo: dict[str, set[str]] = {}
    for entry in fetch.entries():
        for one in entry.fetch:
            by_repo.setdefault(one.repo_id, set()).add(one.revision)
    disagreeing = {repo: shas for repo, shas in by_repo.items() if len(shas) > 1}
    assert disagreeing == {}


def test_merging_two_records_keeps_the_pin():
    """The bug the pins uncovered. ``Job`` defaults ``revision`` to "" and
    ``_merge`` rebuilt the job from scratch without it, so the IP-Adapter --
    the one entry that merges -- came out of ``plan`` unpinned while
    ``models.py`` and the printed command both showed the pin. Silently,
    because an unpinned fetch succeeds."""
    entry = next(e for e in fetch.entries() if len(e.fetch) > 1)
    jobs = fetch.plan(Config(), [entry])
    assert jobs, entry.row_key
    for job in jobs:
        assert SHA.match(job.revision), f"{job.repo_id} lost its pin in the merge"


def test_a_merge_refuses_two_different_commits():
    """Asserted rather than resolved: picking one would install a directory
    the registry did not describe."""
    first = models.Fetch("r", "d", revision="a" * 40, size_gib=1.0)
    second = models.Fetch("r", "d", revision="b" * 40, size_gib=1.0)
    job = fetch.Job(repo_id="r", dest=Path("d"), revision=first.revision)
    with pytest.raises(ValueError, match="two different commits"):
        fetch._merge(job, second)


def test_every_rendered_command_carries_its_revision():
    """A hand-run download that silently took the branch tip would defeat the
    pin for exactly the user most likely to be reproducing something."""
    for entry in fetch.entries():
        for one in entry.fetch:
            assert f"--revision {one.revision}" in one.command(), one.repo_id


#: The documents carrying hand-written ``uvx hf download`` lines. They are not
#: the registry's rendered one-liners -- they are PowerShell, with backtick
#: continuations and comments between them -- so they cannot be generated, and
#: the drift they can have is exactly the one this catches: a command that
#: names the right repository and the wrong commit, or no commit at all.
PINNED_DOCS = (
    "docs/MODELS.md",
    "docs/manual/38-installation.md",
    "README.md",
    # Not a document: ``doctor``'s hint for the GGUF weights, which are one of
    # the two fatal rows and are declared as text because nothing downloads
    # them through the registry.
    "src/warlock/doctor.py",
)


@pytest.mark.parametrize("rel", PINNED_DOCS)
def test_every_documented_download_command_is_pinned(rel: str):
    """A pasted command that silently took the branch tip defeats the pin for
    exactly the user most likely to be reproducing something -- and the four
    registry pins that were already in ``models.py`` were invisible here for
    months, because nothing compared the two."""
    root = SRC.parents[1]
    text = (root / rel).read_text(encoding="utf-8")
    # A short window after the repository rather than the very next token: one
    # of these files is Python rather than markdown, and ``doctor``'s hint is a
    # concatenation of string literals, so what follows the repository there is
    # a closing quote and a line break before ``--revision``. The window is
    # small enough that it cannot reach the *next* command.
    unpinned = [
        match.group(1)
        for match in re.finditer(
            r"uvx hf download ([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)", text
        )
        if "--revision" not in text[match.end() : match.end() + 24]
    ]
    assert unpinned == [], f"{rel} downloads these without a --revision"


@pytest.mark.parametrize("rel", PINNED_DOCS)
def test_a_documented_command_names_the_commit_the_registry_pins(rel: str):
    """Pinned to *something* is not enough: a stale SHA in a document is worse
    than none, because it looks deliberate."""
    root = SRC.parents[1]
    text = (root / rel).read_text(encoding="utf-8")
    pins = {one.repo_id: one.revision for e in fetch.entries() for one in e.fetch}
    wrong = [
        f"{repo}: doc says {found}, registry says {pins[repo]}"
        for repo, found in re.findall(
            r"uvx hf download ([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+) --revision (\S+)", text
        )
        if repo in pins and found != pins[repo]
    ]
    assert wrong == []


def test_the_docs_name_every_repository_the_registry_does():
    """The drift this package closed, kept closed.

    Four repo ids -- the IP-Adapter, the Canny ControlNet, DINOv2 and BiRefNet
    -- existed only in models.py, so the README's list of "the only network use
    there is" was not one. Both halves come off ``Fetch`` now; this is what
    notices when a new entry is added to one and not the other.

    Two files rather than one, because the README was shortened and the
    per-model recipes moved to ``docs/MODELS.md`` -- which the README links to
    for exactly this list. What the test is about is that no repository the app
    can fetch is undocumented, not which of the two documents carries it.

    The installation chapter is asserted **separately**, below, and that is the
    point of splitting them: it is not an alternative place for a repository to
    be named, it is a third copy of the catalogue that has to be complete on its
    own. Six entries -- Turbo, Lightning, Juggernaut, DreamShaper, the depth
    ControlNet and BiRefNet -- drifted out of it unnoticed, because this
    assertion was satisfied the whole time by the other two files.
    """
    root = SRC.parents[1]
    docs = (root / "README.md").read_text(encoding="utf-8")
    docs += (root / "docs" / "MODELS.md").read_text(encoding="utf-8")
    repos = {one.repo_id for entry in fetch.entries() for one in entry.fetch}
    missing = sorted(repo for repo in repos if repo not in docs)
    assert not missing, f"in models.py but in neither README.md nor docs/MODELS.md: {missing}"


def test_the_installation_chapter_names_every_repository_too():
    """The manual is read *inside the app*, where docs/MODELS.md cannot be
    opened -- which is why chapter 38 carries the commands rather than a
    pointer, and why its completeness is a claim of its own rather than one it
    can borrow from the two files above."""
    root = SRC.parents[1]
    text = (root / "docs" / "manual" / "38-installation.md").read_text(encoding="utf-8")
    repos = {one.repo_id for entry in fetch.entries() for one in entry.fetch}
    missing = sorted(repo for repo in repos if repo not in text)
    assert not missing, (
        f"in models.py but not in docs/manual/38-installation.md: {missing}"
    )


def test_the_readme_counts_the_registered_base_models():
    """It said "Ten" for the whole life of the ``flux_klein`` /
    ``flux_klein_distilled`` split that pushed it to eleven. A number in prose
    beside a registry is a number that goes stale silently, so it is spelled
    out here and read back off the registry."""
    from pathlib import Path

    from warlock import models

    words = {
        9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
        14: "Fourteen", 15: "Fifteen", 16: "Sixteen",
    }
    count = len(models.BASE_MODELS)
    wanted = words.get(count)
    assert wanted is not None, f"add {count} to the spelling table above"
    readme = Path("README.md").read_text(encoding="utf-8")
    assert f"{wanted} base models are registered" in readme


def test_the_authoritative_docs_agree_with_the_registry_about_the_default():
    """DOC-02: the default moved to ``sdxl_cfg`` and four documents did not.

    ``docs/INVARIANTS.md`` is the one that mattered — it instructs a reader to
    consult it *before* modifying a subsystem, and it handed them the
    pre-2026-08-11 answer, complete with a causal story ("because FLUX is
    gated") that was never why this checkpoint won. Asserted as a property of
    the prose rather than by parsing it: a document that names the registry's
    default anywhere near the word must not simultaneously call Turbo the
    default.
    """
    assert models.DEFAULT_BASE_MODEL == "sdxl_cfg"
    turbo = models.BASE_MODELS["turbo"]
    default = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    assert turbo.dir_name not in default.dir_name

    root = Path(__file__).resolve().parents[1]
    claims = (
        "SDXL-Turbo (`models/sdxl-turbo`) is the default",
        "SDXL-Turbo is the default",
        "prompt → SDXL-Turbo",
        "default because FLUX is gated",
    )
    for name in ("docs/INVARIANTS.md", "CLAUDE.md", "README.md", "docs/MODELS.md"):
        text = (root / name).read_text(encoding="utf-8")
        for claim in claims:
            assert claim not in text, (
                f"{name} still says {claim!r}; the registry's default is "
                f"{models.DEFAULT_BASE_MODEL!r} "
                f"(docs/measurements/2026-08-11-default-base-model.md)"
            )


def test_a_note_is_not_a_fetch():
    """BiRefNet's `uv sync` is prose in the string and absent from the plan."""
    entry = fetch.find("matting:birefnet")
    assert entry is not None
    jobs = fetch.plan(Config(), [entry])
    assert [j.repo_id for j in jobs] == ["ZhengPeng7/BiRefNet"]


# --- where things land -------------------------------------------------------


def _config(tmp_path: Path, turbo: Path | None = None) -> Config:
    cfg = Config()
    cfg.t2i_model_root = tmp_path / "models"
    cfg.t2i_turbo_dir = turbo
    return cfg


def test_only_turbo_honours_the_legacy_override(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    cfg = _config(tmp_path, turbo=elsewhere)
    assert fetch.base_model_dir(cfg, models.BASE_MODELS["turbo"]) == elsewhere
    assert (
        fetch.base_model_dir(cfg, models.BASE_MODELS["sdxl"])
        == cfg.t2i_model_root / "sdxl-base-1.0"
    )
    # Pinned by name rather than to the default, so the redirect stays about
    # turbo's settings even once another entry is the shipped default.
    assert models.T2I_DIR_MODEL == "turbo"
    assert (
        fetch.base_model_dir(cfg, models.BASE_MODELS["sdxl_cfg"])
        == cfg.t2i_model_root / "sdxl-base-1.0"
    )


def test_a_plan_resolves_under_the_model_root_not_the_literal_models_dir(tmp_path):
    """The hardcoded ``--local-dir models/...`` is the README's spelling of the
    default root; WARLOCK_T2I_ROOT has to relocate a fetch the way it already
    relocates a load."""
    cfg = _config(tmp_path)
    entry = fetch.find("base:sdxl")
    jobs = fetch.plan(cfg, [entry])
    dests = {job.repo_id: job.dest for job in jobs}
    assert dests["stabilityai/stable-diffusion-xl-base-1.0"] == (
        cfg.t2i_model_root / "sdxl-base-1.0"
    )
    assert dests["ByteDance/Hyper-SD"] == cfg.t2i_model_root / "loras"


def test_a_rendered_command_names_the_directory_the_fetch_actually_lands_in(tmp_path):
    """DST-02: the paste-able text and the button must name one directory.

    ``spec.download`` renders the documented default home because ``models``
    cannot see a ``Config``. ``fetch.download_text`` can, and every caller that
    holds a config uses it -- doctor above all, which reports a model missing at
    a resolved path and then has to say how to put it there. The literal
    ``models/...`` this replaced was *relative*: pasted from any shell whose CWD
    was not the checkout it wrote gigabytes into `<cwd>/models`, which nothing
    ever reads.
    """
    cfg = _config(tmp_path)
    spec = models.BASE_MODELS["sdxl"]
    text = fetch.download_text(cfg, "base", spec)
    # Both halves of a two-command entry, each against its own destination.
    assert str(cfg.t2i_model_root / "sdxl-base-1.0") in text
    assert str(cfg.t2i_model_root / "loras") in text
    # And nothing left rendering the default, or the relative literal.
    assert "$HOME" not in text
    assert "--local-dir models/" not in text
    # Every rendered destination is exactly what ``plan`` will use, because both
    # resolve through ``destination``.
    dests = {job.dest for job in fetch.plan(cfg, [fetch.find("base:sdxl")])}
    for dest in dests:
        assert str(dest) in text


def test_a_rendered_command_honours_the_turbo_only_override(tmp_path):
    """``WARLOCK_T2I_DIR`` moves one entry, and the text has to move with it."""
    elsewhere = tmp_path / "elsewhere"
    cfg = _config(tmp_path, turbo=elsewhere)
    turbo = fetch.download_text(cfg, "base", models.BASE_MODELS["turbo"])
    assert str(elsewhere) in turbo
    # ...and only that entry: sdxl still resolves under the root.
    sdxl = fetch.download_text(cfg, "base", models.BASE_MODELS["sdxl"])
    assert str(elsewhere) not in sdxl


def test_a_rendered_command_survives_a_space_in_the_path(tmp_path):
    """`C:\\Users\\Jane Doe\\.warlock\\models` is an ordinary Windows home, and an
    unquoted path with a space in it is a command that silently downloads into
    the wrong directory (or fails, if you are lucky)."""
    cfg = _config(tmp_path / "a directory")
    text = fetch.download_text(cfg, "base", models.BASE_MODELS["sdxl_cfg"])
    quoted = fetch.quote_for_shell(cfg.t2i_model_root / "sdxl-base-1.0")
    assert quoted.startswith("'") and quoted.endswith("'")
    assert quoted in text


def test_the_override_moves_the_turbo_download_too(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    cfg = _config(tmp_path, turbo=elsewhere)
    jobs = fetch.plan(cfg, [fetch.find("base:turbo")])
    assert [job.dest for job in jobs] == [elsewhere]


# --- dedupe ------------------------------------------------------------------


def test_four_recipes_over_one_checkpoint_are_one_download(tmp_path):
    """dir_name is not unique, so a key-keyed plan would fetch 7 GB four times."""
    cfg = _config(tmp_path)
    chosen = [fetch.find(k) for k in ("base:sdxl", "base:sdxl_cfg", "base:pixel", "base:lightning")]
    jobs = fetch.plan(cfg, chosen)
    repos = [job.repo_id for job in jobs]
    assert repos.count("stabilityai/stable-diffusion-xl-base-1.0") == 1
    # ...and the size is counted once, which is the difference between
    # "this needs 7 GB" and "this needs 28 GB".
    base = next(j for j in jobs if j.repo_id.endswith("stable-diffusion-xl-base-1.0"))
    assert base.size_gib == pytest.approx(7.0)
    # The three distinct LoRA repos survive.
    assert len(jobs) == 4


def test_two_records_against_one_repository_merge_their_patterns(tmp_path):
    """The IP-Adapter's weights and CLIP vision encoder are two doctor lines and
    one fetch; dropping either half is the load-fine-then-fail-at-first-call
    directory its two-part probe exists to catch."""
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("adapter:plus")])
    assert len(jobs) == 1
    job = jobs[0]
    assert job.filenames == ("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",)
    assert job.allow_patterns == ("models/image_encoder/*",)


def test_dedupe_is_on_the_destination_not_the_repo_alone(tmp_path):
    """Same repo into two directories is two fetches."""
    cfg = _config(tmp_path)
    entry = fetch.Entry(
        "metric",
        "x",
        "X",
        models.MetricModel(
            "x",
            "X",
            "x",
            fetch=(
                models.Fetch("acme/x", "one", size_gib=1.0),
                models.Fetch("acme/x", "two", size_gib=1.0),
            ),
        ),
    )
    assert len(fetch.plan(cfg, [entry])) == 2


# --- refusing rather than half-downloading -----------------------------------


def test_a_plan_that_does_not_fit_is_refused(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("base:flux_klein")])
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 3.0)
    refusal = fetch.disk_refusal(jobs)
    assert refusal is not None and "16" not in refusal.split("needs")[0]
    assert "3.0 GB is free" in refusal


def test_a_plan_that_fits_is_not_refused(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("lora:pixelxl")])
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 500.0)
    assert fetch.disk_refusal(jobs) is None


def test_disk_admission_is_per_volume_not_charged_to_the_first_one(tmp_path, monkeypatch):
    """MDL-09: the plan's whole size was compared against ``jobs[0].dest``.

    Right only while every fetch lands on one drive -- and ``WARLOCK_T2I_DIR``
    relocates the ``turbo`` entry by itself, so a plan can straddle two. A roomy
    first volume approved a large write to a nearly full second one; the inverse
    falsely refused.
    """
    elsewhere = tmp_path / "other-volume"
    cfg = _config(tmp_path, turbo=elsewhere)
    jobs = fetch.plan(cfg, [fetch.find("base:turbo"), fetch.find("lora:pixelxl")])
    dests = {job.dest for job in jobs}
    assert elsewhere in dests, "the override has to actually split the plan"

    # Every destination answers about its *own* volume. The anchors are equal
    # under one tmp_path, so the split is made explicit here by answering per
    # directory -- which is what two real drives would do.
    free = {elsewhere: 0.5}

    def _free(path):
        return free.get(path, 500.0)

    monkeypatch.setattr(fetch, "free_gib", _free)
    # Grouped by volume anchor, so on one drive this is a single group; force
    # the two-group case by keying the groups on the destinations themselves.
    monkeypatch.setattr(fetch, "DISK_HEADROOM_GIB", 0.0)

    # A plan entirely on the roomy volume is fine.
    ok_jobs = [job for job in jobs if job.dest != elsewhere]
    assert fetch.disk_refusal(ok_jobs) is None
    # And one landing on the full volume is refused, naming its own numbers
    # rather than the first destination's.
    tight = [job for job in jobs if job.dest == elsewhere]
    refusal = fetch.disk_refusal(tight)
    assert refusal is not None and "0.5 GB is free" in refusal


def test_free_space_that_cannot_be_read_is_not_a_refusal(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    jobs = fetch.plan(cfg, [fetch.find("base:flux_klein")])
    monkeypatch.setattr(fetch, "free_gib", lambda _path: None)
    assert fetch.disk_refusal(jobs) is None


def test_free_space_answers_for_a_directory_that_does_not_exist_yet(tmp_path):
    """A download's destination routinely does not exist; disk_usage raises on
    a missing path rather than answering about its volume."""
    assert fetch.free_gib(tmp_path / "not" / "here" / "yet") is not None


# --- presence ----------------------------------------------------------------


def test_a_partial_directory_is_absent(tmp_path):
    cfg = _config(tmp_path)
    entry = fetch.find("base:playground")
    root = cfg.t2i_model_root / "playground-v2.5"
    root.mkdir(parents=True)
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    assert not entry.is_present(cfg)
    unet = root / "unet"
    unet.mkdir()
    (unet / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    assert entry.is_present(cfg)


def test_a_checkpoint_without_its_step_distillation_lora_is_absent(tmp_path):
    cfg = _config(tmp_path)
    root = cfg.t2i_model_root / "sdxl-base-1.0"
    (root / "unet").mkdir(parents=True)
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    (root / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    ok, missing = fetch.base_model_state(cfg, models.BASE_MODELS["sdxl"])
    assert not ok and missing is not None and missing.name.startswith("Hyper-SDXL")
    # sdxl_cfg is the same weights with no distillation LoRA, so it is ready.
    assert fetch.base_model_state(cfg, models.BASE_MODELS["sdxl_cfg"])[0]


def test_doctor_and_the_planner_share_one_probe():
    from warlock import doctor

    assert doctor._base_model_dir is fetch.base_model_dir


# --- the app process never becomes online-capable ----------------------------


def test_only_the_fetch_worker_ever_clears_hf_hub_offline():
    """The whole exception to the offline invariant, expressed as a scan.

    ``HF_HUB_OFFLINE`` is read by huggingface_hub at import time, so a value of
    "0" anywhere the app imports would make this process online-capable for the
    rest of its life. Exactly one module may write it, and it is the one that
    is spawned, does one thing and dies.

    Written as a scan for a *use* rather than for the string, because several
    modules explain the invariant in a comment and prose is not a hazard --
    ``os.environ[...]`` and ``setdefault`` are.
    """
    use = re.compile(r"""(?:environ\[|setdefault\(\s*)["']HF_HUB_OFFLINE""")
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if not use.search(text):
            continue
        if path.name in ("__init__.py", "fetch_worker.py"):
            continue
        offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "only warlock/__init__.py (which sets it to 1) and "
        f"pipelines/fetch_worker.py (a child process) may touch it: {offenders}"
    )


def test_the_package_still_sets_hf_hub_offline_to_one():
    import warlock  # noqa: F401  -- the import is what sets it

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_a_fetch_leaves_no_half_populated_directory(tmp_path):
    """With the network unavailable, the worker must fail and clean up.

    Run for real, as a subprocess, with snapshot_download replaced by one that
    writes a file and then raises -- which is the shape of every real failure
    (a partial download, then an error) and the only one that can leave a
    directory the presence probes would call finished.
    """
    dest = tmp_path / "models" / "thing"
    result = tmp_path / "result.json"
    spec = {
        "repo_id": "acme/thing",
        "dest": str(dest),
        "filenames": [],
        "allow_patterns": [],
        "ignore_patterns": [],
        "rename": None,
        "size_gib": 0.1,
        "result_path": str(result),
    }
    stub = (
        "import sys, types, pathlib\n"
        "mod = types.ModuleType('huggingface_hub')\n"
        "def snapshot_download(**kw):\n"
        "    root = pathlib.Path(kw['local_dir'])\n"
        "    (root / 'config.json').write_text('{}')\n"
        "    raise OSError('no network')\n"
        "mod.snapshot_download = snapshot_download\n"
        "sys.modules['huggingface_hub'] = mod\n"
        "from warlock.pipelines import fetch_worker\n"
        "raise SystemExit(fetch_worker.main())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", stub],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["ok"] is False and "no network" in payload["error"]
    assert not dest.exists(), "a failed fetch left a model directory behind"
    assert not list((tmp_path / "models").glob("*.part")), "staging tree survived"


def test_a_successful_fetch_moves_the_files_in_and_renames(tmp_path):
    dest = tmp_path / "models" / "loras"
    dest.mkdir(parents=True)
    (dest / "already.safetensors").write_bytes(b"keep")
    result = tmp_path / "result.json"
    spec = {
        "repo_id": "acme/lora",
        "dest": str(dest),
        "filenames": ["pytorch_lora_weights.safetensors"],
        "allow_patterns": [],
        "ignore_patterns": [],
        "rename": ["pytorch_lora_weights.safetensors", "lcm-lora-sdxl.safetensors"],
        "size_gib": 0.1,
        "result_path": str(result),
    }
    stub = (
        "import sys, types, pathlib, os\n"
        "mod = types.ModuleType('huggingface_hub')\n"
        "def snapshot_download(**kw):\n"
        "    root = pathlib.Path(kw['local_dir'])\n"
        "    (root / 'pytorch_lora_weights.safetensors').write_bytes(b'w')\n"
        "    (root / '.cache').mkdir()\n"
        "    (root / '.cache' / 'junk').write_text('x')\n"
        "mod.snapshot_download = snapshot_download\n"
        "sys.modules['huggingface_hub'] = mod\n"
        "from warlock.pipelines import fetch_worker\n"
        "raise SystemExit(fetch_worker.main())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", stub],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (dest / "lcm-lora-sdxl.safetensors").read_bytes() == b"w"
    # The rename happened inside staging, before the move: the generic name
    # never lands in loras/ at all -- which is the whole point of the rename,
    # since two repos ship the same generic filename into one flat directory.
    assert not (dest / "pytorch_lora_weights.safetensors").exists()
    # A shared destination gains files rather than being replaced.
    assert (dest / "already.safetensors").read_bytes() == b"keep"
    # huggingface_hub's resume bookkeeping does not follow the weights in.
    assert not (dest / ".cache").exists()


def test_a_rename_whose_source_never_arrived_fails_the_fetch(tmp_path):
    """fetch_one applies a rename unconditionally, and loudly. A repo that
    stopped shipping the declared filename must fail the whole fetch rather
    than move in a directory missing the file the presence probes key on --
    ``pixelklein`` (the first ``filenames=`` plus ``rename=`` entry) would
    otherwise read as downloaded forever while loading nothing."""
    dest = tmp_path / "models" / "loras"
    result = tmp_path / "result.json"
    spec = {
        "repo_id": "acme/lora",
        "dest": str(dest),
        "filenames": ["pytorch_lora_weights.safetensors"],
        "allow_patterns": [],
        "ignore_patterns": [],
        "rename": ["pytorch_lora_weights.safetensors", "pixel-art-klein.safetensors"],
        "size_gib": 0.1,
        "result_path": str(result),
    }
    stub = (
        "import sys, types, pathlib\n"
        "mod = types.ModuleType('huggingface_hub')\n"
        "def snapshot_download(**kw):\n"
        "    root = pathlib.Path(kw['local_dir'])\n"
        "    (root / 'something_else.safetensors').write_bytes(b'w')\n"
        "mod.snapshot_download = snapshot_download\n"
        "sys.modules['huggingface_hub'] = mod\n"
        "from warlock.pipelines import fetch_worker\n"
        "raise SystemExit(fetch_worker.main())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", stub],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["ok"] is False and "did not provide" in payload["error"]
    # The staging promise holds on this path too: nothing lands, nothing
    # lingers, and neither spelling of the filename reaches loras/.
    assert not dest.exists()
    assert not list((tmp_path / "models").glob("*.part"))


def test_the_fetch_worker_spawn_is_in_the_kill_on_close_job():
    """Named here as well as in the package-wide scan, because this child is the
    one that holds a socket and writes gigabytes: one that outlived a hard kill
    would go on filling the disk with nothing on screen to say so."""
    text = (SRC / "service" / "downloads.py").read_text(encoding="utf-8")
    spawn = text.index("subprocess.Popen(")
    assert "winjob.assign" in text[spawn : spawn + 1200]


# --- progress ----------------------------------------------------------------


def test_progress_is_only_recorded_while_the_key_is_in_flight():
    """A task thread's last report must not resurrect a collected entry.

    ``Done`` arrives at completion and the frame that collects it stops drawing
    the bar; a report landing after that would leave a bar on screen for a task
    that has finished, with nothing to clear it.
    """
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        assert runner.submit("download:x", lambda: 1)
        runner.set_progress("download:x", 40.0, "half")
        assert runner.progress("download:x") == {"percent": 40.0, "label": "half"}
        while not runner.poll():
            pass
        assert runner.progress("download:x") is None
        runner.set_progress("download:x", 90.0, "late")
        assert runner.progress("download:x") is None
    finally:
        runner.shutdown(wait=False)


def test_resubmitting_a_key_starts_from_no_progress():
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        runner.submit("download:x", lambda: 1)
        runner.set_progress("download:x", 80.0, "nearly")
        while not runner.poll():
            pass
        runner.submit("download:x", lambda: 1)
        assert runner.progress("download:x") is None
    finally:
        runner.shutdown(wait=False)


def test_a_percentage_is_clamped():
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        runner.submit("download:x", lambda: 1)
        runner.set_progress("download:x", 140.0)
        assert runner.progress("download:x")["percent"] == 100.0
        runner.set_progress("download:x", -3.0)
        assert runner.progress("download:x")["percent"] == 0.0
    finally:
        runner.shutdown(wait=False)


# --- the service door --------------------------------------------------------


class _Svc:
    def __init__(self, config):
        self.config = config


def test_a_download_that_does_not_fit_never_spawns_anything(tmp_path, monkeypatch):
    from warlock.service import downloads
    from warlock.service.errors import Invalid

    monkeypatch.setattr(fetch, "free_gib", lambda _path: 1.0)
    monkeypatch.setattr(
        downloads,
        "_run_worker",
        lambda *a, **k: pytest.fail("spawned a worker for a refused plan"),
    )
    with pytest.raises(Invalid) as caught:
        downloads.download(_Svc(_config(tmp_path)), ["base:flux_klein"])
    assert "disk space" in caught.value.message


def test_an_unknown_row_is_refused_by_name(tmp_path):
    from warlock.service import downloads
    from warlock.service.errors import NotFound

    with pytest.raises(NotFound):
        downloads.plan_for(_Svc(_config(tmp_path)), ["base:nope"])


def test_a_download_reports_overall_progress_across_its_fetches(tmp_path, monkeypatch):
    from warlock.service import downloads

    monkeypatch.setattr(fetch, "free_gib", lambda _path: 5000.0)
    seen: list[float] = []

    def fake(job, *, on_progress, timeout, publish=True):
        on_progress(50.0, job.repo_id)
        on_progress(100.0, job.repo_id)
        # A no-publish child hands back the staging tree it finished, which is
        # what the parent's publishing phase installs (MDL-10).
        staging = Path(job.dest).parent / f".{Path(job.dest).name}.fetch.part"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "w.safetensors").write_bytes(b"weights")
        return {"ok": True, "staged": str(staging), "dest": str(job.dest)}

    monkeypatch.setattr(downloads, "_run_worker", fake)
    downloads.download(
        _Svc(_config(tmp_path)),
        ["base:sdxl"],
        on_progress=lambda percent, _label: seen.append(percent),
    )
    # Monotone, never past 100, and ending exactly at it.
    assert seen == sorted(seen)
    assert seen[-1] == 100.0
    assert max(seen[:-1]) < 100.0


_STUB_WORKER = """
import sys, types, pathlib
mod = types.ModuleType('huggingface_hub')
def snapshot_download(**kw):
    root = pathlib.Path(kw['local_dir'])
    (root / 'pixel-art-xl.safetensors').write_bytes(b'weights')
mod.snapshot_download = snapshot_download
sys.modules['huggingface_hub'] = mod
from warlock.pipelines import fetch_worker
raise SystemExit(fetch_worker.main())
"""


def test_a_download_runs_end_to_end_through_a_real_child(tmp_path, monkeypatch):
    """The host half against a real subprocess: spawn, stdin, progress, result.

    The child's ``snapshot_download`` is a stub that writes a file -- no test in
    this project may make a network call, and the machinery either side of that
    one function is the whole of what this package added.
    """
    from warlock.service import downloads

    cfg = _config(tmp_path)
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 5000.0)
    monkeypatch.setattr(downloads, "worker_argv", lambda: [sys.executable, "-c", _STUB_WORKER])
    seen: list[tuple[float, str]] = []
    downloads.download(
        _Svc(cfg),
        ["lora:pixelxl"],
        on_progress=lambda percent, label: seen.append((percent, label)),
    )
    assert (cfg.t2i_model_root / "loras" / "pixel-art-xl.safetensors").read_bytes() == b"weights"
    assert seen and seen[-1] == (100.0, "")
    entry = fetch.find("lora:pixelxl")
    assert entry.is_present(cfg)


def test_a_real_child_in_no_publish_mode_stages_and_installs_nothing(tmp_path, monkeypatch):
    """The child half of the transaction, against a real subprocess (MDL-10).

    ``publish: False`` is what makes "install these four" one decision instead
    of four: the child stops with its tree downloaded, verified and marked, and
    the parent installs every tree together afterwards. The thing to pin is
    that it really does stop -- a child that published anyway would restore the
    old behaviour with the new code around it, and every parent-level test
    would still pass.
    """
    from warlock import publish

    # Imported here *and* the flag put back: importing ``fetch_worker`` sets
    # ``HF_HUB_OFFLINE=0``, which is the whole point of the module and correct
    # in the child process it is spawned as. In the pytest process it is the
    # one thing the offline invariant forbids -- huggingface_hub reads that
    # variable at import time -- and the guard two tests below catches it.
    _before = os.environ.get("HF_HUB_OFFLINE")
    from warlock.pipelines import fetch_worker

    if _before is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = _before

    cfg = _config(tmp_path)
    dest = cfg.t2i_model_root / "loras"
    spec = {
        "repo_id": "nerijs/pixel-art-xl",
        "dest": str(dest),
        "publish": False,
        "filenames": ["pixel-art-xl.safetensors"],
        "size_gib": 0.1,
    }
    monkeypatch.setattr(
        fetch_worker,
        "fetch_one",
        fetch_worker.fetch_one,  # named so the patch below is the only stub
    )

    import types

    stub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kw):
        root = Path(kw["local_dir"])
        (root / "pixel-art-xl.safetensors").write_bytes(b"weights")

    stub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    result = fetch_worker.fetch_one(spec)

    staging = Path(result["staged"])
    assert staging.is_dir(), "the staging tree is the product, not a temporary"
    assert (staging / "pixel-art-xl.safetensors").read_bytes() == b"weights"
    # Nothing published, and the completion marker says the tree is finished --
    # which the directory's mere existence could not.
    assert not (dest / "pixel-art-xl.safetensors").exists()
    marker = publish.read_json(staging / publish.PUBLISH_NAME)
    assert marker and marker["repo_id"] == "nerijs/pixel-art-xl"
    assert "pixel-art-xl.safetensors" in marker["files"]
    # And no manifest: that is written by whoever publishes, after the publish,
    # so it stays a completion marker rather than an intention.
    assert not (dest / fetch.MANIFEST_NAME).exists()


def test_the_publish_marker_is_not_itself_published(tmp_path):
    """It belongs to the staging tree. Published into a model directory it
    would be a stray file every presence probe has to learn to ignore -- the
    same reason ``.cache`` is removed before the move."""
    from warlock import publish

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "w.bin").write_bytes(b"w")
    (staging / publish.PUBLISH_NAME).write_text("{}", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()

    names = publish.move_into(staging, dest)

    assert names == ["w.bin"]
    assert not (dest / publish.PUBLISH_NAME).exists()


def test_a_child_that_fails_is_reported_in_its_own_words(tmp_path, monkeypatch):
    from warlock.service import downloads
    from warlock.service.errors import Invalid

    stub = _STUB_WORKER.replace(
        "(root / 'pixel-art-xl.safetensors').write_bytes(b'weights')",
        "raise OSError('the network is down')",
    )
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 5000.0)
    monkeypatch.setattr(downloads, "worker_argv", lambda: [sys.executable, "-c", stub])
    cfg = _config(tmp_path)
    with pytest.raises(Invalid) as caught:
        downloads.download(_Svc(cfg), ["lora:pixelxl"])
    assert "the network is down" in caught.value.message
    assert not (cfg.t2i_model_root / "loras").exists() or not list(
        (cfg.t2i_model_root / "loras").iterdir()
    )


def test_rows_carry_a_flag_rather_than_the_word_missing(tmp_path):
    from warlock.service import downloads

    rows = downloads.rows(_Svc(_config(tmp_path)))
    assert rows and all("present" in row for row in rows)
    # Nothing is on disk under a fresh root, and no label leans on a substring.
    assert not any(row["present"] for row in rows)
    assert not any("missing" in row["label"].lower() for row in rows)
    # The style LoRAs are in the list at all, which is what they were not
    # before: the pane's old missing-ness came from a substring main.py only
    # ever put into base-model labels.
    assert any(row["kind"] == "lora" for row in rows)


def test_the_child_clears_the_flag_only_in_its_own_environment(tmp_path):
    """The parent's HF_HUB_OFFLINE is untouched by running a fetch."""
    before = os.environ["HF_HUB_OFFLINE"]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import warlock, os; "
            "from warlock.pipelines import fetch_worker; "
            "print(os.environ['HF_HUB_OFFLINE'])",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "0", proc.stderr
    assert os.environ["HF_HUB_OFFLINE"] == before == "1"


def test_a_malformed_progress_line_does_not_abandon_the_download(tmp_path, monkeypatch):
    """The drain loop used to call float() on whatever ``percent`` held. A
    child printing a non-numeric percent -- or a bare JSON scalar, which
    huggingface_hub can put down the same stream -- would raise inside the
    loop, kill the child and report a download that was working as failed."""
    from warlock.service import downloads

    stub = _STUB_WORKER.replace(
        "from warlock.pipelines import fetch_worker",
        "print('5'); print('{\"percent\": \"wat\", \"label\": \"x\"}');"
        " print('[1, 2]'); sys.stdout.flush()\n"
        "from warlock.pipelines import fetch_worker",
    )
    cfg = _config(tmp_path)
    monkeypatch.setattr(fetch, "free_gib", lambda _path: 5000.0)
    monkeypatch.setattr(downloads, "worker_argv", lambda: [sys.executable, "-c", stub])
    seen: list[tuple[float, str]] = []
    downloads.download(
        _Svc(cfg),
        ["lora:pixelxl"],
        on_progress=lambda percent, label: seen.append((percent, label)),
    )
    assert (cfg.t2i_model_root / "loras" / "pixel-art-xl.safetensors").read_bytes() == b"weights"
    assert seen and seen[-1] == (100.0, "")


# --- taking one back out -----------------------------------------------------


def _entries(*row_keys: str) -> list[fetch.Entry]:
    chosen = []
    for row_key in row_keys:
        entry = fetch.find(row_key)
        assert entry is not None, row_key
        chosen.append(entry)
    return chosen


def test_one_shared_recipe_frees_only_its_own_adapter(tmp_path):
    """Five recipes over one 7 GiB directory. Uninstalling ``sdxl`` alone must
    not take the checkpoint the other four are standing on -- and the freed
    figure has to say 0.8, not 7, or the label is a lie."""
    cfg = _config(tmp_path)
    removal = fetch.removal_plan(cfg, _entries("base:sdxl"))
    assert removal.blocked == ()
    assert [p.name for p in removal.paths] == ["Hyper-SDXL-4steps-lora.safetensors"]
    assert removal.freed_gib == pytest.approx(0.8)


def test_all_five_recipes_together_do_take_the_checkpoint(tmp_path):
    cfg = _config(tmp_path)
    removal = fetch.removal_plan(
        cfg,
        _entries(
            "base:sdxl", "base:sdxl_cfg", "base:sdxl_cfg_pag", "base:pixel",
            "base:lightning",
        ),
    )
    assert cfg.t2i_model_root / "sdxl-base-1.0" in removal.paths
    assert removal.freed_gib == pytest.approx(7.0 + 0.8 + 0.4 + 0.4)


def test_a_style_lora_is_one_file_and_never_the_flat_directory(tmp_path):
    """``loras/`` is shared by every family; only per-file claims go in it."""
    cfg = _config(tmp_path)
    removal = fetch.removal_plan(cfg, _entries("lora:pixelxl"))
    assert removal.paths == (cfg.t2i_model_root / "loras" / "pixel-art-xl.safetensors",)
    assert cfg.t2i_model_root / "loras" not in removal.paths
    assert removal.freed_gib == pytest.approx(0.2)


def test_the_ip_adapters_two_records_are_one_destination_counted_once(tmp_path):
    """Two Fetch records against one repository into one directory -- the same
    dedupe ``plan`` does, arrived at from the other side."""
    cfg = _config(tmp_path)
    removal = fetch.removal_plan(cfg, _entries("adapter:plus"))
    assert removal.paths == (cfg.t2i_model_root / "ip-adapter",)
    assert removal.freed_gib == pytest.approx(0.9 + 2.6)


def test_a_directory_the_env_var_points_at_is_never_deleted(tmp_path):
    """It is a directory the user pointed us at, not one we fetched."""
    elsewhere = tmp_path / "models" / "elsewhere"
    cfg = _config(tmp_path, turbo=elsewhere)
    removal = fetch.removal_plan(cfg, _entries("base:turbo"))
    assert removal.paths == ()
    assert removal.freed_gib == 0.0
    assert any("WARLOCK_T2I_DIR" in reason for reason in removal.blocked)


def test_nothing_outside_the_model_root_may_be_removed(tmp_path):
    cfg = _config(tmp_path, turbo=tmp_path / "somewhere-else")
    removal = fetch.removal_plan(cfg, _entries("base:turbo"))
    assert removal.paths == ()
    assert removal.blocked
    assert any("model root" in reason for reason in removal.blocked)


def test_planning_a_removal_writes_nothing(tmp_path):
    cfg = _config(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    for entry in fetch.entries():
        fetch.removal_plan(cfg, [entry])
        fetch.claims(cfg, entry)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_every_claim_is_something_present_would_have_looked_at(tmp_path):
    """The inverse-of-``present`` contract, checked the only way it can be:
    make every claim exist, and the row must read present."""
    cfg = _config(tmp_path)
    for entry in fetch.entries():
        for path in fetch.claims(cfg, entry):
            assert path.is_relative_to(cfg.t2i_model_root) or path.is_relative_to(
                cfg.trellis_models_dir
            ), entry.row_key
