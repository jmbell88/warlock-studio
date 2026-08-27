"""Regressions for the defects the 2026-08-24 release-readiness audit found.

One file rather than a hunk in each subsystem's suite, because the value here is
that every one of these is *the same class of mistake* -- a guard that exists
somewhere in the tree and was not applied at one particular site -- and reading
them together is what makes the pattern visible. Each test names the sibling
that already did it right.
"""

from __future__ import annotations

import io
import struct
import threading
import zipfile
import zlib

import pytest

from warlock import leases
from warlock.studio import zipguard

# --- the zip claimed-size ceiling was bypassable -------------------------------


def _lying_zip(declared: int, real: bytes) -> bytes:
    """An archive whose central directory declares ``declared`` bytes for a
    member that actually inflates to ``len(real)``.

    Hand-built, because ``zipfile`` cannot be made to write a dishonest
    directory -- which is the point: the four container doors summed
    ``info.file_size`` over exactly this field and an attacker writes it.
    """
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    blob = comp.compress(real) + comp.flush()
    out = io.BytesIO()
    name, crc = b"a.bin", zlib.crc32(real)
    lfh = struct.pack(
        "<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0, crc, len(blob), declared, len(name), 0
    )
    off = out.tell()
    out.write(lfh + name + blob)
    cd = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50, 20, 20, 0, 8, 0, 0, crc, len(blob), declared, len(name),
        0, 0, 0, 0, 0, off,
    )
    cdoff = out.tell()
    out.write(cd + name)
    out.write(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(cd) + len(name), cdoff, 0))
    return out.getvalue()


def test_a_member_that_lies_about_its_size_is_refused_before_it_is_held():
    """The bomb the ``file_size`` sum could not see.

    16 MiB of real payload behind a directory entry claiming 10 bytes. Stock
    ``zipfile`` discovers the lie by CRC, *after* accumulating the whole
    inflated stream; measured on the 512 MiB version, peak allocation was
    1,070 MiB against a ceiling nominally set at 1 GiB.
    """
    data = _lying_zip(10, b"\0" * (16 << 20))
    with zipguard.BoundedZip(io.BytesIO(data)) as zf:
        assert zf.infolist()[0].file_size == 10  # what the old ceiling summed
        with pytest.raises((ValueError, zipfile.BadZipFile)):
            zf.read("a.bin")


def test_an_honest_member_reads_back_byte_for_byte():
    """The guard is worthless if it costs the ordinary path anything."""
    payload = bytes(range(256)) * 4096
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.bin", payload)
        zf.writestr("empty.bin", b"")
    with zipguard.BoundedZip(io.BytesIO(buf.getvalue())) as zf:
        assert zf.read("big.bin") == payload
        assert zf.read("empty.bin") == b""
        assert zf.namelist() == ["big.bin", "empty.bin"]


def test_an_honest_member_past_the_ceiling_is_still_refused():
    """The absolute ceiling survives, for an archive that is merely enormous."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.bin", b"\0" * 4096)
    with zipguard.BoundedZip(io.BytesIO(buf.getvalue())) as zf:
        zf.ceiling = 100
        with pytest.raises(ValueError, match="past the 100"):
            zf.read("big.bin")


def test_every_container_door_reads_through_the_bounded_reader():
    """The rule that keeps the four doors from drifting apart again.

    A scan rather than four behavioural tests, for the reason ``winjob`` is
    scanned for: what matters is that no *reading* door opens a plain
    ``zipfile.ZipFile``, and a door added next year is exactly the one no
    behavioural test would cover.
    """
    from pathlib import Path

    doors = [
        Path("src/warlock/studio/inker/ora.py"),
        Path("src/warlock/studio/clay/serialize.py"),
        Path("src/warlock/studio/packwright/wpack.py"),
        Path("src/warlock/studio/plotter/wmap.py"),
    ]
    for door in doors:
        source = door.read_text(encoding="utf-8")
        assert "zipguard.BoundedZip(" in source, f"{door.name} opens no bounded archive"
        for line in source.splitlines():
            stripped = line.strip()
            if "zipfile.ZipFile(" in stripped:
                # Writers only. A writer produces the archive, so there is
                # nothing untrusted for a bound to protect.
                assert '"w"' in stripped, f"{door.name} reads through a plain ZipFile"


# --- the maintainer deadlocked against its own shared lease --------------------


def test_the_maintainer_may_run_its_own_model_operation():
    """``queue._unload_under_lease`` under ``WARLOCK_T2I_IN_PROCESS=1``.

    It holds ``maintain`` and then calls ``pipe.unload()``, which takes
    ``use``. ``use``'s ``wait_for`` has no timeout, so before the fix this hung
    the shutdown forever -- one thread waiting on a condition only that same
    thread could clear.
    """
    lease = leases.ModelLease()
    done = threading.Event()

    def shutdown() -> None:
        with lease.maintain(timeout=5.0), lease.use():  # what pipe.unload() does
            pass
        done.set()

    thread = threading.Thread(target=shutdown, daemon=True)
    thread.start()
    thread.join(timeout=10.0)
    assert done.is_set(), "the maintainer deadlocked against its own shared lease"


def test_another_thread_still_waits_for_the_maintainer():
    """The exemption is for the maintaining *thread*, not for everybody.

    Without this the fix would have turned the exclusive lease into no lease at
    all, which is a far worse bug than the one it repaired.
    """
    lease = leases.ModelLease()
    holding = threading.Event()
    entered = threading.Event()
    release = threading.Event()

    def maintainer() -> None:
        with lease.maintain(timeout=5.0):
            holding.set()
            release.wait(timeout=5.0)

    def user() -> None:
        with lease.use():
            entered.set()

    m = threading.Thread(target=maintainer, daemon=True)
    m.start()
    assert holding.wait(timeout=5.0)
    u = threading.Thread(target=user, daemon=True)
    u.start()
    assert not entered.wait(timeout=0.5), "a foreign thread walked past the maintainer"
    release.set()
    u.join(timeout=5.0)
    assert entered.is_set()
    m.join(timeout=5.0)


# --- .aseprite inflated without a bound ----------------------------------------


def test_a_cel_that_inflates_past_its_rectangle_is_refused():
    """``asein`` was the one parser whose siblings all bound this and it did not.

    ``ora.py`` sums declared sizes against a ceiling and ``tmx.py`` uses a
    ``decompressobj`` with the layer's own arithmetic as the bound; ``asein``
    called bare ``zlib.decompress`` three times and checked the size in
    ``_decode``, which runs after the allocation it would have prevented.
    """
    from warlock.studio.inker import asein

    # A 4x4 RGBA cel is 64 bytes. This stream inflates to 8 MiB.
    payload = b"\0" * (8 << 20)
    with pytest.raises(ValueError, match="unpacks past the 64 bytes"):
        asein._inflate(zlib.compress(payload), 4 * 4 * 4, "a cel on layer 0")


def test_a_chunk_declaring_more_than_the_ceiling_is_refused_outright():
    """The second line of defence: the rectangle itself is two u16s, so an
    honest-looking 65535x65535 RGBA cel still asks for 17 GiB."""
    from warlock.studio.inker import asein

    with pytest.raises(ValueError, match="past the"):
        asein._inflate(zlib.compress(b"x"), 65535 * 65535 * 4, "a cel on layer 0")


def test_an_honest_cel_still_inflates():
    from warlock.studio.inker import asein

    payload = bytes(range(64))
    assert asein._inflate(zlib.compress(payload), 64, "a cel on layer 0") == payload


def test_an_infinite_maps_chunk_cannot_outrun_the_extent_cap():
    """``tmx`` capped a *fixed* map's dimensions through ``MapDoc.__init__``,
    but an infinite map's real dimensions arrive as ``<chunk>`` attributes,
    which went straight into the decode bound uncapped."""
    from warlock.studio.plotter import tmx
    from warlock.studio.plotter.tilemap import MAX_DIMENSION

    assert tmx._chunk_side(16, "width") == 16
    assert tmx._chunk_side(MAX_DIMENSION, "width") == MAX_DIMENSION
    with pytest.raises(ValueError, match="past the"):
        tmx._chunk_side(MAX_DIMENSION + 1, "width")
    with pytest.raises(ValueError, match="at least 1"):
        tmx._chunk_side(0, "height")


# --- a persisted window size could brick the window ----------------------------


@pytest.mark.parametrize(
    "stored",
    [None, "junk", [], [0, 0], [-5, -5], {"w": 1}, [1, 1], ["a", "b"], [None, None]],
)
def test_a_junk_window_size_never_reaches_set_mode(stored):
    """``_ui_scale`` right above it already stated the rule this skipped.

    The consequence was not cosmetic: if ``set_mode`` raises, ``run``'s handler
    reports the crash but never rewrites the key, so a non-developer gets a
    window that refuses to open on every launch with no way back in.
    """
    from warlock.studio.main import MIN_SIZE, _window_size

    size = _window_size(stored, override=None, first_run_scale=1.0, desktop=(1920, 1080))
    assert size[0] >= MIN_SIZE[0] and size[1] >= MIN_SIZE[1]


def test_the_default_window_is_clamped_to_the_desktop():
    """1600x950 at the 125% Windows recommends for many 1080p laptops asks for
    2000x1187, which does not fit the panel it was scaled for."""
    from warlock.studio.main import _window_size

    assert _window_size(None, override=None, first_run_scale=1.25, desktop=(1920, 1080)) == (
        1920,
        1080,
    )
    # And a *stored* size too: the display a window closed on may not be the
    # one it reopens on.
    assert _window_size(
        [3000, 2000], override=None, first_run_scale=1.0, desktop=(1366, 768)
    ) == (1366, 768)


def test_an_explicit_override_is_never_clamped():
    """The screenshot harness asks for an exact framebuffer, and a clamp there
    would silently produce shots of a size nothing asked for."""
    from warlock.studio.main import _window_size

    assert _window_size(
        [1700, 1000], override=(640, 480), first_run_scale=1.0, desktop=(1920, 1080)
    ) == (640, 480)


def test_a_display_that_reports_nothing_falls_back_to_the_default():
    from warlock.studio.main import DEFAULT_SIZE, _window_size

    assert _window_size(None, override=None, first_run_scale=1.0, desktop=None) == DEFAULT_SIZE


# --- writers that truncated the user's only copy -------------------------------


def test_saving_a_png_in_place_is_staged_and_replaced(tmp_path, monkeypatch):
    """``WRITABLE_SUFFIXES`` deliberately allows saving back over an opened
    ``.png``, and that branch was a bare ``write_bytes``. Every other document
    writer in the app stages to a temp and replaces.

    The helper has since moved out of ``inker_mode`` and into ``studio.atomic``
    -- see ``tests/test_atomic_writes.py`` for why it had to stop being one
    module's private idiom -- but the property is this one and stays here."""
    from warlock.studio import atomic

    target = tmp_path / "drawing.png"
    target.write_bytes(b"the user's only copy")

    boom = RuntimeError("disk full, encoder half way through")

    def explode(self, data):
        raise boom

    monkeypatch.setattr("pathlib.Path.write_bytes", explode)
    with pytest.raises(RuntimeError):
        atomic.write_bytes(target, b"new content")

    # The point of the whole exercise: the original survived the failed write.
    assert target.read_bytes() == b"the user's only copy"
    monkeypatch.undo()
    assert not list(tmp_path.glob(".*.tmp")), "a staging file was left behind"


def test_a_successful_atomic_write_replaces_and_leaves_no_temp(tmp_path):
    from warlock.studio import atomic

    target = tmp_path / "drawing.png"
    target.write_bytes(b"old")
    atomic.write_bytes(target, b"new")
    assert target.read_bytes() == b"new"
    assert [p.name for p in tmp_path.iterdir()] == ["drawing.png"]


def test_write_ora_leaves_no_staging_file_when_the_encode_fails(tmp_path, monkeypatch):
    """Not data loss -- ``replace`` only runs on success -- but ``plotter_io``,
    ``packwright_io`` and ``journal`` all unlink theirs and this one did not."""
    from warlock.studio.inker import ora

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("encode failed")

    target = tmp_path / "doc.ora"
    with pytest.raises(RuntimeError):
        ora.write_ora(Boom(), target)
    assert list(tmp_path.iterdir()) == [], "write_ora left its .tmp behind"


# --- the app was silent about what the manual is candid about ------------------


def test_every_mode_says_what_it_is_for():
    """Six of eleven rail labels are invented names, and the rail is the
    primary navigation. ``rail._item`` already took a ``tooltip`` no call site
    passed."""
    from warlock.studio import modes

    assert set(modes.PURPOSE) == set(modes.KEYS)
    for key, text in modes.PURPOSE.items():
        assert text.strip() and text[0].isupper() and text.endswith("."), key
        # A tooltip repeating the label is the noise ``_item`` suppresses.
        labels = {k: label.lower() for k, label, _icon in modes.MODES}
        assert text.strip().lower() != labels[key]


def test_troupe_is_the_mode_marked_experimental():
    """It is code-complete and a user really can get a rendered sheet -- but
    three of its own phases are unstarted and its keyframes are provisional.
    ``docs/manual/11`` is candid about that and the app was not.

    **Sirens stayed on the list**, and what kept it there was writing its
    manual chapter: four of a cell's five columns are drawn and take no
    keyboard input, so every effect the synth implements is unreachable from
    the UI. Asserting the exact dict rather than membership is the point -- a
    mode joining or leaving this list is a claim about the app that should have
    to be written down twice.
    """
    from warlock.studio import modes

    assert modes.MATURITY == {"troupe": "Experimental", "sirens": "Experimental"}
    assert set(modes.MATURITY_NOTE) == set(modes.MATURITY)
    assert set(modes.MATURITY) <= set(modes.KEYS)


def test_the_rail_passes_the_purpose_and_the_badge_through():
    """A pin on the wiring, because the data being right is only half of it --
    the call site passed no tooltip at all before."""
    import inspect

    from warlock.studio import rail

    source = inspect.getsource(rail)
    assert "modes.PURPOSE.get(key" in source
    assert "badge=modes.MATURITY.get(key" in source
    assert "modes.MATURITY_NOTE.get(key" in source


# --- doctor was amber on a host that cannot run the headline feature -----------


def test_no_cuda_device_is_fatal_once_it_has_actually_been_looked_for():
    """``vram.plan`` reports "admission control is off", and an amber row
    saying a budget is not being enforced reads as good news. There is no CPU
    fallback, so the real refusal used to arrive as
    ``RuntimeError("trellis-server exited during startup")`` two minutes into
    the first reconstruction."""
    from warlock import doctor
    from warlock.config import get_config

    config = get_config()
    check = doctor._vram_check(config, probe=True)
    if "no CUDA device" in check.detail:
        assert check.fatal and not check.ok
        assert "no CPU fallback" in check.detail


def test_a_deferred_probe_is_never_reported_as_a_missing_card():
    """"We have not looked yet" is not "there is no card".

    Conflating them put a red row on every cold start of a perfectly good
    machine: startup passes ``probe_slow=False`` so the torch import is
    deferred, and ``device_memory`` then has nothing to read.
    """
    import sys

    from warlock import doctor
    from warlock.config import get_config

    if "torch" in sys.modules:
        pytest.skip("torch already imported; the deferred path cannot be observed")
    check = doctor._vram_check(get_config(), probe=False)
    assert check.ok and not check.fatal
    assert "still checking" in check.detail


# --- the licence of the weights was never disclosed ----------------------------


def test_every_base_model_declares_its_licence():
    """A tool whose purpose is producing assets people sell shipped two
    checkpoints that restrict exactly that, and said so nowhere."""
    from warlock import models

    for key, spec in models.BASE_MODELS.items():
        assert spec.license, f"{key} declares no licence"


def test_the_two_restricted_models_are_marked_as_such():
    """SDXL-Turbo is Stability's non-commercial research licence -- and was
    promoted in the README as "the fast option". Playground v2.5 permits
    commercial use only below 1M monthly users and requires an attribution
    string this project owes."""
    from warlock import models

    turbo = models.BASE_MODELS["turbo"]
    assert not turbo.commercial
    assert "Non-Commercial" in turbo.license
    assert turbo.license_note

    playground = models.BASE_MODELS["playground"]
    assert playground.commercial and playground.license_note
    assert "1M monthly active users" in playground.license_note

    # And nothing else claims a restriction it does not have.
    restricted = {k for k, m in models.BASE_MODELS.items() if not m.commercial}
    assert restricted == {"turbo"}


def test_the_licence_reaches_the_row_the_download_button_is_on():
    """Metadata nobody can see is not disclosure. The models table is where the
    ~7 GB fetch is agreed to."""
    from warlock.service import downloads
    from warlock.studio.panes import app_settings

    row = {"license": "Some Non-Commercial Licence", "commercial": False,
           "license_note": "Ask first."}
    note = app_settings.licence_note(row)
    assert "may NOT be used commercially" in note and "Ask first." in note
    assert app_settings.licence_note({}) == ""
    assert "commercial" not in app_settings.licence_note(
        {"license": "MIT", "commercial": True}
    )
    # And the service actually puts it there.
    import inspect

    assert '"license"' in inspect.getsource(downloads.rows)


def test_the_notices_file_covers_every_bundled_binary():
    """MIT requires the notice to travel with the binary. ``build.ps1`` copied
    eleven of them with no licence text at all."""
    import json
    from pathlib import Path

    notices = Path("THIRD-PARTY-NOTICES.md").read_text(encoding="utf-8")
    manifest = json.loads(Path("installer/runtime-manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        name = Path(entry["path"]).name
        assert name in notices, f"{name} is bundled but not in THIRD-PARTY-NOTICES.md"


def test_the_project_declares_a_licence():
    """Without one, default copyright applies: nobody may legally use, copy,
    modify or distribute the work, and nobody can contribute to it."""
    import tomllib
    from pathlib import Path

    text = Path("LICENSE").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in text and "Version 3, 29 June 2007" in text
    # The sections that make it the licence rather than a summary of one.
    for section in ("TERMS AND CONDITIONS", "15. Disclaimer of Warranty",
                    "16. Limitation of Liability"):
        assert section in text, section

    meta = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert meta["license"] == "GPL-3.0-or-later"
    assert "LICENSE" in meta["license-files"]
