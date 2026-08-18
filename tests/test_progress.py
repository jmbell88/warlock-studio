"""Parser + progress-bus tests. Replays a real captured trellis run; no GPU needed."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from warlock.progress import (
    PHASES_IMAGE,
    PHASES_TEXT,
    SUBSTEPS,
    TRELLIS_STAGES,
    ProgressBus,
    TrellisProgressParser,
    pump,
)

FIXTURE = Path(__file__).parent / "fixtures" / "trellis_1024.log"


def read_fixture() -> list[str]:
    raw = FIXTURE.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    return [ln for ln in text.split("\n") if ln]


def replay(lines, *, kind="image", job_id="j1") -> list[float]:
    """Feed lines through parser -> bus, returning the percent after each line."""
    bus = ProgressBus()
    bus.begin(job_id, kind)
    parser = TrellisProgressParser(
        lambda phase, label, inner, inner_next, nominal, fields: bus.update(
            job_id, phase=phase, label=label, inner=inner,
            inner_next=inner_next, nominal=nominal, **fields,
        )
    )
    out = []
    for ln in lines:
        parser.feed(ln)
        snap = bus.snapshot(job_id)
        out.append(snap["percent"] if snap else 0.0)
    return out


# --- weight tables ---


def test_phase_weights_cover_the_whole_bar():
    for table in (PHASES_TEXT, PHASES_IMAGE):
        spans = sorted(table.values())
        assert spans[0][0] == 0.0
        assert spans[-1][1] == 1.0
        for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
            assert end == pytest.approx(start), "phases must be contiguous"


def test_every_phase_the_worker_emits_is_declared_for_generate_jobs():
    """update() falls back to (0.0, 1.0) for an unknown phase, which the
    module's own docstring warns 'would drag the bar back to zero'. So every
    phase queue.Worker emits for a text/image job must be declared in both
    tables -- including the zero-width tail phases."""
    for table in (PHASES_TEXT, PHASES_IMAGE):
        for phase in ("trellis", "optimize", "scale", "audit"):
            assert phase in table, phase


def test_trellis_stage_weights_sum_to_one():
    total = sum(w for _, w in TRELLIS_STAGES.values())
    assert total == pytest.approx(1.0)


def test_substep_fractions_are_ordered():
    for key, steps in SUBSTEPS.items():
        fracs = [s.frac for s in steps]
        assert fracs == sorted(fracs), f"{key} milestones out of order"
        assert all(0.0 <= f <= 1.0 for f in fracs)


# --- fixture replay ---


def test_replay_is_monotonic_and_completes():
    pcts = replay(read_fixture())
    assert pcts == sorted(pcts), "progress went backwards"
    assert pcts[-1] == pytest.approx(100.0)


def test_denominator_switch_does_not_regress():
    """Stage headers go [1/6]..[3/6] then [4/7]..[7/7]; the bar must not jump back."""
    lines = read_fixture()
    pcts = replay(lines)
    at = {}
    for ln, p in zip(lines, pcts, strict=True):
        if ln.startswith("[") and "]" in ln[:6]:
            at[ln[: ln.index("]") + 1]] = p
    assert at["[3/6]"] < at["[4/7]"] < at["[5/7]"] < at["[6/7]"] < at["[7/7]"]


def test_cascade_stage_has_two_flow_runs_without_rewinding():
    """[4/7] with 'cascade' contains an LR and an HR flow run, both 0..12."""
    lines = read_fixture()
    pcts = replay(lines)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("[4/7]"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("[5/7]"))
    seg = lines[start:end]
    starts = sum(1 for ln in seg if re.search(r"\s0/12\s", ln))
    assert starts == 2, "fixture should have two flow runs"
    window = pcts[start:end]
    assert window == sorted(window)


def test_single_run_stage_four_completes_without_cascade():
    """The non-cascade [4/7] variant has one flow run; it must still fill the stage."""
    two = replay([
        "[trellis-server] generate: 1-byte image, seed 42, res 512",
        "[4/7] shape SLAT flow (LR 512 -> upsample -> HR 1024 cascade, max_tok=49152)",
        *[f"      [flow] [####] {i}/12    0.1s  ~1s left" for i in range(1, 13)],
        "      [flow] 12 steps, 20 forwards, 1.2s",
    ])
    one = replay([
        "[trellis-server] generate: 1-byte image, seed 42, res 512",
        "[4/7] shape SLAT flow (512)",
        *[f"      [flow] [####] {i}/12    0.1s  ~1s left" for i in range(1, 13)],
        "      [flow] 12 steps, 20 forwards, 1.2s",
    ])
    # One run fills the whole stage; the cascade's first run only fills half.
    assert one[-1] > two[-1]


def test_stage_seven_advances_in_many_increments():
    """Guards the anti-freeze property: stage 7 is ~64% of wall time and would
    otherwise be a single frozen step."""
    lines = read_fixture()
    pcts = replay(lines)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("[7/7]"))
    distinct = sorted({round(p, 3) for p in pcts[start:]})
    assert len(distinct) >= 8, f"stage 7 only moved {len(distinct)} times"


def test_flow_steps_always_advance_the_bar(monkeypatch):
    """Regression: entering a stage used to creep toward the first milestone. In
    stage 6 that milestone sits at 0.98, so creep saturated within a second and
    every one of the 12 denoise lines then fell below it -- the bar sat frozen
    for the whole stage while the detail text counted up."""
    import warlock.progress as mod

    t = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: t[0])
    bus = ProgressBus()
    bus.begin("j", "image")
    parser = TrellisProgressParser(
        lambda p, lbl, i, n, nom, f: bus.update(
            "j", phase=p, label=lbl, inner=i, inner_next=n, nominal=nom, **f
        )
    )
    parser.feed("[trellis-server] generate: 1-byte image, seed 42, res 1024")
    parser.feed("[6/7] texture SLAT flow + PBR decode")
    t[0] += 1.5  # let entry creep settle before the first flow line
    seen = []
    for i in range(1, 13):
        parser.feed(f"      [flow] [####] {i}/12    0.3s  ~2s left")
        t[0] += 0.3
        seen.append(bus.snapshot("j")["percent"])
    assert len(set(seen)) >= 10, f"bar froze during the flow run: {seen}"
    assert seen == sorted(seen)


def test_cold_model_load_keeps_moving(monkeypatch):
    """Stage 1 on a cold server absorbs the ~8 GB CUDA load (~14 s) and prints
    nothing until stage 2. It is the first thing a user sees, so a frozen bar
    there is the worst case."""
    import warlock.progress as mod

    t = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: t[0])
    bus = ProgressBus()
    bus.begin("j", "image")
    parser = TrellisProgressParser(
        lambda p, lbl, i, n, nom, f: bus.update(
            "j", phase=p, label=lbl, inner=i, inner_next=n, nominal=nom, **f
        )
    )
    parser.feed("[trellis-server] generate: 1-byte image, seed 42, res 1024")
    parser.feed("[1/6] preprocess in.png (BiRefNet bg removal, 1024 cascade)")
    seen = []
    for _ in range(28):  # 14 s at the UI's ~0.5 s poll cadence
        t[0] += 0.5
        seen.append(bus.snapshot("j")["percent"])
    assert seen == sorted(seen)
    # No 5-second window may pass without visible movement.
    for i in range(len(seen) - 10):
        assert seen[i + 10] > seen[i] + 0.05, f"stalled at sample {i}: {seen[i]}"


def test_silence_after_a_flow_run_still_creeps(monkeypatch):
    """Stage 6 spends ~3.5 s in a silent PBR decode once its flow run ends."""
    import warlock.progress as mod

    t = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: t[0])
    bus = ProgressBus()
    bus.begin("j", "image")
    parser = TrellisProgressParser(
        lambda p, lbl, i, n, nom, f: bus.update(
            "j", phase=p, label=lbl, inner=i, inner_next=n, nominal=nom, **f
        )
    )
    parser.feed("[trellis-server] generate: 1-byte image, seed 42, res 1024")
    parser.feed("[6/7] texture SLAT flow + PBR decode")
    for i in range(1, 13):
        parser.feed(f"      [flow] [####] {i}/12    0.3s  ~2s left")
    parser.feed("      [flow] 12 steps, 12 forwards, 3.5s")
    before = bus.snapshot("j")["percent"]
    t[0] += 3.0
    assert bus.snapshot("j")["percent"] > before


def test_second_job_on_a_warm_server_starts_over():
    """A warm server serves many jobs from one stdout stream. Each job calls
    begin(), which is what resets the bar -- the parser only resets its own
    stage bookkeeping on the 'generate:' header."""
    lines = read_fixture()
    bus = ProgressBus()
    current = {"id": "job1"}
    parser = TrellisProgressParser(
        lambda p, lbl, i, n, nom, f: bus.update(
            current["id"], phase=p, label=lbl, inner=i, inner_next=n, nominal=nom, **f
        )
    )
    bus.begin("job1", "image")
    for ln in lines:
        parser.feed(ln)
    assert bus.snapshot("job1")["percent"] == pytest.approx(100.0)

    bus.end("job1")
    current["id"] = "job2"
    bus.begin("job2", "image")
    seen = []
    for ln in lines:
        parser.feed(ln)
        snap = bus.snapshot("job2")
        seen.append(snap["percent"] if snap else 0.0)
    assert seen[-1] == pytest.approx(100.0)
    assert min(seen) < 50.0, "second job did not start over"
    assert seen == sorted(seen)


def test_parser_reset_clears_stage_state():
    """Without the reset on 'generate:', stage 7 of the previous job would still
    be current and the next job's early lines would be misattributed."""
    bus = ProgressBus()
    parser = TrellisProgressParser(
        lambda p, lbl, i, n, nom, f: bus.update(
            "j", phase=p, label=lbl, inner=i, inner_next=n, nominal=nom, **f
        )
    )
    bus.begin("j", "image")
    parser.feed("[7/7] write out.glb")
    parser.feed("  weld: V 10->9")
    bus.end("j")
    bus.begin("j", "image")
    parser.feed("[trellis-server] generate: 1-byte image, seed 42, res 1024")
    parser.feed("  weld: V 10->9")  # stale stage-7 milestone, must not apply
    assert bus.snapshot("j")["percent"] < 5.0


def test_unknown_lines_are_ignored():
    before = replay([
        "[trellis-server] generate: 1-byte image, seed 42, res 1024",
        "[3/6] sparse-structure flow + decode",
        "      [flow] [####]  6/12    1.0s  ~3s left",
    ])
    after = replay([
        "[trellis-server] generate: 1-byte image, seed 42, res 1024",
        "[3/6] sparse-structure flow + decode",
        "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 32606 MiB)",
        "      [stats] cond_512 (DINOv3@512) n=1053696 mean=-0.0000",
        "      [flow] [####]  6/12    1.0s  ~3s left",
    ])
    assert before[-1] == pytest.approx(after[-1])


def test_generate_failure_is_surfaced():
    bus = ProgressBus()
    bus.begin("j1", "image")
    parser = TrellisProgressParser(
        lambda p, lbl, i, n, nom, f: bus.update(
            "j1", phase=p, label=lbl, inner=i, inner_next=n, nominal=nom, **f
        )
    )
    parser.feed("[trellis-server] generate failed: out of memory")
    assert bus.snapshot("j1")["error"] == "out of memory"


def test_eta_and_step_are_reported():
    bus = ProgressBus()
    bus.begin("j1", "image")
    parser = TrellisProgressParser(
        lambda p, lbl, i, n, nom, f: bus.update(
            "j1", phase=p, label=lbl, inner=i, inner_next=n, nominal=nom, **f
        )
    )
    parser.feed("[trellis-server] generate: 1-byte image, seed 42, res 1024")
    parser.feed("[3/6] sparse-structure flow + decode")
    parser.feed("      [flow] [#####...............]  3/12    1.5s  ~5s left    ")
    snap = bus.snapshot("j1")
    assert snap["step"] == 3
    assert snap["step_total"] == 12
    assert snap["stage_eta"] == 5.0
    assert snap["label"] == "Sparse structure"


# --- text jobs ---


def test_text_job_hands_off_to_trellis_at_twenty_percent():
    lines = read_fixture()
    pcts = replay(lines, kind="text")
    assert pcts[-1] == pytest.approx(100.0)
    # Trellis owns 0.20..1.00 for text jobs, so its first real update is at ~20.
    # (The leading server banner emits nothing, hence the first *nonzero* value.)
    assert next(p for p in pcts if p > 0) >= 20.0


# --- bus semantics ---


def test_snapshot_never_leaks_across_jobs():
    bus = ProgressBus()
    bus.begin("a", "image")
    bus.update("a", phase="trellis", label="x", inner=0.5)
    assert bus.snapshot("a") is not None
    assert bus.snapshot("b") is None
    assert bus.snapshot() is not None  # no id filter -> whatever is current
    bus.end("a")
    assert bus.snapshot() is None


def test_end_from_a_stale_job_does_not_clear_the_current_one():
    bus = ProgressBus()
    bus.begin("a", "image")
    bus.end("b")
    assert bus.snapshot("a") is not None


def test_update_never_regresses():
    bus = ProgressBus()
    bus.begin("a", "image")
    bus.update("a", phase="trellis", label="x", inner=0.8)
    bus.update("a", phase="trellis", label="x", inner=0.2)
    assert bus.snapshot("a")["percent"] >= 80.0


def test_creep_moves_the_bar_without_new_lines(monkeypatch):
    """A silent stage must still visibly advance, but never past the next milestone."""
    import warlock.progress as mod

    t = [1000.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: t[0])
    bus = ProgressBus()
    bus.begin("a", "image")
    bus.update("a", phase="trellis", label="x", inner=0.10, inner_next=0.50, nominal=10.0)
    first = bus.snapshot("a")["percent"]
    t[0] += 5.0
    mid = bus.snapshot("a")["percent"]
    t[0] += 300.0
    late = bus.snapshot("a")["percent"]
    assert first == pytest.approx(10.0)
    assert mid > first
    assert mid < 50.0
    assert late > mid
    assert late <= 50.0, "creep must never pass the next milestone"


def test_percent_is_capped_at_one_hundred():
    bus = ProgressBus()
    bus.begin("a", "image")
    bus.update("a", phase="trellis", label="x", inner=5.0, inner_next=9.0)
    assert bus.snapshot("a")["percent"] == 100.0


# --- pump ---


def test_pump_splits_crlf_and_mirrors_bytes_exactly():
    data = b"one\r\ntwo\r\nthree\r\n"
    seen, mirrored = [], bytearray()
    pump(io.BytesIO(data), mirrored.extend, seen.append)
    assert seen == ["one", "two", "three"]
    assert bytes(mirrored) == data


def test_pump_handles_bare_cr_and_lf():
    seen = []
    pump(io.BytesIO(b"a\rb\nc\r\nd"), None, seen.append)
    assert seen == ["a", "b", "c", "d"]


def test_pump_reassembles_lines_split_across_reads():
    class Chunked(io.RawIOBase):
        def __init__(self, chunks):
            self._chunks = list(chunks)

        def read(self, _n=-1):
            return self._chunks.pop(0) if self._chunks else b""

    seen = []
    pump(Chunked([b"hel", b"lo\r", b"\nwor", b"ld\r\n"]), None, seen.append)
    assert seen == ["hello", "world"]


def test_pump_keeps_draining_after_a_parser_error():
    """If the drain stopped, the child would block on a full pipe and the job hangs."""
    seen = []

    def flaky(line):
        if line == "two":
            raise ValueError("boom")
        seen.append(line)

    pump(io.BytesIO(b"one\r\ntwo\r\nthree\r\n"), None, flaky)
    assert seen == ["one", "three"]


def test_pump_skips_empty_lines():
    seen = []
    pump(io.BytesIO(b"a\r\n\r\n\r\nb\r\n"), None, seen.append)
    assert seen == ["a", "b"]


# --- sprite synthesis --------------------------------------------------------


def test_the_sprite_phases_cover_the_whole_bar_contiguously():
    from warlock.progress import PHASES_SPRITE

    spans = sorted(PHASES_SPRITE.values())
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 1.0
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end == pytest.approx(start), "phases must be contiguous"


def test_every_phase_the_sprite_worker_emits_is_declared():
    """``update()`` falls back to (0.0, 1.0) for an unknown phase, which would
    drag the bar back to zero twice per job -- which is exactly why
    ``_sprite_synthesis`` does not reuse ``_t2i_state``."""
    from warlock.progress import phases_for

    table = phases_for("sprite_synthesis")
    for phase in ("condition", "generate_a", "assemble_a", "generate_b", "assemble_b"):
        assert phase in table, phase
    assert "t2i_load" not in table and "t2i_sample" not in table


def test_the_worker_never_hands_the_sprite_bar_a_t2i_phase():
    """Asserted against the source, because the mistake is an easy one to make
    by copying ``_pixel_sheet``: passing ``self._t2i_state`` as ``on_state``."""
    import inspect

    from warlock.queue import Worker

    source = inspect.getsource(Worker._sprite_synthesis)
    # Comments stripped, because the method *explains* the trap in one -- and a
    # scan test tripped by its own prose is a scan test nobody keeps.
    code = "\n".join(
        line for line in source.split("\n") if not line.lstrip().startswith("#")
    )
    assert "_t2i_state" not in code


# --- pixel sheets and re-textures --------------------------------------------


@pytest.mark.parametrize("kind", ["pixel_sheet", "retexture", "tile_sheet"])
def test_the_multi_pass_kinds_have_their_own_contiguous_tables(kind):
    """CON-02: none of these kinds had a table, and ``phases_for`` falls back
    to ``PHASES_IMAGE`` -- whose only real phase is ``trellis``.

    Named for what the three have in common rather than for how many there
    are: this said "the two img2img kinds" while carrying three parameters, and
    ``tile_sheet`` is txt2img anyway. What actually binds them is that none of
    them is the plain text path -- each has a tail the fallback knows nothing
    about, which is the shape that trips it. ``tile_sheet`` is the sharpest
    case: with a single generation there is nothing to average the fallback's
    errors out, so an unregistered table finishes the bar before the reduction
    has started.

    An unknown phase maps onto the *whole* bar, so the last sampling step of the
    **first** band (or view) emitted 100%, and the never-regress creep then
    pinned it there for the rest of a multi-minute job. The bar said finished
    while five more views were still to render.
    """
    from warlock.progress import PHASES_IMAGE, phases_for

    table = phases_for(kind)
    assert table is not PHASES_IMAGE, f"{kind} still falls back to the image table"
    spans = sorted(table.values())
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 1.0
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end == pytest.approx(start), "phases must be contiguous"


@pytest.mark.parametrize(
    "kind,emitted",
    [
        ("pixel_sheet", ("restyle", "t2i_load", "t2i_sample", "quantize")),
        (
            "retexture",
            ("views", "restyle", "t2i_load", "t2i_sample", "project", "assemble"),
        ),
        ("tile_sheet", ("guide", "t2i_load", "t2i_sample", "slice", "quantize")),
    ],
)
def test_every_phase_these_kinds_emit_is_declared(kind, emitted):
    """The names are the ones ``_q_sprite`` actually passes to
    ``progress.update`` -- including the two the shared ``_t2i_state`` callback
    emits, which is how these kinds acquired the fallback in the first place."""
    from warlock.progress import phases_for

    table = phases_for(kind)
    for phase in emitted:
        assert phase in table, phase


@pytest.mark.parametrize("kind", ["pixel_sheet", "retexture", "tile_sheet"])
def test_the_first_sampling_pass_does_not_reach_the_end_of_the_bar(kind):
    """The failure in its own terms: finishing the first pass's last step must
    leave room for the passes after it."""
    from warlock.progress import ProgressBus

    bus = ProgressBus()
    bus.begin("job", kind)
    # The last step of the first pass: ``inner`` is 1.0 *within* the phase, so
    # this is as far as one sampling pass can possibly push the bar.
    bus.update("job", phase="t2i_sample", label="Drawing", inner=1.0)
    snap = bus.snapshot("job")
    assert snap is not None
    assert snap["percent"] < 100.0, "the bar finished during the first pass"
