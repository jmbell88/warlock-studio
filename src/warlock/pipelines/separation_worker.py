"""Hybrid Demucs in a short-lived child: one mix in, four stems out.

Reads a JSON spec on stdin (see ``rigging.run_worker``). Writes its result to
``spec["result_path"]`` and progress to stdout as ``[separate] <frac> <label>``.

**Why a one-shot child, when both existing model workers are resident.** Both
of those argue from *N calls per user action*: ``music_worker`` because a
retake, a repaint and a second take are three generates against the same
8.3 GiB, and ``matting_worker`` because an asset's icon, sprite and pixel sets
are three calls against one twelve-second load. Separation is **one call per
take, minutes apart, against a ~2-second, 300 MB load.** Neither argument
survives contact with those numbers, and the counter-pressure is real:
``matting_worker``'s own docstring records BiRefNet leaving 1053 MB resident
after ``gc.collect()``, and that "Warlock's worst crash to date is host-commit
exhaustion."

So: spawn, separate, write, exit. Which means **cancel is a kill**, which means
no ``_workerio`` import, no third hand-copy of the stdin reader, and no
vendored modification. ``music_worker`` needed the vendored ``cancel_event``
hook precisely because the alternative was throwing away a warm 8.3 GiB pipe;
here it throws away 300 MB and two seconds. This is the shape ``_workerio``'s
docstring explicitly reserves for ``matting_worker``/``blender_worker``, so
taking it is agreeing with the invariant rather than working around it.

**Why ``torchaudio.models.hdemucs_high`` and not the ``demucs`` package.** The
model class is already installed -- ``torchaudio`` is in the ``music`` extra for
ACE-Step -- so this adds **zero packages**. The PyPI ``demucs`` would drag in
``dora-search``, ``julius``, ``lameenc``, ``openunmix``, ``submitit`` and
``treetable`` for a class already on disk, and ``lameenc`` would be a second MP3
encoder beside the libsndfile one ``pipelines/audioout`` uses -- which is
exactly the "two spellings of one thing" shape this repo writes tests against.

**Never ``HDEMUCS_HIGH_MUSDB_PLUS.get_model()``.** That bundle calls
``torchaudio.utils.download_asset`` at load time, which is a runtime download
outside ``fetch_worker`` -- the one thing ``HF_HUB_OFFLINE=1`` exists to make
impossible. The model is built and ``load_state_dict``-ed from our own path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: The checkpoint's filename inside the model directory. ``models.Fetch``'s
#: ``filename`` for the same entry, restated here because a pipeline may not
#: import the registry.
CHECKPOINT = "hdemucs_high_trained.pt"


def _emit(fraction: float, label: str) -> None:
    """One progress line, in the format ``rigging.run_worker`` parses."""
    print(f"[separate] {fraction:.3f} {label}", flush=True)


def separate(spec: dict) -> dict:
    """Split one mix into its stems. -> the result the host reads.

    Segmented rather than fed whole, and that is what makes this survivable on
    a card the music pipe may still be releasing: four minutes of 44.1 kHz
    stereo through a U-Net at once is an allocation nobody has measured, while
    ten seconds at a time is bounded by a figure the registry declares. The
    segments are blended with a triangular window, which is what stops each
    boundary being an audible step -- an unwindowed concatenation puts a seam
    every ``segment_seconds`` in all four stems at once.
    """
    import soundfile as sf
    import torch
    from torchaudio.models import hdemucs_high

    sources = tuple(spec["sources"])
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit(0.02, "Loading the separation model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = hdemucs_high(sources=list(sources))
    state = torch.load(
        str(Path(spec["model_dir"]) / CHECKPOINT),
        map_location="cpu",
        # ``WARLOCK 4/5``'s argument on a different checkpoint: these weights
        # arrive through ``warlock.fetch`` and are digest-pinned, so they are
        # not untrusted input -- but restricting the load to tensors costs
        # nothing and closes the same class of risk.
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(device).eval()

    _emit(0.10, "Reading the take")
    audio, rate = sf.read(str(spec["source"]), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        # The model is stereo-in by construction. A mono take is widened here
        # rather than refused: what comes back is four mono-equivalent stems,
        # which is the right answer rather than an error about channel counts.
        audio = audio.repeat(2, axis=1)
    mix = torch.from_numpy(audio.T).to(device)

    # Normalised by the mix's own statistics and put back afterwards. Demucs is
    # trained on level-normalised input, so a quiet take separates measurably
    # worse without this -- and restoring the scale is what keeps the four
    # stems summing back to the take rather than to a louder version of it.
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std().clamp(min=1e-6)
    mix = (mix - mean) / std

    segment = int(float(spec.get("segment_seconds", 10.0)) * rate)
    overlap = segment // 4
    total = mix.shape[-1]
    stems = torch.zeros(len(sources), 2, total, device=device)
    weights = torch.zeros(total, device=device)
    window = torch.hann_window(segment * 2, device=device)[:segment] if segment else None

    start = 0
    with torch.no_grad():
        while start < total:
            end = min(start + segment, total)
            chunk = mix[:, start:end]
            piece = model(chunk[None])[0]
            span = end - start
            shape = window[:span] if window is not None else torch.ones(span, device=device)
            stems[..., start:end] += piece * shape
            weights[start:end] += shape
            _emit(0.10 + 0.85 * (end / max(total, 1)), "Separating")
            if end >= total:
                break
            start += segment - overlap

    stems = stems / weights.clamp(min=1e-6)
    stems = stems * std + mean

    _emit(0.97, "Writing the stems")
    written = []
    for index, name in enumerate(sources):
        path = out_dir / f"{name}.wav"
        # Staged and renamed, like every other write onto a name something else
        # reads: ``stems.json`` is the completion gate, but a half-written stem
        # beside it would still be a file the library offers.
        tmp = path.with_name(f".{path.name}.tmp")
        sf.write(
            str(tmp),
            stems[index].T.cpu().numpy(),
            int(rate),
            subtype="PCM_16",
        )
        tmp.replace(path)
        written.append(f"{name}.wav")

    return {"ok": True, "files": written, "rate": int(rate), "device": device}


def main() -> int:
    # ``blender_worker``'s rule: a malformed spec is reported in a sentence and
    # an exit code, never as a traceback. The result path is resolved before
    # anything is loaded, so a spec that could never hand anything back is
    # refused before the weights are read.
    try:
        spec = json.loads(sys.stdin.read())
        result_path = Path(spec["result_path"])
    except (ValueError, TypeError, KeyError) as exc:
        print(f"the worker spec on stdin is not usable: {exc}", file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except ImportError as exc:
        print(f"the music extra is not installed: {exc}", file=sys.stderr)
        return 3
    try:
        result = separate(spec)
    except Exception as exc:  # noqa: BLE001 -- the host reads the sentence
        result = {"ok": False, "error": str(exc)}
    tmp = result_path.with_name(result_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(result_path)
    finally:
        tmp.unlink(missing_ok=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
