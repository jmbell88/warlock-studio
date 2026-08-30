"""Queue a graded prop corpus at the shipped defaults: P3's submitter, and P12's.

The graded-mesh programme wants "a representative corpus through text ->
sdxl_cfg -> TRELLIS at the shipped defaults", and until this file there was
nothing that could submit one. The corpus and the protocol are pre-registered
in ``docs/measurements/2026-08-30-art-verdicts-preregistration.md``.

The plan had named ``scripts/qualify_tiers.py`` as the harness, which it is
not: that script qualifies the *gltfpack tiers* against meshes a human has
already accepted, and documents that it deliberately does not drive
``optimize_job`` because doing so would consume its own inputs. It cannot
generate a corpus.

**A corpus is not a sweep, and this is the reason.** ``service.sweeps.SweepPlan``
carries exactly one ``prompt``; ``unit_kwargs`` hands that same string to every
unit and ``prompt`` is in neither ``KWARG_AXES`` nor ``guidance.form_fields()``,
so a vector naming it is refused. That is correct -- a sweep varies *settings*
while holding the subject still, which is what makes its units comparable. A
corpus varies the *subject* while holding the settings still. They are opposite
instruments and the sweep machinery is the wrong one here.

So these go in as ordinary library jobs, which costs nothing: ``review_mode``'s
first bucket "is not a sweep at all -- it is the recent finished meshes nobody
has judged", and ``JudgingPass`` walks every bucket with work left in one run.
The grading loop is therefore already the one this corpus needs. It also means
``sweeps.cleanup_sweep`` cannot reach these rows: it refuses ``RECENT_ID``, so
nothing auto-deletes the corpus the moment its last unit is graded.

**Retention is a requirement here, not a preference.** The 2026-08-13 run's
library was cleaned after grading -- "the 20 job rows and every GLB are gone" --
and that document closes by telling any future run to plan around it by
retaining its evidence until the writeup exists. Every job this queues carries
the corpus tag for exactly that: it is how the corpus is found again afterwards.

**The tag is the corpus id and nothing else.** The corpus file classes each
subject easy/medium/hard, and that class is deliberately *not* written onto the
job. A difficulty label visible in the review pane is a thumb on the scale of a
pass whose whole point is that it is blind; the class is joined back on the
prompt at writeup time instead, where it cannot bias a grade.

A submitter, not a runner, like ``_campaign.py``: it writes ``queued`` rows and
exits, and the app's worker drains them at next launch. Nothing here touches the
card, the trellis port or the event loop.

    uv run python scripts/campaign_props.py --dry-run
    uv run python scripts/campaign_props.py
    uv run python scripts/campaign_props.py --seeds 42,11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _campaign  # noqa: E402

from warlock import guidance  # noqa: E402
from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402
from warlock.service import jobs as jobs_mod  # noqa: E402
from warlock.service.core import WarlockService  # noqa: E402
from warlock.service.errors import ServiceError  # noqa: E402
from warlock.service.validation import (  # noqa: E402
    check_prompt,
    check_seed,
    check_vram,
    check_weights,
)

#: The corpus this campaign exists to submit. A path rather than a literal so
#: the file is what the pre-registration cites and what a re-run reproduces.
DEFAULT_CORPUS = _ROOT / "docs" / "measurements" / "corpora" / "props-v1.txt"

#: The tag every queued job carries. Constant across the corpus on purpose --
#: see the module docstring on why the difficulty class is not written here.
DEFAULT_TAG = "props-v1"

#: The classes the corpus file may use. Checked so a typo becomes a refusal
#: rather than a fourth silent class nothing reports on.
CLASSES = ("easy", "medium", "hard", "humanoid")


class Subject:
    """One line of the corpus: its class, its bare prompt, and where it came
    from, so a refusal can name a line the reader can open."""

    __slots__ = ("cls", "prompt", "line")

    def __init__(self, cls: str, prompt: str, line: int) -> None:
        self.cls = cls
        self.prompt = prompt
        self.line = line

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Subject({self.cls!r}, {self.prompt!r})"


def read_corpus(path: Path) -> list[Subject]:
    """Parse ``class | prompt`` lines. Blank and ``#`` lines are skipped.

    A malformed line raises rather than being dropped: a corpus silently one
    subject short is a corpus whose N does not mean what the writeup says.
    """
    if not path.is_file():
        raise SystemExit(f"no corpus file at {path}")
    subjects: list[Subject] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if "|" not in text:
            raise SystemExit(f"{path}:{number}: expected `class | prompt`, got {text!r}")
        cls, _, prompt = text.partition("|")
        cls, prompt = cls.strip().lower(), prompt.strip()
        if cls not in CLASSES:
            raise SystemExit(
                f"{path}:{number}: unknown class {cls!r}; one of {list(CLASSES)}"
            )
        if not prompt:
            raise SystemExit(f"{path}:{number}: no prompt after the class")
        subjects.append(Subject(cls, prompt, number))
    if not subjects:
        raise SystemExit(f"{path} names no subjects")
    return subjects


def parse_seeds(raw: str) -> tuple[int, ...]:
    """``"42,11"`` -> ``(42, 11)``, refusing a repeat.

    A repeated seed would queue two byte-identical jobs for one subject, which
    is ``sweeps._validate``'s duplicate-unit refusal in miniature: GPU spent
    redrawing one picture, and two rows in the corpus that look like evidence
    of agreement.
    """
    try:
        seeds = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit(f"--seeds takes whole numbers: {raw!r}") from exc
    if not seeds:
        raise SystemExit("--seeds names none")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"--seeds repeats a value: {raw!r}")
    return seeds


def shipped_params(svc: WarlockService) -> dict:
    """What a corpus job actually runs with: ``guidance.normalize`` over an
    empty form, with the same background-removal default ``create_job`` applies.

    Empty on purpose. Every override this script could pass is a departure from
    the configuration the verdict is *about* -- ``Config.mesh_profile`` stays
    ``raw``, no LoRA, no conditioning -- so the honest submitter passes nothing
    and lets the shipped defaults be the thing under test.
    """
    return guidance.normalize(
        {}, bg_default=guidance.default_bg_removal(svc.config.trellis_models_dir)
    )


def validate(
    svc: WarlockService, subjects: list[Subject], seeds: tuple[int, ...]
) -> None:
    """Everything ``create_job`` would refuse, refused before anything is written.

    ``sweeps._validate``'s all-or-nothing rule and its reason: a corpus that
    queues fourteen jobs and is then refused on the fifteenth leaves a partial
    run nobody asked for and nobody can interpret. The weights and VRAM halves
    do not vary across subjects -- the settings are identical by construction --
    but they are asked once so that a missing checkpoint refuses here rather
    than after twenty rows exist.
    """
    params = shipped_params(svc)
    check_weights(svc, "text", params)
    check_vram(svc, "text", "model", params)
    for seed in seeds:
        check_seed("seed", seed)
    for subject in subjects:
        try:
            check_prompt(subject.prompt)
        except ServiceError as exc:
            # Named by line, because "one of your twenty-two subjects is bad" is
            # not something anyone can act on -- ``sweeps._validate``'s rule.
            raise SystemExit(f"{subject.line}: {exc.message}") from exc


def spread(subjects: list[Subject], seeds: tuple[int, ...]) -> str:
    """"8 easy, 3 hard, 3 humanoid, 8 medium" -- the corpus as jobs, by class."""
    counts: dict[str, int] = {}
    for subject in subjects:
        counts[subject.cls] = counts.get(subject.cls, 0) + len(seeds)
    return ", ".join(f"{n} {cls}" for cls, n in sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--seeds",
        default="42",
        help="comma-separated; one seed per subject by default (breadth over depth)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate everything, write nothing"
    )
    args = parser.parse_args()

    subjects = read_corpus(args.corpus)
    seeds = parse_seeds(args.seeds)
    total = len(subjects) * len(seeds)

    config = Config()
    db_path = Path(config.db_path)
    _campaign.require_no_live_writer(db_path)

    store = JobStore(db_path)
    svc = WarlockService(config, store)
    try:
        try:
            validate(svc, subjects, seeds)
        except ServiceError as exc:
            print(f"refused: {exc.message}", file=sys.stderr)
            return 1

        print(f"{args.corpus}: {len(subjects)} subjects ({spread(subjects, seeds)})")
        print(f"seeds: {', '.join(str(s) for s in seeds)} -> {total} jobs")
        print(f"tag: {args.tag}   db: {db_path}")

        if args.dry_run:
            print("dry run: nothing written")
            return 0

        queued = 0
        for subject in subjects:
            for seed in seeds:
                job = jobs_mod.create_job(
                    svc,
                    kind="text",
                    prompt=subject.prompt,
                    output="model",
                    seed=seed,
                )
                # Through the door rather than ``store.set_meta``, so the tag is
                # normalized exactly as the library's own retag normalizes it.
                jobs_mod.update_job(svc, job["id"], {"tags": [args.tag]})
                queued += 1
        print(
            f"\n{queued} jobs queued. Launch Warlock Studio; the worker drains "
            "them and Review's recent bucket lists them.\n"
            "Do not clean the library until the writeup exists -- filter on "
            f"'{args.tag}' to find the corpus again."
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
