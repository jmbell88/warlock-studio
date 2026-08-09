"""The lazy-derivation engine behind every artifact download.

Everything but the GLB, the reference PNG and the rig is a pure function of
``model.glb``, produced on first request and cached beside it. One lock per
(job, artifact) means two callers asking at once wait for one conversion
rather than starting two -- which for the FBX path is two Blender subprocesses
writing the same filename.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from .. import rigging
from . import files
from .core import WarlockService
from .errors import Failed, Invalid, NotFound, NotReady
from .files import MEDIA

log = logging.getLogger(__name__)


def get_file(
    svc: WarlockService,
    job_id: str,
    name: str,
    *,
    pixel_colors: int | None = None,
    pixel_palette: str | None = None,
    pixel_dither: bool | None = None,
) -> Path:
    """The path of one artifact, deriving it first if that is possible.

    Blocking: a cold STL of a 300k-face mesh is seconds and an FBX is a Blender
    subprocess, so this must never be called from the frame thread.

    ``pixel_colors`` is the derive-time palette preference for the pixel-art
    artifacts: ``None`` means no opinion (serve whatever is on disk if fresh,
    which is every pre-knob caller verbatim), an int means the served file must
    have been quantized to that cap (0 = full colour) and a mismatch re-derives
    an otherwise-fresh file. It is validated against a literal tuple and never
    becomes part of a path.
    """
    if not name:
        # Before the allowlist, and a different sentence (O117). "unknown file"
        # is deliberately ambiguous -- it covers a name that does not exist and
        # a name that exists and is not servable, and telling those apart leaks
        # what a job directory holds. An *empty* name leaks nothing, because it
        # is not a question about this job at all: it is a caller that composed
        # a request with a blank in it, and answering it with the ambiguous
        # sentence sent the reader looking for a file nobody asked for.
        raise NotFound("no file was named")
    if name not in MEDIA:
        raise NotFound("unknown file")
    if pixel_colors is not None and pixel_colors not in files.PIXEL_COLOR_CHOICES:
        choices = ", ".join(str(n) for n in files.PIXEL_COLOR_CHOICES)
        raise Invalid(f"pixel_colors must be one of {choices}")
    # The row is fetched, not just the id checked: readiness is a fact about
    # the job (a mesh is only servable once it is *done*), and an orphaned
    # directory used to serve files for a job that no longer exists.
    job = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    path = job_dir / name
    glb = job_dir / "model.glb"
    if not files.ready(job, job_dir, name):
        raise NotReady(files.unready_reason(job, job_dir, name))
    # FBX needs a Blender subprocess rather than a trimesh call, so it does not
    # fit the `derived` dict below -- but it takes the same per-artifact lock,
    # for the same reason. Existence checked only under the lock: Blender
    # writes the served filename in place, so a lock-free exists() during the
    # first export would open a truncated FBX. The trimesh conversions below
    # don't need this -- postprocess stages them and renames atomically.
    if name == "model.fbx" and glb.exists():
        with svc.convert_lock(job_id, name):
            if not path.exists():
                spec = rigging.fbx_spec(glb, path, job_dir)
                try:
                    # FBX export is import-plus-export like a pose bake, so it
                    # reuses pose_timeout rather than gaining a knob.
                    rigging.run_worker(spec, timeout=svc.config.pose_timeout)
                except rigging.BlenderError as exc:
                    log.error("fbx export for %s failed: %s", job_id, exc)
                    raise Failed("could not export FBX") from exc

    from ..pipelines import postprocess

    # Everything that is a pure function of model.glb, derived on first request
    # and cached. Each entry takes (glb, out_path).
    derived = {
        "model.stl": postprocess.glb_to_stl,
        "model_obj.zip": postprocess.glb_to_obj_zip,
        "collision.glb": postprocess.glb_to_collision,
        "textures.zip": postprocess.glb_to_textures_zip,
    }
    if not path.exists() and name in derived and glb.exists():
        convert = derived[name]
        with svc.convert_lock(job_id, name):
            # Re-checked inside the lock: whoever waited here was waiting for
            # exactly this file, so the second caller gets the finished
            # artifact instead of exporting a 300K-face mesh again.
            if not path.exists():
                try:
                    convert(glb, path)
                except ValueError as exc:
                    # "this model has no textures" is a fact about the mesh,
                    # not a fault -- the artifact simply cannot exist.
                    raise NotReady(str(exc)) from exc
    # Freshness rather than existence: a hand edit or a revert rewrites
    # input.png in place, and an artifact older than it is a picture of pixels
    # that are gone. files.fresh_2d answers "missing" and "stale" the same way
    # on purpose -- both mean derive it again.
    opts = _pixel_opts(svc, name, pixel_colors, pixel_palette, pixel_dither)
    if name in files.DERIVED_2D and not (
        files.fresh_2d(job_dir, name) and _pixel_current(job_dir, name, pixel_colors, opts)
    ):
        _derive_2d(
            svc, job, job_id, job_dir, name, pixel_colors=pixel_colors, opts=opts
        )
    if not path.exists():
        # After the derivation ran, so this is "it was produced and is still
        # not there" -- a different fact from the gate above, and the reason
        # names the artifact rather than repeating the gate's sentence.
        raise NotReady(f"{name} could not be produced for this job.")
    return path


def derivable(name: str) -> bool:
    """Whether ``name`` can be produced from model.glb on demand.

    The UI uses this to decide between a save button and a "derive, then save"
    one, without having to know which conversions exist.
    """
    return name in files.DERIVED


def derivable_2d(name: str, stage: str | None) -> bool:
    """Whether ``name`` can be produced from *this stage's* input.png.

    The stage is an argument rather than an assumption because the two image
    stages take different halves of the set: a reference has the cutouts and a
    tile has the wrapped view, and neither has the other's.
    """
    return name in files.derived_2d_for(stage)


# The manifest's name, and with it the lock discipline around it. Deriving an
# artifact takes that artifact's lock and the manifest's *inside* it, never the
# other way about: two derivations of different artifacts genuinely race for
# the manifest, and a consistent order is the only thing standing between that
# and a deadlock. The single path that does not nest the two is a request for
# the manifest itself, where they would be one and the same non-reentrant lock
# -- it takes the manifest lock alone (see _derive_2d), which leaves the
# manifest lock innermost on every path there is. Nothing anywhere may hold the
# manifest lock and then reach for an artifact's.
MANIFEST = "manifest.json"


def _pixel_opts(
    svc: WarlockService,
    name: str,
    pixel_colors: int | None,
    pixel_palette: str | None,
    pixel_dither: bool | None,
):
    """One PixelOpts describing what the caller asked for, or None.

    None for every artifact that is not a pixel export and for the caller that
    expressed no opinion at all -- which is every pre-upgrade call site, and is
    what keeps them serving the file already on disk.
    """
    from ..pipelines import pixel as pixelmod

    if name not in files.PIXEL_ARTIFACTS:
        return None
    if pixel_colors is None and pixel_palette is None and not pixel_dither:
        return None
    colors = tuple()
    digest = None
    if pixel_palette:
        from . import palettes

        _name, colors, digest = palettes.load(svc.config, pixel_palette)
    return pixelmod.PixelOpts(
        colors=int(pixel_colors or 0),
        palette_name=pixel_palette or None,
        palette=tuple(colors) or None,
        palette_hash=digest,
        dither=bool(pixel_dither),
    )


def _pixel_current(
    job_dir: Path, name: str, pixel_colors: int | None, opts=None
) -> bool:
    """Whether the on-disk pixel artifact was cut with the requested palette.

    Freshness (fresh_2d) answers "does this file describe the input.png on
    disk"; this answers "was it quantized the way the caller wants" -- a knob
    change leaves the file mtime-fresh, so mtime alone would serve the old
    palette forever. The record is the manifest entry asset2d.pixel already
    writes (``palette``: int or None), read lock-free: _write_manifest only
    ever lands via tmp + os.replace, so a reader sees a whole old or whole new
    file, never a torn one. A missing or mangled manifest returns False --
    one re-derive rebuilds the entry, the same self-healing rule the manifest
    itself follows.
    """
    if name not in files.PIXEL_ARTIFACTS:
        return True
    if pixel_colors is None and opts is None:
        return True
    import json

    try:
        manifest = json.loads((job_dir / MANIFEST).read_text("utf-8"))
        entry = manifest["artifacts"][name]
    except (OSError, ValueError, TypeError, KeyError):
        return False
    if not isinstance(entry, dict):
        return False
    if opts is not None and opts.palette:
        # A palette file supersedes the median-cut cap entirely, so the cap is
        # not compared: the entry records ``palette: None`` whenever one is in
        # use. ``.get`` defaults throughout, so a manifest written before any of
        # these keys existed reads as current for a caller asking for none of
        # them, and stale the moment one is asked for.
        return (
            entry.get("palette_file") == opts.palette_name
            and entry.get("palette_hash") == opts.palette_hash
            and bool(entry.get("dither")) == opts.dither
        )
    if entry.get("palette_file") or entry.get("dither"):
        # On disk is a palette-mapped export and the caller asked for a plain
        # one. That is a different picture, whatever the cap says.
        return False
    if pixel_colors is None:
        return True
    return entry.get("palette") == (pixel_colors or None)


def _staged(job_dir: Path, name: str, write) -> None:
    """Write ``name`` through a dotfile and rename it into place.

    The rule every derivation here follows: a concurrent reader of an artifact
    this job derived a moment ago must never see a half-written file. The
    ``finally`` matters as much as the replace -- an encode that raises part way
    through would otherwise strand its staging file for the life of the job
    directory, where nothing ever looks at a dotfile again.
    """
    tmp = job_dir / f".{name}.tmp"
    try:
        write(tmp)
        os.replace(tmp, job_dir / name)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _derive_material(job_dir: Path, name: str, source: Path) -> None:
    """One map, or the zip of all of them, from a tile's input.png.

    The zip re-derives the maps into its own staging area rather than asking
    ``get_file`` for them, and that is a deliberate refusal to nest artifact
    locks. This function already holds ``material.zip``'s lock; reaching for
    three more would establish an ordering between four artifact locks that
    nothing else in this module has, and the module's one stated lock rule is
    that an artifact's lock is taken before the manifest's and never the other
    way about. The cost is that a zip requested after the three maps are
    already on disk computes them a second time -- a second or so on a 1024²
    tile, against a deadlock that would need three callers to reproduce and
    would hang the task runner.
    """
    from ..pipelines import material as material_lib

    if name in material_lib.MAP_NAMES:
        def write(tmp: Path) -> None:
            if material_lib.write_map(source, tmp, name) is None:
                raise NotReady(f"{source.name} could not be read as an image")

        _staged(job_dir, name, write)
        return

    import json
    import zipfile

    def write_zip(tmp: Path) -> None:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(source, "albedo.png")
            for map_name in material_lib.MAP_NAMES:
                staged = tmp.with_name(f"{tmp.name}.{map_name}")
                try:
                    if material_lib.write_map(source, staged, map_name) is None:
                        raise NotReady(f"{source.name} could not be read as an image")
                    zf.write(staged, map_name)
                finally:
                    with contextlib.suppress(OSError):
                        staged.unlink(missing_ok=True)
            zf.writestr(
                "material.json",
                json.dumps(material_lib.material_json("albedo.png"), indent=2),
            )
            zf.writestr("README.txt", _MATERIAL_README)

    _staged(job_dir, name, write_zip)


# Said in the zip rather than only in a docstring nobody downloading it will
# read. The whole point of pipelines/material is that these maps are estimates,
# and a file called material_normal.png in a folder of textures carries every
# implication of a measured one unless something says otherwise.
_MATERIAL_README = (
    "Warlock material set\n"
    "\n"
    "albedo.png is the generated tile. The other three maps are ESTIMATED from\n"
    "it and describe its contrast, not a measured surface:\n"
    "\n"
    "  material_height.png     darker albedo is treated as deeper\n"
    "  material_normal.png     slopes of that height, glTF/OpenGL green-up\n"
    "  material_roughness.png  local fine detail is treated as roughness\n"
    "\n"
    "Metalness is not estimated at all and material.json declares 0.0.\n"
    "All four images tile seamlessly, as the albedo does.\n"
)

def _derive_2d(
    svc: WarlockService,
    job: dict,
    job_id: str,
    job_dir: Path,
    name: str,
    *,
    pixel_colors: int | None = None,
    opts=None,
) -> None:
    """Produce one 2D export from the reference's input.png.

    Blocking, like every other derivation here: the matte is a model or a
    flood fill and the quantize is real work, so this must never be called
    from the frame thread.
    """
    from PIL import Image

    from ..pipelines import asset2d, matting

    source = job_dir / "input.png"
    if not source.exists():
        raise NotReady("this job has no reference image")
    if name == MANIFEST:
        # Nothing to compute: the manifest is written by the artifacts
        # themselves, so asking for it before any of them exist means writing
        # the header alone. Deliberately *without* an enclosing artifact lock
        # of its own -- the manifest's artifact lock and its manifest lock are
        # the same lock, and convert_lock hands out a plain, non-reentrant
        # threading.Lock, so wrapping the call would deadlock against itself.
        # _write_manifest takes it, and the ordering rule is untouched: the
        # manifest lock is still the innermost thing anyone holds.
        _write_manifest(svc, job, job_id, job_dir, None, None)
        return
    with svc.convert_lock(job_id, name):
        # Re-checked inside the lock, for the reason the mesh exports give:
        # whoever waited here was waiting for exactly this file. The same
        # compound test as the gate outside, not existence and not bare
        # freshness -- after a palette-knob change the file is mtime-fresh, so
        # a fresh_2d-only recheck would return the old palette to the caller
        # who asked for a new one.
        if files.fresh_2d(job_dir, name) and _pixel_current(
            job_dir, name, pixel_colors, opts
        ):
            return
        if name == "wrap_preview.png":
            # The one 2D export that is not a cutout, and so the one that runs
            # no matte: it is a view of the whole frame rather than a
            # measurement of a subject in it. Nothing about it goes in the
            # manifest for the same reason -- an entry there records the matte
            # that cut an artifact and the alpha it came out with, and this has
            # neither.
            from ..pipelines import seam

            tmp = job_dir / f".{name}.tmp"
            try:
                seam.wrap_preview(source, tmp)
                os.replace(tmp, job_dir / name)
            finally:
                # A failed or half-written roll must not leave its staging file
                # in the job directory: nothing ever looks at a dotfile again,
                # so it would sit there until the job was pruned. Suppressed
                # and not merely missing_ok, because a cleanup that raises here
                # would replace the failure the caller needs to see. Harmless
                # after a success, where the replace has already consumed it.
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            return
        if name in files.MATERIAL_2D:
            # No matte here either, and for the wrapped view's reason: every
            # cutout is the operation of lifting a subject off a background,
            # and a seamless texture *is* background. These are whole-frame
            # transforms of it. Nothing goes in the manifest for the same
            # reason -- an entry there records the matte that cut an artifact
            # and the alpha it came out with, and these have neither.
            _derive_material(job_dir, name, source)
            return
        with Image.open(source) as image:
            image.load()
            mask, matte = matting.mask(image, svc.config)
            try:
                if name == "icon.png":
                    out, meta = asset2d.icon(image, mask)
                elif name == "sprite.png":
                    out, meta = asset2d.sprite(image, mask)
                elif name in files.PIXEL_ARTIFACTS:
                    out, meta = asset2d.pixel(
                        image, mask,
                        size=files.PIXEL_ARTIFACTS[name],
                        colors=pixel_colors or 0,
                        opts=opts,
                    )
                else:
                    # Unreachable while DERIVED_2D and the branches above agree,
                    # which test_every_2d_artifact_has_a_derivation asserts at
                    # edit time. Spelled out anyway so that a seventh artifact
                    # added to the tuple and nowhere else surfaces as a service
                    # error the UI can show, rather than as a bare KeyError out
                    # of a dict lookup.
                    raise Failed(f"no derivation for {name}")
            except asset2d.NoSubject as exc:
                # A fact about the image, not a fault -- the same shape
                # glb_to_textures_zip's "this model has no textures" takes.
                raise NotReady(str(exc)) from exc
        meta["matte"] = matte
        meta["alpha"] = asset2d.alpha_report(out)
        # Staged and renamed: a concurrent reader of an artifact this job
        # derived a moment ago must never see a half-written PNG. The finally
        # is the same rule the wrapped view above follows -- an encode that
        # raises half way through would otherwise strand its staging file for
        # the life of the job directory.
        tmp = job_dir / f".{name}.tmp"
        try:
            out.save(tmp, "PNG")
            os.replace(tmp, job_dir / name)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        _write_manifest(svc, job, job_id, job_dir, name, meta)


def _write_manifest(
    svc: WarlockService,
    job: dict,
    job_id: str,
    job_dir: Path,
    name: str | None,
    meta: dict | None,
) -> None:
    """Merge one artifact's metadata into the job's manifest.

    Read-modify-write under the manifest's own lock, because that is what it
    is: several artifacts derived concurrently each add their own entry, and
    a whole-file write from a stale read would drop whichever landed first.

    The merge prunes as well as adds, and that is the half that keeps the
    manifest honest. An entry describes a file: its pivot, its alpha, which
    matte cut it. So an entry whose file is gone -- or is stale, because a hand
    edit rewrote input.png under it -- is a claim about pixels nobody can look
    at, and after an edit the whole set of them would be a verdict on a deleted
    image. Pruning against ``fresh_2d`` is deliberately the *only* mechanism
    here: dropping the file wholesale when it is itself stale would say the
    same thing in a second way, and would still leave the deleted-artifact case
    unhandled, whereas asking the freshness question per entry handles both.
    """
    import json

    from ..pipelines import asset2d

    path = job_dir / MANIFEST
    with svc.convert_lock(job_id, MANIFEST):
        try:
            manifest = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            # A truncated or hand-mangled manifest starts again from nothing.
            # Every entry in it is reproducible by re-deriving the artifact it
            # describes, so the recoverable loss is preferable to failing an
            # export on a file that is pure metadata.
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest.setdefault("version", 1)
        manifest["job"] = job_id
        manifest["prompt"] = job.get("prompt") or ""
        params = job.get("params") or {}
        manifest["recipe"] = asset2d.recipe_hash((params.get("recipe") or {}).get("reference"))
        # The recipe alone claims a seed and a model produced these pixels,
        # which after a hand edit is no longer the whole truth -- the same
        # reason files._remeasure records the flag in the first place. B4 shows
        # the manifest to the user, so the caveat has to travel with the claim.
        manifest["hand_edited"] = bool(params.get("hand_edited"))
        previous = manifest.get("artifacts")
        artifacts = previous if isinstance(previous, dict) else {}
        artifacts = {
            key: entry
            for key, entry in artifacts.items()
            if key != name and files.fresh_2d(job_dir, key)
        }
        if name is not None and meta is not None:
            artifacts[name] = meta
        manifest["artifacts"] = artifacts
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, path)
