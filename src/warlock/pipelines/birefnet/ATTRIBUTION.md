# BiRefNet, vendored

## What this is

`modeling.py` and `configuration.py` are **third-party source**, copied from the
published BiRefNet checkpoint repository so that this application never executes
Python it downloaded at runtime.

| | |
|---|---|
| Upstream | <https://huggingface.co/ZhengPeng7/BiRefNet> |
| Project | <https://github.com/ZhengPeng7/BiRefNet> |
| Commit | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` |
| Vendored on | 2026-08-12 |
| Licence | MIT |

Source files as fetched, before any modification:

| Upstream name | Here | SHA-256 of the original |
|---|---|---|
| `birefnet.py` | `modeling.py` | `208771ae626f653d64128fbf2d6ac9f8e645c5cc5e286258a73ec3322bbfe5ef` |
| `BiRefNet_config.py` | `configuration.py` | `e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52` |

`modeling.py` also carries a Microsoft copyright notice on the Swin Transformer
section it incorporates (line ~599 of the original), likewise MIT.

## Why

`matting` used to build the model with
`AutoModelForImageSegmentation.from_pretrained(..., trust_remote_code=True)`.
That flag executes whatever `birefnet.py` the snapshot on disk happens to hold —
in the app's own process, with the user's filesystem and network permissions.
It was already the *narrow* version of that risk: `local_files_only=True` meant
nothing was fetched at load time, and since MDL-03's pins the snapshot is a
named commit rather than a moving branch.

What neither of those fixes is that `uv.lock` says nothing about code
downloaded at runtime, and no reviewer of this repository could read what the
process was going to run. Vendoring closes that: the code that runs is the code
in this checkout.

The dependency on `einops`, `kornia`, `timm` and `torchvision` did **not** go
away — the same imports are at the top of `modeling.py`, and the registry entry
still carries its `uv sync --extra text2image` note.

## The modifications

Four, each marked with a `WARLOCK n/4:` comment in the source. Every other line
is upstream's, byte for byte, because the point of vendoring is that the code is
auditable against upstream and a fork that tidied while it was in there is a
fork nobody can diff.

None of them changes the arithmetic. `tests/test_birefnet_parity.py` compares
the mask this produces against one captured through the old remote-code path
before the switch; it came out **bit-identical**, 0 differing pixels of 65,536.

1. **The torchvision weights-enum imports and download paths are deleted.**
   `VGG16_Weights.DEFAULT` and friends are URLs — touching one reaches out to
   `download.pytorch.org`, which this app may never do. They were reachable only
   through `pretrained=True`, which the checkpoint's own `config.json` sets
   False (`bb_pretrained: false`), so the branch was dead weight pointing at the
   network. `build_backbone`'s `pretrained` default is now `False`, and the
   three torchvision backbones raise a clear refusal rather than trying.

2. **`build_backbone`'s `eval()` is an explicit table.** A backbone name out of
   a config file reaching `eval` is arbitrary execution with extra steps.
   `params_settings` — which only ever carries `in_channels=4` — is parsed
   rather than executed.

3. **The four remaining `eval()` config dispatches are explicit lookups**
   (`BLOCKS` and `REFINERS`, defined at the end of the module). `dec_blk`,
   `lat_blk`, `squeeze_block` and `refine` are all strings a config file can
   set. An unknown name is now a `KeyError` rather than whatever the string
   evaluates to.

   `HierarAttDecBlk` is named by `Config`'s own list of choices and is defined
   nowhere in upstream's file; selecting it raised `NameError` out of `eval`
   before and raises `KeyError` now. It is deliberately not stubbed — inventing
   a class upstream does not have would be the one modification that changes
   behaviour.

4. **`BACKBONES` is populated at import.** `build_backbone` builds it lazily
   too; doing it at the end of the module as well means a caller can read the
   table without constructing a model, which is what the `pvt_v2_b0` CPU shape
   test does.

Plus one rename that is not a code change: `BiRefNet_config` → `configuration`,
so the one import inside `modeling.py` points at it.

## Loading

`birefnet.load(path)` builds the architecture and loads `model.safetensors`
with **`strict=True`**. That is a parity assertion, not a setting: the loose
load `from_pretrained` performs would leave a layer randomly initialised if the
vendored source and the checkpoint ever drifted, and a BiRefNet with one random
decoder block does not fail — it returns a plausible, wrong mask.

## Updating

If the checkpoint is ever re-pinned to a newer commit:

1. Re-capture the golden mask through the **current** code first.
2. Copy the new `birefnet.py` over `modeling.py`, re-apply the four
   modifications, and update the commit, date and SHA-256 above.
3. Run `uv run pytest -m gpu -k birefnet`. `strict=True` will catch an
   architecture change loudly; the parity test will catch an arithmetic one.
