# ACE-Step, vendored

## What this is

Every file under this directory is **third-party source**, copied from the
ACE-Step repository so that this application never executes Python it
downloaded at runtime. It is the model code behind the Muse mode.

| | |
|---|---|
| Upstream | <https://github.com/ace-step/ACE-Step> |
| Project | <https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B> |
| Commit | `1bee4c9f5b43e30995f8d4d33b3919197ce1bd68` |
| Vendored on | 2026-09-04 |
| Licence | Apache-2.0 |

Upstream's package layout is mirrored one-for-one, path for path: flattening it
would make a re-vendoring undiffable, which is the whole value of vendoring.

Source files as fetched, before any modification:

| Path | SHA-256 of the original |
|---|---|
| `__init__.py` | `b9246213861cf6107bd05e0aeef487196e86c3389a98718a7a43298429367207` |
| `apg_guidance.py` | `8217c8857d8817f8648f1ef09ae65c6fb92151a79b654148eba0e7dd1adaf1b9` |
| `cpu_offload.py` | `46faea0454bb41e769e4ab9315ec216e102b136758bef24adff5eda5300d628f` |
| `language_segmentation/LangSegment.py` | `b89aa89648401fc814f64aa473f4a64b7393139bacd9d3a7b92f54a4ff21de01` |
| `language_segmentation/__init__.py` | `2028bc12b7ffbd48dbe28c9ba0445b3ac482f94932b83af1817bf3dedc66508b` |
| `language_segmentation/language_filters.py` | `1051902f86bae96aa1df310e8dad4bc5920c280225f4d827510d964e425e4881` |
| `language_segmentation/utils/__init__.py` | `06a19fa9bc30fed0b58b8105bd06e3348c185051570a1421be5b76fbf08bb141` |
| `language_segmentation/utils/num.py` | `75c493c49b84e2317a7068a546d5a0081d33fee59317b5f9ee8a59aa465a6d24` |
| `models/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `models/ace_step_transformer.py` | `d1366bd4c3a779180e62bfdd88e57b96a6fcea3cfe43213cae4b3f6db5cdbfcc` |
| `models/attention.py` | `95ed7d7f2afb8949f1f0f4a1dc441016af9124b5d27496f919aa8ee6792db8a8` |
| `models/config.json` | `bfc4026895594e278bdb3b42ec4b7bce8314e7862bb54e2ff1e9a12b17a4913e` |
| `models/customer_attention_processor.py` | `1e88ec3b3362020b8df33096804779d81a26c29fe2726b31b242a7c0e23d30b0` |
| `models/lyrics_utils/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `models/lyrics_utils/lyric_encoder.py` | `9591753bf014ef55aa835e43c0f4daae00d21e86423fb33ae9a4509c87e63a05` |
| `models/lyrics_utils/lyric_normalizer.py` | `1d083350d2570cfbe4990f0a0ccb82fedab55a5f4de7e55767663c99bc315f6c` |
| `models/lyrics_utils/lyric_tokenizer.py` | `0a34860d09f6a8d798f5e14422888e8a91b68607cc68bbeba93a51cb38ee45a0` |
| `models/lyrics_utils/vocab.json` | `69a44d1be0d8c46bad439046b1a339791c68da63e61112be0ded70180d80bccb` |
| `models/lyrics_utils/zh_num2words.py` | `af3423dd2861e64224a55cb1c3d95dccfed1320a5d1de940836e8a4830a6ed36` |
| `music_dcae/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `music_dcae/music_dcae_pipeline.py` | `9a94af45c799eff515e459ab1bde41b721c2f6ac026643ed4c33c864a722437b` |
| `music_dcae/music_log_mel.py` | `c9829a3d7ee337fd1b3288e633b85a68d7ecf487faf6d398a88d2acd4aee868a` |
| `music_dcae/music_vocoder.py` | `74bdfaf2a3f91533ef0550d691345ea7151b31203767533f6cb39ce663eccc07` |
| `pipeline_ace_step.py` | `cac6b40c12508467bad7a146cf77411030a7be9767f83e314b6f4c51adb203f5` |
| `schedulers/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `schedulers/scheduling_flow_match_euler_discrete.py` | `d5999388b4cd92fff04b143282f8c19e9a99190286b7068083a89404847fe064` |
| `schedulers/scheduling_flow_match_heun_discrete.py` | `dfde88ef5013060bc261f8d2a7e437314fb3a83af868d2b798548a1c8880f77b` |
| `schedulers/scheduling_flow_match_pingpong.py` | `8da87cd29f953acbc815c53f0ad579a551d0219adfc9561f0cd90f4ee3e433ac` |

## What is not here

`gui.py`, `ui/`, `data_sampler.py` and `text2music_dataset.py`, plus everything
outside the `acestep/` package (trainers, notebooks, Docker, LoRA data). The
vendoring surface is exactly `pipeline_ace_step.py`'s import closure and no
more.

All three schedulers are here even though a given generation uses one:
`text2music_diffusion_process` dispatches on a *string* out of the recipe, so
an unvendored scheduler would fail at call time rather than at import, which is
the worst place to find out.

## Why

The same argument the vendored BiRefNet makes: `uv.lock` says nothing about
code fetched at runtime, and no reviewer of this repository could otherwise
read what the process is going to run. ACE-Step is not on PyPI, so the
alternative was a git dependency — which is the same problem with a URL in
front of it.

The dependencies did not go away. `torch`, `diffusers`, `transformers`,
`torchaudio`, `librosa`, `soundfile`, `accelerate`, `peft`, `loguru` and the
lyric-language stack are declared as the `music` extra in `pyproject.toml`.

## The modifications

Three, each marked with a `WARLOCK n/3:` comment in the source. Every other
line is upstream's, byte for byte.

1. **`cancel_event` and `on_step` threaded into the sampling loop**
   (`pipeline_ace_step.py`). `ACEStepPipeline.__call__` takes no cancel hook of
   any kind, and this is the one modification the feature cannot work without:
   without it a two-minute generation cannot be stopped, and the only remaining
   cancel is killing the child and throwing away a warm 8 GiB pipe on an action
   users take routinely.

   Both are new keyword arguments defaulting to `None`, forwarded from
   `__call__` to `text2music_diffusion_process`, which checks the event once
   before the loop — load and encode have no interruption point of their own —
   and once per step, raising the locally defined `WarlockCancelled`. This is
   the shape `Text2Image.generate` already uses, so the queue's cancel
   semantics are identical across the two model workers.

   `on_step(i + 1, total)` rides along in the same loop rather than earning a
   second modification of its own; without it a music job's progress bar has
   nothing between "load" and "done".

2. **`get_checkpoint_path` and `load_lora` refuse rather than download.**
   Upstream falls through to `snapshot_download` when the directory is
   unpopulated. Weights arrive here through `warlock.fetch` and its own
   subprocess, never at load time, so an absent directory raises
   `FileNotFoundError` naming the Settings page instead. Belt and braces beside
   `HF_HUB_OFFLINE=1`, exactly as the vendored BiRefNet passes
   `local_files_only=True`.

3. **`__init__.py` registers the package under the bare name `acestep`.**
   Upstream imports itself absolutely (`from acestep.models... import ...`),
   which cannot resolve from `warlock.pipelines.acestep`. One
   `sys.modules.setdefault` at the foot of `__init__.py` makes all of them
   resolve to these files. The alternative — rewriting the import block at the
   top of a dozen vendored modules — would make each of them undiffable against
   upstream for no gain. Only the music worker subprocess imports this package,
   so the bare name never appears in the app process.

## Updating

If the model is ever re-pinned to a newer commit:

1. Re-clone at the new commit, copy the same file set over this directory, and
   re-apply the three modifications.
2. Update the commit, date and SHA-256 table above.
3. Run `uv run pytest -m gpu -k music` — the seeded golden-checksum parity
   check is what will catch an arithmetic change.
