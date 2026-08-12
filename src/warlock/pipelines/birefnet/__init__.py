"""BiRefNet's modelling code, vendored so nothing runs downloaded Python.

``matting`` used to build this model through
``AutoModelForImageSegmentation.from_pretrained(..., trust_remote_code=True)``,
which executes whatever ``birefnet.py`` the repository happens to ship, in the
app's own process, with the user's filesystem and network permissions --
and ``uv.lock`` says nothing at all about code fetched at runtime (MDL-03).
Pinning the revision narrowed that to "whatever that commit shipped". This
closes it: the code that runs is the code in this checkout.

:func:`load` is the whole public surface. It builds the architecture from the
vendored source and loads the published weights with ``strict=True`` -- which
is the parity assertion, not merely a setting: a state dict that does not map
exactly onto this architecture means the vendored copy and the checkpoint have
drifted, and the only safe answer is to refuse rather than to run a model with
randomly-initialised layers in it.

See ``ATTRIBUTION.md`` for the upstream source, its licence and the four
deliberate modifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load", "build"]

#: The weights file inside a BiRefNet snapshot. One name, because the registry
#: entry downloads exactly one.
WEIGHTS_NAME = "model.safetensors"


def build(config: Any = None) -> Any:
    """The architecture, with no weights in it. Randomly initialised."""
    from .configuration import BiRefNetConfig
    from .modeling import BiRefNet

    return BiRefNet(config=config or BiRefNetConfig())


def load(path: Path) -> Any:
    """The published checkpoint at ``path``, as an eval-mode model.

    ``strict=True`` deliberately. The loose load ``from_pretrained`` performs
    would silently leave a layer randomly initialised if the vendored source
    and the checkpoint disagreed, and a BiRefNet with one random decoder block
    does not fail -- it returns a plausible, wrong mask, which is the failure
    this whole exercise is meant to make impossible.
    """
    import torch
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path / WEIGHTS_NAME))
    model = build()
    model.load_state_dict(state, strict=True)
    model.eval()
    # No ``to(device)`` and no dtype cast here: ``matting._load`` owns both, and
    # its reasoning about fp16 on the CPU is measured rather than assumed.
    del torch
    return model
