import os

# Belt-and-suspenders: every model load uses a local path with
# local_files_only=True, but also make sure no transitive huggingface_hub
# code can ever phone home. huggingface_hub reads these at import time, and
# it is only imported lazily from package modules -- so setting them here in
# the package __init__ is guaranteed to run first for every entry point
# (cli, ASGI runner, tests). setdefault so a deliberate user override wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Keep in step with pyproject.toml's [project] version -- the two drifted to
# 0.0.9 against 0.0.11, and tests/test_offline.py pins them together by reading
# the installed distribution's metadata rather than re-parsing the TOML.
# CHANGELOG.md's top heading is the third surface and tests/test_changelog.py
# pins that one; a bump that moves fewer than all three leaves Home printing a
# version whose "What's new" entry does not exist (2026-08-11, v0.0.22).
__version__ = "0.0.22"
