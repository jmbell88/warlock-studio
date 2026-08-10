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
# 0.0.9 against 0.0.11, and tests/test_models.py pins them together by reading
# the installed distribution's metadata rather than re-parsing the TOML.
__version__ = "0.0.17"
