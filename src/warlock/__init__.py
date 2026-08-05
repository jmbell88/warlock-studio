import os

# Belt-and-suspenders: every model load uses a local path with
# local_files_only=True, but also make sure no transitive huggingface_hub
# code can ever phone home. huggingface_hub reads these at import time, and
# it is only imported lazily from package modules -- so setting them here in
# the package __init__ is guaranteed to run first for every entry point
# (cli, ASGI runner, tests). setdefault so a deliberate user override wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

__version__ = "0.0.8"
