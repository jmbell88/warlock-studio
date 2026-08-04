"""A fixed benchmark: the same prompts, the same seeds, measured the same way.

This package exists to answer one question the app cannot: did a change make
the assets better, or does it only feel that way? It is a developer tool, not
a feature -- nothing in ``studio/`` or ``service/`` imports it, and it reaches
the app the same way a user does, through ``service.jobs.create_job`` on a
real ``studio.runtime.Runtime``.

The split mirrors pipelines/sheet.py: everything decidable (the suite, the
recipes, the manifest, the grid, the scoring) is pure and torch-free, and only
the two stages that genuinely need a GPU (running jobs, rendering views) touch
one. That is what lets a metric change be re-applied to every past run without
regenerating anything.

Entry point: ``python -m warlock.bench <subcommand>``. Deliberately not
``warlock``'s own CLI -- that is the user-facing binary, and its flat
``choices=[...]`` would have to become a subparser tree to host this.
"""

from __future__ import annotations
