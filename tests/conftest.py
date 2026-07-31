from __future__ import annotations

import pytest

from animancer3d.db import JobStore


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()
