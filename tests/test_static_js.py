"""The frontend's one automated check: does it parse?

There is deliberately no build step, no bundler and no JS test harness (see
CLAUDE.md), which leaves a gap nothing else in this suite covers: a syntax
error in ``static/app.js`` is not a broken feature, it is a broken *page*. The
module never executes, so every select stays empty, no handler binds and the
viewer never initialises -- and the server keeps serving the file happily,
because to FastAPI it is just bytes.

That is not hypothetical. ``updateNode`` grew a second ``const p`` in a scope
that already had one; a const redeclaration is an early error, so the whole
module stopped parsing and the entire UI went dead. Seven code reviews read
that diff and none of them caught it, because reading JS is not running it.

Parsing is not building: no artifact is produced, nothing is bundled, and node
is not a project dependency -- the test skips when it is absent rather than
demanding it. It is the cheapest possible check that the page can run at all.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from warlock import guidance

STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "warlock" / "static"

# vendor/ is third-party three.js shipped as-is; it is not ours to lint and
# parsing all of it would dominate the runtime of this file.
OUR_SCRIPTS = sorted(p for p in STATIC_DIR.glob("*.js"))


def _node() -> str | None:
    return shutil.which("node")


@pytest.mark.skipif(_node() is None, reason="node not installed; JS parse check skipped")
@pytest.mark.parametrize("script", OUR_SCRIPTS, ids=lambda p: p.name)
def test_static_script_parses(script: Path) -> None:
    """``node --check`` the file exactly as the browser would load it.

    ``--input-type=module`` matters: app.js uses import statements, and checking
    it as a classic script reports every one of them as a syntax error.
    """
    proc = subprocess.run(
        [_node(), "--input-type=module", "--check"],
        stdin=script.open("rb"),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{script.name} does not parse:\n{proc.stderr}"


def test_there_is_a_script_to_check() -> None:
    """A guard on the guard: if static/ is reorganised and the glob stops
    matching, every test above silently becomes zero test cases."""
    assert OUR_SCRIPTS, f"no *.js found in {STATIC_DIR}"


def test_every_element_the_module_grabs_at_load_time_exists_in_the_html():
    """A top-level getElementById that returns null kills the whole page.

    The bulk-export bar wires its handlers at module scope, so a typo'd or
    missing id is not a dead button -- it is a TypeError before the viewer
    initialises, exactly the failure mode this file exists to catch. Checked
    by id text rather than by running the page, which is all that is possible
    without a DOM.
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "bulk-bar",
        "bulk-count",
        "bulk-zip",
        "bulk-folder",
        "g-profile",
        "dl-collision",
        "dl-textures",
        "dl-fbx",
    ):
        assert f'id="{element_id}"' in html, (
            f"app.js grabs #{element_id}; index.html has no such id"
        )


def test_every_guidance_field_has_a_matching_select_and_row():
    """loadGuidance() in app.js queries `[data-guidance="<field>"]` for every
    name in GUIDANCE_FIELDS and toggles `#g-<field>-row` on tab switch. A
    field present in one but missing the matching HTML breaks initialization
    silently (loadGuidance's promise chain ends in .catch(console.error)) or
    throws from inside the tab-click handler. Parse app.js's own array so
    this test can't drift from the source of truth.
    """
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    match = re.search(r"const GUIDANCE_FIELDS = \[(.*?)\];", app_js, re.DOTALL)
    assert match, "GUIDANCE_FIELDS array not found in app.js"
    fields = re.findall(r'"([\w]+)"', match.group(1))
    assert fields, "GUIDANCE_FIELDS parsed empty"

    for field in fields:
        assert f'data-guidance="{field}"' in html, f"no select for guidance field {field!r}"
        assert f'id="g-{field}-row"' in html, f"no row div for guidance field {field!r}"

    # And the reverse direction: guidance.py is the source of truth for which
    # fields the API accepts, so a taxonomy field added there but never wired
    # into GUIDANCE_FIELDS would be invisible in the UI while still accepted
    # by the API -- silently unreachable rather than loudly broken.
    assert set(guidance.form_fields()) <= set(fields)
