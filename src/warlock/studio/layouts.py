"""Saved workspace arrangements: the data half, with no imgui in it.

What a layout is, and deliberately is not. It captures **arrangement,
visibility and shares** -- which panes are in which column, in what order, and
how tall. It does *not* capture the sidebar width, the rail, the UI scale or
the theme: a workspace switch that collapsed your navigation is the same class
of surprise as the eighteen-failing-tests incident, and those four are
app-level preferences that happen to be stored nearby.

**Top-level settings keys**, ``workspace_layouts`` and ``active_layout``,
following ``profiles.py``'s pattern rather than living inside
``settings["layout"]`` -- ``Settings.set`` replaces a whole dict and
``test_a_save_writes_no_width_key_at_all`` asserts that dict's exact key set,
so a fifth key in it would be a test failure and a silently dropped preference
the first time anything else saved.

The blob is **sparse and per workspace**; absent means "the built-in", so the
migration story is empty and ``settings._migrate`` gains nothing.
``settings.VERSION`` stays 1 -- a bump discards the whole file -- and each blob
carries its own ``v`` instead: an unknown higher one is **kept verbatim and
listed greyed** rather than coerced, because the alternative is a newer build's
layout quietly rewritten by an older one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: This build's layout-blob version. Independent of ``settings.VERSION``.
VERSION = 1

#: The two layouts every profile has. ``default`` is the built-in arrangement
#: and ``mirrored`` is Aseprite's own second built-in -- the columns swapped --
#: which is one line of code and the proof the machinery works end to end.
BUILT_IN = ("default", "mirrored")

LAYOUTS_KEY = "workspace_layouts"
ACTIVE_KEY = "active_layout"


@dataclass
class Arrangement:
    """One workspace's saved shape, inside one named layout."""

    columns: dict[str, list[str]] = field(default_factory=dict)
    hidden: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "columns": {key: list(value) for key, value in sorted(self.columns.items())},
            "hidden": sorted(self.hidden),
        }

    @classmethod
    def from_json(cls, raw: Any) -> Arrangement:
        if not isinstance(raw, dict):
            return cls()
        columns = {}
        for key, value in (raw.get("columns") or {}).items():
            if isinstance(value, list):
                columns[str(key)] = [str(item) for item in value]
        hidden = [str(item) for item in (raw.get("hidden") or []) if isinstance(item, str)]
        return cls(columns=columns, hidden=hidden)


@dataclass
class Layout:
    """A named layout: one arrangement per workspace that has been touched."""

    name: str
    v: int = VERSION
    workspaces: dict[str, Arrangement] = field(default_factory=dict)
    #: A blob this build does not understand, kept exactly as it was found.
    opaque: Any = None

    @property
    def readable(self) -> bool:
        """Whether this build may apply it. See the module docstring."""

        return self.opaque is None

    def to_json(self) -> Any:
        if self.opaque is not None:
            return self.opaque
        return {
            "v": VERSION,
            "workspaces": {
                key: value.to_json() for key, value in sorted(self.workspaces.items())
            },
        }

    @classmethod
    def from_json(cls, name: str, raw: Any) -> Layout:
        if not isinstance(raw, dict):
            return cls(name=name)
        try:
            version = int(raw.get("v", VERSION))
        except (TypeError, ValueError):
            version = VERSION
        if version > VERSION:
            # Kept verbatim rather than coerced: an older build rewriting a
            # newer layout is the one way this feature can destroy something.
            return cls(name=name, v=version, opaque=raw)
        spaces = {
            str(key): Arrangement.from_json(value)
            for key, value in (raw.get("workspaces") or {}).items()
        }
        return cls(name=name, v=version, workspaces=spaces)


class Library:
    """Every saved layout, and which one is in use.

    Reads on construction and writes only when something changes -- **never on
    launch**: a start-up that rewrites the settings file is one that can lose
    it to a crash it had no reason to be exposed to.
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        raw = settings.get(LAYOUTS_KEY) or {}
        self.layouts: dict[str, Layout] = {}
        if isinstance(raw, dict):
            for name, blob in raw.items():
                self.layouts[str(name)] = Layout.from_json(str(name), blob)
        for name in BUILT_IN:
            self.layouts.setdefault(name, Layout(name=name))
        wanted = str(settings.get(ACTIVE_KEY) or BUILT_IN[0])
        self.active = wanted if wanted in self.layouts else BUILT_IN[0]

    # -- reading ------------------------------------------------------------

    def current(self) -> Layout:
        return self.layouts[self.active]

    def arrangement(self, workspace: str) -> Arrangement:
        """This layout's shape for one workspace, or an empty one."""

        return self.current().workspaces.get(workspace) or Arrangement()

    def order(self, workspace: str, column: str, builtin: list[str]) -> list[str]:
        """The slots of one column, in the order this layout wants them.

        Reconciled against the built-in list every time it is read and **never
        written back**: a retired pane disappears, a new one lands after its
        last already-placed predecessor rather than at the bottom, and a launch
        that changes nothing leaves the file alone.
        """

        from .layout_skeleton import reconcile

        if not self.current().readable:
            return list(builtin)
        stored = self.arrangement(workspace).columns.get(column)
        return list(builtin) if stored is None else reconcile(builtin, stored)

    def hidden(self, workspace: str) -> set[str]:
        if not self.current().readable:
            return set()
        return set(self.arrangement(workspace).hidden)

    # -- writing ------------------------------------------------------------

    def set_active(self, name: str) -> None:
        if name not in self.layouts:
            return
        self.active = name
        self._settings.set(ACTIVE_KEY, name)

    def record(self, workspace: str, columns: dict[str, list[str]], hidden: set[str]) -> None:
        """Store an arrangement for one workspace of the active layout."""

        layout = self.current()
        if not layout.readable:
            return
        layout.workspaces[workspace] = Arrangement(
            columns={key: list(value) for key, value in columns.items()},
            hidden=sorted(hidden),
        )
        self.save()

    def duplicate(self, name: str, into: str) -> bool:
        if name not in self.layouts or not into or into in self.layouts:
            return False
        source = self.layouts[name]
        self.layouts[into] = Layout.from_json(into, source.to_json())
        self.save()
        return True

    def rename(self, name: str, into: str) -> bool:
        """Rename a layout. **A built-in cannot be renamed**, because its name
        is what the reset commands and this docstring refer to."""

        if name in BUILT_IN or name not in self.layouts or not into or into in self.layouts:
            return False
        layout = self.layouts.pop(name)
        layout.name = into
        self.layouts[into] = layout
        if self.active == name:
            self.active = into
            self._settings.set(ACTIVE_KEY, into)
        self.save()
        return True

    def delete(self, name: str) -> bool:
        """Delete a layout. A built-in is **reset** instead of removed: there
        is no reachable state in which a pane cannot be got back."""

        if name not in self.layouts:
            return False
        if name in BUILT_IN:
            self.layouts[name] = Layout(name=name)
        else:
            del self.layouts[name]
            if self.active == name:
                self.set_active(BUILT_IN[0])
        self.save()
        return True

    def reset(self, name: str = "") -> None:
        """Put a layout back to the built-in arrangement."""

        wanted = name or self.active
        if wanted in self.layouts:
            self.layouts[wanted] = Layout(name=wanted)
            self.save()

    def save(self) -> None:
        self._settings.set(
            LAYOUTS_KEY,
            {name: layout.to_json() for name, layout in sorted(self.layouts.items())},
        )


def mirrored(columns: dict[str, list[str]]) -> dict[str, list[str]]:
    """Aseprite's second built-in: the two sidebars swapped.

    One function, and it is the proof the machinery works end to end -- a
    layout that is not the default, expressible in the same data, with nothing
    special about it anywhere else.
    """

    out = dict(columns)
    left, right = out.get("left"), out.get("right")
    if left is not None and right is not None:
        out["left"], out["right"] = list(right), list(left)
    return out
