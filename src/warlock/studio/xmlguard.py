"""The one XML door: a DTD refused by the parser, and a depth the stack survives.

Three doors in this repo parse XML a user was handed -- ``plotter/tsx.py``'s
``xml_root`` (which ``tmx.py`` shares), and ``inker/ora.py``'s ``_parse_stack``.
Both refused a DTD with the same four lines::

    if b"<!DOCTYPE" in data[:4096].upper():

and both said in a docstring why that was enough. **Both were bypassed by direct
execution on this machine**, two ways:

* **UTF-16.** The declaration encodes as ``<\\x00!\\x00D\\x00...``, so the byte
  substring never matches. expat honours the file's own
  ``encoding="UTF-16"`` and parses the entities happily.
* **A padded prolog.** A comment is legal prolog content, so five thousand
  bytes of ``<!-- ... -->`` put the declaration past byte 4096. The load-bearing
  sentence in both docstrings -- *"4 KiB is far past any prolog"* -- is the
  assumption that breaks, and it breaks for free.

So the refusal moved onto the **parser's own DOCTYPE event**, which is a
property of the parser object the way a bound is a property of
``zipguard.BoundedZip``: it fires for every encoding and at every offset,
because by then expat has decoded the document and found the declaration for
itself. Worth recording, because the obvious route does not exist:
``ElementTree.XMLParser`` does **not** expose its expat parser (no ``.parser``
attribute under the C accelerator), so there is nothing to hang a
``StartDoctypeDeclHandler`` on. What it does do is call ``doctype()`` on its
*target* -- so the target is where this lives.

**Depth, on the same target, for a failure that outlives the parse.**
``ET.fromstring`` will build a 20,001-deep tree out of a 300 KB file; Python's
recursion limit is 1000. Seven walkers in this repo then descend that tree, and
a document nested a few hundred deep is worse than one nested twenty thousand:
the shallow one *loads*, and the frame-thread walkers -- ``scene.resolve``,
``render``, the layers pane -- blow up once a frame afterwards, which is a
session that cannot be closed rather than a file that will not open. A ceiling
at the door is the only place that failure can be spent once.

Element and attribute counts get the same treatment, because expat's own
amplification limit (CPython >= 3.12.2) bounds only *entity expansion*: a merely
enormous element tree, or one element with a million attributes, is unbounded at
every XML door regardless, and ``requires-python = ">=3.13"`` admits builds
without even that limit.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

#: Deeper than any hand-built document and shallower than anything the walkers
#: cannot descend. Tiled group layers and ORA stacks both nest a handful deep in
#: practice; a hundred leaves the frame-thread recursions two orders of
#: headroom under a limit of 1000. Module-level so a test lowers it rather than
#: generating a megabyte of angle brackets.
MAX_DEPTH = 100

#: The backstop, not the primary bound: a ``.tmx`` is already held to
#: ``MAX_MAP_SOURCE_BYTES`` (100 MB) and ``stack.xml`` to the archive ceiling,
#: and at roughly eighteen bytes per ``<tile gid="1"/>`` neither can reach this.
#: It is here so that the element count is bounded by something that is not the
#: byte ceiling, and it sits above the largest tree a legitimate file produces:
#: a 2048-square map written in the one-element-per-cell form of ``<data>``.
MAX_ELEMENTS = 4_000_000

#: One element with a million attributes is a dictionary built one string at a
#: time, and no format here writes more than a few dozen.
MAX_ATTRIBUTES = 4096


class _Guarded(ET.TreeBuilder):
    """A tree builder that counts what it is being asked to build.

    Subclassing the builder rather than post-walking the tree, because the
    point is to refuse *before* the allocation: a depth check run over a
    finished ``Element`` has already paid for every node in it, and the
    20,001-deep case never returns a tree to walk.
    """

    def __init__(self, what: str) -> None:
        super().__init__()
        self._what = what
        self._depth = 0
        self._elements = 0

    def start(self, tag, attrs):  # type: ignore[override]
        self._depth += 1
        self._elements += 1
        if self._depth > MAX_DEPTH:
            raise ValueError(
                f"{self._what} nests elements more than {MAX_DEPTH} deep, which is"
                " deeper than this build reads"
            )
        if self._elements > MAX_ELEMENTS:
            raise ValueError(
                f"{self._what} holds more than {MAX_ELEMENTS} elements, which is"
                " more than this build reads"
            )
        if len(attrs) > MAX_ATTRIBUTES:
            raise ValueError(
                f"an element in {self._what} carries more than {MAX_ATTRIBUTES}"
                " attributes"
            )
        return super().start(tag, attrs)

    def end(self, tag):  # type: ignore[override]
        self._depth -= 1
        return super().end(tag)

    def doctype(self, _name, _pubid, _system):
        """The DTD refusal, on the event rather than on a substring.

        ``ExpatParser`` expands internal entities, so the billion-laughs shape
        -- ten nested entities each referencing the previous one ten times --
        turns a few hundred bytes of file into gigabytes of string *inside* the
        parse, where the byte ceiling above has already had its turn and every
        ceiling below has not had one yet. No writer of any format this app
        opens emits a DTD, so nothing legitimate is refused.
        """
        raise ValueError(f"{self._what} declares a DTD, which this build does not read")


def fromstring(data: bytes, what: str) -> ET.Element:
    """``ET.fromstring`` with the guards above wired to it.

    ``ET.ParseError`` is translated because it is a ``SyntaxError`` subclass
    rather than a ``ValueError``, and every door here is written against
    ``ValueError`` as the shape a refusal arrives in.
    """
    builder = _Guarded(what)
    parser = ET.XMLParser(target=builder)
    try:
        parser.feed(data)
        root = parser.close()
    except ET.ParseError as exc:
        raise ValueError(f"{what} is not a readable XML document: {exc}") from exc
    if root is None:
        raise ValueError(f"{what} carries no XML document")
    return root
