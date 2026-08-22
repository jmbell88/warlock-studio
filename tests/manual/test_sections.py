"""The sub-section machinery behind the TOC tree, headlessly.

The TOC listed chapters and nothing else, so the only way to reach a section of
a 900-line chapter was to open it and scroll. The tree that fixes that needs
three decisions -- which headings a chapter offers, which one you are currently
looking at, and which ones a search matched -- and all three are pure, so they
are asserted here rather than through a frame.
"""

from warlock.studio.manual import loader, parser

CHAPTER = """# Inker

Intro prose.

## Tools

The brush.

### Image brushes

A stamp.

## Layers

Groups and such.
"""


def test_sections_are_the_headings_under_the_title():
    """Level 2 and 3, in source order. The H1 is the chapter and is not one of
    its own sections -- it is already the row the tree hangs from."""
    found = loader.sections(parser.parse(CHAPTER))
    assert [(s.level, s.title, s.anchor) for s in found] == [
        (2, "Tools", "tools"),
        (3, "Image brushes", "image-brushes"),
        (2, "Layers", "layers"),
    ]


def test_sections_of_a_chapter_with_no_headings_is_empty():
    assert loader.sections(parser.parse("# Bare\n\nJust prose.\n")) == []


def test_section_anchors_match_the_ones_links_resolve_against():
    """The tree navigates by anchor, and ``test_cross_links_resolve`` checks
    links against the same slugs -- so a tree row that could not be linked to
    would be a row that scrolls nowhere."""
    blocks = parser.parse(CHAPTER)
    headings = {b.anchor for b in blocks if isinstance(b, parser.Heading)}
    assert {s.anchor for s in loader.sections(blocks)} <= headings


class TestActiveAnchor:
    """Which section the reader is in, from the scroll position.

    Content-space offsets: imgui's ``get_cursor_pos_y`` adds the scroll back,
    so a heading's recorded y is stable across frames and compares directly
    against ``get_scroll_y``.
    """

    OFFSETS = (("tools", 100.0), ("image-brushes", 260.0), ("layers", 400.0))

    def test_above_the_first_heading_is_no_section(self):
        assert loader.active_anchor(self.OFFSETS, 0.0) is None
        assert loader.active_anchor(self.OFFSETS, 99.0) is None

    def test_the_heading_you_have_scrolled_to_is_active(self):
        assert loader.active_anchor(self.OFFSETS, 100.0) == "tools"
        assert loader.active_anchor(self.OFFSETS, 150.0) == "tools"

    def test_it_is_the_last_heading_above_the_line_not_the_nearest(self):
        """A long section keeps its row lit all the way down, rather than
        handing over to the next one at the halfway point."""
        assert loader.active_anchor(self.OFFSETS, 399.0) == "image-brushes"
        assert loader.active_anchor(self.OFFSETS, 400.0) == "layers"
        assert loader.active_anchor(self.OFFSETS, 99999.0) == "layers"

    def test_no_headings_is_no_section_rather_than_an_error(self):
        assert loader.active_anchor((), 500.0) is None

    def test_offsets_out_of_order_do_not_confuse_it(self):
        """The renderer appends in draw order, which is source order -- but the
        function sorts rather than trusting it, because a cache rebuilt
        mid-scroll can hand over a partial list."""
        jumbled = (("layers", 400.0), ("tools", 100.0))
        assert loader.active_anchor(jumbled, 150.0) == "tools"


class TestMatchingSections:
    def test_a_search_returns_the_sections_that_contain_it(self):
        found = loader.matching_sections("groups", parser.parse(CHAPTER))
        assert [s.title for s in found] == ["Layers"]

    def test_a_heading_matches_on_its_own_title(self):
        found = loader.matching_sections("image", parser.parse(CHAPTER))
        assert [s.title for s in found] == ["Image brushes"]

    def test_prose_before_the_first_heading_matches_no_section(self):
        """It belongs to the chapter, not to any section under it -- listing it
        under whichever heading happens to follow would send the reader past
        the text they searched for."""
        assert loader.matching_sections("intro", parser.parse(CHAPTER)) == []

    def test_a_miss_is_empty(self):
        assert loader.matching_sections("nothing here", parser.parse(CHAPTER)) == []

    def test_the_search_is_case_insensitive(self):
        assert loader.matching_sections("LAYERS", parser.parse(CHAPTER))

    def test_a_hit_lights_the_innermost_section_not_its_parents(self):
        """"stamp" is prose under ``### Image brushes``, which is inside
        ``## Tools``. Listing both would put the parent of every hit beside it
        -- half the rows scrolling somewhere the word does not appear."""
        found = loader.matching_sections("stamp", parser.parse(CHAPTER))
        assert [s.title for s in found] == ["Image brushes"]

    def test_a_section_matched_by_a_subsection_is_not_listed_twice(self):
        """"brush" is in the Tools prose *and* in the Image brushes heading
        under it. Two rows, one per heading -- not Tools twice."""
        found = loader.matching_sections("brush", parser.parse(CHAPTER))
        assert [s.title for s in found] == ["Tools", "Image brushes"]


class TestScrollReset:
    """Arriving at a chapter puts you at the top of it.

    The page is one imgui child with one id, so its scroll offset outlives the
    chapter drawn into it: opening a short chapter while scrolled deep into a
    long one landed the reader at the short one's *footer*, past everything it
    said. Only a screenshot showed it -- the chapter drew correctly, at the
    wrong offset -- and the TOC tree made it constant rather than occasional,
    because a tree is a thing people click through.
    """

    def test_a_new_chapter_starts_at_the_top(self):
        assert loader.needs_scroll_reset("28-inker", "30-clay", None) is True

    def test_staying_in_a_chapter_keeps_your_place(self):
        """Every frame is a redraw of the same chapter, so answering True here
        would pin the page to the top and make it unscrollable."""
        assert loader.needs_scroll_reset("28-inker", "28-inker", None) is False

    def test_an_anchor_does_its_own_scrolling(self):
        """``set_scroll_here_y`` fires on the heading as it draws. Resetting to
        the top as well would fight it for the same frame."""
        assert loader.needs_scroll_reset("28-inker", "30-clay", "materials") is False
        assert loader.needs_scroll_reset("28-inker", "28-inker", "layers") is False

    def test_the_first_chapter_of_a_session_starts_at_the_top(self):
        """There is no previous chapter, and the page has no scroll to keep."""
        assert loader.needs_scroll_reset("", "00-index", None) is True


def test_every_real_chapter_offers_sections():
    """A chapter with no ``##`` at all would be a tree row that never expands.

    00-index is the exception by construction: its headings are the parts, and
    it is the one 'chapter' that is a table of contents rather than a chapter.
    """
    bare = [
        c.key
        for c in loader.chapters()
        if c.key != "00-index" and not loader.sections(parser.parse(loader.load(c.key)))
    ]
    assert bare == []
