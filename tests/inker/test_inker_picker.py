"""The colour picker's arithmetic, which is everything about it worth pinning.

The sliders themselves are imgui and are walked by ``tests/test_studio_smoke``.
What is asserted here is what the pane *decides*: which colour it is pointed
at, whether that is a palette entry or a free colour, and what a typed hex
string means -- a hex field that silently ignores what was typed looks exactly
like one that is not wired up.
"""

from __future__ import annotations

import colorsys

from warlock.studio.panes import inker_picker


class _Doc:
    def __init__(self, palette=None, indexed=False):
        self.palette = list(palette or [])
        self.is_indexed = indexed
        self.recoloured: list[tuple[int, tuple]] = []

    def recolour_slot(self, index, colour):
        self.recoloured.append((index, tuple(colour)))
        self.palette[index] = tuple(colour)
        return True


class _Tab:
    def __init__(self, doc):
        self.doc = doc


class _State:
    def __init__(self, **kwargs):
        self.fg = (10, 20, 30, 255)
        self.bg = (255, 255, 255, 255)
        self.fg_slot = None
        self.picker_target = "fg"
        self.palette_usage = "stale"
        self.active = None
        self.__dict__.update(kwargs)

    def set_fg(self, colour, slot=None):
        self.fg = tuple(int(c) for c in tuple(colour)[:4])
        self.fg_slot = None if slot is None else int(slot)


def test_the_target_defaults_to_the_foreground_and_survives_nonsense():
    assert inker_picker.target_of(_State()) == "fg"
    assert inker_picker.target_of(_State(picker_target="bg")) == "bg"
    assert inker_picker.target_of(_State(picker_target="sideways")) == "fg"


def test_only_an_indexed_document_holding_a_slot_edits_the_entry():
    """Three separate conditions, and each one alone is not enough.

    The brush can be holding a slot in a palette-constrained *RGB* document,
    where the pixels are colours and editing the table would not move them;
    and the background never came from a slot, because ``set_fg`` is the one
    door that records one.
    """
    palette = [(0, 0, 0, 255), (255, 0, 0, 255)]
    indexed = _Tab(_Doc(palette, indexed=True))
    rgb = _Tab(_Doc(palette, indexed=False))

    assert inker_picker.slot_of(_State(fg_slot=1), indexed) == 1
    assert inker_picker.slot_of(_State(fg_slot=None), indexed) is None
    assert inker_picker.slot_of(_State(fg_slot=1), rgb) is None
    assert inker_picker.slot_of(_State(fg_slot=1, picker_target="bg"), indexed) is None
    assert inker_picker.slot_of(_State(fg_slot=1), None) is None
    # A slot the palette no longer has -- a Remove between the click and this
    # frame -- reads as no slot rather than as an IndexError.
    assert inker_picker.slot_of(_State(fg_slot=9), indexed) is None


def test_writing_a_slot_repaints_the_entry_and_keeps_the_brush_on_it():
    state = _State(fg_slot=1)
    tab = _Tab(_Doc([(0, 0, 0, 255), (255, 0, 0, 255)], indexed=True))
    inker_picker.write(None, state, tab, 1, (0, 128, 255, 255))
    assert tab.doc.recoloured == [(1, (0, 128, 255, 255))]
    # Still holding slot 1: the picker changed what the slot *is*, not which
    # slot the brush came from.
    assert state.fg_slot == 1
    assert state.fg == (0, 128, 255, 255)
    # And the cached usage histogram is dropped, the way every other palette
    # write in the app drops it.
    assert state.palette_usage is None


def test_a_free_write_clamps_rather_than_handing_the_engine_a_bad_channel():
    """The HSV round trip lands a shade over 255 often enough to matter."""
    state = _State()
    inker_picker.write(None, state, None, None, (255.4, -3.0, 40.6, 255))
    assert state.fg == (255, 0, 41, 255)


def test_reading_a_short_colour_still_answers_four_channels():
    """A palette entry stored as a triple is a real shape in this codebase."""
    tab = _Tab(_Doc([(1, 2, 3)], indexed=True))
    assert inker_picker.read(_State(fg_slot=0), tab, 0) == (1, 2, 3, 255)


def test_the_hex_field_takes_every_shape_a_palette_is_written_down_in():
    assert inker_picker.parse_hex("#3b4252") == (0x3B, 0x42, 0x52, 255)
    assert inker_picker.parse_hex("3b4252") == (0x3B, 0x42, 0x52, 255)
    assert inker_picker.parse_hex("  #3B4252  ") == (0x3B, 0x42, 0x52, 255)
    assert inker_picker.parse_hex("f00") == (255, 0, 0, 255)
    assert inker_picker.parse_hex("f008") == (255, 0, 0, 0x88)
    assert inker_picker.parse_hex("3b425280") == (0x3B, 0x42, 0x52, 0x80)


def test_a_half_typed_or_junk_hex_is_ignored_rather_than_guessed():
    # ``3b42`` is *not* junk -- it is the four-digit shorthand, and reading it
    # as one is the whole reason that branch exists.
    for text in ("", "#", "3b425", "3b42525", "zzzzzz", "3b 42 52"):
        assert inker_picker.parse_hex(text) is None, text


def test_the_grey_of_a_colour_is_the_same_luma_the_palette_sorts_by():
    """Rec. 601, and the weights sum to one -- a Gray tab whose weights did not
    would darken or lighten every colour it was asked about."""
    assert inker_picker.LUMA == (0.299, 0.587, 0.114)
    assert sum(inker_picker.LUMA) == 1.0

    def grey(colour):
        pairs = zip(inker_picker.LUMA, colour, strict=True)
        return inker_picker.clamp8(sum(weight * channel for weight, channel in pairs))

    assert grey((255, 255, 255)) == 255
    assert grey((0, 0, 0)) == 0
    assert grey((128, 128, 128)) == 128


def test_the_two_wheel_spaces_round_trip_through_colorsys():
    """HSL is HLS with two arguments swapped, and getting that backwards is a
    picker whose Saturation slider changes the lightness."""
    for colour in ((10, 200, 90), (255, 0, 0), (128, 128, 128), (0, 0, 0)):
        r, g, b = (channel / 255.0 for channel in colour)
        hue, sat, light = inker_picker._to_hsl(r, g, b)
        assert inker_picker._from_hsl(hue, sat, light) == colorsys.hls_to_rgb(hue, light, sat)
        back = tuple(round(channel * 255) for channel in inker_picker._from_hsl(hue, sat, light))
        assert back == colour
