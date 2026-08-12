"""Named Lucide glyphs.

Codepoints are Private-Use-Area assignments taken from lucide-static
**0.525.0**'s ``font/info.json`` and are only valid against the exact
``resources/fonts/lucide.ttf`` vendored beside them -- the project assigns
codepoints per release, so bumping the font means regenerating these numbers
from that release's info.json, never guessing them. A stale constant renders
as a missing-glyph box, which is cosmetic, not a crash.

The font is merged into every Inter face (:mod:`.fonts`), so an icon is just a
character in a string: ``f"{icons.STAR} Favourite"``.

**This is a catalogue, and an unreferenced constant here is not dead code.**
The numbers are transcribed from a named release's ``info.json`` and are only
ever regenerated wholesale from a later one, so a name with no call site today
costs nothing and is *correct* -- while re-deriving one by hand when a pane
finally wants it is precisely the guessing the paragraph above forbids. Do not
prune this list to what is currently drawn.
"""

from __future__ import annotations

ACTIVITY = "\uE038"  # activity
ARROW_LEFT = "\uE04C"  # arrow-left
BLEND = "\uE5A1"  # blend
BONE = "\uE35C"  # bone
BOOK_OPEN = "\uE063"  # book-open
BOX = "\uE065"  # box
BRUSH = "\uE1D3"  # brush
CAMERA = "\uE068"  # camera
CHECK = "\uE070"  # check
CHEVRON_DOWN = "\uE071"  # chevron-down
CHEVRON_RIGHT = "\uE073"  # chevron-right
CIRCLE = "\uE07A"  # circle
CIRCLE_ALERT = "\uE07B"  # circle-alert
CIRCLE_CHECK = "\uE226"  # circle-check
CIRCLE_X = "\uE088"  # circle-x
COPY = "\uE0A2"  # copy
CROP = "\uE0AF"  # crop
CROSSHAIR = "\uE0B0"  # crosshair
DOWNLOAD = "\uE0B6"  # download
EGG = "\uE25D"  # egg
ELLIPSIS = "\uE0BA"  # ellipsis
ERASER = "\uE28F"  # eraser
EYE = "\uE0BE"  # eye
EYE_OFF = "\uE0BF"  # eye-off
FILE_IMAGE = "\uE31D"  # file-image
FILE_JSON = "\uE36F"  # file-json
FILM = "\uE0D4"  # film
FLAG = "\uE0D5"  # flag
FLIP_HORIZONTAL = "\uE362"  # flip-horizontal-2
FOLDER_OPEN = "\uE247"  # folder-open
GRID = "\uE0ED"  # grid-3x3
HAND = "\uE1D7"  # hand
HEART = "\uE0F6"  # heart
HOUSE = "\uE0F9"  # house
IMAGE = "\uE0FA"  # image
INFO = "\uE0FF"  # info
KEYBOARD = "\uE284"  # keyboard
LASSO = "\uE1CE"  # lasso
LASSO_SELECT = "\uE1CF"  # lasso-select
LAYERS = "\uE52E"  # layers
LIST = "\uE10C"  # list
LOADER = "\uE10E"  # loader-circle
LOCK = "\uE10F"  # lock
LOCK_OPEN = "\uE110"  # lock-open
MAGNET = "\uE2B5"  # magnet
MAXIMIZE = "\uE116"  # maximize
MINUS = "\uE120"  # minus
MOVE = "\uE125"  # move
PAINT_BUCKET = "\uE2E6"  # paint-bucket
PALETTE = "\uE1DD"  # palette
PENCIL = "\uE1F9"  # pencil
PEN_TOOL = "\uE135"  # pen-tool
PERSON_STANDING = "\uE21E"  # person-standing
PIPETTE = "\uE13F"  # pipette
PLAY = "\uE140"  # play
PLUS = "\uE141"  # plus
POWER = "\uE144"  # power
RECTANGLE = "\uE37A"  # rectangle-horizontal
REDO = "\uE2A0"  # redo-2
REFRESH = "\uE149"  # refresh-cw
ROTATE_CW = "\uE14D"  # rotate-cw
RULER = "\uE14F"  # ruler
SAVE = "\uE151"  # save
SCALING = "\uE2EC"  # scaling
SEARCH = "\uE155"  # search
SEND = "\uE156"  # send
SETTINGS = "\uE158"  # settings
SHUFFLE = "\uE162"  # shuffle
SLASH = "\uE522"  # slash
SLIDERS = "\uE29A"  # sliders-horizontal
SPARKLES = "\uE417"  # sparkles
SPRAY_CAN = "\uE49A"  # spray-can
SQUARE = "\uE16B"  # square
SQUARE_DASHED = "\uE1CB"  # square-dashed
STAR = "\uE17A"  # star
TRASH = "\uE18E"  # trash-2
TRIANGLE_ALERT = "\uE193"  # triangle-alert
TYPE = "\uE198"  # type
UNDO = "\uE2A1"  # undo-2
UPLOAD = "\uE19E"  # upload
WAND = "\uE35B"  # wand-sparkles
WRENCH = "\uE1B1"  # wrench
X = "\uE1B2"  # x
ZOOM_IN = "\uE1B6"  # zoom-in
ZOOM_OUT = "\uE1B7"  # zoom-out
