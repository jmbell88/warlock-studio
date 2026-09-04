"""A take's envelope, and the one place time and pixels are converted.

**Why a waveform at all**, given ``panes/muse_results``' own docstring says "a
picture of a waveform tells a listener nothing that pressing play does not tell
them better". That argument is correct and it survives -- *for a card*. A card
is a thing you press.

A player's waveform is a different claim. It is not a picture but **the
coordinate system the other four controls are expressed in**: a playhead is
nowhere without it, a seek is a click on it, and a loop region is two positions
in it. The tray's line stands unedited; this is the distinction, stated where
the second one is drawn.

**Peaks, not samples.** Four minutes of 44.1 kHz stereo is 10.6 M frames and no
pane is 10.6 M pixels wide, so the display is a min/max per column. Computed
once, on the task thread that already reads the WAV; the pane's per-frame cost
is one :func:`window` over 4096 floats.
"""

from __future__ import annotations

import numpy as np

#: How many columns :func:`peaks` reduces a take to.
#:
#: More than any physical pane is wide -- a 4K display at 100% is under 4000
#: design pixels and the player is not full-screen -- so :func:`window` only
#: ever buckets *down*. That direction is exact: a column of the display is a
#: whole number of columns of this array, minimum of the mins and maximum of the
#: maxes, which is the same answer computing from the samples would have given.
#: Upsampling would not be, which is why the figure is generous rather than
#: tuned.
COLUMNS = 4096


def peaks(pcm: np.ndarray, columns: int = COLUMNS) -> np.ndarray:
    """The min and max of each block of ``pcm``. -> ``(2, columns)`` float32.

    ``pcm`` is ``(n,)`` or ``(n, channels)`` in any numeric dtype; a stereo take
    is downmixed, because the envelope is a picture of the piece and two
    overlaid channels at this scale are one slightly thicker picture.

    Scaled into [-1, 1] from the *dtype's* range rather than from the take's own
    peak. Normalising per take would make a quiet piece draw as loud as a loud
    one, so the picture would stop being comparable between two takes -- which
    is exactly what a tray of candidates is for.
    """
    data = np.asarray(pcm)
    if data.ndim == 2:
        data = data.mean(axis=1)
    elif data.ndim != 1:
        raise ValueError("pcm is (n,) or (n, channels)")

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        # Divided by the positive peak, matching ``wavout.to_int16``'s inverse
        # for the reason that function's own comment gives: the pair is exact.
        data = data.astype(np.float32) / float(max(info.max, 1))
    else:
        data = data.astype(np.float32)

    columns = max(1, int(columns))
    if data.size == 0:
        return np.zeros((2, columns), dtype=np.float32)

    # Padded to a whole number of blocks rather than truncated: dropping the
    # tail would draw a take as very slightly shorter than it is, and the
    # playhead -- which is placed by :func:`at`, from the true duration -- would
    # run past the end of its own picture.
    block = int(np.ceil(data.size / columns))
    padded = np.zeros(block * columns, dtype=np.float32)
    padded[: data.size] = data
    blocks = padded.reshape(columns, block)
    return np.stack([blocks.min(axis=1), blocks.max(axis=1)]).astype(np.float32)


def window(env: np.ndarray, width: int) -> np.ndarray:
    """``env`` reduced to ``width`` columns. -> ``(2, width)`` float32.

    The per-frame call. Exact whenever ``width <= env.shape[1]``, which
    :data:`COLUMNS` is chosen to guarantee -- see its docstring. A wider request
    is answered by repeating columns, which is honest about being a stretch
    rather than pretending to detail the array does not hold.
    """
    have = int(env.shape[1])
    width = max(1, int(width))
    if width >= have:
        index = np.minimum((np.arange(width) * have) // width, have - 1)
        return env[:, index].astype(np.float32)
    edges = (np.arange(width + 1) * have) // width
    lows = np.minimum.reduceat(env[0], edges[:-1])
    highs = np.maximum.reduceat(env[1], edges[:-1])
    return np.stack([lows, highs]).astype(np.float32)


def at(seconds: float, duration: float, width: float) -> float:
    """Where ``seconds`` falls across ``width`` pixels. -> a pixel offset.

    Here rather than at each of the four call sites, which is the whole reason
    this module carries a conversion at all: the playhead, the two loop markers
    and the click hit-test are four readings of one mapping, and four copies of
    ``x / duration * width`` disagree by half a column the first time one of
    them rounds differently.
    """
    if duration <= 0.0:
        return 0.0
    return float(np.clip(seconds / duration, 0.0, 1.0) * width)


def seconds_at(x: float, duration: float, width: float) -> float:
    """The inverse of :func:`at`: a pixel offset back to a time. -> seconds."""
    if width <= 0.0:
        return 0.0
    return float(np.clip(x / width, 0.0, 1.0) * duration)


__all__ = ["COLUMNS", "at", "peaks", "seconds_at", "window"]
