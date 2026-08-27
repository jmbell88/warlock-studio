"""Sirens' engine: chiptune synthesis and the song document, with no window.

The ``inker``/``clay``/``plotter``/``packwright`` rule, fifth instance -- no
imgui, no moderngl, **no pygame** and no ``service`` anywhere under here, pinned
by ``tests/sirens/test_sirens_imports.py``. The reason is sharper for this
package than for the other four: the thing it produces is *audio*, and the one
piece of the app that needs a sound card is playback. Keeping the device out of
the engine is what lets a machine with no audio hardware at all -- CI, a
headless build box, a laptop with the driver uninstalled -- still open a song,
edit it, render it and export a WAV. ``studio/sirens_audio.py`` is the only
module in the repo that touches ``pygame.mixer``, and it is not in here.

The modules, in dependency order:

``notes``        note numbers, names and frequencies
``instruments``  the per-tick sequences an instrument is made of
``voices``       the oscillators, as pure functions over sample arrays
``document``     the song: channels, patterns, order, one-shots, history
``edits``        the reversible steps over it
``synth``        the tick loop that turns a document into samples
``wavout``       16-bit PCM WAV, with loop points
``wsng``         the ``.wsng`` container
"""

from __future__ import annotations
