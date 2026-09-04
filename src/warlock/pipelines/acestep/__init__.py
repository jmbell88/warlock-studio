"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

# WARLOCK 3/5: see ATTRIBUTION.md. Upstream imports itself absolutely
# (`from acestep.models... import ...`), which cannot resolve from
# `warlock.pipelines.acestep`. Registering this package under the bare name it
# expects makes every one of those imports resolve to these files, and is the
# one alternative to rewriting the import line at the top of a dozen vendored
# modules -- which would make each of them undiffable against upstream, for no
# gain. Only the music worker subprocess ever imports this package.
import sys as _sys

_sys.modules.setdefault("acestep", _sys.modules[__name__])
