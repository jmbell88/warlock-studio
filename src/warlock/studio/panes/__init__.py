"""The panels. Each module draws one and owns none of the state it draws.

Every ``draw`` takes the app's :class:`~warlock.studio.app_ctx.Ctx` and reads
and writes through it, so the panes can be built in any order in a frame and
none of them can hold a stale copy of anything.
"""
