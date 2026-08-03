"""The ModernGL viewport: GLB loading, PBR shading, camera, gizmos, capture.

Nothing here imports imgui or pygame. The viewer renders into framebuffers and
answers questions about rays and bones; :mod:`warlock.studio.viewer_embed` is
what turns that into a panel the UI can show and click on. Keeping the split
means the whole renderer is testable against a standalone GL context with no
window at all -- which is what the golden-image tests do.
"""
