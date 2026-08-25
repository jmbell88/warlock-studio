---
name: Bug report
about: Something behaved wrong
labels: bug
---

**What happened, and what you expected instead**

**Steps to reproduce**
1.
2.

**Diagnostics** — please paste the output of:
```powershell
uv run warlock doctor
```
It reports your GPU, VRAM, which models are present and which subsystems are
available, which is usually the whole answer.

**Log** — `~/.warlock/warlock.log`, and `~/.warlock/assets/trellis.log` if the
problem was a 3D reconstruction. Paste the relevant part rather than the file.

**Version** — shown next to the title on the Home screen.
