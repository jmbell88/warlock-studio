# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Use GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (Security -> Report a vulnerability), which opens a private
advisory only the maintainers can see.

Expect an acknowledgement within a week. There is no bounty; this is one
person's project.

## What is in scope

Warlock Studio is an offline desktop application. It has no server, no account
system and no network listener beyond `127.0.0.1`, so the realistic threat is
**a malicious file**, not a malicious peer. In scope:

- **Any file the app opens.** `.ora`, `.aseprite`, `.tmx`/`.tsx`, `.wmap`,
  `.wblk`, `.wpack`, `.glb`, and every image format Pillow handles. These are
  files people download from asset sites, so a crafted one reaching code
  execution, a decompression bomb, or a write outside the chosen directory is a
  real finding. So is a hang or an unbounded allocation.
- **Path traversal** through any archive member, external tileset reference or
  export template.
- **Anything that makes the app reach the network.** The offline guarantee
  (`HF_HUB_OFFLINE=1`, set before any import) is a security property here, not
  only a convenience. The single exception is the user-initiated
  `fetch_worker` subprocess.
- **Subprocess handling** -- `trellis-server.exe`, the Blender worker, the
  matting worker, the fetch worker.

## What is not in scope

- **Model weights and what they generate.** The app runs whatever checkpoint you
  point it at, in-process, by design. A malicious `.safetensors` is a supply
  chain question about where you downloaded it from.
- **`WARLOCK_*` environment variables.** They are configuration, set by whoever
  is already running the process.
- Anything requiring an attacker who already has code execution as your user.
- The known, documented traversal allowance in `.tmx`/`.tsx` external
  references: Tiled's real folder layouts use `../`, so relative traversal is
  permitted deliberately while absolute and UNC paths are refused. It is a
  same-user read in an offline app; if you have a way to turn it into something
  more, that *is* in scope.

## Supported versions

The latest release only.
