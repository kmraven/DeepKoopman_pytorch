## uv

Codex sandbox may block `uv` when it tries to access `~/.cache/uv`.
Use a workspace-local cache instead.

Run uv commands as:

```bash
mkdir -p .cache/uv
UV_CACHE_DIR="$PWD/.cache/uv" uv run <command>
```
Do not use or request access to ~/.cache/uv.

Add to .gitignore:
```
.cache/uv/
.venv/```
