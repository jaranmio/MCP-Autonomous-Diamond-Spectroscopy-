# MCP-Autonomous-Diamond-Spectroscopy

Autonomous diamond color center spectral microscopy pipeline using the Model Context Protocol (MCP).

---

## Dev Environment Notes

### Why `uv` instead of `pip`

This project uses [`uv`](https://github.com/astral-sh/uv) as the package manager instead of plain `pip`. If you're reproducing this environment, use `uv` — it's faster and handles dependency tracking automatically.

The key practical difference:

| pip | uv |
|-----|----|
| `pip install mcp` | `uv add mcp` |
| `pip freeze > requirements.txt` (manual) | `pyproject.toml` updated automatically |
| `pip install -r requirements.txt` | `uv sync` |

With `uv`, every `uv add` writes the dependency into `pyproject.toml` immediately — you never have to remember to run `pip freeze`. Anyone cloning this repo can reproduce the exact environment with:

```bash
uv venv
source .venv/bin/activate
uv sync
```

`uv` is fully compatible with `pip` and the same PyPI ecosystem — it's just faster (written in Rust) and less manual. The `pyproject.toml` it generates is also readable by `pip` if needed.

---

### Folder name vs. Python package name mismatch

The root folder is named `MCP-Autonomous-Diamond-Spectroscopy-` (with uppercase letters and hyphens), which is not a valid Python package name. When initializing with `uv`, this causes an error:

```
error: The target directory is not a valid package name.
```

This was resolved by explicitly passing a valid package name at init time:

```bash
uv init . --name diamond-spectroscopy
```

The folder name and the `pyproject.toml` package name are **completely independent** — the folder name is just where files live, the package name is metadata. Your imports and code are unaffected by this mismatch.

---

## Quickstart

```bash
# Clone and enter the repo
git clone <repo-url>
cd MCP-Autonomous-Diamond-Spectroscopy-

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```
