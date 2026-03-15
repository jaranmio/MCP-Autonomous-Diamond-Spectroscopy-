# MCP-Autonomous-Diamond-Spectroscopy

Autonomous diamond color center spectral microscopy pipeline using the Model Context Protocol (MCP).

---

## Dev Environment Notes

### Why `uv` instead of `pip`

This project uses [`uv`](https://github.com/astral-sh/uv) as the package manager instead of plain `pip`. If you're reproducing this environment, use `uv` — it's faster and handles dependency tracking automatically.


```bash
uv venv
source .venv/bin/activate
uv sync
```

`uv` is fully compatible with `pip` and the same PyPI ecosystem — it's just faster (written in Rust) and less manual.

---

### Folder name vs. Python package name mismatch

Top-Level python package name is mcp-diamond-spectroscopy

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
