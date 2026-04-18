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

### Folder name vs. Python package name difference

Top-Level python package name is mcp-diamond-spectroscopy (different from top-level folder name)

---

### Not using Colab anymore, runs loally

The file "Update of Automated optical alignment using digital twins.ipynb" initially ran on google server to use colab functionalities like "drive.mount('/content/drive')". To make the project more self-contained, I changed that in favor of accessing files locally (from presumably cloned repo).

---

## Power meter and VISA

Instrument I/O for the Thorlabs PM goes through **PyVISA**; the Python package depends on a **VISA implementation on the OS** (commonly **NI-VISA**) for USB/LAN discovery and sessions.  Install that vendor stack on the bench PC in addition to `uv sync`.

**NI-VISA and Thorlabs Optical Power Monitor** do not always coexist cleanly: installing or reinstalling one can break the other.  After NI-VISA is in place, **do not** reinstall Thorlabs Optical Power Monitor as a troubleshooting step—it is a common way to end up with both broken.  Prefer a stable order of installation and minimal churn.

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
