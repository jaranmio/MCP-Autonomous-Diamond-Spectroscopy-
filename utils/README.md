# `utils/` — CLI helpers for optics

Small **command-line wrappers** around `mcp_servers.optics_server`. Run them from the **repository root** so imports resolve.

```bash
uv run utils/<script>.py [args]
```

Same environment as the project (see root `pyproject.toml`). Alternatively: `.\.venv\Scripts\python.exe utils\<script>.py`.

**Hardware:** unset `OPTICS_SIMULATE`, use 64-bit Windows, set `OPTICS_XA_SERIALS` as in `optics_server` (read once at process start — restart the process after env changes).

---

## `sweep_mirror.py`

Calls `sweep_mirror()`: stepped moves between **min/max** from `get_mirror_limits` along chosen axes.

| Argument | Meaning | Default |
|----------|---------|---------|
| `--mirror-index` | 0-based mirror | `0` |
| `--step-mdeg` | Spacing between setpoints (mdeg) | `250` |
| `--settle-ms` | Wait after each move (ms) | optics `SETTLE_MS` |
| `--axes` | `theta` \| `phi` \| `both` | `theta` |

Examples:

```bash
uv run utils/sweep_mirror.py --mirror-index 0 --step-mdeg 250
uv run utils/sweep_mirror.py --axes both --settle-ms 300
```

Prints JSON result on success; exits `1` on error.

---

## `set_mirror_angle.py`

Calls `set_mirror_angle()`: **absolute** θ/φ in **millidegrees**.

| Argument | Meaning | Default |
|----------|---------|---------|
| `--mirror-index` | 0-based mirror | `0` |
| `--theta-mdeg` | Pitch | `0` |
| `--phi-mdeg` | Yaw | `0` |
| `--settle-ms` | Post-move wait (ms) | optics `SETTLE_MS` |

Examples:

```bash
uv run utils/set_mirror_angle.py --theta-mdeg 100 --phi-mdeg 0
uv run utils/set_mirror_angle.py --mirror-index 0 --settle-ms 300 --theta-mdeg 0 --phi-mdeg 0
```

---

## `move_steps.py`

Calls the same XA stack as optics, but **`IMove` relative steps in raw **device units** (encoder counts), **not** mdeg. **Hardware only** — refuses simulation.

| Argument | Meaning | Default |
|----------|---------|---------|
| `N` or `--n-steps N` | Signed step count (required); negative = opposite direction | — |
| `--mirror-index` | 0-based mirror | `0` |
| `--axis` | `theta` \| `phi` | `theta` |

Examples:

```bash
uv run utils/move_steps.py 1000
uv run utils/move_steps.py --n-steps -500
uv run utils/move_steps.py -- -1000
```

Use **`--n-steps -500`** (or **`-- -1000`** as the positional argument) for negative steps — a bare `-500` on the command line is parsed as a flag unless it comes after `--`.

**`-v` / `--verbose`** prints **mdeg readback** before/after the move (same axis as `--axis`). Use this if the controller **display** looks wrong: the box may show velocity, a different unit, or bounce near **travel limits** / stall recovery while the encoder position still moves monotonically.

The script calls **`_release_thorlabs_hw()`** on exit so the USB session closes cleanly; if you still see **FTDI / transport** errors, close **Kinesis** or any other app holding the same serial.

For mdeg-based moves, use optics MCP or the other utils.
