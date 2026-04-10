"""
CLI wrapper for ``mcp_servers.optics_server.sweep_mirror``.

From repo root:

  uv run utils/sweep_mirror.py --mirror-index 0 --step-mdeg 250
  uv run utils/sweep_mirror.py --axes both --settle-ms 200

Defaults match optics_server: ``step_mdeg=250``, ``settle_ms=SETTLE_MS``, ``axes=theta``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from mcp_servers import optics_server as opt
except ImportError:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from mcp_servers import optics_server as opt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=("Call optics_server.sweep_mirror: stepped sweep within reported limits.")
    )
    p.add_argument(
        "--mirror-index",
        type=int,
        default=0,
        help=f"0-based mirror index (default 0; valid 0..{opt.NUM_MIRRORS - 1}).",
    )
    p.add_argument(
        "--step-mdeg",
        type=float,
        default=250.0,
        help="Spacing between setpoints along each swept axis in mdeg (default 250).",
    )
    p.add_argument(
        "--settle-ms",
        type=int,
        default=opt.SETTLE_MS,
        metavar="MS",
        help=f"Wait after each absolute move in ms (default {opt.SETTLE_MS}).",
    )
    p.add_argument(
        "--axes",
        choices=("theta", "phi", "both"),
        default="theta",
        help="Which axis range(s) to sweep (default theta).",
    )
    args = p.parse_args(argv)

    try:
        out = opt.sweep_mirror(
            args.mirror_index,
            step_mdeg=args.step_mdeg,
            settle_ms=args.settle_ms,
            axes=args.axes,
        )
    except (ValueError, RuntimeError) as e:
        print(f"sweep_mirror: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
