"""
CLI wrapper for ``mcp_servers.optics_server.set_mirror_angle`` (absolute theta/phi in mdeg).

From repo root:

  uv run utils/set_mirror_angle.py --theta-mdeg 100 --phi-mdeg 0
  uv run utils/set_mirror_angle.py --mirror-index 0 --settle-ms 300 --theta-mdeg 0 --phi-mdeg 0

Defaults: ``mirror_index=0``, ``theta_mdeg=0``, ``phi_mdeg=0``, ``settle_ms`` = optics_server.SETTLE_MS.
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
        description=(
            "Call optics_server.set_mirror_angle: absolute mirror tilt (theta, phi) in millidegrees."
        )
    )
    p.add_argument(
        "--mirror-index",
        type=int,
        default=0,
        help=f"0-based mirror index (default 0; valid 0..{opt.NUM_MIRRORS - 1}).",
    )
    p.add_argument(
        "--theta-mdeg",
        type=float,
        default=0.0,
        help="Vertical/pitch angle in millidegrees (default 0).",
    )
    p.add_argument(
        "--phi-mdeg",
        type=float,
        default=0.0,
        help="Horizontal/yaw angle in millidegrees (default 0).",
    )
    p.add_argument(
        "--settle-ms",
        type=int,
        default=opt.SETTLE_MS,
        metavar="MS",
        help=f"Post-move settle time in ms (default {opt.SETTLE_MS}).",
    )
    args = p.parse_args(argv)

    try:
        out = opt.set_mirror_angle(
            args.mirror_index,
            args.theta_mdeg,
            args.phi_mdeg,
            settle_ms=args.settle_ms,
        )
    except (ValueError, RuntimeError) as e:
        print(f"set_mirror_angle: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
