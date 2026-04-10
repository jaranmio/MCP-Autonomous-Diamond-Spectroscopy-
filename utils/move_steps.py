"""
Move one Thorlabs XA axis by an exact number of device units (relative IMove).

Uses the same connection, serial layout, and locking as ``mcp_servers.optics_server``
(``_get_hw``, ``_move_relative_device``). This is raw controller counts, not millidegrees.

From the repo root (no package install needed; ``uv`` uses this project's environment):

  uv run move_steps.py 1000
  uv run move_steps.py --n-steps 1000

Requires real hardware: never uses simulation (``OPTICS_SIMULATE`` must be unset, Windows, valid ``OPTICS_XA_SERIALS``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from mcp_servers import optics_server as opt
except ImportError:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from mcp_servers import optics_server as opt


def move_device_steps_forward(
    n_steps: int,
    *,
    mirror_index: int = 0,
    axis: str = "theta",
) -> None:
    """
    Relative move by ``n_steps`` device units on the selected hardware axis (positive = forward
    in the controller's sign convention). ``n_steps`` must be strictly positive.
    Refuses to run if optics_server would use simulation (no stub moves).
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer (forward move only).")
    if opt._use_simulation():
        raise RuntimeError(
            "move_steps requires Thorlabs hardware only; simulation is disabled for this tool. "
            "Unset OPTICS_SIMULATE, use 64-bit Windows, and set OPTICS_XA_SERIALS to "
            "NUM_MIRRORS or 2×NUM_MIRRORS serials (see optics_server)."
        )
    opt._validate_mirror_index(mirror_index)
    axis = axis.strip().lower()
    if axis not in ("theta", "phi"):
        raise ValueError("axis must be 'theta' or 'phi'.")
    axis_index = opt._axis_theta(mirror_index) if axis == "theta" else opt._axis_phi(mirror_index)

    hw = opt._get_hw()
    if hw is None:
        err = getattr(opt, "_hw_init_error", None) or "unknown error"
        raise RuntimeError(f"Thorlabs XA not available: {err}")

    device = hw._handle_per_axis[axis_index]
    if device is None:
        raise RuntimeError(
            f"No hardware channel for mirror {mirror_index} axis {axis!r} "
            "(software-only axis in theta-only layout)."
        )

    with hw._lock:
        hw._move_relative_device(device, int(n_steps))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Relative move by N device units on a Thorlabs XA axis "
            "(hardware only, same stack as optics_server; N is raw IMove counts, not mdeg)."
        )
    )
    p.add_argument(
        "n_steps_positional",
        nargs="?",
        type=int,
        help="Device steps (positive integer; forward in firmware sign).",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=None,
        dest="n_steps_flag",
        metavar="N",
        help="Same as positional N (either form may be used).",
    )
    p.add_argument(
        "--mirror-index",
        type=int,
        default=0,
        help="0-based mirror index (default 0).",
    )
    p.add_argument(
        "--axis",
        choices=("theta", "phi"),
        default="theta",
        help="Which mirror axis to move (default theta).",
    )
    args = p.parse_args(argv)

    n = args.n_steps_flag if args.n_steps_flag is not None else args.n_steps_positional
    if n is None:
        p.error("provide N as a positional argument or --n-steps N")
    if n <= 0:
        p.error("N must be a positive integer")

    try:
        move_device_steps_forward(n, mirror_index=args.mirror_index, axis=args.axis)
    except (RuntimeError, ValueError) as e:
        print(f"move_steps: {e}", file=sys.stderr)
        return 1
    print(f"OK: relative move by {n} device steps ({args.axis}, mirror {args.mirror_index}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
