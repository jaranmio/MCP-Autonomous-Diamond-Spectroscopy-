"""
Move one Thorlabs XA axis by an exact number of device units (relative IMove).

Uses the same connection, serial layout, and locking as ``mcp_servers.optics_server``
(``_get_hw``, ``_move_relative_device``). This is raw controller counts, not millidegrees.

From the repo root (no package install needed; ``uv`` uses this project's environment):

  uv run utils/move_steps.py 1000
  uv run utils/move_steps.py --n-steps -500
  uv run utils/move_steps.py -- -1000

Requires real hardware: never uses simulation (``OPTICS_SIMULATE`` must be unset, Windows, valid ``OPTICS_XA_SERIALS``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from mcp_servers import optics_server as opt
except ImportError:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from mcp_servers import optics_server as opt


def move_device_steps(
    n_steps: int,
    *,
    mirror_index: int = 0,
    axis: str = "theta",
    verbose: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Relative move by ``n_steps`` device units on the selected hardware axis (sign = direction
    in the controller's convention). ``n_steps`` must be non-zero.
    Refuses to run if optics_server would use simulation (no stub moves).

    If ``verbose``, returns (theta_or_phi_mdeg_before, theta_or_phi_mdeg_after) for that hardware
    axis so you can confirm direction; otherwise (None, None).
    """
    if n_steps == 0:
        raise ValueError("n_steps must be non-zero.")
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

    before: Optional[float] = None
    after: Optional[float] = None
    with hw._lock:
        if verbose:
            before = float(hw.read_axis_mdeg(axis_index))
        hw._move_relative_device(device, int(n_steps))
        if verbose:
            after = float(hw.read_axis_mdeg(axis_index))
    return (before, after)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Relative move by N device units on a Thorlabs XA axis "
            "(hardware only, same stack as optics_server; N is raw IMove counts, not mdeg). "
            "N may be negative to move in the opposite direction."
        )
    )
    p.add_argument(
        "n_steps_positional",
        nargs="?",
        type=int,
        help="Device steps (signed integer; use 'utils/move_steps.py -- -100' if N is negative).",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=None,
        dest="n_steps_flag",
        metavar="N",
        help="Same as positional N; use this form for negative N (e.g. --n-steps -500).",
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
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print axis position in mdeg before and after the move (same readback as optics_server).",
    )
    args = p.parse_args(argv)

    n = args.n_steps_flag if args.n_steps_flag is not None else args.n_steps_positional
    if n is None:
        p.error("provide N as a positional argument or --n-steps N")
    if n == 0:
        p.error("N must be non-zero")

    exit_code = 0
    try:
        try:
            before, after = move_device_steps(
                n,
                mirror_index=args.mirror_index,
                axis=args.axis,
                verbose=args.verbose,
            )
        except (RuntimeError, ValueError) as e:
            print(f"move_steps: {e}", file=sys.stderr)
            exit_code = 1
        else:
            if args.verbose and before is not None and after is not None:
                print(
                    f"axis {args.axis!r} mdeg: {before:.6g} -> {after:.6g} (delta {after - before:+.6g})",
                    file=sys.stderr,
                )
            print(f"OK: relative move by {n} device steps ({args.axis}, mirror {args.mirror_index}).")
    finally:
        # Short-lived CLI exits without optics_server's atexit; release USB so the native stack
        # does not log FTDI / "Transport not available" during interpreter teardown.
        try:
            opt._release_thorlabs_hw()
        except Exception:
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
