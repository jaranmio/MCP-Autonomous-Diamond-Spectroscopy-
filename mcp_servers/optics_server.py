"""
optics_server.py
----------------
MCP server for tilt-mirror actuator control in the diamond spectroscopy setup.

Each mirror has two angular degrees of freedom:
  - theta : vertical tilt   (pitch)
  - phi   : horizontal tilt (yaw)

Mirror indexing is 0-based and consistent across all tools.
All angles are in millidegrees (mdeg) to allow fine-grained control.

Exposed tools (for LLM use):
  1. get_mirror_state       - read current (theta, phi) of one mirror
  2. get_all_mirrors_state  - read (theta, phi) of every mirror at once
  3. set_mirror_angle       - set absolute (theta, phi) of one mirror
  4. set_all_mirrors_angles - set absolute (theta, phi) of every mirror at once
  5. step_mirror_angle      - apply a relative delta to one mirror
  6. step_all_mirrors_angles- apply relative deltas to every mirror at once
  7. get_mirror_limits      - query the safe angle bounds for one mirror
  8. home_mirror            - move one mirror to its home position (0, 0)
  9. home_all_mirrors       - move all mirrors to home position (0, 0)
"""

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import List
import time

# ---------------------------------------------------------------------------
# Configuration – edit these to match your hardware
# ---------------------------------------------------------------------------

NUM_MIRRORS = 2          # number of tilt mirrors in the beam path
SETTLE_MS   = 200        # default settle time after a move (milliseconds)

# Per-mirror angle limits in millidegrees
THETA_MIN = -5_000.0   # mdeg
THETA_MAX =  5_000.0   # mdeg
PHI_MIN   = -5_000.0   # mdeg
PHI_MAX   =  5_000.0   # mdeg

# ---------------------------------------------------------------------------
# Internal state (replace with real actuator SDK calls)
# ---------------------------------------------------------------------------

_mirror_states: List[dict] = [
    {"theta": 0.0, "phi": 0.0} for _ in range(NUM_MIRRORS)
]


def _validate_mirror_index(mirror_index: int) -> None:
    if not (0 <= mirror_index < NUM_MIRRORS):
        raise ValueError(
            f"mirror_index {mirror_index} is out of range. "
            f"Valid indices are 0 to {NUM_MIRRORS - 1}."
        )


def _validate_angles(theta: float, phi: float) -> None:
    if not (THETA_MIN <= theta <= THETA_MAX):
        raise ValueError(
            f"theta={theta} mdeg is outside safe limits "
            f"[{THETA_MIN}, {THETA_MAX}] mdeg."
        )
    if not (PHI_MIN <= phi <= PHI_MAX):
        raise ValueError(
            f"phi={phi} mdeg is outside safe limits "
            f"[{PHI_MIN}, {PHI_MAX}] mdeg."
        )


def _move_mirror(mirror_index: int, theta: float, phi: float, settle_ms: int) -> None:
    """
    LOW-LEVEL: Send move command to the physical actuator and wait to settle.
    Replace the body of this function with your actual SDK / serial call.
    """
    # TODO: replace with real actuator SDK call, e.g.:
    #   actuator.set_position(mirror_index, axis="theta", value=theta)
    #   actuator.set_position(mirror_index, axis="phi",   value=phi)
    _mirror_states[mirror_index]["theta"] = theta
    _mirror_states[mirror_index]["phi"]   = phi
    time.sleep(settle_ms / 1000.0)


def _read_mirror(mirror_index: int) -> dict:
    """
    LOW-LEVEL: Read current angles from the physical actuator.
    Replace with your actual SDK readback call.
    """
    # TODO: replace with real actuator SDK readback, e.g.:
    #   theta = actuator.get_position(mirror_index, axis="theta")
    #   phi   = actuator.get_position(mirror_index, axis="phi")
    return dict(_mirror_states[mirror_index])


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="optics_server",
    instructions=(
        "Controls tilt mirrors in the diamond spectroscopy beam path. "
        "Each mirror is identified by a 0-based integer index (0 to "
        f"{NUM_MIRRORS - 1}). "
        "Angles are always in millidegrees (mdeg). "
        "Use get_mirror_state or get_all_mirrors_state to read positions "
        "before making moves. "
        "Use set_mirror_angle or set_all_mirrors_angles for absolute moves. "
        "Use step_mirror_angle or step_all_mirrors_angles for relative moves. "
        "Always call get_mirror_limits before large moves to avoid hitting "
        "hardware bounds. "
        "Call home_mirror or home_all_mirrors to reset to (0, 0)."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1 – read one mirror
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Read the current tilt angles of a single mirror. "
        "Returns theta (vertical/pitch) and phi (horizontal/yaw) in millidegrees. "
        "Use this before any move to know the current position. "
        "mirror_index is 0-based."
    )
)
def get_mirror_state(
    mirror_index: int = Field(
        ...,
        description=f"0-based mirror index. Must be between 0 and {NUM_MIRRORS - 1}.",
    )
) -> dict:
    """
    Returns: {"mirror_index": int, "theta_mdeg": float, "phi_mdeg": float}
    """
    _validate_mirror_index(mirror_index)
    state = _read_mirror(mirror_index)
    return {
        "mirror_index": mirror_index,
        "theta_mdeg":   state["theta"],
        "phi_mdeg":     state["phi"],
    }


# ---------------------------------------------------------------------------
# Tool 2 – read all mirrors
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Read the current tilt angles of ALL mirrors in one call. "
        "Returns a list of {mirror_index, theta_mdeg, phi_mdeg} objects. "
        "Preferred over calling get_mirror_state repeatedly when you need "
        "a snapshot of the full optical path."
    )
)
def get_all_mirrors_state() -> List[dict]:
    """
    Returns: list of {"mirror_index": int, "theta_mdeg": float, "phi_mdeg": float}
    """
    return [
        {
            "mirror_index": i,
            "theta_mdeg":   _read_mirror(i)["theta"],
            "phi_mdeg":     _read_mirror(i)["phi"],
        }
        for i in range(NUM_MIRRORS)
    ]


# ---------------------------------------------------------------------------
# Tool 3 – absolute move, one mirror
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Set the absolute tilt angles of a single mirror. "
        "Both theta (vertical/pitch) and phi (horizontal/yaw) must be provided "
        "in millidegrees. "
        "The mirror will move and then wait settle_ms milliseconds before "
        "this tool returns. "
        "Raises an error if angles exceed hardware limits — call "
        "get_mirror_limits first if unsure. "
        "mirror_index is 0-based."
    )
)
def set_mirror_angle(
    mirror_index: int = Field(
        ...,
        description=f"0-based mirror index. Must be between 0 and {NUM_MIRRORS - 1}.",
    ),
    theta_mdeg: float = Field(
        ...,
        description=(
            f"Target vertical tilt (pitch) in millidegrees. "
            f"Safe range: [{THETA_MIN}, {THETA_MAX}] mdeg."
        ),
    ),
    phi_mdeg: float = Field(
        ...,
        description=(
            f"Target horizontal tilt (yaw) in millidegrees. "
            f"Safe range: [{PHI_MIN}, {PHI_MAX}] mdeg."
        ),
    ),
    settle_ms: int = Field(
        SETTLE_MS,
        description=(
            "Milliseconds to wait after the move for the mirror to mechanically "
            "settle before returning. Default is "
            f"{SETTLE_MS} ms."
        ),
    ),
) -> dict:
    """
    Returns: {"mirror_index": int, "theta_mdeg": float, "phi_mdeg": float, "settled_ms": int}
    """
    _validate_mirror_index(mirror_index)
    _validate_angles(theta_mdeg, phi_mdeg)
    _move_mirror(mirror_index, theta_mdeg, phi_mdeg, settle_ms)
    return {
        "mirror_index": mirror_index,
        "theta_mdeg":   theta_mdeg,
        "phi_mdeg":     phi_mdeg,
        "settled_ms":   settle_ms,
    }


# ---------------------------------------------------------------------------
# Tool 4 – absolute move, all mirrors
# ---------------------------------------------------------------------------

class MirrorTarget(BaseModel):
    mirror_index: int   = Field(..., description="0-based mirror index.")
    theta_mdeg:   float = Field(..., description="Target vertical tilt (pitch) in mdeg.")
    phi_mdeg:     float = Field(..., description="Target horizontal tilt (yaw) in mdeg.")


@mcp.tool(
    description=(
        "Set the absolute tilt angles of ALL mirrors in a single call. "
        "Accepts a list of {mirror_index, theta_mdeg, phi_mdeg} objects — "
        "you must include every mirror you want to move. "
        "Mirrors are moved sequentially in index order. "
        "Use this instead of calling set_mirror_angle in a loop to reduce "
        "round-trip latency. "
        "Raises an error if any angle exceeds hardware limits."
    )
)
def set_all_mirrors_angles(
    targets: List[MirrorTarget] = Field(
        ...,
        description=(
            "List of mirror targets. Each entry must have mirror_index, "
            "theta_mdeg, and phi_mdeg."
        ),
    ),
    settle_ms: int = Field(
        SETTLE_MS,
        description=f"Settle time in ms applied after each mirror move. Default {SETTLE_MS} ms.",
    ),
) -> List[dict]:
    """
    Returns: list of {"mirror_index": int, "theta_mdeg": float, "phi_mdeg": float}
    """
    results = []
    for t in targets:
        _validate_mirror_index(t.mirror_index)
        _validate_angles(t.theta_mdeg, t.phi_mdeg)
        _move_mirror(t.mirror_index, t.theta_mdeg, t.phi_mdeg, settle_ms)
        results.append({
            "mirror_index": t.mirror_index,
            "theta_mdeg":   t.theta_mdeg,
            "phi_mdeg":     t.phi_mdeg,
        })
    return results


# ---------------------------------------------------------------------------
# Tool 5 – relative (delta) move, one mirror
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Apply a RELATIVE angular step to a single mirror. "
        "The new angle = current_angle + delta. "
        "Safer than set_mirror_angle when making small iterative adjustments "
        "during alignment, because you only need to specify how much to move, "
        "not the absolute target. "
        "Raises an error if the resulting angle would exceed hardware limits. "
        "mirror_index is 0-based."
    )
)
def step_mirror_angle(
    mirror_index: int = Field(
        ...,
        description=f"0-based mirror index. Must be between 0 and {NUM_MIRRORS - 1}.",
    ),
    delta_theta_mdeg: float = Field(
        ...,
        description="Step size for vertical tilt (pitch) in millidegrees. Can be negative.",
    ),
    delta_phi_mdeg: float = Field(
        ...,
        description="Step size for horizontal tilt (yaw) in millidegrees. Can be negative.",
    ),
    settle_ms: int = Field(
        SETTLE_MS,
        description=f"Settle time in ms after the move. Default {SETTLE_MS} ms.",
    ),
) -> dict:
    """
    Returns: {"mirror_index": int, "theta_mdeg": float, "phi_mdeg": float, "delta_theta_mdeg": float, "delta_phi_mdeg": float}
    """
    _validate_mirror_index(mirror_index)
    current = _read_mirror(mirror_index)
    new_theta = current["theta"] + delta_theta_mdeg
    new_phi   = current["phi"]   + delta_phi_mdeg
    _validate_angles(new_theta, new_phi)
    _move_mirror(mirror_index, new_theta, new_phi, settle_ms)
    return {
        "mirror_index":     mirror_index,
        "theta_mdeg":       new_theta,
        "phi_mdeg":         new_phi,
        "delta_theta_mdeg": delta_theta_mdeg,
        "delta_phi_mdeg":   delta_phi_mdeg,
    }


# ---------------------------------------------------------------------------
# Tool 6 – relative (delta) move, all mirrors
# ---------------------------------------------------------------------------

class MirrorDelta(BaseModel):
    mirror_index:     int   = Field(..., description="0-based mirror index.")
    delta_theta_mdeg: float = Field(..., description="Relative vertical tilt step in mdeg.")
    delta_phi_mdeg:   float = Field(..., description="Relative horizontal tilt step in mdeg.")


@mcp.tool(
    description=(
        "Apply RELATIVE angular steps to ALL mirrors in a single call. "
        "For each mirror: new_angle = current_angle + delta. "
        "Use this during iterative closed-loop alignment to nudge every "
        "mirror simultaneously rather than one at a time. "
        "Raises an error if any resulting angle would exceed hardware limits."
    )
)
def step_all_mirrors_angles(
    deltas: List[MirrorDelta] = Field(
        ...,
        description=(
            "List of per-mirror deltas. Each entry must have mirror_index, "
            "delta_theta_mdeg, and delta_phi_mdeg."
        ),
    ),
    settle_ms: int = Field(
        SETTLE_MS,
        description=f"Settle time in ms after each mirror move. Default {SETTLE_MS} ms.",
    ),
) -> List[dict]:
    """
    Returns: list of {"mirror_index", "theta_mdeg", "phi_mdeg", "delta_theta_mdeg", "delta_phi_mdeg"}
    """
    results = []
    for d in deltas:
        _validate_mirror_index(d.mirror_index)
        current   = _read_mirror(d.mirror_index)
        new_theta = current["theta"] + d.delta_theta_mdeg
        new_phi   = current["phi"]   + d.delta_phi_mdeg
        _validate_angles(new_theta, new_phi)
        _move_mirror(d.mirror_index, new_theta, new_phi, settle_ms)
        results.append({
            "mirror_index":     d.mirror_index,
            "theta_mdeg":       new_theta,
            "phi_mdeg":         new_phi,
            "delta_theta_mdeg": d.delta_theta_mdeg,
            "delta_phi_mdeg":   d.delta_phi_mdeg,
        })
    return results


# ---------------------------------------------------------------------------
# Tool 7 – query limits
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Query the safe hardware angle limits for a single mirror. "
        "Always call this before large absolute moves to avoid hardware errors. "
        "Returns the min and max allowed values for both theta and phi "
        "in millidegrees. "
        "mirror_index is 0-based."
    )
)
def get_mirror_limits(
    mirror_index: int = Field(
        ...,
        description=f"0-based mirror index. Must be between 0 and {NUM_MIRRORS - 1}.",
    )
) -> dict:
    """
    Returns: {"mirror_index": int, "theta_min_mdeg": float, "theta_max_mdeg": float,
              "phi_min_mdeg": float, "phi_max_mdeg": float}
    """
    _validate_mirror_index(mirror_index)
    return {
        "mirror_index":    mirror_index,
        "theta_min_mdeg":  THETA_MIN,
        "theta_max_mdeg":  THETA_MAX,
        "phi_min_mdeg":    PHI_MIN,
        "phi_max_mdeg":    PHI_MAX,
    }


# ---------------------------------------------------------------------------
# Tool 8 – home one mirror
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Move a single mirror to its home position: theta=0, phi=0 mdeg. "
        "Use this to reset a mirror to a known neutral state before a new "
        "alignment run or after an error. "
        "mirror_index is 0-based."
    )
)
def home_mirror(
    mirror_index: int = Field(
        ...,
        description=f"0-based mirror index. Must be between 0 and {NUM_MIRRORS - 1}.",
    ),
    settle_ms: int = Field(
        SETTLE_MS,
        description=f"Settle time in ms after the move. Default {SETTLE_MS} ms.",
    ),
) -> dict:
    """
    Returns: {"mirror_index": int, "theta_mdeg": 0.0, "phi_mdeg": 0.0}
    """
    _validate_mirror_index(mirror_index)
    _move_mirror(mirror_index, 0.0, 0.0, settle_ms)
    return {"mirror_index": mirror_index, "theta_mdeg": 0.0, "phi_mdeg": 0.0}


# ---------------------------------------------------------------------------
# Tool 9 – home all mirrors
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Move ALL mirrors to their home position: theta=0, phi=0 mdeg. "
        "Use this at the start of a new experiment run or after an emergency "
        "stop to return the optical path to a known neutral state. "
        "Equivalent to calling home_mirror for every mirror in sequence."
    )
)
def home_all_mirrors(
    settle_ms: int = Field(
        SETTLE_MS,
        description=f"Settle time in ms applied after each mirror move. Default {SETTLE_MS} ms.",
    ),
) -> List[dict]:
    """
    Returns: list of {"mirror_index": int, "theta_mdeg": 0.0, "phi_mdeg": 0.0}
    """
    results = []
    for i in range(NUM_MIRRORS):
        _move_mirror(i, 0.0, 0.0, settle_ms)
        results.append({"mirror_index": i, "theta_mdeg": 0.0, "phi_mdeg": 0.0})
    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
