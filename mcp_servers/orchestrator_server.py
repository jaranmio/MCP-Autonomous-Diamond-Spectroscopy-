"""
orchestrator_server.py
----------------------
FastMCP **orchestration** layer for autonomous optical alignment: combines

1. **Live optics** — in-process calls to ``optics_server`` (tilt mirrors in millidegrees),
   the only hardware backend implemented in this repo today.
2. **Fiber-coupling surrogate** — ``FiberCouplingNet`` and normalization from
   ``surrogate-NN/Update of Automated optical alignment using digital twins.ipynb``,
   loaded from ``fiber_coupling_model.pth`` + ``normalisation_parameters.npz``.

**9-D surrogate state (physical NN input space, *before* linear normalization)**

Matches the notebook notation
``(φ₁, ψ₁, φ₂, ψ₂, δx, δy, δz, φ_x, φ_y)`` with array layout::

  index 0 — mirror-0 first angle (paired with index 2 in mesh plots)
  index 1 — mirror-1 first angle (paired with index 3)
  index 2 — mirror-0 second angle
  index 3 — mirror-1 second angle
  indices 4–8 — collimation / misalignment (δx, δy, δz, φ_x, φ_y)

**Mapping to ``optics_server`` (theta_mdeg / phi_mdeg)**

The notebook uses Zemax/simulator units; the MCP optics server uses **millidegrees**.
There is **no universal conversion** encoded in the notebook. This orchestrator uses
**explicit scale factors** (tool arguments and/or env defaults)::

  theta_mdeg = state[nn_theta_index] * nn_theta_to_mdeg
  phi_mdeg   = state[nn_phi_index]   * nn_phi_to_mdeg

with ``nn_theta_index`` / ``nn_phi_index`` = (0, 2) for mirror 0 and (1, 3) for mirror 1.
You must choose scales that match your calibration between the trained twin and the bench.

Environment (read once at import; restart process to apply)
-----------------------------------------------------------
``ORCH_SURROGATE_DIR``
    Directory containing ``fiber_coupling_model.pth`` and
    ``normalisation_parameters.npz``. Default: ``<repo>/surrogate-NN/NN_parameter_data``.
``ORCH_DEFAULT_NN_THETA_TO_MDEG``, ``ORCH_DEFAULT_NN_PHI_TO_MDEG``
    Default scale factors for tools that move optics (default ``1000.0``).

Optics uses existing ``OPTICS_*`` variables (see ``optics_server`` docstring).

Run (stdio MCP)::

  python orchestrator_server.py

with ``mcp_servers`` on ``sys.path`` (e.g. cwd = ``mcp_servers``, or pass the absolute path
to this file as the script argument).
"""


from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

_ORCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ORCH_DIR.parent
if str(_ORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCH_DIR))

import optics_server  # noqa: E402
import surrogate_fiber_coupling as sfc  # noqa: E402

_TOOL_HINTS_READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_TOOL_HINTS_COMPUTE = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_TOOL_HINTS_MOTION = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    openWorldHint=False,
)

_DEFAULT_SURROGATE_DIR = _REPO_ROOT / "surrogate-NN" / "NN_parameter_data"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _resolve_surrogate_dir() -> Path:
    raw = os.environ.get("ORCH_SURROGATE_DIR", "").strip()
    if raw:
        return Path(raw)
    return _DEFAULT_SURROGATE_DIR


_SURROGATE_DIR = _resolve_surrogate_dir()
_DEFAULT_NN_THETA_TO_MDEG = _env_float("ORCH_DEFAULT_NN_THETA_TO_MDEG", 1000.0)
# assuming NN angles are in Zemax-like units (degrees), this default corresponds to 1000 mdeg = 1 NN angle unit (e.g. 1 mdeg = 0.001 radian)
_DEFAULT_NN_PHI_TO_MDEG = _env_float("ORCH_DEFAULT_NN_PHI_TO_MDEG", 1000.0)
# assuming similar scales for phi

_surrogate: Optional[sfc.SurrogateBundle] = None
_surrogate_load_error: Optional[str] = None

try:
    _surrogate = sfc.SurrogateBundle.load(_SURROGATE_DIR)
except Exception as e:
    _surrogate_load_error = f"{type(e).__name__}: {e}"


def _require_surrogate() -> sfc.SurrogateBundle:
    if _surrogate is None:
        msg = _surrogate_load_error or "Surrogate not loaded."
        raise RuntimeError(
            f"{msg} Set ORCH_SURROGATE_DIR to a folder with fiber_coupling_model.pth and "
            "normalisation_parameters.npz (see surrogate-NN notebook)."
        )
    return _surrogate


def _mirror_nn_indices(mirror_index: int) -> tuple[int, int]:
    """Return (nn_index_theta, nn_index_phi) for optics mirror_index."""
    if mirror_index == 0:
        return 0, 2
    if mirror_index == 1:
        return 1, 3
    raise ValueError(
        f"mirror_index {mirror_index} unsupported for NN coupling layout "
        f"(notebook defines two mirrors in indices 0–3 only)."
    )


def build_state_9_from_optics_and_lens(
    lens_delta_x: float,
    lens_delta_y: float,
    lens_delta_z: float,
    lens_rotate_x: float,
    lens_rotate_y: float,
    *,
    nn_theta_to_mdeg: float,
    nn_phi_to_mdeg: float,
) -> List[float]:
    """
    Read all mirrors from ``optics_server``, map mdeg → NN units via inverse scales,
    fill indices 4–8 from lens parameters.
    """
    if nn_theta_to_mdeg == 0 or nn_phi_to_mdeg == 0:
        raise ValueError("nn_theta_to_mdeg and nn_phi_to_mdeg must be non-zero.")
    n = optics_server.NUM_MIRRORS
    state = [0.0] * 9
    state[4] = lens_delta_x
    state[5] = lens_delta_y
    state[6] = lens_delta_z
    state[7] = lens_rotate_x
    state[8] = lens_rotate_y
    for i in range(min(n, 2)):
        st = optics_server.get_mirror_state(i)
        ti, pi = _mirror_nn_indices(i)
        state[ti] = float(st["theta_mdeg"]) / nn_theta_to_mdeg
        state[pi] = float(st["phi_mdeg"]) / nn_phi_to_mdeg
    return state


def apply_nn_mirror_angles_to_optics(
    state_9: List[float],
    *,
    nn_theta_to_mdeg: float,
    nn_phi_to_mdeg: float,
    settle_ms: int,
) -> List[dict]:
    """Command mirrors 0–1 from NN indices; mirrors ≥2 are left unchanged (no surrogate slots)."""
    n = optics_server.NUM_MIRRORS
    s = np.asarray(state_9, dtype=float).reshape(9)
    out: List[dict] = []
    for i in range(n):
        if i <= 1:
            ti, pi = _mirror_nn_indices(i)
            th = float(s[ti]) * nn_theta_to_mdeg
            ph = float(s[pi]) * nn_phi_to_mdeg
            out.append(optics_server.set_mirror_angle(i, th, ph, settle_ms=settle_ms))
        else:
            st = optics_server.get_mirror_state(i)
            out.append(
                {
                    "mirror_index": i,
                    "theta_mdeg": st["theta_mdeg"],
                    "phi_mdeg": st["phi_mdeg"],
                    "skipped": True,
                    "reason": "no NN angle slots for mirror_index>=2 in twin layout",
                }
            )
    return out


mcp = FastMCP(
    name="orchestrator_server",
    instructions=(
        "Orchestrates optical alignment using the fiber-coupling surrogate (digital twin) "
        "and the real optics MCP backend (optics_server). "
        "Surrogate input is a 9-D physical vector [m0_a0, m1_a0, m0_a1, m1_a1, δx, δy, δz, φx, φy]; "
        "mirrors 0–1 map to NN indices (0,2) and (1,3). Output η is fiber coupling efficiency "
        "after log-domain denormalization. "
        "Millidegree moves use explicit nn_*_to_mdeg scales — calibrate to your bench. "
        "If surrogate weights are missing, prediction tools error until ORCH_SURROGATE_DIR is set."
    ),
)


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Orchestrator and backend status: optics simulation flag, NUM_MIRRORS, surrogate "
        "load state, ORCH_SURROGATE_DIR resolved path, and default NN→mdeg scale placeholders."
    ),
)
def orchestrator_status() -> dict:
    return {
        "optics_server_name": optics_server.mcp.name,
        "optics_simulate": optics_server.settings.simulate,
        "num_mirrors": optics_server.NUM_MIRRORS,
        "surrogate_loaded": _surrogate is not None,
        "surrogate_load_error": _surrogate_load_error,
        "surrogate_dir": str(_SURROGATE_DIR.resolve()),
        "default_nn_theta_to_mdeg": _DEFAULT_NN_THETA_TO_MDEG,
        "default_nn_phi_to_mdeg": _DEFAULT_NN_PHI_TO_MDEG,
        "surrogate_state_layout": (
            "[m0_angle_a, m1_angle_a, m0_angle_b, m1_angle_b, dx, dy, dz, rot_x, rot_y] "
            "in physical NN training space; pair (0,2) for mirror 0, (1,3) for mirror 1."
        ),
    }


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Return training-box bounds for the 9-D surrogate input from normalisation_parameters.npz "
        "(inputs_min / inputs_max per index). Use to keep optimization inside the digital-twin domain."
    ),
)
def surrogate_input_bounds() -> dict:
    b = _require_surrogate()
    return {
        "inputs_min": b.norm.inputs_min.tolist(),
        "inputs_max": b.norm.inputs_max.tolist(),
        "outputs_min": b.norm.outputs_min,
        "outputs_max": b.norm.outputs_max,
        "output_is_efficiency_eta": True,
        "output_denormalization": "log10 scaling via stored outputs_min/max (see notebook).",
    }


@mcp.tool(
    annotations=_TOOL_HINTS_COMPUTE,
    description=(
        "Evaluate the surrogate MLP: predict fiber coupling efficiency η for one or more 9-D states "
        "(physical NN space, not millidegrees). Network applies linear input normalization then "
        "inverse log normalization on the scalar output. Raises if weights are not loaded."
    ),
)
def surrogate_predict_coupling(states: List[List[float]]) -> dict:
    bundle = _require_surrogate()
    if not states:
        raise ValueError("states must be a non-empty list of 9-float vectors.")
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError("Each state must be a length-9 list of floats.")
    eff = bundle.predict_efficiency_batch(arr)
    return {
        "efficiency_eta": eff.tolist(),
        "n_states": int(arr.shape[0]),
    }


@mcp.tool(
    annotations=_TOOL_HINTS_COMPUTE,
    description=(
        "Gradient-based **in-silico** optimization (notebook cell 27 style): maximize η w.r.t. "
        "mirror angles (indices 0–3) while optionally freezing lens/misalignment indices 4–8. "
        "Does not move hardware. Returns optimized 9-vector and final η."
    ),
)
def surrogate_optimize_mirror_angles(
    initial_state_9: List[float],
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    freeze_lens_parameters: bool = True,
) -> dict:
    bundle = _require_surrogate()
    if len(initial_state_9) != 9:
        raise ValueError("initial_state_9 must have length 9.")
    init = np.asarray(initial_state_9, dtype=np.float64)
    final, eff = sfc.optimize_mirror_angles_maximize_efficiency(
        bundle,
        init,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        freeze_lens=freeze_lens_parameters,
    )
    return {
        "optimized_state_9": final.tolist(),
        "efficiency_eta": eff,
        "num_epochs": num_epochs,
        "freeze_lens_parameters": freeze_lens_parameters,
    }


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Proxy to optics_server.get_all_mirrors_state: current theta_mdeg and phi_mdeg for every mirror."
    ),
)
def optics_get_all_mirrors_state() -> List[dict]:
    return optics_server.get_all_mirrors_state()


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description="Proxy to optics_server.get_mirror_limits for one mirror (mdeg bounds).",
)
def optics_get_mirror_limits(mirror_index: int) -> dict:
    return optics_server.get_mirror_limits(mirror_index)


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description="Proxy to optics_server.get_mirror_state for one mirror.",
)
def optics_get_mirror_state(mirror_index: int) -> dict:
    return optics_server.get_mirror_state(mirror_index)


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Build the surrogate 9-vector from **current** optics positions (mirrors 0–1 mapped to "
        "NN indices) plus explicit lens/misalignment components for indices 4–8. "
        "Inverse maps: nn_angle = mdeg / nn_*_to_mdeg."
    ),
)
def build_surrogate_state_from_live_optics(
    lens_delta_x: float = 0.0,
    lens_delta_y: float = 0.0,
    lens_delta_z: float = 0.0,
    lens_rotate_x: float = 0.0,
    lens_rotate_y: float = 0.0,
    nn_theta_to_mdeg: float = _DEFAULT_NN_THETA_TO_MDEG,
    nn_phi_to_mdeg: float = _DEFAULT_NN_PHI_TO_MDEG,
) -> dict:
    if nn_theta_to_mdeg == 0 or nn_phi_to_mdeg == 0:
        raise ValueError("nn_theta_to_mdeg and nn_phi_to_mdeg must be non-zero.")
    st = build_state_9_from_optics_and_lens(
        lens_delta_x,
        lens_delta_y,
        lens_delta_z,
        lens_rotate_x,
        lens_rotate_y,
        nn_theta_to_mdeg=nn_theta_to_mdeg,
        nn_phi_to_mdeg=nn_phi_to_mdeg,
    )
    return {"state_9": st}


@mcp.tool(
    annotations=_TOOL_HINTS_COMPUTE,
    description=(
        "End-to-end **read-only** alignment suggestion: read live optics, build the 9-vector, "
        "run surrogate mirror-angle optimization (freeze lens block 4–8), return predicted η and "
        "target NN state **without** commanding the stage."
    ),
)
def suggest_alignment_from_live_optics(
    lens_delta_x: float = 0.0,
    lens_delta_y: float = 0.0,
    lens_delta_z: float = 0.0,
    lens_rotate_x: float = 0.0,
    lens_rotate_y: float = 0.0,
    nn_theta_to_mdeg: float = _DEFAULT_NN_THETA_TO_MDEG,
    nn_phi_to_mdeg: float = _DEFAULT_NN_PHI_TO_MDEG,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
) -> dict:
    st = build_state_9_from_optics_and_lens(
        lens_delta_x,
        lens_delta_y,
        lens_delta_z,
        lens_rotate_x,
        lens_rotate_y,
        nn_theta_to_mdeg=nn_theta_to_mdeg,
        nn_phi_to_mdeg=nn_phi_to_mdeg,
    )
    bundle = _require_surrogate()
    init = np.asarray(st, dtype=np.float64)
    final, eff = sfc.optimize_mirror_angles_maximize_efficiency(
        bundle,
        init,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        freeze_lens=True,
    )
    eta_before = bundle.predict_efficiency(st)
    return {
        "initial_state_9": st,
        "optimized_state_9": final.tolist(),
        "efficiency_eta_before": eta_before,
        "efficiency_eta_after_optimization": eff,
        "num_epochs": num_epochs,
    }


@mcp.tool(
    annotations=_TOOL_HINTS_MOTION,
    description=(
        "Closed-loop step: build 9-D state from live optics + lens parameters, optimize mirror "
        "angles in surrogate space (freeze indices 4–8), then command **all** mirrors via "
        "optics_server.set_mirror_angle using nn_theta_to_mdeg / nn_phi_to_mdeg. "
        "Mirrors without NN slots (index > 1) keep their current angles."
    ),
)
def alignment_closed_loop_step(
    lens_delta_x: float = 0.0,
    lens_delta_y: float = 0.0,
    lens_delta_z: float = 0.0,
    lens_rotate_x: float = 0.0,
    lens_rotate_y: float = 0.0,
    nn_theta_to_mdeg: float = _DEFAULT_NN_THETA_TO_MDEG,
    nn_phi_to_mdeg: float = _DEFAULT_NN_PHI_TO_MDEG,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    settle_ms: Optional[int] = None,
) -> dict:
    if nn_theta_to_mdeg == 0 or nn_phi_to_mdeg == 0:
        raise ValueError("nn_theta_to_mdeg and nn_phi_to_mdeg must be non-zero.")
    settle = optics_server.SETTLE_MS if settle_ms is None else int(settle_ms)
    st = build_state_9_from_optics_and_lens(
        lens_delta_x,
        lens_delta_y,
        lens_delta_z,
        lens_rotate_x,
        lens_rotate_y,
        nn_theta_to_mdeg=nn_theta_to_mdeg,
        nn_phi_to_mdeg=nn_phi_to_mdeg,
    )
    bundle = _require_surrogate()
    init = np.asarray(st, dtype=np.float64)
    final, eff = sfc.optimize_mirror_angles_maximize_efficiency(
        bundle,
        init,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        freeze_lens=True,
    )
    moves = apply_nn_mirror_angles_to_optics(
        final.tolist(),
        nn_theta_to_mdeg=nn_theta_to_mdeg,
        nn_phi_to_mdeg=nn_phi_to_mdeg,
        settle_ms=settle,
    )
    return {
        "initial_state_9": st,
        "optimized_state_9": final.tolist(),
        "efficiency_eta_after_surrogate_optimization": eff,
        "optics_move_results": moves,
        "settle_ms": settle,
    }


if __name__ == "__main__":
    mcp.run()
