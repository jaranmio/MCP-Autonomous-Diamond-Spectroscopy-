"""
orchestrator_server.py
----------------------
FastMCP **orchestration** layer for autonomous optical alignment: combines

1. **Live optics** — in-process calls to ``optics_server`` (tilt mirrors in millidegrees),
   the only hardware backend implemented in this repo today.
2. **Fiber-coupling surrogate** — ``FiberCouplingNet`` and normalization from
   ``surrogate-NN/Update of Automated optical alignment using digital twins.ipynb``,
   loaded from ``fiber_coupling_model.pth`` + ``normalisation_parameters.npz``.
3. **Power meter** — in-process calls to ``power_meter_server`` (Thorlabs PM via VISA / PyVISA)
   for ground-truth fiber coupling measurements used to validate surrogate predictions.

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
import power_meter_server  # noqa: E402
import surrogate_fiber_coupling as sfc  # noqa: E402

_TOOL_HINTS_READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_TOOL_HINTS_COMPUTE = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
# Used for all write operations (mirror moves, wavelength, zeroing) — none are destructive.
_TOOL_HINTS_WRITE = ToolAnnotations(
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
        "Orchestrates optical alignment using three backends: "
        "(1) optics_server — tilt mirrors in millidegrees; "
        "(2) fiber-coupling surrogate (digital twin) — FiberCouplingNet predicts η from a 9-D "
        "physical vector [m0_a0, m1_a0, m0_a1, m1_a1, δx, δy, δz, φx, φy], mirrors 0–1 map to "
        "NN indices (0,2) and (1,3), output η is fiber coupling efficiency after log-domain "
        "denormalization, millidegree↔NN-unit conversion uses explicit nn_*_to_mdeg scales; "
        "(3) power_meter_server — Thorlabs PM provides ground-truth power readings in Watts "
        "to validate surrogate predictions. "
        "Recommended session sequence: pm_setup (set wavelength + zero with beam blocked) → "
        "compare_surrogate_to_measurement (assess twin accuracy) → alignment_closed_loop_step "
        "(move mirrors, read power before/after). "
        "If surrogate weights are missing, prediction tools error until ORCH_SURROGATE_DIR is set. "
        "If power meter is unavailable, set PM_SIMULATE=1 for a stub."
    ),
)


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Orchestrator and backend status: optics simulation flag, NUM_MIRRORS, surrogate "
        "load state, ORCH_SURROGATE_DIR resolved path, default NN→mdeg scale placeholders, "
        "and power meter connection state."
    ),
)
def orchestrator_status() -> dict:
    pm_status: dict = {}
    try:
        pm_status = power_meter_server.get_power_meter_status()
    except Exception as exc:
        pm_status = {"error": str(exc)}
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
        "power_meter": pm_status,
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
        "inverse log normalization on the scalar output. Raises if weights are not loaded. "
        "input_power_scale: dimensionless fitting coefficient applied as "
        "``surrogate_output * input_power_scale`` (mirrors notebook regression parameter "
        "``input_power``, optimized in [0, 10 000] to match photodiode/sensor magnitudes). "
        "Default 1.0 returns raw η. Fit this value by minimizing SSR between "
        "``surrogate_predict_coupling * input_power_scale`` and your measured sensor readings."
    ),
)
def surrogate_predict_coupling(
    states: List[List[float]],
    input_power_scale: float = 1.0,
) -> dict:
    bundle = _require_surrogate()
    if not states:
        raise ValueError("states must be a non-empty list of 9-float vectors.")
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError("Each state must be a length-9 list of floats.")
    eff = bundle.predict_efficiency_batch(arr, input_power_scale=input_power_scale)
    return {
        "efficiency_eta": eff.tolist(),
        "input_power_scale": input_power_scale,
        "n_states": int(arr.shape[0]),
    }


@mcp.tool(
    annotations=_TOOL_HINTS_COMPUTE,
    description=(
        "Gradient-based **in-silico** optimization (notebook cell 27 style): maximize η w.r.t. "
        "mirror angles (indices 0–3) while optionally freezing lens/misalignment indices 4–8. "
        "Does not move hardware. Returns optimized 9-vector and final η. "
        "input_power_scale does not affect the optimum (it is a positive multiplier on η; "
        "the argmax is unchanged) so optimization always runs on raw η regardless of calibration."
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
        "target NN state **without** commanding the stage. "
        "When measure_current_power=True, also reads the power meter at the current (pre-move) "
        "position as a real-hardware baseline alongside the surrogate's η_before (default False)."
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
    measure_current_power: bool = False,
    n_samples_pm: int = 10,
    interval_ms_pm: int = 100,
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
    result: dict = {
        "initial_state_9": st,
        "optimized_state_9": final.tolist(),
        "efficiency_eta_before": eta_before,
        "efficiency_eta_after_optimization": eff,
        "num_epochs": num_epochs,
    }
    if measure_current_power:
        try:
            result["current_power_reading"] = power_meter_server.read_power_averaged(
                n_samples=n_samples_pm, interval_ms=interval_ms_pm
            )
        except Exception as exc:
            result["current_power_reading"] = {"error": str(exc)}
    return result


@mcp.tool(
    annotations=_TOOL_HINTS_WRITE,
    description=(
        "Closed-loop step: build 9-D state from live optics + lens parameters, optimize mirror "
        "angles in surrogate space (freeze indices 4–8), then command **all** mirrors via "
        "optics_server.set_mirror_angle using nn_theta_to_mdeg / nn_phi_to_mdeg. "
        "Mirrors without NN slots (index > 1) keep their current angles. "
        "When measure_power=True, reads actual power from the power meter before and after "
        "the move so you can compare surrogate predictions against reality (default False — "
        "safe when the power meter is not connected). "
        "n_samples_pm / interval_ms_pm control software averaging on the power meter. "
        "Call pm_setup first to set wavelength and zero the meter."
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
    measure_power: bool = False,
    n_samples_pm: int = 10,
    interval_ms_pm: int = 100,
) -> dict:
    if nn_theta_to_mdeg == 0 or nn_phi_to_mdeg == 0:
        raise ValueError("nn_theta_to_mdeg and nn_phi_to_mdeg must be non-zero.")
    settle = optics_server.SETTLE_MS if settle_ms is None else int(settle_ms)

    # Optionally capture baseline power before moving
    power_before: Optional[dict] = None
    if measure_power:
        try:
            power_before = power_meter_server.read_power_averaged(
                n_samples=n_samples_pm, interval_ms=interval_ms_pm
            )
        except Exception as exc:
            power_before = {"error": str(exc)}

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

    # Read actual power after move if requested
    power_after: Optional[dict] = None
    if measure_power:
        try:
            power_after = power_meter_server.read_power_averaged(
                n_samples=n_samples_pm, interval_ms=interval_ms_pm
            )
        except Exception as exc:
            power_after = {"error": str(exc)}

    result: dict = {
        "initial_state_9": st,
        "optimized_state_9": final.tolist(),
        "efficiency_eta_after_surrogate_optimization": eff,
        "optics_move_results": moves,
        "settle_ms": settle,
    }
    if measure_power:
        result["power_before_move"] = power_before
        result["power_after_move"] = power_after
        # Compute fractional change if both readings are valid numbers
        try:
            p_before = power_before["mean_power_W"]  # type: ignore[index]
            p_after = power_after["mean_power_W"]  # type: ignore[index]
            if p_before > 0:
                result["power_change_fraction"] = (p_after - p_before) / p_before
        except Exception:
            pass
    return result


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Compare the surrogate's predicted fiber coupling efficiency against the power meter's "
        "real measurement at the current mirror positions. "
        "Reads live optics state, evaluates the surrogate, and takes a software-averaged power "
        "reading. Returns both side-by-side so you can assess how well the digital twin matches "
        "the bench at the current operating point. "
        "input_power_scale: dimensionless fitting coefficient (default 1.0, notebook bounds "
        "[0, 10 000]) applied as ``surrogate_eta * input_power_scale`` to match the surrogate's "
        "normalized output to sensor measurement magnitudes. Not a physical power value — "
        "fit it by minimizing SSR between scaled surrogate predictions and sensor readings "
        "(mirrors notebook regression for ``input_power``)."
    ),
)
def compare_surrogate_to_measurement(
    lens_delta_x: float = 0.0,
    lens_delta_y: float = 0.0,
    lens_delta_z: float = 0.0,
    lens_rotate_x: float = 0.0,
    lens_rotate_y: float = 0.0,
    nn_theta_to_mdeg: float = _DEFAULT_NN_THETA_TO_MDEG,
    nn_phi_to_mdeg: float = _DEFAULT_NN_PHI_TO_MDEG,
    input_power_scale: float = 1.0,
    n_samples_pm: int = 10,
    interval_ms_pm: int = 100,
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
    eta = bundle.predict_efficiency(st)

    pm_result: dict = {}
    try:
        pm_result = power_meter_server.read_power_averaged(
            n_samples=n_samples_pm, interval_ms=interval_ms_pm
        )
    except Exception as exc:
        pm_result = {"error": str(exc)}

    result: dict = {
        "state_9": st,
        "surrogate_efficiency_eta": eta,
        "surrogate_scaled_output": eta * input_power_scale,
        "input_power_scale": input_power_scale,
        "power_meter": pm_result,
    }
    try:
        measured = pm_result["mean_power_W"]
        if measured > 0 and eta > 0:
            # Single-point estimate of input_power_scale = PM_reading_W / η.
            # NOTE: the notebook's ``input_power`` was fitted against photodiode *voltage*
            # readings (arbitrary units), not Watts, so this ratio is in different units
            # than the notebook's regression coefficient and cannot be plugged in directly.
            # It is useful as an order-of-magnitude starting point or when your sensor
            # is already calibrated in Watts.
            result["implied_input_power_scale_W_per_eta"] = measured / eta
    except Exception:
        pass
    return result


@mcp.tool(
    annotations=_TOOL_HINTS_WRITE,
    description=(
        "Convenience setup for the power meter before a measurement run: sets the operating "
        "wavelength (nm) then performs dark/background zeroing in one call. "
        "IMPORTANT: block the laser beam before calling — the zero step measures the dark offset. "
        "Equivalent to calling pm_set_wavelength then pm_zero_power_meter separately. "
        "Returns the confirmed wavelength_nm and zero_offset_W."
    ),
)
def pm_setup(wavelength_nm: float) -> dict:
    wl = power_meter_server.set_wavelength(wavelength_nm=wavelength_nm)
    zero = power_meter_server.zero_power_meter()
    return {
        "wavelength_nm": wl["wavelength_nm"],
        "zero_offset_W": zero["zero_offset_W"],
        "zero_elapsed_s": zero["elapsed_s"],
    }


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description="Proxy to power_meter_server.get_power_meter_status: connection state, wavelength, range, and averaging settings.",
)
def pm_get_status() -> dict:
    return power_meter_server.get_power_meter_status()


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description="Proxy to power_meter_server.read_power: single instantaneous power reading in Watts.",
)
def pm_read_power() -> dict:
    return power_meter_server.read_power()


@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Proxy to power_meter_server.read_power_averaged: software-averaged power over n_samples "
        "readings spaced interval_ms apart. Returns mean_power_W and std_power_W. "
        "Preferred over pm_read_power during mirror sweeps — use this after each alignment step."
    ),
)
def pm_read_power_averaged(n_samples: int = 10, interval_ms: int = 100) -> dict:
    return power_meter_server.read_power_averaged(n_samples=n_samples, interval_ms=interval_ms)


@mcp.tool(
    annotations=_TOOL_HINTS_WRITE,
    description=(
        "Proxy to power_meter_server.set_wavelength: set responsivity-correction wavelength (nm). "
        "Call at the start of every calibration run before any sweeps or zeroing."
    ),
)
def pm_set_wavelength(wavelength_nm: float) -> dict:
    return power_meter_server.set_wavelength(wavelength_nm=wavelength_nm)


@mcp.tool(
    annotations=_TOOL_HINTS_WRITE,
    description=(
        "Proxy to power_meter_server.zero_power_meter: dark/background subtraction. "
        "IMPORTANT: block the laser beam completely before calling. "
        "Blocks until zeroing completes (up to 30 s). "
        "Call once per session before any quantitative measurements."
    ),
)
def pm_zero_power_meter() -> dict:
    return power_meter_server.zero_power_meter()


if __name__ == "__main__":
    mcp.run()
