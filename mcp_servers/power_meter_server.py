"""
power_meter_server.py
---------------------
MCP server for the Thorlabs PM101A optical power meter in the diamond spectroscopy setup.

Communication is via SCPI over USB (USBTMC) using pyvisa.  All SCPI commands follow the
PM101x programmer's reference (mtn013681-d04).

Configuration (environment variables, read once at import; restart the MCP process to apply
changes):

  PM_VISA_ADDRESS  — VISA resource string for the power meter, e.g.
                     "USB0::0x1313::0x8078::P0012345::INSTR".
                     If unset, the server auto-scans connected VISA resources and picks the
                     first Thorlabs power meter it can identify via *IDN?.
  PM_SIMULATE      — if 1/true, use in-memory stub (no pyvisa load), even when hardware
                     is present.  Defaults to off.

Exposed tools (for LLM use):
  1. get_power_meter_status   - read connection state, wavelength, range, averaging settings
  2. read_power               - single instantaneous power reading (W)
  3. read_power_averaged      - software-averaged power reading: mean + std over N samples
  4. set_wavelength           - set operating wavelength for responsivity correction (nm)
  5. zero_power_meter         - dark/background subtraction (block the beam before calling!)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# ---------------------------------------------------------------------------
# Tool annotation hints
# ---------------------------------------------------------------------------

_TOOL_HINTS_READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_TOOL_HINTS_CONFIG = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    openWorldHint=False,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hardware defaults
_DEFAULT_TIMEOUT_MS = 5_000       # VISA read/query timeout
_ZERO_POLL_INTERVAL_S = 0.2       # how often to poll zeroing-state during zero_power_meter
_ZERO_TIMEOUT_S = 30.0            # give up waiting for zeroing after this many seconds


def _truthy_string(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PowerMeterSettings:
    """Process configuration snapshot from environment (see module docstring)."""

    simulate: bool
    visa_address: str   # empty string → auto-detect at init time

    @classmethod
    def from_env(cls) -> PowerMeterSettings:
        return cls(
            simulate=_truthy_string(os.environ.get("PM_SIMULATE", "")),
            visa_address=os.environ.get("PM_VISA_ADDRESS", "").strip(),
        )


settings = PowerMeterSettings.from_env()

# ---------------------------------------------------------------------------
# Hardware backend — real pyvisa instrument
# ---------------------------------------------------------------------------

class _PowerMeterHW:
    """Thin wrapper around a pyvisa resource for the PM101A.

    query() sends a SCPI command and returns the stripped response string.
    write() sends a SCPI command with no response expected.
    """

    def __init__(self, resource) -> None:  # resource: pyvisa.resources.Resource
        self._inst = resource

    def query(self, cmd: str) -> str:
        return self._inst.query(cmd).strip()

    def write(self, cmd: str) -> None:
        self._inst.write(cmd)

    def close(self) -> None:
        try:
            self._inst.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Simulation stub
# ---------------------------------------------------------------------------

class _SimulatedPowerMeter:
    """In-memory stub for offline/macOS development (PM_SIMULATE=1)."""

    def __init__(self) -> None:
        self._wavelength_m: float = 1064e-9   # 1064 nm
        self._auto_range: bool = True
        self._range_upper_W: float = 0.02     # 20 mW
        self._unit: str = "W"
        self._avg_count: int = 1
        self._zero_offset_W: float = 0.0
        self._base_power_W: float = 1e-3      # 1 mW simulated signal

    def query(self, cmd: str) -> str:
        cmd_upper = cmd.strip().upper()
        if cmd_upper == "*IDN?":
            return "THORLABS,PM101A,SIM000001,1.0.0"
        if cmd_upper == "SENS:CORR:WAV?":
            return f"{self._wavelength_m:.6e}"
        if cmd_upper == "SENS:POW:DC:RANG:AUTO?":
            return "1" if self._auto_range else "0"
        if cmd_upper == "SENS:POW:DC:RANG:UPP?":
            return f"{self._range_upper_W:.6e}"
        if cmd_upper == "SENS:POW:DC:UNIT?":
            return self._unit
        if cmd_upper == "SENS:AVG:CNT?":
            return str(self._avg_count)
        if cmd_upper == "SENS:CORR:COLL:ZERO:MAG?":
            return f"{self._zero_offset_W:.6e}"
        if cmd_upper == "SENS:CORR:COLL:ZERO:STAT?":
            return "0"   # always idle in simulation
        if cmd_upper == "MEAS:POW?":
            import random
            noise = random.gauss(0.0, 5e-6)   # ±5 µW noise
            raw = self._base_power_W + noise
            return f"{max(0.0, raw - self._zero_offset_W):.6e}"
        return ""

    def write(self, cmd: str) -> None:
        parts = cmd.strip().split(None, 1)
        key = parts[0].upper()
        val = parts[1].strip() if len(parts) > 1 else ""
        if key == "SENS:CORR:WAV":
            self._wavelength_m = float(val)
        elif key == "SENS:CORR:COLL:ZERO:INIT":
            # Simulate zeroing: capture current noise floor as offset
            import random
            self._zero_offset_W = random.gauss(0.0, 2e-6)
        elif key == "SENS:AVG:CNT":
            self._avg_count = max(1, int(float(val)))

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Module-level hardware initialisation (lazy, once)
# ---------------------------------------------------------------------------

_hw: Optional[_PowerMeterHW | _SimulatedPowerMeter] = None
_hw_init_error: str = ""


def _init_hw() -> None:
    """Attempt to open the VISA connection.  Sets module globals _hw / _hw_init_error."""
    global _hw, _hw_init_error

    if settings.simulate:
        _hw = _SimulatedPowerMeter()
        return

    try:
        import pyvisa  # type: ignore
    except ImportError:
        _hw_init_error = (
            "pyvisa not installed — run `uv add pyvisa pyvisa-py` or set PM_SIMULATE=1."
        )
        return

    rm = pyvisa.ResourceManager()

    address = settings.visa_address
    if not address:
        # Auto-detect: scan for the first resource whose *IDN? identifies a Thorlabs PM
        for r in rm.list_resources():
            if "1313" not in r:   # Thorlabs USB vendor ID 0x1313
                continue
            try:
                candidate = rm.open_resource(r)
                candidate.timeout = _DEFAULT_TIMEOUT_MS
                idn = candidate.query("*IDN?").strip().upper()
                if "THORLABS" in idn and ("PM" in idn):
                    address = r
                    candidate.close()
                    break
                candidate.close()
            except Exception:
                pass

    if not address:
        _hw_init_error = (
            "No Thorlabs power meter found on any VISA resource. "
            "Set PM_VISA_ADDRESS or PM_SIMULATE=1."
        )
        return

    try:
        inst = rm.open_resource(address)
        inst.timeout = _DEFAULT_TIMEOUT_MS
        # Confirm identity
        idn = inst.query("*IDN?").strip()
        if "THORLABS" not in idn.upper():
            inst.close()
            _hw_init_error = f"Device at {address!r} does not identify as Thorlabs: {idn!r}"
            return
        _hw = _PowerMeterHW(inst)
    except Exception as exc:
        _hw_init_error = f"Failed to open {address!r}: {exc}"


_init_hw()


def _get_hw() -> _PowerMeterHW | _SimulatedPowerMeter:
    """Return the active backend.  Raises RuntimeError when hardware unavailable."""
    if _hw is not None:
        return _hw
    raise RuntimeError(
        f"Power meter not available: {_hw_init_error or 'unknown error'}. "
        "Set PM_SIMULATE=1 to run without hardware."
    )


def _is_simulating() -> bool:
    return isinstance(_hw, _SimulatedPowerMeter)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="power_meter_server",
    instructions=(
        "Controls the Thorlabs PM101A optical power meter in the diamond spectroscopy setup. "
        "All power readings are in Watts (W) unless noted otherwise. "
        "Wavelength is always specified in nanometres (nm). "
        "Call get_power_meter_status first to confirm the meter is connected and correctly configured. "
        "Call set_wavelength to match the laser wavelength before any calibration run — the meter "
        "applies a responsivity correction curve that depends on wavelength. "
        "Call zero_power_meter with the beam BLOCKED to subtract the dark/background offset before "
        "any quantitative measurement. "
        "Use read_power for a single instantaneous reading; use read_power_averaged for a stable "
        "mean + std estimate over multiple samples (preferred during mirror sweeps). "
        "Set PM_SIMULATE=1 to run without hardware (in-memory stub with ~1 mW simulated signal). "
        "Set PM_VISA_ADDRESS to a specific VISA resource string; otherwise the server auto-detects "
        "the first Thorlabs power meter found among USB VISA resources."
    ),
)


# ---------------------------------------------------------------------------
# Tool 1 — status
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Read the current configuration and connection state of the power meter. "
        "Returns: device identity string, wavelength_nm (operating wavelength for responsivity "
        "correction), auto_range (bool), range_upper_W (current range ceiling in W), "
        "power_unit ('W' or 'DBM'), zero_offset_W (dark offset from last zero_power_meter call), "
        "avg_count (device-side averaging count), and simulate (true when running in stub mode). "
        "Call this first to verify the meter is ready before a calibration run."
    ),
)
def get_power_meter_status() -> dict:
    hw = _get_hw()
    idn = hw.query("*IDN?")
    wavelength_m = float(hw.query("SENS:CORR:WAV?"))
    auto_range = hw.query("SENS:POW:DC:RANG:AUTO?") in ("1", "ON")
    range_upper_W = float(hw.query("SENS:POW:DC:RANG:UPP?"))
    power_unit = hw.query("SENS:POW:DC:UNIT?")
    zero_offset_W = float(hw.query("SENS:CORR:COLL:ZERO:MAG?"))
    avg_count = int(hw.query("SENS:AVG:CNT?"))
    return {
        "identity": idn,
        "wavelength_nm": wavelength_m * 1e9,
        "auto_range": auto_range,
        "range_upper_W": range_upper_W,
        "power_unit": power_unit,
        "zero_offset_W": zero_offset_W,
        "avg_count": avg_count,
        "simulate": _is_simulating(),
    }


# ---------------------------------------------------------------------------
# Tool 2 — single power reading
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Read a single instantaneous optical power value from the meter (in Watts). "
        "The meter applies the configured responsivity correction (wavelength) and subtracts "
        "any previously stored zero offset. "
        "For noisy or mechanically disturbed beams prefer read_power_averaged instead. "
        "Returns: power_W (float) and timestamp_s (Unix time of the reading)."
    ),
)
def read_power() -> dict:
    hw = _get_hw()
    power_W = float(hw.query("MEAS:POW?"))
    return {
        "power_W": power_W,
        "timestamp_s": time.time(),
    }


# ---------------------------------------------------------------------------
# Tool 3 — software-averaged power reading
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_TOOL_HINTS_READ,
    description=(
        "Read optical power averaged over n_samples consecutive measurements, spaced "
        "interval_ms milliseconds apart (software-side averaging — does not change device "
        "averaging settings). "
        "Returns mean_power_W, std_power_W (sample standard deviation), n_samples, and "
        "interval_ms. "
        "Use this during mirror sweeps or after mirror moves to get a stable efficiency "
        "estimate; the standard deviation indicates measurement stability. "
        "Typical values: n_samples=10, interval_ms=100 (1 second total collection). "
        "n_samples must be ≥ 1; std_power_W is 0.0 when n_samples=1."
    ),
)
def read_power_averaged(n_samples: int = 10, interval_ms: int = 100) -> dict:
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if interval_ms < 0:
        raise ValueError("interval_ms must be >= 0")

    hw = _get_hw()
    samples: list[float] = []
    for i in range(n_samples):
        samples.append(float(hw.query("MEAS:POW?")))
        if i < n_samples - 1:
            time.sleep(interval_ms / 1000.0)

    mean_W = sum(samples) / len(samples)
    if n_samples > 1:
        variance = sum((s - mean_W) ** 2 for s in samples) / (n_samples - 1)
        std_W = variance ** 0.5
    else:
        std_W = 0.0

    return {
        "mean_power_W": mean_W,
        "std_power_W": std_W,
        "n_samples": n_samples,
        "interval_ms": interval_ms,
    }


# ---------------------------------------------------------------------------
# Tool 4 — set wavelength
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_TOOL_HINTS_CONFIG,
    description=(
        "Set the operating wavelength of the power meter for responsivity correction (in nm). "
        "The PM101A applies a wavelength-dependent correction curve to convert raw photodiode "
        "current to optical power — this must match the actual laser wavelength for accurate "
        "readings. "
        "Always call this at the start of a calibration run before any sweeps or zero. "
        "Returns the wavelength_nm that was actually stored by the device (may be rounded to "
        "the nearest integer nm by the firmware)."
    ),
)
def set_wavelength(wavelength_nm: float) -> dict:
    hw = _get_hw()
    wavelength_m = wavelength_nm * 1e-9   # SCPI unit is metres
    hw.write(f"SENS:CORR:WAV {wavelength_m:.6e}")
    # Read back to confirm what the device accepted (firmware may round)
    actual_m = float(hw.query("SENS:CORR:WAV?"))
    return {
        "wavelength_nm": actual_m * 1e9,
    }


# ---------------------------------------------------------------------------
# Tool 5 — zero (dark offset subtraction)
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=_TOOL_HINTS_CONFIG,
    description=(
        "Perform dark/background subtraction on the power meter (zero adjustment). "
        "IMPORTANT: block the laser beam completely before calling this tool. "
        "The meter measures residual signal with no light present and stores it as an offset "
        "that is subtracted from all subsequent readings. "
        "The function blocks until zeroing completes (polls SENS:CORR:COLL:ZERO:STAT? at "
        f"{int(_ZERO_POLL_INTERVAL_S * 1000)} ms intervals, timeout {int(_ZERO_TIMEOUT_S)} s). "
        "Returns zero_offset_W (the measured dark offset in W) and elapsed_s. "
        "Raises an error if zeroing does not complete within the timeout."
    ),
)
def zero_power_meter() -> dict:
    hw = _get_hw()

    hw.write("SENS:CORR:COLL:ZERO:INIT")

    # Poll until zeroing finishes (state == "0") or timeout
    t_start = time.time()
    while True:
        state = hw.query("SENS:CORR:COLL:ZERO:STAT?").strip()
        if state == "0":
            break
        elapsed = time.time() - t_start
        if elapsed > _ZERO_TIMEOUT_S:
            raise RuntimeError(
                f"zero_power_meter timed out after {elapsed:.1f} s. "
                "Check that the beam is blocked and the meter is responsive."
            )
        time.sleep(_ZERO_POLL_INTERVAL_S)

    zero_offset_W = float(hw.query("SENS:CORR:COLL:ZERO:MAG?"))
    elapsed_s = time.time() - t_start
    return {
        "zero_offset_W": zero_offset_W,
        "elapsed_s": round(elapsed_s, 3),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
