"""
Thorlabs XA — positional repeatability study with research-oriented logging.

Commands nominal positions via the .NET API (same patterns as ``Miscellaneous/thorlabs_xa_simple_motion_test.py``
and SDK ``ExamplesUsingInterfaces``). After each approach and dwell, records indicated position
from ``IStatusItems`` + ``IUnitConverter`` (distance scale).

Outputs a timestamped directory (default under the repo ``repeatability_logs/``) containing:

  - ``manifest.json`` — protocol parameters, environment metadata, serial number
  - ``samples.csv`` — one row per measurement
  - ``summary.json`` — per-target statistics (n, mean, std, min, max, range)

Protocol (default, unidirectional): for each trial, move to a *retreat* position
(nominal − retreat offset, clamped to travel limits when available), dwell, move to nominal,
dwell, then sample position. This separates approach repeatability from single-point noise.

Override output directory: ``THORLABS_REPEATABILITY_LOG_DIR``.

Run from repo root::

  python tests/thorlabs_xa_repeatability_study.py

Requires: 64-bit Windows Python, pythonnet, Thorlabs XA managed DLL (``OPTICS_THORLABS_DOTNET_DIR``).
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from mcp_servers.dotnet_cast import cast_clr_iface as _cast
except ImportError:
    _mcp_servers_dir = str(_REPO_ROOT / "mcp_servers")
    if _mcp_servers_dir not in sys.path:
        sys.path.insert(0, _mcp_servers_dir)
    from dotnet_cast import cast_clr_iface as _cast

# ---------------------------------------------------------------------------
# Study configuration (edit for your protocol; cite values in publication)
# ---------------------------------------------------------------------------

DEVICE_SERIAL = "27272875"

# Nominal positions in physical units for the device's distance scale (typically mm).
TARGET_POSITIONS_MM: list[float] = [0.0, 1.0, 0.0, 2.0, 0.0]

# Full repeatability cycles: each cycle visits every target in order.
NUM_CYCLES = 25

# Before each nominal, retreat by this offset (same units as targets), then approach.
RETREAT_OFFSET_MM = 2.5

# Dwell after retreat move and after nominal move (seconds).
SETTLE_AFTER_RETREAT_S = 1.5
SETTLE_AFTER_NOMINAL_S = 2.0

# Averages per commanded point (set >1 to reduce read noise; report in methods).
SAMPLES_PER_POINT = 1
SAMPLE_INTERVAL_S = 0.05

THORLABS_XA_MANAGED_DLL = "tlmc_xa_dotnet.dll"
_DEFAULT_THORLABS_XA_DOTNET_DIR = (
    r"C:\Program Files\Thorlabs XA\SDK\.NET Framework (C#)\Libraries\x64"
)


def _dotnet_dir() -> str:
    env = os.environ.get("OPTICS_THORLABS_DOTNET_DIR", "").strip()
    if env:
        return env.rstrip("\\/")
    d = _DEFAULT_THORLABS_XA_DOTNET_DIR
    if os.path.isfile(os.path.join(d, THORLABS_XA_MANAGED_DLL)):
        return d
    return ""


def _load_xa() -> Any:
    d = _dotnet_dir()
    if not d:
        raise FileNotFoundError(
            f"Set OPTICS_THORLABS_DOTNET_DIR to the folder containing {THORLABS_XA_MANAGED_DLL}."
        )
    path = os.path.join(d, THORLABS_XA_MANAGED_DLL)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not found: {path}")
    import clr  # type: ignore[import-not-found]

    if sys.platform == "win32":
        os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
    clr.AddReference(path)
    import Thorlabs.MotionControl.XA as xa  # type: ignore[import-not-found]

    return xa


def _iface(xa: Any, name: str) -> Any:
    if hasattr(xa, name):
        return getattr(xa, name)
    df = importlib.import_module("Thorlabs.MotionControl.XA.DeviceFeatures")
    return getattr(df, name)


def _try_open_device(sm: Any, serial: str, xa: Any) -> Any:
    om = xa.OperatingModes.Default
    try:
        result = sm.TryOpenDevice(serial, "", om, None)
    except TypeError:
        result = sm.TryOpenDevice(serial, "", om)
    if isinstance(result, tuple):
        if len(result) >= 2:
            ok, dev = bool(result[0]), result[1]
        else:
            ok, dev = True, result[0]
    else:
        ok, dev = True, result
    if not ok or dev is None:
        raise RuntimeError(f"TryOpenDevice returned false for serial {serial!r}.")
    return dev


def _close_device(xa: Any, device: Any) -> None:
    disc = _cast(device, _iface(xa, "IDisconnect"))
    if disc is not None:
        try:
            disc.Disconnect()
        except Exception:
            pass
    try:
        device.Close()
    except Exception:
        pass


def _prepare_device(device: Any, xa: Any) -> None:
    loadp = _cast(device, _iface(xa, "ILoadParams"))
    en = _cast(device, _iface(xa, "IEnableState"))
    if loadp is None or en is None:
        raise RuntimeError("ILoadParams / IEnableState required.")
    loadp.LoadParams()
    en.SetEnableState(xa.EnableState.Enabled, xa.Timeout.Infinite)


def _distance_unit_type(device: Any, xa: Any) -> Any:
    u = _cast(device, _iface(xa, "IUnitConverter"))
    items = _cast(device, _iface(xa, "IStatusItems"))
    if u is None or items is None:
        raise RuntimeError("IUnitConverter or IStatusItems missing.")
    si = items.GetStatusItem(xa.StatusItemId.Position)
    if not si.IsInteger:
        raise RuntimeError("Position status item must be integer-typed.")
    pos = int(si.GetInteger())
    res = u.FromDeviceUnitToPhysical(xa.ScaleType.Distance, pos)
    ut = getattr(res, "UnitType", None) or getattr(res, "unitType", None)
    if ut is None:
        raise RuntimeError("UnitConversionResult has no UnitType.")
    return ut


def _unit_type_label(ut: Any) -> str:
    try:
        return str(ut)
    except Exception:
        return repr(ut)


def _travel_limits(device: Any, xa: Any) -> tuple[float, float] | None:
    iprod = _cast(device, _iface(xa, "IConnectedProduct"))
    if iprod is None:
        return None
    info = iprod.GetConnectedProductInfo()
    lo = float(info.MinPosition if hasattr(info, "MinPosition") else info.minPosition)
    hi = float(info.MaxPosition if hasattr(info, "MaxPosition") else info.maxPosition)
    return (min(lo, hi), max(lo, hi))


def read_position_mm(device: Any, xa: Any) -> float:
    u = _cast(device, _iface(xa, "IUnitConverter"))
    items = _cast(device, _iface(xa, "IStatusItems"))
    si = items.GetStatusItem(xa.StatusItemId.Position)
    dev = int(si.GetInteger())
    res = u.FromDeviceUnitToPhysical(xa.ScaleType.Distance, dev)
    return float(res.Value)


def absolute_move_mm(device: Any, xa: Any, target_mm: float, distance_ut: Any) -> None:
    u = _cast(device, _iface(xa, "IUnitConverter"))
    mv = _cast(device, _iface(xa, "IMove"))
    du = int(u.FromPhysicalToDeviceUnit(xa.ScaleType.Distance, distance_ut, float(abs(target_mm))))
    if target_mm < 0:
        du = -du
    if du < -(2**31) or du > 2**31 - 1:
        raise ValueError("Absolute position does not fit int32 for IMove.Move.")
    mv.Move(xa.MoveMode.Absolute, du, xa.Timeout.Infinite)


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _retreat_for_target(nominal_mm: float, retreat_offset_mm: float, lim: tuple[float, float] | None) -> float:
    r = nominal_mm - retreat_offset_mm
    if lim is not None:
        lo, hi = lim
        r = max(lo, min(hi, r))
        if abs(r - nominal_mm) < 1e-9:
            r = max(lo, min(hi, nominal_mm + retreat_offset_mm))
    return r


def run_study(
    out_root: Path,
    *,
    xa: Any,
    device: Any,
    distance_ut: Any,
) -> Path:
    lim = _travel_limits(device, xa)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{uuid.uuid4().hex[:8]}"
    out_dir = out_root / f"repeatability_{DEVICE_SERIAL}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "study_protocol_name": "thorlabs_xa_repeatability_unidirectional_retreat",
        "study_protocol_version": "1.0",
        "device_serial": DEVICE_SERIAL,
        "target_positions": list(TARGET_POSITIONS_MM),
        "num_cycles": NUM_CYCLES,
        "retreat_offset": RETREAT_OFFSET_MM,
        "settle_after_retreat_s": SETTLE_AFTER_RETREAT_S,
        "settle_after_nominal_s": SETTLE_AFTER_NOMINAL_S,
        "samples_per_point": SAMPLES_PER_POINT,
        "sample_interval_s": SAMPLE_INTERVAL_S,
        "distance_unit_type": _unit_type_label(distance_ut),
        "travel_limits_raw": list(lim) if lim else None,
        "thorlabs_dotnet_dir": _dotnet_dir(),
    }

    manifest = {
        "run_uuid": str(uuid.uuid4()),
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": _git_commit(),
        "script": str(Path(__file__).resolve()),
        "protocol": protocol,
    }

    csv_path = out_dir / "samples.csv"
    fieldnames = [
        "utc_iso",
        "global_sequence",
        "cycle_index",
        "target_index",
        "target_nominal_mm",
        "retreat_mm",
        "position_measured_mm",
        "residual_mm",
        "sample_within_point",
    ]
    measurements: list[dict[str, Any]] = []
    seq = 0

    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()

        for cycle in range(NUM_CYCLES):
            for ti, nominal in enumerate(TARGET_POSITIONS_MM):
                retreat = _retreat_for_target(float(nominal), RETREAT_OFFSET_MM, lim)
                absolute_move_mm(device, xa, retreat, distance_ut)
                time.sleep(SETTLE_AFTER_RETREAT_S)
                absolute_move_mm(device, xa, float(nominal), distance_ut)
                time.sleep(SETTLE_AFTER_NOMINAL_S)

                for s in range(SAMPLES_PER_POINT):
                    if s > 0:
                        time.sleep(SAMPLE_INTERVAL_S)
                    pos = read_position_mm(device, xa)
                    residual = pos - float(nominal)
                    seq += 1
                    row = {
                        "utc_iso": datetime.now(timezone.utc).isoformat(),
                        "global_sequence": seq,
                        "cycle_index": cycle + 1,
                        "target_index": ti,
                        "target_nominal_mm": nominal,
                        "retreat_mm": retreat,
                        "position_measured_mm": pos,
                        "residual_mm": residual,
                        "sample_within_point": s + 1,
                    }
                    w.writerow(row)
                    measurements.append(row)
                    fcsv.flush()

    # Summary per (target_index, target_nominal_mm)
    by_target: dict[tuple[int, float], list[float]] = {}
    for m in measurements:
        key = (int(m["target_index"]), float(m["target_nominal_mm"]))
        by_target.setdefault(key, []).append(float(m["position_measured_mm"]))

    summary_targets = []
    for (ti, nom), vals in sorted(by_target.items()):
        n = len(vals)
        mean = statistics.fmean(vals)
        stdev = statistics.stdev(vals) if n > 1 else 0.0
        summary_targets.append(
            {
                "target_index": ti,
                "target_nominal_mm": nom,
                "n": n,
                "mean_mm": mean,
                "std_sample_mm": stdev,
                "min_mm": min(vals),
                "max_mm": max(vals),
                "range_mm": max(vals) - min(vals),
                "mean_residual_mm": mean - nom,
            }
        )

    manifest["utc_end"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_targets"] = summary_targets
    manifest["csv_file"] = "samples.csv"

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"targets": summary_targets, "protocol": protocol}, indent=2),
        encoding="utf-8",
    )

    return out_dir


def main() -> int:
    if sys.platform != "win32":
        print("This script requires Windows.", file=sys.stderr)
        return 1
    try:
        xa = _load_xa()
    except (FileNotFoundError, ImportError) as e:
        print(e, file=sys.stderr)
        return 1

    _default_logs = _REPO_ROOT / "repeatability_logs"
    out_root = Path(
        os.environ.get("THORLABS_REPEATABILITY_LOG_DIR", str(_default_logs))
    ).resolve()

    sm = None
    device = None
    try:
        sm = xa.SystemManager.Create()
        sm.Startup()
        device = _try_open_device(sm, DEVICE_SERIAL, xa)
        _prepare_device(device, xa)
        distance_ut = _distance_unit_type(device, xa)
        print(
            f"Repeatability study: serial={DEVICE_SERIAL}, cycles={NUM_CYCLES}, "
            f"targets={len(TARGET_POSITIONS_MM)}, logging under {out_root}",
            flush=True,
        )
        out_dir = run_study(out_root, xa=xa, device=device, distance_ut=distance_ut)
        print(f"Complete. Data directory: {out_dir}", flush=True)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        if device is not None:
            _close_device(xa, device)
        if sm is not None:
            try:
                sm.Shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
