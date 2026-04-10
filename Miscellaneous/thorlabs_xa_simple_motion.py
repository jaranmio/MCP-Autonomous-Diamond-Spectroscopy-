"""
Thorlabs XA — relative moves in millimetres via the .NET API (pythonnet).

Aligned with the official SDK sources under
``C:\\Program Files\\Thorlabs XA\\SDK\\.NET Framework (C#)\\``:

  - ``Examples\\Program.cs`` — ``SystemManager.Create()``, ``Startup()``,
    ``TryOpenDevice(id, \"\", OperatingModes.Default, out device)``, ``device.Close()``,
    ``Shutdown()``.
  - ``Examples\\ExamplesUsingInterfaces.cs`` — ``LoadParams``, ``SetEnableState(Enabled, …)``,
    ``IUnitConverter`` (``FromDeviceUnitToPhysical`` then ``FromPhysicalToDeviceUnit`` with
    ``UnitType`` from the conversion result, same as ``UnitConverter`` example),
    ``IMove.Move(MoveMode.RelativeMove, …, Timeout.Infinite)``.

Managed assemblies: ``Libraries\\x64\\tlmc_xa_dotnet.dll`` (override with
``OPTICS_THORLABS_DOTNET_DIR``).

Requires: 64-bit Python on Windows; ``pip install pythonnet``.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from typing import Any

try:
    from mcp_servers.dotnet_cast import cast_clr_iface as _cast
except ImportError:
    _mcp_servers_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_servers"
    )
    if _mcp_servers_dir not in sys.path:
        sys.path.insert(0, _mcp_servers_dir)
    from dotnet_cast import cast_clr_iface as _cast

DEVICE_SERIAL = "27272875"

RELATIVE_MOVE_MM = 2.0
OSCILLATION_AMPLITUDE_MM = 1.0
OSCILLATION_CYCLES = 20
OSCILLATION_PAUSE_S = 0.0

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
            f"Set OPTICS_THORLABS_DOTNET_DIR to the folder containing {THORLABS_XA_MANAGED_DLL} "
            r"(default: XA SDK '.NET Framework (C#)\Libraries\x64')."
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
    """``TryOpenDevice`` only — if false, device unavailable (``Program.cs`` OpenOptionsExample)."""
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
        raise RuntimeError(
            f"TryOpenDevice returned false for serial {serial!r} (device unavailable). "
            "See SDK Examples\\Program.cs GetDeviceListExample / OpenOptionsExample."
        )
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
    ILoadParams = _iface(xa, "ILoadParams")
    IEnableState = _iface(xa, "IEnableState")
    loadp = _cast(device, ILoadParams)
    if loadp is None:
        raise RuntimeError("ILoadParams not supported.")
    loadp.LoadParams()
    en = _cast(device, IEnableState)
    if en is None:
        raise RuntimeError("IEnableState not supported.")
    en.SetEnableState(xa.EnableState.Enabled, xa.Timeout.Infinite)


def _distance_unit_type(device: Any, xa: Any) -> Any:
    """Match ``ExamplesUsingInterfaces.UnitConverter`` (use ``UnitType`` from ``FromDeviceUnitToPhysical``)."""
    IUnitConverter = _iface(xa, "IUnitConverter")
    IStatusItems = _iface(xa, "IStatusItems")
    u = _cast(device, IUnitConverter)
    items = _cast(device, IStatusItems)
    if u is None or items is None:
        raise RuntimeError("IUnitConverter or IStatusItems missing.")
    si = items.GetStatusItem(xa.StatusItemId.Position)
    if not si.IsInteger:
        raise RuntimeError("Position status item is not integer-typed (see SDK GetStatus example).")
    pos = int(si.GetInteger())
    res = u.FromDeviceUnitToPhysical(xa.ScaleType.Distance, pos)
    ut = getattr(res, "UnitType", None) or getattr(res, "unitType", None)
    if ut is None:
        raise RuntimeError("UnitConversionResult has no UnitType.")
    return ut


def relative_move_mm(device: Any, xa: Any, delta_mm: float, distance_ut: Any) -> bool:
    """``MoveRelative`` + ``UnitConverter`` pattern from ``ExamplesUsingInterfaces``."""
    IUnitConverter = _iface(xa, "IUnitConverter")
    IMove = _iface(xa, "IMove")
    u = _cast(device, IUnitConverter)
    mv = _cast(device, IMove)
    if u is None or mv is None:
        print("IUnitConverter or IMove missing.", file=sys.stderr)
        return False
    sign = 1 if delta_mm >= 0 else -1
    try:
        steps = int(
            u.FromPhysicalToDeviceUnit(
                xa.ScaleType.Distance,
                distance_ut,
                float(abs(delta_mm)),
            )
        )
    except Exception as e:
        print(f"FromPhysicalToDeviceUnit failed: {e}", file=sys.stderr)
        return False
    steps *= sign
    if steps < -(2**31) or steps > 2**31 - 1:
        print("Converted distance does not fit int32 for IMove.Move.", file=sys.stderr)
        return False
    try:
        mv.Move(xa.MoveMode.RelativeMove, steps, xa.Timeout.Infinite)
    except Exception as e:
        print(f"IMove.Move failed: {e}", file=sys.stderr)
        return False
    return True


def run_single_motion(device: Any, xa: Any, distance_ut: Any) -> bool:
    print(f"Single move: {RELATIVE_MOVE_MM} mm (relative).")
    return relative_move_mm(device, xa, RELATIVE_MOVE_MM, distance_ut)


def oscillate_back_and_forth(
    device: Any,
    xa: Any,
    distance_ut: Any,
    *,
    amplitude_mm: float = OSCILLATION_AMPLITUDE_MM,
    cycles: int = OSCILLATION_CYCLES,
    pause_s: float = OSCILLATION_PAUSE_S,
) -> bool:
    print(f"Oscillate: ±{amplitude_mm} mm, {cycles} cycle(s), pause {pause_s} s between half-strokes.")
    for _ in range(cycles):
        if not relative_move_mm(device, xa, amplitude_mm, distance_ut):
            return False
        if pause_s > 0:
            time.sleep(pause_s)
        if not relative_move_mm(device, xa, -amplitude_mm, distance_ut):
            return False
        if pause_s > 0:
            time.sleep(pause_s)
    return True


def main() -> int:
    if sys.platform != "win32":
        print("This script requires Windows.", file=sys.stderr)
        return 1
    try:
        xa = _load_xa()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"pythonnet not available: {e}", file=sys.stderr)
        return 1

    sm = None
    device = None
    try:
        sm = xa.SystemManager.Create()
        sm.Startup()
        device = _try_open_device(sm, DEVICE_SERIAL, xa)
        _prepare_device(device, xa)
        distance_ut = _distance_unit_type(device, xa)
        ok = oscillate_back_and_forth(device, xa, distance_ut, pause_s=0.5)
        if not ok:
            return 1
        print("Done.")
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
