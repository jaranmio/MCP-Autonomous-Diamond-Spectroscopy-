"""
Thorlabs XA — relative moves in millimetres via the native C API (ctypes).

Ground truth: Native (C, C++)\\Examples\\main.c and tlmc_xa_native_api.h in the Thorlabs XA SDK:
  - TLMC_Startup(NULL), TLMC_Open(pDevice, NULL, TLMC_OperatingMode_Default, &h)
  - TLMC_LoadParams (main.c calls Example_LoadParams)
  - TLMC_SetEnableState(..., TLMC_Enabled, TLMC_InfiniteWait) — Example_ChangeEnableState
  - TLMC_ConvertFromPhysicalToDevice(..., TLMC_ScaleType_Distance, TLMC_Unit_Millimetres, ...) — Example_UnitConverter
  - TLMC_Move(..., TLMC_MoveMode_Relative, int32_device_units, TLMC_InfiniteWait) — Example_MoveRelative
  - TLMC_Close, TLMC_Shutdown

Requires: 64-bit Python on Windows; TLMC_XA_Native.dll (x64) from the SDK install.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import POINTER, byref, c_int32, c_int64, c_uint16, c_uint32, c_uint8

# --- Same literals as tlmc_xa_native_api.h / main.c ---------------------------------
TLMC_Success = 0

TLMC_OperatingMode_Default = 0x00000100  # TLMC_OperatingMode_Default == TLMC_OperatingMode_Apt

TLMC_MoveMode_Relative = 2  # TLMC_MoveMode_Relative

TLMC_ScaleType_Distance = 1  # TLMC_ScaleType_Distance
TLMC_Unit_Millimetres = 1  # TLMC_Unit_Millimetres

TLMC_Enabled = 0x01  # TLMC_Enabled

TLMC_InfiniteWait = -1  # TLMC_InfiniteWait

# Device serial for TLMC_Open (main.c passes a string literal; no device-list step).
DEVICE_SERIAL = "27272875"

# Single-shot demo: relative move in millimetres (used by run_single_motion).
RELATIVE_MOVE_MM = 2.0

# Oscillation: ± this distance (mm), this many full back-and-forth cycles (each cycle = +A then −A).
OSCILLATION_AMPLITUDE_MM = 1.0
OSCILLATION_CYCLES = 20
OSCILLATION_PAUSE_S = 0.0  # optional pause between half-strokes

# Default x64 native DLL (SDK layout).
DEFAULT_DLL = r"C:\Program Files\Thorlabs XA\SDK\Native (C, C++)\Libraries\x64\TLMC_XA_Native.dll"


def _load_dll() -> ctypes.WinDLL:
    path = os.environ.get("TLMC_XA_NATIVE_DLL", "").strip() or DEFAULT_DLL
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Native DLL not found: {path}")
    return ctypes.WinDLL(path)


def _bind(dll: ctypes.WinDLL) -> None:
    dll.TLMC_Startup.argtypes = [ctypes.c_char_p]
    dll.TLMC_Startup.restype = c_uint16

    dll.TLMC_Shutdown.argtypes = []
    dll.TLMC_Shutdown.restype = c_uint16

    dll.TLMC_Open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, c_uint32, POINTER(c_uint32)]
    dll.TLMC_Open.restype = c_uint16

    dll.TLMC_Close.argtypes = [c_uint32]
    dll.TLMC_Close.restype = c_uint16

    dll.TLMC_LoadParams.argtypes = [c_uint32]
    dll.TLMC_LoadParams.restype = c_uint16

    dll.TLMC_SetEnableState.argtypes = [c_uint32, c_uint8, c_int64]
    dll.TLMC_SetEnableState.restype = c_uint16

    dll.TLMC_ConvertFromPhysicalToDevice.argtypes = [
        c_uint32,
        c_uint16,
        c_uint16,
        ctypes.c_double,
        POINTER(c_int64),
    ]
    dll.TLMC_ConvertFromPhysicalToDevice.restype = c_uint16

    dll.TLMC_Move.argtypes = [c_uint32, c_uint8, c_int32, c_int64]
    dll.TLMC_Move.restype = c_uint16


def relative_move_mm(dll: ctypes.WinDLL, handle: int, delta_mm: float) -> bool:
    """
    One TLMC_Move in relative mode: convert |delta_mm| mm to device units, apply sign, move.
    """
    sign = 1 if delta_mm >= 0 else -1
    du = c_int64(0)
    if (
        dll.TLMC_ConvertFromPhysicalToDevice(
            handle,
            TLMC_ScaleType_Distance,
            TLMC_Unit_Millimetres,
            ctypes.c_double(abs(delta_mm)),
            byref(du),
        )
        != TLMC_Success
    ):
        print("TLMC_ConvertFromPhysicalToDevice failed.", file=sys.stderr)
        return False

    steps = int(du.value) * sign
    if steps < -(2**31) or steps > 2**31 - 1:
        print("Converted distance does not fit int32 for TLMC_Move.", file=sys.stderr)
        return False

    if (
        dll.TLMC_Move(handle, TLMC_MoveMode_Relative, c_int32(steps), c_int64(TLMC_InfiniteWait))
        != TLMC_Success
    ):
        print("TLMC_Move failed.", file=sys.stderr)
        return False
    return True


def run_single_motion(dll: ctypes.WinDLL, handle: int) -> bool:
    """Single relative move by RELATIVE_MOVE_MM (same as the original one-shot script)."""
    print(f"Single move: {RELATIVE_MOVE_MM} mm (relative).")
    return relative_move_mm(dll, handle, RELATIVE_MOVE_MM)


def oscillate_back_and_forth(
    dll: ctypes.WinDLL,
    handle: int,
    *,
    amplitude_mm: float = OSCILLATION_AMPLITUDE_MM,
    cycles: int = OSCILLATION_CYCLES,
    pause_s: float = OSCILLATION_PAUSE_S,
) -> bool:
    """
    Each cycle: +amplitude_mm, then −amplitude_mm (one full back-and-forth).
    `cycles` is the number of those oscillations (2 * cycles moves total).
    """
    print(f"Oscillate: ±{amplitude_mm} mm, {cycles} cycle(s), pause {pause_s} s between half-strokes.")
    for c in range(cycles):
        if not relative_move_mm(dll, handle, amplitude_mm):
            return False
        if pause_s > 0:
            time.sleep(pause_s)
        if not relative_move_mm(dll, handle, -amplitude_mm):
            return False
        if pause_s > 0:
            time.sleep(pause_s)
    return True


def main() -> int:
    try:
        dll = _load_dll()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    _bind(dll)
    handle = 0

    try:
        if dll.TLMC_Startup(None) != TLMC_Success:
            print("TLMC_Startup failed.", file=sys.stderr)
            return 1

        h = c_uint32(0)
        if (
            dll.TLMC_Open(
                DEVICE_SERIAL.encode("utf-8"),
                None,
                TLMC_OperatingMode_Default,
                byref(h),
            )
            != TLMC_Success
        ):
            print(f"TLMC_Open failed (serial={DEVICE_SERIAL!r}).", file=sys.stderr)
            return 1

        handle = h.value

        if dll.TLMC_LoadParams(handle) != TLMC_Success:
            print("TLMC_LoadParams failed.", file=sys.stderr)
            return 1

        if (
            dll.TLMC_SetEnableState(handle, TLMC_Enabled, c_int64(TLMC_InfiniteWait))
            != TLMC_Success
        ):
            print("TLMC_SetEnableState(Enabled) failed.", file=sys.stderr)
            return 1

        # --- Swap which motion runs (comment one, uncomment the other) -----------------
        ok = oscillate_back_and_forth(dll, handle, pause_s=0.5)
        # ok = oscillate_back_and_forth(dll, handle)
        # ok = run_single_motion(dll, handle)

        if not ok:
            return 1

        print("Done.")
        return 0
    finally:
        if handle:
            dll.TLMC_Close(handle)
        dll.TLMC_Shutdown()


if __name__ == "__main__":
    sys.exit(main())
