"""
optics_server.py
----------------
MCP server for tilt-mirror actuator control in the diamond spectroscopy setup.

Each mirror has two angular degrees of freedom:
  - theta : vertical tilt   (pitch)
  - phi   : horizontal tilt (yaw)

Mirror indexing is 0-based and consistent across all tools.
All angles are in millidegrees (mdeg) to allow fine-grained control.

Hardware (optional): Thorlabs Motion Control XA **.NET Framework** API via **pythonnet**
(``Thorlabs.MotionControl.XA``), matching the official C# examples under
``SDK\\.NET Framework (C#)\\Examples`` (``Program.cs``, ``ExamplesUsingInterfaces.cs``)
with assemblies loaded from ``SDK\\.NET Framework (C#)\\Libraries\\x64``:
``SystemManager.Create``, ``Startup``, ``TryOpenDevice``, ``ILoadParams.LoadParams``,
``IEnableState.SetEnableState(Enabled, Timeout.Infinite)``, ``IUnitConverter`` distance
conversion, ``IMove.Move`` (``MoveMode.Absolute`` / ``MoveMode.RelativeMove``),
``IStatusItems.GetStatusItem(StatusItemId.Position)``, ``IConnectedProduct.GetConnectedProductInfo``
for travel limits. Requires a 64-bit Windows Python and the Thorlabs .NET assemblies from the
XA SDK (see ``OPTICS_THORLABS_DOTNET_DIR``).

Configuration: all knobs are read once at import into ``settings`` (:class:`OpticsSettings`).
Change environment variables and ``importlib.reload(optics_server)`` (or restart the process)
to apply updates. Variables:

  OPTICS_XA_SERIALS    — comma-separated XA device serial numbers. Either:
                         • NUM_MIRRORS serials: one hardware axis per mirror (maps to theta; phi is
                           software-only for that mirror), or
                         • 2 * NUM_MIRRORS serials: theta then phi per mirror.
                         If unset, defaults to this repo bench (single mirror, one mm axis).
  OPTICS_THORLABS_DOTNET_DIR — folder containing ``tlmc_xa_dotnet.dll`` and
                         dependency assemblies. If unset, the default is the XA SDK x64 library
                         folder (``…\\SDK\\.NET Framework (C#)\\Libraries\\x64``).
  OPTICS_SIMULATE      — if 1/true, use in-memory stub (no .NET load), even on Windows.
  OPTICS_XA_ANGLE_MODE — "linear_mm" (default) or "degrees" for how MCP millidegrees map to
                         physical values passed to ``IUnitConverter``:
                         linear_mm: mdeg → mm via OPTICS_MDEG_TO_MM (default 0.001 ⇒ 1000 mdeg = 1 mm).
                         degrees: mdeg / 1000 → degrees.
                         Choose the mode that matches the device's distance ``UnitType`` from the
                         API (typically ``linear_mm`` for linear stages reporting millimetres).

  The above map to fields on :class:`OpticsSettings` (``simulate``, ``angle_mode``,
  ``mdeg_to_mm_factor``, ``xa_serials_raw``, ``thorlabs_dotnet_dir``). No other code path
  should read these env keys for optics behaviour.

Exposed tools (for LLM use):
  1. get_mirror_state        - read current (theta, phi) of one mirror
  2. get_all_mirrors_state   - read (theta, phi) of every mirror at once
  3. set_mirror_angle        - set absolute (theta, phi) of one mirror
  4. set_all_mirrors_angles  - set absolute (theta, phi) of every mirror at once
  5. step_mirror_angle       - apply a relative delta to one mirror
  6. step_all_mirrors_angles - apply relative deltas to every mirror at once
  7. get_mirror_limits       - query the safe angle bounds for one mirror
  8. home_mirror             - move one mirror to its home position (0, 0)
  9. home_all_mirrors        - move all mirrors to home position (0, 0)
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

try:
    from mcp_servers.dotnet_cast import cast_clr_iface
except ImportError:
    from dotnet_cast import cast_clr_iface

THORLABS_XA_MANAGED_DLL = "tlmc_xa_dotnet.dll"

# Default managed assembly directory (Thorlabs XA SDK, .NET x64 — see SDK layout).
_DEFAULT_THORLABS_XA_DOTNET_DIR = (
    r"C:\Program Files\Thorlabs XA\SDK\.NET Framework (C#)\Libraries\x64"
)

# ---------------------------------------------------------------------------
# Configuration – edit these to match your hardware
# ---------------------------------------------------------------------------

# Default bench: one Thorlabs channel (27272875), travel commanded in mm — matches
# Miscellaneous/thorlabs_xa_simple_motion_test.py. MCP theta drives that axis; phi is
# software-only unless you add a second serial per mirror (see OPTICS_XA_SERIALS).
NUM_MIRRORS = 1
SETTLE_MS = 200

DEFAULT_OPTICS_XA_SERIALS = "27272875"

THETA_MIN = -5_000.0
THETA_MAX = 5_000.0
PHI_MIN = -5_000.0
PHI_MAX = 5_000.0

# mdeg → mm for OPTICS_XA_ANGLE_MODE=linear_mm (1000 mdeg = 1 mm by default)
_DEFAULT_MDEG_TO_MM = 0.001


def _truthy_string(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "on")


def _truthy_env(name: str) -> bool:
    """Read any env flag by name (live lookup; used by tests and ad-hoc checks)."""
    return _truthy_string(os.environ.get(name, ""))


def _normalize_angle_mode(raw: str) -> str:
    m = raw.strip().lower()
    return "degrees" if m == "degrees" else "linear_mm"


def _resolve_thorlabs_dotnet_dir() -> str:
    """
    Directory containing ``tlmc_xa_dotnet.dll`` and sibling Thorlabs assemblies.

    ``OPTICS_THORLABS_DOTNET_DIR`` overrides when set. Otherwise the XA SDK .NET x64
    library folder is used if the managed DLL is present there. Returns ``\"\"`` if not
    found (hardware init will fail with a clear message).
    """
    env = os.environ.get("OPTICS_THORLABS_DOTNET_DIR", "").strip()
    if env:
        return env.rstrip("\\/")
    d = _DEFAULT_THORLABS_XA_DOTNET_DIR
    if os.path.isfile(os.path.join(d, THORLABS_XA_MANAGED_DLL)):
        return d
    return ""


@dataclass(frozen=True)
class OpticsSettings:
    """
    Process configuration snapshot from environment (see module docstring).

    Built once when the module loads; ``importlib.reload(optics_server)`` rebuilds it.
    """

    simulate: bool
    angle_mode: str
    mdeg_to_mm_factor: float
    xa_serials_raw: str
    thorlabs_dotnet_dir: str

    @classmethod
    def from_env(cls) -> OpticsSettings:
        return cls(
            simulate=_truthy_string(os.environ.get("OPTICS_SIMULATE", "")),
            angle_mode=_normalize_angle_mode(os.environ.get("OPTICS_XA_ANGLE_MODE", "linear_mm")),
            mdeg_to_mm_factor=float(os.environ.get("OPTICS_MDEG_TO_MM", str(_DEFAULT_MDEG_TO_MM))),
            xa_serials_raw=os.environ.get("OPTICS_XA_SERIALS", DEFAULT_OPTICS_XA_SERIALS),
            thorlabs_dotnet_dir=_resolve_thorlabs_dotnet_dir(),
        )

    @property
    def parsed_serials(self) -> List[str]:
        raw = self.xa_serials_raw.strip()
        if not raw:
            return []
        return [s.strip() for s in raw.split(",") if s.strip()]


settings = OpticsSettings.from_env()


def _angle_mode() -> str:
    """Current module angle mode (from ``settings``)."""
    return settings.angle_mode


def _mdeg_to_mm(mdeg: float) -> float:
    return mdeg * settings.mdeg_to_mm_factor


def _mm_to_mdeg(mm: float) -> float:
    f = settings.mdeg_to_mm_factor
    if f == 0:
        return 0.0
    return mm / f


def _mdeg_to_deg(mdeg: float) -> float:
    return mdeg / 1000.0


def _deg_to_mdeg(deg: float) -> float:
    return deg * 1000.0


def _mdeg_to_physical(mdeg: float, mode: str) -> float:
    return _mdeg_to_deg(mdeg) if mode == "degrees" else _mdeg_to_mm(mdeg)


def _physical_to_mdeg(physical: float, mode: str) -> float:
    return _deg_to_mdeg(physical) if mode == "degrees" else _mm_to_mdeg(physical)


_xa_clr_lock = threading.Lock()
_xa_clr_module: Any = None


def _dotnet_xa_dll_path() -> str:
    d = settings.thorlabs_dotnet_dir
    if not d:
        raise FileNotFoundError(
            "Thorlabs XA .NET assemblies not found. Set OPTICS_THORLABS_DOTNET_DIR to the folder "
            f"containing {THORLABS_XA_MANAGED_DLL} (expected under the XA SDK "
            r"'.NET Framework (C#)\Libraries\x64' unless you override)."
        )
    p = os.path.join(d, THORLABS_XA_MANAGED_DLL)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"{THORLABS_XA_MANAGED_DLL} not found under {d!r} (check OPTICS_THORLABS_DOTNET_DIR)."
        )
    return p


def _ensure_xa_clr_loaded() -> Any:
    """Load ``Thorlabs.MotionControl.XA`` once via pythonnet; return the XA module object."""
    global _xa_clr_module
    with _xa_clr_lock:
        if _xa_clr_module is not None:
            return _xa_clr_module
        try:
            import clr  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "pythonnet is required for Thorlabs XA hardware control. "
                "Install with: pip install pythonnet (64-bit Python to match Thorlabs assemblies)."
            ) from e
        dll_path = _dotnet_xa_dll_path()
        lib_dir = os.path.dirname(os.path.abspath(dll_path))
        if sys.platform == "win32":
            os.add_dll_directory(lib_dir)
        clr.AddReference(dll_path)
        import Thorlabs.MotionControl.XA as xa  # type: ignore[import-not-found]

        _xa_clr_module = xa
        return xa


def _xa_iface(xa: Any, name: str) -> Any:
    if hasattr(xa, name):
        return getattr(xa, name)
    import importlib

    df = importlib.import_module("Thorlabs.MotionControl.XA.DeviceFeatures")
    return getattr(df, name)


class _ThorlabsXAOptics:
    """
    One ``IDevice`` per hardware axis (theta / phi layout), using the Thorlabs XA .NET API
    (see SDK ``.NET Framework (C#)\\Examples``).
    """

    def __init__(self, serials: List[str]) -> None:
        self._serials = serials
        n = NUM_MIRRORS
        if len(serials) == 2 * n:
            self._layout = "full"
        elif len(serials) == n:
            self._layout = "theta_only"
        else:
            raise ValueError(
                f"Expected {n} serial(s) (theta-only per mirror) or {2 * n} (theta+phi), "
                f"got {len(serials)}."
            )
        self._mode = settings.angle_mode
        self._xa: Any = None
        self._system_manager: Any = None
        self._handle_per_axis: List[Optional[Any]] = [None] * (2 * n)
        self._lock = threading.Lock()
        # ``UnitType`` for distance conversions, from ``UnitConverter`` example (ExamplesUsingInterfaces.UnitConverter).
        self._distance_unit_by_device: dict[int, Any] = {}

    def _bind_dotnet_enums_and_ifaces(self) -> None:
        xa = self._xa
        self._OperatingModes = xa.OperatingModes
        self._MoveMode = xa.MoveMode
        self._ScaleType = xa.ScaleType
        self._Unit = xa.Unit
        self._EnableState = xa.EnableState
        self._StatusItemId = xa.StatusItemId
        self._Timeout = xa.Timeout
        self._ILoadParams = _xa_iface(xa, "ILoadParams")
        self._IEnableState = _xa_iface(xa, "IEnableState")
        self._IMove = _xa_iface(xa, "IMove")
        self._IStatusItems = _xa_iface(xa, "IStatusItems")
        self._IUnitConverter = _xa_iface(xa, "IUnitConverter")
        self._IConnectedProduct = _xa_iface(xa, "IConnectedProduct")
        self._IDisconnect = _xa_iface(xa, "IDisconnect")

    @staticmethod
    def _clr_enum_equals(a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        try:
            return bool(a.Equals(b))
        except Exception:
            return a == b or int(a) == int(b)

    def _try_open_device(self, serial: str) -> Any:
        """
        Match ``Program.cs``: ``TryOpenDevice(id, \"\", OperatingModes.Default, out device)``.
        If the device is unavailable the API returns false; do not call ``OpenDevice`` (that throws
        ``XADeviceNotFoundException`` per ``OpenOptionsExample``).
        """
        sm = self._system_manager
        om = self._OperatingModes.Default
        try:
            result = sm.TryOpenDevice(serial, "", om, None)
        except TypeError:
            result = sm.TryOpenDevice(serial, "", om)
        if isinstance(result, tuple):
            if len(result) >= 2:
                ok, device = bool(result[0]), result[1]
            else:
                ok, device = True, result[0]
        else:
            ok, device = True, result
        if not ok or device is None:
            raise RuntimeError(
                f"XA device unavailable for serial {serial!r} (TryOpenDevice returned false). "
                "Check power, USB, and that the serial appears in SystemManager.GetDeviceList() "
                "(see SDK Examples\\Program.cs GetDeviceListExample)."
            )
        return device

    def _close_one_device(self, device: Any) -> None:
        """``Disconnect`` then ``Close`` — see ``ExamplesUsingInterfaces.Disconnect`` and ``Program.cs``."""
        if device is None:
            return
        disc = cast_clr_iface(device, self._IDisconnect)
        if disc is not None:
            try:
                disc.Disconnect()
            except Exception:
                pass
        try:
            device.Close()
        except Exception:
            pass
        self._distance_unit_by_device.pop(id(device), None)

    def _converter_distance_unit_type(self, device: Any) -> Any:
        """
        ``UnitConverter`` example: ``FromPhysicalToDeviceUnit`` uses ``UnitType`` from the result of
        ``FromDeviceUnitToPhysical`` for the same scale (Distance).
        """
        key = id(device)
        if key in self._distance_unit_by_device:
            return self._distance_unit_by_device[key]
        uconv = cast_clr_iface(device, self._IUnitConverter)
        if uconv is None:
            raise RuntimeError("IUnitConverter not available on device.")
        items = cast_clr_iface(device, self._IStatusItems)
        if items is None:
            raise RuntimeError("IStatusItems not available on device.")
        status_item = items.GetStatusItem(self._StatusItemId.Position)
        if not status_item.IsInteger:
            raise RuntimeError(
                "Position status item is not integer-typed (SDK GetStatus example expects int64 position)."
            )
        dev_pos = int(status_item.GetInteger())
        res = uconv.FromDeviceUnitToPhysical(self._ScaleType.Distance, dev_pos)
        ut = getattr(res, "UnitType", None)
        if ut is None:
            ut = getattr(res, "unitType", None)
        if ut is None:
            raise RuntimeError("UnitConversionResult has no UnitType; cannot match SDK UnitConverter pattern.")
        self._distance_unit_by_device[key] = ut
        return ut

    def _physical_scalar_to_mdeg(self, value: float, unit_type: Any) -> float:
        """Map converter physical ``Value`` (in ``UnitType``) to MCP millidegrees."""
        if self._clr_enum_equals(unit_type, self._Unit.Millimetres):
            return _mm_to_mdeg(value)
        if self._clr_enum_equals(unit_type, self._Unit.Degrees):
            return _deg_to_mdeg(value)
        return _physical_to_mdeg(value, self._mode)

    def _open_serial(self, serial: str) -> Any:
        try:
            device = self._try_open_device(serial)
        except Exception as e:
            raise RuntimeError(f"Could not open device serial {serial!r}: {e}") from e

        try:
            loadp = cast_clr_iface(device, self._ILoadParams)
            if loadp is None:
                raise RuntimeError(f"ILoadParams not supported for serial {serial!r}.")
            loadp.LoadParams()

            en = cast_clr_iface(device, self._IEnableState)
            if en is None:
                raise RuntimeError(f"IEnableState not supported for serial {serial!r}.")
            en.SetEnableState(self._EnableState.Enabled, self._Timeout.Infinite)
            self._converter_distance_unit_type(device)
            return device
        except Exception:
            self._close_one_device(device)
            raise

    def connect(self) -> None:
        if any(h is not None for h in self._handle_per_axis):
            return
        self._xa = _ensure_xa_clr_loaded()
        self._bind_dotnet_enums_and_ifaces()
        sm = self._xa.SystemManager.Create()
        sm.Startup()
        self._system_manager = sm
        opened: List[Any] = []
        n = NUM_MIRRORS
        try:
            if self._layout == "full":
                for k, serial in enumerate(self._serials):
                    dev = self._open_serial(serial)
                    opened.append(dev)
                    self._handle_per_axis[k] = dev
            else:
                for i, serial in enumerate(self._serials):
                    dev = self._open_serial(serial)
                    opened.append(dev)
                    self._handle_per_axis[2 * i] = dev
        except Exception:
            for dev in opened:
                self._close_one_device(dev)
            self._handle_per_axis = [None] * (2 * n)
            try:
                sm.Shutdown()
            except Exception:
                pass
            self._system_manager = None
            self._xa = None
            raise

    def close(self) -> None:
        sm = self._system_manager
        if sm is None:
            return
        seen: set[int] = set()
        for dev in self._handle_per_axis:
            if dev is not None and id(dev) not in seen:
                seen.add(id(dev))
                self._close_one_device(dev)
        self._handle_per_axis = [None] * (2 * NUM_MIRRORS)
        self._distance_unit_by_device.clear()
        try:
            sm.Shutdown()
        except Exception:
            pass
        self._system_manager = None
        self._xa = None

    def _convert_to_device(self, device: Any, physical: float) -> int:
        uconv = cast_clr_iface(device, self._IUnitConverter)
        if uconv is None:
            raise RuntimeError("IUnitConverter not available on device.")
        ut = self._converter_distance_unit_type(device)
        du = int(uconv.FromPhysicalToDeviceUnit(self._ScaleType.Distance, ut, float(abs(physical))))
        if physical < 0:
            du = -du
        if du < -(2**31) or du > 2**31 - 1:
            raise ValueError("Converted position does not fit int32 for IMove.Move.")
        return du

    def _move_relative_device(self, device: Any, delta_device: int) -> None:
        mv = cast_clr_iface(device, self._IMove)
        if mv is None:
            raise RuntimeError("IMove not available on device.")
        mv.Move(self._MoveMode.RelativeMove, int(delta_device), self._Timeout.Infinite)

    def _move_absolute_device(self, device: Any, target_device: int) -> None:
        mv = cast_clr_iface(device, self._IMove)
        if mv is None:
            raise RuntimeError("IMove not available on device.")
        mv.Move(self._MoveMode.Absolute, int(target_device), self._Timeout.Infinite)

    def read_axis_mdeg(self, axis_index: int) -> float:
        device = self._handle_per_axis[axis_index]
        if device is None:
            raise RuntimeError("read_axis_mdeg called for software-only axis")
        items = cast_clr_iface(device, self._IStatusItems)
        if items is None:
            raise RuntimeError("IStatusItems not available on device.")
        status_item = items.GetStatusItem(self._StatusItemId.Position)
        if not status_item.IsInteger:
            raise RuntimeError("Position status item is not integer-typed.")
        dev = int(status_item.GetInteger())
        uconv = cast_clr_iface(device, self._IUnitConverter)
        if uconv is None:
            raise RuntimeError("IUnitConverter not available on device.")
        res = uconv.FromDeviceUnitToPhysical(self._ScaleType.Distance, dev)
        phys = float(res.Value)
        ut = getattr(res, "UnitType", None) or getattr(res, "unitType", None)
        if ut is None:
            return _physical_to_mdeg(phys, self._mode)
        return self._physical_scalar_to_mdeg(phys, ut)

    def move_axis_absolute_mdeg(self, axis_index: int, target_mdeg: float) -> None:
        device = self._handle_per_axis[axis_index]
        if device is None:
            raise RuntimeError("move_axis_absolute_mdeg called for software-only axis")
        phys = _mdeg_to_physical(target_mdeg, self._mode)
        dev = self._convert_to_device(device, phys)
        self._move_absolute_device(device, dev)

    def move_axis_relative_mdeg(self, axis_index: int, delta_mdeg: float) -> None:
        device = self._handle_per_axis[axis_index]
        if device is None:
            raise RuntimeError("move_axis_relative_mdeg called for software-only axis")
        phys = _mdeg_to_physical(delta_mdeg, self._mode)
        steps = self._convert_to_device(device, phys)
        self._move_relative_device(device, steps)

    def axis_limits_mdeg(self, axis_index: int) -> tuple[float, float]:
        device = self._handle_per_axis[axis_index]
        if device is None:
            if axis_index % 2 == 0:
                return (THETA_MIN, THETA_MAX)
            return (PHI_MIN, PHI_MAX)
        iprod = cast_clr_iface(device, self._IConnectedProduct)
        if iprod is None:
            raise RuntimeError("IConnectedProduct not available on device.")
        info = iprod.GetConnectedProductInfo()
        if hasattr(info, "MinPosition"):
            lo_u = float(info.MinPosition)
            hi_u = float(info.MaxPosition)
        else:
            lo_u = float(info.minPosition)
            hi_u = float(info.maxPosition)
        ut = getattr(info, "UnitType", None) or getattr(info, "unitType", None)
        if ut is not None:
            lo = self._physical_scalar_to_mdeg(lo_u, ut)
            hi = self._physical_scalar_to_mdeg(hi_u, ut)
        else:
            lo = _physical_to_mdeg(lo_u, self._mode)
            hi = _physical_to_mdeg(hi_u, self._mode)
        return (min(lo, hi), max(lo, hi))


# ---------------------------------------------------------------------------
# Backend selection: Thorlabs XA vs in-memory simulation
# ---------------------------------------------------------------------------

_mirror_states: List[dict] = [{"theta": 0.0, "phi": 0.0} for _ in range(NUM_MIRRORS)]
_hw: Optional[_ThorlabsXAOptics] = None
_hw_init_error: Optional[str] = None
_backend_lock = threading.Lock()


def _use_simulation() -> bool:
    if settings.simulate:
        return True
    if sys.platform != "win32":
        return True
    serials = settings.parsed_serials
    n = NUM_MIRRORS
    if len(serials) not in (n, 2 * n):
        return True
    return False


def _get_hw() -> Optional[_ThorlabsXAOptics]:
    global _hw, _hw_init_error
    if _use_simulation():
        return None
    with _backend_lock:
        if _hw is not None:
            return _hw
        if _hw_init_error is not None:
            return None
        serials = settings.parsed_serials
        try:
            dev = _ThorlabsXAOptics(serials)
            dev.connect()
            _hw = dev

            def _shutdown_xa_dotnet() -> None:
                h = _hw
                if h is not None:
                    h.close()

            atexit.register(_shutdown_xa_dotnet)
            return _hw
        except Exception as e:
            _hw_init_error = str(e)
            return None


def _validate_mirror_index(mirror_index: int) -> None:
    if not (0 <= mirror_index < NUM_MIRRORS):
        raise ValueError(
            f"mirror_index {mirror_index} is out of range. "
            f"Valid indices are 0 to {NUM_MIRRORS - 1}."
        )


def _validate_angles(theta: float, phi: float) -> None:
    if not (THETA_MIN <= theta <= THETA_MAX):
        raise ValueError(
            f"theta={theta} mdeg is outside safe limits [{THETA_MIN}, {THETA_MAX}] mdeg."
        )
    if not (PHI_MIN <= phi <= PHI_MAX):
        raise ValueError(
            f"phi={phi} mdeg is outside safe limits [{PHI_MIN}, {PHI_MAX}] mdeg."
        )


def _axis_theta(mirror_index: int) -> int:
    return 2 * mirror_index


def _axis_phi(mirror_index: int) -> int:
    return 2 * mirror_index + 1


def _move_mirror(mirror_index: int, theta: float, phi: float, settle_ms: int) -> None:
    hw = _get_hw()
    if hw is None:
        if not _use_simulation() and _hw_init_error:
            raise RuntimeError(f"Thorlabs XA not available: {_hw_init_error}")
        _mirror_states[mirror_index]["theta"] = theta
        _mirror_states[mirror_index]["phi"] = phi
        time.sleep(settle_ms / 1000.0)
        return
    with hw._lock:
        hw.move_axis_absolute_mdeg(_axis_theta(mirror_index), theta)
        if hw._layout == "full":
            hw.move_axis_absolute_mdeg(_axis_phi(mirror_index), phi)
        else:
            _mirror_states[mirror_index]["phi"] = phi
        _mirror_states[mirror_index]["theta"] = hw.read_axis_mdeg(_axis_theta(mirror_index))
        if hw._layout == "full":
            _mirror_states[mirror_index]["phi"] = hw.read_axis_mdeg(_axis_phi(mirror_index))
    time.sleep(settle_ms / 1000.0)


def _read_mirror(mirror_index: int) -> dict:
    hw = _get_hw()
    if hw is None:
        if not _use_simulation() and _hw_init_error:
            raise RuntimeError(f"Thorlabs XA not available: {_hw_init_error}")
        return dict(_mirror_states[mirror_index])
    with hw._lock:
        theta = hw.read_axis_mdeg(_axis_theta(mirror_index))
        if hw._layout == "full":
            phi = hw.read_axis_mdeg(_axis_phi(mirror_index))
        else:
            phi = _mirror_states[mirror_index]["phi"]
    return {"theta": theta, "phi": phi}


def _mirror_limits_from_hw(mirror_index: int) -> Optional[dict]:
    hw = _get_hw()
    if hw is None:
        return None
    with hw._lock:
        t_lo, t_hi = hw.axis_limits_mdeg(_axis_theta(mirror_index))
        p_lo, p_hi = hw.axis_limits_mdeg(_axis_phi(mirror_index))
    return {
        "theta_min_mdeg": t_lo,
        "theta_max_mdeg": t_hi,
        "phi_min_mdeg": p_lo,
        "phi_max_mdeg": p_hi,
    }


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="optics_server",
    instructions=(
        "Controls tilt mirrors in the diamond spectroscopy beam path. "
        f"Each mirror is identified by a 0-based integer index (0 to {NUM_MIRRORS - 1}). "
        "Angles are always in millidegrees (mdeg). "
        "Use get_mirror_state or get_all_mirrors_state to read positions before making moves. "
        "Use set_mirror_angle or set_all_mirrors_angles for absolute moves. "
        "Use step_mirror_angle or step_all_mirrors_angles for relative moves. "
        "Always call get_mirror_limits before large moves to avoid hitting hardware bounds. "
        "Call home_mirror or home_all_mirrors to reset to (0, 0). "
        "Hardware: default is one mm axis (theta) per mirror; set env OPTICS_XA_SERIALS to "
        "NUM_MIRRORS serials or 2×NUM_MIRRORS (theta,phi per mirror). "
        "Use OPTICS_SIMULATE=1 or clear OPTICS_XA_SERIALS for offline testing. "
        "Hardware needs Thorlabs .NET assemblies (pythonnet); default path is the XA SDK x64 library folder. "
        "Optics env vars are read once at server process start (restart MCP after changes)."
    ),
)


@mcp.tool(
    description=(
        "Read the current tilt angles of a single mirror. "
        "Returns theta (vertical/pitch) and phi (horizontal/yaw) in millidegrees. "
        f"mirror_index is 0-based integer between 0 and {NUM_MIRRORS - 1}. "
        "Use this before any move to know the current position."
    )
)
def get_mirror_state(mirror_index: int) -> dict:
    _validate_mirror_index(mirror_index)
    state = _read_mirror(mirror_index)
    return {
        "mirror_index": mirror_index,
        "theta_mdeg": state["theta"],
        "phi_mdeg": state["phi"],
    }


@mcp.tool(
    description=(
        "Read the current tilt angles of ALL mirrors in one call. "
        "Returns a list of {mirror_index, theta_mdeg, phi_mdeg} objects. "
        "Preferred over calling get_mirror_state repeatedly when you need "
        "a snapshot of the full optical path."
    )
)
def get_all_mirrors_state() -> List[dict]:
    return [
        {
            "mirror_index": i,
            "theta_mdeg": _read_mirror(i)["theta"],
            "phi_mdeg": _read_mirror(i)["phi"],
        }
        for i in range(NUM_MIRRORS)
    ]


@mcp.tool(
    description=(
        "Set the absolute tilt angles of a single mirror. "
        "theta_mdeg is vertical/pitch, phi_mdeg is horizontal/yaw, both in millidegrees. "
        f"Safe range for both: [{THETA_MIN}, {THETA_MAX}] mdeg. "
        f"mirror_index is 0-based integer between 0 and {NUM_MIRRORS - 1}. "
        f"settle_ms is how long to wait for mechanical settling after the move (default {SETTLE_MS} ms). "
        "Raises an error if angles exceed hardware limits — call get_mirror_limits first if unsure."
    )
)
def set_mirror_angle(
    mirror_index: int,
    theta_mdeg: float,
    phi_mdeg: float,
    settle_ms: int = SETTLE_MS,
) -> dict:
    _validate_mirror_index(mirror_index)
    _validate_angles(theta_mdeg, phi_mdeg)
    _move_mirror(mirror_index, theta_mdeg, phi_mdeg, settle_ms)
    return {
        "mirror_index": mirror_index,
        "theta_mdeg": theta_mdeg,
        "phi_mdeg": phi_mdeg,
        "settled_ms": settle_ms,
    }


class MirrorTarget(BaseModel):
    mirror_index: int = Field(..., description="0-based mirror index.")
    theta_mdeg: float = Field(..., description="Target vertical tilt (pitch) in mdeg.")
    phi_mdeg: float = Field(..., description="Target horizontal tilt (yaw) in mdeg.")


@mcp.tool(
    description=(
        "Set the absolute tilt angles of ALL mirrors in a single call. "
        "Accepts a list of {mirror_index, theta_mdeg, phi_mdeg} objects. "
        "Mirrors are moved sequentially in the order provided. "
        "Use this instead of calling set_mirror_angle in a loop to reduce round-trip latency. "
        f"settle_ms is applied after each individual mirror move (default {SETTLE_MS} ms). "
        "Raises an error if any angle exceeds hardware limits."
    )
)
def set_all_mirrors_angles(
    targets: List[MirrorTarget],
    settle_ms: int = SETTLE_MS,
) -> List[dict]:
    results = []
    for t in targets:
        _validate_mirror_index(t.mirror_index)
        _validate_angles(t.theta_mdeg, t.phi_mdeg)
        _move_mirror(t.mirror_index, t.theta_mdeg, t.phi_mdeg, settle_ms)
        results.append(
            {
                "mirror_index": t.mirror_index,
                "theta_mdeg": t.theta_mdeg,
                "phi_mdeg": t.phi_mdeg,
            }
        )
    return results


@mcp.tool(
    description=(
        "Apply a RELATIVE angular step to a single mirror. "
        "new_angle = current_angle + delta for both axes. "
        "delta_theta_mdeg is the vertical/pitch step, delta_phi_mdeg is the horizontal/yaw step, "
        "both in millidegrees — can be negative. "
        "Safer than set_mirror_angle for small iterative alignment adjustments "
        "because you only specify how much to move, not the absolute target. "
        f"mirror_index is 0-based integer between 0 and {NUM_MIRRORS - 1}. "
        f"settle_ms is wait time after the move (default {SETTLE_MS} ms). "
        "Raises an error if the resulting angle would exceed hardware limits."
    )
)
def step_mirror_angle(
    mirror_index: int,
    delta_theta_mdeg: float,
    delta_phi_mdeg: float,
    settle_ms: int = SETTLE_MS,
) -> dict:
    _validate_mirror_index(mirror_index)
    current = _read_mirror(mirror_index)
    new_theta = current["theta"] + delta_theta_mdeg
    new_phi = current["phi"] + delta_phi_mdeg
    _validate_angles(new_theta, new_phi)
    hw = _get_hw()
    if hw is None:
        if not _use_simulation() and _hw_init_error:
            raise RuntimeError(f"Thorlabs XA not available: {_hw_init_error}")
        _mirror_states[mirror_index]["theta"] = new_theta
        _mirror_states[mirror_index]["phi"] = new_phi
        time.sleep(settle_ms / 1000.0)
    else:
        with hw._lock:
            hw.move_axis_relative_mdeg(_axis_theta(mirror_index), delta_theta_mdeg)
            if hw._layout == "full":
                hw.move_axis_relative_mdeg(_axis_phi(mirror_index), delta_phi_mdeg)
            else:
                _mirror_states[mirror_index]["phi"] = new_phi
        time.sleep(settle_ms / 1000.0)
    final = _read_mirror(mirror_index)
    return {
        "mirror_index": mirror_index,
        "theta_mdeg": final["theta"],
        "phi_mdeg": final["phi"],
        "delta_theta_mdeg": delta_theta_mdeg,
        "delta_phi_mdeg": delta_phi_mdeg,
    }


class MirrorDelta(BaseModel):
    mirror_index: int = Field(..., description="0-based mirror index.")
    delta_theta_mdeg: float = Field(..., description="Relative vertical tilt step in mdeg. Can be negative.")
    delta_phi_mdeg: float = Field(..., description="Relative horizontal tilt step in mdeg. Can be negative.")


@mcp.tool(
    description=(
        "Apply RELATIVE angular steps to ALL mirrors in a single call. "
        "For each mirror: new_angle = current_angle + delta. "
        "Accepts a list of {mirror_index, delta_theta_mdeg, delta_phi_mdeg} objects. "
        "Use this during iterative closed-loop alignment to nudge every mirror simultaneously. "
        f"settle_ms is applied after each individual mirror move (default {SETTLE_MS} ms). "
        "Raises an error if any resulting angle would exceed hardware limits."
    )
)
def step_all_mirrors_angles(
    deltas: List[MirrorDelta],
    settle_ms: int = SETTLE_MS,
) -> List[dict]:
    results = []
    for d in deltas:
        _validate_mirror_index(d.mirror_index)
        current = _read_mirror(d.mirror_index)
        new_theta = current["theta"] + d.delta_theta_mdeg
        new_phi = current["phi"] + d.delta_phi_mdeg
        _validate_angles(new_theta, new_phi)
        hw = _get_hw()
        if hw is None:
            if not _use_simulation() and _hw_init_error:
                raise RuntimeError(f"Thorlabs XA not available: {_hw_init_error}")
            _mirror_states[d.mirror_index]["theta"] = new_theta
            _mirror_states[d.mirror_index]["phi"] = new_phi
            time.sleep(settle_ms / 1000.0)
        else:
            with hw._lock:
                hw.move_axis_relative_mdeg(_axis_theta(d.mirror_index), d.delta_theta_mdeg)
                if hw._layout == "full":
                    hw.move_axis_relative_mdeg(_axis_phi(d.mirror_index), d.delta_phi_mdeg)
                else:
                    _mirror_states[d.mirror_index]["phi"] = new_phi
            time.sleep(settle_ms / 1000.0)
        final = _read_mirror(d.mirror_index)
        results.append(
            {
                "mirror_index": d.mirror_index,
                "theta_mdeg": final["theta"],
                "phi_mdeg": final["phi"],
                "delta_theta_mdeg": d.delta_theta_mdeg,
                "delta_phi_mdeg": d.delta_phi_mdeg,
            }
        )
    return results


@mcp.tool(
    description=(
        "Query the safe hardware angle limits for a single mirror. "
        "Returns min and max allowed values for both theta and phi in millidegrees. "
        "Always call this before large absolute moves to avoid hardware errors. "
        f"mirror_index is 0-based integer between 0 and {NUM_MIRRORS - 1}."
    )
)
def get_mirror_limits(mirror_index: int) -> dict:
    _validate_mirror_index(mirror_index)
    lim = _mirror_limits_from_hw(mirror_index)
    if lim is not None:
        return {"mirror_index": mirror_index, **lim}
    return {
        "mirror_index": mirror_index,
        "theta_min_mdeg": THETA_MIN,
        "theta_max_mdeg": THETA_MAX,
        "phi_min_mdeg": PHI_MIN,
        "phi_max_mdeg": PHI_MAX,
    }


@mcp.tool(
    description=(
        "Move a single mirror to its home position: theta=0, phi=0 mdeg. "
        "Use this to reset a mirror to a known neutral state before a new alignment run or after an error. "
        f"mirror_index is 0-based integer between 0 and {NUM_MIRRORS - 1}. "
        f"settle_ms is wait time after the move (default {SETTLE_MS} ms)."
    )
)
def home_mirror(mirror_index: int, settle_ms: int = SETTLE_MS) -> dict:
    _validate_mirror_index(mirror_index)
    _move_mirror(mirror_index, 0.0, 0.0, settle_ms)
    return {"mirror_index": mirror_index, "theta_mdeg": 0.0, "phi_mdeg": 0.0}


@mcp.tool(
    description=(
        "Move ALL mirrors to their home position: theta=0, phi=0 mdeg. "
        "Use this at the start of a new experiment run or after an emergency stop "
        "to return the full optical path to a known neutral state. "
        "Equivalent to calling home_mirror for every mirror in sequence. "
        f"settle_ms is applied after each mirror move (default {SETTLE_MS} ms)."
    )
)
def home_all_mirrors(settle_ms: int = SETTLE_MS) -> List[dict]:
    results = []
    for i in range(NUM_MIRRORS):
        _move_mirror(i, 0.0, 0.0, settle_ms)
        results.append({"mirror_index": i, "theta_mdeg": 0.0, "phi_mdeg": 0.0})
    return results


if __name__ == "__main__":
    mcp.run()
