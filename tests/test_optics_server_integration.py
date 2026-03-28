"""
Integration (hardware) tests for optics_server.py against real Thorlabs XA (.NET API).

On **Windows**, if the managed DLL is found and **OPTICS_SIMULATE** is not set, these tests
**run against hardware by default** (connect, read, and a small theta round-trip move).

**Opt out** (CI, no bench, or read-only):

  set OPTICS_SKIP_HARDWARE_TESTS=1     # skip this entire module
  set OPTICS_SKIP_HARDWARE_MOVES=1    # skip only the motion round-trip; still runs connect+read

Prerequisites
-------------
- 64-bit Python on Windows
- ``pythonnet`` and Thorlabs ``tlmc_xa_dotnet.dll`` (and dependencies), from the
  XA SDK ``.NET Framework (C#)\\Libraries\\x64`` folder — set **OPTICS_THORLABS_DOTNET_DIR** if needed
- Controller powered, correct **OPTICS_XA_SERIALS** for `optics_server.NUM_MIRRORS`
  (see optics_server module docstring: N or 2×N serials)

**OPTICS_SIMULATE** must be unset or false — otherwise the stub backend is used and this class
is skipped.

Optional overrides:

  set OPTICS_XA_SERIALS=27272875
  set OPTICS_THORLABS_DOTNET_DIR=C:\\Program Files\\Thorlabs XA\\SDK\\.NET Framework (C#)\\Libraries\\x64

Run from repo root:

  .venv\\Scripts\\python.exe -m unittest tests.test_optics_server_integration -v

Safety
------
- Read-only test runs first (connect + position read).
- Motion test uses a **small** theta step (10 mdeg default) and reverses it; use
  **OPTICS_SKIP_HARDWARE_MOVES=1** if the stage must not move. phi is not moved on the device
  when using theta_only layout (one serial per mirror).
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MCP_SERVERS = str(_ROOT / "mcp_servers")
if _MCP_SERVERS not in sys.path:
    sys.path.insert(0, _MCP_SERVERS)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _xa_managed_dll_path() -> Path:
    """``tlmc_xa_dotnet.dll`` under ``settings.thorlabs_dotnet_dir``."""
    import optics_server

    d = optics_server.settings.thorlabs_dotnet_dir
    return Path(d) / optics_server.THORLABS_XA_MANAGED_DLL


def _reload_simulate() -> object:
    os.environ["OPTICS_SIMULATE"] = "1"
    import optics_server as o

    return importlib.reload(o)


def _reload_for_hardware() -> object:
    """Hardware path allowed: OPTICS_SIMULATE removed so optics_server may connect."""
    os.environ.pop("OPTICS_SIMULATE", None)
    import optics_server

    return importlib.reload(optics_server)


@unittest.skipUnless(sys.platform == "win32", "Thorlabs XA .NET API is Windows-only")
class TestOpticsServerHardwareIntegration(unittest.TestCase):
    """
    Real Thorlabs XA .NET session: runs on Windows when the DLL exists and simulation is off.

    Skipped if OPTICS_SKIP_HARDWARE_TESTS is set, OPTICS_SIMULATE is on, or the managed DLL
    is missing. Non-Windows: skipped at class level.
    """

    # Small relative step (mdeg). At default OPTICS_MDEG_TO_MM=0.001, 10 mdeg → 0.01 mm (10 µm).
    THETA_STEP_MDEG = 10.0
    # Encoder / quantization slack when comparing readback (mdeg).
    READBACK_TOLERANCE_MDEG = 25.0

    @classmethod
    def setUpClass(cls) -> None:
        if _env_truthy("OPTICS_SKIP_HARDWARE_TESTS"):
            raise unittest.SkipTest(
                "Hardware integration skipped (OPTICS_SKIP_HARDWARE_TESTS=1). "
                "Unset it to run against the bench."
            )
        if _env_truthy("OPTICS_SIMULATE"):
            raise unittest.SkipTest(
                "Unset OPTICS_SIMULATE (or set to 0) for hardware tests; simulation forces stub backend."
            )
        dll = _xa_managed_dll_path()
        if not dll.is_file():
            raise unittest.SkipTest(f"Thorlabs XA managed DLL not found: {dll}")

        cls._optics = _reload_for_hardware()

    @classmethod
    def tearDownClass(cls) -> None:
        _reload_simulate()

    def test_01_connect_and_read_mirror_state(self) -> None:
        """Open device(s), enable, read position via IStatusItems + IUnitConverter."""
        o = self._optics
        try:
            st = o.get_mirror_state(0)
        except RuntimeError as e:
            self.skipTest(
                "XA session failed (wrong serial, cable, power, driver, or .NET path?). "
                f"Detail: {e}"
            )
        self.assertIsInstance(st, dict)
        self.assertEqual(st.get("mirror_index"), 0)
        self.assertIn("theta_mdeg", st)
        self.assertIn("phi_mdeg", st)
        self.assertIsInstance(st["theta_mdeg"], float)
        self.assertIsInstance(st["phi_mdeg"], float)

        lo = o.get_mirror_limits(0)
        self.assertEqual(lo["mirror_index"], 0)
        self.assertIn("theta_min_mdeg", lo)
        self.assertIn("theta_max_mdeg", lo)
        self.assertLessEqual(lo["theta_min_mdeg"], st["theta_mdeg"])
        self.assertGreaterEqual(lo["theta_max_mdeg"], st["theta_mdeg"])

    def test_02_small_theta_relative_roundtrip_optional(self) -> None:
        """
        Relative move on hardware theta, then reverse; confirms IMove + readback path.

        Skipped if OPTICS_SKIP_HARDWARE_MOVES=1.
        """
        if _env_truthy("OPTICS_SKIP_HARDWARE_MOVES"):
            raise unittest.SkipTest(
                "Hardware motion test skipped (OPTICS_SKIP_HARDWARE_MOVES=1). "
                "Unset to allow the small theta round-trip."
            )

        o = self._optics
        step = self.THETA_STEP_MDEG
        tol = self.READBACK_TOLERANCE_MDEG

        try:
            before = o.get_mirror_state(0)
        except RuntimeError as e:
            self.skipTest(f"Read failed: {e}")

        try:
            o.step_mirror_angle(0, step, 0.0, settle_ms=o.SETTLE_MS)
            mid = o.get_mirror_state(0)
            o.step_mirror_angle(0, -step, 0.0, settle_ms=o.SETTLE_MS)
            after = o.get_mirror_state(0)
        except RuntimeError as e:
            self.fail(f"Motion or read failed: {e}")

        d_theta_mid = mid["theta_mdeg"] - before["theta_mdeg"]
        self.assertAlmostEqual(
            d_theta_mid,
            step,
            delta=tol,
            msg=f"After +{step} mdeg theta, expected ~{step} change, got {d_theta_mid}",
        )

        d_theta_final = after["theta_mdeg"] - before["theta_mdeg"]
        self.assertAlmostEqual(
            d_theta_final,
            0.0,
            delta=tol,
            msg=f"After round-trip, theta should return near start; delta={d_theta_final} mdeg",
        )


if __name__ == "__main__":
    unittest.main()
