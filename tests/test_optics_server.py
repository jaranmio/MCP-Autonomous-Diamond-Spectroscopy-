"""
Tests for optics_server.py.

Hardware / Thorlabs XA .NET assemblies are never loaded in these tests (simulation,
mocks, or pure helpers only). On a Windows machine without OPTICS_SIMULATE, importing
the module could otherwise attempt a real connection; all cases here force simulation
or patch the .NET load path.

Run from repo root:
  .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Hardware / Thorlabs XA integration (runs on Windows by default when DLL present; opt out with
  OPTICS_SKIP_HARDWARE_TESTS): see tests/test_optics_server_integration.py

Module ``optics_server.settings`` (``OpticsSettings``) is rebuilt on each
``importlib.reload(optics_server)`` used in these tests.
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

# Import the module as `optics_server` (directory mcp_servers/ is on sys.path, not a package).
_ROOT = Path(__file__).resolve().parent.parent
_MCP_SERVERS = str(_ROOT / "mcp_servers")
if _MCP_SERVERS not in sys.path:
    sys.path.insert(0, _MCP_SERVERS)


def _reload_optics_server(
    *,
    env: dict[str, str] | None = None,
    pop_env: tuple[str, ...] = (),
) -> object:
    """
    Reload optics_server after applying env changes.

    Does not restore os.environ; tests must call `_reload_simulate()` in tearDown (or otherwise
    leave OPTICS_SIMULATE=1) so later imports do not attempt hardware .NET init on Windows.
    """
    import optics_server as o  # noqa: F401 — must be on sys.path

    for k in pop_env:
        os.environ.pop(k, None)
    for k, v in (env or {}).items():
        os.environ[k] = v
    return importlib.reload(o)


def _reload_simulate() -> object:
    """Fresh module state; hardware path disabled via OPTICS_SIMULATE=1."""
    return _reload_optics_server(env={"OPTICS_SIMULATE": "1"})


def tearDownModule() -> None:
    """
    Each case leaves ``OPTICS_SIMULATE=1`` in ``os.environ`` so the reloaded module stays on
    the stub. Remove it when this file is done so ``unittest discover`` can run
    ``test_optics_server_integration`` in the same process without tripping the
    "simulation forces stub" skip.
    """
    os.environ.pop("OPTICS_SIMULATE", None)


class TestEnvParsingAndHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.o = _reload_simulate()

    def tearDown(self) -> None:
        _reload_simulate()

    def test_truthy_env(self) -> None:
        o = self.o
        with mock.patch.dict(os.environ, {"X_Q": "1"}, clear=False):
            self.assertTrue(o._truthy_env("X_Q"))
        with mock.patch.dict(os.environ, {"X_Q": "TRUE"}, clear=False):
            self.assertTrue(o._truthy_env("X_Q"))
        with mock.patch.dict(os.environ, {"X_Q": "yes"}, clear=False):
            self.assertTrue(o._truthy_env("X_Q"))
        with mock.patch.dict(os.environ, {"X_Q": "On"}, clear=False):
            self.assertTrue(o._truthy_env("X_Q"))
        with mock.patch.dict(os.environ, {"X_Q": "0"}, clear=False):
            self.assertFalse(o._truthy_env("X_Q"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("X_MISSING", None)
            self.assertFalse(o._truthy_env("X_MISSING"))

    def test_angle_mode(self) -> None:
        import optics_server as mod

        with mock.patch.dict(os.environ, {"OPTICS_XA_ANGLE_MODE": "linear_mm"}, clear=False):
            o = importlib.reload(mod)
            self.assertEqual(o.settings.angle_mode, "linear_mm")
        with mock.patch.dict(os.environ, {"OPTICS_XA_ANGLE_MODE": "degrees"}, clear=False):
            o = importlib.reload(mod)
            self.assertEqual(o.settings.angle_mode, "degrees")
        with mock.patch.dict(os.environ, {"OPTICS_XA_ANGLE_MODE": "  DEGREES "}, clear=False):
            o = importlib.reload(mod)
            self.assertEqual(o.settings.angle_mode, "degrees")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPTICS_XA_ANGLE_MODE", None)
            o2 = importlib.reload(mod)
            self.assertEqual(o2.settings.angle_mode, "linear_mm")

    def test_mdeg_physical_linear_mm_vs_degrees(self) -> None:
        o = self.o
        self.assertAlmostEqual(o._mdeg_to_physical(1000.0, "linear_mm"), 1.0)
        self.assertAlmostEqual(o._physical_to_mdeg(1.0, "linear_mm"), 1000.0)
        self.assertAlmostEqual(o._mdeg_to_physical(500.0, "degrees"), 0.5)
        self.assertAlmostEqual(o._physical_to_mdeg(0.5, "degrees"), 500.0)

    def test_settings_object_matches_helpers(self) -> None:
        o = self.o
        self.assertEqual(o._angle_mode(), o.settings.angle_mode)
        self.assertTrue(o.settings.simulate)
        self.assertIsInstance(o.settings, o.OpticsSettings)

    def test_mdeg_mm_round_trip_default_factor(self) -> None:
        o = self.o
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPTICS_MDEG_TO_MM", None)
            o2 = importlib.reload(o)
            self.assertAlmostEqual(o2._mdeg_to_mm(1000.0), 1.0)
            self.assertAlmostEqual(o2._mm_to_mdeg(1.0), 1000.0)

    def test_mdeg_mm_zero_factor(self) -> None:
        o = self.o
        with mock.patch.dict(os.environ, {"OPTICS_MDEG_TO_MM": "0"}, clear=False):
            o2 = importlib.reload(o)
            self.assertEqual(o2._mm_to_mdeg(5.0), 0.0)

    def test_settings_parsed_serials_default_when_key_missing(self) -> None:
        o = self.o
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPTICS_XA_SERIALS", None)
            o2 = importlib.reload(o)
            self.assertEqual(o2.settings.parsed_serials, [o2.DEFAULT_OPTICS_XA_SERIALS])

    def test_settings_parsed_serials_empty_when_key_explicit_empty(self) -> None:
        o = self.o
        with mock.patch.dict(os.environ, {"OPTICS_XA_SERIALS": ""}, clear=False):
            o2 = importlib.reload(o)
            self.assertEqual(o2.settings.parsed_serials, [])

    def test_settings_parsed_serials_splits_and_strips(self) -> None:
        o = self.o
        with mock.patch.dict(os.environ, {"OPTICS_XA_SERIALS": " a , b , "}, clear=False):
            o2 = importlib.reload(o)
            self.assertEqual(o2.settings.parsed_serials, ["a", "b"])


class TestUseSimulationMatrix(unittest.TestCase):
    """
    _use_simulation() gates whether _get_hw() may try to connect (no .NET assembly load here).

    Implementation reads ``optics_server.settings`` (snapshot at reload). These tests patch
    os.environ, ``importlib.reload(optics_server)``, then assert both ``settings`` fields and
    ``_use_simulation()`` so the matrix stays tied to :class:`OpticsSettings`.
    """

    def tearDown(self) -> None:
        _reload_simulate()

    def test_optics_simulate_forces_true(self) -> None:
        import optics_server as o

        # settings.simulate short-circuits: OS / serial list do not matter.
        with mock.patch.dict(os.environ, {"OPTICS_SIMULATE": "1"}, clear=False):
            o = importlib.reload(o)
            self.assertTrue(o.settings.simulate)
            with mock.patch.object(o.sys, "platform", "win32"):
                self.assertTrue(o._use_simulation())

    def test_non_windows_forces_true(self) -> None:
        import optics_server as o

        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("OPTICS_SIMULATE", None)
            o = importlib.reload(o)
            self.assertFalse(o.settings.simulate)
            with mock.patch.object(o.sys, "platform", "linux"):
                self.assertTrue(o._use_simulation())

    def test_windows_no_serials_forces_true(self) -> None:
        import optics_server as o

        with mock.patch.dict(os.environ, {"OPTICS_XA_SERIALS": ""}, clear=False):
            os.environ.pop("OPTICS_SIMULATE", None)
            o = importlib.reload(o)
            self.assertFalse(o.settings.simulate)
            self.assertEqual(o.settings.parsed_serials, [])
            with mock.patch.object(o.sys, "platform", "win32"):
                self.assertTrue(o._use_simulation())

    def test_windows_wrong_serial_count_forces_true(self) -> None:
        import optics_server as o

        # NUM_MIRRORS == 1: valid lengths are 1 or 2; three serials → stub path.
        with mock.patch.dict(os.environ, {"OPTICS_XA_SERIALS": "a,b,c"}, clear=False):
            os.environ.pop("OPTICS_SIMULATE", None)
            o = importlib.reload(o)
            self.assertFalse(o.settings.simulate)
            self.assertEqual(o.settings.parsed_serials, ["a", "b", "c"])
            with mock.patch.object(o.sys, "platform", "win32"):
                self.assertTrue(o._use_simulation())

    def test_windows_valid_one_serial_not_simulation_by_count(self) -> None:
        import optics_server as o

        with mock.patch.dict(os.environ, {"OPTICS_XA_SERIALS": "27272875"}, clear=False):
            os.environ.pop("OPTICS_SIMULATE", None)
            o = importlib.reload(o)
            self.assertFalse(o.settings.simulate)
            self.assertEqual(o.settings.parsed_serials, ["27272875"])
            with mock.patch.object(o.sys, "platform", "win32"):
                self.assertFalse(o._use_simulation())


class TestThorlabsXAOpticsLayout(unittest.TestCase):
    def setUp(self) -> None:
        self.o = _reload_simulate()

    def tearDown(self) -> None:
        _reload_simulate()

    def test_theta_only_layout(self) -> None:
        o = self.o
        dev = o._ThorlabsXAOptics(["27272875"])
        self.assertEqual(dev._layout, "theta_only")
        self.assertEqual(dev._handle_per_axis, [None, None])

    def test_full_layout_two_serials(self) -> None:
        o = self.o
        dev = o._ThorlabsXAOptics(["111", "222"])
        self.assertEqual(dev._layout, "full")
        self.assertEqual(len(dev._handle_per_axis), 2)

    def test_invalid_serial_count_raises(self) -> None:
        o = self.o
        with self.assertRaises(ValueError) as ctx:
            o._ThorlabsXAOptics(["a", "b", "c"])
        self.assertIn("Expected", str(ctx.exception))


class TestThorlabsXAOpticsStubAxis(unittest.TestCase):
    """Software-only phi axis (theta_only layout): limits and read/move guards."""

    def setUp(self) -> None:
        self.o = _reload_simulate()

    def tearDown(self) -> None:
        _reload_simulate()

    def test_axis_limits_software_phi(self) -> None:
        o = self.o
        dev = o._ThorlabsXAOptics(["one"])
        lo, hi = dev.axis_limits_mdeg(1)
        self.assertEqual(lo, o.PHI_MIN)
        self.assertEqual(hi, o.PHI_MAX)

    def test_read_axis_mdeg_raises_for_software_axis(self) -> None:
        o = self.o
        dev = o._ThorlabsXAOptics(["one"])
        with self.assertRaises(RuntimeError) as ctx:
            dev.read_axis_mdeg(1)
        self.assertIn("software-only", str(ctx.exception).lower())

    def test_move_raises_for_software_axis(self) -> None:
        o = self.o
        dev = o._ThorlabsXAOptics(["one"])
        with self.assertRaises(RuntimeError) as ctx:
            dev.move_axis_absolute_mdeg(1, 0.0)
        self.assertIn("software-only", str(ctx.exception).lower())
        with self.assertRaises(RuntimeError):
            dev.move_axis_relative_mdeg(1, 1.0)


class TestMirrorGeometryHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.o = _reload_simulate()

    def tearDown(self) -> None:
        _reload_simulate()

    def test_axis_indices(self) -> None:
        o = self.o
        self.assertEqual(o._axis_theta(0), 0)
        self.assertEqual(o._axis_phi(0), 1)


class TestValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.o = _reload_simulate()

    def tearDown(self) -> None:
        _reload_simulate()

    def test_validate_mirror_index(self) -> None:
        o = self.o
        o._validate_mirror_index(0)
        with self.assertRaises(ValueError):
            o._validate_mirror_index(-1)
        with self.assertRaises(ValueError):
            o._validate_mirror_index(o.NUM_MIRRORS)

    def test_validate_angles_theta_bounds(self) -> None:
        o = self.o
        with self.assertRaises(ValueError):
            o._validate_angles(0, o.THETA_MIN - 1.0, 0.0)
        with self.assertRaises(ValueError):
            o._validate_angles(0, o.THETA_MAX + 1.0, 0.0)

    def test_validate_angles_phi_bounds(self) -> None:
        o = self.o
        with self.assertRaises(ValueError):
            o._validate_angles(0, 0.0, o.PHI_MIN - 1.0)


class TestSimulationMcpTools(unittest.TestCase):
    """End-to-end tool behaviour with in-memory state (OPTICS_SIMULATE=1)."""

    def setUp(self) -> None:
        self.o = _reload_simulate()

    def tearDown(self) -> None:
        _reload_simulate()

    def test_get_mirror_state_initial(self) -> None:
        o = self.o
        st = o.get_mirror_state(0)
        self.assertEqual(st["mirror_index"], 0)
        self.assertEqual(st["theta_mdeg"], 0.0)
        self.assertEqual(st["phi_mdeg"], 0.0)

    def test_get_all_mirrors_state_length(self) -> None:
        o = self.o
        all_s = o.get_all_mirrors_state()
        self.assertEqual(len(all_s), o.NUM_MIRRORS)
        self.assertEqual(
            {x["mirror_index"] for x in all_s},
            set(range(o.NUM_MIRRORS)),
        )

    def test_set_and_read_mirror(self) -> None:
        o = self.o
        with mock.patch("optics_server.time.sleep") as sl:
            out = o.set_mirror_angle(0, 100.0, -200.0, settle_ms=0)
            sl.assert_called()
        self.assertEqual(out["theta_mdeg"], 100.0)
        self.assertEqual(out["phi_mdeg"], -200.0)
        st = o.get_mirror_state(0)
        self.assertEqual(st["theta_mdeg"], 100.0)
        self.assertEqual(st["phi_mdeg"], -200.0)

    def test_set_mirror_angle_validates_before_sleep(self) -> None:
        o = self.o
        with mock.patch("optics_server.time.sleep") as sl:
            with self.assertRaises(ValueError):
                o.set_mirror_angle(0, o.THETA_MAX + 1.0, 0.0, settle_ms=0)
            sl.assert_not_called()

    def test_step_mirror_angle(self) -> None:
        o = self.o
        o.set_mirror_angle(0, 100.0, 50.0, settle_ms=0)
        with mock.patch("optics_server.time.sleep"):
            out = o.step_mirror_angle(0, 10.0, -25.0, settle_ms=0)
        self.assertEqual(out["theta_mdeg"], 110.0)
        self.assertEqual(out["phi_mdeg"], 25.0)
        self.assertEqual(out["delta_theta_mdeg"], 10.0)
        self.assertEqual(out["delta_phi_mdeg"], -25.0)

    def test_step_mirror_rejects_out_of_range_result(self) -> None:
        o = self.o
        o.set_mirror_angle(0, o.THETA_MAX, 0.0, settle_ms=0)
        with self.assertRaises(ValueError):
            o.step_mirror_angle(0, 1.0, 0.0, settle_ms=0)

    def test_set_all_mirrors_angles_pydantic(self) -> None:
        o = self.o
        targets = [
            o.MirrorTarget(mirror_index=0, theta_mdeg=10.0, phi_mdeg=20.0),
        ]
        with mock.patch("optics_server.time.sleep"):
            res = o.set_all_mirrors_angles(targets, settle_ms=0)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["mirror_index"], 0)

    def test_step_all_mirrors_angles(self) -> None:
        o = self.o
        with mock.patch("optics_server.time.sleep"):
            o.set_mirror_angle(0, 10.0, 10.0, settle_ms=0)
            deltas = [
                o.MirrorDelta(mirror_index=0, delta_theta_mdeg=5.0, delta_phi_mdeg=-3.0),
            ]
            res = o.step_all_mirrors_angles(deltas, settle_ms=0)
        self.assertEqual(res[0]["theta_mdeg"], 15.0)
        self.assertEqual(res[0]["phi_mdeg"], 7.0)

    def test_get_mirror_limits_stub_returns_constants(self) -> None:
        o = self.o
        lim = o.get_mirror_limits(0)
        self.assertEqual(lim["theta_min_mdeg"], o.THETA_MIN)
        self.assertEqual(lim["theta_max_mdeg"], o.THETA_MAX)
        self.assertEqual(lim["phi_min_mdeg"], o.PHI_MIN)
        self.assertEqual(lim["phi_max_mdeg"], o.PHI_MAX)

    def test_home_mirror(self) -> None:
        o = self.o
        o.set_mirror_angle(0, 123.0, 456.0, settle_ms=0)
        with mock.patch("optics_server.time.sleep"):
            h = o.home_mirror(0, settle_ms=0)
        self.assertEqual(h["theta_mdeg"], 0.0)
        self.assertEqual(h["phi_mdeg"], 0.0)
        st = o.get_mirror_state(0)
        self.assertEqual(st["theta_mdeg"], 0.0)

    def test_home_all_mirrors(self) -> None:
        o = self.o
        with mock.patch("optics_server.time.sleep"):
            o.set_mirror_angle(0, 50.0, 50.0, settle_ms=0)
            res = o.home_all_mirrors(settle_ms=0)
        self.assertEqual(len(res), o.NUM_MIRRORS)
        self.assertTrue(all(r["theta_mdeg"] == 0.0 and r["phi_mdeg"] == 0.0 for r in res))


class TestHardwareInitFailure(unittest.TestCase):
    """
    If Windows + valid serial count but Thorlabs .NET stack cannot load, reads should raise RuntimeError.
    """

    def tearDown(self) -> None:
        _reload_simulate()

    def test_get_mirror_state_raises_when_dotnet_missing(self) -> None:
        saved = os.environ.get("OPTICS_SIMULATE")
        try:
            os.environ.pop("OPTICS_SIMULATE", None)
            with mock.patch.dict(os.environ, {"OPTICS_XA_SERIALS": "27272875"}, clear=False):
                o = importlib.import_module("optics_server")
                importlib.reload(o)
            with mock.patch.object(o.sys, "platform", "win32"):
                with mock.patch.object(
                    o, "_ensure_xa_clr_loaded", side_effect=FileNotFoundError("no dotnet")
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        o.get_mirror_state(0)
                    self.assertIn("Thorlabs", str(ctx.exception))
        finally:
            if saved is None:
                os.environ.pop("OPTICS_SIMULATE", None)
            else:
                os.environ["OPTICS_SIMULATE"] = saved
            _reload_simulate()


class TestFastMCPSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.o = _reload_simulate()

    def tearDown(self) -> None:
        _reload_simulate()

    def test_mcp_instance_exists(self) -> None:
        o = self.o
        self.assertTrue(hasattr(o, "mcp"))
        self.assertEqual(o.mcp.name, "optics_server")


if __name__ == "__main__":
    unittest.main()
