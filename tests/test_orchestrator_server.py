"""
Tests for orchestrator_server.py and surrogate_fiber_coupling.py.

Forces OPTICS_SIMULATE=1 and a temporary ORCH_SURROGATE_DIR with minimal valid weights
so the Thorlabs .NET path is never touched and the surrogate loads without notebook assets.

Run from repo root:
  .venv\\Scripts\\python.exe -m unittest tests.test_orchestrator_server -v
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
_MCP_SERVERS = str(_ROOT / "mcp_servers")
if _MCP_SERVERS not in sys.path:
    sys.path.insert(0, _MCP_SERVERS)


def _write_minimal_surrogate(dir_path: Path) -> None:
    import surrogate_fiber_coupling as sfc

    m = sfc.FiberCouplingNet()
    torch.save(m.state_dict(), dir_path / "fiber_coupling_model.pth")
    np.savez(
        dir_path / "normalisation_parameters.npz",
        inputs_min=np.full(9, -10.0, dtype=np.float64),
        inputs_max=np.full(9, 10.0, dtype=np.float64),
        outputs_min=np.array(0.01, dtype=np.float64),
        outputs_max=np.array(1.0, dtype=np.float64),
    )


def _reload_orchestrator_stack(*, surrogate_dir: str | None) -> object:
    os.environ["OPTICS_SIMULATE"] = "1"
    if surrogate_dir is not None:
        os.environ["ORCH_SURROGATE_DIR"] = surrogate_dir
    else:
        os.environ.pop("ORCH_SURROGATE_DIR", None)
    for name in ("orchestrator_server", "optics_server", "surrogate_fiber_coupling"):
        sys.modules.pop(name, None)
    import optics_server  # noqa: F401

    return importlib.import_module("orchestrator_server")


class TestSurrogateFiberCoupling(unittest.TestCase):
    def test_normalize_log_round_trip_scalar_output(self) -> None:
        import surrogate_fiber_coupling as sfc

        y = np.array([[0.1, 0.5]], dtype=np.float64)
        mn = np.array([0.01, 0.01])
        mx = np.array([1.0, 1.0])
        n = sfc.normalize(y, mn, mx, scales=["log", "log"])
        u = sfc.unnormalize(n, mn, mx, scales=["log", "log"])
        np.testing.assert_allclose(u, y, rtol=1e-5, atol=1e-6)

    def test_optimize_runs_and_clamps(self) -> None:
        import surrogate_fiber_coupling as sfc

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_minimal_surrogate(d)
            b = sfc.SurrogateBundle.load(d, device="cpu")
            init = np.zeros(9, dtype=np.float64)
            final, eff = sfc.optimize_mirror_angles_maximize_efficiency(
                b, init, num_epochs=5, learning_rate=0.05, freeze_lens=True
            )
            self.assertEqual(final.shape, (9,))
            self.assertTrue(np.all(final >= b.norm.inputs_min - 1e-5))
            self.assertTrue(np.all(final <= b.norm.inputs_max + 1e-5))
            self.assertGreaterEqual(eff, 0.0)


class TestOrchestratorServer(unittest.TestCase):
    _surrogate_tmp: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._td = tempfile.TemporaryDirectory()
        cls._surrogate_tmp = cls._td.name
        _write_minimal_surrogate(Path(cls._surrogate_tmp))
        cls.orch = _reload_orchestrator_stack(surrogate_dir=cls._surrogate_tmp)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._td.cleanup()
        os.environ.pop("ORCH_SURROGATE_DIR", None)

    def tearDown(self) -> None:
        os.environ["OPTICS_SIMULATE"] = "1"

    def test_orchestrator_status(self) -> None:
        o = self.orch
        s = o.orchestrator_status()
        self.assertTrue(s["surrogate_loaded"])
        self.assertIsNone(s["surrogate_load_error"])
        self.assertTrue(s["optics_simulate"])
        self.assertEqual(s["surrogate_dir"], str(Path(self._surrogate_tmp).resolve()))

    def test_surrogate_predict_and_bounds(self) -> None:
        o = self.orch
        b = o.surrogate_input_bounds()
        self.assertEqual(len(b["inputs_min"]), 9)
        eta = o.surrogate_predict_coupling([[0.0] * 9])
        self.assertEqual(len(eta["efficiency_eta"]), 1)

    def test_optics_proxies(self) -> None:
        o = self.orch
        alls = o.optics_get_all_mirrors_state()
        self.assertEqual(len(alls), o.optics_server.NUM_MIRRORS)
        lim = o.optics_get_mirror_limits(0)
        self.assertIn("theta_min_mdeg", lim)

    def test_closed_loop_moves_stub(self) -> None:
        o = self.orch
        with mock.patch("optics_server.time.sleep"):
            r = o.alignment_closed_loop_step(
                nn_theta_to_mdeg=1.0,
                nn_phi_to_mdeg=1.0,
                num_epochs=3,
                learning_rate=0.05,
                settle_ms=0,
            )
        self.assertIn("optimized_state_9", r)
        self.assertEqual(len(r["optics_move_results"]), o.optics_server.NUM_MIRRORS)

    def test_missing_surrogate_reports_in_status(self) -> None:
        empty = tempfile.mkdtemp()
        try:
            bad = _reload_orchestrator_stack(surrogate_dir=empty)
            s = bad.orchestrator_status()
            self.assertFalse(s["surrogate_loaded"])
            self.assertIsNotNone(s["surrogate_load_error"])
            with self.assertRaises(RuntimeError):
                bad.surrogate_predict_coupling([[0.0] * 9])
        finally:
            Path(empty).rmdir()


if __name__ == "__main__":
    unittest.main()
