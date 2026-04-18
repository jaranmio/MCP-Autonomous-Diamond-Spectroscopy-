"""
Fiber-coupling surrogate neural network — logic aligned with
``surrogate-NN/Update of Automated optical alignment using digital twins.ipynb``.

**Physics / IO (from the notebook)**

- **Input:** a 9-D vector in *physical* simulator space (before linear normalization to [-1, 1]):

  ``[φ₁, φ₂, ψ₁, ψ₂, δx, δy, δz, φ_x, φ_y]``

  i.e. four mirror-related angles (two per mirror in the Zemax model), then five collimation /
  misalignment parameters (shifts and tilts). The notebook meshes *Thetas* and *Phis* into
  columns ``0+_j`` and ``2+_j`` for mirror index ``_j ∈ {0,1}``, so:

  - Mirror 0: components **0** (first angle) and **2** (second angle).
  - Mirror 1: components **1** and **3**.

- **Output:** scalar **fiber coupling efficiency** η. The network head is trained on **log₁₀ η**;
  after inference, outputs are **unnormalized** from normalized logits using the stored
  ``outputs_min`` / ``outputs_max`` bounds with **log** scaling (same as
  ``unnormalize(..., scales=['log'])`` in the notebook).

- **Architecture:** ``FiberCouplingNet`` — Linear 9→64, ReLU; 64→1024, ReLU; 1024→1.
  Weights expect **linear-normalized** inputs in [-1, 1] per feature (``normalize`` /
  ``normalize_with_grad`` with ``scale='linear'``).

This module does **not** perform MCP I/O; it is imported by ``orchestrator_server.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiberCouplingNet(nn.Module):
    """Same MLP as the notebook (expects normalized 9-D inputs)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def normalize(
    y: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    scales: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Notebook ``normalize`` — map columns to [-1, 1] (linear and/or log per column)."""
    normalised_data = np.zeros_like(y, dtype=np.float64)
    if scales is not None:
        scales_list = list(scales)
        for _i, scale in enumerate(scales_list):
            if scale == "log":
                log_scaled = np.log10(y[:, _i])
                denom = np.log10(y_max[_i]) - np.log10(y_min[_i])
                linear_scaled = 2 * ((log_scaled - np.log10(y_min[_i])) / denom) - 1
            elif scale == "linear":
                linear_scaled = 2 * ((y[:, _i] - y_min[_i]) / (y_max[_i] - y_min[_i])) - 1
            else:
                raise ValueError(f"Unknown scale {scale!r}")
            normalised_data[:, _i] = linear_scaled
    else:
        normalised_data = 2 * ((y - y_min) / (y_max - y_min)) - 1
    return normalised_data.astype(np.float32)


def unnormalize(
    y_norm: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
    scales: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Notebook ``unnormalize`` — invert ``normalize``."""
    y = np.zeros_like(y_norm, dtype=np.float64)
    if scales is not None:
        scales_list = list(scales)
        for _i, scale in enumerate(scales_list):
            if scale == "log":
                log_scaled = ((y_norm[:, _i] + 1) / 2) * (
                    np.log10(y_max[_i]) - np.log10(y_min[_i])
                ) + np.log10(y_min[_i])
                linear_scaled = 10**log_scaled
            elif scale == "linear":
                linear_scaled = ((y_norm[:, _i] + 1) / 2) * (y_max[_i] - y_min[_i]) + y_min[_i]
            else:
                raise ValueError(f"Unknown scale {scale!r}")
            y[:, _i] = linear_scaled
    else:
        y = ((y_norm + 1) / 2) * (y_max - y_min) + y_min
    return y.astype(np.float32)


def normalize_with_grad(
    y: torch.Tensor,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    scale: Optional[str] = None,
) -> torch.Tensor:
    """Notebook ``normalize_with_grad`` (linear or log on all elements)."""
    if scale is None or scale == "linear":
        return (2 * ((y - y_min) / (y_max - y_min)) - 1).float()
    if scale == "log":
        y_log = torch.log10(y)
        min_log = torch.log10(y_min)
        max_log = torch.log10(y_max)
        return (2 * ((y_log - min_log) / (max_log - min_log)) - 1).float()
    raise ValueError(f"Unknown scale {scale!r}")


def unnormalize_with_grad(
    y_norm: torch.Tensor,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    scale: Optional[str] = None,
) -> torch.Tensor:
    """Notebook ``unnormalize_with_grad``."""
    if scale is None or scale == "linear":
        return (((y_norm + 1) / 2) * (y_max - y_min) + y_min).float()
    if scale == "log":
        min_log = torch.log10(y_min)
        max_log = torch.log10(y_max)
        log_scaled = ((y_norm + 1) / 2) * (max_log - min_log) + min_log
        return (10**log_scaled).float()
    raise ValueError(f"Unknown scale {scale!r}")


def _to_float_tensor(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


@dataclass
class SurrogateNormParams:
    inputs_min: np.ndarray  # (9,)
    inputs_max: np.ndarray  # (9,)
    outputs_min: float
    outputs_max: float


@dataclass
class SurrogateBundle:
    """Loaded ``FiberCouplingNet`` plus normalization bounds from ``normalisation_parameters.npz``."""

    model: FiberCouplingNet
    norm: SurrogateNormParams
    device: torch.device
    weights_path: Path
    norm_path: Path

    @property
    def inputs_min_t(self) -> torch.Tensor:
        return torch.tensor(self.norm.inputs_min, device=self.device, dtype=torch.float32)

    @property
    def inputs_max_t(self) -> torch.Tensor:
        return torch.tensor(self.norm.inputs_max, device=self.device, dtype=torch.float32)

    @property
    def outputs_min_t(self) -> torch.Tensor:
        return torch.tensor(
            [self.norm.outputs_min], device=self.device, dtype=torch.float32
        )

    @property
    def outputs_max_t(self) -> torch.Tensor:
        return torch.tensor(
            [self.norm.outputs_max], device=self.device, dtype=torch.float32
        )

    def predict_efficiency_batch(
        self,
        state_9: np.ndarray,
        input_power_scale: float = 1.0,
    ) -> np.ndarray:
        """
        state_9: shape (N, 9) physical inputs; returns η * input_power_scale of shape (N,)
        after log unnormalization.

        input_power_scale mirrors the notebook's ``input_power`` regression parameter:
        a dimensionless fitting coefficient (notebook optimizes it in [0, 10 000]) that
        maps the surrogate's normalized output onto the magnitude of the photodiode /
        sensor readings.  It is **not** a physical power value — it is fitted by minimizing
        SSR between ``prediction * input_power`` and the measured voltage array.
        Default 1.0 returns raw dimensionless η.
        """
        if state_9.ndim == 1:
            state_9 = state_9.reshape(1, -1)
        if state_9.shape[1] != 9:
            raise ValueError("state_9 must have 9 columns")
        x = normalize(state_9, self.norm.inputs_min, self.norm.inputs_max)
        xt = torch.from_numpy(x).to(self.device, torch.float32)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(xt)
        out = raw.cpu().numpy()
        eff = unnormalize(
            out,
            np.array([self.norm.outputs_min], dtype=np.float64),
            np.array([self.norm.outputs_max], dtype=np.float64),
            scales=["log"],
        )
        return (eff.reshape(-1) * input_power_scale).astype(np.float64)

    def predict_efficiency(self, state_9: list[float], input_power_scale: float = 1.0) -> float:
        arr = np.asarray(state_9, dtype=np.float64).reshape(1, 9)
        return float(self.predict_efficiency_batch(arr, input_power_scale=input_power_scale)[0])

    @staticmethod
    def load(
        directory: Path,
        *,
        device: Optional[str] = None,
    ) -> "SurrogateBundle":
        """
        Load ``fiber_coupling_model.pth`` and ``normalisation_parameters.npz`` from *directory*.
        Filenames match the notebook.
        """
        d = Path(directory)
        pth = d / "fiber_coupling_model.pth"
        npz_path = d / "normalisation_parameters.npz"
        if not pth.is_file():
            raise FileNotFoundError(f"Missing surrogate weights: {pth}")
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing normalization file: {npz_path}")

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = FiberCouplingNet().to(dev)
        try:
            state_obj = torch.load(pth, map_location=dev, weights_only=True)
        except Exception:
            state_obj = torch.load(pth, map_location=dev, weights_only=False)
        model.load_state_dict(state_obj)
        model.eval()

        nz = np.load(npz_path)
        imin = np.asarray(nz["inputs_min"], dtype=np.float64).reshape(9)
        imax = np.asarray(nz["inputs_max"], dtype=np.float64).reshape(9)
        omin = _to_float_tensor(nz["outputs_min"])
        omax = _to_float_tensor(nz["outputs_max"])
        norm = SurrogateNormParams(
            inputs_min=imin, inputs_max=imax, outputs_min=omin, outputs_max=omax
        )
        return SurrogateBundle(
            model=model, norm=norm, device=dev, weights_path=pth, norm_path=npz_path
        )


def optimize_mirror_angles_maximize_efficiency(
    bundle: SurrogateBundle,
    initial_state_9: np.ndarray,
    *,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    freeze_lens: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Gradient ascent on η(θ) mirroring notebook cell 27 ("Optimization of Mirror angles"):
    Adam on the 9-vector in *physical* NN input space, **maximize** efficiency, optionally
    zeroing gradients on indices 4..8 so only mirror angles (0–3) move; clamp to training box.
    """
    s = np.asarray(initial_state_9, dtype=np.float32).reshape(9)
    guess = torch.tensor(s, device=bundle.device, requires_grad=True)
    opt = torch.optim.Adam([guess], lr=learning_rate)
    ins_min = bundle.inputs_min_t
    ins_max = bundle.inputs_max_t
    out_min = bundle.outputs_min_t
    out_max = bundle.outputs_max_t
    bundle.model.train()  # only affects dropout/batchnorm — there are none; harmless

    last_eff = 0.0
    for _ in range(num_epochs):
        inp = normalize_with_grad(guess, ins_min, ins_max, scale="linear")
        out = bundle.model(inp.unsqueeze(0))
        eff = unnormalize_with_grad(out, out_min, out_max, scale="log").squeeze()
        cost = -eff
        cost.backward()
        if freeze_lens and guess.grad is not None:
            guess.grad[4:].zero_()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            guess.clamp_(ins_min, ins_max)
        last_eff = float(eff.detach().cpu().item())

    bundle.model.eval()
    final = guess.detach().cpu().numpy().reshape(9)
    return final, last_eff
