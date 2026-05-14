"""HazardPINN: combines SurrogateNetwork, CoefficientNetwork, and β parameter."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .networks import SurrogateNetwork, CoefficientNetwork, BaselineCumHazardNetwork


class HazardPINN(nn.Module):
    """Full PINN model for Cox hazard estimation.

    Three parameterizations are supported:

    - 'surrogate': canonical PINN. Λ_φ(t, x) is a free MLP. ODE and IC are
      enforced softly via L_ODE and L_IC. Exhibits the v1 identifiability
      collapse on this data without an oracle (see phaseA_v1).
    - 'quadrature': ODE-by-construction. Λ(t, x) = exp(xᵀβ) · ∫₀ᵗ exp(γ(s)) ds
      evaluated by trapezoidal rule. ODE and IC are exact to discretization
      error; L_ODE, L_IC contribute nothing useful.
    - 'factored_surrogate': true PINN with the v1 identifiability fix. The
      surrogate is restricted to depend only on t — Λ_φ(t) — and the model
      enforces the Cox PH factorization Λ(t, x) = exp(xᵀβ) · Λ_φ(t)
      architecturally. The ODE residual (now genuinely active) ties Λ_φ to γ.
    """

    def __init__(
        self,
        surrogate,
        coefficient: CoefficientNetwork,
        n_covariates: int,
        parameterization: str = "surrogate",
        quadrature_grid: int = 200,
    ):
        super().__init__()
        self.surrogate = surrogate
        self.coefficient = coefficient
        self.beta = nn.Parameter(torch.zeros(n_covariates))
        self.n_covariates = n_covariates
        if parameterization not in ("surrogate", "quadrature", "factored_surrogate"):
            raise ValueError(f"Unknown parameterization: {parameterization}")
        self.parameterization = parameterization
        self.quadrature_grid = int(quadrature_grid)

    def _lambda_quadrature(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute Λ(t,x) = exp(xᵀβ) · ∫₀ᵗ exp(γ(s)) ds via trapezoidal cumulative integration.

        The integral is evaluated on a fixed grid of `quadrature_grid` points
        on the normalized time domain [0, 1]; t is then linearly interpolated.
        IC Λ(0,x)=0 and ODE ∂Λ/∂t = exp(γ+xᵀβ) hold exactly (up to discretization).
        """
        device, dtype = t.device, t.dtype
        n = self.quadrature_grid

        t_grid = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).unsqueeze(1)  # [G,1]
        gamma_grid = self.coefficient(t_grid).squeeze(-1)  # [G]
        exp_gamma = torch.exp(gamma_grid)  # [G]

        dt = 1.0 / (n - 1)
        segments = (exp_gamma[:-1] + exp_gamma[1:]) * 0.5 * dt  # [G-1]
        zero = torch.zeros(1, device=device, dtype=dtype)
        G_cum = torch.cat([zero, torch.cumsum(segments, dim=0)])  # [G]

        t_clamped = t.squeeze(-1).clamp(0.0, 1.0)  # [B]
        idx_float = t_clamped * (n - 1)
        idx_lo = idx_float.floor().long().clamp(0, n - 2)
        frac = idx_float - idx_lo.to(dtype)
        G_at_t = G_cum[idx_lo] * (1.0 - frac) + G_cum[idx_lo + 1] * frac  # [B]

        xb = x @ self.beta  # [B]
        return (G_at_t * torch.exp(xb)).unsqueeze(-1)  # [B, 1]

    def _lambda_factored(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute Λ(t, x) = exp(xᵀβ) · Λ_φ(t) with the 1-D baseline surrogate."""
        Lambda_baseline = self.surrogate(t)  # [B, 1]; depends only on t
        xb = (x @ self.beta).unsqueeze(-1)  # [B, 1]
        return Lambda_baseline * torch.exp(xb)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            t: shape [batch, 1]
            x: shape [batch, p]
        Returns:
            dict with keys:
                Lambda_hat: cumulative hazard [batch, 1]
                gamma_hat:  log-baseline hazard [batch, 1]
                beta:       covariate coefficients [p]
        """
        if self.parameterization == "quadrature":
            Lambda_hat = self._lambda_quadrature(t, x)
        elif self.parameterization == "factored_surrogate":
            Lambda_hat = self._lambda_factored(t, x)
        else:
            Lambda_hat = self.surrogate(t, x)
        gamma_hat = self.coefficient(t)
        return {"Lambda_hat": Lambda_hat, "gamma_hat": gamma_hat, "beta": self.beta}

    def compute_Lambda_derivative(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute ∂Λ̂/∂t.

        For 'surrogate' parameterization this is taken via autograd of the surrogate.
        For 'quadrature' it is exact by FTC: ∂Λ/∂t = exp(γ(t) + xᵀβ).

        Args:
            t: shape [batch, 1]; for 'surrogate' must have requires_grad=True
            x: shape [batch, p]
        Returns:
            dLambda_dt: shape [batch, 1]
        """
        if self.parameterization == "quadrature":
            gamma = self.coefficient(t)  # [B, 1]
            xb = (x @ self.beta).unsqueeze(-1)  # [B, 1]
            return torch.exp(gamma + xb)

        if self.parameterization == "factored_surrogate":
            # Λ(t, x) = exp(xᵀβ) · Λ_φ(t)  →  ∂Λ/∂t = exp(xᵀβ) · ∂Λ_φ/∂t
            t_req = t.requires_grad_(True)
            Lambda_baseline = self.surrogate(t_req)
            grad_baseline = torch.autograd.grad(
                Lambda_baseline,
                t_req,
                grad_outputs=torch.ones_like(Lambda_baseline),
                create_graph=True,
                retain_graph=True,
            )[0]
            xb = (x @ self.beta).unsqueeze(-1)
            return grad_baseline * torch.exp(xb)

        t_req = t.requires_grad_(True)
        Lambda = self.surrogate(t_req, x)
        grad = torch.autograd.grad(
            Lambda,
            t_req,
            grad_outputs=torch.ones_like(Lambda),
            create_graph=True,
            retain_graph=True,
        )[0]
        return grad

    @classmethod
    def from_config(cls, config: dict) -> "HazardPINN":
        """Instantiate from a configuration dictionary.

        Expected keys:
            n_covariates: int
            parameterization: 'surrogate' | 'quadrature' (optional, default 'surrogate')
            quadrature_grid: int (optional, default 200; only used for 'quadrature')
            surrogate.hidden_dims: list[int]
            surrogate.activation: str
            surrogate.use_layer_norm: bool (optional, default False)
            coefficient.hidden_dims: list[int]
            coefficient.activation: str
            coefficient.use_layer_norm: bool (optional, default False)
            coefficient.time_features: list[str] (optional, default ["t"])
            coefficient.log_time_offset: float (optional, default 1e-6)
        """
        p = config["n_covariates"]
        s_cfg = config.get("surrogate", {})
        c_cfg = config.get("coefficient", {})
        parameterization = config.get("parameterization", "surrogate")
        quadrature_grid = int(config.get("quadrature_grid", 200))

        if parameterization == "factored_surrogate":
            surrogate = BaselineCumHazardNetwork(
                hidden_dims=s_cfg.get("hidden_dims", [64, 64, 64]),
                activation=s_cfg.get("activation", "silu"),
                use_layer_norm=s_cfg.get("use_layer_norm", False),
            )
        else:
            surrogate = SurrogateNetwork(
                n_covariates=p,
                hidden_dims=s_cfg.get("hidden_dims", [64, 64, 64]),
                activation=s_cfg.get("activation", "tanh"),
                use_layer_norm=s_cfg.get("use_layer_norm", False),
            )
        coefficient = CoefficientNetwork(
            hidden_dims=c_cfg.get("hidden_dims", [32, 32]),
            activation=c_cfg.get("activation", "tanh"),
            use_layer_norm=c_cfg.get("use_layer_norm", False),
            time_features=c_cfg.get("time_features", ["t"]),
            log_time_offset=c_cfg.get("log_time_offset", 1e-6),
            input_clamp_min=c_cfg.get("input_clamp_min", 0.0),
            input_clamp_max=c_cfg.get("input_clamp_max", 1.0),
        )
        return cls(
            surrogate,
            coefficient,
            n_covariates=p,
            parameterization=parameterization,
            quadrature_grid=quadrature_grid,
        )
