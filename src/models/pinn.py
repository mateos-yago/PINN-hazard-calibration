"""HazardPINN: combines SurrogateNetwork, CoefficientNetwork, and β parameter."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .networks import SurrogateNetwork, CoefficientNetwork


class HazardPINN(nn.Module):
    """Full PINN model for Cox hazard estimation.

    Components:
        surrogate  (Λ_φ): (t, x) → cumulative hazard
        coefficient (γ_θ): t → log baseline hazard
        beta: learnable covariate coefficient vector
    """

    def __init__(
        self,
        surrogate: SurrogateNetwork,
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
        if parameterization not in ("surrogate", "quadrature"):
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
