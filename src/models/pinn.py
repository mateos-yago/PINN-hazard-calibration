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

    def __init__(self, surrogate: SurrogateNetwork, coefficient: CoefficientNetwork, n_covariates: int):
        super().__init__()
        self.surrogate = surrogate
        self.coefficient = coefficient
        self.beta = nn.Parameter(torch.zeros(n_covariates))
        self.n_covariates = n_covariates

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
        Lambda_hat = self.surrogate(t, x)
        gamma_hat = self.coefficient(t)
        return {"Lambda_hat": Lambda_hat, "gamma_hat": gamma_hat, "beta": self.beta}

    def compute_Lambda_derivative(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute ∂Λ̂/∂t via automatic differentiation.

        Args:
            t: shape [batch, 1], must have requires_grad=True
            x: shape [batch, p]
        Returns:
            dLambda_dt: shape [batch, 1]
        """
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
        )
        return cls(surrogate, coefficient, n_covariates=p)
