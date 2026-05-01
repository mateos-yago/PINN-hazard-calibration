"""Loss function components for the Cox PINN."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..data.pipeline import SurvivalDataset
from ..models.pinn import HazardPINN


class MLELoss(nn.Module):
    """Negative log-likelihood (grounds both networks).

    L_MLE = -1/n * Σ_i [ Δ_i · (γ̂(Y_i) + x_i^T β) - Λ̂(Y_i, x_i) ]
    """

    def forward(
        self,
        Lambda_hat: torch.Tensor,
        gamma_hat: torch.Tensor,
        beta: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        covariates: torch.Tensor,
    ) -> torch.Tensor:
        xb = covariates @ beta  # [n]
        log_hazard = gamma_hat.squeeze() + xb  # [n]
        nll = -(event * log_hazard - Lambda_hat.squeeze())
        return nll.mean()


class PartialLikelihoodLoss(nn.Module):
    """Negative Cox partial log-likelihood (stabilizes β estimation).

    L_PL = -1/n_e * Σ_{i:Δ_i=1} [ x_i^T β - log Σ_{j∈R(Y_i)} exp(x_j^T β) ]
    Uses precomputed risk_mask from SurvivalDataset.
    """

    def forward(
        self,
        beta: torch.Tensor,
        covariates: torch.Tensor,
        event: torch.Tensor,
        risk_mask: torch.Tensor,
    ) -> torch.Tensor:
        xb = covariates @ beta  # [n]
        n_events = event.sum()
        if n_events == 0:
            return torch.tensor(0.0, requires_grad=True)

        # log Σ_{j∈R(i)} exp(x_j^T β)  for each i
        # risk_mask: [n, n], risk_mask[i, j] = 1 if j in R(Y_i)
        log_denom = torch.logsumexp(
            xb.unsqueeze(0) + torch.log(risk_mask + 1e-10),
            dim=1,
        )  # [n]

        pl = event * (xb - log_denom)
        return -pl.sum() / n_events


class ODEResidualLoss(nn.Module):
    """ODE physics residual at collocation points.

    L_ODE = 1/N * Σ_j ( ∂Λ̂/∂t(t_j, x_j) - exp(γ̂(t_j) + x_j^T β) )²
    """

    def forward(
        self,
        model: HazardPINN,
        t_col: torch.Tensor,
        x_col: torch.Tensor,
    ) -> torch.Tensor:
        t_col = t_col.detach().requires_grad_(True)
        out = model(t_col, x_col)
        gamma_hat = out["gamma_hat"]
        beta = out["beta"]

        dLambda_dt = model.compute_Lambda_derivative(t_col, x_col)
        xb = (x_col @ beta).unsqueeze(1)
        rhs = torch.exp(gamma_hat + xb)

        residual = dLambda_dt - rhs
        return (residual ** 2).mean()


class InitialConditionLoss(nn.Module):
    """Enforces Λ̂(0, x_i) = 0.

    L_IC = 1/n * Σ_i ( Λ̂(0, x_i) )²
    """

    def forward(
        self,
        model: HazardPINN,
        covariates: torch.Tensor,
    ) -> torch.Tensor:
        n = covariates.shape[0]
        t_zero = torch.zeros(n, 1, device=covariates.device, dtype=covariates.dtype)
        Lambda_at_zero = model.surrogate(t_zero, covariates)
        return (Lambda_at_zero ** 2).mean()


class CompositeLoss(nn.Module):
    """Weighted sum of the four loss components.

    Set any weight to 0.0 to disable that component entirely.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        defaults = {"mle": 1.0, "pl": 0.5, "ode": 1.0, "ic": 1.0}
        w = defaults if weights is None else {**defaults, **weights}
        self.w_mle = w["mle"]
        self.w_pl = w["pl"]
        self.w_ode = w["ode"]
        self.w_ic = w["ic"]

        self.mle_loss = MLELoss()
        self.pl_loss = PartialLikelihoodLoss()
        self.ode_loss = ODEResidualLoss()
        self.ic_loss = InitialConditionLoss()

    def forward(
        self,
        model: HazardPINN,
        dataset: "SurvivalDataset",
        t_col: torch.Tensor,
        x_col: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor
            components: dict of scalar float values for logging
        """
        t = dataset.time.unsqueeze(1)
        x = dataset.covariates
        event = dataset.event
        risk_mask = dataset.risk_mask

        out = model(t, x)
        Lambda_hat = out["Lambda_hat"]
        gamma_hat = out["gamma_hat"]
        beta = out["beta"]

        components: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=t.device)

        if self.w_mle > 0:
            l = self.mle_loss(Lambda_hat, gamma_hat, beta, t, event, x)
            components["mle"] = l
            total = total + self.w_mle * l

        if self.w_pl > 0:
            l = self.pl_loss(beta, x, event, risk_mask)
            components["pl"] = l
            total = total + self.w_pl * l

        if self.w_ode > 0:
            l = self.ode_loss(model, t_col, x_col)
            components["ode"] = l
            total = total + self.w_ode * l

        if self.w_ic > 0:
            l = self.ic_loss(model, x)
            components["ic"] = l
            total = total + self.w_ic * l

        components_float = {k: v.item() for k, v in components.items()}
        return total, components_float
