"""Loss function components for the Cox PINN."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.pipeline import SurvivalDataset
from ..models.pinn import HazardPINN


class MLELoss(nn.Module):
    """Negative log-likelihood (grounds both networks).

    L_MLE = -1/n * Σ_i [ Δ_i · (γ̂(Y_i) + x_i^T β) - Λ̂(Y_i, x_i) ]
    """

    def __init__(
        self,
        weighting: str = "uniform",
        weight_power: float = 0.5,
        max_weight: float = 10.0,
        weight_target: str = "full",
    ):
        super().__init__()
        self.weighting = weighting
        self.weight_power = float(weight_power)
        self.max_weight = float(max_weight)
        self.weight_target = weight_target

    def _sample_weights(
        self,
        risk_mask: Optional[torch.Tensor],
        n: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.weighting == "uniform":
            return None
        if self.weighting != "inverse_at_risk":
            raise ValueError(f"Unknown MLE weighting: {self.weighting}")
        if risk_mask is None:
            raise RuntimeError("inverse_at_risk MLE weighting requires risk_mask.")

        at_risk = risk_mask.sum(dim=1).clamp_min(1.0).to(device=device, dtype=dtype)
        weights = (float(n) / at_risk).pow(self.weight_power)
        weights = weights / weights.mean().clamp_min(1e-12)
        if self.max_weight > 0:
            weights = weights.clamp(max=self.max_weight)
            weights = weights / weights.mean().clamp_min(1e-12)
        return weights.detach()

    def forward(
        self,
        Lambda_hat: torch.Tensor,
        gamma_hat: torch.Tensor,
        beta: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        covariates: torch.Tensor,
        risk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xb = covariates @ beta  # [n]
        log_hazard = gamma_hat.squeeze() + xb  # [n]
        event_nll = -(event * log_hazard)
        cumulative_nll = Lambda_hat.squeeze()
        weights = self._sample_weights(
            risk_mask,
            n=event_nll.shape[0],
            device=event_nll.device,
            dtype=event_nll.dtype,
        )
        if weights is not None:
            if self.weight_target == "full":
                event_nll = event_nll * weights
                cumulative_nll = cumulative_nll * weights
            elif self.weight_target == "event":
                event_nll = event_nll * weights
            else:
                raise ValueError(f"Unknown MLE weight target: {self.weight_target}")
        nll = event_nll + cumulative_nll
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


class BaselineMonotonicityLoss(nn.Module):
    """Penalizes decreasing log-baseline hazard on collocation points."""

    def forward(self, model: HazardPINN, t_col: torch.Tensor) -> torch.Tensor:
        t_req = t_col.detach().requires_grad_(True)
        gamma = model.coefficient(t_req)
        dgamma_dt = torch.autograd.grad(
            gamma,
            t_req,
            grad_outputs=torch.ones_like(gamma),
            create_graph=True,
            retain_graph=True,
        )[0]
        return (torch.relu(-dgamma_dt) ** 2).mean()


class BaselineMinSlopeLoss(nn.Module):
    """Penalizes log-baseline slopes below a small positive margin."""

    def __init__(self, margin: float = 0.25):
        super().__init__()
        self.margin = margin

    def forward(self, model: HazardPINN, t_col: torch.Tensor) -> torch.Tensor:
        t_req = t_col.detach().requires_grad_(True)
        gamma = model.coefficient(t_req)
        dgamma_dt = torch.autograd.grad(
            gamma,
            t_req,
            grad_outputs=torch.ones_like(gamma),
            create_graph=True,
            retain_graph=True,
        )[0]
        return (torch.relu(self.margin - dgamma_dt) ** 2).mean()


class BaselineSmoothnessLoss(nn.Module):
    """Discrete-Laplacian penalty on γ over an interior grid.

    Shape-agnostic regularizer that removes the MLE-spike pathology of a free
    continuous γ. Implemented as a finite-difference second derivative on an
    interior grid `[t_min, 1]` (avoiding the t=0 log-time singularity that
    blows up an autograd implementation when log_t is in the input features).
    """

    def __init__(self, n_grid: int = 200, t_min: float = 0.01):
        super().__init__()
        self.n_grid = int(n_grid)
        self.t_min = float(t_min)

    def forward(self, model: HazardPINN, t_col: torch.Tensor) -> torch.Tensor:
        device, dtype = t_col.device, t_col.dtype
        t_grid = torch.linspace(
            self.t_min, 1.0, self.n_grid, device=device, dtype=dtype
        ).unsqueeze(1)
        gamma = model.coefficient(t_grid).squeeze(-1)  # [n_grid]
        d2 = gamma[2:] - 2.0 * gamma[1:-1] + gamma[:-2]  # [n_grid - 2]
        return (d2 ** 2).mean()


class BaselineReferenceLoss(nn.Module):
    """Supervised gamma loss for simulated experiments with known baseline."""

    def __init__(self, baseline_hazard=None, pipeline=None, true_beta=None):
        super().__init__()
        self.baseline_hazard = baseline_hazard
        self.pipeline = pipeline
        self.true_beta = None if true_beta is None else np.asarray(true_beta, dtype=float)

    def forward(self, model: HazardPINN, t_col: torch.Tensor) -> torch.Tensor:
        if self.baseline_hazard is None or self.pipeline is None or self.true_beta is None:
            raise RuntimeError("BaselineReferenceLoss requires baseline_hazard, pipeline, and true_beta.")

        time_scaler, cov_scaler = self.pipeline.get_scalers()
        t_np = t_col.detach().cpu().numpy().reshape(-1)
        t_orig = self.pipeline.inverse_transform_time(t_np)
        time_range = float(time_scaler.data_range_[0])
        baseline_scale = float(np.exp(np.dot(cov_scaler.mean_, self.true_beta)))
        target = np.log(self.baseline_hazard.hazard(t_orig) * time_range * baseline_scale + 1e-8)
        target_tensor = torch.tensor(target, dtype=t_col.dtype, device=t_col.device)
        gamma = model.coefficient(t_col).squeeze()
        return ((gamma - target_tensor) ** 2).mean()


class CompositeLoss(nn.Module):
    """Weighted sum of the four loss components.

    Set any weight to 0.0 to disable that component entirely.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        baseline_hazard=None,
        pipeline=None,
        true_beta=None,
    ):
        super().__init__()
        defaults = {
            "mle": 1.0,
            "pl": 0.5,
            "ode": 1.0,
            "ic": 1.0,
            "monotonic": 0.0,
            "min_slope": 0.0,
            "min_slope_margin": 0.25,
            "smoothness": 0.0,
            "baseline_ref": 0.0,
            "mle_weighting": "uniform",
            "mle_weight_power": 0.5,
            "mle_max_weight": 10.0,
            "mle_weight_target": "full",
        }
        w = defaults if weights is None else {**defaults, **weights}
        self.w_mle = w["mle"]
        self.w_pl = w["pl"]
        self.w_ode = w["ode"]
        self.w_ic = w["ic"]
        self.w_monotonic = w["monotonic"]
        self.w_min_slope = w["min_slope"]
        self.min_slope_margin = w["min_slope_margin"]
        self.w_smoothness = w["smoothness"]
        self.w_baseline_ref = w["baseline_ref"]

        self.mle_loss = MLELoss(
            weighting=w["mle_weighting"],
            weight_power=w["mle_weight_power"],
            max_weight=w["mle_max_weight"],
            weight_target=w["mle_weight_target"],
        )
        self.pl_loss = PartialLikelihoodLoss()
        self.ode_loss = ODEResidualLoss()
        self.ic_loss = InitialConditionLoss()
        self.monotonic_loss = BaselineMonotonicityLoss()
        self.min_slope_loss = BaselineMinSlopeLoss(margin=self.min_slope_margin)
        self.smoothness_loss = BaselineSmoothnessLoss()
        self.baseline_ref_loss = BaselineReferenceLoss(baseline_hazard, pipeline, true_beta)

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
            l = self.mle_loss(Lambda_hat, gamma_hat, beta, t, event, x, risk_mask)
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

        if self.w_monotonic > 0:
            l = self.monotonic_loss(model, t_col)
            components["monotonic"] = l
            total = total + self.w_monotonic * l

        if self.w_min_slope > 0:
            l = self.min_slope_loss(model, t_col)
            components["min_slope"] = l
            total = total + self.w_min_slope * l

        if self.w_smoothness > 0:
            l = self.smoothness_loss(model, t_col)
            components["smoothness"] = l
            total = total + self.w_smoothness * l

        if self.w_baseline_ref > 0:
            l = self.baseline_ref_loss(model, t_col)
            components["baseline_ref"] = l
            total = total + self.w_baseline_ref * l

        components_float = {k: v.item() for k, v in components.items()}
        return total, components_float
