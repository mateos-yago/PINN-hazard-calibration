"""Accuracy metrics and plotting utilities for PINN evaluation."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.pipeline import SurvivalDataset, SurvivalDataPipeline
from ..models.pinn import HazardPINN
from ..simulation.cox_simulator import BaselineHazard, CoxSimulator


def beta_accuracy(estimated_beta: np.ndarray, true_beta: np.ndarray) -> Dict:
    """Compare estimated vs true covariate coefficients."""
    est = np.asarray(estimated_beta)
    true = np.asarray(true_beta)
    diff = est - true
    rel_err = np.abs(diff) / (np.abs(true) + 1e-8)
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "per_coeff_relative_error": rel_err.tolist(),
        "estimated": est.tolist(),
        "true": true.tolist(),
    }


def baseline_hazard_accuracy(
    model: HazardPINN,
    baseline: BaselineHazard,
    pipeline: SurvivalDataPipeline,
    n_grid: int = 200,
) -> Dict:
    """Compare estimated α̂(t) = exp(γ̂_θ(t)) to true α(t) on a normalized time grid."""
    time_scaler, _ = pipeline.get_scalers()
    t_max_norm = 1.0
    t_norm = np.linspace(1e-4, t_max_norm, n_grid)
    t_orig = pipeline.inverse_transform_time(t_norm)

    true_hazard = baseline.hazard(t_orig)

    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        gamma_hat = model.coefficient(t_tensor).squeeze().numpy()
    est_hazard = np.exp(gamma_hat)

    # Normalize estimated hazard to match scale of true hazard (up to a constant)
    # because the PINN learns γ in normalized time; rescale by time Jacobian
    dt_norm_dt_orig = 1.0 / (time_scaler.data_range_[0] + 1e-8)
    est_hazard_rescaled = est_hazard * dt_norm_dt_orig

    diff = est_hazard_rescaled - true_hazard
    denom = true_hazard + 1e-8
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # NumPy ≥2.0 renamed trapz
    integrated_mse = float(_trapz(diff ** 2, t_orig) / (t_orig[-1] - t_orig[0]))
    integrated_rel_mse = float(_trapz((diff / denom) ** 2, t_orig) / (t_orig[-1] - t_orig[0]))

    return {
        "integrated_mse": integrated_mse,
        "integrated_relative_mse": integrated_rel_mse,
        "pointwise_max_error": float(np.max(np.abs(diff))),
        "t_grid_orig": t_orig.tolist(),
        "true_hazard": true_hazard.tolist(),
        "estimated_hazard": est_hazard_rescaled.tolist(),
    }


def concordance_index(model: HazardPINN, dataset: SurvivalDataset) -> float:
    """Harrell's C-index using predicted cumulative hazard as risk score."""
    model.eval()
    with torch.no_grad():
        t = dataset.time.unsqueeze(1)
        x = dataset.covariates
        out = model(t, x)
        risk = out["Lambda_hat"].squeeze().numpy()  # higher = higher risk

    times = dataset.time.numpy()
    events = dataset.event.numpy()

    concordant = 0
    permissible = 0
    n = len(times)
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if times[j] <= times[i]:
                continue
            permissible += 1
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5

    return float(concordant / permissible) if permissible > 0 else float("nan")


def plot_loss_history(loss_history: Dict[str, List], save_path: str) -> None:
    """Plot total loss and each component over training epochs."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    keys = ["total", "mle", "pl", "ode", "ic"]
    titles = ["Total Loss", "L_MLE", "L_PL", "L_ODE (physics)", "L_IC (initial cond.)"]
    epochs = loss_history["epoch"]

    for ax, key, title in zip(axes, keys, titles):
        vals = loss_history.get(key, [])
        if any(v != 0.0 for v in vals):
            ax.plot(epochs, vals)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_yscale("log" if min(v for v in vals if v > 0) > 0 else "linear")
            ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_baseline_hazard(
    model: HazardPINN,
    baseline: BaselineHazard,
    pipeline: SurvivalDataPipeline,
    save_path: str,
    n_grid: int = 200,
) -> None:
    """Plot true vs estimated baseline hazard α(t)."""
    metrics = baseline_hazard_accuracy(model, baseline, pipeline, n_grid)
    t_orig = np.array(metrics["t_grid_orig"])
    true_h = np.array(metrics["true_hazard"])
    est_h = np.array(metrics["estimated_hazard"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_orig, true_h, label="True α(t)", linewidth=2)
    ax.plot(t_orig, est_h, "--", label="Estimated α̂(t)", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Baseline hazard")
    ax.set_title(f"Baseline hazard: IRMSE={metrics['integrated_relative_mse']:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_beta_comparison(estimated: np.ndarray, true: np.ndarray, save_path: str) -> None:
    """Bar chart comparing true vs estimated β coefficients."""
    p = len(true)
    x = np.arange(p)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, p * 1.5), 5))
    ax.bar(x - width / 2, true, width, label="True β")
    ax.bar(x + width / 2, estimated, width, label="Estimated β̂", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"β_{i+1}" for i in range(p)])
    ax.set_ylabel("Coefficient value")
    rmse = float(np.sqrt(np.mean((np.array(estimated) - np.array(true)) ** 2)))
    ax.set_title(f"β coefficients  (RMSE={rmse:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


class EvaluationReport:
    """Generates and saves a complete evaluation for one experiment."""

    def generate(
        self,
        model: HazardPINN,
        simulator: CoxSimulator,
        dataset: SurvivalDataset,
        pipeline: SurvivalDataPipeline,
        save_path: str,
        loss_history: Optional[Dict] = None,
    ) -> Dict:
        """
        Args:
            model: trained HazardPINN
            simulator: the CoxSimulator used to generate data (provides true β and baseline)
            dataset: the SurvivalDataset used for training
            pipeline: fitted SurvivalDataPipeline
            save_path: directory to write outputs
            loss_history: optional dict from Trainer.train()

        Returns:
            metrics dict (also written to metrics.json)
        """
        os.makedirs(save_path, exist_ok=True)

        # β accuracy
        estimated_beta = model.beta.detach().numpy()
        true_beta = simulator.beta
        b_metrics = beta_accuracy(estimated_beta, true_beta)

        # Baseline hazard accuracy
        h_metrics = baseline_hazard_accuracy(model, simulator.baseline_hazard, pipeline)

        # C-index
        c_idx = concordance_index(model, dataset)

        metrics = {
            "beta": b_metrics,
            "hazard": {k: v for k, v in h_metrics.items() if not isinstance(v, list)},
            "c_index": c_idx,
            "thresholds_met": {
                "beta_rmse": b_metrics["rmse"] < 0.10,
                "hazard_irmse": h_metrics["integrated_relative_mse"] < 0.05,
                "c_index": c_idx > 0.75,
            },
        }

        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Plots
        plot_beta_comparison(
            estimated_beta, true_beta, os.path.join(save_path, "beta_comparison.png")
        )
        plot_baseline_hazard(
            model, simulator.baseline_hazard, pipeline,
            os.path.join(save_path, "baseline_hazard.png")
        )
        if loss_history is not None:
            plot_loss_history(loss_history, os.path.join(save_path, "loss_history.png"))

        all_passed = all(metrics["thresholds_met"].values())
        print(
            f"\n{'='*50}\nEvaluation complete:\n"
            f"  β RMSE:    {b_metrics['rmse']:.4f}  {'✓' if metrics['thresholds_met']['beta_rmse'] else '✗'}\n"
            f"  Haz IRMSE: {h_metrics['integrated_relative_mse']:.4f}  {'✓' if metrics['thresholds_met']['hazard_irmse'] else '✗'}\n"
            f"  C-index:   {c_idx:.4f}  {'✓' if metrics['thresholds_met']['c_index'] else '✗'}\n"
            f"  All passed: {all_passed}\n{'='*50}"
        )
        return metrics
