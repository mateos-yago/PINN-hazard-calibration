"""Run a single experiment from a YAML config file.

Usage:
    python -m experiments.run_experiment --config experiments/configs/stage1_exponential_p1.yaml
"""

import argparse
import copy
import os
import sys

import numpy as np
import yaml
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import CoxSimulator, ExponentialBaseline, WeibullBaseline, GompertzBaseline, PiecewiseConstantBaseline
from src.data import SurvivalDataPipeline
from src.models import HazardPINN
from src.training import CompositeLoss, Trainer, ExperimentConfig
from src.evaluation import EvaluationReport


_BASELINES = {
    "ExponentialBaseline": ExponentialBaseline,
    "WeibullBaseline": WeibullBaseline,
    "GompertzBaseline": GompertzBaseline,
    "PiecewiseConstantBaseline": PiecewiseConstantBaseline,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_simulator(sim_cfg: dict) -> CoxSimulator:
    baseline_cls = _BASELINES[sim_cfg["baseline"]]
    baseline = baseline_cls(**sim_cfg.get("baseline_params", {}))
    return CoxSimulator(
        baseline_hazard=baseline,
        beta=sim_cfg["beta"],
        n_covariates=sim_cfg["n_covariates"],
        censoring_rate=sim_cfg.get("censoring_rate", 0.3),
        censoring_type=sim_cfg.get("censoring_type", "random_exponential"),
        covariate_dist=sim_cfg.get("covariate_dist", "normal"),
        random_seed=sim_cfg.get("random_seed", None),
    )


def initialize_beta_from_cox_pl(model: HazardPINN, dataset) -> None:
    """Initialize standardized beta by optimizing Cox partial likelihood."""
    x = dataset.covariates.numpy()
    event = dataset.event.numpy()
    risk_mask = dataset.risk_mask.numpy()

    def objective(beta):
        xb = x @ beta
        denom = (risk_mask * np.exp(xb)[None, :]).sum(axis=1) + 1e-12
        pll = event * (xb - np.log(denom))
        return float(-pll.sum() / max(event.sum(), 1.0))

    result = minimize(objective, np.zeros(model.n_covariates), method="BFGS")
    if result.success:
        model.beta.data = model.beta.data.new_tensor(result.x)
    else:
        print(f"Warning: Cox PL beta initialization failed: {result.message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results_dir", default="experiments/results")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Simulate data
    simulator = build_simulator(cfg["simulation"])
    df = simulator.simulate(cfg["simulation"]["n_samples"])

    # Preprocess
    pipeline = SurvivalDataPipeline()
    dataset = pipeline.fit_transform(df)

    # Build model
    model_cfg = copy.deepcopy(cfg["model"])
    coeff_cfg = model_cfg.get("coefficient", {})
    if "log_t_shifted" in coeff_cfg.get("time_features", []):
        time_scaler, _ = pipeline.get_scalers()
        data_range = float(time_scaler.data_range_[0])
        data_min = float(time_scaler.data_min_[0])
        coeff_cfg.setdefault("log_time_offset", data_min / (data_range + 1e-8))
    model = HazardPINN.from_config(model_cfg)
    if cfg["training"].get("beta_initialization") == "cox_pl":
        initialize_beta_from_cox_pl(model, dataset)

    # Build loss
    loss_weights = cfg["training"]["loss_weights"]
    loss_fn = CompositeLoss(
        weights=loss_weights,
        baseline_hazard=simulator.baseline_hazard if loss_weights.get("baseline_ref", 0.0) > 0 else None,
        pipeline=pipeline if loss_weights.get("baseline_ref", 0.0) > 0 else None,
        true_beta=simulator.beta if loss_weights.get("baseline_ref", 0.0) > 0 else None,
    )

    # Build trainer config
    t_cfg = cfg["training"]
    exp_config = ExperimentConfig(
        experiment_name=cfg["experiment_name"],
        rationale=cfg.get("rationale", ""),
        n_epochs=t_cfg["n_epochs"],
        lr=t_cfg["lr"],
        lr_beta=t_cfg.get("lr_beta"),
        lr_coefficient=t_cfg.get("lr_coefficient"),
        optimizer_name=t_cfg.get("optimizer_name", "adam"),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        n_collocation_points=t_cfg["n_collocation_points"],
        collocation_method=t_cfg.get("collocation_method", "uniform"),
        checkpoint_every=t_cfg.get("checkpoint_every", 200),
        gradient_clip=t_cfg.get("gradient_clip"),
        lr_scheduler=t_cfg.get("lr_scheduler"),
        lr_patience=t_cfg.get("lr_patience", 100),
        loss_weights=t_cfg["loss_weights"],
        model_config=model_cfg,
        simulation_config=cfg["simulation"],
    )

    trainer = Trainer(model, loss_fn, exp_config)
    loss_history = trainer.train(dataset)

    # Load best weights before evaluation
    trainer._load_best_weights(model)

    # Evaluate
    report = EvaluationReport()
    exp_dir = os.path.join(args.results_dir, cfg["experiment_name"])
    eval_cfg = cfg.get("evaluation", {})
    metrics = report.generate(
        model,
        simulator,
        dataset,
        pipeline,
        exp_dir,
        loss_history,
        thresholds=eval_cfg.get("thresholds"),
        hazard_time_extension=eval_cfg.get("hazard_time_extension", 0.10),
    )

    # Save experiment
    trainer.save_experiment(args.results_dir, metrics)


if __name__ == "__main__":
    main()
