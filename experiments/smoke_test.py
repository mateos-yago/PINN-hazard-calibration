"""End-to-end smoke test: verifies all modules wire together without errors.

Usage:
    python -m experiments.smoke_test
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.simulation import CoxSimulator, ExponentialBaseline
from src.data import SurvivalDataPipeline
from src.models import HazardPINN
from src.training import CompositeLoss, Trainer, ExperimentConfig
from src.evaluation import EvaluationReport


def main():
    print("=== Smoke Test ===")

    # 1. Simulate
    print("[1/5] Simulating data...")
    baseline = ExponentialBaseline(lam=0.5)
    simulator = CoxSimulator(baseline, beta=[1.0], n_covariates=1,
                             censoring_rate=0.3, random_seed=0)
    df = simulator.simulate(200)
    print(f"      Simulated {len(df)} subjects, event rate={df['event'].mean():.2f}")

    # 2. Preprocess
    print("[2/5] Preprocessing...")
    pipeline = SurvivalDataPipeline()
    dataset = pipeline.fit_transform(df)
    t_col, x_col = dataset.get_collocation_points(50, method="uniform")
    print(f"      Dataset: {dataset.n} obs, {dataset.p} covariates, {dataset.n_events} events")

    # 3. Model
    print("[3/5] Building model...")
    model_cfg = {
        "n_covariates": 1,
        "surrogate": {"hidden_dims": [32, 32], "activation": "tanh"},
        "coefficient": {"hidden_dims": [16, 16], "activation": "tanh"},
    }
    model = HazardPINN.from_config(model_cfg)

    # Forward pass
    t = dataset.time.unsqueeze(1)
    x = dataset.covariates
    out = model(t, x)
    assert out["Lambda_hat"].shape == (dataset.n, 1)
    assert out["gamma_hat"].shape == (dataset.n, 1)
    print(f"      Forward pass OK. β init: {model.beta.data.tolist()}")

    # Derivative
    dL = model.compute_Lambda_derivative(t_col, x_col)
    assert dL.shape == t_col.shape
    print(f"      Autograd derivative OK.")

    # 4. Loss
    print("[4/5] Testing loss...")
    loss_fn = CompositeLoss(weights={"mle": 1.0, "pl": 0.5, "ode": 1.0, "ic": 1.0})
    total, components = loss_fn(model, dataset, t_col, x_col)
    assert total.item() == total.item()  # not NaN
    print(f"      Loss OK: total={total.item():.4f}  components={components}")

    # 5. Training (5 epochs)
    print("[5/5] Training (5 epochs)...")
    config = ExperimentConfig(
        experiment_name="smoke_test",
        rationale="smoke test",
        n_epochs=5,
        lr=1e-3,
        n_collocation_points=50,
        checkpoint_every=5,
    )
    trainer = Trainer(model, loss_fn, config)
    history = trainer.train(dataset)
    assert len(history["epoch"]) == 5
    print("      Training OK.")

    # Evaluate
    metrics = EvaluationReport().generate(
        model, simulator, dataset, pipeline,
        "experiments/results/smoke_test", history
    )
    print(f"\nSmoke test PASSED. Metrics: {metrics['beta']['rmse']:.4f} β RMSE (untrained — expected to be large)")


if __name__ == "__main__":
    main()
