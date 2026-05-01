# Implementation Plan: PINN Hazard Calibration

## Overview

Implement a Physics-Informed Neural Network (PINN) for Cox model hazard function estimation,
following `Hazard_Function_Estimation_with_PINN.pdf`. The PINN jointly learns the cumulative
hazard Λ(t,x) and log-baseline hazard γ(t) using four physics-informed loss components.

---

## Project Structure

```
PINN_hazard_calibration/
├── src/
│   ├── __init__.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── cox_simulator.py        # CoxSimulator + BaselineHazard variants
│   ├── data/
│   │   ├── __init__.py
│   │   └── pipeline.py             # SurvivalDataset + SurvivalDataPipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── networks.py             # SurrogateNetwork + CoefficientNetwork
│   │   └── pinn.py                 # HazardPINN (both networks + β parameter)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py                 # 4 loss components + CompositeLoss
│   │   └── trainer.py              # Trainer + experiment logging
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py              # Accuracy metrics + plotting
├── experiments/
│   ├── configs/                    # YAML experiment configurations
│   ├── results/                    # Per-experiment output folders
│   └── architecture_log.md        # Running log of all architecture changes
├── .claude/docs/
│   ├── CLAUDE.md                   # Project coding guidelines
│   ├── IMPLEMENTATION_PLAN.md      # This file
│   └── AUTONOMOUS_TRAINING.md      # Instructions for autonomous training loop
└── requirements.txt
```

---

## Cox Model & PINN Formulation (from paper)

**Model:** λ_x(t) = α(t)·exp(x^T β)

**ODE form:** Λ'_x(t) = exp(γ(t) + x^T β), where γ(t) = log α(t)

**Initial condition:** Λ_x(0) = 0

**Networks:**
- Surrogate Λ_φ: (t, x) → Λ̂(t, x)  [cumulative hazard]
- Coefficient γ_θ: t → γ̂(t)         [log baseline hazard]
- β: learnable parameter vector

**Loss:** L_total = w1·L_MLE + w2·L_PL + w3·L_ODE + w4·L_IC

- L_MLE = -1/n Σ_i [Δ_i(γ̂(Y_i) + x_i^T β) - Λ̂(Y_i, x_i)]
- L_PL  = -1/n_e Σ_{i:Δ_i=1} [x_i^T β - log Σ_{j∈R(Y_i)} exp(x_j^T β)]
- L_ODE = 1/N Σ_j (∂Λ̂/∂t(t_j,x_j) - exp(γ̂(t_j) + x_j^T β))²
- L_IC  = 1/n Σ_i (Λ̂(0, x_i))²

---

## Module Specifications

### `src/simulation/cox_simulator.py`

```
BaselineHazard (ABC)
  .hazard(t), .cumulative_hazard(t), .inverse_cumulative_hazard(u)

ExponentialBaseline   # α(t) = λ  (constant)
WeibullBaseline       # α(t) = k·λ·(λt)^{k-1}
GompertzBaseline      # α(t) = b·exp(a·t)
PiecewiseConstantBaseline

CoxSimulator(baseline_hazard, beta, n_covariates, censoring_rate,
             censoring_type, covariate_dist, random_seed)
  .simulate(n_samples) → pd.DataFrame [time, event, x_1..x_p]
  Sampling: U~Unif(0,1), T = Λ_0^{-1}(-log(U)/exp(x^T β))
```

### `src/data/pipeline.py`

```
SurvivalDataset(Dataset)
  Normalized time [0,1], standardized covariates
  Precomputed risk set masks for PL loss
  .get_collocation_points(n, method) → Tensor  # 'uniform'|'observed_times'|'random'

SurvivalDataPipeline
  .fit_transform(df) → SurvivalDataset
  .transform(df) → SurvivalDataset
```

### `src/models/networks.py` + `pinn.py`

```
SurrogateNetwork(n_inputs, hidden_dims, activation)  # output: Softplus
CoefficientNetwork(hidden_dims, activation)           # output: unconstrained

HazardPINN(surrogate, coefficient, n_covariates)
  .beta: nn.Parameter
  .forward(t, x) → {Lambda_hat, gamma_hat, beta}
  .compute_Lambda_derivative(t, x) → ∂Λ̂/∂t  via autograd
  .from_config(config: dict) → HazardPINN
```

### `src/training/loss.py`

```
MLELoss, PartialLikelihoodLoss, ODEResidualLoss, InitialConditionLoss

CompositeLoss(weights: dict)
  weights keys: 'mle', 'pl', 'ode', 'ic'  — set to 0.0 to disable
  .forward(...) → (total_loss, component_dict)
```

### `src/training/trainer.py`

```
ExperimentConfig (dataclass)
  model_config, loss_weights, optimizer_name, lr, n_epochs,
  n_collocation_points, experiment_name, rationale

Trainer(model, loss_fn, config)
  .train(dataset) → loss_history
  .save_experiment(results_dir)
    → weights_final.pt, weights_best.pt, loss_history.csv, config.yaml, metrics.json
```

### `src/evaluation/metrics.py`

```
beta_accuracy(estimated, true) → {rmse, mae, per_coeff_relative_error}
baseline_hazard_accuracy(model, baseline, time_grid, pipeline) → {integrated_mse, integrated_relative_mse}
concordance_index(model, dataset, pipeline) → float
plot_loss_history(history, save_path)
plot_baseline_hazard(model, baseline, time_grid, pipeline, save_path)
plot_beta_comparison(estimated, true, save_path)
EvaluationReport.generate(model, simulator, dataset, pipeline, save_path)
```

---

## Staged Training Curriculum

| Stage | Baseline hazard | Covariates | Goal |
|-------|----------------|------------|------|
| 1     | Exponential (constant) | p=1 | Validate basic PINN works |
| 2     | Weibull (non-constant) | p=1 | Validate γ(t) network |
| 3     | Exponential (constant) | p=3 | Validate β scaling |
| 4     | Weibull / Gompertz     | p=3 | Full model validation |

**Acceptance thresholds (all must hold):**
- β RMSE < 0.10
- Baseline hazard integrated relative MSE < 0.05
- C-index > 0.75

---

## Implementation Order

1. `.claude/docs/AUTONOMOUS_TRAINING.md` ← first (session-persistent record)
2. `requirements.txt`
3. `src/simulation/cox_simulator.py`
4. `src/data/pipeline.py`
5. `src/models/networks.py` + `pinn.py`
6. `src/training/loss.py`
7. `src/training/trainer.py`
8. `src/evaluation/metrics.py`
9. `experiments/architecture_log.md` (template)
10. `experiments/configs/stage1_exponential_p1.yaml`
11. All `__init__.py` files
