# Autonomous Training Instructions: PINN Hazard Calibration

This document gives Claude everything needed to autonomously train, evaluate, and iteratively
improve the PINN until all acceptance thresholds are met — without asking the user any questions.
Read this file at the start of every session before doing anything else.

---

## 1. Session Resumption Protocol

When starting a new session (or resuming after token exhaustion):

1. Read `experiments/architecture_log.md` — find the last completed entry to determine current stage and last experiment name.
2. Read `experiments/results/{last_experiment}/metrics.json` — check whether thresholds were met.
3. If thresholds were met for a stage: advance to the next stage (see §3).
4. If thresholds were NOT met: apply the next improvement from the protocol in §5 and run a new experiment.
5. Never restart from scratch unless the architecture log explicitly says to.

---

## 2. Project Entry Points

```bash
# Run a single experiment from a config file:
python -m experiments.run_experiment --config experiments/configs/<name>.yaml

# Evaluate a saved model:
python -m experiments.evaluate --experiment experiments/results/<name>/

# Quick smoke test (end-to-end sanity check):
python -m experiments.smoke_test
```

**Key source files:**
- `src/simulation/cox_simulator.py` — data generation
- `src/data/pipeline.py` — preprocessing
- `src/models/pinn.py` — HazardPINN with `from_config()`
- `src/training/loss.py` — CompositeLoss (set weights to 0.0 to disable components)
- `src/training/trainer.py` — Trainer + ExperimentConfig
- `src/evaluation/metrics.py` — EvaluationReport

**Config file location:** `experiments/configs/`
**Results location:** `experiments/results/{experiment_name}/`
**Architecture log:** `experiments/architecture_log.md`

---

## 3. Staged Training Curriculum

Complete each stage fully before advancing. A stage is complete when ALL three acceptance
thresholds are met (see §4).

### Stage 1 — Constant hazard, 1 covariate
```yaml
# experiments/configs/stage1_exponential_p1.yaml
simulation:
  baseline: ExponentialBaseline
  baseline_params: {lam: 0.5}
  beta: [1.0]
  n_covariates: 1
  censoring_rate: 0.3
  censoring_type: random_exponential
  n_samples: 500
  random_seed: 42
```
**Goal:** Validate that the PINN can recover a known constant hazard and a single β coefficient.
This is the simplest possible case. If this fails, the bug is in the core architecture or loss.

### Stage 2 — Non-constant hazard, 1 covariate
```yaml
simulation:
  baseline: WeibullBaseline
  baseline_params: {k: 1.5, lam: 0.5}
  beta: [1.0]
  n_covariates: 1
  censoring_rate: 0.3
  n_samples: 500
```
**Goal:** Validate that the γ(t) coefficient network can learn a non-trivial hazard shape.

### Stage 3 — Constant hazard, multiple covariates
```yaml
simulation:
  baseline: ExponentialBaseline
  baseline_params: {lam: 0.5}
  beta: [1.0, -0.5, 0.3]
  n_covariates: 3
  censoring_rate: 0.3
  n_samples: 1000
```
**Goal:** Validate that β estimation scales to p=3 coefficients.

### Stage 4 — Non-constant hazard, multiple covariates (full model)
```yaml
simulation:
  baseline: WeibullBaseline
  baseline_params: {k: 1.5, lam: 0.5}
  beta: [1.0, -0.5, 0.3]
  n_covariates: 3
  censoring_rate: 0.3
  n_samples: 1000
```
**Goal:** Full Cox PINN as described in the paper.

---

## 4. Acceptance Thresholds

All three must hold simultaneously before a stage is considered complete:

| Metric | Threshold |
|--------|-----------|
| β RMSE | < 0.10 |
| Baseline hazard integrated relative MSE | < 0.05 |
| C-index | > 0.75 |

These values are stored in `experiments/results/{name}/metrics.json` after each run.

---

## 5. Diagnostic Guide — Reading Loss History

Open `experiments/results/{name}/loss_history.csv` and examine the per-epoch curves for each
component. Use these patterns to decide what to change:

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| L_ODE not decreasing | LR too high or surrogate too shallow | Lower LR or add hidden layers to surrogate |
| L_MLE flat / oscillating | β poorly initialized or L_PL dominating | Reduce w2 (PL weight) or re-init β |
| L_IC > 0.01 after 500 epochs | Initial condition not enforced | Increase w4 or add explicit t=0 collocation points |
| β estimates keep drifting | L_PL gradient noise | Increase w2 or add AdamW weight decay (1e-4) |
| All losses decrease but β wrong | Surrogate has absorbed covariate effect | Reduce surrogate width; increase L_PL weight |
| All losses decrease but hazard wrong | γ(t) network underfitting | Increase coefficient network depth |
| Training unstable (loss spikes) | LR too high | Halve LR; add gradient clipping (max_norm=1.0) |

---

## 6. Architecture Improvement Protocol

Apply changes in this order (cheapest first). After each change, run a new experiment and log it.
Only move to the next step if the current one did not help after 3 attempts with different values.

**Step 1 — Adjust loss weights (w1, w2, w3, w4)**
- Default start: w1=1.0, w2=0.5, w3=1.0, w4=1.0
- Try: w3=2.0 (emphasize ODE), or w4=2.0 (emphasize IC), or w2=0.0 (disable PL)

**Step 2 — Adjust learning rate / optimizer**
- Default: Adam, lr=1e-3
- Try: lr=5e-4, lr=1e-4; switch to AdamW with weight_decay=1e-4

**Step 3 — Surrogate network architecture**
- Default: hidden_dims=[64, 64, 64], activation=tanh
- Try: [128, 128, 128], [64, 64, 64, 64], or [32, 32]

**Step 4 — Coefficient network architecture**
- Default: hidden_dims=[32, 32], activation=tanh
- Try: [64, 64], [32, 32, 32]

**Step 5 — Activation functions**
- Try: silu (often better than tanh for PINNs), relu

**Step 6 — Collocation points**
- Default: n_collocation=200
- Try: 500, 1000, or use 'observed_times' method instead of 'uniform'

**Step 7 — Disable L_PL**
- Set w2=0.0 — the paper notes PL may not be necessary for convergence

**Step 8 — Add layer normalization**
- Add LayerNorm between each hidden layer in both networks

**Step 9 — Increase dataset size**
- Try n_samples=1000 (Stage 1/2) or 2000 (Stage 3/4)

**Step 10 — Increase training epochs**
- Try n_epochs=2000 or 5000 with LR scheduler (ReduceLROnPlateau, patience=100)

---

## 7. Logging Protocol

After EVERY experiment (regardless of success or failure), append an entry to
`experiments/architecture_log.md` in this format:

```markdown
## Experiment: {stage}_{name}  [YYYY-MM-DD]

**Stage:** {1|2|3|4}
**Changes vs previous:** {list every changed hyperparameter or architectural change}
**Rationale:** {why this change was made, based on the diagnostic guide}
**Results:**
  - β RMSE: {value}  [PASS/FAIL]
  - Hazard IMSE: {value}  [PASS/FAIL]
  - C-index: {value}  [PASS/FAIL]
  - Loss components at epoch end: MLE={}, PL={}, ODE={}, IC={}
**Next planned change:** {what to try next if this failed, or "ADVANCE TO STAGE N" if passed}
```

Save all experiment artifacts to `experiments/results/{stage}_{name}/`.

---

## 8. Token Limit Handling

If the context window is exhausted mid-run:
- All state is persisted in `experiments/architecture_log.md` and `experiments/results/`
- On resume: read this file (§1, §3–§6), then read the architecture log to find current position
- Continue from where the log left off — no need to ask the user
- The user will renew tokens and return; proceed autonomously as soon as the session starts

---

## 9. Permitted Autonomous Actions

Claude is permitted to do all of the following without asking the user:
- Create, edit, move, or delete any file in the project directory
- Run Python scripts and training experiments
- Commit and push to the GitHub remote (do this after every successful experiment and after any significant code change)
- Modify experiment YAML configs
- Modify source code in `src/` to implement architectural improvements
- Create new experiment configs for each iteration

---

## 10. Stopping Condition

Stop the autonomous loop only when Stage 4 acceptance thresholds are all met simultaneously.
Log the final entry in `architecture_log.md` as:

```markdown
## FINAL: All stages complete [YYYY-MM-DD]
Stage 4 thresholds met. β RMSE={}, Hazard IMSE={}, C-index={}.
Model weights saved to experiments/results/stage4_{name}/weights_best.pt
```

Then commit and push all remaining changes.
