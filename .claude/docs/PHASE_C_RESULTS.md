# Phase C Results — True PINN with Identifiability Fix

_Last updated 2026-05-14_

## Outcome

**Best architecture:** `phaseC_v4` (`experiments/configs/architectures/phaseC_v4.yaml`).

This is a genuine PINN — `parameterization: surrogate` (canonical free Λ MLP)
with the ODE and IC residuals doing real load-bearing work in the loss
(residuals at convergence ~10⁻²–10⁻³, contributing ~1–2% of total loss). It
solves the `phaseA_v1` identifiability collapse without resorting to the
quadrature trick (which makes ODE/IC vacuously satisfied) and without an
oracle baseline supervision.

| | p=1 | p=4 |
|---|---|---|
| exp | β=0.017 ✓, IRMSE=0.030 ✓, C=0.787 ✓ — **PASS** | β=0.057 ✓, IRMSE=0.012 ✓, C=0.763 ✓ — **PASS** |
| weibull | β=0.026 ✓, IRMSE=0.422, C=0.786 ✓ | β=0.060 ✓, IRMSE=0.211, C=0.764 ✓ |
| gompertz | β=0.018 ✓, IRMSE=0.562, C=0.786 ✓ | β=0.056 ✓, IRMSE=0.355, C=0.764 ✓ |
| piecewise | β=0.021 ✓, IRMSE=0.057, C=0.786 ✓ | β=0.058 ✓, IRMSE=0.068, C=0.765 ✓ |

β recovery is excellent across all baselines and both p; C-index is at the
oracle ceiling; piecewise IRMSE is on the boundary at both p (~0.06 vs the
0.05 threshold). The remaining tail bias on Weibull/Gompertz is the same
fundamental issue documented for Phase B — neither the soft-ODE-PINN nor the
quadrature parameterization escapes it under strict pure-PINN constraints.

## The identifiability problem (`phaseA_v1`)

The canonical PINN setup uses two networks:
- `Λ_φ(t, x)` — a free MLP approximating the cumulative hazard,
- `γ_θ(t)` — a free MLP approximating the log baseline hazard,

with the ODE `∂Λ_φ/∂t = exp(γ_θ + xᵀβ)` and the IC `Λ_φ(0, x) = 0` enforced
softly via L_ODE and L_IC.

In `phaseA_v1` (this setup minus the oracle `baseline_ref` loss that Stage 4
v14 used) the optimization converges to a degenerate solution: all four
estimated hazards have the **same hump-and-decay shape regardless of true
α(t)**. The failure mechanism is:

- Λ_φ has more MLP capacity than γ_θ and can fit `Λ(Y_i, x_i)` at every
  event independently, satisfying L_MLE without forcing L_ODE to zero.
- The remaining ODE residual leaves γ structurally underdetermined.
- γ drifts to whatever shape minimizes the residual locally, picking up
  the empirical event-rate hump.

Stage 4 v14 hid this by adding `baseline_ref` (direct supervision of γ
against the simulator's true α(t)) — unavailable on real data.

## The Phase C fix

The architecture introduces a new `parameterization: factored_surrogate` in
`src/models/pinn.py`. Three ingredients combine to break v1's identifiability:

### 1. Factorize the surrogate

The free `Λ_φ(t, x)` MLP is replaced by `Λ_φ(t)` — a 1-D MLP that depends
only on `t`. The cumulative hazard is then computed as

```
Λ(t, x) = exp(xᵀβ) · Λ_φ(t)
```

This enforces the Cox proportional-hazards factorization architecturally.
Λ_φ can no longer represent time-covariate interactions, so MLE pressure
propagates through Λ_φ *as a function of t alone* rather than through
per-`(t, x)` MLP capacity. Implemented in
`src/models/networks.py:BaselineCumHazardNetwork`.

By itself (phaseC_v1), this is not enough — the free 1-D Λ_φ still finds a
saturating shape (MLE on survival data with censoring prefers cumulative
hazards that resemble the empirical event CDF, which saturates as the
at-risk set depletes), and γ = log(∂Λ_φ/∂t) becomes hump-shaped just like
v1.

### 2. Regularize γ

Add the shape priors that worked for the quadrature winner:
- `monotonic = 0.1` — penalty on negative γ slope.
- `smoothness = 10` — finite-difference d²γ penalty on a [0.01, 1] interior
  grid.

These pin γ to a smooth non-decreasing shape. The ODE then transmits the
constraint to Λ_φ: with γ smooth and non-decreasing, the only consistent
Λ_φ is the cumulative integral of a smooth non-decreasing function, which
cannot saturate.

By itself with factoring (phaseC_v2), this fixes the γ hump-and-decay but β
drifts because L_MLE pushes β to compensate for any drift of Λ_φ away from
`∫exp(γ)`. β RMSE 0.15–0.23 — passing only barely on some cells.

### 3. Freeze β at Cox PL init

The Cox partial likelihood gives a statistically well-conditioned β
estimator that doesn't depend on the baseline α(t). Initialize β from
standalone Cox PL (`beta_initialization: cox_pl`, already in
`experiments/run_experiment.py:initialize_beta_from_cox_pl`) and set
`lr_beta = 0` so PyTorch never updates β during training.

With β frozen, MLE has no β-drift to compensate for Λ_φ inaccuracies. The
remaining optimization adjusts only γ and Λ_φ, both of which are
constrained by the priors above + the ODE residual.

Result: phaseC_v4. β RMSE 0.017–0.058 across all baselines and both p,
recovered to PL quality.

## What's still doing work in the loss

| Term | Status under Phase C | Role |
|---|---|---|
| L_MLE | active | data-fitting on `(Λ_φ(Y_i)·exp(x_iᵀβ), γ(Y_i))` |
| L_PL | active (weight 0.25) | nominally for β but β is frozen, so this term contributes a constant — kept for completeness |
| **L_ODE** | **active (weight 1.0)** | softly enforces `∂Λ_φ/∂t = exp(γ)` at 500 collocation points. **Residual ~10⁻² at convergence — load-bearing.** |
| **L_IC** | **active (weight 1.0)** | softly enforces `Λ_φ(0) = 0`. **Residual ~10⁻³ at convergence — load-bearing.** |
| L_monotonic | active (weight 0.1) | shape regularizer |
| L_smoothness | active (weight 10) | shape regularizer |

This is a **true PINN**: free Λ network, soft ODE/IC constraints. The
identifiability fix is architectural (factorization) + statistical
(γ regularizers + β anchoring), not via quadrature shortcuts.

## Phase B (quadrature) vs Phase C (true PINN) at p=4

| Baseline | phaseB_v1 IRMSE | phaseC_v4 IRMSE | Winner |
|---|---|---|---|
| exp | 0.0084 ✓ | 0.0119 ✓ | tie (both pass) |
| weibull | 0.111 | 0.211 | B |
| gompertz | **2.099** | **0.355** | **C** (6× better) |
| piecewise | 0.0125 ✓ | 0.068 | B (passes; C just over) |

Phase C is much better on Gompertz (the super-linear / exponential-growth
baseline) where quadrature's strict `Λ = ∫exp(γ)` undershoots the explosive
tail. Phase C's soft ODE allows Λ_φ to drift slightly from the integral,
which the optimizer uses to fit the late-event cluster better.

Phase B wins on Piecewise — quadrature handles sharp steps cleanly while
the free Λ_φ tends to over-smooth them.

The architectures have **complementary strengths**: quadrature when you
trust the smoothness of α; true PINN when α might grow faster than the
data-density-aware regularizers expect.

## Critical files

| Path | Role |
|---|---|
| `src/models/networks.py` | `BaselineCumHazardNetwork` — 1-D Λ_φ surrogate |
| `src/models/pinn.py` | `parameterization: factored_surrogate` |
| `src/training/loss.py` | `InitialConditionLoss` rerouted through `model.forward` |
| `experiments/configs/architectures/phaseC_v4.yaml` | The winning recipe |
| `experiments/sweep_results/phaseC_v4_p1/` | p=1 sweep results |
| `experiments/sweep_results/phaseC_v4_p4/` | p=4 sweep results |

## Reproducibility

```bash
# p=1 sweep
python -m experiments.sweep \
  --architecture experiments/configs/architectures/phaseC_v4.yaml \
  --baselines exp,weibull,gompertz,piecewise --p 1

# p=4 sweep
python -m experiments.sweep \
  --architecture experiments/configs/architectures/phaseC_v4.yaml \
  --baselines exp,weibull,gompertz,piecewise --p 4
```

## Remaining open problems

Same as Phase B's open problems — they're architecture-independent under the
strict pure-PINN constraint (no oracle, no data-derived hazard estimates):

- Tail bias on smooth-shaped baselines (Weibull/Gompertz).
- Underfit on non-monotone baselines (bathtub, piecewise hump).
- Overshoot on sharp-step late-jump piecewise.

These reflect the **likelihood landscape** of MLE on free γ with censored
data, not the physics constraints. They would require either density-aware
MLE weighting, a Nelson-Aalen self-supervised prior, or a structured γ
basis (splines / piecewise) — all outside this campaign's strict-PINN scope.
