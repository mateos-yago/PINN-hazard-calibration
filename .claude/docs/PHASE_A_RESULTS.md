# Phase A Results — Baseline-Hazard-Agnostic Architecture Search (p=1)

_Last updated 2026-05-13_

## Outcome

**Best architecture:** `phaseA_v6` (`experiments/configs/architectures/phaseA_v6.yaml`).

| Baseline | β RMSE | Hazard IRMSE | C-index | Pass |
|---|---|---|---|---|
| exp       | 0.021 | **0.011** ✓ | 0.787 | ✓ |
| gompertz  | 0.024 | 0.065 (vs 0.05 threshold) | 0.786 | β & C ✓, hazard ✗ |
| weibull   | 0.014 | 0.288 | 0.786 | β & C ✓, hazard ✗ |
| piecewise | 0.042 | 0.557 | 0.786 | β & C ✓, hazard ✗ |

All baselines pass β-recovery and concordance thresholds. Only Exp passes the
hazard IRMSE threshold cleanly; the others fail in the right tail.

## Architecture (the recipe)

The accepted architecture for the post-Stage-4 campaign:

1. **Λ via quadrature** (no Λ-surrogate). `Λ(t,x) = exp(xᵀβ) · ∫₀ᵗ exp(γ(s)) ds`
   on a 200-point trapezoidal grid in normalised time. ODE and IC are exact
   by construction; their loss weights are set to 0.
2. **γ-network**: SiLU MLP `[64, 64]` with `time_features = [t, log_t]` and
   input clamp `[0.01, 1.0]` (constant extrapolation past the training range).
3. **β**: Cox-PL initialised, `lr_beta = 1e-4` (low) so it stays near the
   PL optimum during training.
4. **Loss**: `mle=1.0`, `pl=0.25`, `ode=0`, `ic=0`, `monotonic=0.1`,
   `smoothness=10.0`, `baseline_ref=0`.
5. **Optimiser**: AdamW, `lr=3e-4` (Λ surrogate, unused), `lr_coefficient=1e-3`,
   `lr_beta=1e-4`, `weight_decay=1e-4`, gradient clip 1.0,
   `ReduceLROnPlateau(patience=200)`.
6. **Reproducibility**: `torch.manual_seed(simulation.random_seed)` set in
   `run_from_config` (added 2026-05-13).

## Key findings (chronological)

1. **v1** (Stage 4 v14 minus oracle, minus monotonic): catastrophic — all four
   baselines collapse to the same hump-and-decay shape regardless of true α(t).
   The Λ-surrogate fits the empirical Λ(Y_i, x_i) via its MLP capacity while
   the ODE residual stays nonzero, leaving γ structurally underdetermined.

2. **v2** (quadrature, no monotonic, no smoothness): β passes perfectly on
   every baseline (RMSE 0.002–0.025), but γ collapses into a near-delta
   spike at very small t. MLE on a free continuous γ has a degenerate
   optimum at delta functions placed on observed event clusters.

3. **v3** (smoothness, autograd-based): the autograd second derivative of γ
   blows up at small t because `d²log_t/dt² = −1/t²`. Switched to
   finite-difference smoothness on an interior grid.

4. **v4** (quadrature + γ-monotonicity): kills the spike but leaves
   uncontrolled MLP extrapolation past `t_norm = 1`, IRMSE = inf.

5. **v5** (+ input clamp + FD smoothness=0.1): exp passes; others fail with
   tail spikes within [0.85, 1.0]. Smoothness weight 100× too low to compete
   with MLE.

6. **v6** (smoothness=10): the working architecture. Exp passes; others
   still over-threshold but by 1.3×–11× rather than 100×.

7. **v7** (smaller [32,32] γ-net): no improvement. **Capacity is not the
   binding constraint.**

8. **v8** (n_epochs 3000): slightly worse. **Overfitting is not the binding
   constraint either.**

9. **v9** (no eval extension): Gompertz almost passes (0.053). Piecewise
   gets *worse* due to PyTorch RNG variance between runs — fixed by adding
   torch determinism in the runner.

10. **v10** (smoothness=100): backfires on Weibull — over-smoothing in the
    interior forces a compensating delta at `t_norm=1`. **Smoothness alone
    has been exhausted as a lever.**

## Open problem: hazard tail bias

The (MLE + monotonic + smoothness) combination on a free γ has a structural
tail-bias under sparse late-event data:

- Sublinear-in-t baselines (Weibull, piecewise plateau): γ̂ continues to rise
  through the data-sparse right tail because monotonic permits any
  non-decreasing shape and the bulk-slope from the interior extrapolates
  upward.
- Super-linear baselines (Gompertz): γ̂ undershoots because smoothness
  caps the rate of γ-rise below what the true exponential demands.

No single global smoothness weight resolves the two directions simultaneously.

Plausible directions for future work (all outside the strict pure-PINN
constraint of this campaign):

- **Density-aware MLE weighting** (e.g., inverse at-risk count) rebalances
  gradient signal across the time axis.
- **Nelson-Aalen self-supervised prior** — a kernel-smoothed Nelson-Aalen
  estimator from the data as a soft anchor for γ. Data-derived, not
  oracle, but excluded by the user's "strict pure-PINN" choice in this
  campaign.
- **Spline parameterisation of γ** with density-adaptive knot placement.
- **Non-uniform smoothness** that scales with `1/density(t)`.

## Next step

Phase B: scale v6 architecture to p=4 (β = `[1.0, -0.5, 0.3, -0.2]`) on the
same four baselines. The tail-bias issue is independent of covariate count,
so Phase B is expected to show the same pass/fail pattern as Phase A.

## Reproducibility

To regenerate Phase A:

```bash
python -m experiments.sweep \
  --architecture experiments/configs/architectures/phaseA_v6.yaml \
  --baselines exp,weibull,gompertz,piecewise \
  --p 1
```

Artifacts: `experiments/sweep_results/phaseA_v6/{summary.csv,summary.json,summary.md,plot_matrix.png}`
plus four leaf-run directories under `experiments/results/phaseA_v6_*`.
